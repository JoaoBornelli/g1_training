"""Currículo da Levanta-Caixa: Prioritized Level Replay (PLR) sobre ALTURAS.

Cada "nível" = uma altura de repouso da caixa (`shelf_top`). O conjunto CRESCE: a
cada degrau que o currículo desce, uma altura nova entra na lista e as anteriores
continuam no sorteio (anti-esquecimento). A cada reset, o sampler escolhe qual altura
cada env vai treinar, de uma distribuição RANK-BASED:

    P(nível) = ρ/N  +  (1−ρ) · (1/rank^(1/β))
               └─ piso ─┘      └── foco na dificuldade ──┘

- ρ (`floor_rho`): piso uniforme → TODA altura recebe pelo menos ρ/N (nunca esquece).
- rank: nível mais DIFÍCIL = rank 1 = mais massa; β (`focus_beta`) = agressividade.
- dificuldade do nível = EMA de (1 − performance); performance = fração do episódio
  com a caixa sustentada no alvo, lida da soma do episódio do termo `sustain_precise`
  (`reward_manager._episode_sums`, ainda viva quando o curriculum roda no reset).
  Nível que o robô domina → performance alta → dificuldade baixa → perde massa.
  Rebalanceia sozinho conforme o robô melhora/piora.

Roda como CURRICULUM term, que no reset dispara ANTES do evento de posicionamento
(`manager_based_rl_env.py:554` vs `:560`) → aqui a altura é sorteada e gravada em
`env.plr_shelf_top[env_ids]`; o evento `reset_scene_plr` (events.py) lê esse buffer e
posiciona prateleira+caixa. Loga `Curriculum/plr_heights/*` (massa e score por altura).

Warm-start-safe: mexe só em QUAL altura cada env vê no reset — obs/ação/rede intactos.
Molde: `mjlab/tasks/velocity/mdp/curriculums.py::terrain_levels_vel`. Ver [[g1-lift-box-task]].
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


class PlrHeights:
    """PLR rank-based sobre alturas de prateleira (curriculum term COM ESTADO).

    Stateful: o mjlab auto-instancia (`func(cfg, env)`), chama `__call__(env, env_ids)`
    a cada reset dos envs que terminaram, e `reset(env_ids)` (que aqui é no-op de
    propósito — o score do PLR PERSISTE entre resets; é o ponto do método)."""

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv"):
        p = cfg.params
        dev = env.device
        levels = tuple(p["shelf_levels"])
        assert len(levels) >= 1, "PLR precisa de >=1 altura em shelf_levels"
        self.levels = torch.tensor(levels, device=dev, dtype=torch.float32)  # [L]
        self.rho = float(p.get("floor_rho", 0.30))
        self.beta = float(p.get("focus_beta", 1.0))
        self.alpha = float(p.get("ema_alpha", 0.1))
        self.sustain_term = str(p.get("sustain_term", "sustain_precise"))
        self.sustain_weight = float(p.get("sustain_weight", 1.0))
        box_half_z = float(p.get("box_half_z", 0.10))

        L = self.levels.numel()
        # score = dificuldade EMA por nível. Init uniforme (0.5); opcional semear o
        # nível mais NOVO (menor altura = último da lista) alto pra já focar nele.
        self.scores = torch.full((L,), 0.5, device=dev)
        if bool(p.get("seed_newest_high", True)) and L > 1:
            self.scores[-1] = 1.0

        # buffers POR-ENV (canal p/ o evento reset_scene_plr e p/ o lift_reward).
        # Criados aqui pra existirem antes do 1º reset/step.
        self.env_level = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        self._visited = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        env.plr_shelf_top = self.levels[self.env_level].clone()   # [B] altura do nível
        env.plr_rest_z = env.plr_shelf_top + box_half_z           # [B] z de repouso da caixa

    def _distribution(self) -> torch.Tensor:
        """P(nível) rank-based, [L], soma 1."""
        L = self.levels.numel()
        if L == 1:
            return torch.ones(1, device=self.scores.device)
        order = torch.argsort(self.scores, descending=True)   # rank 1 = mais difícil
        rank = torch.empty(L, device=self.scores.device)
        rank[order] = torch.arange(1, L + 1, device=self.scores.device, dtype=rank.dtype)
        p_dif = rank.pow(-1.0 / self.beta)
        p_dif = p_dif / p_dif.sum()
        P = self.rho / L + (1.0 - self.rho) * p_dif
        return P / P.sum()

    def __call__(self, env, env_ids, **_):
        dev = self.scores.device
        # 1. ATUALIZA o score dos níveis que os envs que TERMINARAM estavam treinando
        #    (ignora os que nunca completaram um episódio — 1º reset, _visited=False).
        valid = self._visited[env_ids]
        if bool(valid.any()):
            sums = env.reward_manager._episode_sums[self.sustain_term][env_ids]
            denom = env.max_episode_length_s * self.sustain_weight
            perf = (sums / denom).clamp(0.0, 1.0)     # performance ∈ [0,1] (episódio)
            diff = 1.0 - perf                          # dificuldade
            lv = self.env_level[env_ids][valid]
            dv = diff[valid]
            for l in torch.unique(lv):                 # L pequeno → loop barato
                m = dv[lv == l].mean()
                self.scores[l] = (1.0 - self.alpha) * self.scores[l] + self.alpha * m

        # 2. SORTEIA nível novo pros envs que resetaram (rank-based) → grava no buffer.
        P = self._distribution()
        new = torch.multinomial(P, len(env_ids), replacement=True)
        self.env_level[env_ids] = new
        env.plr_shelf_top[env_ids] = self.levels[new]
        self._visited[env_ids] = True

        # 3. LOG (Curriculum/plr_heights/*): massa e dificuldade por altura.
        out: dict[str, torch.Tensor] = {
            "num_levels": torch.tensor(float(self.levels.numel()), device=dev)}
        for i in range(self.levels.numel()):
            h = float(self.levels[i])
            out[f"prob_{h:.2f}"] = P[i]
            out[f"score_{h:.2f}"] = self.scores[i]
        return out

    def reset(self, env_ids=None):
        # NÃO limpa os scores — o estado do PLR persiste entre resets (é o método).
        pass
