"""O termo de ação residual: o BFM dá equilíbrio, a RL dá a habilidade.

    z     = projeta_16( prior[tarefa] + 0,3 * (c @ B) )
    alvo  = a_BFM(obs_bfm, z) * ESCALA_BFM * 5,0 + PADRAO_BFM  +  clamp(delta)

A política emite **49 números**: 29 de residual por junta e 20 de coeficiente de
comportamento.

**Por que subclasse do `JointPositionAction` e não peça nova.** O
`apply_actions` do fabricante já faz o que precisa: subtrai o `encoder_bias` (que é
a nossa DR) e chama `set_joint_position_target`. Só a MATEMÁTICA muda, e ela mora no
`process_actions`. Assim o caminho de escrita no sim continua sendo o testado.

**A divisão de trabalho, e é ela que responde "o BFM e a RL vão brigar?".**
Só a RL tem gradiente; o BFM é função fixa. Então não é cabo de guerra — a RL
aprende SOBRE o BFM, vendo o resultado somado na reward. O que pode dar errado é a
RL gastar o orçamento cancelando o BFM, e o clamp é o que decide isso:

    perna, tornozelo, cintura   0,35 rad (20°)   o BFM continua dono do equilíbrio
    braço (ombro, cotovelo, punho)  2,0 rad (115°)  a RL é dona da tarefa

Medido: com ação 1,0 o BFM move o ombro 125,6° e o joelho 100,5°. Um clamp de
±0,2 rad (11,5°) daria à RL 9% da autoridade do ombro — ela não alcançaria nada que
o BFM já não alcance. Daí o braço solto.

⚠️ **A pose padrão do BFM não é a nossa.** Elas diferem em até 34,4°. O alvo do BFM
sai somado ao padrão DELE; nada aqui usa `default_joint_pos` do mjlab.
"""
import pathlib
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_residual.base_z import DIM_C, ESCALA_C, BaseZ  # noqa: E402
from g1_residual.bfm import PESO, AtorBFM  # noqa: E402
from g1_residual.obs_bfm import ObsBFM  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg  # noqa: E402
from mjlab.utils.spec_config import resolve_matching_names_values  # noqa: E402

LIMITE_PADRAO: dict[str, float] = {
    # cadeia de equilíbrio — o BFM manda, a RL só corrige
    r".*_hip_.*": 0.35,
    r".*_knee_joint": 0.35,
    r".*_ankle_.*": 0.35,
    r"waist_.*": 0.35,
    # efetuador — a RL manda
    r".*_shoulder_.*": 2.0,
    r".*_elbow_joint": 2.0,
    r".*_wrist_.*": 2.0,
}
"""Clamp do residual, em RADIANOS de alvo de junta.

Em radianos de propósito. Em unidade de ação o mesmo número vale coisas diferentes:
a escala varia 7,4x entre juntas (0,0596 no punho contra 0,438 no quadril), e ainda
difere 6,25x entre a nossa convenção e a do BFM."""


@dataclass(kw_only=True)
class ResidualBFMActionCfg(JointPositionActionCfg):
    """Config do termo. Herda `scale`/`offset`/`clip` mas NÃO os usa na conta.

    O `scale` do pai fica para o `__init__` do fabricante montar `_target_names` e
    validar; a conversão de unidade aqui é explícita e usa as constantes do plant do
    BFM, gravadas no `.pt` pelo `extrai_ator.py`."""

    limite_rad: dict[str, float] | None = None
    escala_c: float = ESCALA_C
    dim_c: int = DIM_C
    prior_unico: bool = False
    rolagens_por_passo: int = 2
    caminho_peso: str = str(PESO)
    semente_z: int | None = None

    def __post_init__(self):
        self.class_type = ResidualBFMAction


class ResidualBFMAction(JointPositionAction):
    def __init__(self, cfg: ResidualBFMActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        self.cfg: ResidualBFMActionCfg = cfg
        dev = self.device

        self._n_juntas = int(self._action_dim)
        assert self._n_juntas == 29, f"esperava 29 juntas, achei {self._n_juntas}"

        # A política emite 29 + 20. O `_action_dim` do pai é atributo simples, e o
        # `_raw_actions` foi dimensionado por ele — os dois crescem juntos.
        self._action_dim = self._n_juntas + cfg.dim_c
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=dev)
        # `_processed_actions` continua com 29: é o ALVO DE JUNTA que o
        # `apply_actions` do pai escreve no sim.
        self._processed_actions = torch.zeros(self.num_envs, self._n_juntas, device=dev)

        self._ator = AtorBFM(pathlib.Path(cfg.caminho_peso), device=dev)
        plant = torch.load(cfg.caminho_peso, map_location=dev,
                           weights_only=True)["plant"]
        # alvo do BFM = a * ACTION_SCALES * 5.0 + DEFAULT_JOINT_POS  (dele!)
        self._ganho_bfm = (plant["action_scales"].to(dev)
                           * plant["action_rescale"].to(dev)).view(1, -1)
        self._padrao_bfm = plant["default_joint_pos"].to(dev).view(1, -1)

        self._obs = ObsBFM(env, plant["default_joint_pos"],
                           rolagens_por_passo=cfg.rolagens_por_passo)
        self._base = BaseZ(self._ator.z_tabela, device=dev, dim=cfg.dim_c,
                           prior_unico=cfg.prior_unico)

        limites = cfg.limite_rad or LIMITE_PADRAO
        idx, nomes, vals = resolve_matching_names_values(limites, self._target_names)
        self._limite = torch.zeros(1, self._n_juntas, device=dev)
        self._limite[0, idx] = torch.tensor(vals, device=dev)
        assert bool((self._limite > 0).all()), (
            "alguma junta ficou sem limite — o clamp viraria zero e a RL perderia "
            f"ela: {[n for n, v in zip(self._target_names, self._limite[0].tolist()) if v <= 0]}")

        self._z = torch.zeros(self.num_envs, self._ator.z_dim, device=dev)
        print(f"[RESIDUAL] ação {self._action_dim} = {self._n_juntas} residual + "
              f"{cfg.dim_c} comportamento | base cobre "
              f"{self._base.energia:.1%} da energia dos 41 | "
              f"prior {'ÚNICO' if cfg.prior_unico else 'por tarefa'}")

    # ------------------------------------------------------------------ ciclo
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        delta_bruto = actions[:, :self._n_juntas]
        c = actions[:, self._n_juntas:]

        # 1. o comportamento que a RL escolheu, partindo do prior da tarefa
        tarefa = getattr(self._env, "tarefa_sorteada", None)
        if tarefa is None:                      # env sem currículo (o `play` cru)
            tarefa = torch.zeros(actions.shape[0], dtype=torch.long,
                                 device=actions.device)
        self._z = self._base.z(tarefa, c, escala=self.cfg.escala_c)

        # 2. o BFM, congelado, com a obs DELE
        estado, ultima, historico = self._obs.monta()
        a_bfm = self._ator(estado, ultima, historico, self._z)
        self._obs.guarda_acao(a_bfm)

        # 3. alvo absoluto do BFM, na pose padrão DELE
        alvo = a_bfm * self._ganho_bfm + self._padrao_bfm

        # 4. residual em radianos. `clamp(d, -1, 1) * limite` em vez de
        #    `clamp(d * escala, -limite, limite)`: assim `d = ±1` é exatamente o
        #    limite e a política não precisa descobrir a escala.
        self._processed_actions = alvo + delta_bruto.clamp(-1.0, 1.0) * self._limite

    def reset(self, env_ids=None) -> None:
        # O BFM começa episódio com histórico ZERADO, igual ao `env.py:397` dele.
        self._obs.reset(env_ids)
        pai = getattr(super(), "reset", None)
        if callable(pai):
            pai(env_ids)

    # -------------------------------------------------------------- leitura
    @property
    def z_atual(self) -> torch.Tensor:
        return self._z

    def relatorio(self) -> dict[str, float]:
        """O que a rede escolheu, em nome de comportamento e em graus.

        É a tabela de contribuição, mas para comportamento."""
        tarefa = getattr(self._env, "tarefa_sorteada", None)
        if tarefa is None:
            return {}
        nomes, cos = self._base.mais_proximo(self._z)
        graus = self._base.graus_do_prior(tarefa, self._z)
        out: dict[str, float] = {"z_graus_do_prior": float(graus.mean())}
        # qual comportamento domina, por tarefa presente no lote
        for t in tarefa.unique().tolist():
            m = tarefa == t
            votos: dict[str, int] = {}
            for n in (n for n, keep in zip(nomes, m.tolist()) if keep):
                votos[n] = votos.get(n, 0) + 1
            vencedor = max(votos, key=votos.get)
            out[f"z_perto/{t}/{vencedor}"] = votos[vencedor] / int(m.sum())
            out[f"z_graus/{t}"] = float(graus[m].mean())
        out["z_cos_do_mais_proximo"] = float(cos.mean())
        return out
