"""Contribuição de cada termo de reward, SEPARADA POR TAREFA.

Tarefa nova, não está na §13 do doc. Existe porque o workflow de blocos de 2k-3k
exige olhar entre um bloco e o outro, e **o log default não serve**: o
`RewardManager` loga `Episode_Reward/<termo>` como média sobre TODOS os envs, e com
7 tarefas intercaladas um termo gateado a uma tarefa aparece com ~1/7 da magnitude
real. Olhando só esse número não há como saber qual termo domina em qual tarefa —
que é justamente a pergunta que se faz entre blocos.

Custo zero de recomputação: o `RewardManager.compute` já guarda o valor PONDERADO de
cada termo por env em `_step_reward[:, idx]` (`reward_manager.py:132`). Aqui a gente
só soma essa matriz por tarefa ativa.

São dois termos cooperando, porque o mjlab não tem um único ponto que rode a cada
passo E possa emitir dict de log:

  `Contribuicao`  — termo de MÉTRICA, roda a cada passo, acumula a matriz
  `Relatorio`     — termo de CURRÍCULO, roda no reset, emite o dict e zera

O `PlrHeights` da Lift já usa esse canal de log do currículo (`Curriculum/...`), então
não é mecanismo novo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class Contribuicao:
    """Acumula `peso × valor` de cada termo, por tarefa. Termo de métrica.

    O retorno é a reward total do passo (útil de logar por si), mas o trabalho real é
    o efeito colateral: preencher `env.contrib_soma` e `env.contrib_cont`."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        n_termos = len(env.reward_manager.active_terms)
        env.contrib_soma = torch.zeros(T.NUM_TASKS, n_termos, device=env.device)
        env.contrib_cont = torch.zeros(T.NUM_TASKS, device=env.device)
        env.contrib_nomes = list(env.reward_manager.active_terms)

    def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
        del params
        passo = env.reward_manager._step_reward            # [B, n_termos], ponderado
        tarefa = env.active_task                           # [B]
        # index_add_ soma cada linha na fatia da tarefa dela: uma passada, sem loop
        env.contrib_soma.index_add_(0, tarefa, passo)
        env.contrib_cont.index_add_(
            0, tarefa, torch.ones_like(tarefa, dtype=passo.dtype))
        return passo.sum(dim=-1)

    def reset(self, env_ids=None):
        pass    # a matriz é zerada pelo `Relatorio`, não pelo reset de env


class Relatorio:
    """Emite a matriz de contribuição como dict de log e zera. Termo de currículo.

    Chaves saem como `Contrib/<tarefa>/<termo>`, e mais `Contrib/<tarefa>/_total`.
    O `entre_blocos.py` lê exatamente essas chaves do TensorBoard.

    ⚠️ Emite **só quando a amostra é suficiente**: com poucos passos numa tarefa a
    média é ruído, e ruído num relatório que orienta ajuste de peso é pior que
    silêncio. O limiar é `min_amostras`."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        self.min_amostras = float(cfg.params.get("min_amostras", 500))
        self.top = int(cfg.params.get("top", 0))
        """0 = loga todos os termos. >0 = só os `top` de maior |contribuição|, pra o
        TensorBoard não virar 210 séries. O `entre_blocos.py` prefere todos."""

    def __call__(self, env, env_ids, **_):
        soma, cont = env.contrib_soma, env.contrib_cont
        if float(cont.sum()) < self.min_amostras:
            return {}
        out: dict[str, torch.Tensor] = {}
        media = soma / cont.clamp(min=1.0).unsqueeze(-1)
        for t in range(T.NUM_TASKS):
            if float(cont[t]) < self.min_amostras / T.NUM_TASKS:
                continue          # tarefa quase não sorteada: não polui o log
            linha = media[t]
            out[f"{T.NAMES[t]}/_total"] = linha.sum()
            ordem = torch.argsort(linha.abs(), descending=True)
            alvo = ordem[: self.top] if self.top > 0 else ordem
            for i in alvo.tolist():
                out[f"{T.NAMES[t]}/{env.contrib_nomes[i]}"] = linha[i]
        soma.zero_()
        cont.zero_()

        # --- item 4: as duas linhas que respondem "o sucesso é real?" ---
        # Mascaradas por `tarefa_sorteada`, a MESMA chave que o `_medir` do currículo
        # usa pra creditar — é essa a comparação que importa. `contrib` acima usa
        # `active_task`, e as duas divergem na janela de pré-gatilho.
        dsoma, dcont = getattr(env, "diag_soma", None), getattr(env, "diag_cont", None)
        if dsoma is not None:
            dm = dsoma / dcont.clamp(min=1.0).unsqueeze(-1)
            for t in range(T.NUM_TASKS):
                if float(dcont[t]) < self.min_amostras / T.NUM_TASKS:
                    continue
                # fração do tempo em que a condição FÍSICA da tarefa vale
                out[f"{T.NAMES[t]}/cond_fisica"] = dm[t, 0]
                # fração dos sucessos registrados com a condição dela FALSA. Tem que
                # ser ZERO. Diferente de zero = crédito falso, e o log denuncia na hora
                # em vez de exigir uma caçada de horas como a de 31/07.
                out[f"{T.NAMES[t]}/atribuicao_divergente"] = dm[t, 1]
            dsoma.zero_()
            dcont.zero_()
        return out

    def reset(self, env_ids=None):
        pass
