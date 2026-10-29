"""O PPO do `rsl_rl`, com a vantagem normalizada POR TAREFA (D9 do plano, §11; origem
`g1_limpo/algoritmo.py:155-224`).

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `rsl_rl` (framework) e `torch`.

O DEFEITO QUE ISTO CONSERTA: `rsl_rl/algorithms/ppo.py` normaliza a vantagem com `mean` e
`std` sobre o LOTE INTEIRO. Quando uma tarefa destrava, a vantagem dela fica grande e
dispersa, o `std` do lote cresce, e as vantagens das outras tarefas encolhem para perto de
zero. Medido no g1_limpo: a locomoção perdia a fatia da magnitude do gradiente e continuava
sendo arrastada pelo gradiente da manipulação. A normalização por grupo separa a escala de
cada tarefa. Os pesos seguem INTEIRAMENTE compartilhados (é isso que o CARREGAR, andar
segurando a caixa, precisa); o que muda é só a estatística de agregação do PPO.

⚠ O GRUPO é a tarefa corrente, lida do crítico: o argmax das 6 últimas colunas da observação
(`tarefa_critico`, §6.5). É o crítico que estima o retorno, e o retorno é o que a normalização
escala.
"""
from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

__all__ = ["PPOPorTarefa", "normaliza_por_grupo", "CAMINHO"]

# ⚠ O caminho QUALIFICADO, para o `class_name` do cfg (`rl_cfg()` da D10). Uma string, e não
# a classe, porque o logger do mjlab despeja o cfg em disco e uma classe não serializa.
CAMINHO = "g1_v3.algoritmo:PPOPorTarefa"

# índice de TAREFAS (§6.2 `knobs.TAREFAS`): ANDAR é o 0. É o grupo que fica com a vantagem
# crua quando o grupo é ralo (ver `normaliza_por_grupo`).
ANDAR = 0


def normaliza_por_grupo(adv: torch.Tensor, grupo: torch.Tensor) -> torch.Tensor:
    """Vantagem normalizada dentro de cada grupo. `adv` e `grupo` são (N,); a saída é (N,).

    Cada grupo com 2 ou mais amostras sai com média 0 e desvio 1 (o `std()` do torch).

    ⚠ GRUPO RALO (< 2 amostras). Com uma amostra só o `std` é NaN, e o NaN se propaga para o
    gradiente inteiro no passo seguinte. Por isso:
      · ANDAR ralo fica com a vantagem CRUA (a amostra não tem escala própria);
      · os outros ralos usam a "manipulação inteira" (todas as amostras não-ANDAR do lote,
        o próprio grupo ralo incluído). Dá a mesma ordem de grandeza sem inventar escala.
    Se a manipulação tem menos de 2 amostras, o ralo também fica cru.
    """
    saida = adv.clone()
    manip = grupo != ANDAR
    media_manip = desvio_manip = None
    if int(manip.sum()) >= 2:
        media_manip, desvio_manip = adv[manip].mean(), adv[manip].std()
    for t in torch.unique(grupo).tolist():
        m = grupo == t
        n = int(m.sum())
        if n < 2:
            if t != ANDAR and desvio_manip is not None:
                saida[m] = (adv[m] - media_manip) / (desvio_manip + 1e-8)
            continue
        a = adv[m]
        saida[m] = (a - a.mean()) / (a.std() + 1e-8)
    return saida


class PPOPorTarefa(PPO):
    """PPO com a vantagem normalizada por tarefa. Só `compute_returns` é diferente."""

    def compute_returns(self, obs) -> None:
        """Recalcula a vantagem por grupo de tarefa, sobre a vantagem CRUA.

        ⚠ Chama o `super()` e REFAZ a normalização em vez de reimplementar o GAE. O GAE do
        rsl_rl está certo; reimplementá-lo aqui seria uma segunda fonte de verdade que derivaria
        no primeiro upgrade.
        """
        super().compute_returns(obs)
        st = self.storage

        # ⚠ O one-hot da tarefa corrente são as ÚLTIMAS 6 colunas do crítico (§6.5, último termo).
        # Se a ordem dos termos mudar, o grupo sai errado sem erro nenhum: a trava é a soma.
        bloco = st.observations["critic"][..., -6:]
        soma = bloco.sum(-1)
        assert torch.allclose(soma, torch.ones_like(soma), atol=1e-3), (
            f"as 6 últimas colunas do crítico não são um one-hot (soma média "
            f"{float(soma.mean()):.4f}); `tarefa_critico` deixou de ser o último termo?")

        # ⚠ Sobre a vantagem CRUA, e não sobre a que o `super()` já normalizou: renormalizar o
        # normalizado misturaria as duas escalas.
        crua = st.returns - st.values
        st.advantages = normaliza_por_grupo(
            crua.reshape(-1), bloco.argmax(-1).reshape(-1)).reshape(crua.shape)


