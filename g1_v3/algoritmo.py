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
escala. A M1 (13/11) usa o mesmo grupo para a média de V que ela soma na queda e na perda da caixa.
"""
from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

__all__ = ["PPOPorTarefa", "normaliza_por_grupo", "fixa_normalizador", "CAMINHO", "PISO_DESVIO"]

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


# Piso do desvio do normalizador da observação. O `rsl_rl` divide por `desvio + 0,01`: um canal que ficou constante
# (a `caixa` vale zero no ANDAR, e o currículo começa só com ANDAR) entraria na rede com ganho de até 100 quando
# mudasse. Com o piso o ganho máximo é ~17. Medido no model_11200: o menor desvio de um canal vivo é ~0,05.
PISO_DESVIO = 0.05


def fixa_normalizador(nz, fatias, piso: float = PISO_DESVIO) -> None:
    """Põe o piso no desvio do normalizador `nz` e fixa cada fatia de `fatias` em média 0 e no desvio do seu par
    `(fatia, desvio)`. Sem normalizador (`Identity`), não faz nada.

    ⚠ O piso vai só no `_std`, que é o que o `forward` lê; a `_var` segue a estatística real. Nas fatias fixas o desvio
    é o do contrato (`runner._FIXOS`), e não o empírico: um one-hot é categórico, e padronizá-lo amplifica o slot raro
    (g1_limpo, model_3450: 94 no lugar de 1).
    """
    if not hasattr(nz, "_std"):
        return
    nz._std.clamp_(min=piso)
    for sl, dv in fatias:
        nz._mean[..., sl] = 0.0
        nz._std[..., sl] = dv
        nz._var[..., sl] = dv * dv


class PPOPorTarefa(PPO):
    """PPO com a vantagem normalizada por tarefa e os canais fixos fora do normalizador da observação."""

    # {"actor": [(slice, desvio), ...], "critic": [...]}: o `RunnerV3` preenche pelos nomes dos termos (`runner._FIXOS`).
    fatias_fixas: dict = {}

    # Fatia do one-hot `alvo_macro` no crítico. O `RunnerV3` preenche; None faz o grupo da vantagem ser só a tarefa.
    fatia_alvo = None

    # Média de `storage.values` por grupo, do rollout anterior (1 + 6·n posições; o `compute_returns` preenche).
    v_grupo = None

    def _grupo(self, obs_critico: torch.Tensor) -> torch.Tensor:
        """Grupo da vantagem e da M1, de uma observação do crítico (..., D): (...,) long.

        A tarefa é o argmax das 6 últimas colunas. Sem `fatia_alvo`, o grupo é a tarefa. Com ela, o grupo é 0 para o
        ANDAR com o slot "nenhum" (o último da fatia) e `1 + tarefa·n + slot` no resto: o ANDAR final (B4 fechado) e o
        ANDAR depois da perda ganham grupo próprio, como as outras tarefas (13/11, M7). Desde 11/11, um grupo por tarefa e
        alvo, como o g1_limpo normalizava por elo: com um grupo por tarefa, o P4 tinha 6,7 % da magnitude da vantagem.
        """
        tarefa = obs_critico[..., -6:].argmax(-1)
        if self.fatia_alvo is None:
            return tarefa
        fatia = obs_critico[..., self.fatia_alvo]
        n = fatia.shape[-1]
        slot = fatia.argmax(-1)
        return torch.where((tarefa == ANDAR) & (slot == n - 1), torch.zeros_like(tarefa), 1 + tarefa * n + slot)

    def process_env_step(self, obs, rewards, dones, extras):
        """M1 (13/11): a queda e a perda da caixa nunca escapam de um futuro negativo. Numa terminação ruim
        (`dones` sem `time_outs`) e no passo da perda (`extras["v3_perdeu_agora"]`), a recompensa ganha
        γ·min(V̄_g, 0), com V̄_g a média do valor do grupo no rollout anterior (`v_grupo`, sem gradiente).
        Com V̄_g ≥ 0 nada muda. Com V̄_g < 0, cair custa a perda mais o futuro negativo que o robô evitaria.

        ⚠ A média do grupo, e não o V do próprio estado: o alvo de um estado que cai viraria r + γ·min(V(s), 0), e o
        crítico convergiria para r/(1 − γ) nesses estados. A perda da caixa não termina o episódio, e vale pelo
        mesmo motivo: com V < −5,8, largar a caixa seria a saída mais barata.
        ⚠ O grupo vem das observações do PASSO (`transition.observations`, gravadas no `act`), e não das de `obs`, que
        já são as do reset.
        O `super()` atualiza os normalizadores em TODOS os canais; a fixação desfaz a mexida antes de a rede ler.
        """
        if self.fatia_alvo is not None and self.v_grupo is not None:
            g = self._grupo(self.transition.observations["critic"])
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
            ruim = ((dones > 0) & ~time_outs) | extras["v3_perdeu_agora"]
            rewards = rewards + self.gamma * self.v_grupo[g].clamp(max=0.0) * ruim.float()
        saida = super().process_env_step(obs, rewards, dones, extras)
        for rede in ("actor", "critic"):
            fixa_normalizador(getattr(self, rede).obs_normalizer, self.fatias_fixas.get(rede, ()))
        return saida

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
        grupo = self._grupo(st.observations["critic"])
        st.advantages = normaliza_por_grupo(crua.reshape(-1), grupo.reshape(-1)).reshape(crua.shape)
        if self.fatia_alvo is not None:
            # M1: V̄ de cada grupo, para o rollout seguinte (sem gradiente). Grupo ausente fica com o valor anterior.
            n = self.fatia_alvo.stop - self.fatia_alvo.start
            soma = torch.zeros(1 + bloco.shape[-1] * n, device=st.values.device)
            cont = torch.zeros_like(soma)
            g, v = grupo.reshape(-1), st.values.detach().reshape(-1)
            soma.index_add_(0, g, v)
            cont.index_add_(0, g, torch.ones_like(v))
            anterior = self.v_grupo if self.v_grupo is not None else torch.zeros_like(soma)
            self.v_grupo = torch.where(cont > 0, soma / cont.clamp(min=1.0), anterior)
