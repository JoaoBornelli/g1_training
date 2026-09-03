"""O PPO do `rsl_rl`, com a vantagem normalizada POR GRUPO DE ELO.

⚠ ZERO IMPORT DE CÓDIGO DE OUTRO MÓDULO DO PROJETO. Só `rsl_rl` (framework) e o próprio
`g1_limpo`.

O DEFEITO QUE ISTO CONSERTA está em `rsl_rl/algorithms/ppo.py:188`:

    st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

Um `mean` e um `std` sobre o LOTE INTEIRO, misturando envs de locomoção e de
manipulação. Quando a manipulação destrava, as vantagens dela ficam grandes e dispersas,
o `std` do lote cresce, e as vantagens da LOCOMOÇÃO são divididas por ele — elas encolhem
para perto de zero. A locomoção para de receber sinal de gradiente e continua sendo
arrastada pelo gradiente da outra tarefa.

MEDIDO no bloco 7 (`[ADV]` a cada 50 iterações), com ~32% dos envs em locomoção:

    it 1600   loco 0,112   manip 0,446   razão 0,251
    it 1650   loco 0,162   manip 1,042   razão 0,155
    it 1800   loco 0,164   manip 1,174   razão 0,139

A fatia da locomoção na MAGNITUDE do gradiente, que é `fração × desvio normalizado`:

    global    10,4%  ->  6,7%  ->  5,6%      (piorando conforme a manipulação avança)
    por elo   ~30%   (a própria fatia de amostra)

E o resultado, comparando a MESMA iteração com o MESMO nível de manipulação:

                        sem patch   com patch
    descarga (manip)      0,991       0,994
    marcha                0,484       0,762
    fell_over            51,9%        0,9%
    duração do episódio    425         888

⚠ POR QUE ISTO NÃO É "SEPARAR AS TAREFAS". Os pesos continuam INTEIRAMENTE
compartilhados, e é isso que o elo `CARREGAR` — andar segurando a caixa — precisa. O que
muda é só a estatística de agregação do PPO, que não carrega conhecimento nenhum. Uma
arquitetura com roteamento por elo daria separação matando a transferência; esta não.

⚠ O SEGUNDO CANAL GLOBAL SEGUE ABERTO, e é declarado: a taxa de aprendizado
(`ppo.py:246-249`) é UMA só, dirigida pelo `kl_mean` agregado das duas tarefas. Uma
virada grande na manipulação derruba o passo da locomoção junto. Não entra aqui porque
uma variável por bloco.
"""
from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from g1_limpo.comando import ANDAR
from g1_limpo.observacoes import fatia_do_elo_interno

__all__ = ["PPOPorElo", "CAMINHO"]

# ⚠ O caminho QUALIFICADO, para o `class_name` do cfg. O `resolve_callable` do rsl_rl
# aceita `"modulo:Atributo"` (`rsl_rl/utils/utils.py:103`). Uma string, e não a classe,
# porque o logger do mjlab despeja o cfg em disco e uma classe não serializa.
CAMINHO = "g1_limpo.algoritmo:PPOPorElo"

# a cada quantas chamadas o diagnóstico vai para o log
_INTERVALO = 50


class PPOPorElo(PPO):
    """PPO com a vantagem normalizada por grupo de elo. Ver o docstring do módulo."""

    _chamadas = 0

    def compute_returns(self, obs) -> None:
        """Recalcula a normalização da vantagem, por grupo, sobre a vantagem CRUA.

        ⚠ Chama o `super()` primeiro e REFAZ, em vez de reimplementar o GAE. O cálculo
        de retorno do rsl_rl é o que queremos; o que não queremos é só a normalização
        final. Reimplementar o GAE aqui seria uma segunda fonte de verdade para a parte
        que está CERTA, e ela derivaria no primeiro upgrade.
        """
        super().compute_returns(obs)
        st = self.storage

        # ⚠ O ELO INTERNO DO CRÍTICO, e não o publicado do ator (spec §6.1): nas duas
        # esperas o ator vê ANDAR, mas a espera final carrega retorno de MANIPULAÇÃO —
        # agrupá-la com a locomoção inflaria o desvio da locomoção, que é exatamente o
        # defeito que esta classe existe para evitar.
        criticos = st.observations["critic"]
        bloco = criticos[..., fatia_do_elo_interno(criticos.shape[-1])]

        # ⚠ INVARIANTE, e ele é a trava de runtime. Se a fatia deixar de apontar para o
        # one-hot — alguém acrescentou um termo depois do `caixa`, ou o molde mudou —
        # isto falha na PRIMEIRA iteração, e não vira um treino silenciosamente errado
        # duas mil iterações depois.
        soma = bloco.sum(-1)
        assert torch.allclose(soma, torch.ones_like(soma), atol=1e-3), (
            f"a fatia do elo não é um one-hot (soma média {float(soma.mean()):.4f}); "
            f"alguém acrescentou observação ao crítico DEPOIS do elo_interno?")

        # ⚠ Sobre a vantagem CRUA, e não sobre a que o `super()` já normalizou:
        # renormalizar o que já foi normalizado misturaria as duas escalas.
        crua = st.returns - st.values
        eh_loco = (bloco.argmax(-1) == ANDAR).unsqueeze(-1)

        saida = crua.clone()
        desvios: dict[str, float] = {}
        for nome, mascara in (("loco", eh_loco), ("manip", ~eh_loco)):
            # ⚠ `< 2` e não `== 0`: com uma amostra só o `std` é NaN, e o NaN se
            # propaga para o gradiente inteiro no passo seguinte.
            if int(mascara.sum()) < 2:
                continue
            a = crua[mascara]
            desvios[nome] = float(a.std())
            saida[mascara] = (a - a.mean()) / (a.std() + 1e-8)
        st.advantages = saida

        PPOPorElo._chamadas += 1
        if PPOPorElo._chamadas % _INTERVALO == 1 and len(desvios) == 2:
            razao = desvios["loco"] / max(desvios["manip"], 1e-8)
            print(f"[ADV] std cru  loco={desvios['loco']:.4f}  "
                  f"manip={desvios['manip']:.4f}  razao={razao:.3f}", flush=True)
