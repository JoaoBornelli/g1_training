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

from g1_limpo.comando import ANDAR, ELOS
from g1_limpo.observacoes import N_SLOTS, fatia_do_elo, fatia_do_elo_interno

__all__ = ["PPOPorElo", "CAMINHO"]

# ⚠ O caminho QUALIFICADO, para o `class_name` do cfg. O `resolve_callable` do rsl_rl
# aceita `"modulo:Atributo"` (`rsl_rl/utils/utils.py:103`). Uma string, e não a classe,
# porque o logger do mjlab despeja o cfg em disco e uma classe não serializa.
CAMINHO = "g1_limpo.algoritmo:PPOPorElo"

# a cada quantas chamadas o diagnóstico vai para o log
_INTERVALO = 50

# ⚠ Abaixo deste desvio o slot do one-hot NUNCA acendeu de verdade: no `model_3450` o
# CARREGAR tem 0,0006 e o BOTAR 0,0012, e os slots treinados vão de 0,03 (REORIENTAR) a 0,5.
# A dobra do `load` preserva só os slots acima dele; nos de baixo a escala volta a 1.
_DESVIO_MIN_DA_DOBRA = 0.01


def _one_hots(ator, critico):
    """`(normalizador, mlp, fatias)` do one-hot, no ator e no crítico (ver `_fixa_one_hot`)."""
    for rede, fatias in ((ator, lambda D: [fatia_do_elo(D)]),
                         (critico, lambda D: [fatia_do_elo(D - N_SLOTS),
                                              fatia_do_elo_interno(D)])):
        nz = getattr(rede, "obs_normalizer", None)
        if nz is not None and hasattr(nz, "_mean"):
            yield nz, getattr(rede, "mlp", None), fatias(nz._mean.shape[-1])


class PPOPorElo(PPO):
    """PPO com a vantagem normalizada por grupo de elo, e o one-hot fora da normalização
    da observação (`_fixa_one_hot`). Ver o docstring do módulo para a vantagem."""

    _chamadas = 0

    def _fixa_one_hot(self) -> None:
        """O one-hot FORA da normalização empírica (spec g1-limpo-curriculo-de-cadeia §6).

        Os canais do one-hot ficam com média 0 e desvio 1, fixos, no ator e no crítico: o
        publicado do ator, o publicado do crítico e o `elo_interno` do crítico. Um one-hot
        é categórico, e padronizá-lo só amplifica o slot raro.

        ⚠ MEDIDO no `model_3450` da zero15: o `rsl_rl` normaliza por
        `(x − média) / (desvio + 0,01)`, com estatística acumulada o treino todo, e um
        slot que quase nunca acendeu tem desvio ~0. O slot CARREGAR tinha desvio 0,00055 e
        o BOTAR 0,0012, portanto o slot aceso entrava na rede como ~94 e ~89, no lugar de
        1. Cada troca de fase do currículo de cadeia acende um slot pouco visto.

        ⚠ LAYOUT (contado do fim, ver `observacoes.fatia_do_elo`): ator
        `[…, elo(5), caixa(10)]`; crítico `[…, elo(5), caixa(10), elo_interno(5)]`. O
        `elo` publicado do crítico se acha descontando o `elo_interno` do fim:
        `fatia_do_elo(131 − 5)` = `slice(111, 116)`; o interno é `slice(126, 131)`.

        ⚠ RODA ANTES de cada `act` e DEPOIS de cada atualização da normalização
        (`process_env_step`): o `update` do `rsl_rl` mexe em todos os canais, e a
        fixação desfaz a mexida nos do one-hot antes de a rede os ler.
        """
        for nz, _mlp, fatias in _one_hots(self.actor, self.critic):
            for sl in fatias:
                nz._mean[..., sl] = 0.0
                nz._var[..., sl] = 1.0
                nz._std[..., sl] = 1.0

    def _absorve_one_hot(self) -> None:
        """DOBRA a normalização antiga do one-hot na 1ª camada, e só depois a fixa.

        ⚠ UM CHECKPOINT ANTERIOR a 30/09 treinou com o one-hot normalizado. Fixar a
        normalização sem mais nada muda a entrada da rede. MEDIDO no `model_3450`: a 1ª
        camada do ator desloca 1,5 a 1,8 no ANDAR e no PEGAR, e 38 no REORIENTAR, cuja
        entrada acesa cai de 24,9 para 1. A dobra reescreve a 1ª camada e a função não muda:

            W'_j = W_j · (1 + ε) / (σ_j + ε)      b' = b − Σ_j W_j · μ_j / (σ_j + ε)

        ⚠ SÓ nos slots com `σ_j ≥ _DESVIO_MIN_DA_DOBRA`. Num slot que nunca acendeu (o
        CARREGAR e o BOTAR da linhagem zero), a dobra guardaria o ganho de ~95 no peso, e o
        Adam levaria milhares de passos para desfazê-lo: ali a escala volta a 1, que é o
        conserto. Num checkpoint já fixado (média 0, desvio 1), a dobra é a identidade.
        """
        with torch.no_grad():
            for nz, mlp, fatias in _one_hots(self.actor, self.critic):
                lin = mlp[0]
                for sl in fatias:
                    for j in range(sl.start, sl.stop):
                        dp = float(nz._std[0, j])
                        if dp < _DESVIO_MIN_DA_DOBRA:
                            continue
                        s = dp + nz.eps
                        lin.bias -= lin.weight[:, j] * (float(nz._mean[0, j]) / s)
                        lin.weight[:, j] *= (1.0 + nz.eps) / s
        self._fixa_one_hot()

    def load(self, loaded_dict, load_cfg, strict):
        saida = super().load(loaded_dict, load_cfg, strict)
        self._absorve_one_hot()
        return saida

    def act(self, obs):
        self._fixa_one_hot()
        return super().act(obs)

    def process_env_step(self, *args, **kwargs):
        saida = super().process_env_step(*args, **kwargs)
        self._fixa_one_hot()
        return saida

    def compute_returns(self, obs) -> None:
        """Recalcula a normalização da vantagem, por grupo, sobre a vantagem CRUA.

        ⚠ Chama o `super()` primeiro e REFAZ, em vez de reimplementar o GAE. O cálculo
        de retorno do rsl_rl é o que queremos; o que não queremos é só a normalização
        final. Reimplementar o GAE aqui seria uma segunda fonte de verdade para a parte
        que está CERTA, e ela derivaria no primeiro upgrade.

        ⚠⚠ CINCO GRUPOS, um por slot de `elo_interno` (spec `g1-limpo-dois-bits.md`
        §3.4), e não dois (`ANDAR` vs. o resto). Até a v3, a cauda `CARREGAR` (retorno
        alto, variância baixa) caía no MESMO grupo "manip" que `PEGAR` e `BOTAR`
        (curtos, variáveis) — a vantagem do elo de trabalho era dividida pelo desvio
        da cauda. Agrupar pelos 5 slots separa cada estado da sua própria escala.
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
        argmax = bloco.argmax(-1)

        # ⚠ O GRUPO "MANIPULAÇÃO INTEIRA" (revisão independente, item A7): a união
        # dos 4 slots que não são ANDAR. É o FALLBACK de quem cai num grupo RALO
        # (< 2 amostras) — REORIENTAR (5% do sorteio) e o BOTAR cedo no treino caem
        # nisso o tempo todo. Deixar a vantagem CRUA ali (o `continue` antigo) a
        # tirava da escala normalizada dos outros grupos; a normalização pela
        # manipulação inteira mantém a mesma ORDEM de grandeza sem inventar uma
        # escala própria para uma amostra só.
        mascara_manip = (argmax != ANDAR).unsqueeze(-1)
        media_manip = desvio_manip = None
        if int(mascara_manip.sum()) >= 2:
            a_manip = crua[mascara_manip]
            media_manip = a_manip.mean()
            desvio_manip = a_manip.std()

        saida = crua.clone()
        desvios: dict[str, float] = {}
        for elo_id, nome in enumerate(ELOS):
            mascara = (argmax == elo_id).unsqueeze(-1)
            n = int(mascara.sum())
            if n == 0:
                continue
            # ⚠ `< 2` e não `== 0`: com uma amostra só o `std` é NaN, e o NaN se
            # propaga para o gradiente inteiro no passo seguinte.
            if n < 2:
                if elo_id != ANDAR and desvio_manip is not None:
                    a = crua[mascara]
                    saida[mascara] = (a - media_manip) / (desvio_manip + 1e-8)
                continue
            a = crua[mascara]
            desvios[nome] = float(a.std())
            saida[mascara] = (a - a.mean()) / (a.std() + 1e-8)
        st.advantages = saida

        PPOPorElo._chamadas += 1
        if PPOPorElo._chamadas % _INTERVALO == 1 and len(desvios) >= 2:
            partes = "  ".join(f"{nome}={v:.4f}" for nome, v in desvios.items())
            print(f"[ADV] std cru por elo  {partes}", flush=True)
