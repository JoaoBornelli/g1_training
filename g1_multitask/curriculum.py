"""Orquestrador: quem escolhe tarefa e nível, mede competência, e destrava.

É UM curriculum term só, e não dois, porque medir e sortear são a mesma decisão: a
competência medida no episódio que acabou define a distribuição do que começa agora.

⚠️ **Por que curriculum term e não evento nem comando.** No reset a ordem é currículo
(`manager_based_rl_env.py:554`) -> eventos (`:560`) -> comando (`:581`). O evento
`reset_segurando` precisa saber QUAL tarefa o env tirou pra decidir se ele nasce com a
caixa nas mãos; se o sorteio ficasse no termo de comando, aconteceria 21 linhas depois
de a informação ser necessária.

Interface com o resto do pacote:

    env.tarefa_sorteada   [num_envs] long          — lida pelo evento e pelo comando
    env.nivel             dict eixo -> [num_envs]  — lida pelo comando
    env.success_buf       [num_envs] float         — ESCRITA por metrics.py, lida aqui

O molde é o `PlrHeights` da Lift (`g1_training/common/curriculums.py`): mesma fórmula
rank-based, mesmo piso `ρ/L`, mesma disciplina de criar buffers no `__init__`. Duas
diferenças de fundo:

  1. `scores` deixa de ser `[L]` e passa a ser um por CÉLULA `(tarefa, eixo)`;
  2. a performance vem de `env.success_buf` — **fato físico** — e não da soma de
     reward do episódio. É isso que faz ajustar peso entre blocos ser Categoria A
     (grátis) em vez de Categoria C disfarçada (§15).

E ganha o que o `PlrHeights` não tem: `state_dict`/`load_state_dict`. Sem isso, com
blocos de 2k-3k, o currículo voltaria ao nível 0 **10 a 15 vezes**.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

# Grafo de dependências (§7b). `parado` -> `andar` exige competência em TODOS os
# níveis de push, não só no atual. `reorientar` e `pegar` são IRMÃOS, ambos filhos do
# `andar`: manipulação não-preênsil é mais difícil de controlar, e como gate duro ela
# travaria `pegar`, `carregar` e `botar` atrás dela.
FILHOS: dict[int, tuple[int, ...]] = {
    T.PARADO: (T.ANDAR,),
    # ⚠️ `pegar` ANTES de `reorientar`, e a ordem importa em 1 evento. Os dois são
    # irmãos e nenhum trava o outro, mas o `andar` abre UM por evento (F9), e só o
    # `pegar` está na cadeia crítica: é ele que produz o estado de segurar, que o
    # `parado c/ caixa` consome, que leva ao `andar c/ caixa` e ao `botar`. Abrindo
    # o `reorientar` primeiro, a cadeia até o `botar` vira 10 eventos; abrindo o
    # `pegar` primeiro, 9 — que é o número da §7b.
    T.ANDAR: (T.PEGAR, T.REORIENTAR),
    T.PEGAR: (T.PARADO_CAIXA,),
    T.PARADO_CAIXA: (T.ANDAR_CAIXA,),
    T.ANDAR_CAIXA: (T.BOTAR,),
    T.REORIENTAR: (),
    T.BOTAR: (),
}

PUSH = "push"
"""O eixo global. Único sem piso `ρ/L` (é aninhado por construção: o nível 4 contém o
0), único com RECUO, e o único cuja competência se mede sobre TODAS as tarefas em vez
de uma — porque ele se aplica a todos os envs ao mesmo tempo."""


class Orquestrador:
    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv"):
        p = cfg.params
        k = p["curriculum"]
        dev = env.device
        self.dev = dev
        self.rho = float(k.rho)
        self.beta = float(k.focus_beta)
        self.alpha = float(k.ema_alpha)
        self.alpha_lenta = float(k.ema_alpha_lenta)
        self.limiar = float(k.limiar_competencia)
        self.congela_queda = float(k.congela_queda)
        self.descongela = float(k.descongela_dist_pico)
        self.platô_amostras = int(k.platô_amostras)
        self.alarme_transicoes = float(k.alarme_transicoes)
        self.min_amostras_evento = int(p.get("min_amostras_evento", 200))
        self.verboso = bool(p.get("verboso", True))
        """Episódios NOVOS que a tarefa precisa acumular desde o último
        destravamento dela. Não é conservadorismo: abrir uma tarefa-filha não mexe
        nas células da mãe, então o `min` da mãe continua satisfeito e ela
        dispararia a cada reset — o que daria dezenas de destravamentos na mesma
        iteração em vez da cascata de ~10 eventos do desenho."""

        # ---------------- estado INVARIANTE a num_envs (vai no checkpoint) --------
        self.celulas: list[tuple[int, str]] = [
            (t, eixo) for t, eixos in T.AXES.items() for eixo in eixos
        ] + [(T.PARADO, PUSH)]
        self.perf: dict[tuple[int, str], torch.Tensor] = {}
        self.ref: dict[tuple[int, str], torch.Tensor] = {}
        """Referência contra a qual a queda é medida. Era `pico`, o máximo corrido;
        virou uma EMA de `alpha_lenta` na S3. O nome mudou junto porque `pico` deixou
        de descrever o conteúdo."""
        self.amostras: dict[tuple[int, str], torch.Tensor] = {}
        self.congelado: dict[tuple[int, str], torch.Tensor] = {}
        self.abertos: dict[tuple[int, str], int] = {}
        for cel in self.celulas:
            n = len(self._niveis(cel))
            # EMA começa em 0.0, não em 0.5. O 0.5 era um prior otimista e ele
            # ARMAVA O CONGELAMENTO SOZINHO: o `_congelamento` punha a referência em
            # max(ref, perf), então o 0.5 inicial virava referência; a medição real
            # levava a EMA a ~0 (o robô cai); a queda de 0.5 passava do limiar de
            # 0.10 e a célula congelava por volta da 8ª atualização. Medido na
            # sessão de 30/07: `parado_push/congeladas = 1.0` na iteração 29, sem
            # nenhuma regressão real. Zero é honesto — célula não medida vale
            # "sem competência", não "meia competência".
            #
            # Zero continua honesto com a referência nova: uma EMA lenta partindo de
            # zero só sobe, então `queda = ref − perf` nasce negativa e nenhuma
            # célula pode congelar antes de a referência subir de verdade.
            self.perf[cel] = torch.zeros(n, device=dev)
            self.ref[cel] = torch.zeros(n, device=dev)
            self.amostras[cel] = torch.zeros(n, device=dev)
            self.congelado[cel] = torch.zeros(n, dtype=torch.bool, device=dev)
            self.abertos[cel] = 1
        self.abertas: list[int] = [T.PARADO]
        self.rr: dict[int, int] = {t: 0 for t in T.AXES}
        self.desde_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self.eventos = 0
        self.transicoes_sem_evento = 0.0
        # --- S15: diagnóstico. Nenhuma destas séries muda o treino. ---
        self.iteracoes_desde_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self.amostras_no_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self.chamadas_congelado: dict[tuple, float] = {}
        self._passo_ultimo_evento: dict[int, int] = {t: 0 for t in T.AXES}
        self._passos_por_iter = float(
            int(getattr(env, "num_envs", 1)) * 24)
        """Passos de ambiente por iteração de PPO. `num_steps_per_env = 24` é o valor
        do `rl_cfg`; multiplicado pelos envs dá o incremento de `common_step_counter`
        por iteração. Serve só para converter a série de diagnóstico de passos para
        iterações, que é a unidade em que o orçamento de 30 000 é escrito."""

        # ---------------- estado POR-ENV (descartável, não vai no checkpoint) -----
        env.tarefa_sorteada = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        env.nivel = {eixo: torch.zeros(env.num_envs, dtype=torch.long, device=dev)
                     for eixo in T.LEVELS}
        # Os três buffers que ligam o currículo à CENA e à FÍSICA (S1). Ficam aqui,
        # junto de `env.nivel`, porque são a mesma classe de estado: por-env,
        # re-derivados a cada reset, e portanto FORA do `state_dict`.
        #
        #   plr_shelf_top   escrito por `_amostrar`, lido por `reset_scene_plr`
        #   plr_rest_z      escrito por `reset_scene_plr`, lido pelo `lift_reward`
        #   peso_amostrado  escrito por `payload_por_nivel`, lido pela S14 (v_max)
        #
        # `peso_amostrado` nasce em 1.0 e não em 0.0: é uma MASSA, e massa zero não é
        # um estado válido pra quem lê. 1.0 é o nível 0 do eixo de peso.
        env.plr_shelf_top = torch.zeros(env.num_envs, device=dev)
        env.plr_rest_z = torch.zeros(env.num_envs, device=dev)
        env.peso_amostrado = torch.ones(env.num_envs, device=dev)
        self._alturas = torch.tensor(T.LEVELS["altura"], device=dev)
        # Os dois do eixo `push` (S2). Mesma classe: por-env, re-derivados no reset,
        # fora do `state_dict`. O NÍVEL de push é global (`self.push_nivel`);
        # `push_nivel_t` é a cópia por env que os eventos leem sem precisar do
        # objeto do currículo.
        env.push_fator = torch.zeros(env.num_envs, device=dev)
        env.push_nivel_t = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        self._visitou = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        # Marca do `common_step_counter` na última medição, pra contar transição por
        # DIFERENÇA. Fica FORA do checkpoint de propósito: o contador do env zera em
        # processo novo, e esta marca zera com ele — assim o resume não vê um salto.
        self._ultimo_passo = 0
        # o nível de push é GLOBAL: um número, não um por env
        self.push_nivel = 0

    # ------------------------------------------------------------------ helpers
    def _base(self, t: int, eixo: str) -> int:
        """Índice INICIAL da tarefa nesse eixo, dentro de `T.LEVELS[eixo]`.

        Existe porque há DUAS convenções de índice, e confundi-las é bug silencioso
        (foi, em 30/07): as células deste orquestrador indexam a partir do início da
        tarefa (nível 0 da célula = o primeiro nível QUE ELA USA), enquanto
        `env.nivel[eixo]` — que o termo de comando lê — guarda índice ABSOLUTO em
        `T.LEVELS[eixo]`.

        A diferença só é diferente de zero na `distancia` de quem ANDA, que começa em
        0.3 (índice 1). Foi exatamente ali que o bug apareceu: o orquestrador sorteava
        "nível 0 da célula" e o comando lia `T.LEVELS['distancia'][0] = 0.0`, ou seja
        destino em cima do próprio robô — e o `twist` mandava ficar parado numa tarefa
        de locomoção."""
        if eixo == PUSH:
            return 0
        return T.AXES[t][eixo]

    def _niveis(self, cel) -> tuple[float, ...]:
        t, eixo = cel
        if eixo == PUSH:
            return T.LEVELS[PUSH]
        return T.axis_levels(t, eixo)

    def _dist(self, cel) -> torch.Tensor:
        """P(nível) rank-based sobre os níveis DESTRAVADOS. Fórmula do `PlrHeights`:

            P = ρ/L + (1−ρ) · rank^(−1/β) / Σ

        ρ é piso uniforme: toda altura já vista recebe pelo menos ρ/L, e é isso que
        impede o esquecimento. rank 1 = mais difícil = mais massa."""
        L = self.abertos[cel]
        if L == 1:
            return torch.ones(1, device=self.dev)
        dificuldade = 1.0 - self.perf[cel][:L]
        # célula congelada puxa massa: ela é o que está travando o `min` da tarefa
        dificuldade = dificuldade + self.congelado[cel][:L].float()
        ordem = torch.argsort(dificuldade, descending=True)
        rank = torch.empty(L, device=self.dev)
        rank[ordem] = torch.arange(1, L + 1, device=self.dev, dtype=rank.dtype)
        foco = rank.pow(-1.0 / self.beta)
        foco = foco / foco.sum()
        P = self.rho / L + (1.0 - self.rho) * foco
        return P / P.sum()

    def _min_tarefa(self, t: int) -> float:
        """Competência da tarefa = `min` sobre TODOS os níveis destravados dela.

        `min`, não média: dominar o nível fácil e ignorar o difícil não passa. E é
        absoluto em 0.90, sem escape hatch — o mesmo limiar em todas as tarefas."""
        vals = []
        for eixo in T.AXES[t]:
            cel = (t, eixo)
            vals.append(self.perf[cel][: self.abertos[cel]])
        if not vals:
            return 1.0          # `parado` não tem eixo próprio; quem manda é o push
        return float(torch.cat(vals).min())

    def _push_competente(self) -> bool:
        """TODOS os níveis de push destravados em competência (gate do `parado`)."""
        cel = (T.PARADO, PUSH)
        return bool((self.perf[cel][: self.abertos[cel]] >= self.limiar).all())

    # ------------------------------------------------------------------- medir
    def _medir(self, env, env_ids: torch.Tensor) -> None:
        """EMA da taxa de sucesso na célula que cada env que terminou estava treinando."""
        valido = self._visitou[env_ids]
        if not bool(valido.any()):
            self._visitou[env_ids] = True
            return
        ids = env_ids[valido]
        sucesso = env.success_buf[ids]
        tarefa = env.tarefa_sorteada[ids]

        for t in torch.unique(tarefa).tolist():
            m = tarefa == t
            self.desde_evento[t] += float(m.sum())
            for eixo in T.AXES[t]:
                cel = (t, eixo)
                # `env.nivel` guarda índice ABSOLUTO em `T.LEVELS[eixo]`; a célula
                # indexa a partir do início DELA. Ver `_base`.
                lv = env.nivel[eixo][ids][m] - self._base(t, eixo)
                s = sucesso[m]
                for nivel in torch.unique(lv).tolist():
                    if nivel < 0 or nivel >= self.abertos[cel]:
                        continue
                    media = float(s[lv == nivel].mean())
                    p = self.perf[cel]
                    p[nivel] = (1.0 - self.alpha) * p[nivel] + self.alpha * media
                    self.amostras[cel][nivel] += float((lv == nivel).sum())
                    self._congelamento(cel, nivel)

        # push: competência do NÍVEL ATUAL, medida só nos envs da tarefa `parado`.
        #
        # ⚠️ Era `sucesso.mean()` sobre TODAS as tarefas. Medido em 05/08 no harness
        # ruidoso: abrir o `andar` derrubava esta perf de 0.900 para 0.696, queda de
        # 0.206. A queda é REAL, e não desvio do máximo — a S3 não a conserta. Mas ela
        # não mede regressão de robustez a push: ela mede a COMPOSIÇÃO da população,
        # porque a tarefa nova é mais difícil que a antiga.
        #
        # Consequência de manter a média global: `perf[push]` é comparada com um
        # limiar ABSOLUTO de 0.90 no `_destravar_push` e no `_push_competente`, e com
        # população mista esse 0.90 é inatingível por construção.
        #
        # Restringir ao `parado` deixa a população estável, e aí a queda volta a
        # significar regressão. Na Fase 0 o resultado é IDÊNTICO ao de antes, porque só
        # o `parado` está aberto — o gate da Fase 0 não muda.
        #
        # A célula continua sendo `(PARADO, PUSH)`, e o `parado` é justamente a tarefa
        # cujo critério de sucesso é sobreviver: é o que o push testa.
        cel = (T.PARADO, PUSH)
        so_parado = tarefa == T.PARADO
        n_parado = int(so_parado.sum())
        if n_parado > 0:
            media = float(sucesso[so_parado].mean())
            p = self.perf[cel]
            p[self.push_nivel] = ((1.0 - self.alpha) * p[self.push_nivel]
                                  + self.alpha * media)
            self.amostras[cel][self.push_nivel] += float(n_parado)
            self._congelamento(cel, self.push_nivel)
        # Transições EXATAS desde a medição anterior, pelo contador do próprio env.
        # A versão de antes fazia `len(ids) * max_episode_length` e contava cada env
        # que terminou como um episódio COMPLETO de 1000 passos. Com episódio real de
        # 11 passos isso inflava ~90x — medido 30/07: o alarme acusava 1.15e9 contra
        # 1.4e7 transições reais na iteração 146, ou seja disparava ~1900 iterações
        # antes da hora e enchia o log de centenas de linhas.
        passo = int(env.common_step_counter)
        self.transicoes_sem_evento += (float(passo - self._ultimo_passo)
                                       * float(env.num_envs))
        self._ultimo_passo = passo
        self._visitou[env_ids] = True

    def _congelamento(self, cel, nivel: int) -> None:
        """Queda > `congela_queda` contra a MÉDIA LENTA congela a célula (S3).

        A referência é uma EMA de `alpha_lenta`, e não o máximo corrido. O máximo tem
        desvio de +2.5σ a +3σ; com σ de 0.037 a 0.062 esse desvio SOZINHO passa do
        limiar de 0.10, e a célula congelava sem regressão nenhuma.

        Com referência sem desvio, o 0.10 volta a significar 3σ em p = 0.90. Em
        p = 0.50 ele vale 1.6σ, e ali ainda há falso positivo em ~5% das medições.
        Aceito: o congelamento não re-trava nível, e solta a < 0.05.

        Não usar mediana em janela: ela exige buffer circular por nível, aumenta o
        `state_dict`, e não melhora o resultado.

        Congelar NÃO re-trava o nível (Decisão 2): ele continua no sorteio, e ainda
        recebe massa extra, porque é ele que está travando o `min` da tarefa. O que a
        marca faz é aparecer no log e bloquear novo destravamento até recuperar."""
        p = float(self.perf[cel][nivel])
        r = float(self.ref[cel][nivel])
        a = self.alpha_lenta
        r = (1.0 - a) * r + a * p
        self.ref[cel][nivel] = r
        queda = r - p
        if queda > self.congela_queda:
            self.congelado[cel][nivel] = True
        elif queda < self.descongela:
            self.congelado[cel][nivel] = False

    # --------------------------------------------------------------- destravar
    def _destravar(self, t: int) -> str | None:
        """UM destravamento na tarefa `t`, na prioridade do F9. Devolve o rótulo."""
        # 1. TAREFA NOVA primeiro — abre trabalho paralelo que não depende de nada
        #    mais. Esgotar o eixo antes exigiria andar 2 m com heading 360° ANTES de
        #    encostar na caixa.
        for filho in FILHOS[t]:
            if filho not in self.abertas:
                self.abertas.append(filho)
                return f"abriu_{T.NAMES[filho]}"

        # 2. UM EIXO, o de maior competência (mais folga), empate por ROUND-ROBIN.
        #    O desempate é obrigatório, não cosmético: no 1º evento há um nível em
        #    cada eixo, todo episódio tem a mesma configuração, e cada célula recebe
        #    o MESMO fluxo de dados -> EMAs idênticas -> empate por construção.
        eixos = [e for e in T.AXIS_ORDER if e in T.AXES[t]
                 and self.abertos[(t, e)] < len(self._niveis((t, e)))]
        if not eixos:
            return None
        folga = {e: self._min_cel((t, e)) for e in eixos}
        melhor = max(folga.values())
        empatados = [e for e in eixos if abs(folga[e] - melhor) < 1e-6]
        if len(empatados) == 1:
            escolhido = empatados[0]
        else:
            # round-robin ciclando a ordem fixa: mesma filosofia breadth-first que o
            # desenho já usa no nível das tarefas
            ordem = [e for e in T.AXIS_ORDER if e in empatados]
            escolhido = ordem[self.rr[t] % len(ordem)]
            self.rr[t] += 1
        self.abertos[(t, escolhido)] += 1
        return f"{T.NAMES[t]}_{escolhido}_n{self.abertos[(t, escolhido)] - 1}"

    def _min_cel(self, cel) -> float:
        return float(self.perf[cel][: self.abertos[cel]].min())

    def _destravar_push(self) -> str | None:
        cel = (T.PARADO, PUSH)
        if self.abertos[cel] >= len(T.LEVELS[PUSH]):
            return None
        self.abertos[cel] += 1
        self.push_nivel = self.abertos[cel] - 1
        return f"push_n{self.push_nivel}"

    # ------------------------------------------------------------------ sortear
    def _amostrar(self, env, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        abertas = torch.tensor(self.abertas, device=self.dev)
        forcada = getattr(env, "task_dist", None)
        if forcada is not None:
            tarefa = torch.multinomial(forcada.to(self.dev), n, replacement=True)
        else:
            tarefa = abertas[torch.randint(0, len(self.abertas), (n,), device=self.dev)]
        env.tarefa_sorteada[env_ids] = tarefa

        for eixo in T.LEVELS:
            if eixo == PUSH:
                continue
            alvo = env.nivel[eixo]
            # O eixo que a tarefa sorteada NÃO possui recebe o nível MAIS FÁCIL
            # (índice absoluto 0 — `T.LEVELS` é ordenado do fácil pro difícil).
            #
            # ⚠️ Sem esta linha o env mantinha o valor do episódio ANTERIOR dele, e
            # isso era leitura obsoleta. Enquanto só o termo de comando lia `env.nivel`
            # o estrago era limitado; desde que `plr_shelf_top` passou a sair daqui
            # (S1), o lixo decidiria a POSIÇÃO DA PRATELEIRA em `parado`, `andar`,
            # `parado c/ caixa` e `andar c/ caixa` — que não têm o eixo `altura`.
            #
            # ⚠️ Nível mais fácil, e NÃO o corrente. O corrente daria ao `andar c/
            # caixa` giros de ±180° que o currículo dele nunca mediu, e portanto
            # nunca controlou. Com o nível 0 a cena fica sempre bem definida e
            # nenhuma tarefa fica mais difícil do que a lista de eixos dela declara.
            alvo[env_ids] = 0
            for t in torch.unique(tarefa).tolist():
                if eixo not in T.AXES[t]:
                    continue
                m = tarefa == t
                cel = (t, eixo)
                sorteado = torch.multinomial(
                    self._dist(cel), int(m.sum()), replacement=True)
                # converte pra ABSOLUTO antes de escrever: é assim que o termo de
                # comando lê. Ver `_base`.
                alvo[env_ids[m]] = sorteado + self._base(t, eixo)

        # A altura do nível vira POSIÇÃO DA PRATELEIRA. O `reset_scene_plr` lê este
        # buffer no evento de reset, que roda 6 linhas depois do currículo
        # (`manager_based_rl_env.py:554` contra `:560`) — sem off-by-one.
        env.plr_shelf_top[env_ids] = self._alturas[env.nivel["altura"][env_ids]]

        # O eixo `push` vira PERTURBAÇÃO (S2). O fator é sorteado em `U(0, teto)` e não
        # fixado no teto, pela mesma razão do peso: com valor fixo o nível 4 não contém
        # o nível 0, e o push fraco desaparece do treino assim que o eixo sobe. É o
        # sorteio que torna verdadeira a afirmação do `knobs.py` de que este eixo é
        # "aninhado por construção" — ele é o único sem piso `ρ/L` justamente por isso.
        teto = float(T.LEVELS[PUSH][self.push_nivel])
        env.push_fator[env_ids] = torch.rand(len(env_ids), device=self.dev) * teto
        env.push_nivel_t[env_ids] = self.push_nivel

    # -------------------------------------------------------------------- termo
    def __call__(self, env, env_ids, **_):
        if env_ids is None or len(env_ids) == 0:
            return {}
        self._medir(env, env_ids)

        # PUSH primeiro: os 4 destravamentos dele são a Fase 0 inteira, e acontecem
        # antes de qualquer tarefa abrir. É deles que sai a medida de "quantas
        # iterações do destravamento até a competência", de graça.
        rotulos = []
        cel_push = (T.PARADO, PUSH)
        if (self.amostras[cel_push][self.push_nivel] >= self.min_amostras_evento
                and float(self.perf[cel_push][self.push_nivel]) >= self.limiar):
            r = self._destravar_push()
            if r:
                rotulos.append(r)
                self.amostras[cel_push][self.push_nivel] = 0.0

        for t in list(self.abertas):
            if self.desde_evento[t] < self.min_amostras_evento:
                continue
            if t == T.PARADO and not self._push_competente():
                continue        # o `parado` só abre o `andar` com push COMPLETO
            if any(bool(self.congelado[(t, e)].any()) for e in T.AXES[t]):
                continue        # célula congelada bloqueia novo destravamento
            if self._min_tarefa(t) < self.limiar:
                continue
            r = self._destravar(t)
            if r:
                rotulos.append(r)
                # S15 — as duas séries que a rodada precisa medir, registradas NO
                # momento do destravamento (depois elas já mudaram).
                #
                # `iteracoes_desde_evento` é o número mais valioso da rodada: sem ele,
                # "54 destravamentos em 30 000 iterações" é aposta, não plano.
                #
                # `amostras_no_evento` responde se o portão `min` decidiu por
                # COMPETÊNCIA ou por SORTE — com amostra pequena, a EMA que cruzou
                # 0.90 pode ser ruído.
                passo = int(env.common_step_counter)
                self.iteracoes_desde_evento[t] = (
                    (passo - self._passo_ultimo_evento[t]) / self._passos_por_iter)
                self._passo_ultimo_evento[t] = passo
                self.amostras_no_evento[t] = min(
                    (float(self.amostras[(t, e)][: self.abertos[(t, e)]].min())
                     for e in T.AXES[t]), default=float("nan"))
                self.desde_evento[t] = 0.0

        if rotulos:
            self.eventos += len(rotulos)
            self.transicoes_sem_evento = 0.0
            if self.verboso:
                print(f"[CURRICULO] evento {self.eventos}/{T.total_unlocks()}: "
                      f"{', '.join(rotulos)}")

        self._amostrar(env, env_ids)
        return self._log()

    def _log(self) -> dict[str, torch.Tensor]:
        d = self.dev
        out = {
            "eventos": torch.tensor(float(self.eventos), device=d),
            "tarefas_abertas": torch.tensor(float(len(self.abertas)), device=d),
            "push_nivel": torch.tensor(float(self.push_nivel), device=d),
        }
        # --- S15: diagnóstico por tarefa. Só log; não muda nada no treino. ---
        for t in T.AXES:
            nome = T.NAMES[t]
            out[f"diag/{nome}/iteracoes_desde_evento"] = torch.tensor(
                float(self.iteracoes_desde_evento[t]), device=d)
            out[f"diag/{nome}/amostras_no_evento"] = torch.tensor(
                float(self.amostras_no_evento[t]), device=d)
        for cel in self.celulas:
            t, eixo = cel
            base = f"{T.NAMES[t]}_{eixo}"
            # duração acumulada do congelamento, em chamadas de reset (S15)
            n_cong = int(self.congelado[cel][: self.abertos[cel]].sum())
            if n_cong:
                self.chamadas_congelado[cel] = self.chamadas_congelado.get(cel, 0.0) + 1
            out[f"diag/congelado_chamadas/{base}"] = torch.tensor(
                float(self.chamadas_congelado.get(cel, 0.0)), device=d)
            out[f"{base}/abertos"] = torch.tensor(float(self.abertos[cel]), device=d)
            out[f"{base}/min"] = self.perf[cel][: self.abertos[cel]].min()
            out[f"{base}/congeladas"] = self.congelado[cel].float().sum()
            for i in range(self.abertos[cel]):
                out[f"{base}/perf_n{i}"] = self.perf[cel][i]
                # PLATÔ é DIAGNÓSTICO, não portão (§14): >= 2000 amostras na célula E
                # máximo não superado. Loga; não bloqueia nem destrava.
                if float(self.amostras[cel][i]) >= self.platô_amostras:
                    out[f"{base}/plato_n{i}"] = torch.tensor(1.0, device=d)
        if self.transicoes_sem_evento > self.alarme_transicoes:
            out["ALARME_estagnacao"] = torch.tensor(1.0, device=d)
            print(f"[CURRICULO] ⚠️ ALARME DE ESTAGNAÇÃO: "
                  f"{self.transicoes_sem_evento:.2e} transições sem destravamento "
                  f"(limite {self.alarme_transicoes:.1e})")
        return out

    # -------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        """Só o que é INVARIANTE a `num_envs`. O estado por-env é descartável: qual
        nível cada env está treinando se re-sorteia no primeiro reset.

        Sem isto, com blocos de 2k-3k, o currículo voltaria ao nível 0 de 10 a 15
        vezes ao longo da run — e em silêncio."""
        chave = lambda cel: f"{cel[0]}|{cel[1]}"      # noqa: E731
        return {
            "perf": {chave(c): self.perf[c].cpu() for c in self.celulas},
            "ref": {chave(c): self.ref[c].cpu() for c in self.celulas},
            "amostras": {chave(c): self.amostras[c].cpu() for c in self.celulas},
            "congelado": {chave(c): self.congelado[c].cpu() for c in self.celulas},
            "abertos": {chave(c): self.abertos[c] for c in self.celulas},
            "abertas": list(self.abertas),
            "rr": dict(self.rr),
            "desde_evento": dict(self.desde_evento),
            "push_nivel": self.push_nivel,
            "eventos": self.eventos,
            "transicoes_sem_evento": self.transicoes_sem_evento,
        }

    def load_state_dict(self, estado: dict) -> None:
        chave = lambda cel: f"{cel[0]}|{cel[1]}"      # noqa: E731
        for c in self.celulas:
            k = chave(c)
            # ⚠️ Checkpoint anterior à S3 tem a chave `"pico"`, não `"ref"`. Ele carrega
            # sem erro e a referência começa em zero. Zero na referência significa
            # "sem regressão possível", o que é seguro: a EMA lenta só sobe a partir
            # dali. Não há código de migração de propósito.
            for nome, destino in (("perf", self.perf), ("ref", self.ref),
                                  ("amostras", self.amostras),
                                  ("congelado", self.congelado)):
                if k in estado.get(nome, {}):
                    destino[c] = estado[nome][k].to(self.dev)
            if k in estado.get("abertos", {}):
                self.abertos[c] = int(estado["abertos"][k])
        self.abertas = list(estado.get("abertas", self.abertas))
        self.rr = {int(k): int(v) for k, v in estado.get("rr", self.rr).items()}
        self.desde_evento = {int(k): float(v)
                             for k, v in estado.get("desde_evento",
                                                    self.desde_evento).items()}
        self.push_nivel = int(estado.get("push_nivel", self.push_nivel))
        self.eventos = int(estado.get("eventos", self.eventos))
        self.transicoes_sem_evento = float(
            estado.get("transicoes_sem_evento", self.transicoes_sem_evento))
        print(f"[CURRICULO] retomado: {self.eventos}/{T.total_unlocks()} eventos, "
              f"{len(self.abertas)} tarefas abertas, push nível {self.push_nivel}")

    def reset(self, env_ids=None):
        pass    # o estado do currículo PERSISTE entre resets — é o método
