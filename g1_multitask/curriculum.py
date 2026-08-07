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
    env.nivel             dict eixo -> [num_envs]  — índice ABSOLUTO em T.LEVELS
    env.teto_velocidade   [num_envs] float         — lido pela subclasse de comando
    env.dr_peso           [num_envs] bool          — lido pelo `payload_dr`
    env.plr_shelf_top     [num_envs] float         — lido pelo `reset_scene_plr`
    env.success_buf       [num_envs] float         — ESCRITA por metrics.py, lida aqui

O molde do sorteio de nível é o `PlrHeights` da Lift
(`g1_training/common/curriculums.py`): mesma fórmula rank-based, mesmo piso `ρ/L`,
mesma disciplina de criar buffers no `__init__`. Duas diferenças de fundo:

  1. `scores` é por CÉLULA `(tarefa, eixo)`, e não `[L]`;
  2. a performance vem de `env.success_buf` — **fato físico** — e não da soma de
     reward do episódio. É isso que faz ajustar peso entre blocos ser Categoria A.

E ganha o que o `PlrHeights` não tem: `state_dict`/`load_state_dict`. Sem isso, com
blocos de 2k-3k, o currículo voltaria ao nível 0 de 10 a 15 vezes.

--------------------------------------------------------------------------------
O QUE A REFORMA DE 07/08 TIROU DAQUI  (ver `EXPERIMENTO.md` §9)
--------------------------------------------------------------------------------
  - o `FILHOS` com prioridade "filho antes de eixo" e a regra F9 de um filho por
    evento -> viraram `PAIS` com junção AND;
  - o round-robin e o `AXIS_ORDER` -> um eixo por tarefa, nada a desempatar;
  - o `_min_tarefa` e o `_min_cel` -> o portão lê o nível CORRENTE;
  - o condicionamento do `_medir` -> não há outro eixo para condicionar;
  - o congelamento inteiro (`ref`, `congelado`, histerese) -> ele era o único
    mecanismo capaz de bloquear a abertura de um filho, e a referência já é EMA
    lenta desde a S3, que suaviza sozinha;
  - a célula `(PARADO, PUSH)` e o eixo `push` -> o push virou evento fixo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

# Grafo de dependências (§9). Quatro camadas. Declarado por PAIS, e não por FILHOS,
# porque a camada 3 é uma JUNÇÃO: o `locomover_carregando` exige os DOIS pais.
#
#   1  locomover                    nasce aberta
#   2  pegar        reorientar      abrem JUNTOS (mesmo pai, mesmo evento)
#   3  locomover_carregando         pais: pegar E reorientar
#   4  botar                        pai: locomover_carregando
#
# A ordem tem razão física. O `reorientar` exige aproximar, tocar e aplicar força. O
# `pegar` exige tudo isso, mais força normal suficiente e mais erguer.
PAIS: dict[int, tuple[int, ...]] = {
    T.LOCOMOVER: (),
    T.PEGAR: (T.LOCOMOVER,),
    T.REORIENTAR: (T.LOCOMOVER,),
    T.LOCOMOVER_CARREGANDO: (T.PEGAR, T.REORIENTAR),
    T.BOTAR: (T.LOCOMOVER_CARREGANDO,),
}

EIXOS: tuple[str, ...] = tuple(
    dict.fromkeys(e for eixos in T.AXES.values() for e in eixos))
"""Os eixos que existem de fato, na ordem de primeira aparição.

Derivado de `T.AXES`, e não digitado. O `peso` NÃO aparece aqui: ele está em
`T.LEVELS` como tabela de DR, mas não é eixo — não tem célula, não tem EMA e não tem
portão."""


class Orquestrador:
    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv"):
        p = cfg.params
        k = p["curriculum"]
        dev = env.device
        self.dev = dev
        self.rho = float(k.rho)
        self.beta = float(k.focus_beta)
        self.alpha = float(k.ema_alpha)
        self.limiar = float(k.limiar_competencia)
        self.platô_amostras = int(k.platô_amostras)
        self.alarme_transicoes = float(k.alarme_transicoes)
        self.min_amostras_evento = int(p.get("min_amostras_evento", 200))
        self.verboso = bool(p.get("verboso", True))
        """Episódios NOVOS que a tarefa precisa acumular desde o último evento dela.
        Não é conservadorismo: sem ele a tarefa dispararia a cada reset."""

        # ---------------- estado INVARIANTE a num_envs (vai no checkpoint) --------
        self.celulas: list[tuple[int, str]] = [
            (t, T.eixo_de(t)) for t in T.AXES
        ]
        self.perf: dict[tuple[int, str], torch.Tensor] = {}
        self.amostras: dict[tuple[int, str], torch.Tensor] = {}
        self.abertos: dict[tuple[int, str], int] = {}
        for cel in self.celulas:
            n = len(self._niveis(cel))
            # EMA começa em 0.0, não em 0.5. Célula não medida vale "sem competência",
            # não "meia competência". O 0.5 era um prior otimista, e ele armava o
            # congelamento sozinho no desenho antigo.
            self.perf[cel] = torch.zeros(n, device=dev)
            self.amostras[cel] = torch.zeros(n, device=dev)
            self.abertos[cel] = 1

        self.abertas: list[int] = [T.LOCOMOVER]
        self.eventos_tarefa: dict[int, int] = {t: 0 for t in T.AXES}
        """Quantos eventos cada tarefa já teve. **É o portão do filho.**

        Um inteiro, monotônico. O primeiro evento de uma tarefa dispara exatamente
        quando ela chega ao limiar na configuração mais fácil dela — portanto
        `eventos_tarefa[P] >= 1` significa "P passou do nível 0".

        O portão do filho nunca olha nível difícil. A cadeia não trava atrás de um
        nível duro."""
        self.dr_peso: dict[int, bool] = {t: False for t in T.AXES}
        """A DR de carga desta tarefa já alargou?

        `False` = caixa de 1 kg fixo. `True` = massa em `U(1, 5)` kg.

        ⚠️ **Não é eixo.** O peso não tem célula, não tem EMA e não tem portão
        próprio, porque o sucesso não é atrelado à massa — o critério é "fez o que
        tinha que fazer". O booleano vira `True` no PRIMEIRO evento da tarefa, antes
        de o eixo específico avançar. Assim a tarefa nunca recebe as duas
        dificuldades no mesmo passo."""

        self.desde_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self.eventos = 0
        self.transicoes_sem_evento = 0.0
        self.iteracoes_desde_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self._passo_ultimo_evento: dict[int, int] = {t: 0 for t in T.AXES}
        self._passos_por_iter = 24.0
        """Incremento de `common_step_counter` por iteração de PPO.

        🔧 **Conserto de 07/08.** Era `num_envs × 24`, e o docstring antigo justificava
        assim: "multiplicado pelos envs dá o incremento por iteração". Está errado. O
        `manager_based_rl_env.py:431` faz `common_step_counter += 1` por chamada de
        `step()`, **independente de `num_envs`** — então a iteração de PPO avança o
        contador em `num_steps_per_env = 24`, e só.

        Com 4 096 envs o divisor estava 4 096× grande, e a série
        `iteracoes_desde_evento` — que o doc chama de "o número mais valioso da
        rodada" — reportava ~0 sempre. Sem ela, "24 destravamentos em N iterações" é
        aposta, não plano.

        ⚠️ O `transicoes_sem_evento` continua multiplicando por `num_envs`, e ali está
        certo: transição de ambiente é passo × env."""

        # ---------------- estado POR-ENV (descartável, fora do checkpoint) --------
        env.tarefa_sorteada = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        env.nivel = {eixo: torch.zeros(env.num_envs, dtype=torch.long, device=dev)
                     for eixo in EIXOS}
        env.plr_shelf_top = torch.zeros(env.num_envs, device=dev)
        env.plr_rest_z = torch.zeros(env.num_envs, device=dev)
        env.peso_amostrado = torch.ones(env.num_envs, device=dev)
        env.dr_peso = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        env.teto_velocidade = torch.full(
            (env.num_envs,), float(T.LEVELS["velocidade"][0]), device=dev)
        """Teto de `lin_vel_x`/`lin_vel_y` por env, lido pela subclasse de comando.

        Escrito aqui, e não lido lá de `T.LEVELS`, porque o nível é por env e por
        tarefa: o `locomover` e o `locomover_carregando` têm células independentes."""

        self._alturas = torch.tensor(T.LEVELS["altura"], device=dev)
        self._velocidades = torch.tensor(T.LEVELS["velocidade"], device=dev)
        self._visitou = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        self._ultimo_passo = 0

    # ------------------------------------------------------------------ helpers
    def _base(self, t: int, eixo: str) -> int:
        """Índice INICIAL da tarefa nesse eixo, dentro de `T.LEVELS[eixo]`.

        Existe porque há DUAS convenções de índice: as células indexam a partir do
        início da tarefa, e `env.nivel[eixo]` guarda índice ABSOLUTO. Hoje todos os
        inícios são 0, mas a função fica — confundi-las já foi bug silencioso."""
        return T.AXES[t][eixo]

    def _niveis(self, cel) -> tuple[float, ...]:
        t, eixo = cel
        return T.axis_levels(t, eixo)

    def _dist(self, cel) -> torch.Tensor:
        """P(nível) rank-based sobre os níveis DESTRAVADOS. Fórmula do `PlrHeights`:

            P = ρ/L + (1−ρ) · rank^(−1/β) / Σ

        ρ é piso uniforme: todo nível já visto recebe pelo menos ρ/L, e é isso que
        impede o esquecimento. rank 1 = mais difícil = mais massa."""
        L = self.abertos[cel]
        if L == 1:
            return torch.ones(1, device=self.dev)
        dificuldade = 1.0 - self.perf[cel][:L]
        ordem = torch.argsort(dificuldade, descending=True)
        rank = torch.empty(L, device=self.dev)
        rank[ordem] = torch.arange(1, L + 1, device=self.dev, dtype=rank.dtype)
        foco = rank.pow(-1.0 / self.beta)
        foco = foco / foco.sum()
        P = self.rho / L + (1.0 - self.rho) * foco
        return P / P.sum()

    def _topo(self, cel) -> int:
        """Índice do nível corrente — o último aberto. É o que o portão lê."""
        return self.abertos[cel] - 1

    # ------------------------------------------------------------------- medir
    def _medir(self, env, env_ids: torch.Tensor) -> None:
        """EMA da taxa de sucesso na célula que cada env que terminou treinava.

        ⚠️ **Sem condicionamento.** No desenho de dois eixos, medir um eixo exigia os
        outros no nível base, senão a célula media marginalizada e o portão travava
        (o bug de 06/08). Com um eixo por tarefa não há outro eixo, e as cinco linhas
        do condicionamento saem.

        O que resta de marginalização é a DR de carga, e ela é aceita: depois do
        alargamento a distribuição PARA de se mover, então a política converge contra
        ela. O portão de 0,90 passa a significar "0,90 sobre a faixa inteira de
        carga", que é a tarefa real."""
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
            eixo = T.eixo_de(t)
            cel = (t, eixo)
            # `env.nivel` guarda índice ABSOLUTO; a célula indexa a partir do início
            # dela. Ver `_base`.
            lv = env.nivel[eixo][ids][m] - self._base(t, eixo)
            s = sucesso[m]
            for nivel in torch.unique(lv).tolist():
                if nivel < 0 or nivel >= self.abertos[cel]:
                    continue
                media = float(s[lv == nivel].mean())
                p = self.perf[cel]
                p[nivel] = (1.0 - self.alpha) * p[nivel] + self.alpha * media
                self.amostras[cel][nivel] += float((lv == nivel).sum())

        # Transições EXATAS desde a medição anterior, pelo contador do próprio env.
        passo = int(env.common_step_counter)
        self.transicoes_sem_evento += (float(passo - self._ultimo_passo)
                                       * float(env.num_envs))
        self._ultimo_passo = passo
        self._visitou[env_ids] = True

    # --------------------------------------------------------------- destravar
    def _abre_filhos(self) -> list[str]:
        """Abre toda tarefa fechada cujos pais já tiveram o primeiro evento.

        É a ação 2 da regra do evento. A junção AND cai fora sozinha: o
        `locomover_carregando` tem dois pais, e o `all()` exige os dois.

        O `pegar` e o `reorientar` têm o MESMO pai, então abrem na mesma chamada. É
        isso que substitui a regra F9 de um filho por evento."""
        rotulos = []
        for f, pais in PAIS.items():
            if f in self.abertas:
                continue
            if all(self.eventos_tarefa[p] >= 1 for p in pais):
                self.abertas.append(f)
                rotulos.append(f"abriu_{T.NAMES[f]}")
        return rotulos

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

        for eixo in EIXOS:
            alvo = env.nivel[eixo]
            # O eixo que a tarefa sorteada NÃO possui recebe o nível MAIS FÁCIL
            # (índice absoluto 0 — `T.LEVELS` é ordenado do fácil pro difícil).
            #
            # ⚠️ Sem esta linha o env manteria o valor do episódio ANTERIOR dele, e
            # isso é leitura obsoleta: o `plr_shelf_top` sai daqui, então o lixo
            # decidiria a POSIÇÃO DA PRATELEIRA numa tarefa sem eixo de altura.
            #
            # ⚠️ Nível mais fácil, e NÃO o corrente. O corrente daria à tarefa uma
            # dificuldade que o currículo dela nunca mediu.
            alvo[env_ids] = 0
            for t in torch.unique(tarefa).tolist():
                if eixo not in T.AXES[t]:
                    continue
                m = tarefa == t
                cel = (t, eixo)
                sorteado = torch.multinomial(
                    self._dist(cel), int(m.sum()), replacement=True)
                alvo[env_ids[m]] = sorteado + self._base(t, eixo)

        # A altura do nível vira POSIÇÃO DA PRATELEIRA. O `reset_scene_plr` lê este
        # buffer no evento de reset, que roda 6 linhas depois do currículo
        # (`manager_based_rl_env.py:554` contra `:560`) — sem off-by-one.
        env.plr_shelf_top[env_ids] = self._alturas[env.nivel["altura"][env_ids]]
        # O teto de velocidade vira a faixa do comando sorteado, por env.
        env.teto_velocidade[env_ids] = self._velocidades[
            env.nivel["velocidade"][env_ids]]
        # A DR de carga: por env, a partir do booleano da tarefa sorteada.
        largou = torch.tensor([self.dr_peso[t] for t in range(T.NUM_TASKS)],
                              dtype=torch.bool, device=self.dev)
        env.dr_peso[env_ids] = largou[tarefa]

    # -------------------------------------------------------------------- termo
    def __call__(self, env, env_ids, **_):
        if env_ids is None or len(env_ids) == 0:
            return {}
        self._medir(env, env_ids)

        # ------------------------------------------------------ regra do evento
        #   condição:  episódios(T) ≥ 200  E  perf[T][topo] ≥ 0,90
        #   ação 1:    se a DR de peso está fechada:  abre a DR
        #              senão:                         abertos[T] += 1
        #   ação 2:    abre todo filho cujos pais já tiveram o 1º evento
        #
        # As duas ações rodam no MESMO evento. A tarefa nova começa no nível 0
        # enquanto a tarefa mãe avança. Nada serializa.
        rotulos: list[str] = []
        for t in list(self.abertas):
            if self.desde_evento[t] < self.min_amostras_evento:
                continue
            cel = (t, T.eixo_de(t))
            if float(self.perf[cel][self._topo(cel)]) < self.limiar:
                continue

            tem_dr = t in T.COM_DR_PESO and not self.dr_peso[t]
            tem_eixo = self.abertos[cel] < len(self._niveis(cel))
            if not (tem_dr or tem_eixo):
                continue        # tarefa esgotada; os filhos dela já abriram

            nome = T.NAMES[t]
            if tem_dr:
                # DR primeiro, sempre. O 1º evento alarga a carga; o 2º avança o
                # eixo. A tarefa nunca recebe as duas dificuldades no mesmo passo.
                self.dr_peso[t] = True
                rotulos.append(f"{nome}_dr_peso")
            else:
                self.abertos[cel] += 1
                rotulos.append(f"{nome}_{cel[1]}_n{self.abertos[cel] - 1}")

            self.eventos_tarefa[t] += 1
            rotulos.extend(self._abre_filhos())

            passo = int(env.common_step_counter)
            self.iteracoes_desde_evento[t] = (
                (passo - self._passo_ultimo_evento[t]) / self._passos_por_iter)
            self._passo_ultimo_evento[t] = passo
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
        }
        for t in T.AXES:
            nome = T.NAMES[t]
            out[f"diag/{nome}/iteracoes_desde_evento"] = torch.tensor(
                float(self.iteracoes_desde_evento[t]), device=d)
            out[f"diag/{nome}/dr_peso"] = torch.tensor(
                float(self.dr_peso[t]), device=d)
        for cel in self.celulas:
            t, eixo = cel
            base = f"{T.NAMES[t]}_{eixo}"
            out[f"{base}/abertos"] = torch.tensor(float(self.abertos[cel]), device=d)
            out[f"{base}/min"] = self.perf[cel][: self.abertos[cel]].min()
            for i in range(self.abertos[cel]):
                out[f"{base}/perf_n{i}"] = self.perf[cel][i]
                # PLATÔ é DIAGNÓSTICO, não portão: >= 2000 amostras na célula. Loga;
                # não bloqueia nem destrava.
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
            "amostras": {chave(c): self.amostras[c].cpu() for c in self.celulas},
            "abertos": {chave(c): self.abertos[c] for c in self.celulas},
            "abertas": list(self.abertas),
            "eventos_tarefa": dict(self.eventos_tarefa),
            "dr_peso": dict(self.dr_peso),
            "desde_evento": dict(self.desde_evento),
            "eventos": self.eventos,
            "transicoes_sem_evento": self.transicoes_sem_evento,
        }

    def load_state_dict(self, estado: dict) -> None:
        """⚠️ Checkpoint anterior à reforma de 07/08 NÃO carrega aqui de forma útil.

        As chaves de célula mudaram junto com os índices de tarefa e os eixos, e as
        chaves antigas (`ref`, `congelado`, `rr`, `push_nivel`) não têm destino. O
        `get` com default deixa o load passar sem erro e o currículo começa do zero.
        Isso é o correto: o desenho mudou, e retomar níveis medidos contra outra
        distribuição seria pior que recomeçar."""
        chave = lambda cel: f"{cel[0]}|{cel[1]}"      # noqa: E731
        for c in self.celulas:
            k = chave(c)
            for nome, destino in (("perf", self.perf), ("amostras", self.amostras)):
                if k in estado.get(nome, {}):
                    destino[c] = estado[nome][k].to(self.dev)
            if k in estado.get("abertos", {}):
                self.abertos[c] = int(estado["abertos"][k])
        self.abertas = list(estado.get("abertas", self.abertas))
        self.eventos_tarefa = {int(k): int(v) for k, v
                               in estado.get("eventos_tarefa",
                                             self.eventos_tarefa).items()}
        self.dr_peso = {int(k): bool(v) for k, v
                        in estado.get("dr_peso", self.dr_peso).items()}
        self.desde_evento = {int(k): float(v) for k, v
                             in estado.get("desde_evento",
                                           self.desde_evento).items()}
        self.eventos = int(estado.get("eventos", self.eventos))
        self.transicoes_sem_evento = float(
            estado.get("transicoes_sem_evento", self.transicoes_sem_evento))
        print(f"[CURRICULO] retomado: {self.eventos}/{T.total_unlocks()} eventos, "
              f"{len(self.abertas)} tarefas abertas, "
              f"DR de peso em {sum(self.dr_peso.values())}/{len(T.COM_DR_PESO)}")

    def reset(self, env_ids=None):
        pass    # o estado do currículo PERSISTE entre resets — é o método
