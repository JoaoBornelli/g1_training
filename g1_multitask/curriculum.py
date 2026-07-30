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
        self.pico: dict[tuple[int, str], torch.Tensor] = {}
        self.amostras: dict[tuple[int, str], torch.Tensor] = {}
        self.congelado: dict[tuple[int, str], torch.Tensor] = {}
        self.abertos: dict[tuple[int, str], int] = {}
        for cel in self.celulas:
            n = len(self._niveis(cel))
            self.perf[cel] = torch.full((n,), 0.5, device=dev)
            self.pico[cel] = torch.zeros(n, device=dev)
            self.amostras[cel] = torch.zeros(n, device=dev)
            self.congelado[cel] = torch.zeros(n, dtype=torch.bool, device=dev)
            self.abertos[cel] = 1
        self.abertas: list[int] = [T.PARADO]
        self.rr: dict[int, int] = {t: 0 for t in T.AXES}
        self.desde_evento: dict[int, float] = {t: 0.0 for t in T.AXES}
        self.eventos = 0
        self.transicoes_sem_evento = 0.0

        # ---------------- estado POR-ENV (descartável, não vai no checkpoint) -----
        env.tarefa_sorteada = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        env.nivel = {eixo: torch.zeros(env.num_envs, dtype=torch.long, device=dev)
                     for eixo in T.LEVELS}
        self._visitou = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
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

        # push: competência do NÍVEL ATUAL, medida sobre TODAS as tarefas juntas
        cel = (T.PARADO, PUSH)
        media = float(sucesso.mean())
        p = self.perf[cel]
        p[self.push_nivel] = ((1.0 - self.alpha) * p[self.push_nivel]
                              + self.alpha * media)
        self.amostras[cel][self.push_nivel] += float(len(ids))
        self._congelamento(cel, self.push_nivel)
        self.transicoes_sem_evento += float(len(ids)) * float(env.max_episode_length)
        self._visitou[env_ids] = True

    def _congelamento(self, cel, nivel: int) -> None:
        """Queda > 0.10 do pico congela; volta a < 0.05 do pico descongela (§14).

        Congelar NÃO re-trava o nível (Decisão 2): ele continua no sorteio, e ainda
        recebe massa extra, porque é ele que está travando o `min` da tarefa. O que a
        marca faz é aparecer no log e bloquear novo destravamento até recuperar."""
        p = float(self.perf[cel][nivel])
        self.pico[cel][nivel] = max(float(self.pico[cel][nivel]), p)
        queda = float(self.pico[cel][nivel]) - p
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
                self.desde_evento[t] = 0.0

        if rotulos:
            self.eventos += len(rotulos)
            self.transicoes_sem_evento = 0.0
            if self.verboso:
                print(f"[CURRICULO] evento {self.eventos}/60: "
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
        for cel in self.celulas:
            t, eixo = cel
            base = f"{T.NAMES[t]}_{eixo}"
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
            "pico": {chave(c): self.pico[c].cpu() for c in self.celulas},
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
            for nome, destino in (("perf", self.perf), ("pico", self.pico),
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
        print(f"[CURRICULO] retomado: {self.eventos}/60 eventos, "
              f"{len(self.abertas)} tarefas abertas, push nível {self.push_nivel}")

    def reset(self, env_ids=None):
        pass    # o estado do currículo PERSISTE entre resets — é o método
