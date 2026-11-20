"""Balanceadores e níveis do currículo da v3 (B3 do plano, §6.3; S §11). Puro: sem mjlab, sem mujoco.

Dois balanceadores: o de fatias (ANDAR, B, C; S §11.1) e o de células dentro de cada tarefa (S §11.2),
sobre a tabela de abertura da §6.3 e as células de PD-12. Os knobs entram como argumento (PD-43).
16/11: o ANDAR tem uma célula de velocidade (`andar`) e a nota dela é a média do núcleo da Unitree no segmento
(0 a 1); o portão dela é `portao_andar`, o das outras tarefas é `portao` (sucesso 0 ou 1 por segmento). O nível sobe
com o portão por `iters_nivel` iterações seguidas, nas 4 tarefas.
20/11, T13: cada etapa do PEGAR e do BOTAR (P0..P4, B1..B4) tem uma escala de zona z em [0, 1] (1 = critério largo, 0 = o final);
ela sobe e desce pela taxa de fecho por visita. Visitas e fechos acumulam no `vis_acum` e no `fech_acum`, que a cadeia preenche.
O estado vai ao checkpoint em tipos do Python, no formato de g1_limpo/runner.py:47-61.
"""
from __future__ import annotations

import dataclasses
import math

import torch

TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
# ordem de abertura (§6.3): o celula_id é o índice nesta lista
LAYOUT = (("ANDAR", "andar"), ("ANDAR", "terreno"),
          ("CARREGAR", "reto"), ("CARREGAR", "giro_lugar"), ("CARREGAR", "arco"), ("CARREGAR", "terreno"),
          ("PEGAR", "pegar"), ("BOTAR", "botar"))
_POS = {lay: i for i, lay in enumerate(LAYOUT)}
ABERTAS_NO_COMECO = (("ANDAR", "andar"),)  # S §11.1: só o ANDAR abre no começo
# células de velocidade: o nível sobe quando todas estão abertas e no portão
VELOCIDADE = {"ANDAR": ("andar",),
              "CARREGAR": ("reto", "giro_lugar", "arco"), "PEGAR": ("pegar",), "BOTAR": ("botar",)}
CADEIA = ("ANDAR", "CARREGAR")  # tarefas com cadeia de velocidade e célula de terreno (S §11.2)
# 20/11, T13: etapas do PEGAR e do BOTAR, P0..P4 e B1..B4 (o `len(knobs.ALVOS)`; um módulo puro não importa o knobs)
N_ETAPAS = 9
FATIA_DE = {"ANDAR": 0, "PEGAR": 1, "CARREGAR": 1, "BOTAR": 2}  # ANDAR, B, C (PD-14)


def fracoes(deficits, piso, eps):
    # S §11.1: F = piso + (1 − n·piso)(D + ε)/Σ(D + ε). Tudo no portão dá 1/n.
    n = deficits.shape[0]
    w = deficits + eps
    return piso + (1 - n * piso) * w / w.sum()


def deficit(s, portao):
    return torch.clamp(portao - s, min=0.0)


def no_degrau(ini, fim, d, degraus):
    # §6.9 (P1): o valor do degrau d de um refino que vai de `ini` (d = 0) a `fim` (d = degraus). Uma função só:
    # o dr() a chama aqui, e o recompensas.por_degrau e os módulos do env a importam daqui. `d` é int ou tensor (E,).
    return ini + (fim - ini) * d / degraus


@dataclasses.dataclass
class Celula:
    tarefa: str        # "ANDAR" | "PEGAR" | "CARREGAR" | "BOTAR"
    tipo: str          # ANDAR: "andar","terreno";
                       # CARREGAR: "reto","giro_lugar","arco","terreno"; PEGAR: "pegar"; BOTAR: "botar"
    aberta: bool
    s: float           # EMA da nota da célula (PD-13): sucesso 0 ou 1 por segmento; no ANDAR, a nota do núcleo da Unitree
    n_medidas: int


class Curriculo:
    """Estado inteiro do currículo: células, níveis, fatias, refino. Puro e serializável."""

    def __init__(self, k_bal, k_niv, k_com, k_ter, k_dr=None):
        self.k = k_bal
        self.k_dr = k_dr  # DR (§6.8); sem ele, `dr()` devolve {}
        # ⚠ PEGAR e BOTAR têm os níveis das tabelas de Niveis
        # 16/11: o ANDAR sobe de `andar_inicio` aos tetos finais em passos de `andar_passo` (10 níveis); o CARREGAR segue a tabela
        self.max_nivel = {"ANDAR": round((k_com.frente_tetos[-1] - k_com.andar_inicio) / k_com.andar_passo),
                          "CARREGAR": len(k_com.frente_tetos) - 1,
                          "PEGAR": len(k_niv.pegar) - 1, "BOTAR": len(k_niv.botar) - 1}
        # linha de terreno (PD-20): ANDAR até linhas − 1; CARREGAR até a maior linha_max_carregar (as colunas
        # com cap menor param no env, pelo min da coluna)
        self._max_ter = {"ANDAR": k_ter.linhas - 1,
                         "CARREGAR": max(v for _n, v in k_ter.linha_max_carregar)}
        self.celulas = [Celula(t, tp, (t, tp) in ABERTAS_NO_COMECO, 0.0, 0) for t, tp in LAYOUT]
        self.it = 0
        self._acum_n = [0] * len(LAYOUT)  # contagens da iteração corrente (zeradas no fecho)
        self._acum_suc = [0] * len(LAYOUT)
        self._nivel = dict.fromkeys(TAREFAS, 0)
        self._nivel_ter = dict.fromkeys(TAREFAS, 0)  # só ANDAR e CARREGAR sobem; PEGAR e BOTAR ficam em 0
        # 13/11, M3: subnível do nível 0 do PEGAR e do BOTAR (laje alta primeiro). O índice do nível não muda com ele
        self._sub = {"PEGAR": 0, "BOTAR": 0}
        self._sub_max = len(k_niv.subnivel_extra) - 1
        # 16/11, A4 e P8: iterações seguidas no portão de cada tarefa (a carência por tempo saiu). Uma tarefa que abre
        # tarde começa em 0. Vai no estado
        self._seguidas = dict.fromkeys(TAREFAS, 0)
        self._portao = {t: k_bal.portao_andar if t == "ANDAR" else k_bal.portao for t in TAREFAS}
        self._degrau = dict.fromkeys(TAREFAS, 0)
        self._cont = dict.fromkeys(TAREFAS, 0)
        # travas do DR (§6.8): só ligam, nunca desligam. "portao0": o PEGAR passou no portão do nível 0;
        # "vies": a célula `andar` do ANDAR passou no portão; "subida": o PEGAR ou o CARREGAR já subiu de nível
        self._dr = {"portao0": False, "vies": False, "subida": False}
        # 20/11, T13: zona de cada etapa (índices de `ALVOS`); `taxa` é a última calculada, para o log
        self.z = [1.0] * N_ETAPAS
        self.vis_acum = [0] * N_ETAPAS
        self.fech_acum = [0] * N_ETAPAS
        self.taxa = [0.0] * N_ETAPAS
        # rampa da fatia (§5.3): o fecho em que cada fatia abriu (None = fechada); o ANDAR abre na iteração 0
        self._it_abertura = [0, None, None]
        # degraus da rampa de cada fatia. 18/11, T9: sobe um a cada `rampa_fatia_its` fechos e não desce
        self._rampa = [0, 0, 0]

    # --- consultas internas ---------------------------------------------------------------------

    def _aberta(self, t, tp):
        return self.celulas[_POS[(t, tp)]].aberta

    def _abre_cel(self, t, tp):
        self.celulas[_POS[(t, tp)]].aberta = True

    def _passa(self, i):
        return self.celulas[i].s >= self._portao[self.celulas[i].tarefa]

    def _passa_media(self, ids):
        # 11/11: a média das células no portão. O portão conjuntivo (todas ≥ portão ao mesmo tempo) nunca abriu
        # em 13 TensorBoards; o refino (`_conta_refino`) já usava a média. Os `ids` são de uma tarefa só.
        return bool(ids) and sum(self.celulas[i].s for i in ids) / len(ids) >= self._portao[self.celulas[ids[0]].tarefa]

    def _de(self, t):
        return [i for i, c in enumerate(self.celulas) if c.tarefa == t]

    def _abertos(self, t):
        return [i for i in self._de(t) if self.celulas[i].aberta]

    def _deficit(self, ids):
        return deficit(torch.tensor([self.celulas[i].s for i in ids], dtype=torch.float64),
                       torch.tensor([self._portao[self.celulas[i].tarefa] for i in ids], dtype=torch.float64))

    # --- registro e fecho da iteração -----------------------------------------------------------

    def registra(self, celula_id: int, n: int, soma: float) -> None:
        # `soma`: a soma das notas dos `n` segmentos (0 ou 1 por segmento nas tarefas com sucesso booleano)
        self._acum_n[celula_id] += n
        self._acum_suc[celula_id] += soma

    def fecha_iteracao(self) -> None:
        # ⚠ ordem do fecho (os testes fixam): 1) EMA com o que a iteração registrou; 2) contador do refino,
        # ANTES das subidas, para que a célula recém-subida (s velho, n_medidas = 0) não conte no nível novo;
        # 3) aberturas (B e C, cadeias, terreno); 4) rampa das fatias (por tempo); 5) subidas de nível
        # (a contagem de iterações seguidas no portão); 6) travas do DR (com o s já atualizado). 20/11, T13: a zona das
        # etapas entra entre a EMA e o refino.
        # EMA (PD-13), uma atualização por iteração: com n > 0, x = soma/n; se n_medidas = 0, s = x;
        # senão s += alfa·(x − s). n = 0 não muda s. s começa em 0.
        # Subir de nível zera n_medidas das células da tarefa: a EMA recomeça na 1ª medida depois da subida.
        self.it += 1
        for i, c in enumerate(self.celulas):
            if self._acum_n[i] > 0:
                x = self._acum_suc[i] / self._acum_n[i]
                c.s = x if c.n_medidas == 0 else c.s + self.k.alfa * (x - c.s)
                c.n_medidas += 1
            self._acum_n[i] = self._acum_suc[i] = 0
        # 20/11, T13: a zona das etapas com visitas suficientes; depois, uma etapa não fica mais apertada que a anterior
        for e, n in enumerate(self.vis_acum):
            if n >= self.k.zona_n_min:
                self.taxa[e] = self.fech_acum[e] / n
                self.z[e] = min(1.0, max(0.0, self.z[e] - self.k.zona_ganho * (self.taxa[e] - self.k.zona_t_alvo)))
                self.vis_acum[e] = self.fech_acum[e] = 0
        for e in (*range(1, 5), *range(6, 9)):
            self.z[e] = max(self.z[e], self.z[e - 1] - self.k.zona_folga_cadeia)
        for t in TAREFAS:
            self._conta_refino(t)
        self._abre()
        for f in (1, 2):
            if self._it_abertura[f] is None and any(c.aberta and FATIA_DE[c.tarefa] == f for c in self.celulas):
                self._it_abertura[f] = self.it
        for f in (1, 2):
            # 18/11, T9: a rampa sobe só por tempo e não desce; a nota do ANDAR saiu dela
            d = 0 if self._it_abertura[f] is None else self.it - self._it_abertura[f]
            if d > 0 and d % self.k.rampa_fatia_its == 0:
                self._rampa[f] += 1
        self._sobe()
        self._trava_dr()

    def _trava_dr(self) -> None:
        # §6.8: as travas do DR só ligam. 16/11: a fatia B abre pelo nível do ANDAR e o "vies" liga com a célula `andar`
        # no portão: ele pode ligar antes ou depois da fatia B
        self._dr["portao0"] = self._dr["portao0"] or self.nivel_passou("PEGAR") >= 0
        self._dr["vies"] = self._dr["vies"] or self._passa(_POS[("ANDAR", "andar")])
        self._dr["subida"] = self._dr["subida"] or self._nivel["PEGAR"] + self._nivel["CARREGAR"] > 0

    def _abre(self) -> None:
        # S §11.1 e §11.2 (tabela da §6.3). Um passo por fecho e por cadeia: a célula nova entra com s = 0,
        # então não passa no mesmo fecho e não abre a seguinte.
        # 16/11, B1: a fatia B abre quando o ANDAR chega ao nível `nivel_abre_b` (a subida sai no `_sobe` do fecho
        # anterior, então a B abre no fecho seguinte)
        if not self._aberta("CARREGAR", "reto") and self._nivel["ANDAR"] >= self.k.nivel_abre_b:
            self._abre_cel("CARREGAR", "reto")  # fatia B: o CARREGAR reto e o PEGAR nível 0 abrem juntos
            self._abre_cel("PEGAR", "pegar")
        for t in CADEIA:
            abertas = self._abertos(t)
            fechadas = [tp for tp in VELOCIDADE[t] if not self._aberta(t, tp)]
            # `abertas` não vazio: o CARREGAR só começa depois da fatia B
            if abertas and fechadas and self._passa_media(abertas):
                self._abre_cel(t, fechadas[0])
        for t in CADEIA:
            # terreno: depois do último nível de velocidade, com todas as abertas no portão.
            # ⚠ no CARREGAR, também com o terreno do ANDAR já passado no portão no nível de terreno 0
            # (S §11.2): ou o ANDAR já subiu de linha, ou a célula dele passa agora.
            if self._nivel[t] != self.max_nivel[t] or self._aberta(t, "terreno"):
                continue
            if t == "CARREGAR" and not (self._nivel_ter["ANDAR"] >= 1 or self._passa(_POS[("ANDAR", "terreno")])):
                continue
            if self._passa_media(self._abertos(t)):
                self._abre_cel(t, "terreno")
        # fatia C: o PEGAR nível 0 no portão. O PEGAR não sobe antes disso: subir exige o portão e este passo
        # roda antes de `_sobe` no mesmo fecho.
        if not self._aberta("BOTAR", "botar") and self._passa(_POS[("PEGAR", "pegar")]):
            self._abre_cel("BOTAR", "botar")

    def _conta_refino(self, t: str) -> None:
        # S §11.5 e §6.9: o último nível aberto e a média das células abertas da tarefa ≥ portão por
        # iters_refino its seguidas; o degrau sobe e a conta recomeça; o degrau nunca desce.
        if self._degrau[t] >= self.k.degraus:
            return
        ultimo = self._nivel[t] == self.max_nivel[t] and (
            t not in CADEIA or (self._aberta(t, "terreno") and self._nivel_ter[t] == self._max_ter[t]))
        abertas = self._abertos(t)
        if ultimo and self._passa_media(abertas):
            self._cont[t] += 1
        else:
            self._cont[t] = 0
        if self._cont[t] >= self.k.iters_refino:
            self._degrau[t] += 1
            self._cont[t] = 0

    def _sobe(self) -> None:
        # 16/11, A4 e P8: a tarefa sobe depois de `iters_nivel` fechos seguidos no portão; um fecho abaixo zera a contagem.
        # No nível abaixo do máximo, `ok` é: todas as células de velocidade abertas e a média no portão (PD-12, S §11.2).
        # No máximo, numa tarefa de cadeia com a linha de terreno abaixo do teto: a célula `terreno` aberta e no portão.
        # O que sobe: o subnível (13/11, M3: só PEGAR e BOTAR no nível 0), o nível ou a linha de terreno.
        for t in TAREFAS:
            ids = [_POS[(t, tp)] for tp in VELOCIDADE[t]]
            ter = _POS.get((t, "terreno"))
            if self._nivel[t] < self.max_nivel[t]:
                ok = all(self.celulas[i].aberta for i in ids) and self._passa_media(ids)
            elif t in CADEIA and self._nivel_ter[t] < self._max_ter[t]:
                ok = self.celulas[ter].aberta and self._passa(ter)
            else:
                ok = False
            self._seguidas[t] = self._seguidas[t] + 1 if ok else 0
            if self._seguidas[t] < self.k.iters_nivel:
                continue
            self._seguidas[t] = 0
            if self._nivel[t] == self.max_nivel[t]:
                self._nivel_ter[t] += 1
                self.celulas[ter].n_medidas = 0
                continue
            if self.subnivel(t) < self._sub_max:
                self._sub[t] += 1
            else:
                self._nivel[t] += 1
            for i in self._de(t):
                self.celulas[i].n_medidas = 0

    # --- consultas do env -----------------------------------------------------------------------

    def fracoes_fatias(self) -> torch.Tensor:
        # 18/11, T9: cada fatia aberta recebe a mesma parte; o déficit saiu daqui (ele prendia a B e a C à nota do ANDAR).
        # A fatia nova entra pela rampa: o teto dela começa em `rampa_fatia_inicio` e sobe um passo por degrau. O que o
        # teto corta volta em partes iguais às fatias de teto maior (o ANDAR não tem teto). Fatia fechada = 0
        out = torch.zeros(3, dtype=torch.float64)
        abertas = [f for f in range(3) if any(c.aberta and FATIA_DE[c.tarefa] == f for c in self.celulas)]
        teto = {f: 1.0 if f == 0 else self.k.rampa_fatia_inicio + self.k.rampa_fatia_passo * self._rampa[f] for f in abertas}
        resto = 1.0
        for j, f in enumerate(sorted(abertas, key=teto.get)):
            out[f] = min(teto[f], resto / (len(abertas) - j))
            resto -= float(out[f])
        return out

    def fracoes_celulas(self, tarefa: str) -> tuple[list[int], torch.Tensor]:
        ids = self._abertos(tarefa)
        if not ids:
            return [], torch.zeros(0, dtype=torch.float64)
        return ids, fracoes(self._deficit(ids), self.k.piso_sub, self.k.eps)

    # os sorteios rodam e devolvem no device do gerador (o `v3_gerador` mora na GPU); sem gerador, CPU
    def sorteia_fatias(self, n: int, gerador) -> torch.Tensor:
        dev = "cpu" if gerador is None else gerador.device
        return torch.multinomial(self.fracoes_fatias().to(dev), n, replacement=True, generator=gerador)

    def sorteia_celulas(self, tarefa: str, n: int, gerador, sem_tipo: str | None = None) -> torch.Tensor:
        dev = "cpu" if gerador is None else gerador.device
        ids, f = self.fracoes_celulas(tarefa)
        f = f.to(dev)
        if sem_tipo is not None:
            # ⚠ a fração da célula `sem_tipo` vale 0: o `twist` fora do episódio de terreno nunca cai nela (§6.4,
            # `v3_celula`). Sem `sem_tipo` a sequência de sorteio é a de sempre.
            f = torch.where(torch.tensor([self.celulas[i].tipo != sem_tipo for i in ids], dtype=torch.bool, device=dev),
                            f, torch.zeros_like(f))
        return torch.tensor(ids, dtype=torch.long, device=dev)[
            torch.multinomial(f, n, replacement=True, generator=gerador)]

    def id_celula(self, tarefa: str, tipo: str) -> int:
        # §6.3: `celulas` está na ordem de LAYOUT e nunca muda, então o índice vem da mesma tabela
        return _POS[(tarefa, tipo)]

    def nivel(self, tarefa: str) -> int:
        return self._nivel[tarefa]

    def subnivel(self, tarefa: str) -> int:
        # 13/11, M3: ANDAR e CARREGAR não têm subnível: valem o último
        return self._sub.get(tarefa, self._sub_max)

    def nivel_terreno(self, tarefa: str) -> int:
        return self._nivel_ter[tarefa]

    def niveis_abertos(self, tarefa: str) -> int:
        return self._nivel[tarefa] + 1  # o degrau do freio (S §11.4)

    def degrau(self, tarefa: str) -> int:
        return self._degrau[tarefa]

    def degrau_global(self) -> int:
        # §6.9, PD-41 (P5): o refino do DR só sobe quando as 4 tarefas chegaram: o menor degrau entre elas
        return min(self.degrau(t) for t in TAREFAS)

    def restricao(self) -> float:
        # 11/11 (specs/g1-v3-treino-unico.md §2): as restrições seguem o progresso, como o freio segue o nível.
        # O menor nível relativo entre as 4 tarefas: 0 até todas passarem do nível 0; nunca desce.
        return min(self._nivel[t] / self.max_nivel[t] for t in TAREFAS)

    def nivel_passou(self, tarefa: str) -> int:
        # o maior nível da tarefa cuja célula já passou no portão (−1 = nenhum). Só PEGAR e BOTAR: uma célula por tarefa.
        # 13/11, M3: com o subnível abaixo do último, o nível 0 ainda não passou. C1: no nível 0 o `s` da célula é o do
        # subnível anterior até a primeira medida do último (a subida zera só o `n_medidas`), então exige `n_medidas > 0`
        i = self._de(tarefa)[0]
        ok = self._passa(i) and self.subnivel(tarefa) == self._sub_max and (self._nivel[tarefa] > 0 or self.celulas[i].n_medidas > 0)
        return self._nivel[tarefa] if ok else self._nivel[tarefa] - 1

    def dr(self) -> dict:
        # §6.8: o estado de DR que o env aplica, só tipos do Python. Sobe no fecho e nunca desce (as travas).
        # ⚠ "com_caixa" só age com `DR.com_caixa` (PD-38): desligada, o vetor fica [0, 0].
        # H1 (30/10): o refino (pd, forca_motor, forca_motor_bracos, solref, rfi, atraso_max) segue o degrau GLOBAL g
        # (PD-41, P5) pela `no_degrau`; com g = 0 cada par vale o início. "forca_caixa" é a trava, sem escala (P15).
        if self.k_dr is None:
            return {}
        d, p0, sub = self.k_dr, self._dr["portao0"], self._dr["subida"]
        g, n = self.degrau_global(), self.k.degraus

        def pares(ini, fim):
            return [no_degrau(ini, fim[0], g, n), no_degrau(ini, fim[1], g, n)]

        mu = d.atrito_pad[0]
        if p0:
            mu = max(d.mu_pad_min, d.atrito_pad[0] - d.passo_mu_pad * (self._nivel["PEGAR"] + self._nivel["CARREGAR"]))
        com = d.com_caixa_frac_max if sub else d.com_caixa_frac0 if p0 else (0, 0)
        return {
            "vies_imu": self._dr["vies"],
            "atrito_pad": p0,
            "atrito_laje": p0,
            "mu_min_pad": float(mu),
            "com_caixa": [float(x) for x in com] if d.com_caixa else [0.0, 0.0],
            "atraso_max": (d.atraso_refino[1] if g >= d.degrau_atraso_refino
                           else math.ceil(d.atraso_portao[1] * self.restricao() - 1e-9)),
            "forca_caixa": sub,
            "rfi": no_degrau(d.rfi[0], d.rfi[1], g, n) if sub else 0.0,
            "pd": pares(1, d.pd),
            "forca_motor": pares(1, d.forca_motor),
            "forca_motor_bracos": pares(1, d.forca_motor_bracos),
            "solref": pares(d.solref_base, d.solref),
        }

    # --- estado do checkpoint -------------------------------------------------------------------

    def estado(self) -> dict:
        return {
            "it": self.it,
            "celulas": [{"aberta": c.aberta, "s": c.s, "n": c.n_medidas} for c in self.celulas],
            "acum": [[n, s] for n, s in zip(self._acum_n, self._acum_suc)],
            "tarefas": {t: {"nivel": self._nivel[t], "nivel_ter": self._nivel_ter[t], "seguidas": self._seguidas[t],
                            "degrau": self._degrau[t], "cont": self._cont[t], "sub": self.subnivel(t)} for t in TAREFAS},
            "dr": dict(self._dr),
            "it_abertura": list(self._it_abertura),
            "rampa": list(self._rampa),
            "zona": {"z": list(self.z), "visitas": list(self.vis_acum), "fechos": list(self.fech_acum)},
        }

    def carrega(self, estado: dict) -> None:
        self.it = estado["it"]
        for c, d in zip(self.celulas, estado["celulas"]):
            c.aberta, c.s, c.n_medidas = d["aberta"], d["s"], d["n"]
        self._acum_n = [a[0] for a in estado["acum"]]
        self._acum_suc = [a[1] for a in estado["acum"]]
        for t in TAREFAS:
            d = estado["tarefas"][t]
            self._nivel[t], self._nivel_ter[t] = d["nivel"], d["nivel_ter"]
            self._seguidas[t], self._degrau[t], self._cont[t] = d.get("seguidas", 0), d["degrau"], d["cont"]
            if t in self._sub:
                self._sub[t] = d.get("sub", self._sub_max)  # estado sem a chave: vale o último subnível
        self._dr = dict(estado["dr"])
        self._it_abertura = list(estado.get("it_abertura", [0, None, None]))
        self._rampa = list(estado.get("rampa", [0, 0, 0]))
        # 20/11, T13: estado sem a chave (checkpoint antigo): z = 1 e os acumuladores em 0
        zona = estado.get("zona", {})
        self.z = list(zona.get("z", [1.0] * N_ETAPAS))
        self.vis_acum = list(zona.get("visitas", [0] * N_ETAPAS))
        self.fech_acum = list(zona.get("fechos", [0] * N_ETAPAS))
