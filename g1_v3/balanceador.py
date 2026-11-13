"""Balanceadores e níveis do currículo da v3 (B3 do plano, §6.3; S §11). Puro: sem mjlab, sem mujoco.

Dois balanceadores: o de fatias (ANDAR, B, C; S §11.1) e o de células dentro de cada tarefa (S §11.2),
sobre a tabela de abertura da §6.3 e as células de PD-12. Os knobs entram como argumento (PD-43).
O estado vai ao checkpoint em tipos do Python, no formato de g1_limpo/runner.py:47-61.
"""
from __future__ import annotations

import dataclasses
import math

import torch

TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
# ordem de abertura (§6.3): o celula_id é o índice nesta lista
LAYOUT = (("ANDAR", "reto"), ("ANDAR", "parado"), ("ANDAR", "baixo"), ("ANDAR", "giro_andando"),
          ("ANDAR", "giro_lugar"), ("ANDAR", "terreno"),
          ("CARREGAR", "reto"), ("CARREGAR", "giro_lugar"), ("CARREGAR", "arco"), ("CARREGAR", "terreno"),
          ("PEGAR", "pegar"), ("BOTAR", "botar"))
_POS = {lay: i for i, lay in enumerate(LAYOUT)}
ABERTAS_NO_COMECO = (("ANDAR", "reto"), ("ANDAR", "parado"))  # S §11.1: só o ANDAR abre no começo
# células de velocidade: o nível sobe quando todas estão abertas e no portão
VELOCIDADE = {"ANDAR": ("reto", "parado", "baixo", "giro_andando", "giro_lugar"),
              "CARREGAR": ("reto", "giro_lugar", "arco"), "PEGAR": ("pegar",), "BOTAR": ("botar",)}
CADEIA = ("ANDAR", "CARREGAR")  # tarefas com cadeia de velocidade e célula de terreno (S §11.2)
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
    tipo: str          # ANDAR: "reto","parado","baixo","giro_andando","giro_lugar","terreno";
                       # CARREGAR: "reto","giro_lugar","arco","terreno"; PEGAR: "pegar"; BOTAR: "botar"
    aberta: bool
    s: float           # EMA do sucesso (PD-13)
    n_medidas: int


class Curriculo:
    """Estado inteiro do currículo: células, níveis, fatias, refino. Puro e serializável."""

    def __init__(self, k_bal, k_niv, k_com, k_ter, k_dr=None):
        self.k = k_bal
        self.k_dr = k_dr  # DR (§6.8); sem ele, `dr()` devolve {}
        # ⚠ ANDAR e CARREGAR têm 5 níveis de velocidade (`frente_tetos`); PEGAR e BOTAR, as tabelas de Niveis
        self.max_nivel = {"ANDAR": len(k_com.frente_tetos) - 1, "CARREGAR": len(k_com.frente_tetos) - 1,
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
        # ⚠ a carência conta desde it = 0 para a 1ª subida (PD-12): o ANDAR conta desde a iteração 0; as tarefas que
        # abrem depois contam da abertura (`_abre`; 13/11, A8; antes, a primeira medida da célula, com 1 ou 2
        # episódios, subia o nível sem carência). `_ult` vai no estado
        self._ult = dict.fromkeys(TAREFAS, 0)
        self._degrau = dict.fromkeys(TAREFAS, 0)
        self._cont = dict.fromkeys(TAREFAS, 0)
        # travas do DR (§6.8): só ligam, nunca desligam. "portao0": o PEGAR passou no portão do nível 0;
        # "vies": o ANDAR passou no portão (reto e parado); "subida": o PEGAR ou o CARREGAR já subiu de nível
        self._dr = {"portao0": False, "vies": False, "subida": False}
        # rampa da fatia (§5.3): o fecho em que cada fatia abriu (None = fechada); o ANDAR abre na iteração 0
        self._it_abertura = [0, None, None]

    # --- consultas internas ---------------------------------------------------------------------

    def _aberta(self, t, tp):
        return self.celulas[_POS[(t, tp)]].aberta

    def _abre_cel(self, t, tp):
        self.celulas[_POS[(t, tp)]].aberta = True

    def _passa(self, i):
        return self.celulas[i].s >= self.k.portao

    def _passa_media(self, ids):
        # 11/11: a média das células no portão. O portão conjuntivo (todas ≥ portão ao mesmo tempo) nunca abriu
        # em 13 TensorBoards; o refino (`_conta_refino`) já usava a média.
        return bool(ids) and sum(self.celulas[i].s for i in ids) / len(ids) >= self.k.portao

    def _de(self, t):
        return [i for i, c in enumerate(self.celulas) if c.tarefa == t]

    def _abertos(self, t):
        return [i for i in self._de(t) if self.celulas[i].aberta]

    def _deficit(self, ids):
        return deficit(torch.tensor([self.celulas[i].s for i in ids], dtype=torch.float64), self.k.portao)

    # --- registro e fecho da iteração -----------------------------------------------------------

    def registra(self, celula_id: int, n: int, n_sucesso: int) -> None:
        self._acum_n[celula_id] += n
        self._acum_suc[celula_id] += n_sucesso

    def fecha_iteracao(self) -> None:
        # ⚠ ordem do fecho (os testes fixam): 1) EMA com o que a iteração registrou; 2) contador do refino,
        # ANTES das subidas, para que a célula recém-subida (s velho, n_medidas = 0) não conte no nível novo;
        # 3) aberturas (B e C, cadeias, terreno); 4) subidas de nível; 5) travas do DR (com o s já atualizado).
        # EMA (PD-13), uma atualização por iteração: com n > 0, x = n_sucesso/n; se n_medidas = 0, s = x;
        # senão s += alfa·(x − s). n = 0 não muda s. s começa em 0.
        # Subir de nível zera n_medidas das células da tarefa: a EMA recomeça na 1ª medida depois da subida.
        self.it += 1
        for i, c in enumerate(self.celulas):
            if self._acum_n[i] > 0:
                x = self._acum_suc[i] / self._acum_n[i]
                c.s = x if c.n_medidas == 0 else c.s + self.k.alfa * (x - c.s)
                c.n_medidas += 1
            self._acum_n[i] = self._acum_suc[i] = 0
        for t in TAREFAS:
            self._conta_refino(t)
        self._abre()
        for f in (1, 2):
            if self._it_abertura[f] is None and any(c.aberta and FATIA_DE[c.tarefa] == f for c in self.celulas):
                self._it_abertura[f] = self.it
        self._sobe()
        self._trava_dr()

    def _trava_dr(self) -> None:
        # §6.8: as travas do DR só ligam. Desde 11/11 a fatia B abre pela média e o "vies" exige o reto e o parado no portão: o "vies" pode ligar depois da fatia B
        self._dr["portao0"] = self._dr["portao0"] or self.nivel_passou("PEGAR") >= 0
        self._dr["vies"] = self._dr["vies"] or (self._passa(_POS[("ANDAR", "reto")]) and
                                                self._passa(_POS[("ANDAR", "parado")]))
        self._dr["subida"] = self._dr["subida"] or self._nivel["PEGAR"] + self._nivel["CARREGAR"] > 0

    def _abre(self) -> None:
        # S §11.1 e §11.2 (tabela da §6.3). Um passo por fecho e por cadeia: a célula nova entra com s = 0,
        # então não passa no mesmo fecho e não abre a seguinte.
        andar_reto, andar_parado = _POS[("ANDAR", "reto")], _POS[("ANDAR", "parado")]
        if not self._aberta("CARREGAR", "reto") and self._passa_media([andar_reto, andar_parado]):
            self._abre_cel("CARREGAR", "reto")  # fatia B: o CARREGAR reto e o PEGAR nível 0 abrem juntos
            self._abre_cel("PEGAR", "pegar")
            self._ult["PEGAR"] = self._ult["CARREGAR"] = self.it  # 13/11, A8: a carência conta da abertura
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
            self._ult["BOTAR"] = self.it  # 13/11, A8

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
        # PD-12 e S §11.2: nível k+1 quando todas as células de velocidade estão abertas e no portão, e já
        # passaram iters_min_nivel its desde a última subida da tarefa.
        for t in TAREFAS:
            if self.it - self._ult[t] < self.k.iters_min_nivel:
                continue
            if self._nivel[t] < self.max_nivel[t]:
                ids = [_POS[(t, tp)] for tp in VELOCIDADE[t]]
                if all(self.celulas[i].aberta for i in ids) and self._passa_media(ids):
                    self._nivel[t] += 1
                    self._ult[t] = self.it
                    for i in self._de(t):
                        self.celulas[i].n_medidas = 0
            elif t in CADEIA and self._nivel_ter[t] < self._max_ter[t]:
                # linha de terreno: a célula `terreno` aberta e no portão; a mesma carência das subidas
                ter = _POS[(t, "terreno")]
                if self.celulas[ter].aberta and self._passa(ter):
                    self._nivel_ter[t] += 1
                    self._ult[t] = self.it
                    self.celulas[ter].n_medidas = 0

    # --- consultas do env -----------------------------------------------------------------------

    def fracoes_fatias(self) -> torch.Tensor:
        # S §11.1: média dos déficits das células abertas de cada fatia; fatia fechada = 0
        out = torch.zeros(3, dtype=torch.float64)
        grupos = {f: [i for i, c in enumerate(self.celulas) if c.aberta and FATIA_DE[c.tarefa] == f]
                  for f in range(3)}
        abertas = [f for f in range(3) if grupos[f]]
        if not abertas:
            return out
        d = torch.stack([self._deficit(grupos[f]).mean() for f in abertas])
        out[torch.tensor(abertas)] = fracoes(d, self.k.piso_tarefa, self.k.eps)
        for f in (1, 2):
            if self._it_abertura[f] is not None:
                teto = min(1.0, self.k.rampa_fatia_inicio
                           + self.k.rampa_fatia_passo * ((self.it - self._it_abertura[f]) // self.k.rampa_fatia_its))
                excesso = max(0.0, float(out[f]) - teto)
                out[f] -= excesso
                out[0] += excesso  # o excedente volta para o ANDAR
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
        # o maior nível da tarefa cuja célula já passou no portão (−1 = nenhum). Só PEGAR e BOTAR: uma célula por tarefa
        i = self._de(tarefa)[0]
        return self._nivel[tarefa] if self._passa(i) else self._nivel[tarefa] - 1

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
            "tarefas": {t: {"nivel": self._nivel[t], "nivel_ter": self._nivel_ter[t], "ult": self._ult[t],
                            "degrau": self._degrau[t], "cont": self._cont[t]} for t in TAREFAS},
            "dr": dict(self._dr),
            "it_abertura": list(self._it_abertura),
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
            self._ult[t], self._degrau[t], self._cont[t] = d["ult"], d["degrau"], d["cont"]
        self._dr = dict(estado["dr"])
        self._it_abertura = list(estado.get("it_abertura", [0, None, None]))
