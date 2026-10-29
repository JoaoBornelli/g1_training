"""Balanceadores e níveis do currículo da v3 (B3 do plano, §6.3; S §11). Puro: sem mjlab, sem mujoco.

Dois balanceadores: o de fatias (ANDAR, B, C; S §11.1) e o de células dentro de cada tarefa (S §11.2),
sobre a tabela de abertura da §6.3 e as células de PD-12. Os knobs entram como argumento (PD-43).
O estado vai ao checkpoint em tipos do Python, no formato de g1_limpo/runner.py:47-61.
"""
from __future__ import annotations

import dataclasses

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

    def __init__(self, k_bal, k_niv, k_com, k_ter):
        self.k = k_bal
        # ⚠ ANDAR e CARREGAR têm 6 níveis de velocidade (`frente_tetos`); PEGAR e BOTAR, as tabelas de Niveis
        self._max = {"ANDAR": len(k_com.frente_tetos) - 1, "CARREGAR": len(k_com.frente_tetos) - 1,
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
        # ⚠ a carência conta desde it = 0 para a 1ª subida (PD-12); por isso `_ult` começa em 0 e vai no estado
        self._ult = dict.fromkeys(TAREFAS, 0)
        self._degrau = dict.fromkeys(TAREFAS, 0)
        self._cont = dict.fromkeys(TAREFAS, 0)

    # --- consultas internas ---------------------------------------------------------------------

    def _aberta(self, t, tp):
        return self.celulas[_POS[(t, tp)]].aberta

    def _abre_cel(self, t, tp):
        self.celulas[_POS[(t, tp)]].aberta = True

    def _passa(self, i):
        return self.celulas[i].s >= self.k.portao

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
        # 3) aberturas (B e C, cadeias, terreno); 4) subidas de nível.
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
        self._sobe()

    def _abre(self) -> None:
        # S §11.1 e §11.2 (tabela da §6.3). Um passo por fecho e por cadeia: a célula nova entra com s = 0,
        # então não passa no mesmo fecho e não abre a seguinte.
        andar_reto, andar_parado = _POS[("ANDAR", "reto")], _POS[("ANDAR", "parado")]
        if not self._aberta("CARREGAR", "reto") and self._passa(andar_reto) and self._passa(andar_parado):
            self._abre_cel("CARREGAR", "reto")  # fatia B: o CARREGAR reto e o PEGAR nível 0 abrem juntos
            self._abre_cel("PEGAR", "pegar")
        for t in CADEIA:
            abertas = self._abertos(t)
            fechadas = [tp for tp in VELOCIDADE[t] if not self._aberta(t, tp)]
            # `abertas` não vazio: o CARREGAR só começa depois da fatia B
            if abertas and fechadas and all(self._passa(i) for i in abertas):
                self._abre_cel(t, fechadas[0])
        for t in CADEIA:
            # terreno: depois do último nível de velocidade, com todas as abertas no portão.
            # ⚠ no CARREGAR, também com o terreno do ANDAR já passado no portão no nível de terreno 0
            # (S §11.2): ou o ANDAR já subiu de linha, ou a célula dele passa agora.
            if self._nivel[t] != self._max[t] or self._aberta(t, "terreno"):
                continue
            if t == "CARREGAR" and not (self._nivel_ter["ANDAR"] >= 1 or self._passa(_POS[("ANDAR", "terreno")])):
                continue
            if all(self._passa(i) for i in self._abertos(t)):
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
        ultimo = self._nivel[t] == self._max[t] and (
            t not in CADEIA or (self._aberta(t, "terreno") and self._nivel_ter[t] == self._max_ter[t]))
        abertas = self._abertos(t)
        media = sum(self.celulas[i].s for i in abertas) / len(abertas) if abertas else 0.0
        if ultimo and abertas and media >= self.k.portao:
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
            if self._nivel[t] < self._max[t]:
                ids = [_POS[(t, tp)] for tp in VELOCIDADE[t]]
                if all(self.celulas[i].aberta for i in ids) and all(self._passa(i) for i in ids):
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

    # --- estado do checkpoint -------------------------------------------------------------------

    def estado(self) -> dict:
        return {
            "it": self.it,
            "celulas": [{"aberta": c.aberta, "s": c.s, "n": c.n_medidas} for c in self.celulas],
            "acum": [[n, s] for n, s in zip(self._acum_n, self._acum_suc)],
            "tarefas": {t: {"nivel": self._nivel[t], "nivel_ter": self._nivel_ter[t], "ult": self._ult[t],
                            "degrau": self._degrau[t], "cont": self._cont[t]} for t in TAREFAS},
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
