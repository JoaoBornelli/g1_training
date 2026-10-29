"""Dirige um checkpoint da v3 no MuJoCo clássico, com o teclado (D11 do plano; I §2.1, §2.4; S §4, §15).

    python pilota_v3.py --cena ~/g1_pilota_v3/ --checkpoint ~/Downloads/model_9499.pt

⚠ ZERO IMPORT DE `g1_v3`. Os módulos puros (`knobs`, `alvos`, `estimador`, `percepcao_lidar`) são
carregados POR CAMINHO (`importlib.util.spec_from_file_location`), como o robô real faz (PD-43). Só
`mujoco`, `torch` e `numpy`: o arquivo fica na RAIZ e roda num venv magro.

⚠ A conta da observação mora AQUI, em `monta_observacao`, e a `paridade` do `exporta_cena.py` a IMPORTA
por caminho. Uma implementação só: duas cópias divergem e a paridade passaria a testar a si mesma.
A assinatura é o contrato da docstring do `exporta_cena.py` ("Contrato com o pilota_v3"). Não mexa nela
sem mexer lá.

⚠ FASE D. A caixa entra como zeros (o `caixa_zero` do treino) e o estimador da caixa e o detector de fim
não entram aqui: são da fase E. O mapa de elevação entra, com o lidar no MuJoCo clássico.

A cena vem do `exporta_cena.py`, que LÊ o env do mjlab. Nada aqui é digitado à mão.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

__all__ = ["carrega_cena", "monta_observacao", "Percepcao", "Ator", "TAREFAS", "DIM_ATOR"]


def _carrega(nome: str, rel: str):
    """Carrega um módulo puro do `g1_v3` por caminho, relativo à raiz deste arquivo.

    ⚠ O módulo vai para `sys.modules` ANTES de executar: o `knobs.py` usa `@dataclass` com anotações
    em string (`from __future__ import annotations`), e o dataclass procura `sys.modules[cls.__module__]`.
    Sem o registro, a carga quebra com `AttributeError` no `dataclass`.
    """
    caminho = Path(__file__).resolve().parent / rel
    spec = importlib.util.spec_from_file_location(nome, caminho)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"não achei {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


_K = _carrega("g1v3_knobs", "g1_v3/knobs.py")
_AL = _carrega("g1v3_alvos", "g1_v3/alvos.py")
_ES = _carrega("g1v3_estimador", "g1_v3/estimador.py")
_PL = _carrega("g1v3_percepcao", "g1_v3/percepcao_lidar.py")

_CONTRATO = _K.Contrato()   # I §2.1: blocos do ator, escalas, histórico
_KN = _K.ATIVO             # os números do lidar, da odometria e das referências (knobs.Knobs)
TAREFAS = _K.TAREFAS        # one-hot de 6 canais; a ordem é o índice (I §2.2)
DIM_ATOR = _K.dim_ator()    # 1 447 canais por passo
N_CAIXA = 11                # I §2.1: canais da caixa do ator (zeros na fase D)
_DT = _CONTRATO.dt_fisica * _CONTRATO.decimacao   # 0,02 s: passo da política
_HIST = _CONTRATO.historico                       # 5 passos por bloco com histórico
# Sem acerto, o raio vale 100 m: fora do alcance útil de 6 m, como em `pontos_no_frame_do_mapa`.
_SEM_ACERTO = 100.0

# Blocos com histórico formam o quadro de 113 canais (ordem do knobs). O único bloco sem histórico é o
# `mapa`, que entra com o quadro atual e escala 1,0 (I §2.1).
_QUADRO = [(n, ch, es) for n, ch, es, h in _CONTRATO.blocos_ator if h > 0]
_N_QUADRO = sum(ch for _, ch, _ in _QUADRO)       # 113
_ESC_QUADRO = np.concatenate([np.full(ch, es) for _, ch, es in _QUADRO])
_FATIA: dict[str, tuple[int, int]] = {}
_ini = 0
for _nome, _ch, _ in _QUADRO:
    _FATIA[_nome] = (_ini, _ini + _ch)
    _ini += _ch
assert _N_QUADRO == 113, f"o quadro do histórico tem {_N_QUADRO} canais, esperado 113 (I §2.1)"

# Tarefas que o teclado manda. REORIENTAR e LEVANTAR existem no one-hot como canais reservados, sem env (§6.3):
# por isso só estas quatro têm tecla.
TAREFAS_TECLADO = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")

# Laço de rumo do driver (spec §15, cenário "rumo"; interface §4.1): wz = K × erro de rumo, em rad/s.
_K_RUMO = 2.0


# --------------------------------------------------------------------------- cena

def carrega_cena(pasta: str | Path) -> tuple[mujoco.MjModel, dict[str, np.ndarray], dict]:
    """Lê `cena.mjb` (modelo), `cena.npz` (campos) e `contrato.json` da pasta que o exportador gravou."""
    pasta = Path(pasta).expanduser()
    caminho_mjb = pasta / "cena.mjb"
    caminho_npz = pasta / "cena.npz"
    caminho_json = pasta / "contrato.json"
    if not (caminho_mjb.exists() and caminho_npz.exists() and caminho_json.exists()):
        raise FileNotFoundError(
            f"faltam cena.mjb, cena.npz ou contrato.json em {pasta}. "
            f"Rode `python -m g1_v3.exporta_cena --saida {pasta}` primeiro.")
    m = mujoco.MjModel.from_binary_path(str(caminho_mjb))
    with np.load(caminho_npz, allow_pickle=False) as bruto:
        c = {k: bruto[k] for k in bruto.files}
    contrato = json.loads(caminho_json.read_text(encoding="utf-8"))
    return m, c, contrato


# ------------------------------------------------------------------- quatérnions

def gira_inverso(quat: np.ndarray, vetor: np.ndarray) -> np.ndarray:
    """`R⁻¹ · v`, com o quatérnion em (w, x, y, z), a convenção do MuJoCo.

    ⚠ A MESMA fórmula do `mjlab.utils.lab_api.math.quat_apply_inverse`: o sinal do termo do meio é o que
    separa a rotação da inversa.
    """
    xyz = quat[1:]
    t = 2.0 * np.cross(xyz, vetor)
    return vetor - quat[0] * t + np.cross(xyz, t)


def _yaw(quat: np.ndarray) -> float:
    """Guinada (rad) do quatérnion wxyz. Mesma conta do `_yaw` do `estimador.py` e do `_rumo` do treino."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


# ------------------------------------------------------------------ o mapa (lidar)

class Percepcao:
    """O mapa de elevação do lidar, no MuJoCo clássico (I §2.4; PERC §0; decisões de 29/10 da D11).

    Cada passo de política: a odometria recebe a pose VERDADEIRA da base; o `z_ref` sai do pé mais baixo;
    o `mj_multiRay` lança os 205 raios do passo a partir da pose VIVA do site `mid360`; o filtro do corpo
    tira as cápsulas e o raio mínimo; depois `prediz` transporta o mapa e `atualiza` põe a leitura nova.
    É a ORDEM de `observacoes.passo_de_percepcao` do treino. Sem ruído: o driver é determinístico.

    ⚠ `reinicia(m, d)` no reset da cena; `passo(m, d)` uma vez por passo de política, com `d` fresco
    (`mj_forward` depois do último `mj_step`).

    A assinatura `Percepcao(m, c)` é a que o `exporta_cena.py` já usa (`pilota.Percepcao(m, dados)`, §20, D11);
    a `paridade` do mapa tem de usar esta mesma forma.
    """

    def __init__(self, m: mujoco.MjModel, c: dict[str, np.ndarray]):
        self.k = _KN.percepcao
        self.base = int(c["id_base"])
        self.sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360")
        if self.sensor < 0:
            raise AssertionError("site `mid360` não existe no modelo")
        # ⚠ `exclude_parent_body=True` do treino (cena.py): o lidar não vê o corpo onde o site está (o torso).
        self.excluido = int(m.site_bodyid[self.sensor])
        self.pes = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
        if min(self.pes) < 0:
            raise AssertionError("sites `left_foot`/`right_foot` não existem no modelo")
        # ⚠ PD-9: o raio vê os grupos 0 (terreno), 1 (mobília) e 3 (pads). O mjtByte tem 6 posições.
        self.grupos = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8)

        caps = np.load(_K.caminho("g1_v3/dados/capsulas.npz"), allow_pickle=False)
        # ⚠ O `corpo` repete nomes (33 cápsulas, menos corpos): a pose sai por índice, como em `_capsulas_w`.
        self.corpo_idx = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(n))
                                   for n in caps["corpo"]], dtype=np.int64)
        if (self.corpo_idx < 0).any():
            raise AssertionError("alguma cápsula cita um corpo que não existe no modelo")
        self.p0 = np.asarray(caps["p0"], dtype=np.float64)
        self.p1 = np.asarray(caps["p1"], dtype=np.float64)
        self.raios = torch.as_tensor(caps["raio"], dtype=torch.float32)

        # Pé: a altura do site sobre a sola (`site_pe_de_pe`, D2), lida do mesmo `referencias.npz` do treino.
        self.site_pe_de_pe = _AL.Referencias(str(_K.caminho(_CONTRATO.referencias)), "cpu").site_pe_de_pe

        self.padrao = _PL.PadraoMid360(str(_K.caminho("g1_v3/dados/mid360_passos.npz")), "cpu")
        self.filtro = _PL.FiltroDeElevacao(1, "cpu", self.k)
        # Sem ruído, o `Odometria` não lê o `deriva` (só com gerador): zeros, só para cumprir a assinatura.
        self.odo = _ES.Odometria(1, "cpu", _KN.estimador, None)
        self.deriva = {"odo": 0.0, "rumo": 0.0, "pos_giro": 0.0, "rumo_giro": 0.0}
        # ⚠ IMU sem giro fixo: o treino com ruído sorteia `incl_imu`; aqui a identidade (PD-42).
        self.incl = torch.eye(3, dtype=torch.float32)[None]
        self.n = 0  # passo do padrão do MID-360: `direcoes(n)` (o trecho repete a cada 200)

    def _pose(self, d: mujoco.MjData) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.as_tensor(d.xpos[self.base], dtype=torch.float32)[None]
        quat = torch.as_tensor(d.xquat[self.base], dtype=torch.float32)[None]
        return pos, quat

    def reinicia(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """Reset: o mapa zera, a odometria começa na pose verdadeira e o padrão volta ao trecho 0 (PD-7, PD-18)."""
        ids = torch.tensor([0])
        pos, quat = self._pose(d)
        self.odo.reinicia(ids, pos, quat)
        self.filtro.reinicia(ids)
        self.n = 0

    def passo(self, m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        """Uma leitura → o mapa de 882 canais: 441 alturas (±1 m) e depois 441 confianças (I §2.4)."""
        pos, quat = self._pose(d)

        # 1. Odometria: o incremento VERDADEIRO do passo, com o erro de deriva (sem ruído aqui).
        self.odo.passo(pos, quat, self.deriva)
        dxy, dpsi, _dz = self.odo.delta_b()

        # 2. z_ref: a altura estimada da base mais a altura relativa do pé mais baixo (PD-7).
        # ⚠ `site_pe_de_pe` desce o z do site até a sola, para os dois pés.
        z_pe = float(d.site_xpos[self.pes, 2].min()) - self.site_pe_de_pe
        z_pe_t = torch.tensor([z_pe], dtype=torch.float32)
        z_ref = self.odo.pos_w[:, 2] + (z_pe_t - pos[:, 2])

        # 3. Raios: os 205 do passo, no frame do sensor, girados para o mundo pela pose VIVA do site.
        # ⚠ `direcoes(n)` com o mesmo `n` do passo: o treino copia `direcoes(passo + 1)` para o próximo passo,
        # e a leitura de um passo usa a direção do próprio passo. Aqui os dois são `direcoes(n)`.
        dirs = self.padrao.direcoes(self.n)                     # (205,3) float32, frame do sensor
        R = d.site_xmat[self.sensor].reshape(3, 3).copy()
        p = d.site_xpos[self.sensor].copy()
        vec = dirs.numpy().astype(np.float64) @ R.T              # sensor → mundo: R · dir, por raio
        n_raios = vec.shape[0]
        geomid = np.zeros(n_raios, dtype=np.int32)
        dist = np.zeros(n_raios, dtype=np.float64)
        # ⚠ `flg_static=1`: o terreno e a mobília são geoms estáticos e entram no raio. `cutoff` = sem acerto.
        mujoco.mj_multiRay(m, d, p, vec.reshape(-1), self.grupos, 1, self.excluido,
                           geomid, dist, None, n_raios, _SEM_ACERTO)
        dist = np.where(dist < 0, _SEM_ACERTO, dist)
        dist_t = torch.as_tensor(dist, dtype=torch.float32)[None]

        sensor_pos = torch.as_tensor(p, dtype=torch.float32)[None]
        sensor_rot = torch.as_tensor(R, dtype=torch.float32)[None]
        rumo = torch.tensor([_yaw(d.xquat[self.base])], dtype=torch.float32)
        pontos, pontos_w, validos = _PL.pontos_no_frame_do_mapa(
            dist_t, dirs, sensor_pos, sensor_rot, pos, rumo, z_pe_t, self.incl, self.k, None)

        # 4. Filtro do corpo (PD-10): as cápsulas no mundo, a partir da pose VIVA dos corpos.
        pos_k = d.xpos[self.corpo_idx]
        rot_k = d.xmat[self.corpo_idx].reshape(-1, 3, 3)
        p0 = pos_k + np.einsum("kij,kj->ki", rot_k, self.p0)
        p1 = pos_k + np.einsum("kij,kj->ki", rot_k, self.p1)
        capsulas = torch.as_tensor(np.stack([p0, p1], axis=1), dtype=torch.float32)[None]
        validos = _PL.filtra_corpo(pontos_w, validos, capsulas, self.raios, sensor_pos,
                                   self.k.raio_min, self.k.margem_corpo)

        # 5. Transporte e Kalman. ⚠ `prediz` ANTES de `atualiza`: o transporte move o mapa que a leitura preenche.
        self.filtro.prediz(dxy, dpsi, z_ref, _DT)
        self.filtro.atualiza(pontos, validos, dist_t)
        mapa = self.filtro.recorte()[0].numpy().astype(np.float64)
        self.n += 1
        return mapa


# ------------------------------------------------------------------- a observação

def monta_observacao(
    d: mujoco.MjData,
    c: dict[str, np.ndarray],
    twist: np.ndarray,
    tarefa: int,
    anterior: np.ndarray,
    caixa: np.ndarray,
    mapa: np.ndarray,
    hist: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """A observação do ator (1 447 canais) e o histórico novo. Assinatura do contrato do `exporta_cena.py`.

    Devolve `(obs (1447,), hist novo (5, 113))`. Ordem dos blocos: `Contrato.blocos_ator` (I §2.1).
    `hist` é o histórico JÁ ESCALADO, linha 0 = a mais antiga; `None` = primeiro passo depois do reset, e
    o histórico é preenchido com o quadro atual (o `CircularBuffer.append` do mjlab).

    ⚠ `d` PRECISA ESTAR FRESCO: `mj_forward` depois do último `mj_step`. O `sensordata` e o `xquat` de um
    subpasso atrás mudariam a observação.

    ⚠ SEM RUÍDO e SEM ATRASO: o `play` do treino desliga a corrupção e o atraso zera no exportador.

    ⚠ `anterior` entra CRU (sem escala e sem offset): é a `action_manager.action` do mjlab. Quem escala é o
    alvo do PD (`q_default_acao + escala_acao · a`), no laço.

    ⚠ `caixa` e `mapa` entram como ENTRADA: a caixa é zeros na fase D; o mapa vem do `Percepcao.passo`.
    """
    ids_q = np.asarray(c["ids_junta_qpos"], dtype=np.int64)
    ids_v = np.asarray(c["ids_junta_qvel"], dtype=np.int64)
    a_ang = int(c["adr_ang_vel"])
    base_q = d.xquat[int(c["id_base"])]

    # Valores crus de cada bloco com histórico (o nome é a chave; a ordem vem do knobs).
    valores = {
        # IMU de giro, 3 valores em rad/s (escala 0,2).
        "giro_base": np.asarray(d.sensordata[a_ang:a_ang + 3], dtype=np.float64),
        # Gravidade no frame da base: `R⁻¹ · (0, 0, −1)`.
        "gravidade": gira_inverso(base_q, np.array([0.0, 0.0, -1.0])),
        # Twist pedido: vx, vy, wz no frame da base.
        "comando": np.asarray(twist, dtype=np.float64),
        # ⚠ Relativas ao default: `q − q_default` e `qd − qd_default` (o `joint_pos_rel` do mjlab).
        "juntas": d.qpos[ids_q] - np.asarray(c["q_default"], dtype=np.float64),
        "vel_juntas": d.qvel[ids_v] - np.asarray(c["qd_default"], dtype=np.float64),
        "acao": np.asarray(anterior, dtype=np.float64),
        "tarefa": np.eye(len(TAREFAS), dtype=np.float64)[int(tarefa)],
        "caixa": np.asarray(caixa, dtype=np.float64),
    }
    quadro = np.concatenate([valores[nome] for nome, _, _ in _QUADRO]) * _ESC_QUADRO
    if quadro.shape[0] != _N_QUADRO:
        raise AssertionError(f"o quadro tem {quadro.shape[0]} canais, esperado {_N_QUADRO}")

    # Histórico: o quadro novo entra por último; a linha 0 é a mais antiga (ordem do mjlab).
    if hist is None:
        hist_novo = np.repeat(quadro[None], _HIST, axis=0)
    else:
        hist = np.asarray(hist, dtype=np.float64)
        if hist.shape != (_HIST, _N_QUADRO):
            raise ValueError(f"o histórico tem forma {hist.shape}, esperado ({_HIST}, {_N_QUADRO})")
        hist_novo = np.concatenate([hist[1:], quadro[None]], axis=0)

    # Montagem na ordem de `blocos_ator`. Bloco com histórico: as 5 linhas do bloco, em ordem cronológica
    # (`reshape(E, -1)` do mjlab, t−4 primeiro). Bloco sem histórico (o mapa): quadro atual, escala do knobs.
    partes = []
    for nome, _canais, escala, h in _CONTRATO.blocos_ator:
        if nome == "mapa":
            partes.append(np.asarray(mapa, dtype=np.float64) * escala)
        else:
            a, b = _FATIA[nome]
            partes.append(hist_novo[:, a:b].reshape(-1))
    obs = np.concatenate(partes)
    if obs.shape != (DIM_ATOR,):
        raise AssertionError(f"a observação tem forma {obs.shape}, esperado ({DIM_ATOR},)")
    return obs, hist_novo


# ------------------------------------------------------------------------- o ator

class Ator:
    """O MLP do checkpoint, mais o normalizador dele. Determinístico, CPU, float32."""

    def __init__(self, caminho: str | Path):
        alvo = str(Path(caminho).expanduser())
        try:
            estado = torch.load(alvo, map_location="cpu", weights_only=True)
        except Exception:
            # ⚠ Só para checkpoint seu: `weights_only=False` desserializa objeto arbitrário.
            estado = torch.load(alvo, map_location="cpu", weights_only=False)
        sd = estado["actor_state_dict"] if "actor_state_dict" in estado else estado

        pesos = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
        if not pesos:
            raise KeyError("o checkpoint não tem pesos `mlp.*` no `actor_state_dict`")
        indices = sorted({int(k.split(".")[0]) for k in pesos})
        dims = [tuple(pesos[f"{i}.weight"].shape) for i in indices]

        # ELU entre as camadas e nada depois da última: o `tarefa.rl_cfg` parte do `unitree_g1_ppo_runner_cfg`
        # (`activation="elu"`, `hidden_dims=(512, 256, 128)`, `mjlab/tasks/velocity/config/g1/rl_cfg.py:13-16`).
        camadas: list[torch.nn.Module] = []
        for k in range(len(indices)):
            saida, entrada = dims[k]
            camadas.append(torch.nn.Linear(entrada, saida))
            if k < len(indices) - 1:
                camadas.append(torch.nn.ELU())
        self.rede = torch.nn.Sequential(*camadas)

        renomeado = {}
        for k, i in enumerate(indices):
            renomeado[f"{2 * k}.weight"] = pesos[f"{i}.weight"]
            renomeado[f"{2 * k}.bias"] = pesos[f"{i}.bias"]
        self.rede.load_state_dict(renomeado)
        self.rede.eval()
        for p in self.rede.parameters():
            p.requires_grad_(False)

        self.dim_entrada = dims[0][1]
        self.dim_saida = dims[-1][0]

        # ⚠⚠ O NORMALIZADOR É `(x − mean) / (std + 1e-2)`, e NÃO `sqrt(var + 1e-8)` como diz a spec §3.
        # O `rsl_rl` soma `self.eps` = 1e-2 (`rsl_rl/modules/normalization.py`). Num canal de desvio
        # pequeno a diferença é grande.
        if "obs_normalizer._mean" in sd:
            self.media = sd["obs_normalizer._mean"].reshape(-1).float()
            if "obs_normalizer._std" in sd:
                self.desvio = sd["obs_normalizer._std"].reshape(-1).float()
            else:
                self.desvio = sd["obs_normalizer._var"].reshape(-1).float().sqrt()
            self.eps = 1e-2
        else:
            self.media = torch.zeros(self.dim_entrada)
            self.desvio = torch.ones(self.dim_entrada)
            self.eps = 0.0

        self.iteracao = int(estado.get("iter", -1)) if isinstance(estado, dict) else -1

    def normaliza(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        return (x - self.media) / (self.desvio + self.eps)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rede(self.normaliza(obs)).numpy().astype(np.float64)


# ----------------------------------------------------------------------- o teclado

# ⚠ NENHUMA TECLA QUE O `simulate` RESERVA. Os dígitos, o espaço, as setas, o `[` e o `]` são dele.
# O mapa é DADO: a bandeira `--teclas` sobrepõe entradas, e o cabeçalho sai deste dict.
ACOES: dict[str, str] = {
    "vx+": "vx + passo",
    "vx-": "vx − passo",
    "vy+": "vy + passo",
    "vy-": "vy − passo",
    "wz+": "wz + passo",
    "wz-": "wz − passo",
    "zera": "zera o twist",
    **{f"tarefa:{_t}": f"tarefa {_t}" for _t in TAREFAS_TECLADO},
    "teto-": "os três tetos × 0,8",
    "teto+": "os três tetos × 1,25",
    "tempo-": "fator de tempo ÷ 2",
    "tempo+": "fator de tempo × 2",
    "reset": "volta a cena ao reset do treino",
}

# A tecla é o CARACTERE, em minúscula. O `key_callback` recebe o código GLFW (ASCII maiúsculo para letra).
# ⚠ BOTAR é UM comando só: uma tecla põe a tarefa BOTAR, e a cadeia B1→B4 é da política, não do driver.
TECLAS: dict[str, str] = {
    "w": "vx+", "s": "vx-",
    "a": "vy+", "d": "vy-",
    "q": "wz+", "e": "wz-",
    "x": "zera",
    "z": "tarefa:ANDAR", "v": "tarefa:PEGAR",
    "b": "tarefa:CARREGAR", "n": "tarefa:BOTAR",
    "-": "teto-", "=": "teto+",
    ",": "tempo-", ".": "tempo+",
    "p": "reset",
}

assert set(TECLAS.values()) <= set(ACOES), \
    f"tecla apontando para ação que não existe: {sorted(set(TECLAS.values()) - set(ACOES))}"


def analisa_teclas(texto: str) -> dict[str, str]:
    """Lê `--teclas "w=vx+,s=vx-"` e devolve o `TECLAS` com as entradas sobrepostas.

    Parse igual ao do `pilota.py`: a tecla é o PRIMEIRO caractere da entrada, e o `=` é o segundo. Espaço
    separa sempre; a vírgula separa, MENOS quando o próximo caractere é `=`.
    """
    mapa = dict(TECLAS)
    i, n = 0, len(texto)
    while i < n:
        if texto[i] in " \t" or (texto[i] == "," and texto[i + 1:i + 2] != "="):
            i += 1
            continue
        if texto[i + 1:i + 2] != "=":
            raise SystemExit(f"--teclas: esperava <tecla>=<ação> e achei {texto[i:]!r}")
        tecla = texto[i].lower()
        j = i + 2
        while j < n and texto[j] not in ", \t":
            j += 1
        acao = texto[i + 2:j]
        if acao not in ACOES:
            raise SystemExit(f"--teclas: a ação {acao!r} não existe. "
                             f"As que existem: {', '.join(ACOES)}")
        mapa[tecla] = acao
        i = j
    return mapa


def cabecalho(teclas: dict[str, str]) -> str:
    """O mapa CORRENTE, montado do dict, para a partida imprimir."""
    por_acao: dict[str, list[str]] = {}
    for tecla, acao in teclas.items():
        por_acao.setdefault(acao, []).append(tecla)
    linhas = ["[pilota_v3] teclas:"]
    linhas += [f"  {' '.join(por_acao[a]):<10s} {ACOES[a]}" for a in ACOES if a in por_acao]
    return "\n".join(linhas)


# ⚠ O envelope do TREINO e o teto de quem dirige são coisas diferentes. Acima do envelope a política está
# fora da distribuição e o que se vê não diz nada.
_TETOS_PADRAO = (1.0, 0.5, 0.5)


class Piloto:
    """O estado que o teclado muda. Sem máquina de estados: cada tecla é um efeito."""

    def __init__(self, tetos: tuple[float, float, float], envelope: np.ndarray,
                 teclas: dict[str, str] | None = None):
        self.tetos = np.asarray(tetos, dtype=np.float64)
        self.envelope = np.asarray(envelope, dtype=np.float64)
        self.teclas = dict(TECLAS if teclas is None else teclas)
        self.twist = np.zeros(3)
        self.tarefa = TAREFAS.index("ANDAR")
        self.fator = 1.0
        self.reset_pedido = False
        self.aviso = ""
        self.rumo_ref: float | None = None  # rumo segurado na marcha reta; None fora dela

    @property
    def passo(self) -> np.ndarray:
        """`teto / 5`, e não 0,1 fixo: cinco toques chegam ao teto seja qual for o teto."""
        return self.tetos / 5.0

    def _recorta(self) -> None:
        """O teto recorta o valor CORRENTE: baixar o teto derruba o comando no mesmo passo."""
        self.twist = np.clip(self.twist, -self.tetos, self.tetos)

    def _escala_tetos(self, fator: float) -> None:
        novos = self.tetos * fator
        estourou = novos > self.envelope + 1e-9
        self.tetos = np.minimum(novos, self.envelope)
        if estourou.any():
            eixos = ", ".join(nome for nome, e in zip("vx vy wz".split(), estourou) if e)
            self.aviso = (f"teto saturado no envelope do treino ({eixos}): "
                          f"{self.envelope[0]:.2f} / {self.envelope[1]:.2f} / {self.envelope[2]:.2f}")
        else:
            self.aviso = ""
        self._recorta()

    def tecla(self, codigo: int) -> None:
        """O `key_callback` do viewer. Traduz o código GLFW em caractere e despacha.

        ⚠ TECLA DESCONHECIDA SE ANUNCIA: o `simulate` em C++ engole os atalhos dele antes de nos chamar.
        O `\\n` na frente existe porque o laço escreve a linha de status com `\\r`.
        """
        char = chr(codigo).lower() if 32 <= codigo < 127 else ""
        acao = self.teclas.get(char)
        if acao is None:
            print(f"\n[pilota_v3] tecla {codigo} ('{char or '?'}') não está no mapa", flush=True)
            return
        self.aplica(acao)

    def aplica(self, acao: str) -> None:
        p = self.passo
        if acao == "vx+":
            self.twist[0] += p[0]
        elif acao == "vx-":
            self.twist[0] -= p[0]
        elif acao == "vy+":
            self.twist[1] += p[1]
        elif acao == "vy-":
            self.twist[1] -= p[1]
        elif acao == "wz+":
            self.twist[2] += p[2]
        elif acao == "wz-":
            self.twist[2] -= p[2]
        elif acao == "zera":
            self.twist[:] = 0.0
        elif acao.startswith("tarefa:"):
            self.tarefa = TAREFAS.index(acao.split(":", 1)[1])
        elif acao == "teto-":
            self._escala_tetos(0.8)
        elif acao == "teto+":
            self._escala_tetos(1.25)
        elif acao == "tempo-":
            self.fator = max(self.fator / 2.0, 1.0 / 64.0)
        elif acao == "tempo+":
            self.fator = min(self.fator * 2.0, 8.0)
        elif acao == "reset":
            self.reset_pedido = True
        else:
            return
        self._recorta()

    def twist_da_politica(self, yaw: float) -> np.ndarray:
        """O twist que vai para a política: o pedido, com o wz do laço de rumo na marcha reta.

        Marcha reta = vx ou vy pedidos e wz zero. Nela o rumo do início fica segurado e o wz vira
        `K × erro`, recortado no teto de wz (spec §15, "rumo"). Fora dela o laço solta a referência.
        ⚠ Escolha desta lane: o laço só age na marcha reta; parado, o wz segue o pedido (zero).
        """
        t = self.twist.copy()
        marcha_reta = (t[0] != 0.0 or t[1] != 0.0) and t[2] == 0.0
        if not marcha_reta:
            self.rumo_ref = None
            return t
        if self.rumo_ref is None:
            self.rumo_ref = yaw
        erro = _wrap(self.rumo_ref - yaw)
        t[2] = float(np.clip(_K_RUMO * erro, -self.tetos[2], self.tetos[2]))
        return t


# -------------------------------------------------------------------------- o laço

def restaura(m: mujoco.MjModel, d: mujoco.MjData, c: dict[str, np.ndarray], percepcao: Percepcao) -> None:
    """Devolve a cena ao estado do reset do treino, e reinicia a percepção junto.

    ⚠ NÃO É `m.qpos0`: o `qpos0` traz a pose crua do XML e não sabe onde o evento põe a caixa e a laje. O
    exportador grava o qpos, o qvel e o mocap LIDOS do env logo depois do reset.
    """
    mujoco.mj_resetData(m, d)
    d.qpos[:] = np.asarray(c["qpos_inicial"], dtype=np.float64)
    d.qvel[:] = np.asarray(c["qvel_inicial"], dtype=np.float64)
    if m.nmocap:
        d.mocap_pos[:] = np.asarray(c["mocap_pos_inicial"], dtype=np.float64)
        d.mocap_quat[:] = np.asarray(c["mocap_quat_inicial"], dtype=np.float64)
    d.ctrl[np.asarray(c["ids_atuador"], dtype=np.int64)] = np.asarray(c["q_default_acao"], dtype=np.float64)
    mujoco.mj_forward(m, d)
    percepcao.reinicia(m, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb, cena.npz e contrato.json")
    ap.add_argument("--checkpoint", required=True, help="o model_*.pt da v3")
    ap.add_argument("--vx-max", type=float, default=_TETOS_PADRAO[0])
    ap.add_argument("--vy-max", type=float, default=_TETOS_PADRAO[1])
    ap.add_argument("--wz-max", type=float, default=_TETOS_PADRAO[2])
    ap.add_argument("--teclas", default="", help="sobrepõe entradas do mapa: \"w=vx+,s=vx-\"")
    args = ap.parse_args()
    teclas = analisa_teclas(args.teclas)

    m, c, contrato = carrega_cena(args.cena)
    d = mujoco.MjData(m)
    ator = Ator(args.checkpoint)
    if ator.dim_entrada != int(c["dim_obs"]):
        raise SystemExit(f"o checkpoint espera {ator.dim_entrada} canais e a cena monta "
                         f"{int(c['dim_obs'])}. Checkpoint de outra fase?")

    envelope = np.asarray(c["envelope_treino"], dtype=np.float64)
    tetos = np.minimum([args.vx_max, args.vy_max, args.wz_max], envelope)
    piloto = Piloto(tuple(tetos), envelope, teclas)
    percepcao = Percepcao(m, c)

    ids_atuador = np.asarray(c["ids_atuador"], dtype=np.int64)
    q_default_acao = np.asarray(c["q_default_acao"], dtype=np.float64)
    escala_acao = np.asarray(c["escala_acao"], dtype=np.float64)
    decimation = int(c["decimation"])
    dt = float(c["physics_dt"]) * decimation

    restaura(m, d, c, percepcao)
    acao = np.zeros(ator.dim_saida)
    hist: np.ndarray | None = None
    caixa = np.zeros(N_CAIXA)  # ⚠ fase D: a caixa é zero, como o `caixa_zero` do treino

    print(f"[pilota_v3] cena {args.cena}  checkpoint iter={ator.iteracao}  "
          f"dt={dt * 1000:.0f} ms  decimation={decimation}  dim_obs={int(c['dim_obs'])}")
    print(cabecalho(piloto.teclas))
    # Nota: o estimador da caixa e o detector de fim NÃO entram no laço da fase D; entram na E9 (§20, 29/10).
    print("[pilota_v3] fase D: caixa zerada; estimador da caixa e detector de fim NÃO entram (fase E, E9)")

    # ⚠ `from ... import ... as`, e NÃO `import mujoco.viewer`: a segunda forma liga `mujoco` no escopo LOCAL
    # de `main` e o `mujoco.MjData(m)` lá em cima passa a levantar `UnboundLocalError`.
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(m, d, key_callback=piloto.tecla) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            if piloto.reset_pedido:
                restaura(m, d, c, percepcao)
                acao[:] = 0.0
                hist = None
                piloto.twist[:] = 0.0
                piloto.rumo_ref = None
                piloto.reset_pedido = False

            twist_pol = piloto.twist_da_politica(_yaw(d.xquat[percepcao.base]))
            mapa = percepcao.passo(m, d)
            obs, hist = monta_observacao(d, c, twist_pol, piloto.tarefa, acao, caixa, mapa, hist)
            acao = ator(obs)

            # ⚠ SEM CLAMP NOSSO: os atuadores são servos de posição com `ctrlrange`, e o MuJoCo satura.
            d.ctrl[ids_atuador] = q_default_acao + escala_acao * acao

            for _ in range(decimation):
                mujoco.mj_step(m, d)
            # ⚠ `mj_forward` obrigatório: o `mj_step` deixa `xpos`, `xquat` e `sensordata` um subpasso atrás.
            mujoco.mj_forward(m, d)

            viewer.sync()

            medido = np.array([d.sensordata[int(c["adr_lin_vel"])],
                               d.sensordata[int(c["adr_lin_vel"]) + 1],
                               d.sensordata[int(c["adr_ang_vel"]) + 2]])
            print(f"\r{TAREFAS[piloto.tarefa]:<10s} "
                  f"pedido {piloto.twist[0]:+.2f} {piloto.twist[1]:+.2f} {piloto.twist[2]:+.2f} | "
                  f"rumo {twist_pol[2]:+.2f} | "
                  f"medido {medido[0]:+.2f} {medido[1]:+.2f} {medido[2]:+.2f} | "
                  f"tetos {piloto.tetos[0]:.2f}/{piloto.tetos[1]:.2f}/{piloto.tetos[2]:.2f} | "
                  f"tempo x{piloto.fator:g} {piloto.aviso:<60s}", end="", flush=True)

            atraso = dt / piloto.fator - (time.perf_counter() - t0)
            if atraso > 0:
                time.sleep(atraso)
    print()


if __name__ == "__main__":
    main()
