"""Exporta a cena da v3 para o MuJoCo clássico e prova a paridade da observação do ator (D11 do plano).

    python -m g1_v3.exporta_cena --saida ~/g1_pilota_v3/
    python -m g1_v3.exporta_cena --paridade
    python -m g1_v3.exporta_cena --paridade --cena ~/g1_pilota_v3/

Roda UMA vez, e precisa do mjlab (constrói o env). O consumidor é o `pilota_v3.py`, na raiz do repo, que
carrega `knobs`, `alvos`, `estimador` e `percepcao_lidar` por caminho e não importa nada daqui.

⚠ NADA É DIGITADO À MÃO. Todo número sai do env construído por `make_env_cfg(play=True)` ou do `knobs.py`.

⚠ O `.mjb` NÃO VAI PARA O GIT. O `.gitignore` já tem `*.mjb`. O gerador é que é versionado.

⚠ A CONTA DA OBSERVAÇÃO NÃO MORA AQUI. Ela mora em `pilota_v3.monta_observacao`, e a `paridade` a IMPORTA
por caminho. Duas cópias divergem, e a paridade passaria a testar a si mesma.

⚠ ATRASO DA OBSERVAÇÃO ZERADO NO EXPORTADOR. Em `play`, os cinco termos com atraso (`giro_base`, `gravidade`,
`juntas`, `vel_juntas`, `mapa`) ainda sorteiam lag 0..1 no reset (PD-25). O pilota não tem atraso de
observação: o `constroi_env` põe o teto dos buffers em 0 (como o `encoder_bias`, zerado pelo g1_limpo).

## Contrato com o pilota_v3

A lane D11b escreve o `pilota_v3.py` SÓ por este texto. Se um campo mudar aqui, muda no texto e na lane.

O `cena.mjb` é o robô e a caixa NOMINAIS, sem a DR do treino (atritos, massa do tronco, tamanho e massa da caixa).

### `cena.npz` (`np.load(..., allow_pickle=False)`)

Os campos de `nomes_juntas` a `envelope_treino` saem de `coleta_cena`; os de `qpos_inicial` a
`mocap_quat_inicial` saem de `exporta` (estado do reset). Forma e dtype:

| chave | dtype, forma | unidade e significado |
|---|---|---|
| `nomes_juntas` | str (29,) | nome da junta, na ordem do ator (= `knobs.JUNTAS`, conferido) |
| `ids_junta_qpos` | int64 (29,) | índice de cada junta em `d.qpos` |
| `ids_junta_qvel` | int64 (29,) | índice de cada junta em `d.qvel` |
| `q_default` | f64 (29,) | rad; pose de pé do env. O termo `juntas` é `q[ids_junta_qpos] − q_default` |
| `qd_default` | f64 (29,) | rad/s; zeros. O termo `vel_juntas` é `qd[ids_junta_qvel] − qd_default` |
| `q_default_acao` | f64 (29,) | rad; offset da ação (`default_joint_pos[target_ids]`) |
| `escala_acao` | f64 (29,) | rad por unidade de ação; alvo do PD = `q_default_acao + escala_acao · a` |
| `ids_atuador` | int64 (29,) | índice de cada junta em `d.ctrl` (mapa pelo `actuator_trnid`) |
| `adr_lin_vel` | int64 () | endereço do IMU de velocidade linear em `d.sensordata` (só o crítico usa) |
| `adr_ang_vel` | int64 () | endereço do giro da base em `d.sensordata`: 3 valores, rad/s, frame do IMU |
| `id_base` | int64 () | body id do root (pelve); a gravidade projetada usa `d.xquat[id_base]` |
| `id_caixa`, `id_laje` | int64 () | body ids da caixa e da laje |
| `meia_aresta` | f64 () | m; meia aresta nominal, lida do `geom_size` do modelo |
| `decimation` | int64 () | passos de física por passo da política (4) |
| `physics_dt` | f64 () | s; passo de física (0,005) |
| `dim_obs` | int64 () | 1447 |
| `escala_obs` | f64 (1447,) | escala de cada canal do ator, já expandida pelo histórico (conferida contra o cfg) |
| `envelope_treino` | f64 (3,) | [vx, vy, wz] máximos do twist: m/s, m/s, rad/s |
| `qpos_inicial` | f64 (nq,) | estado inicial do reset (`exporta`) |
| `qvel_inicial` | f64 (nv,) | idem |
| `mocap_pos_inicial` | f64 (nmocap, 3) | m; zeros (0, 3) se não há mocap |
| `mocap_quat_inicial` | f64 (nmocap, 4) | quaternion wxyz; zeros (0, 4) se não há mocap |
| `origens_terreno` | f64 (linhas, colunas, 3) | m; `env.scene["terrain"].terrain_origins`: a origem de cada sub-terreno, [linha, coluna] |
| `colunas_terreno` | str (9,) | nome de cada coluna do gerador, na ordem de `ATIVO.terreno.colunas` |
| `origem_reset` | f64 (3,) | m; `env.scene.env_origins[0]` depois do reset forçado da fatia B no plano: a origem da cena exportada (o `--coluna` parte dela) |

- Reset exportado: a fatia B no plano (coluna `plano`, `knobs.colunas[0]`), com a laje e a caixa à frente do robô
  (E §4.4; S4). O chão do driver é o plano z = 0 (F9 M12). O `pilota_v3` põe o topo da laje de destino sobre z = 0.

### `contrato.json` (UTF-8, `indent=2`)

| chave | tipo, forma | significado |
|---|---|---|
| `pacote` | str | `"g1_v3"` |
| `versao` | str | `git rev-parse --short HEAD` do repo na hora da exportação; `"desconhecida"` se o git falha |
| `mjlab` | str | versão do mjlab instalada (importlib.metadata) |
| `juntas` | list[str] (29) | nomes na ordem do ator |
| `q_de_pe` | list[float] (29) | rad; `knobs.q_de_pe()` |
| `kp`, `kd` | list[float] (29) | Nm/rad e Nm·s/rad; por `Contrato.pd` (regex que casa a junta, exatamente uma) |
| `escala_acao` | list[float] (29) | igual a `cena.npz["escala_acao"]` |
| `dt` | float | s; `physics_dt · decimation` = 0,02 |
| `dt_fisica`, `decimacao` | float, int | `Contrato.dt_fisica` (0,005), `Contrato.decimacao` (4) |
| `blocos_ator` | list[dict] | um dict por bloco, na ordem: `{"nome", "canais", "escala", "historico", "dim"}`; `dim` = canais × max(historico, 1) |
| `dim_ator` | int | 1447 |
| `tarefas` | list[str] (6) | `knobs.TAREFAS`; a ordem é o índice do one-hot |
| `alvos` | list[str] (9) | `knobs.ALVOS`; só para o log |
| `mapa` | dict | `{"celula_m": 0.10, "n_recorte": 21, "n_interno": 51, "altura_clip": 1.0, "canais": 882, "ordem": "..."}` |
| `T_BASE_SENSOR` | list[list[float]] (4, 4) | pose do site `mid360` no frame do body base, em q_de_pé; m |
| `base`, `sensor_site` | str | nome do body base (pelve) e do site (`"mid360"`) |

### `monta_observacao` (em `pilota_v3.py`, chamada pela `paridade`)

```
monta_observacao(
    d: mujoco.MjData,          # estado vivo, com mj_forward feito (espelho do env)
    c: dict[str, np.ndarray],  # os campos de cena.npz (usa ids_junta_qpos, ids_junta_qvel, q_default,
                               #   qd_default, id_base, adr_ang_vel); escalas vêm de knobs.Contrato
    twist: np.ndarray,         # (3,) f64: [vx, vy, wz] do comando, m/s, m/s, rad/s
    tarefa: int,               # 0..5, índice em knobs.TAREFAS
    anterior: np.ndarray,      # (29,) f64: ação do passo anterior, CRUA (sem escala nem offset)
    caixa: np.ndarray,         # (11,) f64: canais atuais da caixa (zeros na fase D)
    mapa: np.ndarray,          # (882,) f64: FiltroDeElevacao.recorte(): 441 alturas, depois 441 confianças
    hist: np.ndarray | None,   # (5, 113) f64: histórico já ESCALADO, linha 0 = mais antigo; None = 1º passo
) -> tuple[np.ndarray, np.ndarray]:
    # devolve (obs (1447,) f64, hist novo (5, 113) f64)
```

- Quadro de 113 canais, na ordem: `giro_base`(3), `gravidade`(3), `comando`(3), `juntas`(29), `vel_juntas`(29),
  `acao`(29), `tarefa`(6, one-hot), `caixa`(11). Cada canal é multiplicado pela escala do bloco ANTES de entrar
  no histórico (pipeline do mjlab: compute → noise → clip → scale → delay → history).
- `hist is None` (primeiro passo depois do reset): preenche as 5 linhas com o quadro atual, como o
  `CircularBuffer.append` do mjlab (`circular_buffer.py`, backfill no primeiro push).
- Histórico: o mjlab faz `buffer.reshape(E, -1)` com buffer (E, 5, C) em ordem CRONOLÓGICA, o MAIS ANTIGO
  PRIMEIRO (`circular_buffer.py:162-175`). Logo, para cada bloco com histórico, o trecho de 5·C valores é
  `[t−4 (C); t−3 (C); t−2 (C); t−1 (C); t0 (C)]`: o mais NOVO é o ÚLTIMO C do bloco.
- Ordem do vetor de 1447: os 9 blocos de `Contrato.blocos_ator` (`giro_base` 15, `gravidade` 15, `comando` 15,
  `juntas` 145, `vel_juntas` 145, `acao` 145, `tarefa` 30, `caixa` 55, `mapa` 882). `mapa` não tem histórico e
  entra com o quadro atual, sem escala (escala 1,0).
- `anterior` entra CRU: `acao` do mjlab é `action_manager.action`, sem escala. Quem escala é o alvo do PD.

### A paridade (`exporta_cena.paridade`)

- Por passo, lê do env: `twist` = `command_manager.get_command("twist")`; `anterior` = `action_manager.action`;
  `tarefa` = `env.v3_tarefa[0]`; `caixa` = os últimos 11 canais do bloco `caixa` da observação do env (o quadro
  atual); `mapa` = o bloco `mapa` da observação do env. Zera `hist` quando o env reinicia (`reset_buf`).
- Compara, bloco a bloco, os 9 blocos de `observation_manager.compute()["actor"]` com a saída do pilota, Δmax < tol.
- ⚠ `caixa` e `mapa` vêm do env, então o bloco é IGUAL POR CONSTRUÇÃO. A paridade prova os outros 7 blocos e a
  ordem e o histórico.
- Extra: `ctrl` (o mapa junta→atuador contra `d.ctrl`), como no g1_limpo, fora do passo de reset.

Decisão de 29/10 (D11a): o pilota recebe `mapa` e `caixa` de fora. A `paridade` NÃO os prova. Ela tem uma
`Percepcao` do pilota (lidar por `mj_multiRay`, reiniciada no reset) e, a cada passo, IMPRIME a diferença entre o
mapa dela e o `mapa` do env: Δmax das alturas nas células vistas pelos dois, e a fração de células em que as
máscaras de vistas concordam (vista = confiança > `knobs.Percepcao.limiar_visto`). Sem portão: não entra no
veredito; quem julga é o dono, na fase I.

Ponto de entrada: `python -m g1_v3.exporta_cena`. `--saida PASTA` exporta; `--paridade` roda a paridade (com
`--cena PASTA`, também confere que o `cena.npz` gravado é o do env). Sai com código 1 se a paridade falhar.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_v3.knobs import ALVOS, ATIVO, JUNTAS, TAREFAS, dim_ator, q_de_pe

__all__ = ["constroi_env", "coleta_cena", "exporta", "paridade", "main"]

# O grupo que o robô vê. O `critic` tem canais privilegiados e não interessa aqui.
GRUPO = "actor"

# Raiz do repo: g1_v3/exporta_cena.py → parents[1]. O `pilota_v3.py` mora ali (D11b).
_RAIZ = Path(__file__).resolve().parents[1]


def _np(x) -> np.ndarray:
    """Tira um `np.ndarray` de tensor torch, de `TorchArray` do mjlab, de escalar ou de lista."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _fatias(c) -> dict[str, slice]:
    """Fatia de cada bloco do ator no vetor de 1447 (ordem de `Contrato.blocos_ator`; bloco com histórico ocupa canais × H)."""
    saida, i = {}, 0
    for nome, canais, _escala, hist in c.blocos_ator:
        n = canais * max(hist, 1)
        saida[nome] = slice(i, i + n)
        i += n
    return saida


# --------------------------------------------------------------------------- o env

def constroi_env():
    """Constrói o env da v3: `make_env_cfg(play=True)`, 1 env, CPU. A v3 não tem elo: a cena é a do `play`.

    ⚠ Duas coisas são zeradas aqui, por necessidade e não por conveniência:

    `encoder_bias` — evento de startup que soma até `DR.vies_encoder` ao `joint_pos` do ATOR (o termo `juntas`
    usa `biased=True`). Não existe no robô real, e o MuJoCo clássico não tem onde guardá-lo. Deixá-lo ligado
    faria a paridade falhar em 29 canais por um motivo que não é erro de convenção.

    teto do atraso da observação — o evento `atraso` (D7) sorteia lag 0..1 nos cinco termos com atraso. O
    pilota não tem atraso de observação; com o teto em 0, o lag sorteado é sempre 0. O atraso do atuador
    (ação) não entra no ator e fica como está (ver o relatório).
    """
    from mjlab.envs import ManagerBasedRlEnv

    from g1_v3.env_cfg import make_env_cfg

    if not ATIVO.percepcao.ligada:
        raise AssertionError("o exportador exige Percepcao.ligada: o contrato tem o bloco `mapa`")
    cfg = make_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    env = ManagerBasedRlEnv(cfg, device="cpu")

    env.scene["robot"].data.encoder_bias.zero_()
    # ⚠ Atraso da OBSERVAÇÃO, teto 0: o `compute` lê o quadro atual. Privado do mjlab 1.5.3, como o `eventos.atraso`.
    for buf in env.observation_manager._group_obs_term_delay_buffer[GRUPO].values():
        buf.max_lag = 0
    return env


# ------------------------------------------------------------------- a coleta

def coleta_cena(env) -> dict[str, np.ndarray]:
    """Lê do env tudo o que o `pilota_v3` precisa para montar a observação fora do env. Todo campo tem fonte no env."""
    m: mujoco.MjModel = env.sim.mj_model
    cfg = env.cfg
    c = ATIVO.contrato
    robo = env.scene["robot"]
    caixa = env.scene["box"]
    laje = env.scene["table"]

    # ---- as juntas, na ordem do ator (= ordem do termo `joint_pos_rel`, que é a do robô)
    nomes_juntas = [str(n) for n in robo.joint_names]
    if tuple(nomes_juntas) != JUNTAS:
        raise AssertionError(f"ordem das juntas do env difere de knobs.JUNTAS:\n  env:   {nomes_juntas}\n"
                             f"  knobs: {list(JUNTAS)}")
    ids_q = _np(robo.indexing.joint_q_adr).astype(np.int64)
    ids_v = _np(robo.indexing.joint_v_adr).astype(np.int64)
    q_default = _np(robo.data.default_joint_pos)[0].astype(np.float64)
    qd_default = _np(robo.data.default_joint_vel)[0].astype(np.float64)

    # ---- a ação: escala, offset e o mapa junta -> ctrl
    termo = env.action_manager._terms["joint_pos"]
    alvo_ids = _np(termo._target_ids).astype(np.int64)
    escala = termo._scale
    escala_acao = (_np(escala)[0] if hasattr(escala, "shape")
                   else np.full(len(alvo_ids), float(escala))).astype(np.float64)
    # ⚠ O offset é `default_joint_pos[target_ids]` (`use_default_offset=True`), e não o `q_default` da ordem do ator.
    q_default_acao = _np(termo._offset)[0].astype(np.float64)
    ids_atuador = _mapa_atuador(m, _np(robo.indexing.joint_ids), alvo_ids)

    # ---- o IMU: giro da base (ator) e velocidade linear (crítico)
    adr_lin = int(m.sensor("robot/imu_lin_vel").adr[0])
    adr_ang = int(m.sensor("robot/imu_ang_vel").adr[0])
    assert int(m.sensor("robot/imu_lin_vel").dim[0]) == 3
    assert int(m.sensor("robot/imu_ang_vel").dim[0]) == 3

    # ---- os corpos e a meia aresta nominal da caixa (do modelo, não de um sorteio)
    id_base = int(robo.indexing.root_body_id)
    id_caixa = int(caixa.indexing.root_body_id)
    id_laje = int(laje.indexing.root_body_id)
    gid = int(_np(caixa.indexing.geom_ids)[0])
    meia = float(m.geom_size[gid, 0])

    # ---- o envelope de velocidade do treino, lido do cfg (modo play)
    faixas = cfg.commands["twist"].ranges
    envelope = np.array([max(abs(v) for v in faixas.lin_vel_x),
                         max(abs(v) for v in faixas.lin_vel_y),
                         max(abs(v) for v in faixas.ang_vel_z)], dtype=np.float64)

    return {
        "nomes_juntas": np.array(nomes_juntas),
        "ids_junta_qpos": ids_q,
        "ids_junta_qvel": ids_v,
        "q_default": q_default,
        "qd_default": qd_default,
        "q_default_acao": q_default_acao,
        "escala_acao": escala_acao,
        "ids_atuador": ids_atuador,
        "adr_lin_vel": np.int64(adr_lin),
        "adr_ang_vel": np.int64(adr_ang),
        "id_base": np.int64(id_base),
        "id_caixa": np.int64(id_caixa),
        "id_laje": np.int64(id_laje),
        "meia_aresta": np.float64(meia),
        "decimation": np.int64(cfg.decimation),
        "physics_dt": np.float64(cfg.sim.mujoco.timestep),
        "dim_obs": np.int64(dim_ator(c)),
        "escala_obs": _escala_por_canal(env),
        "envelope_treino": envelope,
    }


def _mapa_atuador(m: mujoco.MjModel, joint_ids: np.ndarray, alvo_ids: np.ndarray) -> np.ndarray:
    """Junta -> índice em `d.ctrl`, na ordem da AÇÃO.

    ⚠ NÃO SUPÕE IDENTIDADE. O `BuiltinActuatorGroup` escreve `ctrl[ctrl_ids] = joint_pos_target[target_ids]`, e
    nada garante que a ordem dos atuadores no modelo siga a ordem das juntas. A derivação é pelo `actuator_trnid`,
    e a paridade a confere contra o `d.ctrl` que o mjlab escreve de verdade.
    """
    jid_do_alvo = np.asarray(joint_ids, dtype=np.int64)[np.asarray(alvo_ids)]
    saida = np.full(len(alvo_ids), -1, dtype=np.int64)
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(m.actuator_trnid[a, 0])
        casa = np.nonzero(jid_do_alvo == jid)[0]
        if len(casa):
            saida[casa[0]] = a
    if (saida < 0).any():
        faltam = np.nonzero(saida < 0)[0].tolist()
        raise AssertionError(f"juntas sem atuador de posição: índices {faltam}")
    return saida


def _escala_por_canal(env) -> np.ndarray:
    """A escala de cada canal do ator, já expandida pelo histórico (1447,). Confere contra `Contrato.blocos_ator`.

    ⚠ Não suponha 1,0: a escala de `giro_base` é 0,2 e a de `vel_juntas` 0,05. Ler do cfg é o que mantém o
    contrato verdadeiro depois de uma mudança do knobs.
    """
    c = ATIVO.contrato
    cfg_grupo = env.cfg.observations[GRUPO]
    ativos = list(env.observation_manager.active_terms[GRUPO])
    esperado = [nome for nome, *_ in c.blocos_ator]
    if ativos != esperado:
        raise AssertionError(f"a ordem dos termos do ator mudou.\n  mjlab:  {ativos}\n  knobs:  {esperado}")
    pedacos = []
    for nome, canais, escala, hist in c.blocos_ator:
        s = cfg_grupo.terms[nome].scale
        v = (np.ones(canais) if s is None
             else np.broadcast_to(np.asarray(_np(s), dtype=np.float64).reshape(-1), (canais,)))
        if not np.allclose(v, escala):
            raise AssertionError(f"escala do termo {nome} no cfg ({v[0]}) difere do knobs ({escala})")
        # ⚠ o histórico é TIME-MAJOR (5 quadros de C canais): a escala do canal repete a cada quadro.
        pedacos.append(np.tile(v, max(hist, 1)))
    return np.concatenate(pedacos)


# ------------------------------------------------------------------ o estado vivo

def espelha_estado(env, m: mujoco.MjModel) -> mujoco.MjData:
    """Copia o estado do env para um `MjData` clássico e roda `mj_forward`.

    ⚠ `mj_forward`, e não `mj_step`: o mesmo estado com as grandezas derivadas frescas, que é o que o mjlab
    faz antes de calcular a observação.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = _np(env.sim.data.qpos)[0].astype(np.float64)
    d.qvel[:] = _np(env.sim.data.qvel)[0].astype(np.float64)
    d.ctrl[:] = _np(env.sim.data.ctrl)[0].astype(np.float64)
    if m.nmocap:
        d.mocap_pos[:] = _np(env.sim.data.mocap_pos)[0].astype(np.float64)
        d.mocap_quat[:] = _np(env.sim.data.mocap_quat)[0].astype(np.float64)
    mujoco.mj_forward(m, d)
    return d


# ------------------------------------------------------- o contrato em disco

def _pd_por_junta(nomes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """kp e kd de cada junta por `Contrato.pd`. ⚠ Exatamente UMA regex pode casar cada junta."""
    kp, kd = [], []
    for nome in nomes:
        casam = [(p, d) for regex, p, d in ATIVO.contrato.pd if re.fullmatch(regex, nome)]
        if len(casam) != 1:
            raise AssertionError(f"a junta {nome} casa {len(casam)} regex de Contrato.pd (tem de ser 1)")
        kp.append(casam[0][0])
        kd.append(casam[0][1])
    return np.array(kp, dtype=np.float64), np.array(kd, dtype=np.float64)


def _t_base_sensor(m: mujoco.MjModel, d: mujoco.MjData, id_base: int, ids_q: np.ndarray) -> np.ndarray:
    """Pose do site `mid360` no frame do body base (4×4), com as juntas em q_de_pé.

    ⚠ O site está no `torso_link`, e a cintura o move: a pose depende das juntas. Por isso é medida em q_de_pé,
    e não na pose de reset. O pilota, para a pose ao vivo, usa `d.site_xpos` / `d.site_xmat` do próprio modelo.
    """
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    d2.qpos[ids_q] = q_de_pe()
    mujoco.mj_forward(m, d2)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360")
    if sid < 0:
        raise AssertionError("site `mid360` não existe no modelo")
    rb = d2.xmat[id_base].reshape(3, 3)
    pb = d2.xpos[id_base]
    rs = d2.site_xmat[sid].reshape(3, 3)
    ps = d2.site_xpos[sid]
    t = np.eye(4)
    t[:3, :3] = rb.T @ rs
    t[:3, 3] = rb.T @ (ps - pb)
    return t


def _versao_do_repo() -> str:
    """`git rev-parse --short HEAD` do repo, sem shell. `"desconhecida"` se o git falha (sem git, fora de um repo)."""
    try:
        r = subprocess.run(["git", "-C", str(_RAIZ), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=False)
    except OSError:
        return "desconhecida"
    saida = r.stdout.strip()
    return saida if r.returncode == 0 and saida else "desconhecida"


def _contrato(m: mujoco.MjModel, dados: dict[str, np.ndarray], t_base: np.ndarray) -> dict:
    """O `contrato.json`. Tudo sai de `dados`, do modelo ou de `Contrato`/`knobs`, e nada é digitado."""
    c = ATIVO.contrato
    nomes = [str(n) for n in dados["nomes_juntas"]]
    kp, kd = _pd_por_junta(nomes)
    for j, a in enumerate(dados["ids_atuador"]):
        # MuJoCo: o atuador de posição tem `gainprm[0]` = kp e `biasprm[2]` = −kd (`mjlab/utils/spec.py:307-313`).
        if abs(float(m.actuator_gainprm[int(a), 0]) - kp[j]) > 1e-6 * max(1.0, kp[j]):
            raise AssertionError(f"kp da junta {nomes[j]} no modelo ({m.actuator_gainprm[int(a), 0]}) "
                                 f"difere de Contrato.pd ({kp[j]})")
        if abs(-float(m.actuator_biasprm[int(a), 2]) - kd[j]) > 1e-6 * max(1.0, kd[j]):
            raise AssertionError(f"kd da junta {nomes[j]} no modelo ({-m.actuator_biasprm[int(a), 2]}) "
                                 f"difere de Contrato.pd ({kd[j]})")
    dt = float(dados["physics_dt"]) * int(dados["decimation"])
    if abs(dt - c.dt_fisica * c.decimacao) > 1e-12:
        raise AssertionError(f"dt do env ({dt}) difere de Contrato ({c.dt_fisica * c.decimacao})")
    return {
        "pacote": "g1_v3",
        "versao": _versao_do_repo(),
        "mjlab": importlib.metadata.version("mjlab"),
        "juntas": nomes,
        "q_de_pe": q_de_pe().tolist(),
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "escala_acao": dados["escala_acao"].tolist(),
        "dt": dt,
        "dt_fisica": c.dt_fisica,
        "decimacao": c.decimacao,
        "blocos_ator": [{"nome": n, "canais": ch, "escala": e, "historico": h, "dim": ch * max(h, 1)}
                        for n, ch, e, h in c.blocos_ator],
        "dim_ator": dim_ator(c),
        "tarefas": list(TAREFAS),
        "alvos": list(ALVOS),
        "mapa": {"celula_m": ATIVO.percepcao.celula, "n_recorte": ATIVO.percepcao.n_recorte,
                 "n_interno": ATIVO.percepcao.n_interno, "altura_clip": ATIVO.percepcao.altura_clip,
                 "canais": 2 * ATIVO.percepcao.n_recorte ** 2,
                 "ordem": "441 alturas e depois 441 confianças; cada 441 em linha-major, x decrescente, y decrescente"},
        "T_BASE_SENSOR": t_base.tolist(),
        "base": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(dados["id_base"])),
        "sensor_site": "mid360",
    }


# -------------------------------------------------------------------- a exportação

def exporta(pasta: Path) -> None:
    """Grava `cena.mjb`, `cena.npz` e `contrato.json` na pasta. Sem paridade: ela roda em `paridade`."""
    pasta = Path(pasta).expanduser()
    pasta.mkdir(parents=True, exist_ok=True)

    env = constroi_env()
    m: mujoco.MjModel = env.sim.mj_model
    dados = coleta_cena(env)

    # ⚠ A meia do reset é a NOMINAL, a mesma do `cena.mjb`, e não a sorteada pelo startup `tamanho_caixa`: sem isto, o
    # `recuo` e o `topo + v3_meia` do reset forçado usam uma caixa que o `mjb` não tem. O `v3_meia` já existe aqui.
    env.v3_meia[:] = float(dados["meia_aresta"])
    env.reset()
    # S2 (G7): o reset exportado é a fatia B no plano, com a laje e a caixa à frente do robô (E §4.4; S4).
    # ⚠ Mesmo caminho do treino (`sorteia_fatia` e os eventos `cena` e `base`), como o `_abre_b_na_escada` do smoke,
    # sem a coluna de escada. A coluna 0 é o `plano` (`ATIVO.terreno.colunas[0]`): sem terreno, `posiciona_cena` dá rumo 0.
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B

    ids = torch.arange(env.num_envs, device=env.device)
    terreno = env.scene["terrain"]
    terreno.terrain_types[ids] = 0
    env.scene.env_origins[ids] = terreno.terrain_origins[terreno.terrain_levels[ids], 0]
    env.v3_fatia[:] = FATIA_B
    eventos.posiciona_cena(env, ids)
    eventos.reset_base_por_fatia(env, ids, dict(ATIVO.cena.reset_base_andar), dict(ATIVO.cena.reset_base_manipula))
    eventos.reinicia_filtros(env, ids)
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
    # ⚠ O evento `carga` (reset) escreve só `sim.model.body_mass`, o modelo do Warp. O `cena.mjb` sai de `sim.mj_model`,
    # que é nominal por desenho: o `carga` não chega ao `mjb`. No nível 0 do PEGAR a massa sorteada é 1,0 kg, igual à nominal.
    d0 = espelha_estado(env, m)
    dados["qpos_inicial"] = d0.qpos.copy()
    dados["qvel_inicial"] = d0.qvel.copy()
    dados["mocap_pos_inicial"] = d0.mocap_pos.copy() if m.nmocap else np.zeros((0, 3))
    dados["mocap_quat_inicial"] = d0.mocap_quat.copy() if m.nmocap else np.zeros((0, 4))
    # G6 (terreno): a grade do gerador, os nomes das colunas e a origem em que o reset pôs o env 0. O
    # `registra_juntas_v3 --coluna` parte dela.
    dados["origens_terreno"] = _np(env.scene["terrain"].terrain_origins).astype(np.float64)
    dados["colunas_terreno"] = np.array(ATIVO.terreno.colunas)
    dados["origem_reset"] = _np(env.scene.env_origins)[0].astype(np.float64)

    t_base = _t_base_sensor(m, d0, int(dados["id_base"]), dados["ids_junta_qpos"])
    contrato = _contrato(m, dados, t_base)

    caminho_mjb = pasta / "cena.mjb"
    mujoco.mj_saveModel(m, str(caminho_mjb), None)
    np.savez(pasta / "cena.npz", **dados)
    (pasta / "contrato.json").write_text(json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8")

    tamanho = caminho_mjb.stat().st_size / (1024 * 1024)
    print(f"[exporta] {caminho_mjb}  {tamanho:.1f} MB")
    print(f"[exporta] {pasta / 'cena.npz'}  {len(dados)} campos")
    print(f"[exporta] {pasta / 'contrato.json'}  dim_ator={contrato['dim_ator']}")
    print(f"[exporta] nq={m.nq} nv={m.nv} nu={m.nu} nmocap={m.nmocap}  dt={contrato['dt']} s")
    print(f"[exporta] envelope de treino={dados['envelope_treino']}  meia_aresta nominal={float(dados['meia_aresta']):.4f} m")
    env.close()


# ------------------------------------------------------------------- a paridade

def _carrega_pilota():
    """Carrega o `pilota_v3.py` da raiz por caminho. ⚠ Import tardio e por caminho: o arquivo é da lane D11b."""
    caminho_pilota = _RAIZ / "pilota_v3.py"
    spec = importlib.util.spec_from_file_location("pilota_v3", caminho_pilota)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"não achei o pilota_v3 em {caminho_pilota}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _confere_npz(dados: dict[str, np.ndarray], gravado) -> None:
    """Confere que o `cena.npz` gravado é o que o env dá agora (os campos em comum)."""
    for chave, valor in dados.items():
        if chave not in gravado.files:
            continue
        a, b = np.asarray(valor), gravado[chave]
        if a.dtype.kind in "USO":
            ok = a.shape == b.shape and bool((a == b).all())
        else:
            ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-9)
        if not ok:
            raise AssertionError(f"cena.npz gravado diverge do env no campo {chave}; reexporte com --saida")


def _compara_mapas(mapa_pilota: np.ndarray, mapa_env: np.ndarray, limiar: float) -> tuple[float, float]:
    """Δmax das alturas nas células vistas pelos DOIS, e a fração de células em que as máscaras de vistas concordam.

    O mapa são 441 alturas e depois 441 confianças (ordem do contrato). Vista = confiança > `limiar`.
    Sem célula vista pelos dois, o Δmax sai `nan`.
    """
    n = len(mapa_env) // 2
    h_p, c_p = mapa_pilota[:n], mapa_pilota[n:]
    h_e, c_e = mapa_env[:n], mapa_env[n:]
    vista_p, vista_e = c_p > limiar, c_e > limiar
    ambas = vista_p & vista_e
    dmax = float(np.abs(h_p[ambas] - h_e[ambas]).max()) if ambas.any() else float("nan")
    return dmax, float((vista_p == vista_e).mean())


def paridade(pasta: Path | None = None, tol: float = 1e-5, passos: int = 200) -> bool:
    """O PORTÃO. Compara os 9 blocos do ator do env com os do `pilota_v3.monta_observacao`, e o `ctrl`.

    ⚠ Ruído desligado pelo `play=True` (`enable_corruption` do ator). Sem `encoder_bias` e sem atraso de observação
    (`constroi_env`). Sem ruído, a conta do pilota tem de bater com a do env a `tol`.

    ⚠ `caixa` e `mapa` entram no pilota como ENTRADA, lida do env: a paridade não os prova (ver a docstring do
    módulo). Os outros 7 blocos, o histórico e a ordem, a paridade prova de verdade.

    Retorna True se todo bloco ficou abaixo de `tol` em todos os passos.
    """
    pilota = _carrega_pilota()
    gravado = None
    if pasta is not None:
        gravado = np.load(Path(pasta).expanduser() / "cena.npz", allow_pickle=False)

    c = ATIVO.contrato
    fat = _fatias(c)
    canais = {n: ch for n, ch, _e, _h in c.blocos_ator}

    env = constroi_env()
    m: mujoco.MjModel = env.sim.mj_model
    dados = coleta_cena(env)
    if gravado is not None:
        _confere_npz(dados, gravado)
        dados = {k: gravado[k] for k in gravado.files if k in dados}

    nomes = [nome for nome, *_ in c.blocos_ator] + ["ctrl"]
    piores = {nome: 0.0 for nome in nomes}
    primeira_falha: dict[str, str] = {}

    env.reset()
    percepcao = pilota.Percepcao(m, dados)
    percepcao.reinicia(m, espelha_estado(env, m))
    acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    hist = None
    resets = 0
    limiar = ATIVO.percepcao.limiar_visto

    for passo in range(passos):
        env.step(acao)
        reiniciou = bool(_np(env.reset_buf)[0])
        d = espelha_estado(env, m)
        if reiniciou:
            hist = None  # o pilota preenche o histórico no 1º passo, como o mjlab faz no reset
            percepcao.reinicia(m, d)
            resets += 1

        obs_env = _np(env.observation_manager.compute()[GRUPO])[0].astype(np.float64)
        twist = _np(env.command_manager.get_command("twist"))[0].astype(np.float64)
        anterior = _np(env.action_manager.action)[0].astype(np.float64)
        tarefa = int(_np(env.v3_tarefa)[0])
        mapa_env = obs_env[fat["mapa"]]

        # Sem portão (decisão de 29/10): só imprime o quanto a Percepcao do pilota difere do mapa do env.
        dmax, concordancia = _compara_mapas(percepcao.passo(m, d), mapa_env, limiar)
        print(f"[mapa] passo {passo:3d}  Δmax altura (vistas pelos dois) {dmax:.3e} m  "
              f"concordância das vistas {concordancia:.4f}")

        nosso, hist = pilota.monta_observacao(
            d, dados, twist, tarefa, anterior,
            obs_env[fat["caixa"]][-canais["caixa"]:], mapa_env, hist)
        nosso = np.asarray(nosso, dtype=np.float64)
        if nosso.shape != (dim_ator(c),):
            raise AssertionError(f"monta_observacao devolveu forma {nosso.shape}, esperado ({dim_ator(c)},)")

        for nome in nomes[:-1]:
            sl = fat[nome]
            delta = float(np.abs(nosso[sl] - obs_env[sl]).max())
            if delta > piores[nome]:
                piores[nome] = delta
            if delta >= tol and nome not in primeira_falha:
                primeira_falha[nome] = f"passo {passo}"

        # ⚠ No passo de reset o `ctrl` não conta: o `joint_pos_target` zera no reset e o `d.ctrl` vai junto.
        if not reiniciou:
            ctrl_mjlab = _np(env.sim.data.ctrl)[0].astype(np.float64)
            ids = np.asarray(dados["ids_atuador"], dtype=np.int64)
            esperado = (np.asarray(dados["q_default_acao"], dtype=np.float64)
                        + np.asarray(dados["escala_acao"], dtype=np.float64) * anterior)
            erro = float(np.abs(ctrl_mjlab[ids] - esperado).max())
            piores["ctrl"] = max(piores["ctrl"], erro)
            if erro >= tol and "ctrl" not in primeira_falha:
                primeira_falha["ctrl"] = f"passo {passo}"

    tudo_bem = True
    print(f"\n[paridade] {passo + 1} passos  {resets} resets  tol {tol:.0e}")
    for nome in nomes:
        marca = "ok " if piores[nome] < tol else "FALHA"
        rotulo = nome if nome != "ctrl" else "ctrl (extra)"
        dim = (fat[nome].stop - fat[nome].start) if nome in fat else len(dados["ids_atuador"])
        falha = f"  primeira falha {primeira_falha[nome]}" if nome in primeira_falha else ""
        print(f"  {marca} {rotulo:<19s} dim {dim:>4d}  Δmax {piores[nome]:.3e}{falha}")
        if piores[nome] >= tol:
            tudo_bem = False

    print(f"\n[paridade] {'PASSA' if tudo_bem else 'FALHA'}")
    env.close()
    return tudo_bem


# -------------------------------------------------------------------- a linha de comando

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exporta a cena da v3 e prova a paridade da observação.")
    ap.add_argument("--saida", type=Path, help="pasta de saída: cena.mjb, cena.npz e contrato.json")
    ap.add_argument("--paridade", action="store_true", help="roda a paridade do ator contra o pilota_v3")
    ap.add_argument("--cena", type=Path, default=None, help="com --paridade: confere o cena.npz desta pasta")
    args = ap.parse_args(argv)

    if args.saida is not None:
        exporta(args.saida)
    if args.paridade:
        return 0 if paridade(args.cena) else 1
    if args.saida is None:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
