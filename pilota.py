"""Dirige um checkpoint do `g1_limpo` no MuJoCo clássico, com o teclado.

    python pilota.py --cena ~/g1_pilota/ --checkpoint ~/Downloads/model_9499.pt

⚠ ZERO IMPORT DE `g1_limpo`. Só `mujoco`, `torch` e `numpy`. A regra do módulo diz que
o `g1_limpo` não importa código do projeto; aqui o sentido é o inverso, e por isso o
arquivo fica na RAIZ. Assim ele roda num venv magro, sem mjlab e sem warp.

⚠ ISTO NÃO É UMA MÁQUINA DE ESTADOS. Não há espera, sustain, fecho de elo, cadeia,
currículo, recompensa nem terminação. Quem troca o elo é o teclado. Um alvo nasce por
elo, e mais nada.

A cena e as constantes vêm do `g1_limpo/exporta_cena.py`, que as LÊ do env do mjlab.
Nada aqui é digitado à mão.

⚠ A conta da observação mora AQUI, em `monta_observacao`, e o exportador a IMPORTA para
o teste de paridade. Uma implementação só. Duas cópias divergem, e a paridade passaria a
testar a si mesma.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

__all__ = ["Cena", "carrega_cena", "monta_observacao", "alvo_do_elo", "Ator",
           "ELOS", "BLOCOS", "DIM_OBS"]

# 0 ANDAR  1 REORIENTAR  2 PEGAR  3 CARREGAR  4 BOTAR — a mesma ordem do `comando.py`.
ELOS = ("ANDAR", "REORIENTAR", "PEGAR", "CARREGAR", "BOTAR")
ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3, 4

# Os nove blocos do ator, na ordem em que o `observation_manager` os concatena.
# ⚠ A ORDEM É O CONTRATO. Uma inserção no meio desloca todo peso da primeira camada em
# silêncio, e o robô sai andando de lado sem uma linha de erro.
BLOCOS = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("actions", 29),
    ("command", 3),
    ("elo", 5),
    ("caixa", 10),
)
DIM_OBS = sum(n for _, n in BLOCOS)     # 114


# --------------------------------------------------------------------------- cena

class Cena:
    """Os arrays do `cena.npz`, com nome. Só leitura."""

    def __init__(self, dados: dict):
        self._d = dados
        for chave, valor in dados.items():
            setattr(self, chave, valor)

    def __repr__(self) -> str:
        return f"<Cena {len(self._d)} campos, {int(self.dim_obs)} canais>"


def carrega_cena(pasta: str | Path) -> tuple[mujoco.MjModel, Cena]:
    """Lê `cena.mjb` e `cena.npz` da pasta que o `exporta_cena.py` gravou."""
    pasta = Path(pasta).expanduser()
    caminho_mjb = pasta / "cena.mjb"
    caminho_npz = pasta / "cena.npz"
    if not caminho_mjb.exists() or not caminho_npz.exists():
        raise FileNotFoundError(
            f"faltam `cena.mjb` e/ou `cena.npz` em {pasta}. "
            f"Rode `python -m g1_limpo.exporta_cena --saida {pasta}` primeiro.")
    modelo = mujoco.MjModel.from_binary_path(str(caminho_mjb))
    bruto = np.load(caminho_npz, allow_pickle=False)
    return modelo, Cena({k: bruto[k] for k in bruto.files})


# ------------------------------------------------------------------- quatérnions

def gira_inverso(quat: np.ndarray, vetor: np.ndarray) -> np.ndarray:
    """`R⁻¹ · v`, com o quatérnion em (w, x, y, z) — a convenção do MuJoCo.

    ⚠ A MESMA fórmula do `mjlab.utils.lab_api.math.quat_apply_inverse`. O sinal do
    termo do meio é o que separa a rotação da inversa dela.
    """
    xyz = quat[1:]
    t = 2.0 * np.cross(xyz, vetor)
    return vetor - quat[0] * t + np.cross(xyz, t)


def gira_yaw(quat: np.ndarray, vetor: np.ndarray) -> np.ndarray:
    """`R_yaw · v`: só a componente de guinada do quatérnion, como no `comando.py`."""
    w, z = float(quat[0]), float(quat[3])
    norma = np.hypot(w, z)
    if norma < 1e-9:
        return vetor.copy()
    q = np.array([w / norma, 0.0, 0.0, z / norma])
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, vetor)
    return vetor + q[0] * t + np.cross(xyz, t)


def gira(quat: np.ndarray, vetor: np.ndarray) -> np.ndarray:
    """`R · v`, quatérnion em (w, x, y, z)."""
    xyz = quat[1:]
    t = 2.0 * np.cross(xyz, vetor)
    return vetor + quat[0] * t + np.cross(xyz, t)


# ------------------------------------------------------------------ o alvo por elo

def alvo_do_elo(m: mujoco.MjModel, d: mujoco.MjData, c: Cena, elo: int) -> np.ndarray:
    """O alvo em MUNDO, um por elo. Sem estado, sem temporizador, sem fecho.

        ANDAR       irrelevante — os dez canais da caixa zeram no gate.
        REORIENTAR  a própria caixa: o pedido é de atitude, não de posição.
        PEGAR       x,y ancorados no peito do robô; z ABSOLUTO em `altura_carregar`.
        CARREGAR    igual ao PEGAR — a mesma âncora.
        BOTAR       sobre a laje, recuado para a borda perto do robô.

    ⚠ O z DO PEGAR É ABSOLUTO, e isto contradiz a tabela da spec §4. O
    `comando._alvo_ancorado_na_base` sobrescreve `a[:, 2] = altura_carregar` (1,02 m)
    depois de somar o `peito_b`. Com z relativo o robô satisfaria o alvo ANDANDO
    AGACHADO — o alvo desceria com a pelve e a caixa nunca subiria. A spec §2.1 pede o
    knob `altura_carregar` no `.npz`, portanto ela já conta com ele.

    ⚠ O BOTAR RECUA PARA A BORDA. O `comando.py` tira o alvo do centro do tampo por
    `botar_recuo_borda` (0,15 m) na direção do robô: no centro o robô alcançaria por
    cima de 20 cm de tampo, defeito datado de 16/07. E o z é `topo + meia_aresta` — o
    CENTRO da caixa pousada, não a superfície. O jitter aleatório do treino não entra:
    aqui o alvo é determinístico.
    """
    base_p = d.xpos[int(c.id_base)].copy()
    base_q = d.xquat[int(c.id_base)].copy()

    if elo == REORIENTAR:
        return d.xpos[int(c.id_caixa)].copy()

    if elo in (PEGAR, CARREGAR):
        a = base_p + gira(base_q, np.asarray(c.peito_b, dtype=np.float64))
        a[2] = float(c.altura_carregar)
        return a

    if elo == BOTAR:
        laje_p = d.xpos[int(c.id_laje)].copy()
        topo = float(laje_p[2]) + float(c.prateleira_meia_z)
        recuo = np.array([float(c.botar_recuo_borda), 0.0, 0.0])
        a = np.zeros(3)
        a[:2] = laje_p[:2] - gira_yaw(base_q, recuo)[:2]
        a[2] = topo + float(c.meia_aresta)
        return a

    # ANDAR: os canais zeram no gate, mas devolver a caixa mantém o valor finito.
    return d.xpos[int(c.id_caixa)].copy()


# ------------------------------------------------------------------- a observação

def monta_observacao(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    c: Cena,
    twist: np.ndarray,
    elo: int,
    acao_anterior: np.ndarray,
    *,
    caixa_w: np.ndarray | None = None,
    alvo_w: np.ndarray | None = None,
    giro_w: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Os 114 canais do ator, na ordem do `observation_manager`.

    Devolve `(obs, blocos)`. O dicionário `blocos` existe para o teste de paridade
    reportar o Δ POR BLOCO — dizer "falhou" sem dizer qual bloco não ajuda ninguém.

    ⚠ `d` PRECISA ESTAR FRESCO. Chame `mj_forward` depois do último `mj_step` e antes
    daqui: o `mj_step` deixa `xpos`, `xquat` e `sensordata` atrasados um subpasso, e o
    `mjlab` chama `sim.forward()` no fim do laço de decimação justamente para resolver
    isso (`manager_based_rl_env.py:454`).

    ⚠ `caixa_w`, `alvo_w` e `giro_w` são ENTRADAS, com default. O laço do pilota usa os
    defaults; a paridade passa os valores do `command_manager`, porque o gerador de
    alvo da §4 é uma simplificação deliberada da máquina de estados e não deve ser o que
    se testa. O que se testa aqui é a CONTA — frames, ordem e gate.

    ⚠ SEM RUÍDO. O `noise` dos termos do molde é domínio de treino; no `play` o próprio
    mjlab desliga a corrupção (`enable_corruption = False`).

    ⚠ SEM VIÉS DE ENCODER. O termo `joint_pos` do ator usa `biased=True`, e o viés é uma
    randomização de startup de até 0,015 rad. Não existe no robô real nem aqui.
    """
    ids_q = np.asarray(c.ids_junta_qpos, dtype=np.int64)
    ids_v = np.asarray(c.ids_junta_qvel, dtype=np.int64)
    a_lin, a_ang = int(c.adr_lin_vel), int(c.adr_ang_vel)

    base_q = d.xquat[int(c.id_base)]
    base_p = d.xpos[int(c.id_base)]

    if caixa_w is None:
        caixa_w = d.xpos[int(c.id_caixa)]
    if alvo_w is None:
        alvo_w = alvo_do_elo(m, d, c, elo)
    if giro_w is None:
        # ⚠ Nesta versão ninguém pede giro. O canal existe porque o contrato de layout
        # do treino o reserva; zerá-lo é o que o treino faz quando não há pedido.
        giro_w = np.zeros(3)

    # 1-2. O IMU. São sensores do MODELO (velocímetro e giroscópio no site
    # `imu_in_pelvis`), e não `root_link_lin_vel_b` — ler a velocidade do corpo daria
    # outro ponto de aplicação e outro frame.
    base_lin_vel = d.sensordata[a_lin:a_lin + 3].copy()
    base_ang_vel = d.sensordata[a_ang:a_ang + 3].copy()

    # 3. A gravidade no frame da base. O vetor de mundo é (0, 0, −1), normalizado.
    projected_gravity = gira_inverso(base_q, np.array([0.0, 0.0, -1.0]))

    # 4-5. As juntas, RELATIVAS ao default. O default de velocidade é zero.
    joint_pos = d.qpos[ids_q] - np.asarray(c.q_default, dtype=np.float64)
    joint_vel = d.qvel[ids_v] - np.asarray(c.qd_default, dtype=np.float64)

    # 6. A ação do passo ANTERIOR, crua — antes de escala e de offset.
    actions = np.asarray(acao_anterior, dtype=np.float64).copy()

    # 7. O twist pedido: vx, vy, wz, no frame da base.
    command = np.asarray(twist, dtype=np.float64).copy()

    # 8. O one-hot do elo.
    um_de_cinco = np.zeros(5)
    um_de_cinco[int(elo)] = 1.0

    # 9. Os dez canais da caixa, TODOS no frame da base, e GATEADOS.
    # ⚠⚠ O GATE: com o elo publicado em ANDAR os dez canais são ZERO, mesmo com a caixa
    # a 0,5 m. É o que substitui o bit VALIDA. Sem ele a política aprendia "ando" da
    # distância da caixa, e sambava em campo com a caixa perto.
    caixa_b = gira_inverso(base_q, np.asarray(caixa_w, dtype=np.float64) - base_p)
    alvo_b = gira_inverso(base_q, np.asarray(alvo_w, dtype=np.float64) - base_p)
    giro_b = gira_inverso(base_q, np.asarray(giro_w, dtype=np.float64))
    vivo = 0.0 if int(elo) == ANDAR else 1.0
    caixa = np.concatenate([caixa_b, alvo_b, giro_b,
                            [float(c.meia_aresta)]]) * vivo

    blocos = {
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "actions": actions,
        "command": command,
        "elo": um_de_cinco,
        "caixa": caixa,
    }
    obs = np.concatenate([blocos[nome] for nome, _ in BLOCOS])

    # ⚠ A escala dos termos do molde, LIDA DO CFG pelo exportador. Hoje ela é 1,0 em
    # todos os 114 canais (o único termo com `scale` era o `height_scan`, e a variante
    # `flat` o apaga). Multiplicar assim mesmo mantém o pilota correto se algum dia
    # deixar de ser.
    obs = obs * np.asarray(c.escala_obs, dtype=np.float64)
    if obs.shape[0] != DIM_OBS:
        raise AssertionError(f"a observação tem {obs.shape[0]} canais, não {DIM_OBS}")
    return obs, blocos


# ------------------------------------------------------------------------- o ator

class Ator:
    """O MLP do checkpoint, mais o normalizador dele. Determinístico, CPU, float32."""

    def __init__(self, caminho: str | Path):
        alvo = str(Path(caminho).expanduser())
        try:
            estado = torch.load(alvo, map_location="cpu", weights_only=True)
        except Exception:
            # ⚠ O fallback é para checkpoints com `infos` não-tensor. Só rode em
            # arquivo seu: `weights_only=False` desserializa objeto arbitrário.
            estado = torch.load(alvo, map_location="cpu", weights_only=False)
        sd = estado["actor_state_dict"] if "actor_state_dict" in estado else estado

        pesos = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
        if not pesos:
            raise KeyError("o checkpoint não tem pesos `mlp.*` no `actor_state_dict`")
        indices = sorted({int(k.split(".")[0]) for k in pesos})
        dims = [tuple(pesos[f"{i}.weight"].shape) for i in indices]

        # ⚠ ELU entre as camadas, e NADA depois da última. É a receita do
        # `unitree_g1_ppo_runner_cfg` (`activation="elu"`, hidden 512-256-128) e o
        # `GaussianDistribution` com `std_type="scalar"` devolve a média crua na
        # inferência determinística.
        camadas: list[torch.nn.Module] = []
        for k, i in enumerate(indices):
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

        # ⚠⚠ O NORMALIZADOR É `(x − mean) / (std + 1e-2)`, e NÃO
        # `(x − mean) / sqrt(var + 1e-8)` como diz a spec §3. O `rsl_rl` guarda `_std`
        # como buffer e o `EmpiricalNormalization.forward` soma `self.eps`, cujo default
        # é 1e-2 (`rsl_rl/modules/normalization.py:18,46`); o `MLPModel` o constrói sem
        # passar `eps`. Num canal de desvio pequeno — o one-hot do elo, por exemplo — a
        # diferença entre 1e-2 e 1e-8 no denominador é grande, e a rede receberia
        # entrada fora da escala em que treinou.
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

# Códigos GLFW, que é o que o `launch_passive` entrega ao `key_callback`.
_CIMA, _BAIXO, _ESQ, _DIR = 265, 264, 263, 262
_ESPACO, _VIRGULA, _PONTO = 32, 44, 46
_MENOS, _IGUAL = 45, 61
_ABRE, _FECHA = 91, 93
_ERRE = ord("R")
_DIGITOS = {ord(str(i)): i for i in range(5)}

# ⚠ O envelope do TREINO, e o teto de quem dirige, são coisas diferentes. O envelope
# vem do `.npz` (lido do `cfg.commands["twist"].ranges` no modo play): 2,0 / 1,0 / 0,7.
# Acima dele a política está fora da distribuição e o que se vê não diz nada.
_TETOS_PADRAO = (1.0, 0.5, 0.5)


class Piloto:
    """O estado que o teclado muda. Sem máquina de estados: cada tecla é um efeito."""

    def __init__(self, tetos: tuple[float, float, float],
                 envelope: np.ndarray):
        self.tetos = np.asarray(tetos, dtype=np.float64)
        self.envelope = np.asarray(envelope, dtype=np.float64)
        self.twist = np.zeros(3)
        self.elo = ANDAR
        self.fator = 1.0
        self.reset_pedido = False
        self.aviso = ""

    @property
    def passo(self) -> np.ndarray:
        """⚠ `teto / 5`, e não 0,1 fixo: cinco toques chegam ao teto seja qual for o
        teto. Com passo fixo, baixar o teto viraria uma rampa curta demais."""
        return self.tetos / 5.0

    def _recorta(self) -> None:
        """⚠ O teto recorta o valor CORRENTE, e não só os próximos toques. Baixar o
        teto com o robô a 1,0 m/s tem de derrubar o comando no mesmo passo."""
        self.twist = np.clip(self.twist, -self.tetos, self.tetos)

    def _escala_tetos(self, fator: float) -> None:
        novos = self.tetos * fator
        estourou = novos > self.envelope + 1e-9
        self.tetos = np.minimum(novos, self.envelope)
        if estourou.any():
            eixos = ", ".join(nome for nome, e in zip("vx vy wz".split(), estourou) if e)
            self.aviso = (f"teto saturado no envelope do treino ({eixos}): "
                          f"{self.envelope[0]:.2f} / {self.envelope[1]:.2f} / "
                          f"{self.envelope[2]:.2f}")
        else:
            self.aviso = ""
        self._recorta()

    def tecla(self, codigo: int) -> None:
        p = self.passo
        if codigo == _CIMA:
            self.twist[0] += p[0]
        elif codigo == _BAIXO:
            self.twist[0] -= p[0]
        elif codigo == _ESQ:
            self.twist[1] += p[1]
        elif codigo == _DIR:
            self.twist[1] -= p[1]
        elif codigo == _VIRGULA:
            self.twist[2] -= p[2]
        elif codigo == _PONTO:
            self.twist[2] += p[2]
        elif codigo == _ESPACO:
            self.twist[:] = 0.0
        elif codigo == _MENOS:
            self._escala_tetos(0.8)
        elif codigo == _IGUAL:
            self._escala_tetos(1.25)
        elif codigo in _DIGITOS:
            self.elo = _DIGITOS[codigo]
        elif codigo == _ERRE:
            self.reset_pedido = True
        elif codigo == _ABRE:
            self.fator = max(self.fator / 2.0, 1.0 / 64.0)
        elif codigo == _FECHA:
            self.fator = min(self.fator * 2.0, 8.0)
        else:
            return
        self._recorta()


# -------------------------------------------------------------------------- o laço

def restaura(m: mujoco.MjModel, d: mujoco.MjData, c: Cena) -> None:
    """Devolve a cena ao estado do reset do treino.

    ⚠ NÃO É `m.qpos0`, e isto contradiz a spec §6. O `qpos0` traz a pose crua do XML e
    não sabe onde o evento `posiciona_cena` põe a caixa nem a laje mocap. O exportador
    grava o qpos, o qvel e a pose de mocap LIDOS do env logo depois do `reset()`, que é
    a cena que o treino de fato vê.
    """
    mujoco.mj_resetData(m, d)
    d.qpos[:] = np.asarray(c.qpos_inicial, dtype=np.float64)
    d.qvel[:] = np.asarray(c.qvel_inicial, dtype=np.float64)
    if m.nmocap:
        d.mocap_pos[:] = np.asarray(c.mocap_pos_inicial, dtype=np.float64)
        d.mocap_quat[:] = np.asarray(c.mocap_quat_inicial, dtype=np.float64)
    ids_atuador = np.asarray(c.ids_atuador, dtype=np.int64)
    d.ctrl[ids_atuador] = np.asarray(c.q_default_acao, dtype=np.float64)
    mujoco.mj_forward(m, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb e cena.npz")
    ap.add_argument("--checkpoint", required=True, help="o model_*.pt")
    ap.add_argument("--vx-max", type=float, default=_TETOS_PADRAO[0])
    ap.add_argument("--vy-max", type=float, default=_TETOS_PADRAO[1])
    ap.add_argument("--wz-max", type=float, default=_TETOS_PADRAO[2])
    args = ap.parse_args()

    m, c = carrega_cena(args.cena)
    d = mujoco.MjData(m)
    ator = Ator(args.checkpoint)
    if ator.dim_entrada != int(c.dim_obs):
        raise SystemExit(
            f"o checkpoint espera {ator.dim_entrada} canais e a cena monta "
            f"{int(c.dim_obs)}. Checkpoint de outra fase?")

    envelope = np.asarray(c.envelope_treino, dtype=np.float64)
    tetos = np.minimum([args.vx_max, args.vy_max, args.wz_max], envelope)
    piloto = Piloto(tuple(tetos), envelope)

    ids_atuador = np.asarray(c.ids_atuador, dtype=np.int64)
    q_default_acao = np.asarray(c.q_default_acao, dtype=np.float64)
    escala_acao = np.asarray(c.escala_acao, dtype=np.float64)
    decimation = int(c.decimation)
    physics_dt = float(c.physics_dt)
    dt = physics_dt * decimation

    restaura(m, d, c)
    acao = np.zeros(ator.dim_saida)

    print(f"[pilota] cena {args.cena}  checkpoint iter={ator.iteracao}  "
          f"dt={dt * 1000:.0f} ms  decimation={decimation}")
    print("[pilota] setas: vx/vy   , .: wz   espaço: zera   - =: tetos   "
          "0-4: elo   r: reset   [ ]: tempo")

    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d, key_callback=piloto.tecla) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            if piloto.reset_pedido:
                restaura(m, d, c)
                acao[:] = 0.0
                piloto.twist[:] = 0.0
                piloto.reset_pedido = False

            obs, _ = monta_observacao(m, d, c, piloto.twist, piloto.elo, acao)
            acao = ator(obs)

            # ⚠ SEM CLAMP NOSSO. Os atuadores são servos de posição com `ctrlrange` no
            # modelo, e o MuJoCo já satura. Um clamp a mais mudaria o comportamento em
            # relação ao treino.
            d.ctrl[ids_atuador] = q_default_acao + escala_acao * acao

            for _ in range(decimation):
                mujoco.mj_step(m, d)
            # ⚠ O `mj_forward` NÃO ESTÁ NA SPEC §7, e ele é obrigatório. O `mj_step`
            # deixa `xpos`, `xquat` e `sensordata` atrasados um subpasso; o `mjlab`
            # chama `sim.forward()` depois do laço de decimação e antes de calcular a
            # observação. Sem isto o pilota veria a velocidade de 5 ms atrás.
            mujoco.mj_forward(m, d)

            viewer.sync()

            medido = np.array([d.sensordata[int(c.adr_lin_vel)],
                               d.sensordata[int(c.adr_lin_vel) + 1],
                               d.sensordata[int(c.adr_ang_vel) + 2]])
            print(f"\r{ELOS[piloto.elo]:<11s} "
                  f"pedido {piloto.twist[0]:+.2f} {piloto.twist[1]:+.2f} "
                  f"{piloto.twist[2]:+.2f} | "
                  f"medido {medido[0]:+.2f} {medido[1]:+.2f} {medido[2]:+.2f} | "
                  f"tetos {piloto.tetos[0]:.2f}/{piloto.tetos[1]:.2f}/"
                  f"{piloto.tetos[2]:.2f} | tempo x{piloto.fator:g} "
                  f"{piloto.aviso:<60s}", end="", flush=True)

            atraso = dt / piloto.fator - (time.perf_counter() - t0)
            if atraso > 0:
                time.sleep(atraso)
    print()


if __name__ == "__main__":
    main()
