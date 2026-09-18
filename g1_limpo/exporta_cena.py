"""Exporta a cena de treino para o MuJoCo clássico, e prova a paridade da observação.

    python -m g1_limpo.exporta_cena --saida ~/g1_pilota/
    python -m g1_limpo.exporta_cena --paridade
    python -m g1_limpo.exporta_cena --paridade --cena ~/g1_pilota/

Roda UMA vez, e precisa do mjlab. O consumidor é o `pilota.py`, na raiz do repo, que
não importa nada daqui.

⚠ NADA É DIGITADO À MÃO. Todo número sai do env construído por `make_env_cfg(play=True)`
— defaults de junta, escala de ação, mapa de atuador, endereços de sensor, decimação e
passo de física. Um número copiado à mão diverge no primeiro upgrade e ninguém percebe.

⚠ O `.mjb` NÃO VAI PARA O GIT (71 MB). O `.gitignore` já tem `*.mjb`. O gerador é que é
versionado.

⚠ A CONTA DA OBSERVAÇÃO NÃO MORA AQUI. Ela mora em `pilota.monta_observacao`, e este
arquivo a IMPORTA. Duas cópias divergem, e a paridade passaria a testar a si mesma.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

# ⚠ O `pilota` mora na RAIZ do repo, um nível acima deste pacote. Rodar por
# `python -m g1_limpo.exporta_cena` a partir da raiz já basta; o `sys.path` explícito
# cobre quem rodar de outro diretório.
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from pilota import BLOCOS, DIM_OBS, Cena, monta_observacao  # noqa: E402

__all__ = ["constroi_env", "coleta_cena", "exporta", "paridade"]

# O grupo que o robô vê. O `critic` tem 131 canais e não interessa aqui.
GRUPO = "actor"


def _np(x) -> np.ndarray:
    """Tira um `np.ndarray` de tensor torch, de `TorchArray` do mjlab ou de lista."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------- o env

def constroi_env(elo: int | None = None):
    """Constrói o env do mjlab: `play=True`, 1 env, CPU.

    ⚠ DUAS RANDOMIZAÇÕES DE TREINO SÃO DESLIGADAS AQUI, e as duas por necessidade:

    `encoder_bias` — evento de startup que soma até 0,015 rad ao `joint_pos` do ATOR
    (o termo do molde usa `biased=True`). O `play` não o remove. Não existe no robô
    real, e no MuJoCo clássico não há onde guardá-lo; deixá-lo ligado faria a paridade
    falhar em 29 canais por um motivo que não é erro de convenção.

    `tamanho_caixa` — evento de startup que sorteia a meia-aresta da caixa em
    0,07..0,13 e escreve o tamanho no MODELO POR MUNDO (`env.sim.model.geom_size`). O
    `mj_model` que vai para o `.mjb` é o TEMPLATE, e fica com o valor nominal do XML.
    Exportar um `meia_aresta` sorteado daria à política um canal que não descreve a
    caixa que ela vê. Aqui o buffer volta ao nominal, lido do próprio modelo.
    """
    from mjlab.envs import ManagerBasedRlEnv

    from g1_limpo.env_cfg import make_env_cfg

    cfg = make_env_cfg(play=True, elo=elo)
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    env = ManagerBasedRlEnv(cfg, device="cpu")

    robo = env.scene["robot"]
    robo.data.encoder_bias.zero_()

    caixa = env.scene["box"]
    gid = int(_np(caixa.indexing.geom_ids)[0])
    nominal = float(env.sim.mj_model.geom_size[gid, 0])
    if hasattr(env, "limpo_meia_aresta"):
        env.limpo_meia_aresta[:] = nominal
    return env, nominal


# ------------------------------------------------------------------- a coleta

def coleta_cena(env, meia_nominal: float) -> dict[str, np.ndarray]:
    """Lê do env tudo o que o `pilota` precisa. Todo campo tem uma fonte no env."""
    m: mujoco.MjModel = env.sim.mj_model
    cfg = env.cfg
    robo = env.scene["robot"]
    caixa = env.scene["box"]
    laje = env.scene["table"]

    # ---- as juntas, na ordem do MODELO (que é a ordem do termo `joint_pos`)
    nomes_juntas = list(robo.joint_names)
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
    # ⚠ O offset é `default_joint_pos[target_ids]` (`use_default_offset=True`), e não o
    # `q_default` da ordem de observação. Se as duas ordens divergirem um dia, o pilota
    # continua certo porque grava as duas.
    q_default_acao = _np(termo._offset)[0].astype(np.float64)

    ctrl_ids = _np(robo.indexing.ctrl_ids).astype(np.int64)
    ids_atuador = _mapa_atuador(m, ctrl_ids, _np(robo.indexing.joint_ids), alvo_ids)

    # ---- os sensores do IMU. É de onde `base_lin_vel` e `base_ang_vel` saem.
    adr_lin = int(m.sensor("robot/imu_lin_vel").adr[0])
    adr_ang = int(m.sensor("robot/imu_ang_vel").adr[0])
    assert int(m.sensor("robot/imu_lin_vel").dim[0]) == 3
    assert int(m.sensor("robot/imu_ang_vel").dim[0]) == 3

    # ---- os corpos
    id_base = int(robo.indexing.root_body_id)
    id_caixa = int(caixa.indexing.root_body_id)
    id_laje = int(laje.indexing.root_body_id)

    # ---- a escala POR CANAL dos termos do molde, lida do cfg
    escala_obs = _escala_por_canal(env)

    # ---- os knobs do gerador de alvo
    from g1_limpo.knobs import ATIVO
    k = ATIVO

    # ---- o envelope de velocidade do treino, lido do cfg no modo play
    faixas = cfg.commands["twist"].ranges
    envelope = np.array([max(abs(v) for v in faixas.lin_vel_x),
                         max(abs(v) for v in faixas.lin_vel_y),
                         max(abs(v) for v in faixas.ang_vel_z)], dtype=np.float64)

    dados = {
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
        "meia_aresta": np.float64(meia_nominal),
        "peito_b": np.asarray(k.alvo.peito_b, dtype=np.float64),
        "altura_carregar": np.float64(k.alvo.altura_carregar),
        "prateleira_xy": np.asarray(k.cena.prateleira_xy, dtype=np.float64),
        "prateleira_meia_z": np.float64(k.cena.prateleira_meia_z),
        "botar_recuo_borda": np.float64(k.alvo.botar_recuo_borda),
        "decimation": np.int64(cfg.decimation),
        "physics_dt": np.float64(cfg.sim.mujoco.timestep),
        "dim_obs": np.int64(DIM_OBS),
        "escala_obs": escala_obs,
        "envelope_treino": envelope,
    }
    return dados


def _mapa_atuador(m: mujoco.MjModel, ctrl_ids: np.ndarray, joint_ids: np.ndarray,
                  alvo_ids: np.ndarray) -> np.ndarray:
    """Junta -> índice em `d.ctrl`, na ordem da AÇÃO.

    ⚠ NÃO SUPÕE IDENTIDADE. O `BuiltinActuatorGroup` escreve
    `ctrl[ctrl_ids] = joint_pos_target[target_ids]`, e nada garante que a ordem dos
    atuadores no modelo siga a ordem das juntas. A derivação aqui é pelo `actuator_trnid`
    do modelo, e a paridade a confere contra o `d.ctrl` que o mjlab escreve de verdade.
    """
    del ctrl_ids
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
    """A `scale` de cada termo do ator, expandida canal a canal.

    ⚠ Não suponha 1,0. Hoje é 1,0 nos 114 (o único termo com `scale` era o
    `height_scan`, e a variante `flat` o apaga), mas ler do cfg é o que mantém isso
    verdadeiro amanhã.
    """
    cfg_grupo = env.cfg.observations[GRUPO]
    ativos = env.observation_manager.active_terms[GRUPO]
    esperado = [nome for nome, _ in BLOCOS]
    if ativos != esperado:
        raise AssertionError(
            f"a ordem dos termos do ator mudou.\n  mjlab:  {ativos}\n  pilota: {esperado}")
    pedacos = []
    for nome, n in BLOCOS:
        s = cfg_grupo.terms[nome].scale
        if s is None:
            pedacos.append(np.ones(n))
        else:
            pedacos.append(np.broadcast_to(_np(s).astype(np.float64).reshape(-1), (n,)))
    return np.concatenate(pedacos)


# ------------------------------------------------------------------- o estado vivo

def espelha_estado(env, m: mujoco.MjModel) -> mujoco.MjData:
    """Copia o estado do env do mjlab para um `MjData` clássico, e roda `mj_forward`.

    ⚠ `mj_forward`, e não `mj_step`. O que queremos é o mesmo estado com as grandezas
    derivadas frescas — que é exatamente o que o `mjlab` faz antes de calcular a
    observação (`manager_based_rl_env.py:454`).
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


# -------------------------------------------------------------------- a exportação

def exporta(pasta: Path, elo_da_cena: int = 2) -> None:
    """Grava `cena.mjb` e `cena.npz`.

    ⚠⚠ A CENA INICIAL VEM DO ELO `PEGAR`, e não do `ELO_DE_TREINO`. MEDIDO: com
    `elo=None` o `env_cfg` usa `ELO_DE_TREINO = ANDAR`, e o reset do ANDAR manda a laje
    — e a caixa em cima dela — para `afasta_z = 5,0 m` (`comando.py:1458`). O robô
    tropeçaria nela se ficasse no chão, e é correto no treino. Mas o `.mjb` exportado
    daí nasce com a caixa a 5,1 m de altura, e quatro dos cinco elos do pilota ficam
    sem objeto: o dono aperta `2` e o alvo está no céu.

    O reset do PEGAR põe a laje num topo alcançável e a caixa em cima. O ANDAR continua
    dirigível — quem manda no elo é o teclado, e o twist não depende da cena.
    """
    pasta = Path(pasta).expanduser()
    pasta.mkdir(parents=True, exist_ok=True)

    env, meia = constroi_env(elo=elo_da_cena)
    m: mujoco.MjModel = env.sim.mj_model
    dados = coleta_cena(env, meia)

    env.reset()
    d0 = espelha_estado(env, m)
    dados["qpos_inicial"] = d0.qpos.copy()
    dados["qvel_inicial"] = d0.qvel.copy()
    dados["mocap_pos_inicial"] = (d0.mocap_pos.copy() if m.nmocap
                                  else np.zeros((0, 3)))
    dados["mocap_quat_inicial"] = (d0.mocap_quat.copy() if m.nmocap
                                   else np.zeros((0, 4)))

    caminho_mjb = pasta / "cena.mjb"
    mujoco.mj_saveModel(m, str(caminho_mjb), None)
    np.savez(pasta / "cena.npz", **dados)

    tamanho = caminho_mjb.stat().st_size / (1024 * 1024)
    print(f"[exporta] {caminho_mjb}  {tamanho:.1f} MB")
    print(f"[exporta] {pasta / 'cena.npz'}  {len(dados)} campos")
    print(f"[exporta] nq={m.nq} nv={m.nv} nu={m.nu} nmocap={m.nmocap} "
          f"nsensordata={m.nsensordata}")
    print(f"[exporta] decimation={int(dados['decimation'])} "
          f"physics_dt={float(dados['physics_dt'])} "
          f"dt={float(dados['physics_dt']) * int(dados['decimation']):.3f} s")
    print(f"[exporta] meia_aresta nominal={meia:.4f} m  "
          f"envelope de treino={dados['envelope_treino']}")
    print(f"[exporta] ids_atuador identidade? "
          f"{bool((dados['ids_atuador'] == np.arange(len(dados['ids_atuador']))).all())}")
    print(f"[exporta] escala_obs toda 1,0? {bool((dados['escala_obs'] == 1.0).all())}")


# --------------------------------------------------------------------- a paridade

def paridade(pasta: Path | None, tol: float = 1e-5, passos: int = 80,
             teto_passos: int = 400) -> bool:
    """O PORTÃO. Compara os 114 canais do `pilota` com os do `observation_manager`.

    ⚠ RODA NOS CINCO ELOS, e não só no de treino. Com o elo publicado em ANDAR os dez
    canais da caixa são ZERO pelo gate — comparar zeros contra zeros não prova nada
    sobre o bloco que carrega o risco (frame da base, ordem dos três vetores, gate).

    ⚠⚠ E RODA 80 PASSOS NO MÍNIMO, não 5 como diz a spec §5. A janela de espera
    (`knobs.Alvo.espera_s`, 0,3 a 1,0 s) publica ANDAR no começo de TODO episódio.
    MEDIDO: com 5 passos (0,1 s) o elo publicado é 0 nos cinco envs, e a paridade dos
    dez canais da caixa compara zero contra zero. O teste compara em TODO passo, guarda
    o pior, e segue até juntar 20 amostras com o gate ABERTO (teto de 400 passos).
    Exige essa cobertura nos elos 1 a 4; no ANDAR o gate fechado é o certo.

    ⚠ O ALVO E O GIRO VÊM DO `command_manager`, e não do gerador da §4. O gerador é uma
    simplificação declarada da máquina de estados (sem jitter, sem cadeia, sem fecho) e
    não pode ser o que se testa. O que se testa é a CONTA da observação.

    ⚠ A POSIÇÃO DA CAIXA fica no default (`d.xpos[id_caixa]`), de propósito: assim o
    teste também prova o `id_caixa` exportado.

    Um décimo bloco, `ctrl`, confere o mapa junta -> atuador contra o `d.ctrl` que o
    mjlab escreveu. Ele não é da observação, mas é o outro lugar em que um índice errado
    derruba o robô no primeiro passo sem dizer por quê.
    """
    from g1_limpo.comando import ALVO, ANDAR, ELO, GIRO

    gravado = None
    if pasta is not None:
        gravado = np.load(Path(pasta).expanduser() / "cena.npz", allow_pickle=False)

    nomes = [nome for nome, _ in BLOCOS] + ["ctrl"]
    piores: dict[str, float] = {nome: 0.0 for nome in nomes}
    onde: dict[str, str] = {}
    tudo_bem = True

    for elo in range(5):
        env, meia = constroi_env(elo=elo)
        m: mujoco.MjModel = env.sim.mj_model
        dados = coleta_cena(env, meia)

        if gravado is not None:
            _confere_npz(dados, gravado)
            dados = {k: gravado[k] for k in gravado.files if k in dados}
        c = Cena(dados)

        env.reset()
        acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                           device=env.device)

        locais: dict[str, float] = {nome: 0.0 for nome in nomes}
        abertos = resets = 0
        for passo in range(teto_passos):
            env.step(acao)
            reiniciou = bool(_np(env.reset_buf)[0])
            resets += int(reiniciou)

            ref = _np(env.observation_manager.compute()[GRUPO])[0].astype(np.float64)
            cmd = _np(env.command_manager.get_command("alvo_caixa"))[0]
            twist = _np(env.command_manager.get_command("twist"))[0].astype(np.float64)
            anterior = _np(env.action_manager.action)[0].astype(np.float64)
            elo_pub = int(round(float(cmd[ELO])))
            if elo_pub != ANDAR:
                abertos += 1

            d = espelha_estado(env, m)
            nosso, _ = monta_observacao(
                m, d, c, twist, elo_pub, anterior,
                alvo_w=np.asarray(cmd[ALVO], dtype=np.float64),
                giro_w=np.asarray(cmd[GIRO], dtype=np.float64))

            i = 0
            for nome, n in BLOCOS:
                delta = float(np.abs(nosso[i:i + n] - ref[i:i + n]).max())
                if delta > locais[nome]:
                    locais[nome] = delta
                if delta >= tol and locais[nome] == delta:
                    pior = int(np.abs(nosso[i:i + n] - ref[i:i + n]).argmax())
                    print(f"  FALHA {nome} passo {passo} canal {pior}: "
                          f"pilota {nosso[i + pior]:+.6f}  mjlab {ref[i + pior]:+.6f}")
                i += n

            # ⚠ O PASSO DE RESET NÃO CONTA para o `ctrl`. O `EntityData.reset` zera
            # `joint_pos_target` (`entity/data.py:219`) e o `write_data_to_sim` do fim
            # do `step` leva esse zero ao `d.ctrl` — logo, no passo em que o episódio
            # reinicia, o `ctrl` do mjlab é ZERO enquanto a ação já voltou ao default.
            # É artefato do reset, não erro de mapa. O pilota não reinicia sozinho.
            if not reiniciou:
                ctrl_mjlab = _np(env.sim.data.ctrl)[0].astype(np.float64)
                ids = np.asarray(c.ids_atuador, dtype=np.int64)
                esperado = (np.asarray(c.q_default_acao, dtype=np.float64)
                            + np.asarray(c.escala_acao, dtype=np.float64) * anterior)
                erro = float(np.abs(ctrl_mjlab[ids] - esperado).max())
                if erro >= tol and erro > locais["ctrl"]:
                    pior = int(np.abs(ctrl_mjlab[ids] - esperado).argmax())
                    print(f"  FALHA ctrl passo {passo} junta {pior} "
                          f"({c.nomes_juntas[pior]}) -> atuador {ids[pior]}: "
                          f"mjlab {ctrl_mjlab[ids][pior]:+.6f}  "
                          f"pilota {esperado[pior]:+.6f}")
                locais["ctrl"] = max(locais["ctrl"], erro)

            # basta cobertura suficiente do gate aberto; passar de 400 passos só
            # gasta tempo de CPU.
            if passo + 1 >= passos and (elo == ANDAR or abertos >= 20):
                break

        print(f"\n[paridade] elo forçado={elo}  {passo + 1} passos  {resets} resets  "
              f"gate da caixa aberto em {abertos}")
        for nome in nomes:
            n = dict(BLOCOS).get(nome, len(np.asarray(c.ids_atuador)))
            marca = "ok " if locais[nome] < tol else "FALHA"
            rotulo = nome if nome != "ctrl" else "ctrl (extra)"
            print(f"  {marca} {rotulo:<19s} dim {n:>3d}  Δmax {locais[nome]:.3e}")
            if locais[nome] >= tol:
                tudo_bem = False
            if locais[nome] > piores[nome]:
                piores[nome] = locais[nome]
                onde[nome] = f"elo={elo}"
        if elo != ANDAR and abertos < 5:
            print(f"  FALHA cobertura: o gate da caixa abriu só {abertos} vezes no "
                  f"elo {elo}; os dez canais foram zero contra zero quase sempre.")
            tudo_bem = False

        env.close()

    print("\n[paridade] Δ MÁXIMO POR BLOCO, sobre os cinco elos:")
    for nome in nomes:
        print(f"  {nome:<19s} {piores[nome]:.3e}   ({onde.get(nome, '-')})")
    print(f"\n[paridade] {'PASSA' if tudo_bem else 'FALHA'}  (tolerância {tol:.0e})")
    return tudo_bem


def ensaio_a_seco(pasta: Path, checkpoint: Path) -> None:
    """Roda o pilota SEM viewer, direto dos artefatos: cena, alvo, rede e ctrl.

    ⚠ Não é paridade — é a prova de que o caminho todo FECHA antes de o dono abrir o
    viewer. Ele exercita o que a paridade não toca: `carrega_cena`, `restaura`,
    `alvo_do_elo` nos cinco elos, o `Ator` com o normalizador do checkpoint, e a
    escrita no `d.ctrl`. Um erro aqui viraria um traceback na cara de quem abriu o
    viewer, e o viewer é a máquina do dono.
    """
    from pilota import ELOS, Ator, alvo_do_elo, carrega_cena, restaura

    m, c = carrega_cena(pasta)
    d = mujoco.MjData(m)
    ator = Ator(checkpoint)
    print(f"\n[ensaio] {Path(checkpoint).name}  iter={ator.iteracao}  "
          f"entrada={ator.dim_entrada}  saída={ator.dim_saida}")
    if ator.dim_entrada != int(c.dim_obs):
        raise AssertionError(f"checkpoint de {ator.dim_entrada} canais contra cena de "
                             f"{int(c.dim_obs)}")

    restaura(m, d, c)
    acao = np.zeros(ator.dim_saida)
    ids = np.asarray(c.ids_atuador, dtype=np.int64)
    for elo in range(5):
        alvo = alvo_do_elo(m, d, c, elo)
        obs, _ = monta_observacao(m, d, c, np.zeros(3), elo, acao)
        a = ator(obs)
        d.ctrl[ids] = (np.asarray(c.q_default_acao, dtype=np.float64)
                       + np.asarray(c.escala_acao, dtype=np.float64) * a)
        print(f"  {ELOS[elo]:<11s} alvo_w [{alvo[0]:+.3f} {alvo[1]:+.3f} "
              f"{alvo[2]:+.3f}]  |obs| {np.abs(obs).max():.3f}  "
              f"ação [{a.min():+.3f} {a.max():+.3f}]  "
              f"ctrl [{d.ctrl[ids].min():+.3f} {d.ctrl[ids].max():+.3f}]")

    for _ in range(int(c.decimation)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    print(f"  um passo de controle rodou: pelve z = {d.xpos[int(c.id_base)][2]:.3f} m")


def _confere_npz(vivo: dict, gravado) -> None:
    """O `.npz` em disco tem de bater com o que o env vivo diz agora."""
    for chave, valor in vivo.items():
        if chave not in gravado.files:
            raise AssertionError(f"`cena.npz` não tem o campo `{chave}`")
        a, b = np.asarray(valor), np.asarray(gravado[chave])
        if a.dtype.kind in "US":
            igual = bool((a == b).all())
        else:
            igual = bool(np.allclose(a.astype(np.float64), b.astype(np.float64),
                                     rtol=0, atol=1e-9))
        if not igual:
            raise AssertionError(
                f"`cena.npz` está velho no campo `{chave}`: disco {b}, env {a}. "
                f"Reexporte com `--saida`.")


# ------------------------------------------------------------------------- entrada

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, help="pasta de destino de cena.mjb/.npz")
    ap.add_argument("--elo-da-cena", type=int, default=2,
                    help="o elo cujo reset monta a cena inicial (default 2, PEGAR: "
                         "caixa alcançável). Com 0, ANDAR, a caixa vai a +5 m.")
    ap.add_argument("--paridade", action="store_true",
                    help="compara os 114 canais contra o observation_manager")
    ap.add_argument("--cena", type=Path,
                    help="com --paridade: confere também o cena.npz já gravado")
    ap.add_argument("--checkpoint", type=Path,
                    help="com --cena: ensaia o pilota a seco, sem viewer")
    args = ap.parse_args()

    if not args.saida and not args.paridade:
        ap.error("escolha `--saida PASTA` ou `--paridade`")
    if args.checkpoint and not args.cena:
        ap.error("`--checkpoint` precisa de `--cena PASTA`")
    if args.saida:
        exporta(args.saida, elo_da_cena=args.elo_da_cena)
    if args.paridade:
        ok = paridade(args.cena)
        if args.checkpoint:
            ensaio_a_seco(args.cena, args.checkpoint)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
