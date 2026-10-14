"""Teste de queda da caixa na laje, na cena do treino (MuJoCo clássico, CPU).

Mede o que a métrica `impacto_da_caixa` do treino mede: o pico, por episódio, de
|F_apoio_z| / (m·g), com F lido UMA vez por passo de controle (a cada `decimation`
subpassos), como o sensor de contato do mjlab. Também mostra o pico por subpasso.
O robô fica longe, a 3 m atrás; a caixa cai de pé, sem giro e sem velocidade inicial.
"""
import mujoco
import numpy as np

DEC = 4
m = mujoco.MjModel.from_binary_path("/home/joaobornelli/g1_pilota/cena.mjb")
d = mujoco.MjData(m)
gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box/box")
mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table/mocap_base")])
adr = int(m.jnt_qposadr[int(m.body_jntadr[bb])]); vadr = int(m.jnt_dofadr[int(m.body_jntadr[bb])])
meia_laje = float(m.geom_size[gt][2])
f6 = np.zeros(6)


def forca_z():
    fz = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if {c.geom1, c.geom2} == {gb, gt}:
            mujoco.mj_contactForce(m, d, i, f6)
            fz += (c.frame.reshape(3, 3).T @ f6[:3])[2]
    return abs(fz)


def queda(massa, meia, h, topo=0.30, segundos=1.5):
    mujoco.mj_resetData(m, d)
    m.body_mass[bb] = massa
    m.body_inertia[bb] = massa * (2 * meia) ** 2 / 6.0
    m.geom_size[gb] = (meia, meia, meia)
    m.geom_rbound[gb] = meia * np.sqrt(3.0)
    m.geom_aabb[gb, 3:6] = (meia, meia, meia)
    mujoco.mj_setConst(m, d)
    d.qpos[0] = -3.0                                  # robô longe
    d.mocap_pos[mocap] = (0.50, 0.0, topo - meia_laje)
    d.qpos[adr:adr + 3] = (0.35, 0.0, topo + meia + h)
    d.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    mg = massa * 9.81
    pico_ctrl = pico_sub = 0.0
    n = int(segundos / m.opt.timestep)
    for k in range(n):
        mujoco.mj_step(m, d)
        f = forca_z()
        pico_sub = max(pico_sub, f / mg)
        if (k + 1) % DEC == 0:
            pico_ctrl = max(pico_ctrl, f / mg)
    repouso = forca_z() / mg
    return pico_ctrl, pico_sub, repouso


print("queda   massa  meia | pico por passo de controle | pico por subpasso | repouso no fim")
for h in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
    for massa in (1.0, 3.0, 5.0):
        for meia in (0.07, 0.13):
            pc, ps, rep = queda(massa, meia, h)
            print(f"{h*100:4.0f} cm {massa:4.0f} kg {meia:.2f} | {pc:26.1f} | {ps:17.1f} | {rep:6.2f}")
