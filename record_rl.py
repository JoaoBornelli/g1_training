"""Grava um rollout da politica RL (Lift) num .npz pro BFM IMITAR.

A ordem das 29 juntas do G1 do mjlab e IDENTICA a POLICY_JOINT_NAMES do BFM-Zero
(verificado), e nq/nv batem (36/35) -> a conversao e 1:1, sem remapeamento.
Ainda assim este script GRAVA os joint_names e valida a ordem, pra nao virar
"demo corrompida silenciosa".

Uso (mesmos knobs do teu play):
    .venv/bin/python record_rl.py ~/Downloads/model_14150.pt --shelf-top 0.0 --weight 5

Saida: rl_rollout.npz com qpos(T,36), qvel(T,35), dones(T), joint_names.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import g1_training  # noqa: F401  (registra as tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from play import _register_custom_lift

BFM_ORDER = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--shelf-top", type=float, default=0.0)
    p.add_argument("--weight", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=600, help="passos a gravar")
    p.add_argument("--out", default="rl_rollout.npz")
    a = p.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task = _register_custom_lift(a.shelf_top, rehearsal=False, weight=a.weight)

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, __import__("dataclasses").asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = env.unwrapped.scene["robot"]
    jn = list(robot.joint_names)
    print(f"[check] {len(jn)} juntas | ordem == BFM? {jn == BFM_ORDER}")
    if jn != BFM_ORDER:
        perm = [jn.index(n) for n in BFM_ORDER]     # reordena p/ a ordem do BFM
        print("[check] reordenando pra ordem do BFM")
    else:
        perm = list(range(29))

    obs = env.get_observations()          # devolve so o TensorDict (nao e tupla)
    QP, QV, DN = [], [], []
    print(f"gravando {a.steps} passos...")
    with torch.inference_mode():
        for t in range(a.steps):
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
            d = robot.data
            qp = np.concatenate([
                d.root_link_pos_w[0].cpu().numpy(),
                d.root_link_quat_w[0].cpu().numpy(),          # [w,x,y,z]
                d.joint_pos[0].cpu().numpy()[perm],
            ])
            qv = np.concatenate([
                d.root_link_lin_vel_w[0].cpu().numpy(),
                d.root_link_ang_vel_w[0].cpu().numpy(),
                d.joint_vel[0].cpu().numpy()[perm],
            ])
            QP.append(qp); QV.append(qv); DN.append(bool(dones[0].item()))

    QP, QV, DN = np.array(QP), np.array(QV), np.array(DN)
    np.savez(a.out, qpos=QP, qvel=QV, dones=DN, joint_names=np.array(BFM_ORDER))
    print(f"\nsalvo {a.out}: qpos{QP.shape} qvel{QV.shape} | resets: {DN.sum()}")
    print(f"pelve z: min {QP[:,2].min():.3f} max {QP[:,2].max():.3f} "
          f"(agachou {'SIM' if QP[:,2].min() < 0.6 else 'nao'})")
    seg = np.split(np.arange(len(DN)), np.where(DN)[0] + 1)
    seg = [s for s in seg if len(s) > 20]
    print(f"episodios completos gravados: {len(seg)} (maior = {max((len(s) for s in seg), default=0)} passos)")


if __name__ == "__main__":
    main()
