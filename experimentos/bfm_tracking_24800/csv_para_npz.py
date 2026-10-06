"""CSV do `registra_juntas.py` -> `.npz` no formato do `imitate_rl.py` do BFM-Zero.

Teste descartável: o BFM-Zero, por `tracking_inference`, segue o rollout do 24800?

    python experimentos/bfm_tracking_24800/csv_para_npz.py 24800_mu10_m1.csv
    cd ~/Documents/BFM-Zero
    ROLLOUT=~/Documents/g1_training/experimentos/bfm_tracking_24800/24800_mu10_m1.npz \
        python imitate_rl.py            # --dry só mede, sem viewer

Saída: `qpos` (T, 36) = raiz xyz + quat wxyz + 29 juntas; `qvel` (T, 35) = vel. linear
da raiz e das juntas por diferença finita. A vel. angular da raiz fica em ZERO: o
`imitate_rl.py` só lê `qvel[:, 6:35]` e o `set_state` dele zera a da raiz.

A ordem das 29 colunas `q_*` do CSV é checada contra a `POLICY_JOINT_NAMES` do BFM.
"""
import csv
import sys
from pathlib import Path

import numpy as np

POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
RAIZ = ["raiz_x", "raiz_y", "raiz_z", "raiz_qw", "raiz_qx", "raiz_qy", "raiz_qz"]

entrada = Path(sys.argv[1])
linhas = list(csv.DictReader(entrada.open()))
qpos = np.array([[float(r[k]) for k in RAIZ] + [float(r[f"q_{j}"]) for j in POLICY_JOINT_NAMES]
                 for r in linhas])
dt = float(linhas[1]["t"]) - float(linhas[0]["t"])

qvel = np.zeros((len(qpos), 35))
qvel[:, 0:3] = np.gradient(qpos[:, 0:3], dt, axis=0)
qvel[:, 6:35] = np.gradient(qpos[:, 7:36], dt, axis=0)

saida = Path(__file__).parent / f"{entrada.stem}.npz"
np.savez(saida, qpos=qpos, qvel=qvel, dones=np.zeros(len(qpos), dtype=bool),
         joint_names=np.array(POLICY_JOINT_NAMES), elo=np.array([r["elo"] for r in linhas]))
print(f"{saida}: {len(qpos)} passos, dt {dt:g} s, pelve {qpos[:, 2].min():.3f}..{qpos[:, 2].max():.3f} m")
