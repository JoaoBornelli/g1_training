"""PASSO 2, parte A — grava um rastro do caminho ORIGINAL do BFM-Zero.

Roda no ambiente do BFM-Zero, de dentro da pasta dele:

    cd ~/Documents/BFM-Zero
    ~/miniconda3/envs/bmf0/bin/python \
        ~/Documents/g1_training/g1_residual/dump_referencia.py

Por que dois processos: o caminho original precisa de `mujoco` + `scipy` e do XML
com as malhas, que moram no repo do BFM-Zero; o meu caminho precisa do `mjlab`, que
mora no venv do `g1_training`. Em vez de forçar os dois no mesmo ambiente, este
script grava um `.npz` e o `referencia.py` confere contra ele.

O rastro é o do `teste_sim.py`, que é o caminho que ANDA de verdade: a política roda
no `next_obs`, então o `_create_observation` roda duas vezes por passo de controle e
o histórico rola duas vezes. É esse comportamento que o `obs_bfm.py` copia.

Grava, por passo:
    qpos, qvel        estado cru do MuJoCo (para o meu lado reproduzir)
    state, last_action, history_actor, privileged_state    a obs que ELE montou
    acao              a ação que ELE produziu
"""
import pathlib
import sys

import numpy as np
import torch
from torch.utils._pytree import tree_map

AQUI = pathlib.Path(__file__).resolve().parent
BFM = pathlib.Path.home() / "Documents" / "BFM-Zero"
assert (BFM / "env.py").is_file(), f"rode de dentro de {BFM}"
sys.path.insert(0, str(BFM))

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel  # noqa: E402
from common import (ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS,  # noqa: E402
                    KD_GAINS, KP_GAINS)
from env import MuJoCoBFMZeroEnv  # noqa: E402

XML = "bfm_zero_inference_code/g1_for_reward_inference.xml"
PASSOS = 40
COMPORTAMENTO = "move-ego-0-0"
SAIDA = AQUI / "peso" / "referencia.npz"


def main() -> None:
    modelo = FBcprAuxModel.load("./model/checkpoint/model", device="cpu")

    import pickle
    # pickle do release do BFM-Zero, já executado pelo `teste_sim.py` deste repo
    z_tab = pickle.load(open("model/reward_inference/reward_locomotion.pkl", "rb"))
    z = torch.as_tensor(np.asarray(z_tab[COMPORTAMENTO][0])).reshape(1, -1).float()
    z = 16.0 * torch.nn.functional.normalize(z, dim=-1)
    print(f"z = {COMPORTAMENTO} semente 0 | norma {float(z.norm()):.2f}")

    env = MuJoCoBFMZeroEnv(
        robot_xml=XML, kp_gains=KP_GAINS, kd_gains=KD_GAINS,
        default_joint_pos=DEFAULT_JOINT_POS, action_scales=ACTION_SCALES,
        action_rescale=ACTION_RESCALE, enable_video=False)

    obs = env.reset()
    rastro: dict[str, list] = {k: [] for k in (
        "qpos", "qvel", "state", "last_action", "history_actor",
        "privileged_state", "acao")}

    for i in range(PASSOS):
        # a obs que a política de fato consome, e o estado cru que a gerou
        rastro["qpos"].append(env.mjd.qpos.copy())
        rastro["qvel"].append(env.mjd.qvel.copy())
        for k in ("state", "last_action", "history_actor", "privileged_state"):
            v = obs[k]
            rastro[k].append(np.asarray(v).reshape(-1).copy())

        entrada = tree_map(lambda x: torch.as_tensor(x).reshape(1, -1), obs)
        with torch.no_grad():
            acao = modelo.act(entrada, z, mean=True)
        acao = acao.reshape(-1).numpy()
        rastro["acao"].append(acao.copy())

        obs, prox, _, _ = env.step(acao)
        obs = prox

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAIDA, comportamento=COMPORTAMENTO, z=z.numpy(),
                        **{k: np.stack(v) for k, v in rastro.items()})
    print(f"\ngravado {SAIDA}")
    for k, v in rastro.items():
        a = np.stack(v)
        print(f"  {k:18s} {a.shape}")
    z_alt = np.stack(rastro['qpos'])[:, 2]
    print(f"\naltura da pelve: {z_alt[0]:.3f} -> {z_alt[-1]:.3f} m "
          f"(min {z_alt.min():.3f})")
    print("  o robô tem que continuar de pé no rastro; se caiu, o `z` ou os ganhos"
          " estão errados e o rastro não serve de referência.")


if __name__ == "__main__":
    main()
