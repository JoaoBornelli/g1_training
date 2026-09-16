"""Abre no viewer a pose de referência que o `gera_botar` resolveu.

    python -m g1_limpo.ik.ve_botar --cena ~/g1_pilota/ --ref ref_botar.npz
    python -m g1_limpo.ik.ve_botar --cena ~/g1_pilota/ --ref ref_botar.npz --i 7

⚠ É O PORTÃO DA SEÇÃO 3, e ele é obrigatório. Resíduo zero prova que a MÃO chegou ao
alvo. Ele NÃO prova que a pose se parece com alguém pousando uma caixa, nem que ela é
executável sob carga. Erro de IK não levanta exceção.

Teclas do viewer: SETA para a pose seguinte, e a janela mostra as três grandezas.
⚠ A pose é PINADA a cada quadro com `mj_forward`. Nada de física: sem o pino a pose
cai em ~0,4 s, e o que você veria seria a queda, e não a referência.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ⚠ A RAIZ DO REPO NO `sys.path`, e é o mesmo idioma do `play.py`. Sem isto o arquivo
# só roda como `python -m g1_limpo.ik.ve_botar`, e `python g1_limpo/ik/ve_botar.py`
# falha com `No module named 'g1_limpo'`. As duas formas passam a funcionar.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco                                                        # noqa: E402
import mujoco.viewer                                                 # noqa: E402
import numpy as np                                                   # noqa: E402

from g1_limpo.ik.gera_botar import Cena                              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True)
    ap.add_argument("--ref", required=True, help="o .npz do `gera_botar`")
    ap.add_argument("--i", type=int, default=0, help="índice da pose a abrir")
    ap.add_argument("--segundos", type=float, default=6.0,
                    help="quanto tempo cada pose fica na tela; 0 = só a do --i")
    args = ap.parse_args()

    ref = np.load(args.ref)
    c = Cena(Path(args.cena).expanduser())
    if ref["hash_mjcf"] != c.hash:
        raise SystemExit(
            f"⚠ MJCF DIFERENTE. A referência nasceu do hash {ref['hash_mjcf']} e esta "
            f"cena é {c.hash}. Comparar as duas mede um robô contra outro.")

    n = len(ref["topo"])
    ordem = [args.i] if args.segundos <= 0 else list(range(args.i, n))
    print(f"{n} poses.  hash {c.hash}")
    with mujoco.viewer.launch_passive(c.m, c.d) as v:
        for k in ordem:
            topo, a = float(ref["topo"][k]), float(ref["meia"][k])
            c.cenario(topo, a)             # laje no topo pedido, caixa no tamanho
            c.d.qpos[:] = ref["qpos"][k]
            print(f"[{k:2d}/{n}] laje {topo:.2f}  meia {a:.2f}  "
                  f"pelve {ref['pelve_z'][k]:.3f}  tronco {ref['tronco_incl'][k]:5.1f}°  "
                  f"com {ref['com_offset'][k]:.3f}  folga {ref['folga_min'][k] * 1000:.1f} mm  "
                  f"{'ok' if ref['aceita'][k] else 'REJEITADA'}")
            t0 = time.time()
            while v.is_running() and (args.segundos <= 0
                                      or time.time() - t0 < args.segundos):
                # ⚠ O PINO. `mj_forward` recalcula a cinemática sem integrar o tempo.
                mujoco.mj_forward(c.m, c.d)
                v.sync()
                time.sleep(0.02)
            if not v.is_running():
                break


if __name__ == "__main__":
    main()
