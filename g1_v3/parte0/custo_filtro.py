"""Custo do filtro de elevação em CPU: prediz + atualiza + recorte, 4 096 envs e 205 pontos (plano C1).

Referência: 26 ms (viabilidade D7). Rodar da raiz do worktree:
    python -m g1_v3.parte0.custo_filtro
"""
import time

import torch

from g1_v3 import percepcao_lidar
from g1_v3.knobs import Percepcao

E, R, TRECHOS, AQUECE = 4096, 205, 30, 3


def main():
    torch.manual_seed(0)
    k = Percepcao()
    f = percepcao_lidar.FiltroDeElevacao(E, "cpu", k)
    pontos = (torch.rand(E, R, 3) - 0.5) * 4.0  # pontos no frame do mapa, dentro dos 5 m
    pontos[..., 2] = torch.randn(E, R) * 0.02  # alturas perto do chão
    validos = torch.rand(E, R) > k.perda
    dist = pontos.norm(dim=-1)
    dxy = torch.rand(E, 2) * 0.04  # 2 cm por passo de 20 ms, a 1 m/s
    dpsi = torch.randn(E) * 0.02
    z_ref = torch.zeros(E)
    tempos = []
    for passo in range(TRECHOS + AQUECE):
        t0 = time.perf_counter()
        f.prediz(dxy, dpsi, z_ref, 0.02)
        f.atualiza(pontos, validos, dist)
        f.recorte()
        t1 = time.perf_counter()
        if passo >= AQUECE:
            tempos.append((t1 - t0) * 1e3)
    tempos.sort()
    med = tempos[len(tempos) // 2]
    print(f"envs={E} pontos={R} CPU: prediz+atualiza+recorte, mediana de {TRECHOS} passos = {med:.1f} ms "
          f"(mín {tempos[0]:.1f}, máx {tempos[-1]:.1f}); referência 26 ms")


if __name__ == "__main__":
    main()
