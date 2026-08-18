"""Visualiza a cena e um checkpoint do g1_poc.

Dois modos:

    # 1. só a cena, sem política — a verificação da §18
    python -m g1_poc.play --geometria

    # 2. um checkpoint treinado
    python -m g1_poc.play --checkpoint CAMINHO/model_999.pt

O modo `--geometria` é o portão do passo 0 da §17. Ele custa minutos e o repositório
já perdeu um bloco por não fazê-lo. A lição de 16/07 é: **ataque a geometria, e não
o sintoma.**
"""
from __future__ import annotations

import argparse

import g1_poc  # noqa: F401  (registra a task)
from g1_poc.knobs import Knobs


def imprime_geometria() -> None:
    k = Knobs()
    kc, ka, kt = k.cena, k.alvo, k.tol
    repouso_alto = kc.prateleira_topo_teto + kc.caixa_meia_aresta[2]
    repouso_baixo = kc.prateleira_topo_piso + kc.caixa_meia_aresta[2]
    print("== a verificação da §18, em números ==")
    print(f"  1. topo da prateleira, faixa       : {kc.prateleira_topo_piso:.2f} a "
          f"{kc.prateleira_topo_teto:.2f} m")
    print(f"     fundo da laje no piso           : "
          f"{kc.prateleira_topo_piso - 2*kc.prateleira_meia_z:+.3f} m  "
          f"(0,000 = apoia no chão)")
    print(f"  2. a prateleira ocupa x de         : "
          f"{kc.prateleira_xy[0]-kc.prateleira_meia_xy:.2f} a "
          f"{kc.prateleira_xy[0]+kc.prateleira_meia_xy:.2f} m")
    print(f"     a caixa nasce em x              : {kc.caixa_xy[0]:.2f} m")
    print(f"  3. centro da caixa, prateleira alta: {repouso_alto:.2f} m")
    print(f"     centro da caixa, prateleira baixa: {repouso_baixo:.2f} m")
    print(f"  4. alvo do `pegar`, z              : {ka.pegar_z[0]:.2f} a "
          f"{ka.pegar_z[1]:.2f} m  (MUNDO)")
    print(f"     subida mínima da prateleira alta: "
          f"{(ka.pegar_z[0]-repouso_alto)*100:.0f} cm")
    print(f"     subida mínima da prateleira baixa: "
          f"{(ka.pegar_z[0]-repouso_baixo)*100:.0f} cm")
    print(f"     esfera de sucesso                : {kt.raio_sucesso*100:.0f} cm")
    print()
    print("  Confira no viewer, nesta ordem:")
    print("   a) o tampo NÃO cobre a caixa na prateleira baixa")
    print("   b) a pelve, o tronco e a coxa NÃO tocam o tampo no agachamento")
    print("   c) os pads tocam as FACES da caixa, e não as quinas")
    print("   d) subir o topo para 0,55 m com a caixa a 0,82 m não toca a caixa")
    print("      nem os antebraços (folga esperada: 0,17 m)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geometria", action="store_true",
                   help="imprime os números da §18 e abre a cena, sem política")
    p.add_argument("--checkpoint", type=str, default=None)
    args, resto = p.parse_known_args()

    if args.geometria:
        imprime_geometria()

    from mjlab.scripts.play import main as play_main

    play_main()


if __name__ == "__main__":
    main()
