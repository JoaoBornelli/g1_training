"""Prova que a transcrição do g1_limpo bate com as referências.

    python -m g1_limpo.paridade

⚠ ESTE É O ÚNICO ARQUIVO DO PACOTE QUE IMPORTA CÓDIGO DO PROJETO. Ele é
DESCARTÁVEL: não roda em treino, não é importado por nenhum outro módulo, e o
`g1_limpo` funciona sem ele. Se as referências forem apagadas, apague este arquivo.

POR QUE ELE EXISTE. Sob a restrição de zero import, cada número da cena é
transcrito à mão, e **um número errado não levanta erro** — ele muda o
comportamento em silêncio. Um σ de joelho de 0,35 digitado 0,035 achata o passo e a
run morre 1200 iterações depois, num painel.

POR QUE COMPARAR `mjModel` E NÃO `cfg`. Massa, atrito, meia-aresta, altura da laje,
grupo de geom e a posição dos pads vivem dentro de lambdas de `spec_fn`. Comparação
de cfg NÃO PENETRA lambda. Só o modelo compilado vê esses números.

LIMITE ESTRUTURAL, declarado: paridade contra o FABRICANTE não pega erro na metade de
manipulação, porque o fabricante não tem caixa nem mesa. O referencial da
manipulação é o `g1_training/` e o `g1_poc/`, e é o que este arquivo usa.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# --- as referências. NENHUM outro arquivo do pacote pode fazer isto. ---
from g1_training.common.box import get_box_spec, get_shelf_spec
from g1_training.common.robot import add_palm_pads
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from g1_limpo import cena as C
from g1_limpo.knobs import Knobs

_ok = 0
_dif: list[str] = []


def cmp(nome: str, nosso, deles, tol: float = 0.0) -> None:
    global _ok
    a, b = np.asarray(nosso), np.asarray(deles)
    if a.shape != b.shape:
        _dif.append(f"{nome}: SHAPE {a.shape} != {b.shape}")
        return
    if a.dtype.kind in "iub" or tol == 0.0:
        igual = bool(np.array_equal(a, b))
    else:
        igual = bool(np.allclose(a, b, atol=tol, rtol=0.0))
    if igual:
        _ok += 1
    else:
        d = np.abs(a.astype(float) - b.astype(float))
        pior = int(np.argmax(d)) if d.size else -1
        _dif.append(
            f"{nome}: difere em {int((d > tol).sum())} de {d.size} entradas; "
            f"pior no índice {pior}: nosso={a.flat[pior]} deles={b.flat[pior]}")


def modelo(spec):
    return spec.compile()


def campos_de_geom(m) -> dict:
    return {
        "ngeom": m.ngeom,
        "nq": m.nq,
        "nsite": m.nsite,
        "geom_size": m.geom_size,
        "geom_pos": m.geom_pos,
        "geom_friction": m.geom_friction,
        "geom_condim": m.geom_condim,
        "geom_group": m.geom_group,
        "geom_type": m.geom_type,
        "body_mass": m.body_mass,
        "site_pos": m.site_pos,
    }


def compara_modelos(rotulo: str, nosso_spec, deles_spec) -> None:
    a, b = campos_de_geom(modelo(nosso_spec)), campos_de_geom(modelo(deles_spec))
    for campo in a:
        cmp(f"{rotulo}.{campo}", a[campo], b[campo], tol=1e-12)


# =============================================================================
k = Knobs()
c = k.cena

print("=" * 78)
print("PARIDADE — g1_limpo contra g1_training / g1_poc")
print("=" * 78)

# --------------------------------------------------------------- 1. a caixa
print("\n1. a caixa (corpo livre) — com a DIVERGÊNCIA DELIBERADA do marcador")
#
# ⚠ A nossa caixa tem UM GEOM A MAIS que a referência: a placa visual da face alvo,
# sem a qual a inspeção do `reorientar` seria cega (um cubo uniforme girado 90° é
# visualmente idêntico ao original).
#
# Portanto a comparação NÃO é geom a geom. Ela é:
#   (a) a FÍSICA do corpo — massa e inércia — tem de ser BIT-IDÊNTICA
#   (b) o geom de COLISÃO (índice 0) tem de ser idêntico ao da referência
#   (c) o marcador tem de ser PROVADAMENTE inerte: contype = conaffinity = 0
#
# É assim que uma divergência deliberada se declara: delimitada, e com o limite
# afirmado por teste.
_regroup = (lambda s, g: ([setattr(x, "group", g) for x in s.geoms], s)[1])
m_nossa = modelo(C.regroup(C.spec_caixa(c), c.grupo_mobilia))
m_ref = modelo(_regroup(get_box_spec(c.caixa_meia_aresta, c.caixa_massa),
                        c.grupo_mobilia))

# (a) a física
cmp("caixa.body_mass", m_nossa.body_mass, m_ref.body_mass)
cmp("caixa.body_inertia", m_nossa.body_inertia, m_ref.body_inertia)
cmp("caixa.nq", [m_nossa.nq], [m_ref.nq])

# (b) o geom de colisão
for campo in ("geom_size", "geom_pos", "geom_friction", "geom_condim",
              "geom_group", "geom_type"):
    a_ = getattr(m_nossa, campo)
    b_ = getattr(m_ref, campo)
    cmp(f"caixa.colisao.{campo}", a_[0], b_[0], tol=1e-12)

# (c) o marcador é inerte
cmp("caixa.marcador.existe", [m_nossa.ngeom], [2])
cmp("caixa.marcador.contype_zero", [int(m_nossa.geom_contype[1])], [0])
cmp("caixa.marcador.conaffinity_zero", [int(m_nossa.geom_conaffinity[1])], [0])
cmp("caixa.referencia.tem_um_geom_so", [m_ref.ngeom], [1])

# ---------------------------------------------------------- 2. a prateleira
print("2. a prateleira (fixa -> mocap)")
meia_ref = (c.prateleira_meia_xy, c.prateleira_meia_xy, c.prateleira_meia_z)
compara_modelos(
    "prateleira",
    C.regroup(C.spec_prateleira(c), c.grupo_mobilia),
    (lambda s, g: ([setattr(x, "group", g) for x in s.geoms], s)[1])(
        get_shelf_spec(meia_ref), c.grupo_mobilia),
)

# ------------------------------------------------------- 3. o robô com pads
# ⚠ ESTE é o item de maior risco de transcrição. A referência tem os comentários
# dizendo "palma: -Z local" para as DUAS mãos, mas o CÓDIGO usa sinais opostos.
# Se eu tivesse copiado o comentário, `geom_pos` denunciaria aqui.
print("3. o robô com os pads de palma e de dorso")
compara_modelos("robot", C.add_pads_de_palma(get_spec()), add_palm_pads(get_spec()))

# nomes dos geoms, na mesma ordem
m_nosso = modelo(C.add_pads_de_palma(get_spec()))
m_deles = modelo(add_palm_pads(get_spec()))
import mujoco  # noqa: E402


def nomes_de_geom(m) -> list[str]:
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]


cmp("robot.nomes_de_geom", nomes_de_geom(m_nosso), nomes_de_geom(m_deles))

# as cápsulas originais foram removidas nos dois
for capsula in ("left_hand_collision", "right_hand_collision"):
    nossos = nomes_de_geom(m_nosso)
    cmp(f"robot.sem_{capsula}", [capsula in nossos], [capsula in nomes_de_geom(m_deles)])

# ------------------------------------------------------------ 4. constantes
print("4. constantes de nome")
cmp("BOX_GEOM", [C.BOX_GEOM], ["box_geom"])
cmp("TABLE_GEOM", [C.TABLE_GEOM], ["table_geom"])
cmp("PALM_SITES", list(C.PALM_SITES), ["left_palm", "right_palm"])
cmp("PALM_PAD_GEOMS", list(C.PALM_PAD_GEOMS), ["left_palm_pad", "right_palm_pad"])
cmp("BACK_PAD_GEOMS", list(C.BACK_PAD_GEOMS),
    ["left_hand_back_pad", "right_hand_back_pad"])

# ------------------------------------------------------------- 5. sensores
# Comparação de CONTRATO contra o `g1_poc`, que é a referência da manipulação: os
# mesmos padrões, os mesmos `fields`, a mesma redução.
print("5. contrato dos sensores (contra o g1_poc)")
try:
    from g1_poc import cena as PC

    nossos = {s.name: s for s in C.sensores()}
    deles = {s.name: s for s in PC.sensores()}
    # ⚠⚠ DIVERGÊNCIA DECLARADA, e ela é a partição de 31/08. O `g1_poc` tem UM sensor
    # de mesa (`corpo_prateleira`); nós temos TRÊS, um por grupo de geom. O motivo é
    # `reduce="netforce"`: cada sensor entrega UM número para todos os seus geoms,
    # portanto um sensor único diz "encostou" e não diz COM O QUÊ. No bloco 4 o contato
    # ilegal era ~46% dos episódios de manipulação e o log não distinguia tronco de
    # coxa de pad da palma — e sem isso o conserto seguinte seria chute.
    #
    # A partição é de MEDIÇÃO: a união dos grupos é a mesma, o limiar é o mesmo, e o
    # `smoke` afirma as duas coisas. O que muda é o log.
    #
    # ⚠ Declarar aqui NÃO é silenciar: os dois extras são NOMEADOS, e um terceiro nome
    # novo aparecendo faz o check falhar. A comparação de contrato dos sensores que os
    # dois módulos têm em comum continua byte a byte, abaixo.
    NOSSOS_A_MAIS = {C.SENSOR_PALMA_PRATELEIRA, C.SENSOR_DORSO_PRATELEIRA}
    cmp("sensores.nossos_a_mais", sorted(set(nossos) - set(deles)),
        sorted(NOSSOS_A_MAIS))
    cmp("sensores.nenhum_do_poc_falta", sorted(set(deles) - set(nossos)), [])
    cmp("sensores.quantidade", [len(nossos)], [len(deles) + len(NOSSOS_A_MAIS)])
    for nome in sorted(set(nossos) & set(deles)):
        a, b = nossos[nome], deles[nome]
        cmp(f"sensores.{nome}.fields", list(a.fields), list(b.fields))
        cmp(f"sensores.{nome}.reduce", [a.reduce], [b.reduce])
        cmp(f"sensores.{nome}.num_slots", [a.num_slots], [b.num_slots])
        cmp(f"sensores.{nome}.track_air_time",
            [bool(getattr(a, "track_air_time", False))],
            [bool(getattr(b, "track_air_time", False))])
        cmp(f"sensores.{nome}.history_length",
            [getattr(a, "history_length", None) or 0],
            [getattr(b, "history_length", None) or 0])
except Exception as e:      # noqa: BLE001
    _dif.append(f"sensores: não foi possível comparar contra o g1_poc ({e})")

# =============================================================================
print()
print("=" * 78)
if _dif:
    print(f"{_ok} campos idênticos / {len(_dif)} DIFERENÇAS")
    for d in _dif:
        print(f"  ✗ {d}")
    sys.exit(1)
print(f"{_ok} campos idênticos / 0 diferenças")
print("A transcrição bate com as referências.")
