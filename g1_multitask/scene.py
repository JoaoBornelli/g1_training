"""Consertos de cena que o isolamento obriga a fazer daqui.

O problema (item 7) é real e vem do fabricante mais o nosso `common/box.py`:

- `common/box.py::_free_box` e `get_shelf_spec` chamam `add_geom` **sem** `group=`,
  então a caixa e a prateleira caem no **grupo 0**, junto com o chão de verdade;
- o `foot_height_scan` do G1 vem com `include_geom_groups=(0,)` (default testado do
  fabricante) — logo ele lê **a prateleira como chão**;
- e o `foot_height_scan` SOBREVIVE no cfg flat: o `unitree_g1_flat_env_cfg` remove
  só o `terrain_scan`.

Enquanto o multi-tarefa não tinha marcha isso era inofensivo, porque o `base_env`
apaga os rewards que usam esse sensor. Com a locomoção de volta (Tarefa 6), o
`feet_clearance` e o `foot_swing_height` passam a ler a prateleira como piso.

O conserto óbvio seria passar `group=2` no `add_geom`. Mas isso é editar
`g1_training/common/box.py`, e o pacote não edita nada de lá. Então pós-processa.
"""
from __future__ import annotations

import mujoco


def regroup(spec: mujoco.MjSpec, group: int = 2) -> mujoco.MjSpec:
    """Move todos os geoms do spec pro grupo `group`. Devolve o mesmo spec.

    Grupo de geom no MuJoCo é só um rótulo de filtro — não muda colisão, massa nem
    render. Tirar a mobília do grupo 0 basta pra o raycast de altura do pé deixar
    de considerá-la piso, e os sensores de contato (`box_support`, `body_table`)
    continuam casando, porque eles casam por NOME de geom, não por grupo."""
    for geom in spec.geoms:
        geom.group = group
    return spec
