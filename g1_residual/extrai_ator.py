"""PASSO 1 — separa do checkpoint do BFM-Zero só o que a inferência usa.

    python g1_residual/extrai_ator.py [caminho/do/BFM-Zero]

O checkpoint inteiro tem 3,15 GB. Quase tudo ali serve para TREINAR o BFM, e o
BFM fica congelado aqui. A conta por grupo de tensores:

    518,3 MB  _forward_map          treino
    518,3 MB  _target_forward_map   treino
    514,3 MB  _critic               treino  (x4 grupos: critic/aux/target)
     11,1 MB  _discriminator        treino
      0,8 MB  _backward_map         só reward inference, e o `z` já vem pronto
    121,7 MB  _actor                PRECISA
      0,0 MB  _obs_normalizer       PRECISA

Sobram ~122 MB. São 26 vezes menos dados para subir no Kaggle.

⚠️ O `_obs_normalizer` é fácil de esquecer e o erro é SILENCIOSO. O BFM ajusta a
escala da entrada com estatísticas próprias, guardadas dentro do checkpoint
(`BatchNormNormalizer`, uma por chave da obs). Sem ele o ator recebe números na
escala errada e devolve lixo — sem exceção, sem aviso, só comportamento ruim.

O `z` também sai daqui, e não precisa de reward inference: o
`model/reward_inference/reward_locomotion.pkl` já traz 43 comportamentos com 10
sementes cada, todos com norma sqrt(256) = 16. Entre eles estão os que o
currículo precisa — `move-ego-0-0` (parado), `crouch-0` (agachado),
`move-arms-0-0.7-m-m` (anda com os braços erguidos).

A leitura é LAZY (`safetensors.safe_open`): o script nunca carrega os 3,15 GB na
memória, lê tensor por tensor e guarda só os dois grupos que interessam.
"""
import json
import pathlib
import pickle
import sys

import torch
from safetensors import safe_open

BFM = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else "/home/joaobornelli/Documents/BFM-Zero")
CKPT = BFM / "model" / "checkpoint" / "model"
PKL = BFM / "model" / "reward_inference" / "reward_locomotion.pkl"
SAIDA = pathlib.Path(__file__).resolve().parent / "peso" / "bfm_ator.pt"

GRUPOS = ("_actor.", "_obs_normalizer.")
"""Os dois únicos prefixos de chave que a inferência usa.

O `_actor` filtra a obs pelas chaves `state`, `last_action` e `history_actor` —
o `privileged_state` (463 números) entra no espaço de obs mas o ator NÃO o lê, e
o próprio código de inferência do BFM passa zeros ali."""


def main() -> None:
    assert CKPT.is_dir(), f"não achei o checkpoint em {CKPT}"
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. tensores: lê lazy e guarda só os dois grupos --------------------
    pesos: dict[str, torch.Tensor] = {}
    total_bytes = 0
    with safe_open(str(CKPT / "model.safetensors"), framework="pt") as f:
        for chave in f.keys():
            total_bytes += 1
            if chave.startswith(GRUPOS):
                pesos[chave] = f.get_tensor(chave)
    bytes_uteis = sum(t.numel() * t.element_size() for t in pesos.values())
    print(f"tensores: {len(pesos)} de {total_bytes} guardados "
          f"({bytes_uteis / 2**20:.1f} MB)")
    for g in GRUPOS:
        n = sum(1 for k in pesos if k.startswith(g))
        assert n > 0, f"nenhum tensor de {g} — o checkpoint mudou de nome de chave"
        print(f"  {g:20s} {n:>4} tensores")

    # ---- 2. os dois JSON de configuração, verbatim --------------------------
    # São os MESMOS arquivos que o `load_model` do fabricante lê, então o
    # `bfm.py` reconstrói o ator pelo caminho testado, sem redigitar arquitetura.
    cfg = json.loads((CKPT / "config.json").read_text())
    init = json.loads((CKPT / "init_kwargs.json").read_text())
    print(f"\nconfig: z_dim={cfg['archi']['z_dim']} "
          f"actor_std={cfg['actor_std']} amp={cfg['amp']} "
          f"actor={cfg['archi']['actor']['hidden_dim']}x"
          f"{cfg['archi']['actor']['hidden_layers']}")
    print(f"action_dim={init['action_dim']} "
          f"obs keys={sorted(init['obs_space']['spaces'])}")

    # ---- 3. tabela de `z`: 43 comportamentos, 10 sementes cada --------------
    # `pickle.load` aqui é seguro: o arquivo vem do release do BFM-Zero no
    # HuggingFace, já baixado e JÁ EXECUTADO localmente pelo `teste_sim.py` do
    # próprio repo. Não é entrada de terceiro, e este script roda offline, no PC
    # do dono. O que sai daqui é convertido para tensores e regravado em `.pt`,
    # então o Kaggle nunca vê o pickle.
    bruto = pickle.load(open(PKL, "rb"))
    z_tab: dict[str, torch.Tensor] = {}
    for nome, lista in bruto.items():
        # cada item é [1, 256]; empilha as sementes em [10, 256]
        z_tab[nome] = torch.cat([torch.as_tensor(t).reshape(1, -1)
                                 for t in lista], dim=0).float()
    d = z_tab[next(iter(z_tab))].shape[1]
    print(f"\nz: {len(z_tab)} comportamentos x {z_tab[next(iter(z_tab))].shape[0]} "
          f"sementes, dim {d}")
    assert d == cfg["archi"]["z_dim"], "dim do z não bate com o config"

    # ---- 4. constantes do plant do BFM ------------------------------------
    # Sem estas o Kaggle não sabe converter a ação do BFM em alvo de junta. Elas
    # moram no `common.py` do BFM-Zero, que NÃO é vendorizado (é script, não
    # biblioteca), então saem por AST — só as atribuições de topo, sem executar o
    # resto do arquivo.
    import ast
    ns: dict = {"np": __import__("numpy")}
    fonte = (BFM / "common.py").read_text()
    arvore = ast.parse(fonte)
    so_consts = ast.Module(
        body=[n for n in arvore.body
              if isinstance(n, (ast.Assign, ast.Import, ast.ImportFrom))],
        type_ignores=[])
    exec(compile(so_consts, "common.py", "exec"), ns)   # noqa: S102
    plant = {k.lower(): torch.as_tensor(ns[k]).float()
             for k in ("ACTION_SCALES", "DEFAULT_JOINT_POS", "KP_GAINS", "KD_GAINS")}
    plant["action_rescale"] = torch.tensor(float(ns["ACTION_RESCALE"]))
    print(f"\nplant do BFM: escala {plant['action_scales'].min():.4f}"
          f"..{plant['action_scales'].max():.4f}"
          f" | rescale {plant['action_rescale']:.1f}"
          f" | pose padrão máx {plant['default_joint_pos'].abs().max():.3f} rad")
    print("  ⚠️ a pose padrão do BFM difere da nossa em até 0.600 rad (34.4°) —"
          " joelho 0.300 contra 0.669, cotovelo 0.000 contra 0.600.")

    # ---- 5. grava --------------------------------------------------------
    torch.save({"pesos": pesos, "config": cfg, "init_kwargs": init,
                "z": z_tab, "plant": plant}, SAIDA)
    mb = SAIDA.stat().st_size / 2**20
    print(f"\ngravado: {SAIDA}  ({mb:.1f} MB)")
    if mb > 100:
        print("  ⚠️ acima de 100 MB — o GitHub recusa. Suba como Kaggle Dataset;\n"
              "     `peso/` já está fora do git.")


if __name__ == "__main__":
    main()
