"""Expande um checkpoint do ator 112/crítico 125 para 115/128 (`face_normal_b`).

    python -m g1_poc.expande_checkpoint --entrada model_5100.pt --saida model_5100_115.pt
    python -m g1_poc.expande_checkpoint --auto-teste

O canal novo é o ÚLTIMO dos dois grupos, de propósito: a cirurgia é um APPEND na
última dimensão de todo tensor cuja última dimensão é 112 (→115) ou 125 (→128),
onde quer que ele esteja no checkpoint. Isso cobre, com uma regra só:

    actor_state_dict / mlp.0.weight            [512, 112] -> [512, 115]  zeros
    actor_state_dict / obs_normalizer._mean    [1, 112]   -> [1, 115]    zeros
    actor_state_dict / obs_normalizer._var     [1, 112]   -> [1, 115]    UNS
    actor_state_dict / obs_normalizer._std     [1, 112]   -> [1, 115]    UNS
    critic_state_dict / (idem, 125 -> 128)
    optimizer_state_dict / state / * / exp_avg(_sq)  [512, 112|125] -> zeros

Zeros nas colunas de peso: a política começa IGNORANDO o canal novo e aprende a
usá-lo. Uns em `_var`/`_std`: variância unitária é o estado neutro do
`EmpiricalNormalization`. Zeros nos momentos do Adam: momento nulo para colunas
novas. Os demais tensores (camadas ocultas, `distribution.*` [29], biases) não
têm última dimensão 112/125 e passam intactos.

⚠ A PRIMEIRA versão deste script procurava `ckpt["actor"]["network"]` e um
normalizador `mean/var/count` — estrutura que NÃO existe no checkpoint do mjlab
— e o auto-teste "passava" sem nunca exercitar a expansão (criava um checkpoint
já-115 e o recarregava). Medido em 20/08 no Kaggle: "Chaves modificadas:
(nenhuma)" e o resume morria com size mismatch. O auto-teste de agora PODA um
checkpoint recém-salvo para 112/125 antes de expandir, e falha se a poda ou a
expansão tocarem zero tensores.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

# (última dimensão de entrada, última dimensão de saída)
_REGRAS = ((112, 115), (125, 128))


def _expande_tensor(t: torch.Tensor, velho: int, novo: int, fill: float) -> torch.Tensor:
    forma = list(t.shape)
    forma[-1] = novo
    saida = torch.full(forma, fill, dtype=t.dtype)
    saida[..., :velho] = t
    return saida


def expande(ckpt: dict) -> tuple[dict, dict]:
    """Expande IN-PLACE todo tensor com última dimensão 112→115 ou 125→128."""
    mudadas: dict[str, str] = {}

    def caminha(obj, prefixo: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                for velho, novo in _REGRAS:
                    if v.shape[-1] == velho:
                        # `_var`/`_std` do normalizador ganham UNS; todo o resto, zeros
                        fill = 1.0 if ("_var" in str(k) or "_std" in str(k)) else 0.0
                        obj[k] = _expande_tensor(v, velho, novo, fill)
                        mudadas[f"{prefixo}{k}"] = (
                            f"{tuple(v.shape)} -> {tuple(obj[k].shape)} (fill {fill})")
                        break
            elif isinstance(v, dict):
                caminha(v, f"{prefixo}{k}/")

    caminha(ckpt, "")
    return ckpt, mudadas


def _encolhe(ckpt: dict) -> int:
    """SÓ para o auto-teste: poda 115→112 e 128→125, o inverso da cirurgia.

    ⚠ Poda SÓ os tensores que a cirurgia alvo de verdade (a camada de ENTRADA e o
    normalizador). Uma poda genérica por última dimensão cortaria também camadas
    OCULTAS cuja largura coincide com 128 (`mlp.4.bias`, `mlp.6.weight`) — no
    arquivo real elas nunca têm 112/125 na última dimensão, então a EXPANSÃO
    genérica não as toca; a poda tem de espelhar isso.
    """
    n = 0

    def caminha(obj) -> None:
        nonlocal n
        if not isinstance(obj, dict):
            return
        for k, v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                if not ("mlp.0.weight" in str(k) or "obs_normalizer._" in str(k)):
                    continue
                for velho, novo in _REGRAS:
                    if v.shape[-1] == novo:
                        obj[k] = v[..., :velho].clone()
                        n += 1
                        break
            elif isinstance(v, dict):
                caminha(v)

    caminha(ckpt)
    return n


def _auto_teste() -> int:
    """Salva um checkpoint REAL do runner, poda para 112/125, expande e recarrega.

    A poda é o que a primeira versão não fazia — sem ela o teste valida um
    checkpoint que já nasceu 115 e passa sem exercitar nada.
    """
    from dataclasses import asdict

    import g1_poc  # noqa: F401  (registra a task)
    from g1_poc.env_cfg import make_g1_poc_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    env_cfg = make_g1_poc_env_cfg(play=True)
    env_cfg.scene.num_envs = 2
    agent_cfg = load_rl_cfg(g1_poc.TASK_ID)

    base = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cpu")

    caminho = "/tmp/g1_poc_ckpt_115.pt"
    runner.save(caminho)
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)

    podados = _encolhe(ckpt)
    print(f"[AUTO-TESTE] poda 115/128 -> 112/125: {podados} tensores")
    assert podados > 0, "a poda não achou tensor nenhum — a estrutura mudou?"

    ckpt, mudadas = expande(ckpt)
    print(f"[AUTO-TESTE] expansão: {len(mudadas)} tensores")
    for chave, desc in sorted(mudadas.items()):
        print(f"  {chave}: {desc}")
    assert len(mudadas) == podados, (
        f"a expansão ({len(mudadas)}) não desfez a poda ({podados})")

    # os invariantes da cirurgia, medidos no próprio arquivo
    ator = ckpt["actor_state_dict"]
    w = ator["mlp.0.weight"]
    assert w.shape[-1] == 115 and bool((w[:, 112:] == 0).all()), \
        "as colunas novas do peso têm de ser ZERO (a política começa ignorando o canal)"
    for k in ("obs_normalizer._var", "obs_normalizer._std"):
        if k in ator:
            assert bool((ator[k][..., 112:] == 1).all()), f"{k}: as colunas novas têm de ser UM"

    caminho_saida = "/tmp/g1_poc_ckpt_expandido.pt"
    torch.save(ckpt, caminho_saida)

    # recarrega pelo MESMO caminho do train.py (load default, strict)
    runner.load(caminho_saida)
    print("[AUTO-TESTE] recarregado no runner 115/128, pelo caminho do train.py — OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", type=str, default=None,
                   help="checkpoint de entrada (112/125)")
    p.add_argument("--saida", type=str, default=None,
                   help="checkpoint de saída (115/128)")
    p.add_argument("--auto-teste", action="store_true",
                   help="salva um checkpoint real, PODA para 112/125, expande e recarrega")
    args = p.parse_args()

    if args.auto_teste:
        return _auto_teste()

    if args.entrada is None or args.saida is None:
        p.print_help()
        return 1

    entrada = pathlib.Path(args.entrada).expanduser()
    saida = pathlib.Path(args.saida).expanduser()
    if not entrada.is_file():
        print(f"Erro: arquivo de entrada não encontrado: {entrada}")
        return 1

    print(f"Carregando {entrada}...")
    ckpt = torch.load(str(entrada), map_location="cpu", weights_only=False)

    ckpt, mudadas = expande(ckpt)
    print(f"\nChaves modificadas ({len(mudadas)}):")
    for chave, desc in sorted(mudadas.items()):
        print(f"  {chave}: {desc}")

    # ⚠ zero mudanças = o arquivo NÃO era 112/125 (já expandido? estrutura nova?).
    # Salvar uma cópia intacta e seguir foi exatamente o modo de falha de 20/08.
    if not mudadas:
        print("\nERRO: nenhum tensor com última dimensão 112/125 — nada a expandir.")
        print("O arquivo já está expandido, ou a estrutura do checkpoint mudou.")
        return 1

    torch.save(ckpt, str(saida))
    print(f"\nSalvo em {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
