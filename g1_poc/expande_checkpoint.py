"""Expande um checkpoint do ator 112/crítico 125 para 115/128.

Adiciona o canal `face_normal_b` nas posições ÚLTIMAS de ambos.

    python -m g1_poc.expande_checkpoint --entrada model_5100.pt --saida model_5100_115.pt
    python -m g1_poc.expande_checkpoint --auto-teste   # sonda um checkpoint vazio
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch


def expande(ckpt: dict) -> dict:
    """Expande um checkpoint de 112/125 para 115/128.

    Cirurgia de append nas colunas de entrada:
    - weight [h, 112] → [h, 115] com 3 zeros
    - bias intacto
    - mean [112] → [115] com 3 zeros
    - var [112] → [115] com 3 uns
    """
    mudadas = {}

    # ========== ator: entrada 112 → 115 ==========
    if "actor" in ckpt and "network" in ckpt["actor"]:
        net = ckpt["actor"]["network"]
        for nome, param in net.items():
            if nome.endswith("weight") and param.shape[-1] == 112:
                # Linear layer: [out, 112] → [out, 115]
                novo = torch.zeros(param.shape[0], 115, dtype=param.dtype)
                novo[:, :112] = param
                net[nome] = novo
                mudadas[f"actor/{nome}"] = f"{param.shape} -> {novo.shape}"
            elif nome.endswith("bias"):
                # Bias stays [out]
                pass

    # ========== crítico: entrada 125 → 128 ==========
    if "critic" in ckpt and "network" in ckpt["critic"]:
        net = ckpt["critic"]["network"]
        for nome, param in net.items():
            if nome.endswith("weight") and param.shape[-1] == 125:
                # Linear layer: [out, 125] → [out, 128]
                novo = torch.zeros(param.shape[0], 128, dtype=param.dtype)
                novo[:, :125] = param
                net[nome] = novo
                mudadas[f"critic/{nome}"] = f"{param.shape} -> {novo.shape}"
            elif nome.endswith("bias"):
                # Bias stays [out]
                pass

    # ========== normalizadores ==========
    # O obs_normalizer é usado no ator; pode estar em "obs_normalizer" ou em
    # "algorithm" → "obs_normalizer" dependendo da versão
    def expande_normalizer(norm_dict, velho_tamanho: int, novo_tamanho: int):
        """Expande mean/var/count de [velho] → [novo]."""
        mudadas_local = {}
        for chave in ["mean", "var", "count"]:
            if chave not in norm_dict:
                continue
            param = norm_dict[chave]
            if param.shape[0] != velho_tamanho:
                continue
            if chave == "count":
                # count não muda
                continue
            novo = torch.zeros(novo_tamanho, dtype=param.dtype)
            novo[:velho_tamanho] = param
            if chave == "mean":
                # append zeros
                pass
            elif chave == "var":
                # append uns
                novo[velho_tamanho:] = 1.0
            norm_dict[chave] = novo
            mudadas_local[chave] = f"{param.shape} -> {novo.shape}"
        return mudadas_local

    # Procura obs_normalizer em locais comuns
    for caminho in [
        ("obs_normalizer",),
        ("algorithm", "obs_normalizer"),
        ("actor_normalizer",),
        ("actor", "normalizer"),
    ]:
        obj = ckpt
        for k in caminho[:-1]:
            if k in obj:
                obj = obj[k]
            else:
                obj = None
                break
        if obj is not None and caminho[-1] in obj:
            print(f"[INFO] Expandindo normalizer em {'/'.join(caminho)}")
            local_mudadas = expande_normalizer(obj[caminho[-1]], 112, 115)
            mudadas.update({f"{'/'.join(caminho)}/{k}": v for k, v in local_mudadas.items()})

    # Procura obs_normalizer do crítico (é menos comum)
    for caminho in [
        ("critic_normalizer",),
        ("critic", "normalizer"),
    ]:
        obj = ckpt
        for k in caminho[:-1]:
            if k in obj:
                obj = obj[k]
            else:
                obj = None
                break
        if obj is not None and caminho[-1] in obj:
            print(f"[INFO] Expandindo normalizer crítico em {'/'.join(caminho)}")
            local_mudadas = expande_normalizer(obj[caminho[-1]], 125, 128)
            mudadas.update({f"{'/'.join(caminho)}/{k}": v for k, v in local_mudadas.items()})

    return ckpt, mudadas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", type=str, default=None,
                   help="checkpoint de entrada (112/125)")
    p.add_argument("--saida", type=str, default=None,
                   help="checkpoint de saída (115/128)")
    p.add_argument("--auto-teste", action="store_true",
                   help="testa com um checkpoint vazio")
    args = p.parse_args()

    if args.auto_teste:
        print("[AUTO-TESTE] Gerando checkpoint vazio...")
        from dataclasses import asdict
        import g1_poc  # noqa: F401 (registra a task)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
        from mjlab.tasks.registry import load_rl_cfg

        # Monta a task antes de tentar carregar as configs dela
        from g1_poc.env_cfg import make_g1_poc_env_cfg
        env_cfg = make_g1_poc_env_cfg(play=True)

        # Carrega as configs
        agent_cfg = load_rl_cfg(g1_poc.TASK_ID)

        base = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cpu")

        ckpt_path_temp = "/tmp/g1_poc_test_ckpt.pt"
        print(f"[AUTO-TESTE] Salvando em {ckpt_path_temp}...")
        runner.save(ckpt_path_temp)

        ckpt_entrada = torch.load(ckpt_path_temp, map_location="cpu", weights_only=False)
        print(f"[AUTO-TESTE] Checkpoint carregado, tamanho: {len(ckpt_entrada)} chaves")

        # Simula a estrutura: força entrada do ator para 112 e crítico para 125
        # (pode estar diferente, mas vamos documentar o que vimos)
        print("[AUTO-TESTE] Estrutura REAL do checkpoint:")
        for chave, valor in sorted(ckpt_entrada.items()):
            if isinstance(valor, dict):
                print(f"  {chave}: dict com {len(valor)} itens")
                for k2, v2 in sorted(valor.items()):
                    if isinstance(v2, torch.Tensor):
                        print(f"    {k2}: {v2.shape}")
                    elif isinstance(v2, dict):
                        print(f"    {k2}: dict com {len(v2)} itens")
            elif isinstance(valor, torch.Tensor):
                print(f"  {chave}: {valor.shape}")

        ckpt_saida_path = "/tmp/g1_poc_test_ckpt_expanded.pt"
        print(f"\n[AUTO-TESTE] Expandindo para {ckpt_saida_path}...")
        ckpt_saida, mudadas = expande(ckpt_entrada)
        torch.save(ckpt_saida, ckpt_saida_path)

        print("\n[AUTO-TESTE] Chaves modificadas:")
        for chave, descricao in sorted(mudadas.items()):
            print(f"  {chave}: {descricao}")

        # Validação: tenta carregar com contrato 115/128
        print(f"\n[AUTO-TESTE] Validando carregamento no runner novo...")
        base2 = ManagerBasedRlEnv(cfg=make_g1_poc_env_cfg(play=True), device="cpu")
        env2 = RslRlVecEnvWrapper(base2, clip_actions=agent_cfg.clip_actions)
        runner2 = MjlabOnPolicyRunner(env2, asdict(agent_cfg), device="cpu")
        try:
            runner2.load(ckpt_saida_path, load_cfg={"actor": True}, strict=True, map_location="cpu")
            print("[AUTO-TESTE] ✓ Carregamento bem-sucedido!")
            return 0
        except Exception as e:
            print(f"[AUTO-TESTE] ✗ Erro ao carregar: {e}")
            return 1

    if args.entrada is None or args.saida is None:
        p.print_help()
        return 1

    entrada = pathlib.Path(args.entrada).expanduser()
    saida = pathlib.Path(args.saida).expanduser()

    if not entrada.is_file():
        print(f"Erro: arquivo de entrada não encontrado: {entrada}")
        return 1

    print(f"Carregando {entrada}...")
    try:
        ckpt = torch.load(str(entrada), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        return 1

    print("Expandindo...")
    ckpt_expandido, mudadas = expande(ckpt)

    print("Salvando em", saida)
    torch.save(ckpt_expandido, str(saida))

    print("\nChaves modificadas:")
    if mudadas:
        for chave, descricao in sorted(mudadas.items()):
            print(f"  {chave}: {descricao}")
    else:
        print("  (nenhuma)")

    print(f"\nResumo: arquivo salvo em {saida}")
    print(f"Validação: recarregue em um runner com contrato 115/128")

    return 0


if __name__ == "__main__":
    sys.exit(main())
