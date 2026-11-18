"""Coleta os dados brutos, passo a passo, de uma política treinada da v3 (18/11). A análise roda fora do pod.

Por quê: o `model_8150` pega a caixa, devolve à laje e solta no P4 (a medida de 18/11), e o ANDAR parou no nível 6.
Em vez de um script por pergunta, este grava uma linha por env e por passo: o estado ANTES da ação (locomoção, caixa,
mãos, cadeia: d, σ, Φ e os 6 portões), o valor do crítico e, DEPOIS do passo, todos os termos de recompensa (por
segundo, com o peso), as terminações e o fim do episódio. Metade dos episódios novos cai no ANDAR e metade na fatia B
(a política não muda; só o sorteio da fatia).

Saída: `coleta.npz` com `x` (T,E,F) float16 e `colunas`; `ids` (T,E,K) int16 e `colunas_ids`; `rew` (T,E,R) float16 e
`termos_rew`; `term` (T,E,M) bool e `termos_term`; e `meta.json` (argumentos e o estado do currículo do checkpoint).

Uso:
    python -m g1_v3.coleta --checkpoint CAMINHO/model_N.pt [--num-envs 256] [--passos 3000] [--amostra]
                           [--saida ./coleta_saida] [--device cuda:0]
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import g1_v3.tarefa as tarefa  # registra a task
from g1_v3 import alvos, recompensas
from g1_v3.alvos import P4
from g1_v3.env_cfg import make_env_cfg
from g1_v3.metricas import CAIXA_ROBO
from g1_v3.runner import RunnerV3
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--passos", type=int, default=3000)
ap.add_argument("--amostra", action="store_true", help="ação amostrada com o std treinado (como no treino)")
ap.add_argument("--saida", default="./coleta_saida")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
saida = Path(args.saida)
saida.mkdir(parents=True, exist_ok=True)

env_cfg = make_env_cfg(play=False)  # como no treino: empurrão, DR e ruído
env_cfg.scene.num_envs = args.num_envs
rl = tarefa.rl_cfg()
env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
env = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
runner = RunnerV3(env, asdict(rl), device=args.device)
runner.load(args.checkpoint, load_cfg={"actor": True, "critic": True}, strict=True, map_location=args.device)
policy = runner.get_inference_policy(device=args.device)
critico = runner.alg.critic
u = env.unwrapped
meta = {"args": vars(args), "dt": float(u.step_dt), "curriculo": u.v3_cur.estado()}

# todo episódio novo: metade ANDAR, metade fatia B (ANDAR, B, C); os já em curso seguem até o reset
u.v3_cur.fracoes_fatias = lambda: torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)

# o `_phi_env` chama `phi_do_alvo` pelo módulo: a troca guarda d, σ e os portões da última chamada
phi_original, cap = recompensas.phi_do_alvo, {}


def phi_capturando(indice, d, sigma, fator, portoes):
    cap.update(d=d, sigma=sigma, **portoes)
    return phi_original(indice, d, sigma, fator, portoes)


recompensas.phi_do_alvo = phi_capturando
# como o treino (`init_at_random_ep_len=True`): episódios fora de fase entre os envs
u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
cadeia, twist = u.command_manager.get_term("cadeia"), u.command_manager.get_term("twist")
robo, rm, tm = u.scene["robot"], u.reward_manager, u.termination_manager
xs, ids, rews, terms = [], [], [], []

with torch.inference_mode():
    obs = env.get_observations()
    for _ in range(args.passos):
        # estado ANTES da ação
        m, d = cadeia._mede(), robo.data
        phi, conq = recompensas._phi_env(u, u.v3_alvo_idx, u.v3_sigma)  # o mesmo Φ do progresso, no alvo corrente
        faces = u.scene["caixa_faces"].data
        f_esq, f_dir = recompensas.forcas_por_face(
            faces.normal, faces.force, faces.found > 0,
            alvos.frame_da_caixa(u.scene["box"].data.root_link_quat_w, d.heading_w))
        caixa_b = quat_apply_inverse(m["base_q"], m["caixa_p"] - m["base_p"])
        e_p4 = quat_apply_inverse(m["base_q"], m["caixa_p"] - recompensas.alvo_da_caixa_w(u, torch.full_like(u.v3_alvo_idx, P4)))
        cmd = u.command_manager.get_command("twist")
        acao = policy(obs, stochastic_output=args.amostra)
        x = {
            "cmd_vx": cmd[:, 0], "cmd_vy": cmd[:, 1], "cmd_wz": cmd[:, 2],
            "v_bx": d.root_link_lin_vel_b[:, 0], "v_by": d.root_link_lin_vel_b[:, 1], "v_bz": d.root_link_lin_vel_b[:, 2],
            "w_bz": d.root_link_ang_vel_b[:, 2], "h_pelve": m["h"], "pelve_sobre_pe": m["pelve"],
            "tronco_graus": m["tronco_graus"], "dpalmas": m["dpalmas"],
            "caixa_bx": caixa_b[:, 0], "caixa_by": caixa_b[:, 1], "caixa_bz": caixa_b[:, 2], "h_caixa": m["h_caixa"],
            "incl_caixa": m["incl_graus"], "v_caixa": m["v_caixa"].norm(dim=-1), "v_rel": m["v_rel"],
            "apoio": m["apoio"], "apoio_N": m["apoio_N"], "sobre_laje": m["sobre"],
            "contatos": m["contatos"].float().sum(-1), "F_esq": f_esq, "F_dir": f_dir,
            "F_total": u.scene[CAIXA_ROBO].data.force.norm(dim=-1).sum(-1),
            "d": cap["d"], "sigma": cap["sigma"], "phi": phi, "q_palmas": conq[:, 0], "q_pega": conq[:, 1],
            **{f"p_{n}": cap[n] for n in ("desloc", "incl", "maos", "tronco", "recuo", "com")},
            "h_renda": u.v3_h, "phi_melhor": u.v3_phi_melhor, "relogio_alvo": u.v3_relogio_alvo,
            "e_p4_frente": e_p4[:, 0], "e_p4_lado": e_p4[:, 1], "e_p4_z": e_p4[:, 2],
            "topo": u.v3_topo, "meia": u.v3_meia, "massa": u.v3_massa,
            "valor": critico(obs).flatten(), "acao_norma": acao.norm(dim=-1),
        }
        i = {
            "fatia": u.v3_fatia, "tarefa": u.v3_tarefa, "trecho": u.v3_trecho, "alvo": u.v3_alvo_idx, "celula": twist.seg_cel,
            "passo_ep": u.episode_length_buf, "pegou": u.v3_pegou, "perdeu": u.v3_perdeu, "p4_fechado": u.v3_p4_fechado,
            "b4_fechado": u.v3_b4_fechado,
        }
        obs, _, fim, _ = env.step(acao)
        # DEPOIS do passo: a recompensa e as terminações do passo (calculadas antes do reset do env que acabou)
        i["fim"] = fim
        xs.append(torch.stack([v.float() for v in x.values()], -1).half().cpu())
        ids.append(torch.stack([v.to(torch.int16) for v in i.values()], -1).cpu())
        rews.append(rm._step_reward.half().cpu())
        terms.append(torch.stack([tm.get_term(n) for n in tm.active_terms], -1).cpu())

np.savez_compressed(
    saida / "coleta.npz", x=torch.stack(xs).numpy(), colunas=np.array(list(x)), ids=torch.stack(ids).numpy(),
    colunas_ids=np.array(list(i)), rew=torch.stack(rews).numpy(), termos_rew=np.array(rm.active_terms),
    term=torch.stack(terms).numpy(), termos_term=np.array(tm.active_terms))
(saida / "meta.json").write_text(json.dumps(meta, default=str, indent=1))
a = torch.stack(ids)[..., list(i).index("alvo")]
print(f"política: {'amostrada' if args.amostra else 'média'}; passos={args.passos}; envs={args.num_envs}; "
      f"colunas={len(x)}; termos de recompensa={len(rm.active_terms)}")
print("passos por alvo (−1 = sem alvo): " + ", ".join(f"{int(k)}: {int((a == k).sum())}" for k in a.unique()))
print(f"saída em {saida.resolve()} ({(saida / 'coleta.npz').stat().st_size / 1e6:.0f} MB)")
