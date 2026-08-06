"""Calibração ANALÍTICA dos números das recompensas:

    python g1_multitask/calibra.py

Zero GPU, zero run de tuning. É isto que "investigar valores melhores sem fine tune"
significa: os números saem de conta e de curva, não de comparação entre treinos. O
ajuste fino, se for necessário, acontece entre blocos de 2k-3k, e ali só peso muda —
Categoria A, grátis.

Molde: o que o doc já fez com o `upright_std = 0.1`, o único std do desenho com
justificativa numérica escrita ("fator 0.86 a 10°, 0.55 a 20°, 0.26 a 30°, 0.06 a
45° — demandante mas graduado"). A tabela 1 produz isso pra todos os outros.

Três produtos:

  1. CURVA de cada kernel contra erro FÍSICO, com a tolerância de sucesso marcada.
     Puramente analítico — só as fórmulas.
  2. ORÇAMENTO de magnitude por tarefa: `peso × valor típico`, ordenado. Revela quem
     domina. Os valores típicos vêm da matriz de contribuição (`observability.py`),
     medida num rollout curto em CPU.
  3. VARREDURA de conflito: derivada numérica de cada termo em direções físicas
     (caixa pra frente, caixa pra cima, caixa girando, robô pra frente). Pares com
     sinais opostos na mesma direção são conflito.

Critério de aceitação (do plano):
  - nenhum termo com contribuição > 10× a mediana da tarefa sem justificativa escrita
  - nenhum kernel com valor < 0.05 na tolerância de sucesso sem justificativa escrita
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import g1_multitask  # noqa: F401
from g1_multitask import tasks as T
from g1_multitask.configs import ACTIVE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

R, TOL = ACTIVE.reward, ACTIVE.tolerancia
DEVICE = "cpu"
LARG = 78
alertas: list[str] = []

DECIDIDOS = (
    "box_at_*",      # sustain_std: conferido e mantido, justificativa no knobs.py
    "box_shake",     # observar no bloco 1; ação nula infla a medição
    "table_contact",
    "arm_vel",
)
"""Achados que JÁ têm decisão escrita no `knobs.py`. Continuam sendo medidos e
impressos — o número não deixa de importar — mas saem do veredito como `decidido`
em vez de `pendente`. Sem isto o script alertaria pra sempre nos mesmos pontos e o
veredito perderia o sinal."""


def titulo(txt: str) -> None:
    print(f"\n{'=' * LARG}\n {txt}\n{'=' * LARG}")


# ============================================================ 1. curvas de kernel
titulo("1. CURVA DE CADA KERNEL CONTRA O ERRO FÍSICO")
print("""
Todo kernel do desenho é `exp(-erro²/std²)`. A pergunta que a tabela responde é:
QUANTO o termo vale no limiar em que o sucesso é declarado? Se for perto de zero, o
robô que acabou de ter sucesso não recebe quase nada — e pior, não há gradiente que
o leve de "quase" até "lá".
""")

KERNELS = [
    # (nome, std, unidade, pontos de erro, tolerância de sucesso, fator no erro)
    ("box_at_peito / box_at_prateleira", R.sustain_std, "m",
     (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20), TOL.caixa_no_alvo, 1.0),
    ("orienta_face — escala GROSSA", R.angulo_std_grosso_deg, "°",
     (0, 5, 10, 15, 30, 45, 90, 180), TOL.reorienta_angulo_deg, 1.0),
    ("orienta_face — escala FINA", R.angulo_std_fino_deg, "°",
     (0, 5, 10, 15, 30, 45, 90, 180), TOL.reorienta_angulo_deg, 1.0),
    ("orienta_face — desvio xy", R.reorienta_xy_std, "m",
     (0.0, 0.01, 0.02, 0.05, 0.10, 0.15), TOL.reorienta_xy, 1.0),
    ("reaching — GROSSA (soma das 2 palmas)", R.std_coarse, "m",
     (0.0, 0.05, 0.10, 0.20, 0.40, 0.80), None, 2.0),
    ("reaching — FINA (soma das 2 palmas)", R.std_fine, "m",
     (0.0, 0.05, 0.10, 0.20, 0.40, 0.80), None, 2.0),
]

COMBINADOS = {
    # nome do relatório: (peso, std, ...) das escalas que se SOMAM no termo real
    "orienta_face (grossa+fina, é assim que o termo soma)":
        ((0.5, R.angulo_std_grosso_deg), (0.5, R.angulo_std_fino_deg)),
    "reaching (grossa+fina, ×2 palmas)":
        ((0.5, R.std_coarse), (0.5, R.std_fine)),
}

for nome, std, unid, pontos, tolerancia, fator in KERNELS:
    print(f"\n  {nome}   std = {std} {unid}")
    linha_e, linha_v = [], []
    for e in pontos:
        v = math.exp(-fator * e * e / (std * std))
        marca = " <-tol" if tolerancia is not None and abs(e - tolerancia) < 1e-9 else ""
        linha_e.append(f"{e:>7.3g}{marca}" if marca else f"{e:>7.3g}")
        linha_v.append(f"{v:>7.3f}" if not marca else f"{v:>7.3f}     ")
    print("    erro : " + " ".join(f"{x:>11s}" for x in linha_e))
    print("    valor: " + " ".join(f"{x:>11s}" for x in linha_v))
    if tolerancia is not None:
        v_tol = math.exp(-fator * tolerancia * tolerancia / (std * std))
        print(f"    -> nesta escala, na tolerância ({tolerancia} {unid}): {v_tol:.4f}")

# O valor que importa é o do TERMO, não de uma escala isolada: o `orienta_face` e o
# `reaching` somam duas escalas, e julgar a fina sozinha diria "alerta" num termo
# que está saudável. A escala grossa existe exatamente pra cobrir a cauda da fina.
print("\n  --- valor COMBINADO na tolerância de sucesso (é este que decide) ---")
for nome, escalas in COMBINADOS.items():
    tol_uso = (TOL.reorienta_angulo_deg if "orienta" in nome else None)
    if tol_uso is None:
        continue
    v = sum(p * math.exp(-tol_uso * tol_uso / (s * s)) for p, s in escalas)
    print(f"    {nome:<52s} {v:.4f}   "
          f"[{'OK' if v >= 0.05 else 'ALERTA'}]")
    if v < 0.05:
        alertas.append(f"'{nome}' combinado vale {v:.4f} na tolerância")

v_sustain = math.exp(-TOL.caixa_no_alvo ** 2 / R.sustain_std ** 2)
print(f"    {'box_at_peito / box_at_prateleira (escala única)':<52s} {v_sustain:.4f}   "
      f"[{'OK' if v_sustain >= 0.05 else 'ALERTA'}]")
if v_sustain < 0.05:
    alertas.append(f"'box_at_*' vale {v_sustain:.4f} na tolerância de "
                   f"{TOL.caixa_no_alvo} m — e é escala ÚNICA, não tem grossa "
                   f"cobrindo a cauda")

print("""
  Como ler o alerta do `sustain_std`: a tolerância de sucesso é 0.10 m e o std é
  0.05, então o termo vale ~0.018 exatamente onde o sucesso é declarado. Duas
  leituras, e a escolha é de projeto:
    (a) está certo — precisão final DEVE ser exigente, e quem chega a 0.10 m já
        ganhou o sucesso, que é o que o currículo mede; o reward só afina.
    (b) está errado — de 0.15 m pra 0.10 m o gradiente é ~0, então a política não
        tem por onde subir, e a tolerância nunca é alcançada.
  O que decide entre (a) e (b) é se existe OUTRO termo dando gradiente naquela faixa.
  No `pegar` existe: `lift` (progresso de altura, +2.0) e `reaching` (+1.0) cobrem a
  aproximação. No `parado c/ caixa` e no `andar c/ caixa` NÃO existe — o
  `box_at_peito` é o único termo de tarefa. Ver a tabela 2.
""")

# ================================================ 2. orçamento de magnitude
titulo("2. ORÇAMENTO DE MAGNITUDE POR TAREFA")
print("""
`peso × valor típico`, medido num rollout curto com ação nula. Ação nula não é a
política treinada, então os números NÃO são a previsão do regime final — servem pra
achar termo de magnitude absurda e termo morto, que é o que se conserta antes de
gastar bloco de GPU.
""")

cfg = load_env_cfg(g1_multitask.TASK_ID)
cfg.events.pop("base_com", None)      # corrompe heap em CPU; ver knobs.DR.base_com
cfg.scene.num_envs = 128
env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
env.task_dist = torch.ones(T.NUM_TASKS, device=DEVICE)
env.reset()
for _ in range(40):
    env.step(acao)

TERMOS_TAREFA = ("lift", "reaching", "grasp", "box_at_peito", "box_at_prateleira",
                 "orienta_face", "hold_still")
"""Os termos do bloco 2 — os que dizem QUAL tarefa é. O resto é invariante de
equilíbrio e marcha, que vale em todas."""

SEM_TERMO_DE_TAREFA = (T.PARADO, T.ANDAR)
"""Estas duas NÃO têm termo de bloco 2, e isso é desenho, não falta (§6b/B: as
colunas `parado` e `andar` estão vazias no bloco 2). O objetivo delas vem do
`track_linear_velocity`/`track_angular_velocity`, que depois do F6 ficaram LIGADOS em
todas as tarefas porque toda tarefa tem uma regra de destino — mesmo que o destino
seja "onde eu já estou"."""

print("""
  Critério: não é "10x a mediana" (a mediana é dominada por termos deliberadamente
  minúsculos, tipo `joint_acc` a -2.5e-7, e aí tudo que importa passa de 10x). O que
  importa é a relação entre o SINAL DA TAREFA (bloco 2, os termos gateados) e o que
  compete com ele. Duas perguntas:
    - alguma PENALIDADE isolada é maior que a soma do sinal da tarefa?
    - o sinal da tarefa é ~zero, ou seja a tarefa não tem o que ensinar?
""")

media = env.contrib_soma / env.contrib_cont.clamp(min=1.0).unsqueeze(-1)
idx_tarefa = [i for i, n in enumerate(env.contrib_nomes) if n in TERMOS_TAREFA]
for t in range(T.NUM_TASKS):
    if float(env.contrib_cont[t]) < 50:
        print(f"\n  {T.NAMES[t]}: amostra insuficiente "
              f"({int(env.contrib_cont[t])} passos), pulando")
        continue
    linha = media[t]
    sinal = float(linha[idx_tarefa].sum())
    pen = {env.contrib_nomes[i]: float(linha[i])
           for i in range(linha.numel()) if float(linha[i]) < -1e-9}
    inv = float(sum(v for i, v in enumerate(linha.tolist())
                    if v > 0 and i not in idx_tarefa))
    print(f"\n  {T.NAMES[t]}  (n={int(env.contrib_cont[t])})   "
          f"sinal de tarefa={sinal:+.3f}   invariantes={inv:+.3f}   "
          f"penalidades={sum(pen.values()):+.3f}")
    for i in torch.argsort(linha.abs(), descending=True)[:6].tolist():
        v = float(linha[i])
        if abs(v) < 1e-9:
            continue
        marca = " [tarefa]" if i in idx_tarefa else ""
        print(f"      {env.contrib_nomes[i]:<26s} {v:+8.4f}{marca}")

    if t in SEM_TERMO_DE_TAREFA:
        print("      (sem termo de bloco 2 por desenho — o objetivo desta tarefa vem "
              "do `track_*`)")
    elif abs(sinal) < 0.01:
        print("      >>> a tarefa NÃO TEM SINAL neste estado")
        alertas.append(f"`{T.NAMES[t]}`: sinal de tarefa ~0 ({sinal:+.4f}) — "
                       f"nada na reward diz o que fazer")
    else:
        piores = [(k, v) for k, v in pen.items() if abs(v) > abs(sinal)]
        for k, v in sorted(piores, key=lambda kv: kv[1])[:2]:
            print(f"      >>> penalidade `{k}` ({v:+.3f}) MAIOR que o sinal "
                  f"da tarefa ({sinal:+.3f})")
            alertas.append(f"`{T.NAMES[t]}`: penalidade '{k}' ({v:+.3f}) supera o "
                           f"sinal de tarefa ({sinal:+.3f})")

# ==================================================== 3. varredura de conflito
titulo("3. VARREDURA DE CONFLITO")
print("""
Derivada numérica de cada termo em direções FÍSICAS. Dois termos com sinais opostos
na mesma direção estão pedindo coisas contrárias. O doc achou 3 desses à mão
(`com_balance` × andar, `box_shake` × reorientar, `dof_pos_limits` × altura 0.00);
esta tabela varre o resto.
""")


def _estado(e):
    cx = e.scene["box"]
    ro = e.scene["robot"]
    return (cx.data.root_link_pos_w.clone(), cx.data.root_link_quat_w.clone(),
            ro.data.root_link_pos_w.clone(), ro.data.root_link_quat_w.clone())


def _restaura(e, st):
    pos, quat, rpos, rquat = st
    z6 = torch.zeros(e.num_envs, 6, device=e.device)
    e.scene["box"].write_root_state_to_sim(torch.cat([pos, quat, z6], dim=-1))
    e.scene["robot"].write_root_link_pose_to_sim(torch.cat([rpos, rquat], dim=-1))
    e.sim.forward()


LE_DERIVADA = ("box_shake", "joint_acc", "arm_vel", "body_ang_vel",
               "angular_momentum", "soft_landing_feet", "soft_landing_table",
               "feet_slip", "action_rate_l2", "track_linear_velocity",
               "track_angular_velocity", "foot_clearance", "foot_swing_height")
"""Termos que leem VELOCIDADE ou ACELERAÇÃO — excluídos desta varredura.

Não é preguiça, é limite do método: a varredura teleporta o objeto pra medir a
derivada, e teleporte cria descontinuidade de velocidade. O `sim.forward()` depois
dele calcula uma ω espúria, e aí o `box_shake` (que é `‖ω_caixa‖²`) reporta salto de
+2.89 que não existe na física contínua. Medir a derivada destes termos exigiria
perturbar a VELOCIDADE, não a posição — outra varredura."""


def _valores(e) -> dict[str, float]:
    out = {}
    for nome in e.reward_manager.active_terms:
        if nome in LE_DERIVADA:
            continue
        tc = e.reward_manager.get_term_cfg(nome)
        if tc.weight == 0.0:
            continue
        v = tc.func(e, **tc.params)
        out[nome] = float((v * tc.weight).mean())
    return out


DIRECOES = (
    ("caixa +x (2 cm pra frente)", "box", 0, +0.02),
    ("caixa +z (2 cm pra cima)", "box", 2, +0.02),
    ("robô +x (2 cm pra frente)", "robot", 0, +0.02),
)

for tarefa in (T.ANDAR, T.PEGAR, T.REORIENTAR, T.BOTAR):
    env.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env.task_dist[tarefa] = 1.0
    env.reset()
    env.step(acao)
    print(f"\n  {T.NAMES[tarefa]}")
    for rotulo, quem, eixo, delta in DIRECOES:
        st = _estado(env)
        base = _valores(env)
        ent = env.scene[quem]
        pos = ent.data.root_link_pos_w.clone()
        pos[:, eixo] += delta
        if quem == "box":
            env.scene["box"].write_root_state_to_sim(torch.cat(
                [pos, ent.data.root_link_quat_w,
                 torch.zeros(env.num_envs, 6, device=env.device)], dim=-1))
        else:
            env.scene["robot"].write_root_link_pose_to_sim(
                torch.cat([pos, ent.data.root_link_quat_w], dim=-1))
        env.sim.forward()
        depois = _valores(env)
        _restaura(env, st)

        d = {k: depois[k] - base[k] for k in base}
        sobe = sorted([(v, k) for k, v in d.items() if v > 1e-5], reverse=True)
        desce = sorted([(v, k) for k, v in d.items() if v < -1e-5])
        if not sobe and not desce:
            print(f"    {rotulo:<30s} nenhum termo reage")
            continue
        s = ", ".join(f"{k}{v:+.4f}" for v, k in sobe[:3]) or "—"
        c = ", ".join(f"{k}{v:+.4f}" for v, k in desce[:3]) or "—"
        print(f"    {rotulo:<30s} sobe: {s}")
        print(f"    {'':<30s} desce: {c}")
        if sobe and desce:
            alertas.append(f"em `{T.NAMES[tarefa]}`, '{rotulo}': "
                           f"{sobe[0][1]} sobe e {desce[0][1]} desce")

# ================================================================= veredito
titulo("VEREDITO")
decididos = [a for a in alertas if any(d in a for d in DECIDIDOS)]
pendentes = [a for a in alertas if a not in decididos]

if decididos:
    print(f"  {len(decididos)} com decisão JÁ ESCRITA no knobs.py (continuam medidos):\n")
    for a in decididos:
        print(f"    ok  {a}")
if pendentes:
    print(f"\n  {len(pendentes)} PENDENTES — exigem justificativa escrita ao lado do "
          f"número:\n")
    for i, a in enumerate(pendentes, 1):
        print(f"   {i:2d}. {a}")
    print("\n  Pendência não é bug. Cada uma é escolha de projeto, e 'conferido,\n"
          "  mantido' com o número ao lado é resposta válida.")
else:
    print("\n  Nenhuma pendência.")
