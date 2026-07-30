"""Smoke do multi-tarefa — roda local, em CPU, em segundos:

    python g1_multitask/smoke.py

Existe porque a §15 do doc diz que NADA roda no notebook a não ser o `play`, então
não há mais "portão local grátis e ilimitado" pra caçar bug de resume vinte vezes.
O substituto é uma lista ESCRITA de checks, rodada antes de submeter. A lista da
§15 é o roteiro deste arquivo:

  - montagem do config              -> `KeyError: 'twist'` se o comando vier depois
  - os 19+ termos executando 1x     -> shape no `gated()`, `inner` sem kwargs
  - `env.active_task` antes do 1º reward
  - índices do congelamento         -> fatia errada NÃO dá erro, só treina mal
  - as 3 terminações novas
  - `sample`/`report` com env_ids parcial
  - salvar e retomar                -> é o portão do plano inteiro

Cresce junto com o plano: cada tarefa implementada acrescenta uma seção aqui.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import g1_multitask  # noqa: F401  (o import dispara o register_mjlab_task)
from g1_multitask import tasks as T
from g1_multitask.configs import ACTIVE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

DEVICE = "cpu"
NUM_ENVS = 4

# Largura da observação do ATOR (§14). 132 herdada do `g1_training`, menos o bit
# `phase` (F10), mais 20 canais novos: box_rot_b 6 + face_alvo 3 + dir_alvo 3 +
# task_onehot 8. Mudar esta largura é Categoria C — recomeçar do zero.
OBS_ESPERADA = 151

PRIVILEGIO_CRITICO = 12
"""Quanto o crítico é MAIOR que o ator, e tem que continuar sendo.

O crítico vê 4 termos que o ator não vê: `foot_air_time`, `foot_contact`,
`foot_contact_forces`, `foot_height` — grandezas privilegiadas do sim, que o robô
real não mede. O contrato sim-to-real é sobre o ATOR; o crítico pode ver mais,
e o §14 fala de "obs do ator" justamente por isso.

O que este número protege: os dois grupos têm que crescer JUNTOS nos canais de
comando. Se a Tarefa 3 acrescentar um termo só no ator, esta diferença muda e
denuncia — e um canal de comando que o crítico não vê estimaria valor errado."""

falhas: list[str] = []


def check(nome: str, cond: bool, detalhe: str = "") -> None:
    if cond:
        print(f"  OK    {nome}" + (f"  ({detalhe})" if detalhe else ""))
    else:
        print(f"  FALHA {nome}  {detalhe}")
        falhas.append(nome)


# ------------------------------------------------------------------ T1: registro
print("\n-- registro da task --")
check("task registrada", g1_multitask.TASK_ID in list_tasks(), g1_multitask.TASK_ID)
check(
    "não colide com as tasks do g1_training",
    sum(1 for t in list_tasks() if "Lift-Box" in t) == 3,
    "Stand / Stand-Step / Lift intactas",
)

# ---------------------------------------------------- T1: a conta dos 60 eventos
print("\n-- conta dos destravamentos (§14: 60) --")
por_fonte = T.unlock_count()
for fonte, n in por_fonte.items():
    print(f"        {fonte:<14} {n}")
check("total = 60", T.total_unlocks() == 60, f"deu {T.total_unlocks()}")
# Cada linha é derivada dos níveis e do índice inicial. Os valores esperados vêm
# da contagem fechada em 29/07 — se uma linha mudar sozinha, um nível foi mexido.
for fonte, esperado in (
    ("parado", 0), ("andar", 4), ("reorientar", 13), ("pegar", 13),
    ("botar", 10), ("parado_caixa", 4), ("andar_caixa", 6),
    ("push", 4), ("aberturas", 6),
):
    check(f"{fonte} = {esperado}", por_fonte[fonte] == esperado, f"deu {por_fonte[fonte]}")

# ------------------------------------------------- T1: eixos e índices iniciais
print("\n-- escopo do eixo de distância (regra fechada 29/07) --")
# Quem ANDA começa em 0.3; quem MANIPULA começa em 0.0; três tarefas não têm o eixo.
check("andar começa em 0.3", T.axis_levels(T.ANDAR, "distancia")[0] == 0.3)
check("andar c/ caixa começa em 0.3", T.axis_levels(T.ANDAR_CAIXA, "distancia")[0] == 0.3)
check("pegar começa em 0.0", T.axis_levels(T.PEGAR, "distancia")[0] == 0.0)
check("reorientar começa em 0.0", T.axis_levels(T.REORIENTAR, "distancia")[0] == 0.0)
for t in (T.PARADO, T.PARADO_CAIXA, T.BOTAR):
    check(f"{T.NAMES[t]} não tem eixo de distância", "distancia" not in T.AXES[t])
check("giro tem 5 níveis (o 5º é topo/fundo)", len(T.LEVELS["giro"]) == 5)

# ------------------------------------------------------- T1: env monta e roda
print("\n-- env monta, reseta, 1 step --")
def sem_dr_instavel(c):
    """Rede de segurança: garante que o `base_com` não voltou.

    `dr.body_com_offset` corrompe memória em CPU **e em GPU** (medido 30/07, A/B com
    `CUDA_LAUNCH_BLOCKING=1`). Ele está desligado por default no `knobs.DR`, então
    este pop é no-op — existe pra o caso de alguém religar sem rodar o A/B."""
    c.events.pop("base_com", None)
    return c


cfg = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg.scene.num_envs = NUM_ENVS
env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
env.reset()
acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
env.step(acao)
check("1 step sem erro", True)

largura = env.observation_manager.group_obs_dim["actor"][0]
largura_critic = env.observation_manager.group_obs_dim["critic"][0]
print(f"        termos do ator: {list(cfg.observations['actor'].terms)}")
check("obs do ator", largura == OBS_ESPERADA, f"{largura} (esperado {OBS_ESPERADA})")
check("crítico = ator + privilégio", largura_critic - largura == PRIVILEGIO_CRITICO,
      f"critic={largura_critic}, ator={largura}, diff={largura_critic - largura}")
check("cena com 3 entidades", len(cfg.scene.entities) == 3, str(list(cfg.scene.entities)))
check("action_scale_mult aplicado",
      ACTIVE.foundation.action_scale_mult == 0.8, "0.8 = config ativo")

# ------------------------------------------------- T2: os dois termos de comando
print("\n-- comando: layout e ordem --")
from g1_multitask import commands as C  # noqa: E402

check("lift_target tem 17 números", C.COMMAND_DIM == 17)
check("ordem: lift_target antes de twist",
      list(cfg.commands) == ["lift_target", "twist"], str(list(cfg.commands)))
check("bit phase apagado", "phase" not in cfg.observations["actor"].terms)
meta = env.command_manager.get_term("lift_target")
twist = env.command_manager.get_term("twist")
check("shape do lift_target", tuple(meta.command.shape) == (NUM_ENVS, 17),
      str(tuple(meta.command.shape)))
check("shape do twist", tuple(twist.command.shape) == (NUM_ENVS, 3),
      str(tuple(twist.command.shape)))
check("env.active_task existe antes do 1º reward", hasattr(env, "active_task"))

print("\n-- atraso de gatilho: episódio começa em `parado` --")
# Sem passar do atraso, a tarefa ATIVA tem que ser PARADO mesmo que a sorteada não
# seja. É isto que ensina a política a ficar estável antes de receber ordem.
cfg_atraso = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_atraso.scene.num_envs = 32
cfg_atraso.commands["lift_target"].atraso_gatilho_s = (1.9, 2.0)
env_a = ManagerBasedRlEnv(cfg=cfg_atraso, device=DEVICE)
acao_a = torch.zeros(env_a.num_envs, env_a.action_manager.total_action_dim,
                     device=env_a.device)


def pre_gatilho(tarefa: int) -> torch.Tensor:
    """A tarefa ATIVA de `env_a` (atraso 1.9-2.0 s) logo depois do reset."""
    env_a.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_a.task_dist[tarefa] = 1.0
    env_a.reset()
    env_a.step(acao_a)
    return env_a.active_task


# Quem NÃO nasce segurando espera em `parado`. Quem nasce segurando espera em
# `parado c/ caixa` (§4) — senão a caixa escorregaria durante os 2 s de atraso.
check("antes do gatilho, `andar` espera em `parado`",
      bool((pre_gatilho(T.ANDAR) == T.PARADO).all()),
      f"ativa={T.NAMES[int(env_a.active_task[0])]}")
check("antes do gatilho, `botar` espera em `parado c/ caixa`",
      bool((pre_gatilho(T.BOTAR) == T.PARADO_CAIXA).all()),
      f"ativa={T.NAMES[int(env_a.active_task[0])]}")

print("\n-- preenchimento por tarefa (§9) --")
cfg_t = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_t.scene.num_envs = 32
cfg_t.commands["lift_target"].atraso_gatilho_s = (0.0, 0.0)   # dispara já
env_t = ManagerBasedRlEnv(cfg=cfg_t, device=DEVICE)
acao_t = torch.zeros(env_t.num_envs, env_t.action_manager.total_action_dim,
                     device=env_t.device)
meta_t = env_t.command_manager.get_term("lift_target")

for tarefa in range(T.NUM_TASKS):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset()
    env_t.step(acao_t)
    cmd = meta_t.command
    nome = T.NAMES[tarefa]

    check(f"{nome}: one-hot certo",
          bool((cmd[:, C.ONEHOT].argmax(dim=-1) == tarefa).all()),
          f"argmax={cmd[:, C.ONEHOT].argmax(dim=-1)[0].item()}")

    robo = env_t.scene["robot"].data.root_link_pos_w
    caixa = env_t.scene["box"].data.root_link_pos_w
    alvo = cmd[:, C.ALVO]
    dist_robo = (alvo - robo).norm(dim=-1).mean().item()
    dist_caixa = (alvo - caixa).norm(dim=-1).mean().item()

    if tarefa in T.PARADAS:
        # F5: alvo = posição ATUAL do robô, reavaliada a cada passo -> target_pos_b = 0
        check(f"{nome}: alvo_pos = o próprio robô (target_pos_b = 0)",
              dist_robo < 1e-5, f"|alvo−robo|={dist_robo:.2e}")
    elif tarefa in (T.PEGAR, T.REORIENTAR):
        check(f"{nome}: alvo_pos = a caixa", dist_caixa < 1e-5,
              f"|alvo−caixa|={dist_caixa:.2e}")
    elif tarefa in T.ANDA:
        esperado = T.LEVELS["distancia"][T.AXES[tarefa]["distancia"]]
        check(f"{nome}: alvo_pos a {esperado} m do spawn",
              abs(dist_robo - esperado) < 0.05, f"deu {dist_robo:.3f} m")
    elif tarefa == T.BOTAR:
        topo = ACTIVE.scene.shelf_top + ACTIVE.scene.box_half[2]
        check(f"{nome}: alvo_pos no topo da prateleira",
              abs(alvo[:, 2].mean().item() - topo) < 0.02,
              f"z={alvo[:, 2].mean().item():.3f} (esperado {topo:.3f})")

    tem_orientacao = cmd[:, C.FACE].abs().sum(dim=-1) > 0
    if tarefa == T.REORIENTAR:
        check(f"{nome}: face_alvo e dir_alvo presentes", bool(tem_orientacao.all()))
        check(f"{nome}: dir_alvo unitário",
              bool((cmd[:, C.DIR].norm(dim=-1) - 1.0).abs().max() < 1e-4))
        # O INVARIANTE do eixo de giro: o erro no reset tem que ser EXATAMENTE o
        # ângulo do nível. Se não for, `dir_alvo` saiu de uma pose stale e o
        # currículo de giro não comanda rotação nenhuma em particular (bug 30/07).
        erro = meta_t.erro_angulo_deg()
        nivel0 = T.LEVELS["giro"][T.AXES[T.REORIENTAR]["giro"]]
        check(f"{nome}: erro = o ângulo do nível ({nivel0}°)",
              bool((erro - nivel0).abs().max() < 0.5),
              f"min={erro.min().item():.2f}° max={erro.max().item():.2f}°")
    else:
        check(f"{nome}: sem comando de orientação", bool((~tem_orientacao).all()))

print("\n-- T3: observação nova --")
dims = env.observation_manager.group_obs_term_dim["actor"]
nomes = list(cfg.observations["actor"].terms)
por_nome = dict(zip(nomes, [d[0] for d in dims]))
for termo, tam in (("box_rot_b", 6), ("face_alvo", 3), ("dir_alvo", 3),
                   ("task_onehot", 8)):
    check(f"{termo} = {tam} números", por_nome.get(termo) == tam,
          f"deu {por_nome.get(termo)}")
check("target_pos_b escalado ÷2.0",
      cfg.observations["actor"].terms["target_pos_b"].scale == 0.5)
check("ator e crítico com os MESMOS termos",
      nomes == list(cfg.observations["critic"].terms)[:len(nomes)]
      or set(nomes) <= set(cfg.observations["critic"].terms),
      "o crítico só pode ter termos A MAIS")
# A 6D tem que ser ortonormal: se as colunas deixarem de ser unitárias/ortogonais,
# a representação parou de descrever uma rotação e a rede recebe lixo.
from g1_multitask import observations as O  # noqa: E402

r6 = O.object_rot_b(env, "box")
check("box_rot_b: 2 colunas unitárias",
      bool((r6[:, :3].norm(dim=-1) - 1).abs().max() < 1e-4
           and (r6[:, 3:].norm(dim=-1) - 1).abs().max() < 1e-4))
check("box_rot_b: colunas ortogonais",
      bool((r6[:, :3] * r6[:, 3:]).sum(dim=-1).abs().max() < 1e-4))

print("\n-- twist derivado --")
for tarefa, espera_movimento in ((T.PARADO, False), (T.ANDAR, True)):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset()
    env_t.step(acao_t)
    v = env_t.command_manager.get_term("twist").command[:, :2].norm(dim=-1).mean().item()
    nome = T.NAMES[tarefa]
    if espera_movimento:
        # 0.3 m de destino com d_morto 0.25 -> dentro da rampa, v pequena mas > 0
        check(f"{nome}: twist manda andar", v > 0.0, f"|v|={v:.3f} m/s")
    else:
        check(f"{nome}: twist manda parar", v < 1e-5, f"|v|={v:.2e} m/s")

# ------------------------------------- T4: congelamento do normalizador + play
print("\n-- T4: runner e congelamento do normalizador --")
from dataclasses import asdict  # noqa: E402

from g1_multitask.runner import CANAIS_CONGELADOS, MultitaskRunner  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from rsl_rl.modules.normalization import EmpiricalNormalization  # noqa: E402

# CONTRAPROVA: sem congelamento, canal constante em 0 recebe _std = 0 e, ao acender,
# entra 100x amplificado. É este o modo de falha que o congelamento fecha.
nrm = EmpiricalNormalization(shape=8)
for _ in range(50):
    x = torch.randn(64, 8)
    x[:, 3] = 0.0                     # canal constante, como o `face_alvo` na Fase 0
    nrm.update(x)
std_const = nrm._std[0, 3].item()
entraria = (1.0 - nrm._mean[0, 3].item()) / (std_const + 1e-2)
check("contraprova: canal constante ganha _std = 0", std_const == 0.0,
      f"_std={std_const:.6f} -> 1.0 entraria como {entraria:.1f}")

# runner instanciado como o `play` faz: env ENVELOPADO e SEM log_dir
# (`play.py:183` envelopa, `:204` instancia sem log_dir). O runner só alcança o
# observation_manager por `self.env.unwrapped`, então o envelope é obrigatório.
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

rl_cfg = load_rl_cfg(g1_multitask.TASK_ID)
env_wrap = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
runner = MultitaskRunner(env_wrap, asdict(rl_cfg), None, DEVICE)
check("runner monta com log_dir=None (requisito do play)", True)

idx_ator = runner.idx_congelados["actor"]
check("17 canais congelados no ator", idx_ator.numel() == 17,
      f"deu {idx_ator.numel()} — {list(CANAIS_CONGELADOS)}")
# os índices são DERIVADOS: têm que casar com a posição real dos termos
am = env.observation_manager
pos, esperados = 0, []
for nome, dim in zip(am.active_terms["actor"], am.group_obs_term_dim["actor"]):
    if nome in CANAIS_CONGELADOS:
        esperados.extend(range(pos, pos + int(dim[0])))
    pos += int(dim[0])
check("índices batem com a ordem real dos termos",
      idx_ator.tolist() == esperados, f"{idx_ator.tolist()[:4]}...")

# 50 updates com uma tarefa só: os canais de comando ficam constantes
env.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env.task_dist[T.PARADO] = 1.0
env.reset()
ator = runner.alg._raw_actor
for _ in range(50):
    env.step(acao)
    ator.update_normalization(env.observation_manager.compute())
std = ator.obs_normalizer._std[0]
media = ator.obs_normalizer._mean[0]
check("canais de comando: _std continua 1.0 exato",
      bool((std[idx_ator] == 1.0).all()), f"min={std[idx_ator].min().item():.6f}")
check("canais de comando: _mean continua 0.0 exato",
      bool((media[idx_ator] == 0.0).all()), f"max|mean|={media[idx_ator].abs().max():.2e}")
outros = torch.tensor([i for i in range(std.numel()) if i not in set(idx_ator.tolist())])
check("propriocepção NÃO foi congelada (o `until` global seria errado)",
      bool((std[outros] != 1.0).any()),
      f"{int((std[outros] != 1.0).sum())} de {outros.numel()} canais aprenderam")

# ------------------------------------------- T5: cena e randomização de domínio
print("\n-- T5: mobília fora do grupo 0 e DR de startup --")
grupos = {}
for nome in ("box", "table"):
    m = cfg.scene.entities[nome].spec_fn().compile()
    grupos[nome] = sorted(set(int(g) for g in m.geom_group))
check("caixa fora do grupo 0", grupos["box"] == [2], f"grupos={grupos['box']}")
check("prateleira fora do grupo 0", grupos["table"] == [2], f"grupos={grupos['table']}")

sensores = {sv.name: sv for sv in (cfg.scene.sensors or ())}
check("feet_ground_contact casa qualquer contato",
      sensores["feet_ground_contact"].secondary is None)
# os sensores da caixa casam por NOME de geom, então o regrupo não pode tê-los quebrado
for nome in ("palm_L_box", "box_support", "body_table"):
    check(f"sensor {nome} preservado", nome in sensores)

for nome, chave, faixa in (("foot_friction", "ranges", (0.3, 1.2)),
                           ("encoder_bias", "bias_range", (-0.015, 0.015))):
    ev = cfg.events.get(nome)
    check(f"DR {nome} presente e startup",
          ev is not None and ev.mode == "startup", "" if ev else "AUSENTE")
    if ev:
        check(f"DR {nome} com o range do fabricante",
              tuple(ev.params[chave]) == faixa, f"{tuple(ev.params[chave])}")
check("DR foot_friction com geoms do pé preenchidos",
      len(cfg.events["foot_friction"].params["asset_cfg"].geom_names) == 14,
      "14 geoms, preenchidos por robô pelo fabricante")

# O `base_com` tem que estar AUSENTE do config registrado: `dr.body_com_evento`
# corrompe memória em CPU e em GPU (item 0, resolvido 30/07 desligando). Se ele
# reaparecer aqui, alguém religou sem rodar o A/B — e o sintoma em GPU é um
# `illegal memory access` cujo traceback aponta pro lugar ERRADO.
cfg_bruto = load_env_cfg(g1_multitask.TASK_ID)
check("base_com AUSENTE do config registrado (corrompe CPU e GPU)",
      "base_com" not in cfg_bruto.events,
      "ver knobs.DR.base_com — religar só depois do A/B passar")
check("as outras 2 DR seguem ligadas",
      "foot_friction" in cfg_bruto.events and "encoder_bias" in cfg_bruto.events)

# Com DR ligada o mundo tem que continuar finito. O `nonfinite` da base é a rede de
# segurança, mas se ele disparar no 1º step o problema é a DR, não o robô — foi
# exatamente essa a cicatriz de 2026-07-15 que fez a física virar pyramidal/1.0.
for _ in range(5):
    env.step(acao)
pos = env.scene["robot"].data.root_link_pos_w
check("5 steps com DR ligada sem NaN", bool(torch.isfinite(pos).all()))

# ------------------------------------------------------- T6: locomoção de volta
print("\n-- T6: os 5 rewards de marcha, apontados pro twist --")
from g1_multitask import rewards as R  # noqa: E402

kr = ACTIVE.reward
for nome, peso in (("track_linear_velocity", kr.track_linear_velocity),
                   ("track_angular_velocity", kr.track_angular_velocity),
                   ("foot_clearance", kr.foot_clearance),
                   ("foot_swing_height", kr.foot_swing_height),
                   ("soft_landing_feet", kr.soft_landing_feet),
                   ("soft_landing_table", kr.soft_landing_table),
                   ("arm_vel", kr.arm_vel),
                   ("joint_acc", kr.joint_acc)):
    t = cfg.rewards.get(nome)
    check(f"{nome} presente com peso {peso}",
          t is not None and t.weight == peso,
          "AUSENTE" if t is None else f"peso={t.weight}")

# Os termos de MARCHA têm que ler o `"twist"`. Um deles apontando pro `lift_target`
# leria `alvo_pos` como se fosse velocidade — não daria erro, só treinaria errado.
# Os termos de TAREFA leem `lift_target` de propósito: é lá que está `alvo_pos`.
MARCHA = ("track_linear_velocity", "track_angular_velocity", "foot_clearance",
          "foot_swing_height", "soft_landing_feet")
erradas = [n for n in MARCHA
           if cfg.rewards[n].params.get("command_name", "twist") != "twist"]
check("todo reward de marcha aponta pro `twist`", not erradas, str(erradas))

check("track_linear_velocity é a variante com freio de z",
      cfg.rewards["track_linear_velocity"].func is R.track_linear_velocity_freio_z)
check("os dois soft_landing usam sensores DIFERENTES",
      cfg.rewards["soft_landing_feet"].params["sensor_name"]
      != cfg.rewards["soft_landing_table"].params["sensor_name"],
      f"{cfg.rewards['soft_landing_feet'].params['sensor_name']} vs "
      f"{cfg.rewards['soft_landing_table'].params['sensor_name']}")

# Pesos da §14 nos herdados
for nome, peso in (("upright", kr.upright), ("action_rate_l2", kr.action_rate_l2),
                   ("feet_slip", kr.feet_slip), ("dof_pos_limits", kr.dof_pos_limits)):
    check(f"{nome} com o peso da §14 ({peso})", cfg.rewards[nome].weight == peso,
          f"deu {cfg.rewards[nome].weight}")

# Freio de z: dentro do d_morto o erro em z não pode contar. Com `parado` (alvo = a
# própria posição) o robô está sempre dentro, então o termo tem que ficar > o que
# ficaria com a punição de z ativa.
env.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env.task_dist[T.PARADO] = 1.0
env.reset()
env.step(acao)
tw = env.command_manager.get_term("twist")
check("parado está dentro do d_morto", bool(tw.dentro_do_morto().all()))
v = R.track_linear_velocity_freio_z(env, std=0.5, command_name="twist")
check("freio de z: termo finito e em (0,1]", bool(((v > 0) & (v <= 1.0)).all()),
      f"min={v.min().item():.3f} max={v.max().item():.3f}")

# todos os termos executam 1x sem erro de shape (item da lista da §15)
env.step(acao)
n_termos = len(cfg.rewards)
check(f"os {n_termos} termos de reward executam 1x", True, f"{n_termos} termos")

# ------------------------------------------------- T7: gates e as 3 posturas
print("\n-- T7: gate por máscara e postura em 3 escopos --")
check("o `posture` único do base_env saiu", "posture" not in cfg.rewards)
for nome in ("posture_parado", "posture_anda", "posture_manip", "posture_carrega"):
    t = cfg.rewards.get(nome)
    check(f"{nome} presente, gateado, peso {kr.postura}",
          t is not None and t.func is R.gated and t.weight == kr.postura,
          "AUSENTE" if t is None else f"func={t.func.__name__} peso={t.weight}")
# a cobertura das 3 posturas tem que ser EXATA: toda tarefa em uma, nenhuma em duas
cobertura: dict[int, int] = {t: 0 for t in range(T.NUM_TASKS)}
for nome in ("posture_parado", "posture_anda", "posture_manip", "posture_carrega"):
    for t in cfg.rewards[nome].params["tasks"]:
        cobertura[t] += 1
check("as 4 posturas cobrem cada tarefa exatamente 1x",
      all(v == 1 for v in cobertura.values()),
      str({T.NAMES[t]: v for t, v in cobertura.items() if v != 1}))
check("std_walking colhido do fabricante (dict por junta)",
      len(cfg.rewards["posture_anda"].params["std"]) > 5,
      f"{len(cfg.rewards['posture_anda'].params['std'])} entradas por junta")
check("posture_parado é corpo todo, manip é perna+cintura",
      cfg.rewards["posture_parado"].params["asset_cfg"].joint_names == [".*"]
      and cfg.rewards["posture_manip"].params["asset_cfg"].joint_names
      != [".*"])

check("com_balance OFF no andar",
      set(cfg.rewards["com_balance"].params["tasks"]) == set(T.exceto(*T.ANDA)))
check("box_shake OFF no reorientar",
      set(cfg.rewards["box_shake"].params["tasks"]) == set(T.exceto(T.REORIENTAR)))
check("table_contact e back_penalty sem gate",
      cfg.rewards["table_contact"].func is not R.gated
      and cfg.rewards["back_penalty"].func is not R.gated)

# O gate tem que dar ZERO fora do escopo. Sem isto, o termo pontuaria na tarefa
# errada e nenhum teste de peso pegaria.
#
# Lido do `reward_manager`, não do cfg: o manager deepcopia o cfg e SUBSTITUI
# `func` pela instância já construída (`manager_base.py:141-147`). Chamar
# `R.gated(...)` direto construiria uma instância nova em vez de avaliar o termo.
print("\n-- o gate realmente zera fora do escopo --")
# O invariante é `gated == inner × máscara`, não "gated != 0 dentro do escopo".
# Testar por não-zero seria VAZIO quando o termo interno é legitimamente 0 — o
# `com_balance` com o robô em pé é exatamente esse caso: ele pune CoM à frente dos
# pés, e em pé o CoM está sobre os pés.
GATEADOS = ("com_balance", "box_shake", "posture_parado", "posture_anda",
            "posture_manip", "posture_carrega")
for tarefa in range(T.NUM_TASKS):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset()
    env_t.step(acao_t)
    erros = []
    for nome_termo in GATEADOS:
        termo = env_t.reward_manager.get_term_cfg(nome_termo)
        kw = {k: v for k, v in termo.params.items()
              if k not in ("inner", "tasks", "gate_command")}
        interno = termo.func._inner(env_t, **kw)
        v = termo.func(env_t, **termo.params)
        dentro = tarefa in termo.params["tasks"]
        esperado = interno if dentro else torch.zeros_like(interno)
        if not bool(torch.allclose(v, esperado)):
            erros.append(f"{nome_termo}({'dentro' if dentro else 'fora'})")
    check(f"`{T.NAMES[tarefa]}`: gated == inner × máscara nos {len(GATEADOS)} termos",
          not erros, str(erros))

# ------------------------- T8 + T8b: recompensas de tarefa e spawn segurando
print("\n-- T8: os 7 termos de tarefa, com os gates da §6b --")
GATES_ESPERADOS = {
    "lift": (T.PEGAR,),
    "reaching": (T.REORIENTAR, T.PEGAR, T.BOTAR),
    "grasp": (T.PEGAR,),
    "box_at_peito": (T.PEGAR, T.PARADO_CAIXA, T.ANDAR_CAIXA),
    "box_at_prateleira": (T.BOTAR,),
    "orienta_face": (T.REORIENTAR,),
    "hold_still": (T.PARADO_CAIXA, T.ANDAR_CAIXA, T.PEGAR),
}
for nome, esperado in GATES_ESPERADOS.items():
    t = cfg.rewards.get(nome)
    check(f"{nome}: gate = {[T.NAMES[x] for x in esperado]}",
          t is not None and set(t.params["tasks"]) == set(esperado),
          "AUSENTE" if t is None else str([T.NAMES[x] for x in t.params["tasks"]]))
check("botar_fracao_solta = 0.0 (kernel puro, como a §4 especifica)",
      ACTIVE.reward.botar_fracao_solta == 0.0,
      f"deu {ACTIVE.reward.botar_fracao_solta}")

print("\n-- T8b: spawn segurando --")
check("reset_segurando é o ÚLTIMO evento de reset",
      [k for k, v in cfg.events.items() if v.mode == "reset"][-1] == "reset_segurando",
      str([k for k, v in cfg.events.items() if v.mode == "reset"]))
check("sorteio da tarefa é curriculum term (roda antes dos eventos)",
      "orquestrador" in cfg.curriculum, str(list(cfg.curriculum)))

for tarefa in range(T.NUM_TASKS):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset()
    cx = env_t.scene["box"].data.root_link_pos_w
    pelve = env_t.scene["robot"].data.root_link_pos_w
    rel_z = (cx[:, 2] - pelve[:, 2]).mean().item()
    segura = tarefa in T.SPAWN_SEGURANDO
    nome = T.NAMES[tarefa]
    if segura:
        # a caixa nasce no alvo do peito: +0.15 m acima da pelve (§14)
        check(f"{nome}: caixa nasce no peito (+0.15 da pelve)",
              abs(rel_z - 0.15) < 0.03, f"rel_z={rel_z:+.3f}")
        check(f"{nome}: pré-gatilho é `parado c/ caixa`",
              bool((pre_gatilho(tarefa) == T.PARADO_CAIXA).all()),
              f"ativa={T.NAMES[int(env_a.active_task[0])]}")
    else:
        check(f"{nome}: caixa nasce na prateleira, NÃO na mão",
              rel_z < 0.0, f"rel_z={rel_z:+.3f}")

# O achado que motivou a T8b: antes dela, TODOS os termos de tarefa davam 0.0 no
# reset das 3 tarefas c/ caixa -> nenhum caminho de aquisição.
#
# Medido em `env_a` (atraso de ~2 s), porque é a fase PRÉ-GATILHO que interessa: as
# 3 esperam em `parado c/ caixa`, e o `box_at_peito` é gateado nela. No `botar`, ao
# disparar o gatilho ele gateia OFF e o `box_at_prateleira` gateia ON — o
# encadeamento sai do próprio one-hot, sem máquina de fases.
for tarefa in T.SPAWN_SEGURANDO:
    pre_gatilho(tarefa)
    t = env_a.reward_manager.get_term_cfg("box_at_peito")
    v = t.func(env_a, **t.params).mean().item()
    check(f"{T.NAMES[tarefa]}: box_at_peito > 0 antes do gatilho (há gradiente)",
          v > 0.1, f"deu {v:.3f}")

# `botar`: a caixa nasce na MÃO, longe da prateleira -> termo baixo. É isso que
# tira o vale sem precisar de fator de preensão (§4).
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.BOTAR] = 1.0
env_t.reset(); env_t.step(acao_t)
t = env_t.reward_manager.get_term_cfg("box_at_prateleira")
v = t.func(env_t, **t.params).mean().item()
check("botar: caixa longe da prateleira no spawn (sem vale)", v < 0.05,
      f"box_at_prateleira={v:.4f} — sobe conforme aproxima e baixa")

print("\n-- o kernel de 2 escalas do reorientar --")
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.REORIENTAR] = 1.0
env_t.reset(); env_t.step(acao_t)
t = env_t.reward_manager.get_term_cfg("orienta_face")
v = t.func(env_t, **t.params).mean().item()
# Com escala única de 5° o nível 15° daria exp(-(15/5)^2) = 1.2e-4: gradiente nenhum.
check("orienta_face tem gradiente no nível 15° (escala grossa faz o trabalho)",
      v > 0.2, f"deu {v:.3f}  (com escala única de 5° daria 1.2e-4)")

# ------------------------------------------------ T9: terminações e "de pé"
print("\n-- T9: terminações --")
from g1_multitask import terminations as MTT  # noqa: E402

check("6 terminações no total", len(cfg.terminations) == 6, str(list(cfg.terminations)))
check("fell_over em 70° exatos (do fabricante)",
      abs(cfg.terminations["fell_over"].params["limit_angle"] - 1.2217304763960306) < 1e-9)
check("largou gateada só em `c/ caixa`",
      set(cfg.terminations["largou"].params["tasks"]) == set(T.COM_CAIXA),
      str([T.NAMES[x] for x in cfg.terminations["largou"].params["tasks"]]))
check("caixa_caiu gateada só no `reorientar`",
      set(cfg.terminations["caixa_caiu"].params["tasks"]) == {T.REORIENTAR})
check("fora_da_area sem gate (vale em todas)",
      "tasks" not in cfg.terminations["fora_da_area"].params)


def _poe_caixa_em(e, z):
    """Teleporta a caixa pra altura z, parada, e atualiza a cinemática."""
    cx = e.scene["box"]
    pos = cx.data.root_link_pos_w.clone()
    pos[:, 2] = z
    e.scene["box"].write_root_state_to_sim(torch.cat(
        [pos, cx.data.root_link_quat_w, torch.zeros(e.num_envs, 6, device=e.device)],
        dim=-1))
    e.sim.forward()


# A distinção `pegar` × `carregar` é o ponto do desenho: mesma condição física,
# veredito oposto, porque só quando carregar é o estado EXIGIDO é que largar é falha.
for tarefa, esperado in ((T.PARADO_CAIXA, True), (T.ANDAR_CAIXA, True),
                         (T.PEGAR, False), (T.BOTAR, False)):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset(); env_t.step(acao_t)
    _poe_caixa_em(env_t, 0.20)                       # abaixo dos 0.30 do §14
    tc = env_t.termination_manager.get_term_cfg("largou")
    v = bool(tc.func(env_t, **tc.params).any())
    check(f"caixa a 0.20 m em `{T.NAMES[tarefa]}`: "
          f"{'termina' if esperado else 'NÃO termina'}", v == esperado, f"deu {v}")

# O caso que mata a solução certa se for feito errado: no `reorientar`, erguer e
# rolar deixa a caixa NO AR acima da prateleira. Isso não pode terminar.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.REORIENTAR] = 1.0
env_t.reset(); env_t.step(acao_t)
tc = env_t.termination_manager.get_term_cfg("caixa_caiu")
_poe_caixa_em(env_t, 1.00)
check("reorientar: caixa a 1.00 m (erguida) NÃO termina",
      not bool(tc.func(env_t, **tc.params).any()))
_poe_caixa_em(env_t, 0.10)
check("reorientar: caixa a 0.10 m (abaixo da prateleira) TERMINA",
      bool(tc.func(env_t, **tc.params).all()))

print("\n-- item 19: definição de `de pé` --")
env_t.reset(); env_t.step(acao_t)
z = env_t.scene["robot"].data.root_link_pos_w[:, 2].mean().item()
check("robô em pé no reset passa em `de_pe`",
      bool(MTT.de_pe(env_t, ACTIVE.tolerancia.de_pe_z,
                     ACTIVE.tolerancia.de_pe_tilt_rad).all()),
      f"z_pelve={z:.3f} m (limite {ACTIVE.tolerancia.de_pe_z})")
check("`de_pe` é mais exigente que `fell_over`",
      ACTIVE.tolerancia.de_pe_tilt_rad < cfg.terminations["fell_over"].params["limit_angle"],
      f"20° contra 70° — o fell_over daria `de pé` pra um robô dobrado a 65°")

# ----------------------------------------- T10: sucesso em env.success_buf
print("\n-- T10: sucesso é FÍSICO e fora do reward manager --")
check("sucesso registrado com reduce='last'",
      cfg.metrics["sucesso"].reduce == "last",
      "média por passo diria 0.3 pra um episódio com sucesso nos últimos 30%")
check("deriva do parado é métrica, NÃO terminação nem sucesso (F3)",
      "deriva_parado" in cfg.metrics
      and "deriva" not in " ".join(cfg.terminations))
check("env.success_buf existe antes de qualquer step", hasattr(env_t, "success_buf"))
# O que faz a Categoria A ser grátis: nenhum termo de reward alimenta o sucesso.
import inspect  # noqa: E402

fonte_sucesso = inspect.getsource(__import__("g1_multitask.metrics",
                                             fromlist=["Sucesso"]).Sucesso)
check("sucesso NÃO lê reward_manager (senão peso viraria Categoria C)",
      "reward_manager" not in fonte_sucesso)

sucesso = env_t.metrics_manager._term_cfgs[
    list(cfg.metrics).index("sucesso")].func
tol = ACTIVE.tolerancia
EXIGENCIA = {T.PARADO: 0.0, T.ANDAR: tol.sustenta_andar_s,
             T.PEGAR: tol.sustenta_pegar_s, T.BOTAR: tol.sustenta_botar_s,
             T.REORIENTAR: tol.sustenta_reorienta_s,
             T.PARADO_CAIXA: tol.sustenta_pegar_s,
             T.ANDAR_CAIXA: tol.sustenta_andar_s}
for tarefa, esperado in EXIGENCIA.items():
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[tarefa] = 1.0
    env_t.reset(); env_t.step(acao_t)
    s = sucesso._exigencia_s(env_t).max().item()
    check(f"{T.NAMES[tarefa]}: exige {esperado} s seguidos (§14)",
          abs(s - esperado) < 1e-6, f"deu {s}")

# A cadeia inteira do `parado c/ caixa` (caixa no peito E preensão E de pé) tem que
# ser SATISFAZÍVEL: nasce segurando, então o passo 1 já cumpre. E tem que ZERAR
# quando quebra — a caixa escorrega com ação nula, e é isso que o robô vai aprender.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.PARADO_CAIXA] = 1.0
env_t.reset()
env_t.step(acao_t)
c1 = sucesso._contador.max().item()
for _ in range(20):
    env_t.step(acao_t)
c2 = sucesso._contador.max().item()
check("parado c/ caixa: condição satisfazível no passo 1", c1 > 0.0, f"{c1:.2f} s")
check("contador ZERA quando a condição quebra", c2 == 0.0,
      f"{c2:.2f} s — a caixa escorregou, que é o que a tarefa ensina a evitar")

# O sucesso do `parado` depende de `time_outs`, e um episódio terminado por falha
# não conta. Testado com episódio curto pra o timeout ser alcançável.
cfg_curto = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_curto.scene.num_envs = 8
cfg_curto.episode_length_s = 1.0
cfg_curto.commands["lift_target"].atraso_gatilho_s = (0.0, 0.0)
env_c = ManagerBasedRlEnv(cfg=cfg_curto, device=DEVICE)
env_c.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_c.task_dist[T.PARADO] = 1.0
env_c.reset()
acao_c = torch.zeros(env_c.num_envs, env_c.action_manager.total_action_dim,
                     device=env_c.device)
viu_timeout = False
for _ in range(60):
    env_c.step(acao_c)
    if bool(env_c.termination_manager.time_outs.any()):
        viu_timeout = True
        break
check("parado: timeout alcançado com episódio curto", viu_timeout)
check("parado: sucesso só onde NÃO houve terminação por falha",
      bool((env_c.success_buf.bool()
            <= (env_c.termination_manager.time_outs
                & ~env_c.termination_manager.terminated)).all()),
      f"success={env_c.success_buf.mean():.2f}")

# ------------------------------- T11: observabilidade por tarefa x termo
print("\n-- T11: contribuição por tarefa × termo --")
cfg_obs = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_obs.scene.num_envs = 64
cfg_obs.commands["lift_target"].atraso_gatilho_s = (0.0, 0.0)
env_o = ManagerBasedRlEnv(cfg=cfg_obs, device=DEVICE)
acao_o = torch.zeros(env_o.num_envs, env_o.action_manager.total_action_dim,
                     device=env_o.device)
env_o.task_dist = torch.ones(T.NUM_TASKS, device=DEVICE)     # as 7 sorteadas
env_o.reset()
PASSOS = 30
for _ in range(PASSOS):
    env_o.step(acao_o)

check("acumulador conta todos os (env, passo)",
      abs(float(env_o.contrib_cont.sum()) - PASSOS * env_o.num_envs) < 1e-3,
      f"{float(env_o.contrib_cont.sum()):.0f} de {PASSOS * env_o.num_envs}")
check("as 7 tarefas foram amostradas",
      bool((env_o.contrib_cont > 0).all()),
      str([int(x) for x in env_o.contrib_cont]))
check("um nome de termo por coluna",
      len(env_o.contrib_nomes) == env_o.contrib_soma.shape[1] == len(cfg.rewards),
      f"{len(env_o.contrib_nomes)} nomes, {env_o.contrib_soma.shape[1]} colunas")

# RECONCILIAÇÃO: a soma das contribuições de uma tarefa tem que dar o total dela.
# Se não fechar, a máscara do acumulador está errada — e aí o relatório entre blocos
# apontaria o termo errado como dominante.
soma_antes = env_o.contrib_soma.clone()
cont_antes = env_o.contrib_cont.clone()
rel = env_o.curriculum_manager.get_term_cfg("contrib").func
log = rel(env_o, torch.arange(env_o.num_envs, device=DEVICE))
check("relatório emitiu chaves", len(log) > 0, f"{len(log)} chaves")
erros = []
for t in range(T.NUM_TASKS):
    chave_total = f"{T.NAMES[t]}/_total"
    if chave_total not in log:
        continue
    esperado = float(soma_antes[t].sum() / max(float(cont_antes[t]), 1.0))
    if abs(float(log[chave_total]) - esperado) > 1e-4:
        erros.append(T.NAMES[t])
check("cada `_total` = soma das contribuições daquela tarefa", not erros, str(erros))
check("relatório ZERA a matriz depois de emitir",
      float(env_o.contrib_soma.abs().sum()) == 0.0
      and float(env_o.contrib_cont.sum()) == 0.0)

# guarda de ruído: tarefa quase não sorteada não entra no relatório
env_o.contrib_soma.zero_(); env_o.contrib_cont.zero_()
env_o.contrib_cont[T.PARADO] = 10_000.0        # só uma tarefa com amostra
log2 = rel(env_o, torch.arange(env_o.num_envs, device=DEVICE))
tarefas_no_log = {k.split("/")[0] for k in log2}
check("guarda de ruído: só tarefa com amostra suficiente entra",
      tarefas_no_log == {T.NAMES[T.PARADO]}, str(tarefas_no_log))

# o termo mais pesado em `andar` tem que ser identificável — é o que se olha
# entre blocos pra decidir ajuste de Categoria A
env_o.contrib_soma.zero_(); env_o.contrib_cont.zero_()
env_o.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_o.task_dist[T.ANDAR] = 1.0
env_o.reset()
for _ in range(20):
    env_o.step(acao_o)
media = env_o.contrib_soma[T.ANDAR] / max(float(env_o.contrib_cont[T.ANDAR]), 1.0)
ordem = torch.argsort(media.abs(), descending=True)[:3]
print("        3 maiores contribuições em `andar`: "
      + ", ".join(f"{env_o.contrib_nomes[i]}={media[i]:+.3f}" for i in ordem.tolist()))
check("há termo dominante identificável em `andar`",
      float(media.abs().max()) > 0.0)

# -------------------------- T13/T14: a sequência do currículo reproduz o desenho
print("\n-- T13/T14: orquestrador --")
from g1_multitask.sim_curriculo import simula  # noqa: E402

orq_sim, info_sim = simula(num_envs=64)
check("simulação: 60 destravamentos", orq_sim.eventos == 60, f"deu {orq_sim.eventos}")
check("simulação: 7 tarefas abriram", len(orq_sim.abertas) == T.NUM_TASKS)
check("simulação: push completo antes do `andar`",
      info_sim["push_completo_em"] == 4
      and info_sim["push_completo_em"] < info_sim["abertura_em"][T.ANDAR])
check("simulação: `pegar` antes de `reorientar` (cadeia crítica primeiro)",
      info_sim["abertura_em"][T.PEGAR] < info_sim["abertura_em"][T.REORIENTAR])
orq_env = env_t.curriculum_manager.get_term_cfg("orquestrador").func
check("orquestrador tem state_dict e load_state_dict",
      hasattr(orq_env, "state_dict") and hasattr(orq_env, "load_state_dict"),
      "sem isto o currículo volta ao nível 0 a cada um dos 10-15 resumes")
check("14 células (13 tarefa×eixo + push)", len(orq_env.celulas) == 14,
      f"deu {len(orq_env.celulas)}")

# T15 (treinar, salvar, retomar) mora em `smoke_resume.py`: ele instancia PPO de
# verdade e leva minutos, o que quebraria a promessa deste arquivo de rodar em
# segundos. Rode os DOIS antes de submeter.
print("\n-- T15: em `smoke_resume.py` (leva minutos, PPO de verdade) --")

# ------------------------------------------------------------------- veredito
print(f"\n{'=' * 60}")
if falhas:
    print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
    sys.exit(1)
print("smoke do multi-tarefa: tudo OK")
