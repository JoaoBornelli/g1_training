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
import inspect
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import g1_multitask  # noqa: F401  (o import dispara o register_mjlab_task)
from g1_multitask import observations as obs
from g1_multitask import tasks as T
from g1_multitask.configs import ACTIVE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

DEVICE = "cpu"
NUM_ENVS = 4

# Largura da observação do ATOR. 132 herdada do `g1_training`, menos o bit `phase`
# (F10), mais 23 canais novos: box_rot_b 6 + face_alvo 3 + dir_alvo 3 +
# task_onehot 8 + twist_cmd 3. Mudar esta largura é Categoria C — recomeçar do zero.
# 151 -> 154 na S10, e a rodada de validação é a janela em que isso sai de graça.
OBS_ESPERADA = 154

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
print("\n-- conta dos destravamentos (S11: 54) --")
por_fonte = T.unlock_count()
for fonte, n in por_fonte.items():
    print(f"        {fonte:<14} {n}")
check("total = 54", T.total_unlocks() == 54, f"deu {T.total_unlocks()}")
# Cada linha é derivada dos níveis e do índice inicial. Os valores esperados vêm
# da contagem fechada em 29/07 — se uma linha mudar sozinha, um nível foi mexido.
for fonte, esperado in (
    ("parado", 0), ("andar", 4), ("reorientar", 10), ("pegar", 10),
    ("botar", 10), ("parado_caixa", 4), ("andar_caixa", 6),
    ("push", 4), ("aberturas", 6),
):
    check(f"{fonte} = {esperado}", por_fonte[fonte] == esperado, f"deu {por_fonte[fonte]}")

# ------------------------------------------------- T1: eixos e índices iniciais
print("\n-- S11: escopo do eixo de distância --")
# O eixo `distancia` virou `distancia_andar` e SÓ quem anda o possui. Antes ele
# servia também ao `pegar` e ao `reorientar`, o que fazia manipulação exigir
# locomoção antes de encostar na caixa — duas competências numa medição só.
check("o eixo `distancia` não existe mais", "distancia" not in T.LEVELS)
check("o eixo `heading` virou `rumo`",
      "heading" not in T.LEVELS and "rumo" in T.LEVELS,
      "`heading` significava orientação final; agora é a MARCAÇÃO do alvo")
check("`distancia_andar` começa em 1.0 m",
      T.LEVELS["distancia_andar"][0] == 1.0,
      "o primeiro degrau antigo era 0.3 m, e com `andar_raio` em 0.25 o robô "
      "andava 5 cm e o nível media `parado` com sustentação")
for t in (T.ANDAR, T.ANDAR_CAIXA):
    check(f"{T.NAMES[t]} tem `distancia_andar` no índice 0",
          T.AXES[t].get("distancia_andar") == 0)
for t in (T.PARADO, T.PARADO_CAIXA, T.BOTAR, T.PEGAR, T.REORIENTAR):
    check(f"{T.NAMES[t]} não tem eixo de distância",
          "distancia_andar" not in T.AXES[t])
check("os níveis de `rumo` são ±30°, ±90° e ±180°",
      T.LEVELS["rumo"] == (60.0, 180.0, 360.0), str(T.LEVELS["rumo"]))
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

print("\n-- S8: o gatilho é imediato, sem janela de pré-gatilho --")
# O atraso de U(0, 2 s) SAIU. A tarefa sorteada entra ativa no passo 0.
#
# O que se perde, e fica registrado: o desenho não treina transiente de comando nesta
# rodada. O substituto correto é a troca de tarefa no meio do episódio, adiada de
# propósito. O que se ganha: até 2 s dos 20 s voltam para as células duras.
cfg_atraso = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_atraso.scene.num_envs = 32
env_a = ManagerBasedRlEnv(cfg=cfg_atraso, device=DEVICE)
acao_a = torch.zeros(env_a.num_envs, env_a.action_manager.total_action_dim,
                     device=env_a.device)


def ativa_no_spawn(tarefa: int) -> torch.Tensor:
    """A tarefa ATIVA logo depois do reset. Desde a S8 é a própria sorteada."""
    env_a.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_a.task_dist[tarefa] = 1.0
    env_a.reset()
    env_a.step(acao_a)
    return env_a.active_task


for _t8 in (T.ANDAR, T.BOTAR, T.PEGAR, T.PARADO_CAIXA):
    check(f"`{T.NAMES[_t8]}` está ativa já no passo 0",
          bool((ativa_no_spawn(_t8) == _t8).all()),
          f"ativa={T.NAMES[int(env_a.active_task[0])]}")
check("o knob `atraso_gatilho_s` não existe mais",
      not hasattr(ACTIVE.command, "atraso_gatilho_s"),
      "deixar o campo criaria um número que não governa nada")
check("`disparou` é constante em True",
      bool(env_a.command_manager.get_term("lift_target").disparou.all()),
      "o gate continua no `metrics.Sucesso` porque ele volta a ter conteúdo quando "
      "a troca de tarefa no meio do episódio entrar")

print("\n-- preenchimento por tarefa (§9) --")
cfg_t = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_t.scene.num_envs = 32
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
        esperado = T.LEVELS["distancia_andar"][T.AXES[tarefa]["distancia_andar"]]
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
        # Desde a S5: d_morto 0.05 e banda de frenagem 0.125, portanto d_freio = 0.175.
        # O destino do `andar` nasce a 0.3 m, ou seja FORA da banda — o perfil pede
        # `v_max`. O valor lido no 1º passo é pequeno mesmo assim, porque o limitador
        # de taxa sobe `a_max · dt` por passo a partir de zero.
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
check("20 canais congelados no ator", idx_ator.numel() == 20,
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

# Desde a S9 os dois de rastreio são GATEADOS, então a função de fora é a `gated` e a
# variante real fica em `params["inner"]`.
check("track_linear_velocity é a variante com freio de z",
      cfg.rewards["track_linear_velocity"].params["inner"]
      is R.track_linear_velocity_freio_z)
print("\n-- S9: o piso de sobrevivência sai da manipulação --")
# Sem isto, um robô em pé com comando zero recebia 5.5/passo (track 2+2, upright 1,
# posture 0.5), contra 5.0 de TODOS os termos de tarefa do `pegar` somados: ficar
# parado pagava mais que resolver a tarefa.
for _n9 in ("track_linear_velocity", "track_angular_velocity"):
    _g9 = cfg.rewards[_n9].params.get("tasks", ())
    check(f"`{_n9}` só vale onde locomoção é a tarefa",
          set(_g9) == {T.PARADO, T.ANDAR, T.ANDAR_CAIXA, T.PARADO_CAIXA},
          str(sorted(T.NAMES[t] for t in _g9)))
check("os dois de rastreio continuam LIGADOS no `parado`",
      T.PARADO in cfg.rewards["track_linear_velocity"].params["tasks"],
      "o sucesso do `parado` é sobreviver com comando zero; sem eles ele fica sem "
      "sinal nenhum")
check("existe penalidade de terminação",
      "terminacao" in cfg.rewards and cfg.rewards["terminacao"].weight < 0,
      f"peso={cfg.rewards.get('terminacao') and cfg.rewards['terminacao'].weight}")
check("a penalidade usa `is_terminated`, não `time_out`",
      cfg.rewards["terminacao"].func.__name__ == "is_terminated",
      "punir o fim natural do episódio ensina o robô a morrer")
# ⚠️ O peso passa pelo `scale_by_dt`: −200 vira −4,0 de custo real por queda.
_custo_real = cfg.rewards["terminacao"].weight * 0.02
print(f"        custo REAL de uma queda: {_custo_real:.2f} "
      f"(peso {cfg.rewards['terminacao'].weight} x dt 0.02) — a aceitação da S9 "
      "mede se isto basta")

print("\n-- S10: twist na obs do ator, alvo de posição gateado --")
check("`twist_cmd` está nos DOIS grupos",
      "twist_cmd" in cfg.observations["actor"].terms
      and "twist_cmd" in cfg.observations["critic"].terms)
check("`twist_cmd` está em CANAIS_CONGELADOS",
      "twist_cmd" in CANAIS_CONGELADOS,
      "na Fase 0 só o `parado` roda e o twist fica constante em zero -> _std=0 -> "
      "quando o `andar` abre o valor entra ~100x amplificado")
check("`target_pos_b` continua em CANAIS_CONGELADOS",
      "target_pos_b" in CANAIS_CONGELADOS,
      "depois da S10 o ator o vê preenchido em 2 de 7 tarefas; congelar deixou de "
      "ser precaução e virou necessidade")
check("o ATOR usa o `target_pos_b` gateado",
      cfg.observations["actor"].terms["target_pos_b"].func is obs.target_pos_b_gateado)
check("o CRÍTICO mantém o alvo cheio, inclusive no `andar`",
      cfg.observations["critic"].terms["target_pos_b"].func
      is not obs.target_pos_b_gateado,
      "o retorno depende de `chegou`; sem o alvo o crítico não distingue 5 cm de 2 m")
_zeradas = set(cfg.observations["actor"].terms["target_pos_b"].params["tarefas_zeradas"])
check("o alvo é zerado nas 5 tarefas sem waypoint",
      _zeradas == {T.PARADO, T.ANDAR, T.PARADO_CAIXA, T.ANDAR_CAIXA, T.PEGAR},
      str(sorted(T.NAMES[t] for t in _zeradas)))

print("\n-- S12/S13/S14/S15: cena, DR e diagnóstico --")
check("a prateleira tem jitter de xy",
      ACTIVE.scene.table_jitter_xy > 0.0, f"{ACTIVE.scene.table_jitter_xy} m")
check("o jitter de yaw da mesa é o mesmo da caixa (derivado)",
      ACTIVE.scene.table_jitter_yaw_deg == ACTIVE.scene.box_jitter_yaw_deg,
      "a spec pede 'um pouco de yaw' sem número; reusar a amplitude já calibrada "
      "evita inventar uma segunda escala angular para a mesma cena")
check("o alvo do `botar` sai da pose REAL da mesa",
      "self.table.data.root_link_pos_w" in inspect.getsource(C.LiftTargetCommand),
      "derivar de `table_xy` constante faria o robô soltar no lugar errado; derivar "
      "da CAIXA faria o alvo perseguir as mãos e o erro ser sempre zero")
check("o atrito da caixa é randomizado",
      "box_friction" in cfg.events, str(sorted(cfg.events)))
check("a faixa de atrito é ESTREITA em torno do nominal (1.0)",
      0.5 < ACTIVE.dr.box_friction_range[0] < 1.0 < ACTIVE.dr.box_friction_range[1] < 2.0,
      f"{ACTIVE.dr.box_friction_range} — faixa larga acrescenta variância ao "
      "success_buf, e mais variância significa mais congelamento espúrio")
check("não há `dr.body_mass` nem `dr.body_com_offset` na caixa",
      not any("body_mass" in str(getattr(e, 'func', '')) for e in cfg.events.values()),
      "os dois corrompem a heap (CUDA illegal access)")
_tw14 = env_t.command_manager.get_term("twist")
check("`v_max` cai com a carga, e a inclinação é derivada",
      abs(_tw14._inclinacao_carga - 0.0625) < 1e-6,
      f"{_tw14._inclinacao_carga:.4f} m/s por kg = (0.50-0.25)/(5-1)")
check("a banda de frenagem acompanha o `v_max` efetivo",
      "v_max ** 2 / (2.0 * self.cfg.a_max)" in inspect.getsource(
          C.DesiredTwistCommand._update_command),
      "derivada do nominal, a carga pesada começaria a frear meio metro cedo")
_orq15 = env_t.curriculum_manager.get_term_cfg("orquestrador").func
check("o currículo loga iterações desde o destravamento",
      hasattr(_orq15, "iteracoes_desde_evento"),
      "é o número mais valioso da rodada: sem ele, `54 destravamentos em 30 000 "
      "iterações` é aposta, não plano")
check("o currículo loga `amostras` no momento do destravamento",
      hasattr(_orq15, "amostras_no_evento"),
      "diz se o portão `min` decidiu por competência ou por sorte")
check("existe métrica de agachamento no `pegar`",
      "agachamento_pegar" in cfg.metrics)
# ⚠️ Este check nasceu TAUTOLÓGICO: ele lia a fonte do método e passava mesmo com o
# método nunca sendo chamado — que era o caso. Agora verifica a CONEXÃO.
check("o diagnóstico de vantagem está LIGADO ao laço de treino",
      getattr(runner, "_diag_vantagem", None) is not None
      and runner.alg.compute_returns.__name__ == "compute_returns"
      and runner.alg.update.__name__ == "update"
      and runner.alg.update.__qualname__.startswith("MultitaskRunner"),
      "definir o método sem chamá-lo não loga nada — foi o gap que a validação pegou")
_fonte_lig = inspect.getsource(MultitaskRunner._ligar_diagnostico_vantagem)
check("a leitura acontece ANTES da normalização",
      "compute_returns" in _fonte_lig and "_diag_vantagem = self._std_vantagem" in
      _fonte_lig.replace("\n", " ").replace("  ", " "),
      "o `update` normaliza a vantagem destrutivamente (ppo.py:188); ler depois "
      "mediria desvio padrão 1.0 em todas as tarefas")

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
# ⚠️ `PEGAR` é a ÚNICA exceção, e é deliberada (03/08): o termo de postura vale 0,5
# com a perna perto da pose padrão e vai a zero no agachamento, então ele cobrava
# 0,5 de quem agacha. Com o clamp de pitch em 1,05 rad a perna PRECISA sair da pose
# padrão no `pegar`. Nenhuma outra tarefa pode ficar descoberta nem coberta 2x.
_esperado = {t: (0 if t == T.PEGAR else 1) for t in range(T.NUM_TASKS)}
check("as 4 posturas cobrem cada tarefa 1x, menos `pegar` que é 0 de propósito",
      cobertura == _esperado,
      str({T.NAMES[t]: v for t, v in cobertura.items() if v != _esperado[t]}))
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
    # ⚠️ `BOTAR` saiu em 06/08. No `botar` a caixa nasce NA MÃO, exatamente nos alvos
    # por-mão do `reaching` — o termo valia ~0.99 no spawn e cobrava de 0.54 a 0.85
    # por passo de quem soltasse. O critério de sucesso exige `~preensao`, então o
    # argmax da recompensa era o estado que o critério reprova.
    "reaching": (T.REORIENTAR, T.PEGAR),
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
    # ⚠️ UM STEP antes de ler. `root_link_pos_w` de corpo mocap NÃO reflete a escrita
    # do reset sem um `forward()`, então ler direto depois do `reset()` traz a
    # prateleira na posição VELHA — foi o que fez o check acusar `z = 0.53` numa
    # prateleira que no sim estava a 5.53. Um step custa 0.02 s de queda livre, bem
    # dentro da tolerância de 2 cm da `folga`.
    env_t.step(torch.zeros(env_t.num_envs, env_t.action_manager.total_action_dim,
                           device=env_t.device))
    cx = env_t.scene["box"].data.root_link_pos_w
    pelve = env_t.scene["robot"].data.root_link_pos_w
    rel_z = (cx[:, 2] - pelve[:, 2]).mean().item()
    segura = tarefa in T.SPAWN_SEGURANDO
    nome = T.NAMES[tarefa]
    if segura:
        # a caixa nasce no alvo do peito: +0.15 m acima da pelve (§14)
        check(f"{nome}: caixa nasce no peito (+0.15 da pelve)",
              abs(rel_z - 0.15) < 0.03, f"rel_z={rel_z:+.3f}")
        check(f"{nome}: a tarefa sorteada já está ativa no spawn (S8)",
              bool((ativa_no_spawn(tarefa) == tarefa).all()),
              f"ativa={T.NAMES[int(env_a.active_task[0])]}")
    elif tarefa in T.MANIPULA:
        check(f"{nome}: caixa nasce na prateleira, NÃO na mão",
              rel_z < 0.0, f"rel_z={rel_z:+.3f}")
    else:
        # `parado` e `andar` não usam a caixa: o `afasta_cena` a estaciona 5 m acima,
        # junto com a prateleira. Então ela NÃO nasce "na prateleira ao alcance" nem
        # "na mão" — ela nasce fora do episódio, e é isso que o check tem que dizer.
        check(f"{nome}: caixa estacionada fora do alcance",
              rel_z > 2.0, f"rel_z={rel_z:+.3f}")

    # --- afasta_cena: a prateleira NÃO pode ficar no caminho de quem anda ---
    # Ela ocupa x de 0.20 a 0.80 com topo em 0.55 m (altura de joelho), e o destino do
    # `andar` fica a até 2.0 m na direção do heading — com heading perto de zero o robô
    # anda direto contra ela. Achado no `play` em 30/07.
    # Em Z, não em X: deslocar em x punha a prateleira dentro do robô de um env
    # vizinho (as origens ficam lado a lado em x/y) e derrubava ele — medido, e era a
    # causa dos 3 checks que falharam na primeira versão.
    mesa_z = (env_t.scene["table"].data.root_link_pos_w[:, 2]
              - env_t.scene.env_origins[:, 2])
    caixa_z = cx[:, 2] - env_t.scene.env_origins[:, 2]
    sobe = float(ACTIVE.scene.afasta_distancia)
    if tarefa in T.MANIPULA:
        check(f"{nome}: prateleira FICA embaixo (z<1)",
              bool((mesa_z < 1.0).all()), f"z={float(mesa_z.mean()):.2f}")
    else:
        check(f"{nome}: prateleira SUBIU (z>{sobe - 1:.0f})",
              bool((mesa_z > sobe - 1.0).all()), f"z={float(mesa_z.mean()):.2f}")
    if tarefa in (T.PARADO, T.ANDAR):
        # a caixa vai JUNTO: se só a prateleira sai, ela despenca — e o `box_shake`
        # não é gateado no `andar`, então ele puniria a queda.
        check(f"{nome}: caixa vai JUNTO com a prateleira",
              bool((caixa_z > sobe - 1.0).all()), f"z={float(caixa_z.mean()):.2f}")

    # --- a caixa está SOBRE a prateleira, e na altura certa? ---
    # Vale em TODAS as tarefas que não nascem segurando, perto ou a 5 m: o
    # `afasta_cena` move as duas pelo mesmo delta, então a relação tem que sobreviver.
    # Sem este check, a caixa podia estar pendurada no ar ou fora da mesa e nada
    # denunciava — foi o que aconteceu duas vezes em 30/07.
    if not segura:
        mesa_xy = env_t.scene["table"].data.root_link_pos_w[:, :2]
        mesa_z_w = env_t.scene["table"].data.root_link_pos_w[:, 2]
        desvio_xy = (cx[:, :2] - mesa_xy).abs().max().item()
        # o pé da caixa tem que encostar no topo da prateleira
        topo = mesa_z_w + float(ACTIVE.scene.shelf_half_z)
        pe = cx[:, 2] - float(ACTIVE.scene.box_half[2])
        folga = (pe - topo).abs().max().item()
        check(f"{nome}: caixa SOBRE a prateleira (xy dentro de "
              f"{ACTIVE.scene.shelf_half_xy:.2f})",
              desvio_xy <= float(ACTIVE.scene.shelf_half_xy) + 1e-3,
              f"desvio_xy={desvio_xy:.3f}")
        check(f"{nome}: caixa APOIADA (pé no topo da prateleira)",
              folga < 0.02, f"folga={folga:+.4f} m")

# O achado que motivou a T8b: antes dela, TODOS os termos de tarefa davam 0.0 no
# reset das 3 tarefas c/ caixa -> nenhum caminho de aquisição.
#
# ⚠️ Reescrito com a S8. Antes, o teste rodava na fase de pré-gatilho, em que as três
# tarefas esperavam em `parado c/ caixa` e o `box_at_peito` valia para todas. Sem essa
# janela, o gate do `box_at_peito` — `(PEGAR, PARADO_CAIXA, ANDAR_CAIXA)` — decide
# direto, e o `botar` fica FORA dele.
#
# Consequência declarada da S8: o `botar` perdeu o gradiente inicial que vinha do
# pré-gatilho. Ele não fica sem caminho, porque `reaching` é gateado nele e
# `box_at_prateleira` é kernel contínuo do erro — mas o sinal do primeiro passo é
# mais fraco do que era. Registrado para o `Contrib/botar/*` da rodada confirmar.
for tarefa in (T.PARADO_CAIXA, T.ANDAR_CAIXA):
    ativa_no_spawn(tarefa)
    t = env_a.reward_manager.get_term_cfg("box_at_peito")
    v = t.func(env_a, **t.params).mean().item()
    check(f"{T.NAMES[tarefa]}: box_at_peito > 0 no spawn (há gradiente)",
          v > 0.1, f"deu {v:.3f}")
# ⚠️ O `reaching` SAIU do gate do `botar` em 06/08. Ele valia ~0.99 no spawn (a caixa
# nasce exatamente nos alvos por-mão) e cobrava de 0.54 a 0.85 por passo de quem
# soltasse — contra um orçamento de 3.5, e o critério de sucesso EXIGE soltar. O
# argmax da recompensa era o estado que o critério reprova.
ativa_no_spawn(T.BOTAR)
_t_reach = env_a.reward_manager.get_term_cfg("reaching")
check("botar: `reaching` NÃO pontua mais (ele pagava para não soltar)",
      float(_t_reach.func(env_a, **_t_reach.params).abs().max()) == 0.0,
      "com ele ligado, soltar custava 15-24% do orçamento da tarefa")

# `botar`: o que importa NÃO é o valor absoluto no spawn — é a MONOTONICIDADE.
#
# ⚠️ O check antigo exigia `< 0.05` no spawn e passava com o termo valendo 4e-28, que
# é zero em float32. Ele confirmava "termo baixo longe" sem nunca verificar que ele
# SOBE ao aproximar — e o `botar` ficou sem shaping de transporte por causa disso: nos
# primeiros 33 cm de um percurso de 40 o gradiente era indistinguível de zero.
# Com as duas escalas (0.30 grossa + 0.05 fina) ele vale ~0.06 no spawn e cresce.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.BOTAR] = 1.0
env_t.reset(); env_t.step(acao_t)
t = env_t.reward_manager.get_term_cfg("box_at_prateleira")
_alvo_b = env_t.command_manager.get_term("lift_target").command[:, 0:3]
_caixa = env_t.scene["box"]
_serie = []
for _frac in (1.0, 0.6, 0.3, 0.05):        # fração do caminho que FALTA até o alvo
    _pos = _alvo_b + (_caixa.data.root_link_pos_w - _alvo_b) * _frac
    _est = torch.cat([_pos, _caixa.data.root_link_quat_w,
                      torch.zeros(env_t.num_envs, 6, device=DEVICE)], dim=-1)
    _caixa.write_root_state_to_sim(_est)
    env_t.sim.forward()
    _serie.append(float(t.func(env_t, **t.params).mean()))
check("botar: `box_at_prateleira` cresce monotonicamente ao aproximar",
      all(b > a for a, b in zip(_serie, _serie[1:])),
      " -> ".join(f"{v:.4f}" for v in _serie))
check("botar: há gradiente utilizável já no spawn",
      _serie[0] > 1e-3,
      f"{_serie[0]:.5f} — com o `std` único de 0.05 isto valia 4e-28, zero em float32")
env_t.reset()

print("\n-- S1: os eixos `altura` e `peso` chegam à cena e à física --")
# Antes da S1, `env.nivel` tinha um leitor só (`commands.py`), então `altura` e `peso`
# mediam competência num eixo CONSTANTE — 20 dos 54 destravamentos sobre nada.

check("`reset_box` e `reset_table` saíram de cfg.events",
      "reset_box" not in cfg.events and "reset_table" not in cfg.events,
      str([n for n in ("reset_box", "reset_table") if n in cfg.events]))
check("`reset_cena`, `jitter_cena` e `payload` entraram",
      all(n in cfg.events for n in ("reset_cena", "jitter_cena", "payload")),
      str([n for n in ("reset_cena", "jitter_cena", "payload")
           if n not in cfg.events]))
check("`lift` lê o zero de progresso POR-ENV",
      cfg.rewards["lift"].params.get("rest_z_attr") == "plr_rest_z",
      f"rest_z_attr={cfg.rewards['lift'].params.get('rest_z_attr')!r} "
      "— com `None` as alturas baixas ficam sem gradiente nenhum")

# --- a altura do nível vira POSIÇÃO da prateleira, e o zero do lift acompanha ---
_ids = torch.arange(env_t.num_envs, device=DEVICE)
_alturas = torch.tensor(T.LEVELS["altura"], device=DEVICE)
_termo_cena = env_t.event_manager.get_term_cfg("reset_cena")
_metade = env_t.num_envs // 2
env_t.nivel["altura"][:_metade] = 0                       # 0.55 m, o nível fácil
env_t.nivel["altura"][_metade:] = len(T.LEVELS["altura"]) - 1      # 0.00 m, o difícil
env_t.plr_shelf_top[:] = _alturas[env_t.nivel["altura"]]
_termo_cena.func(env_t, _ids, **_termo_cena.params)
env_t.sim.forward()

_jit = float(ACTIVE.scene.level_jitter_z)
_topo = (env_t.scene["table"].data.root_link_pos_w[:, 2]
         + float(ACTIVE.scene.shelf_half_z))
check(f"nível 6 põe o topo da prateleira em 0.00 ± {_jit:.2f}",
      bool((_topo[_metade:].abs() <= _jit + 1e-3).all()),
      f"topo={float(_topo[_metade:].mean()):+.3f}")
check("nível 0 põe o topo em 0.55 (a prateleira não é mais fixa)",
      bool((_topo[:_metade] - 0.55).abs().max() <= _jit + 1e-3),
      f"topo={float(_topo[:_metade].mean()):+.3f}")
check("`plr_rest_z` difere entre envs de níveis diferentes",
      float(env_t.plr_rest_z[:_metade].mean() - env_t.plr_rest_z[_metade:].mean())
      > 0.5, f"fácil={float(env_t.plr_rest_z[:_metade].mean()):.3f} "
             f"difícil={float(env_t.plr_rest_z[_metade:].mean()):.3f}")

# --- o sorteio escreve TODOS os eixos, sempre (passo 8 da S1) ---
# Sentinela: se algum eixo sair do reset com 99, o `_amostrar` não o escreveu e o env
# está com o nível do episódio ANTERIOR dele — que era o bug.
for _eixo in env_t.nivel:
    env_t.nivel[_eixo][:] = 99
env_t.task_dist = torch.ones(T.NUM_TASKS, device=DEVICE)
env_t.reset()
# `push` fica de fora: ele é o eixo GLOBAL (um `self.push_nivel` escalar, não um por
# env), e o `_amostrar` o pula de propósito. `env.nivel["push"]` é buffer morto — existe
# porque o `__init__` cria um por chave de `T.LEVELS`, e ninguém lê.
_sobrou = [e for e in env_t.nivel
           if e != "push" and int(env_t.nivel[e].max()) >= len(T.LEVELS[e])]
check("todo eixo por-env é escrito para todo env em cada reset",
      not _sobrou, str(_sobrou))

# --- eixo que a tarefa não possui = nível MAIS FÁCIL, não o corrente ---
for _t in (T.PARADO, T.ANDAR, T.PARADO_CAIXA, T.ANDAR_CAIXA):
    env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
    env_t.task_dist[_t] = 1.0
    env_t.nivel["altura"][:] = len(T.LEVELS["altura"]) - 1     # suja com o difícil
    env_t.reset()
    check(f"{T.NAMES[_t]} não tem eixo `altura` -> nível 0",
          int(env_t.nivel["altura"].max()) == 0,
          f"ficou em {int(env_t.nivel['altura'].max())} — o corrente daria ao "
          "`andar c/ caixa` giros de ±180° que o currículo nunca mediu")

print("\n-- S2: o eixo `push` chega à física --")
# Antes da S2 o eixo avançava de 0 a 4 e a perturbação não mudava: rodava o
# `push_robot` do fabricante, com o range dele, constante. A Fase 0 media a mesma
# dificuldade cinco vezes — e ela é o portão de `parado` para `andar`.
from g1_multitask import push as P  # noqa: E402

check("`push_robot` é o nosso, não o do fabricante",
      cfg.events["push_robot"].func is P.empurrao,
      f"func={getattr(cfg.events['push_robot'].func, '__name__', '?')} — somar em vez "
      "de substituir faria o segundo evento apagar o primeiro, sem erro e sem log")
check("`push_force` existe no treino",
      "push_force" in cfg.events, str(sorted(cfg.events)))
_cfg_play = load_env_cfg(g1_multitask.TASK_ID, play=True)
check("nenhum dos dois existe no `play`",
      "push_robot" not in _cfg_play.events and "push_force" not in _cfg_play.events,
      str([n for n in ("push_robot", "push_force") if n in _cfg_play.events]))

# --- o fator é zero no nível 0, e o chute some junto ---
env_t.task_dist = torch.ones(T.NUM_TASKS, device=DEVICE)
env_t.reset()
check("`push_fator` é zero enquanto `push_nivel == 0`",
      int(env_t.push_nivel_t.max()) == 0 and float(env_t.push_fator.abs().max()) == 0.0,
      f"nível={int(env_t.push_nivel_t.max())} "
      f"fator_max={float(env_t.push_fator.abs().max()):.3f}")

# --- a janela livre silencia o empurrão no começo do episódio ---
# ⚠️ `forward()` antes de reler: `root_link_vel_w` não reflete a escrita sem ele, e
# sem o forward este teste dá delta zero nos DOIS casos e passa por engano.
_robo = env_t.scene["robot"]
_ids_t = torch.arange(env_t.num_envs, device=DEVICE)
env_t.push_fator[:] = 1.0
_delta = {}
for _rotulo, _buf in (("dentro", 0), ("fora", 100)):
    env_t.episode_length_buf[:] = _buf
    env_t.sim.forward()
    _v0 = _robo.data.root_link_vel_w.clone()
    P.empurrao(env_t, _ids_t, push=ACTIVE.push)
    env_t.sim.forward()
    _delta[_rotulo] = (_robo.data.root_link_vel_w - _v0).abs().max().item()
check(f"nenhum empurrão nos primeiros {P.JANELA_LIVRE_S:.1f} s do episódio",
      _delta["dentro"] == 0.0 and _delta["fora"] > 0.0,
      f"dentro={_delta['dentro']:.4f} fora={_delta['fora']:.4f} — a caixa das 3 "
      "tarefas de spawn-segurando cai 22 cm em 0.5 s com ação nula")

# --- o `and hold` só existe nos níveis 3 e 4 ---
_hold = env_t.event_manager.get_term_cfg("push_force").func
env_t.episode_length_buf[:] = 100
_ativos = {}
for _nivel in (0, 2, 4):
    _hold._inner._active[:] = False
    env_t.push_nivel_t[:] = _nivel
    for _ in range(300):
        _hold(env_t, _ids_t, push=ACTIVE.push)
    _ativos[_nivel] = int(_hold._inner._active.sum())
check("força sustentada só a partir do nível 3",
      _ativos[0] == 0 and _ativos[2] == 0 and _ativos[4] > 0,
      f"impulsos ativos por nível: {_ativos} (de {env_t.num_envs} envs)")

# ⚠️ HIGIENE: os checks acima deixaram `env_t` com fator 1.0, nível 4 e impulsos
# ATIVOS na pelve. `env_t` é reusado pelas seções seguintes, e um robô sob 50 N
# segurados falharia checks que nada têm a ver com push. Zera a força na mão (o
# `_active` do inner não expira sozinho sem mais chamadas) e re-sorteia pelo reset.
_zeros = torch.zeros(env_t.num_envs, _hold._n_bodies, 3, device=DEVICE)
_hold._asset.write_external_wrench_to_sim(
    _zeros, _zeros, env_ids=_ids_t, body_ids=_hold._body_ids)
_hold._inner._active[:] = False
env_t.reset()

print("\n-- S3: congelamento por média lenta, não por pico --")
from g1_multitask.curriculum import PUSH as PUSH_EIXO  # noqa: E402

_orq3 = env_t.curriculum_manager.get_term_cfg("orquestrador").func
_cel3 = (T.PEGAR, "altura")
_abertas_antes = list(_orq3.abertas)

check("a referência se chama `ref`, e `pico` não existe mais",
      hasattr(_orq3, "ref") and not hasattr(_orq3, "pico"))
check("`ema_alpha_lenta` é um décimo de `ema_alpha`",
      abs(_orq3.alpha_lenta - _orq3.alpha / 10.0) < 1e-9,
      f"alpha={_orq3.alpha} lenta={_orq3.alpha_lenta}")
check("`congela_queda` continua em 0.10",
      abs(_orq3.congela_queda - 0.10) < 1e-9, f"{_orq3.congela_queda}")

# A referência PERSEGUE a performance, em vez de guardar o máximo. É a diferença que
# faz o 0.10 voltar a significar 3σ: com o máximo, um pico de sorte vira o alvo para
# sempre e a queda típica passa do limiar sem nenhuma regressão.
_orq3.ref[_cel3][0] = 0.0
_orq3.perf[_cel3][0] = 0.90
for _ in range(400):
    _orq3._congelamento(_cel3, 0)
_subiu = float(_orq3.ref[_cel3][0])
_orq3.perf[_cel3][0] = 0.50                       # queda real de 0.40
for _ in range(5):
    _orq3._congelamento(_cel3, 0)
_congelou_real = bool(_orq3.congelado[_cel3][0])
_orq3.perf[_cel3][0] = 0.90                       # recupera
for _ in range(400):
    _orq3._congelamento(_cel3, 0)
_soltou = not bool(_orq3.congelado[_cel3][0])

check("a referência sobe até a performance (não fica em zero)",
      _subiu > 0.5, f"ref={_subiu:.3f} depois de 400 medições em perf=0.90")
check("queda REAL de 0.40 ainda congela", _congelou_real)
check("recuperar descongela", _soltou)

# Um pico isolado NÃO vira a referência. Com o máximo corrido, um único 1.0 fixava o
# alvo em 1.0 e toda medição seguinte em 0.90 acusava queda de 0.10.
_orq3.ref[_cel3][0] = 0.0
_orq3.perf[_cel3][0] = 0.90
for _ in range(400):
    _orq3._congelamento(_cel3, 0)
_base = float(_orq3.ref[_cel3][0])
_orq3.perf[_cel3][0] = 1.0                        # o pico de sorte
_orq3._congelamento(_cel3, 0)
_orq3.perf[_cel3][0] = 0.90                       # e volta ao normal
_orq3._congelamento(_cel3, 0)
check("um pico isolado não fixa a referência",
      abs(float(_orq3.ref[_cel3][0]) - _base) < 0.01
      and not bool(_orq3.congelado[_cel3][0]),
      f"ref {_base:.4f} -> {float(_orq3.ref[_cel3][0]):.4f}")

# --- a célula de push mede só os envs do `parado` ---
# Antes, ela media `sucesso.mean()` sobre TODAS as tarefas: abrir o `andar` derrubava
# a perf de 0.900 para 0.696 sem nenhuma regressão de robustez a push.
_cel_push = (T.PARADO, PUSH_EIXO)
_orq3.perf[_cel_push][0] = 0.0
_orq3.abertas = [T.PARADO, T.ANDAR]
env_t.tarefa_sorteada[:] = T.ANDAR
env_t.tarefa_sorteada[: env_t.num_envs // 2] = T.PARADO
env_t.success_buf[:] = 0.0
env_t.success_buf[: env_t.num_envs // 2] = 1.0     # `parado` vence, `andar` falha
_orq3._visitou[:] = True
_amostras_antes = float(_orq3.amostras[_cel_push][0])
for _ in range(300):
    _orq3._medir(env_t, torch.arange(env_t.num_envs, device=DEVICE))
check("a perf do push segue o `parado`, e não a média das tarefas",
      float(_orq3.perf[_cel_push][0]) > 0.95,
      f"perf={float(_orq3.perf[_cel_push][0]):.3f} — a média global daria ~0.50, e o "
      "limiar absoluto de 0.90 do `_push_competente` seria inatingível")
check("só os envs do `parado` contam como amostra",
      abs((float(_orq3.amostras[_cel_push][0]) - _amostras_antes)
          - 300 * (env_t.num_envs // 2)) < 1.0,
      f"{float(_orq3.amostras[_cel_push][0]) - _amostras_antes:.0f} de "
      f"{300 * (env_t.num_envs // 2)}")

# ⚠️ HIGIENE: os checks acima mexeram no estado do orquestrador (abriram o `andar`,
# escreveram perf, ref, amostras e congelado) e no `success_buf`. `env_t` é reusado
# pelas seções seguintes. Sem restaurar, o currículo destravaria por dado fabricado.
_orq3.abertas = _abertas_antes
for _c in (_cel3, _cel_push):
    _orq3.perf[_c].zero_()
    _orq3.ref[_c].zero_()
    _orq3.amostras[_c].zero_()
    _orq3.congelado[_c][:] = False
_orq3.desde_evento = {_t: 0.0 for _t in T.AXES}
env_t.success_buf[:] = 0.0
env_t.reset()

print("\n-- S5: perfil de comando gira, depois anda --")
_tw = env_t.command_manager.get_term("twist")
_lt = env_t.command_manager.get_term("lift_target")
_cmd_cfg = cfg.commands["twist"]

check("banda de frenagem = v_max²/2·a_max",
      abs(_cmd_cfg.d_freio_extra
          - _cmd_cfg.v_max ** 2 / (2.0 * _cmd_cfg.a_max)) < 1e-6,
      f"{_cmd_cfg.d_freio_extra} vs {_cmd_cfg.v_max ** 2 / (2.0 * _cmd_cfg.a_max):.4f} "
      "— o número era órfão antes da S5 e voltaria a ser sem esta amarra")
check("banda angular = w_max²/2·alpha_max",
      abs(_tw._banda_ang - _cmd_cfg.w_max ** 2 / (2.0 * _cmd_cfg.alpha_max)) < 1e-9,
      f"{_tw._banda_ang:.4f} rad")
check("o morto angular cabe na tolerância de `alinhado` (10°)",
      _cmd_cfg.morto_angular_rad < math.radians(10.0),
      f"{math.degrees(_cmd_cfg.morto_angular_rad):.1f}° — se fosse maior, o comando "
      "pararia de girar antes de o critério considerar alinhado")

env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.ANDAR] = 1.0
env_t.reset()


def _perfil(offset_x: float, passos: int):
    """Roda o perfil com o destino fixado a `offset_x` metros à frente do robô.

    ⚠️ Chama `_update_command` DIRETO, sem `env.step`. Com ação nula o robô cai, o
    episódio termina dentro do `step`, e o reset re-sorteia `_destino_w` — o alvo
    forçado some e o teste mede outra coisa. Deu 1 falha em 2 execuções antes desta
    mudança. Sem física o perfil é determinístico, que é o que estes checks querem:
    eles testam a lei de controle, não o equilíbrio do robô.
    """
    _p = env_t.scene["robot"].data.root_link_pos_w
    _alvo = torch.zeros_like(_p)
    _alvo[:, 0] = offset_x
    serie = []
    for _ in range(passos):
        _lt._destino_w.copy_(_p + _alvo)
        _lt._update_command()
        _tw._update_command()
        c = _tw.command
        serie.append((float(c[:, 0].abs().max()), float(c[:, 1].abs().max()),
                      float(c[:, 2].abs().max())))
    return serie


# --- alvo ATRÁS do robô: gira sem andar ---
_s = _perfil(-2.0, 50)
_v180 = max(x[0] for x in _s)
_w180 = max(x[2] for x in _s)
check("rumo 180°: v fica em zero e só w trabalha",
      _v180 < 1e-6 and _w180 > 0.5 * _cmd_cfg.w_max,
      f"max|v|={_v180:.2e} max|w|={_w180:.3f} — trava por cos(erro), contínua; "
      "um limiar duro devolveria o degrau que a rampa remove")

# --- limitador de taxa: nenhum degrau acima do teto ---
env_t.reset()
_s = _perfil(2.0, 60)
_mdv = max(abs(b[0] - a[0]) for a, b in zip(_s, _s[1:]))
_mdw = max(abs(b[2] - a[2]) for a, b in zip(_s, _s[1:]))
_teto_v = _cmd_cfg.a_max * env_t.step_dt
_teto_w = _cmd_cfg.alpha_max * env_t.step_dt
check("nenhum degrau de v acima de a_max·dt",
      _mdv <= _teto_v + 1e-6, f"max|dv|={_mdv:.4f} teto={_teto_v:.4f}")
check("nenhum degrau de w acima de alpha_max·dt",
      _mdw <= _teto_w + 1e-6, f"max|dw|={_mdw:.4f} teto={_teto_w:.4f}")
check("o perfil nunca comanda `vy`",
      max(x[1] for x in _s) == 0.0,
      "o robô gira para apontar e anda para frente; marcha lateral fica fora "
      "desta rodada, por decisão da S5")

# --- dentro do morto o zero é EXATO ---
env_t.reset()
_s = _perfil(0.02, 10)               # 0.02 m, dentro do d_morto_andar de 0.05
check("dentro do `d_morto` o comando é exatamente zero",
      _s[-1][0] == 0.0 and _s[-1][2] == 0.0 and bool(_tw.dentro_do_morto().all()),
      f"v={_s[-1][0]:.2e} w={_s[-1][2]:.2e}")
env_t.reset()

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

fonte_sucesso = inspect.getsource(__import__("g1_multitask.metrics",
                                             fromlist=["Sucesso"]).Sucesso)
check("sucesso NÃO lê reward_manager (senão peso viraria Categoria C)",
      "reward_manager" not in fonte_sucesso)
# O `pegar` exige PREENSÃO. Sem ela o critério passa ENCOSTANDO o peito na caixa parada
# na prateleira: na fronteira do `de_pe` (pelve 0.65, tilt 20°) o alvo do peito desce
# para z=0.723 e a caixa está em 0.65 — 0.073 m, dentro dos 0.10 da tolerância. Medido
# em 31/07: 98,6% de sucesso com `grasp = 0` e a caixa subindo 3,8 cm.
_cond_pegar = fonte_sucesso.split("T.PEGAR")[1].split("cond)")[0]
check("o critério do `pegar` exige preensão", "preensao" in _cond_pegar,
      "sem isso ele pontua ENCOSTANDO na caixa, sem nunca pegá-la")
check("as 4 tarefas com caixa na mão citam preensão",
      all(f"T.{n}" in fonte_sucesso for n in
          ("PEGAR", "BOTAR", "PARADO_CAIXA", "ANDAR_CAIXA")))
# O `parado` exige DE PÉ. "Sobreviveu" sozinho aprova SENTAR: o `fell_over` mede só
# inclinação (70°) e sentado com o tronco vertical ela é ~0°, então o robô nunca "cai".
# Medido no `play` em 31/07: `sucesso = 0,9553` com o robô no chão.
_cond_parado = fonte_sucesso.split("T.PARADO,")[1].split("cond)")[0]
check("o critério do `parado` exige de pé", "parado_de_pe" in _cond_parado,
      "sem isso ele pontua SENTADO — o fell_over não pega sentar")
_fonte_de_pe = inspect.getsource(
    __import__("g1_multitask.terminations", fromlist=["de_pe"]).de_pe)
check("`de_pe` mede ALTURA e inclinação, não só inclinação",
      "root_link_pos_w" in _fonte_de_pe and "projected_gravity_b" in _fonte_de_pe,
      "é a altura que separa sentado de em pé — a inclinação sentada é ~0°")
# o limiar tem que ficar ENTRE a pelve sentada e a pelve agachada, senão ele não separa
check("limiar de `de_pe` separa sentado (0.30) de agachado (0.70)",
      0.30 < ACTIVE.tolerancia.de_pe_z < 0.70,
      f"de_pe_z = {ACTIVE.tolerancia.de_pe_z}; pelve no KNEES_BENT é 0.76")

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

print("\n-- S6: sucesso do `andar` com histerese --")
check("`sustenta_andar_s` baixou de 5 para 3 s",
      abs(tol.sustenta_andar_s - 3.0) < 1e-9, f"{tol.sustenta_andar_s} s")
check("o raio de disparo é menor que o de manutenção",
      tol.andar_raio_chega < tol.andar_raio_mantem,
      f"chega={tol.andar_raio_chega} mantem={tol.andar_raio_mantem}")
check("o raio de disparo é maior que o `d_morto_andar`",
      tol.andar_raio_chega > ACTIVE.command.d_morto_andar,
      f"raio={tol.andar_raio_chega} morto={ACTIVE.command.d_morto_andar} — se o morto "
      "fosse maior, o comando zeraria antes de o robô entrar no círculo da régua")
check("o ângulo de disparo é menor que o de manutenção",
      tol.alinhado_chega_deg < tol.alinhado_mantem_deg,
      f"chega={tol.alinhado_chega_deg}° mantem={tol.alinhado_mantem_deg}°")

# --- a histerese muda o veredito no MESMO estado físico ---
# Robô no mesmo lugar, a 0.20 m do alvo: entre o raio de disparo (0.10) e o de
# manutenção (0.25). Sem disparo prévio o critério é falso; com disparo, verdadeiro.
# É esse par que prova que a histerese existe — um raio único daria o mesmo nos dois.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.ANDAR] = 1.0
env_t.reset()
_lt6 = env_t.command_manager.get_term("lift_target")
_p6 = env_t.scene["robot"].data.root_link_pos_w
_off6 = torch.zeros_like(_p6)
_off6[:, 0] = 0.20
_lt6._destino_w.copy_(_p6 + _off6)
# ⚠️ O rumo tem de ser isolado, senão este teste mede o eixo errado. O `_head` é
# sorteado em ±30° no nível 0 e o `reset_base` gira o robô em ±11.5°, então o erro
# pode chegar a ~41° — acima dos 25° de manutenção — e o `alinhado` derrubaria o
# critério por um motivo que nada tem a ver com o raio.
_lt6._head.copy_(env_t.scene["robot"].data.heading_w)
_lt6._update_command()
sucesso._alinhado_ok[:] = True
sucesso._chegou_ok[:] = False
_sem = bool(sucesso._condicao(env_t).any())
sucesso._chegou_ok[:] = True
_com = bool(sucesso._condicao(env_t).all())
check("a 0.20 m: sem disparo é falso, com disparo é verdadeiro",
      (not _sem) and _com, f"sem_disparo={_sem} com_disparo={_com}")

# --- o disparo exige o raio apertado ---
sucesso._chegou_ok[:] = False
_off6[:, 0] = 1.0
_lt6._destino_w.copy_(_p6 + _off6)
_lt6._update_command()
sucesso(env_t)
_longe = bool(sucesso._chegou_ok.any())
_off6[:, 0] = 0.05
_lt6._destino_w.copy_(_p6 + _off6)
_lt6._update_command()
sucesso(env_t)
_perto6 = bool(sucesso._chegou_ok.all())
check("o disparo só ocorre dentro do raio apertado",
      (not _longe) and _perto6, f"a 1.0 m disparou={_longe}  a 0.05 m disparou={_perto6}")

# --- cair continua zerando: `parado_de_pe` fica FORA da histerese ---
_fonte_cond = inspect.getsource(type(sucesso)._condicao)
check("`parado_de_pe` não entra na histerese",
      "_chegou_ok & " in _fonte_cond and "parado_de_pe" not in
      _fonte_cond.split("chegou_andar = ")[1].split("\n")[0],
      "se travasse junto, cair deixaria de zerar o contador")
env_t.reset()

print("\n-- S7: sucesso do `parado` exige ficar de pé --")
check("o critério do `parado` limita o tempo fora de `de pé`",
      "_fora_de_pe_s" in _fonte_cond and "limite_fora_de_pe_s" in _fonte_cond,
      "sem isso um robô permanentemente agachado passa: o `fell_over` mede só "
      "inclinação (70°), e agachado com o tronco vertical ela é ~0°")
check("o limite cabe entre passo protetivo e agachamento permanente",
      1.0 < tol.limite_fora_de_pe_s < 20.0,
      f"{tol.limite_fora_de_pe_s} s — passo protetivo custa 1-2 s e passa; "
      "agachado consome os 20 s e não passa")
check("o acumulador é de TEMPO, não de passos",
      "* self.dt" in inspect.getsource(type(sucesso).__call__),
      "em passos o limite dependeria do `decimation`")

# o acumulador anda quando o robô sai de `de pé`. Com ação nula ele cai, então basta
# rodar passos e comparar — não é preciso forçar pose nenhuma.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.PARADO] = 1.0
env_t.reset()
# ⚠️ O pico tem de ser lido A CADA passo. Com ação nula o robô cai, o `fell_over`
# termina o episódio, e o reset ZERA o acumulador — ler só no fim dá 0,00 s sempre.
_pico_fora = 0.0
for _ in range(120):
    env_t.step(acao_t)
    _pico_fora = max(_pico_fora, float(sucesso._fora_de_pe_s.max()))
check("o tempo fora de `de pé` acumula com o robô caído",
      _pico_fora > 0.0, f"pico de {_pico_fora:.2f} s com ação nula")
env_t.reset()
check("o reset zera o acumulador",
      float(sucesso._fora_de_pe_s.max()) == 0.0,
      f"{float(sucesso._fora_de_pe_s.max()):.2f} s — se vazasse, o episódio novo "
      "nasceria reprovado")

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
env_o = ManagerBasedRlEnv(cfg=cfg_obs, device=DEVICE)
acao_o = torch.zeros(env_o.num_envs, env_o.action_manager.total_action_dim,
                     device=env_o.device)
env_o.task_dist = torch.ones(T.NUM_TASKS, device=DEVICE)     # as 7 sorteadas
env_o.reset()

# ⚠️ A contagem é conferida ANTES do limiar do relatório, não no fim dos 30 passos.
# O `Relatorio` ZERA a matriz quando emite, e ele emite no primeiro reset em que
# `cont.sum()` passa de `min_amostras` (500) — o que acontece no passo 8 de 30. Depois
# disso a contagem mede "passos desde o último reset", não "todo par (env, passo)".
#
# Medido em 30/07: dava 960 de 1920, ou seja um reset no passo 15. E a causa do reset é
# BOA: antes a prateleira ficava embaixo do robô nas tarefas c/ caixa e SEGURAVA a
# caixa que escorregava, então a terminação `largou` (caixa_z < 0.30) nunca disparava.
# Com a prateleira afastada ela cai até o chão e o `largou` funciona como projetado.
#
# Mexer no `min_amostras` para contornar isso quebra os TRÊS checks de emissão logo
# abaixo, que é exatamente o que eles existem para testar. A janela curta resolve sem
# tocar no config.
CURTO = 7
assert CURTO * cfg_obs.scene.num_envs < 500, "a janela tem que caber sob o min_amostras"
for _ in range(CURTO):
    env_o.step(acao_o)
check("acumulador conta todos os (env, passo)",
      abs(float(env_o.contrib_cont.sum()) - CURTO * env_o.num_envs) < 1e-3,
      f"{float(env_o.contrib_cont.sum()):.0f} de {CURTO * env_o.num_envs}")
# Este também vive na janela curta, pelo mesmo motivo: depois do primeiro relatório a
# matriz zera, e no fim dos 30 passos ela pode estar recém-zerada. Aqui ela nunca foi.
check("as 7 tarefas foram amostradas",
      bool((env_o.contrib_cont > 0).all()),
      str([int(x) for x in env_o.contrib_cont]))

PASSOS = 30
for _ in range(PASSOS - CURTO):
    env_o.step(acao_o)
check("um nome de termo por coluna",
      len(env_o.contrib_nomes) == env_o.contrib_soma.shape[1] == len(cfg.rewards),
      f"{len(env_o.contrib_nomes)} nomes, {env_o.contrib_soma.shape[1]} colunas")

# ⚠️ CORRIDA, consertada em 05/08. Os checks abaixo exigem que o relatório EMITA, e
# emitir exige `cont.sum() >= min_amostras` (500). Mas o `Relatorio` roda no reset e
# ZERA quando emite, então a contagem no passo 30 mede "passos desde o último reset" —
# e onde esse reset cai depende da física.
#
# Com reset no passo 15 sobram 960 e o check passa; com reset no passo 23 sobram 448 e
# ele falha. Medido: 2 falhas em 3 execuções depois que a S1 ligou o `level_jitter_z`
# de ±2 cm e o jitter de yaw, que mudam quando a caixa escorrega e portanto quando o
# `largou` dispara.
#
# O conserto é tornar a PRÉ-CONDIÇÃO explícita, e não afrouxar o limiar: mexer no
# `min_amostras` quebraria os três checks de emissão, que é o que eles testam.
LIMITE = 200
while (float(env_o.contrib_cont.sum()) < 500.0
       and PASSOS < LIMITE):
    env_o.step(acao_o)
    PASSOS += 1
assert PASSOS < LIMITE, "a matriz nunca acumulou 500 amostras — não é corrida, é bug"

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
check("simulação: 54 destravamentos", orq_sim.eventos == 54, f"deu {orq_sim.eventos}")
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
# S11: eram 14. O eixo de distância saiu do `pegar` e do `reorientar`, então caem
# duas células tarefa×eixo. 11 = andar 2, pegar 2, botar 2, reorientar 2,
# parado c/ caixa 1, andar c/ caixa 2.
check("12 células (11 tarefa×eixo + push)", len(orq_env.celulas) == 12,
      f"deu {len(orq_env.celulas)}")

# ALARME DE ESTAGNAÇÃO: o contador tem que ser transições DE VERDADE.
# A versão de 30/07 fazia `len(ids) * max_episode_length` e contava cada env que
# terminou como episódio COMPLETO de 1000 passos. Com episódio de 11 passos isso
# inflava 91x e o alarme disparava ~1900 iterações antes da hora. O cenário abaixo
# é exatamente o que expôs o bug: episódio CURTO e nenhum sucesso.
from g1_multitask.curriculum import Orquestrador  # noqa: E402
from g1_multitask.sim_curriculo import _CfgFalso, _EnvFalso  # noqa: E402

_n, _passos, _chamadas = 64, 11, 300
_env_c = _EnvFalso(_n)
_orq_c = Orquestrador(_CfgFalso({"curriculum": ACTIVE.curriculum,
                                 "min_amostras_evento": 10**9,  # nunca destrava
                                 "verboso": False}), _env_c)
_env_c.success_buf.fill_(0.0)
for _ in range(_chamadas):
    _env_c.common_step_counter += _passos
    _orq_c(_env_c, torch.arange(_n))
_esperado = float(_passos * _chamadas * _n)
check("alarme conta transições reais, não episódios cheios",
      abs(_orq_c.transicoes_sem_evento - _esperado) < 1e-6,
      f"deu {_orq_c.transicoes_sem_evento:.3e}, esperado {_esperado:.3e} "
      f"(a versão antiga daria {_chamadas * _n * 1000:.3e}, ou seja 91x)")

# ------------------------------- consertos de 31/07: itens 1 a 5 da fila
print("\n-- consertos de 31/07 (escala_c, pré-gatilho, frame do reorientar, log) --")
from g1_residual.base_z import ESCALA_C as _ESC, PRIOR as _PRIOR  # noqa: E402

# item 1 — a escala da busca de comportamento. Histórico: era 0.3, e nesse valor a
# política precisava de |c| ~ 28 pra trocar de comportamento e emitia ~10 — a busca não
# existia. Subiu pra 1.0, a busca passou a funcionar, e a run desmontou (episódio 765 ->
# 17,9). Desde 03/08 é **0.0**: a busca está DESLIGADA e `z` fica no prior.
# O que o teste protege agora é o valor 0.3, que é o pior dos três: busca que existe no
# papel e não no gradiente.
check("item 1: escala_c não está no limbo de 0.3",
      _ESC == 0.0 or _ESC >= 1.0,
      f"ESCALA_C = {_ESC} — nem desligada nem efetiva")

# item 5 — prior por SEMENTE onde as 10 sementes discordam (60° e 74°). Média só vale
# onde elas concordam (<10°).
check("item 5: PRIOR carrega semente", all(isinstance(v, tuple) and len(v) == 2
                                           for v in _PRIOR.values()))
_por_nome = {nome: sem for nome, sem in _PRIOR.values()}
check("item 5: move-ego-0-0 usa semente, não média",
      _por_nome.get("move-ego-0-0") is not None,
      "as 10 sementes estão a 60° uma da outra; a média não é comportamento nenhum")
# Forma durável do item 5: comportamento cujas 10 sementes DISCORDAM nunca pode entrar
# como média, seja ele prior de quantas tarefas for. Medido em 31/07: `move-ego-0-0` a
# 60° entre sementes, `raisearms-m-m` a 74° — e entre dois comportamentos DIFERENTES o
# ângulo médio é 83°, então duas sementes dessas estão quase tão longe quanto dois
# comportamentos distintos. A média cai num ponto que não é nenhum dos dois.
_DISCORDAM = ("move-ego-0-0", "raisearms-m-m")
_com_media = [n for n, s in _PRIOR.values() if s is None and n in _DISCORDAM]
check("item 5: nenhum prior usa a média de comportamento difuso",
      not _com_media, f"usam média: {_com_media}")

# Desenho de 03/08: com `ESCALA_C = 0` o `z` nunca sai do prior, então os 7 priors
# apontando para o mesmo comportamento é o que faz o BFM ser só equilíbrio.
_nomes_prior = {n for n, _ in _PRIOR.values()}
if _ESC == 0.0:
    check("busca desligada => os 7 priors são o MESMO comportamento",
          len(_nomes_prior) == 1 and "move-ego-0-0" in _nomes_prior,
          f"priors distintos: {sorted(_nomes_prior)} — com ESCALA_C=0 a política não "
          f"pode sair de nenhum deles, então prior por tarefa vira escolha fixa minha")

    # Canal inerte não é neutro: é POÇO DE ENTROPIA GRÁTIS. O bônus de entropia é soma
    # sobre as dimensões e o `log_std` é livre, então canal sem efeito (`ESCALA_C = 0`)
    # e sem custo (o `action_rate_l2` só cobra os 29) deixa a política inflar o desvio
    # dele sem limite. A entropia para de regular a exploração nos canais reais, e o
    # `Loss/entropy` do log fica ilegível.
    from g1_residual.env_residual import build_env_residual as _bld  # noqa: E402
    _dim_c_default = inspect.signature(_bld).parameters["dim_c"].default
    check("busca desligada => os canais de comportamento SAEM da ação",
          _dim_c_default == 0,
          f"dim_c = {_dim_c_default} com ESCALA_C = 0: {_dim_c_default} canais sem "
          f"efeito e sem custo viram entropia de graça")

# item 2 — o pré-gatilho não pode fechar sucesso
_fonte_call = inspect.getsource(
    __import__("g1_multitask.metrics", fromlist=["Sucesso"]).Sucesso.__call__)
check("item 2: sucesso gateado pelo gatilho", "disparou" in _fonte_call,
      "sem isso o critério do `parado` (sustentação 0 s) pontua por outra tarefa")
_meta_t = env_t.command_manager.get_term("lift_target")
# ---- clamp do residual (03/08): curso no plano sagital, correção no resto ----
from g1_residual.acao import LIMITE_PADRAO as _LIM  # noqa: E402
from g1_residual.acao import ResidualBFMActionCfg as _ACfg  # noqa: E402

_SAGITAL = (r".*_hip_pitch_joint", r".*_knee_joint",
            r".*_ankle_pitch_joint", r"waist_pitch_joint")
check("clamp sagital tem curso pra agachar/andar",
      all(_LIM.get(k, 0.0) >= 1.0 for k in _SAGITAL),
      str({k: _LIM.get(k) for k in _SAGITAL if _LIM.get(k, 0.0) < 1.0}))
_rot = {k: v for k, v in _LIM.items()
        if ("roll" in k or "yaw" in k) and "shoulder" not in k and "wrist" not in k}
check("roll e yaw de perna/cintura ficam só em correção",
      all(v <= 0.35 for v in _rot.values()),
      str({k: v for k, v in _rot.items() if v > 0.35}))

# O INVARIANTE que importa, e é o que se perde ao subir um limite sem pensar: o
# `_limite` também escala a exploração INICIAL, porque `delta = clamp(bruto*escala,
# ±1)*limite` e no começo `bruto ~ 0,95`. A âncora medida é a run monolítica, que
# explorava a ±0,32 rad (18,3°) por junta e aprendeu a ficar de pé em ~250 iterações.
# Passar disso é sair do território provado.
_ANCORA_RAD = 0.32
_expl = {k: 0.95 * _ACfg.escala_delta * v for k, v in _LIM.items()}
check(f"exploração inicial fica dentro da âncora de {_ANCORA_RAD} rad (18,3°)",
      all(v <= _ANCORA_RAD + 1e-9 for v in _expl.values()),
      str({k: round(v, 3) for k, v in _expl.items() if v > _ANCORA_RAD})
      + f" | com escala_delta={_ACfg.escala_delta}, o limite máximo é "
        f"{_ANCORA_RAD / (0.95 * _ACfg.escala_delta):.2f} rad")

# ---- critério do `parado` (03/08): velocidade, não posição ----
# `_cond_parado` já foi extraído acima (a fatia do `torch.where` da tarefa `parado`).
check("critério do `parado` mede VELOCIDADE por fração do episódio",
      "_frac_quieto" in _cond_parado and "parado_fracao" in _cond_parado,
      "sem isto o `parado` volta a aprovar quem anda 20 s e para no último passo")
check("a deriva de POSIÇÃO continua só logada, não é portão",
      "deriva_parado" not in _cond_parado,
      "posição como portão comprime o sucesso a zero sob push nível 4 (F3)")
_Suc = __import__("g1_multitask.metrics", fromlist=["Sucesso"]).Sucesso
_cond_src = inspect.getsource(_Suc._condicao)
_call_src = inspect.getsource(_Suc.__call__)
check("o acumulador de `quieto` vive no __call__, não no _condicao",
      "_quieto_passos +=" in _call_src and "_quieto_passos +=" not in _cond_src,
      "o `_condicao` roda 2x por passo (tarefa ativa e sorteada) — contaria dobrado")
check("o acumulador zera no reset do episódio",
      "_quieto_passos[caiu]" in _call_src,
      "sem zerar, a fração do episódio anterior vaza para o próximo")

check("item 2: o comando expõe `disparou`",
      hasattr(_meta_t, "disparou") and tuple(_meta_t.disparou.shape) == (env_t.num_envs,))

# item 3 — o erro de ângulo do `reorientar` NÃO pode depender da pose do robô.
# Check comportamental: gira o ROBÔ 40° em torno de z, com a caixa intacta.
env_t.task_dist = torch.zeros(T.NUM_TASKS, device=DEVICE)
env_t.task_dist[T.REORIENTAR] = 1.0
env_t.reset()
_acao_t = torch.zeros(env_t.num_envs, env_t.action_manager.total_action_dim,
                      device=DEVICE)
env_t.step(_acao_t)
_ang_antes = _meta_t.erro_angulo_deg().clone()
_rb = env_t.scene["robot"]
_q = _rb.data.root_link_quat_w.clone()
_meio = math.radians(40.0) / 2.0
_giro = torch.zeros_like(_q)
_giro[:, 0] = math.cos(_meio)
_giro[:, 3] = math.sin(_meio)          # rotação de 40° em torno de z (w,x,y,z)
_w0, _x0, _y0, _z0 = _q[:, 0], _q[:, 1], _q[:, 2], _q[:, 3]
_w1, _x1, _y1, _z1 = _giro[:, 0], _giro[:, 1], _giro[:, 2], _giro[:, 3]
_qn = torch.stack([
    _w1 * _w0 - _x1 * _x0 - _y1 * _y0 - _z1 * _z0,
    _w1 * _x0 + _x1 * _w0 + _y1 * _z0 - _z1 * _y0,
    _w1 * _y0 - _x1 * _z0 + _y1 * _w0 + _z1 * _x0,
    _w1 * _z0 + _x1 * _y0 - _y1 * _x0 + _z1 * _w0], dim=-1)
_estado = torch.cat([_rb.data.root_link_pos_w, _qn,
                     torch.zeros(env_t.num_envs, 6, device=DEVICE)], dim=-1)
_rb.write_root_state_to_sim(_estado)
env_t.sim.forward()
_ang_depois = _meta_t.erro_angulo_deg()
_delta = float((_ang_depois - _ang_antes).abs().max())
check("item 3: girar o ROBÔ não muda o erro de ângulo da caixa",
      _delta < 1.0,
      f"variação máx {_delta:.3f}° com o robô girado 40° e a caixa intacta "
      f"(antes do conserto ela acompanhava o robô 1:1)")

# item 4 — as duas linhas de log existem e vêm mascaradas por `tarefa_sorteada`
check("item 4: buffers de diagnóstico criados",
      hasattr(env_t, "diag_soma") and hasattr(env_t, "diag_cont")
      and tuple(env_t.diag_soma.shape) == (T.NUM_TASKS, 2))
_fonte_rel = inspect.getsource(
    __import__("g1_multitask.observability", fromlist=["Relatorio"]).Relatorio.__call__)
check("item 4: relatório emite cond_fisica e atribuicao_divergente",
      "cond_fisica" in _fonte_rel and "atribuicao_divergente" in _fonte_rel,
      "se `perf` sobe e `cond_fisica` fica em zero, o crédito é falso")

# ------------------- etiqueta de espaço de ação (o cross-load silencioso de 04/08)
# A decisão é lógica pura sobre um dict, então testa-se com um duplo — instanciar o
# `MultitaskRunner` de verdade exige PPO e mora no `smoke_resume.py`.
print("\n-- etiqueta de espaço de ação --")
from g1_multitask.runner import MultitaskRunner  # noqa: E402


class _RunnerFalso:
    """Só o par de métodos da guarda, com a assinatura vindo de um atributo."""
    _confere_assinatura = MultitaskRunner._confere_assinatura

    def __init__(self, assinatura):
        self._a = assinatura

    def _assinatura(self):
        return self._a


_MT = {"termos": {"joint_pos": "JointPositionAction"}, "dim": 29}
_RES = {"termos": {"joint_pos": "ResidualBFMAction"}, "dim": 29}

check("etiqueta: mesma task carrega",
      _RunnerFalso(_MT)._confere_assinatura({"assinatura": _MT}) is None)

_recusou = False
try:
    _RunnerFalso(_RES)._confere_assinatura({"assinatura": _MT})
except SystemExit:
    _recusou = True
check("etiqueta: checkpoint do multi-tarefa RECUSADO no env residual", _recusou,
      "as duas têm ação 29 e obs 151 desde `dim_c=0` — sem esta guarda o load é "
      "silencioso e o BFM aparece de pé fingindo ser a política")

check("etiqueta: checkpoint SEM etiqueta ainda carrega (só avisa)",
      _RunnerFalso(_MT)._confere_assinatura({"curriculum": {}}) is None,
      "travar aqui quebraria o resume das runs anteriores a 04/08")

_fonte_save = inspect.getsource(MultitaskRunner.save)
check("etiqueta: o `save` grava a assinatura", "assinatura" in _fonte_save)
check("etiqueta: a guarda roda ANTES do bloco de currículo no `load`",
      inspect.getsource(MultitaskRunner.load).index("_confere_assinatura")
      < inspect.getsource(MultitaskRunner.load).index("curriculum"),
      "o `play` passa load_cfg={'actor': True} e sai fora do bloco de currículo")

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
