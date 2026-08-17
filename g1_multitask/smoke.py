"""Portão local do multi-tarefa. Roda em CPU, em segundos:

    python g1_multitask/smoke.py

**Por que ele existe.** O treino roda na Kaggle, com cota. Um bug de montagem que só
aparece lá custa uma submissão. Aqui custa segundos.

**O que carrega peso não são as asserções.** É a instanciação. Montar o `cfg` e dar um
passo executa todos os termos de recompensa, todas as terminações e todos os managers.
Erro de assinatura no `gated()`, chave de comando fora de ordem, buffer lido antes de
existir — os três aparecem ali, sem asserção nenhuma.

**As asserções cobrem só o que um passo NÃO pega:** bug semântico. Fatia errada não
levanta erro; ela treina mal, em silêncio.

    seção 4   fatias do comando        Categoria C — invalida checkpoint
    seção 5   largura da observação    tem de bater com o checkpoint do warm start
    seção 6   orçamento por tarefa     um peso mudado desiguala as tarefas em silêncio
    seção 7   grafo do currículo       substitui o `sim_curriculo.py`
    seção 8   DR de peso               2 níveis, e o nível 1 tem de conter o 0

**O que ele NÃO cobre, de propósito:** valor de recompensa, convergência, e a ORDEM
temporal exata dos 24 destravamentos. A ordem depende de qual tarefa cruza o portão
primeiro, e isso é runtime. O grafo é estático e é o que se verifica.

⚠️ **Este arquivo mira a API PÓS-REFORMA** (ver `EXPERIMENTO.md`, §9 e §10b). Enquanto a
implementação não chega, as seções 7 e 8 reportam FALHA com o nome do que falta. Isso é
proposital: o arquivo serve de checklist da implementação.

Histórico: a versão anterior tinha 1 832 linhas e 229 asserções, das quais pelo menos 51
testavam conceitos que a reforma removeu (`twist` derivado, `d_morto`, `PARADO`, `ANDAR`,
congelamento, eixo `push`). O `sim_curriculo.py`, com 472 linhas, foi apagado — o grafo
ramificado, a prioridade e o round-robin que o justificavam não existem mais.
"""
from __future__ import annotations

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
NUM_ENVS = 8
PASSOS = 5

OBS_ESPERADA = 154
"""Largura da observação do ATOR. Mudá-la é Categoria C — recomeçar do zero.

A reforma NÃO a move: o `ONEHOT_DIM` continua em 8 mesmo com 5 tarefas, e o `twist_cmd`
continua com 3 números depois de virar `generated_commands` do mjlab. Ver §13."""

PRIVILEGIO_CRITICO = 12
"""Quanto o crítico é maior que o ator. Ele vê `foot_air_time`, `foot_contact`,
`foot_contact_forces` e `foot_height` — grandezas privilegiadas do sim.

O que este número protege: os dois grupos têm de crescer JUNTOS nos canais de comando.
Um canal de comando que o crítico não vê faz ele estimar valor sem saber a tarefa."""

ORCAMENTO = 4.0
"""Sinal de tarefa por passo, igual nas cinco tarefas. Derivado do cfg do fabricante:
`track_linear + track_angular` = 2,0 + 2,0."""

falhas: list[str] = []


def check(nome: str, cond: bool, detalhe: str = "") -> None:
    if cond:
        print(f"  OK    {nome}" + (f"  ({detalhe})" if detalhe else ""))
    else:
        print(f"  FALHA {nome}  {detalhe}")
        falhas.append(nome)


def falta(nome: str, alvo: str) -> None:
    """Reporta uma peça da reforma que ainda não existe em código."""
    print(f"  FALTA {nome}  ({alvo} — ver EXPERIMENTO.md)")
    falhas.append(nome)


# =============================================================== 1. registro
print("\n-- registro da task --")
check("task registrada", g1_multitask.TASK_ID in list_tasks(), g1_multitask.TASK_ID)
check("não colide com as tasks do g1_training",
      sum(1 for t in list_tasks() if "Lift-Box" in t) == 3,
      "Stand / Stand-Step / Lift intactas")


# ============================================ 2. o env monta, reseta e roda
# ESTA é a seção que carrega peso. Ela não tem asserção interessante — o valor
# está em não levantar exceção.
print("\n-- env monta, reseta, roda --")


def sem_dr_instavel(c):
    """O `dr.body_com_offset` corrompe memória em CPU e em GPU (A/B de 30/07 com
    `CUDA_LAUNCH_BLOCKING=1`). Ele está desligado no `knobs.DR`; este pop é rede."""
    c.events.pop("base_com", None)
    return c


cfg = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg.scene.num_envs = NUM_ENVS
env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
env.reset()
acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(PASSOS):
    env.step(acao)
check(f"{PASSOS} steps sem exceção", True)
check("env.active_task existe", hasattr(env, "active_task"))
check("cena com 3 entidades", len(cfg.scene.entities) == 3, str(list(cfg.scene.entities)))
check("action_scale_mult do config ativo",
      ACTIVE.foundation.action_scale_mult == 0.8, "0.8")


# ================================================= 3. nada devolve não-finito
# NaN não levanta exceção. Ele contamina o gradiente e o treino morre horas depois.
print("\n-- tudo finito --")
for grupo in ("actor", "critic"):
    o = env.observation_manager.compute()[grupo]
    check(f"obs[{grupo}] finita", bool(torch.isfinite(o).all()))

passo_reward = getattr(env.reward_manager, "_step_reward", None)
if passo_reward is None:
    falta("reward por termo finito", "reward_manager._step_reward")
else:
    finito = torch.isfinite(passo_reward)
    if bool(finito.all()):
        check("todo termo de reward finito", True, f"{passo_reward.shape[1]} termos")
    else:
        ruins = [n for i, n in enumerate(env.reward_manager.active_terms)
                 if not bool(finito[:, i].all())]
        check("todo termo de reward finito", False, f"não-finitos: {ruins}")


# ================================================== 4. fatias do comando (C)
# Fatia errada NÃO levanta erro. Ela alimenta a rede com um número que significa
# outra coisa, e invalida todo checkpoint anterior.
print("\n-- layout do comando --")
from g1_multitask import commands as C  # noqa: E402

check("lift_target tem 17 números", C.COMMAND_DIM == 17)
for nome, fatia, largura in (("ALVO", C.ALVO, 3), ("FACE", C.FACE, 3),
                             ("DIR", C.DIR, 3), ("ONEHOT", C.ONEHOT, 8)):
    check(f"{nome} tem {largura} canais",
          fatia.stop - fatia.start == largura, f"{fatia}")
check("ALVO em [0:3] — o `target_pos_b` do g1_training lê essa fatia",
      C.ALVO.start == 0 and C.ALVO.stop == 3)
check("as fatias cobrem os 17 sem buraco nem sobreposição",
      [C.ALVO.start, C.ALVO.stop, C.FACE.stop, C.DIR.stop, C.ONEHOT.stop]
      == [0, 3, 6, 9, 17])
check("ordem: lift_target antes de twist",
      list(cfg.commands)[:2] == ["lift_target", "twist"], str(list(cfg.commands)))

if hasattr(C, "DesiredTwistCommand"):
    falta("DesiredTwistCommand removido", "commands.py ainda tem o termo derivado")


# ================================================ 5. largura da observação
print("\n-- observação --")
largura = env.observation_manager.group_obs_dim["actor"][0]
largura_critic = env.observation_manager.group_obs_dim["critic"][0]
check("obs do ator", largura == OBS_ESPERADA, f"{largura} (esperado {OBS_ESPERADA})")
check("crítico = ator + privilégio",
      largura_critic - largura == PRIVILEGIO_CRITICO,
      f"critic={largura_critic} ator={largura}")
check("bit phase apagado", "phase" not in cfg.observations["actor"].terms)


# ============================================== 6. orçamento igual por tarefa
# Um peso mudado sem recalcular a escala afunda a tarefa em silêncio.
print("\n-- orçamento de tarefa --")
presentes = [n for n in T.TERMOS_DE_TAREFA if n in cfg.rewards]
soma = dict.fromkeys(range(T.NUM_TASKS), 0.0)
for nome in presentes:
    termo = cfg.rewards[nome]
    escala = termo.params.get("escala") or {}
    for tarefa in termo.params["tasks"]:
        soma[tarefa] += abs(termo.weight) * float(escala.get(tarefa, 1.0))
for t, v in soma.items():
    check(f"{T.NAMES[t]} soma {ORCAMENTO}", abs(v - ORCAMENTO) < 1e-4, f"deu {v:.4f}")


# ================================================== 7. o grafo do currículo
# Substitui o `sim_curriculo.py`. Ele tinha 472 linhas porque grafo ramificado,
# prioridade e round-robin tornavam a ORDEM não óbvia. Os três saíram.
#
# O que se verifica é o grafo, que é estático. A ordem temporal depende de qual
# tarefa cruza o portão primeiro, e isso é runtime.
print("\n-- grafo do currículo --")
from g1_multitask import curriculum as CU  # noqa: E402

PAIS = getattr(CU, "PAIS", None)
if PAIS is None:
    falta("grafo por PAIS", "curriculum.py ainda usa FILHOS com prioridade e F9")
else:
    raizes = [t for t, p in PAIS.items() if not p]
    check("uma raiz só", len(raizes) == 1, str([T.NAMES[t] for t in raizes]))
    check("as cinco tarefas estão no grafo", len(PAIS) == 5, f"{len(PAIS)}")

    # junção AND: locomover_carregando exige pegar E reorientar
    juncoes = {t: p for t, p in PAIS.items() if len(p) > 1}
    check("exatamente uma junção AND", len(juncoes) == 1,
          str({T.NAMES[t]: [T.NAMES[x] for x in p] for t, p in juncoes.items()}))

    # alcançabilidade e ausência de ciclo, por varredura em largura
    alcancado, fronteira = set(raizes), list(raizes)
    while fronteira:
        atual = fronteira.pop()
        for t, p in PAIS.items():
            if t not in alcancado and all(x in alcancado for x in p):
                alcancado.add(t)
                fronteira.append(t)
    check("toda tarefa é alcançável da raiz", len(alcancado) == len(PAIS),
          str([T.NAMES[t] for t in PAIS if t not in alcancado]))

    check("o congelamento saiu", not hasattr(CU, "_congelamento"))
    check("o round-robin saiu", not hasattr(CU, "AXIS_ORDER") and "rr" not in vars(CU))

print("\n-- contagem dos destravamentos --")
ESPERADO = 12
"""4 aberturas de tarefa + 4 alargamentos de DR + 4 níveis de `pegar_alvo`.

Era 24 antes do congelamento de 17/08 (`T.NIVEIS_ATIVOS`). Descongelar um eixo muda
este número — e é isso que o check protege: mexer num nível sem querer denuncia."""
por_fonte = T.unlock_count()
for fonte, n in por_fonte.items():
    print(f"        {fonte:<24} {n}")
check(f"total = {ESPERADO}", T.total_unlocks() == ESPERADO, f"deu {T.total_unlocks()}")
check("um eixo por tarefa",
      all(len(e) == 1 for e in T.AXES.values()),
      str({T.NAMES[t]: list(e) for t, e in T.AXES.items() if len(e) != 1}))
for morto in ("rumo", "distancia_andar", "push"):
    check(f"o eixo `{morto}` não existe mais", morto not in T.LEVELS)

# O eixo do `pegar` gradua QUANTO erguer, não de onde pegar. Trocar isso de volta pra
# `altura` sem descongelar a rampa recria o bloqueio de 22 mil iterações.
check("o `pegar` usa o eixo `alvo`", list(T.AXES[T.PEGAR]) == ["alvo"],
      str(list(T.AXES[T.PEGAR])))
check("a rampa do `alvo` é FRAÇÃO em (0, 1]",
      all(0.0 < f <= 1.0 for f in T.LEVELS["alvo"])
      and T.LEVELS["alvo"][-1] == 1.0, str(T.LEVELS["alvo"]))
check("a rampa do `alvo` está descongelada",
      T.NIVEIS_ATIVOS["alvo"] == len(T.LEVELS["alvo"]),
      f'{T.NIVEIS_ATIVOS["alvo"]} de {len(T.LEVELS["alvo"])}')
# Com todos os eixos congelados, três tarefas passam a ter a DR como única fonte de
# evento — e o `locomover` fica SEM fonte nenhuma. O conserto é a condição
# `eventos_tarefa[t] == 0` no orquestrador; sem ela o `pegar` nunca abre.
congelados = [e for e, n in T.NIVEIS_ATIVOS.items() if n == 1]
if congelados:
    import inspect  # noqa: E402
    fonte = inspect.getsource(CU.Orquestrador.__call__)
    check("o orquestrador dá o 1º evento a tarefa sem nada a destravar",
          "eventos_tarefa[t] == 0" in fonte, f"eixos congelados: {congelados}")


print("\n-- sinais novos do bloco 4 --")
check("`unload` existe e é só do `pegar`",
      "unload" in cfg.rewards
      and tuple(cfg.rewards["unload"].params["tasks"]) == (T.PEGAR,),
      str(cfg.rewards.get("unload") and cfg.rewards["unload"].params["tasks"]))
check("o sensor de apoio dá força (o `unload` depende dela)",
      any(s.name == "box_support" and "force" in s.fields
          for s in (cfg.scene.sensors or ())),
      str([s.fields for s in (cfg.scene.sensors or ()) if s.name == "box_support"]))
check("`box_at_peito` saiu do `pegar`",
      T.PEGAR not in cfg.rewards["box_at_peito"].params["tasks"],
      str(cfg.rewards["box_at_peito"].params["tasks"]))
check("`sucesso_denso` está FORA do orçamento de tarefa",
      "sucesso_denso" in cfg.rewards
      and "sucesso_denso" not in T.TERMOS_DE_TAREFA)
check("`sucesso_denso` não paga o `locomover`",
      T.LOCOMOVER not in cfg.rewards["sucesso_denso"].params["tasks"],
      str(cfg.rewards["sucesso_denso"].params["tasks"]))
check("o `time_out` é por tarefa, e é time_out=True",
      cfg.terminations["time_out"].time_out is True
      and "limites_s" in cfg.terminations["time_out"].params)
_lim = cfg.terminations["time_out"].params["limites_s"]
check("manipulação mais curta que locomoção",
      all(_lim[t] < _lim[T.LOCOMOVER] for t in T.MANIPULA),
      str({T.NAMES[t]: _lim[t] for t in range(T.NUM_TASKS)}))
check("nenhum limite passa do `episode_length_s`",
      max(_lim) <= cfg.episode_length_s, f"{max(_lim)} vs {cfg.episode_length_s}")
check("sustentar cabe no episódio da manipulação",
      ACTIVE.tolerancia.sustenta_pegar_s < ACTIVE.episodio.manipulacao_s,
      f"{ACTIVE.tolerancia.sustenta_pegar_s} s de {ACTIVE.episodio.manipulacao_s} s")
check("piso de amostragem cabe nas 5 tarefas",
      ACTIVE.curriculum.piso_amostragem * T.NUM_TASKS < 1.0,
      f"{ACTIVE.curriculum.piso_amostragem} × {T.NUM_TASKS}")


# ==================================================== 8. a DR de peso
# O peso NÃO é eixo. São 2 níveis de DR, e o nível 1 tem de CONTER o nível 0 —
# senão a carga leve some do treino no momento em que a DR alarga.
print("\n-- DR de peso --")
pesos = T.LEVELS.get("peso")
if pesos is None:
    falta("tabela de 2 níveis do peso", "T.LEVELS['peso']")
else:
    check("o peso tem 2 níveis", len(pesos) == 2, str(pesos))
    check("o nível 0 é 1 kg", pesos[0] == 1.0, str(pesos[0]))
    if len(pesos) == 2:
        # Só faz sentido com a tabela certa: o sorteio é U(piso, teto), e é ele que
        # torna verdadeira a afirmação de que o nível 1 CONTÉM o nível 0.
        check("o nível 1 contém o nível 0", pesos[1] > pesos[0],
              f"U({pesos[0]}, {pesos[1]})")
check("o peso não é eixo de nenhuma tarefa",
      all("peso" not in e for e in T.AXES.values()),
      str([T.NAMES[t] for t, e in T.AXES.items() if "peso" in e]))


# ======================================================== 9. postura: 1 termo
print("\n-- postura --")
posturas = [n for n in cfg.rewards if n.startswith("post")]
check("um termo de postura só", len(posturas) == 1, str(posturas))
if len(posturas) == 1:
    p = cfg.rewards[posturas[0]]
    check("sem gate por tarefa", "tasks" not in p.params, str(list(p.params)))


# ============================================================= resumo
print("\n" + "=" * 60)
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("tudo OK")
