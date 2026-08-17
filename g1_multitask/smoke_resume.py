"""Portão de salvar-e-retomar. Leva MINUTOS, não segundos:

    python g1_multitask/smoke_resume.py

Separado do `smoke.py` porque instancia PPO de verdade e treina 3 iterações. Rode os
dois antes de submeter pra Kaggle.

**É o portão do plano inteiro.** Com a run fatiada em blocos de 2k-3k, `save`/`load`
dispara de 10 a 15 vezes, e um bug aqui perde o currículo em SILÊNCIO: o treino segue
rodando, só volta pro nível 0. A §15 é explícita — "bug de resume na hora 11 da sessão
3 custa uma semana de cota".

O que se verifica:
  - 3 iterações de PPO rodam ponta a ponta (obs 151, 30 termos, 6 terminações)
  - o estado do currículo sobrevive ao round-trip, bit a bit
  - o `play` NÃO reinjeta o currículo (senão atropela o pin manual)
  - o log de TensorBoard sai onde o `entre_blocos.py` procura
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import g1_multitask  # noqa: F401
from g1_multitask import tasks as T
from g1_multitask.runner import MultitaskRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

DEVICE = "cpu"
falhas: list[str] = []


def check(nome, cond, detalhe=""):
    print(f"  {'OK   ' if cond else 'FALHA'} {nome}"
          + (f"  ({detalhe})" if detalhe else ""))
    if not cond:
        falhas.append(nome)


def sem_dr_instavel(c):
    """`dr.body_com_offset` corrompe a heap no backend CPU do warp. Ver knobs.DR."""
    if DEVICE == "cpu":
        c.events.pop("base_com", None)
    return c


print("\n-- T15: treinar 3 iterações, salvar, retomar --")
import shutil  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import asdict as _asdict  # noqa: E402

from mjlab.rl import RslRlVecEnvWrapper as _Wrap  # noqa: E402

tmp = pathlib.Path(tempfile.mkdtemp(prefix="mt_smoke_"))
cfg_tr = sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID))
cfg_tr.scene.num_envs = 4
cfg_tr.episode_length_s = 1.0            # episódio curto: resets de verdade em 3 iters
env_tr = ManagerBasedRlEnv(cfg=cfg_tr, device=DEVICE)
rlc = load_rl_cfg(g1_multitask.TASK_ID)
runner_tr = MultitaskRunner(_Wrap(env_tr, clip_actions=rlc.clip_actions),
                            _asdict(rlc), str(tmp), DEVICE)
runner_tr.learn(num_learning_iterations=3, init_at_random_ep_len=True)
check("3 iterações de PPO rodaram", True)

# mexe no estado do currículo pra o round-trip ter o que provar
#
# ⚠️ As chaves de célula saem de `T.eixo_de`, e não digitadas: o eixo do `pegar` já
# mudou de `altura` para `alvo` (17/08) e a versão com o nome fixo quebrou com
# `KeyError`. Derivar aqui faz o portão sobreviver à próxima troca de eixo.
orq_tr = env_tr.curriculum_manager.get_term_cfg("orquestrador").func
CEL_LOCO = (T.LOCOMOVER, T.eixo_de(T.LOCOMOVER))
CEL_PEGAR = (T.PEGAR, T.eixo_de(T.PEGAR))
orq_tr.abertos[CEL_LOCO] = 1
orq_tr.abertas = [T.LOCOMOVER, T.PEGAR, T.REORIENTAR]
orq_tr.eventos = 7
orq_tr.eventos_tarefa[T.LOCOMOVER] = 2
orq_tr.dr_peso[T.PEGAR] = True
orq_tr.perf[CEL_PEGAR][0] = 0.875   # exato em float32
antes = orq_tr.state_dict()

ckpt = tmp / "model_teste.pt"
runner_tr.save(str(ckpt))
check("checkpoint escrito", ckpt.exists(), f"{ckpt.stat().st_size // 1024} KB")

# runner NOVO, env NOVO: é assim que um bloco seguinte retoma
env_r = ManagerBasedRlEnv(cfg=sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID)),
                          device=DEVICE)
runner_r = MultitaskRunner(_Wrap(env_r, clip_actions=rlc.clip_actions),
                           _asdict(rlc), None, DEVICE)
runner_r.load(str(ckpt))
orq_r = env_r.curriculum_manager.get_term_cfg("orquestrador").func
check("eventos sobreviveram ao resume", orq_r.eventos == 7, f"deu {orq_r.eventos}")
check("tarefas abertas sobreviveram",
      orq_r.abertas == [T.LOCOMOVER, T.PEGAR, T.REORIENTAR],
      str([T.NAMES[t] for t in orq_r.abertas]))
check("nível de eixo sobreviveu", orq_r.abertos[CEL_LOCO] == 1)
check("perf sobreviveu bit a bit",
      float(orq_r.perf[CEL_PEGAR][0]) == 0.875)
# Os dois estados novos da reforma. O `eventos_tarefa` é o PORTÃO DO FILHO: perdê-lo
# no resume reabriria a cadeia do zero, em silêncio. O `dr_peso` é a DR de carga.
check("eventos_tarefa sobreviveu (é o portão do filho)",
      orq_r.eventos_tarefa[T.LOCOMOVER] == 2,
      f"deu {orq_r.eventos_tarefa[T.LOCOMOVER]}")
check("dr_peso sobreviveu", orq_r.dr_peso[T.PEGAR] is True,
      str(orq_r.dr_peso))

# o `play` NÃO pode reinjetar o currículo: lá você fixa tarefa/nível na mão
env_p = ManagerBasedRlEnv(cfg=sem_dr_instavel(load_env_cfg(g1_multitask.TASK_ID)),
                          device=DEVICE)
runner_p = MultitaskRunner(_Wrap(env_p, clip_actions=rlc.clip_actions),
                           _asdict(rlc), None, DEVICE)
runner_p.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
orq_p = env_p.curriculum_manager.get_term_cfg("orquestrador").func
check("play NÃO reinjeta o currículo (preserva o pin manual)",
      orq_p.eventos == 0 and orq_p.abertas == [T.LOCOMOVER],
      f"eventos={orq_p.eventos}, abertas={[T.NAMES[t] for t in orq_p.abertas]}")

logs = list(tmp.rglob("events.out.tfevents.*"))
check("log de tensorboard escrito (o relatório entre blocos lê daqui)",
      len(logs) > 0, str(len(logs)))
print(f"        log de teste em {tmp}")


print(f"\n{'=' * 60}")
if falhas:
    print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
    sys.exit(1)
print("portão de resume: OK — pode submeter")
