"""O runner do PPO, com UMA adição: o estado do currículo da v3 vai para o checkpoint.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

POR QUE ISTO EXISTE. As sessões do Colab e do RunPod morrem, e um bloco de iterações atravessa
vários reinícios. O `mjlab` persiste só o `common_step_counter` (`mjlab/rl/runner.py:73`). Sem
isto, o `Curriculo` (níveis, degraus, a tabela das células e os balanceadores B/C) volta ao
começo a cada retomada, e o currículo fica não-monotônico. Origem: `g1_limpo/runner.py`.

⚠ O caminho é o SUPORTADO, e não um monkeypatch: `register_mjlab_task` aceita `runner_cls`
(`mjlab/tasks/registry.py:27`), e o `tarefa.py` (D10) o passa.

⚠ O QUE VAI: `env.v3_cur.estado()` (§6.7), em tipos do Python. O QUE NÃO VAI: o estado de
episódio (cadeia, alvos, buffers de comando). Um resume começa com todos os envs em reset, e
salvá-los criaria a chance de restaurar um alvo de uma pose que não existe mais.
"""
from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner

__all__ = ["RunnerV3"]


class RunnerV3(MjlabOnPolicyRunner):
    """O runner do mjlab, mais o estado do `Curriculo` no checkpoint (chave `v3_curriculo`)."""

    def save(self, path: str, infos=None) -> None:
        infos = {**(infos or {}), "v3_curriculo": self.env.unwrapped.v3_cur.estado()}
        super().save(path, infos)

    def load(self, path: str, *a, **kw) -> dict:
        carregado = super().load(path, *a, **kw)
        estado = (carregado or {}).get("v3_curriculo")
        if not estado:
            # ⚠ Não falha: um checkpoint anterior à v3 não tem a chave, e o currículo novo
            # começa do zero (o aviso é para o dono não estranhar a rampa no início).
            print("[g1_v3] checkpoint SEM estado de currículo: o currículo recomeça do zero. "
                  "Esperado só ao retomar de um checkpoint anterior à v3.")
            return carregado
        self.env.unwrapped.v3_cur.carrega(estado)
        print(f"[g1_v3] currículo restaurado: it={estado['it']}")
        return carregado
