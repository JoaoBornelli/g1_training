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

import math

from mjlab.rl import MjlabOnPolicyRunner

__all__ = ["RunnerV3"]

# Termos da observação que ficam FORA do normalizador (média 0, desvio 1; `algoritmo.fixa_normalizador`): os one-hot,
# os canais do currículo, que são constantes por milhares de iterações, e o `mapa`, que já vem em [0, 1] e é ruído no
# terreno plano (normalizado, ele seria 882 canais de ruído com desvio 1). Nomes de `observacoes.grupos`.
_FIXOS = ("tarefa", "mapa", "alvo_macro", "niveis", "degrau", "massa", "tarefa_critico")


class RunnerV3(MjlabOnPolicyRunner):
    """O runner do mjlab, mais o estado do `Curriculo` no checkpoint (chave `v3_curriculo`), o std por junta e as
    fatias da observação que ficam fora do normalizador (`_fixa_obs`)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._embrulha_log()
        self._fixa_obs()

    def _fixa_obs(self) -> None:
        """Entrega ao algoritmo as fatias dos termos `_FIXOS` do ator e do crítico, pelos NOMES dos termos.

        ⚠ O layout é posicional: a fatia sai de `active_terms` e `group_obs_term_dim`, nunca de um índice escrito à mão.
        A trava é o tamanho: com normalizador, a soma dos termos tem de ser a entrada da rede.
        """
        om = self.env.unwrapped.observation_manager
        fatias = {}
        for grupo in ("actor", "critic"):
            i, fatias[grupo] = 0, []
            for nome, dim in zip(om.active_terms[grupo], om.group_obs_term_dim[grupo]):
                n = math.prod(dim)
                if nome in _FIXOS:
                    fatias[grupo].append(slice(i, i + n))
                i += n
            nz = getattr(self.alg, grupo).obs_normalizer
            assert not hasattr(nz, "_mean") or nz._mean.shape[-1] == i, (grupo, nz._mean.shape, i)
        self.alg.fatias_fixas = fatias

    def _embrulha_log(self) -> None:
        """Embrulha `logger.log` e grava `Policy/std/<junta>` = média em 0 do `action_std` (S §14, H6).

        ⚠ O rsl_rl grava só `Policy/mean_std` (logger.py:211): a média das 29 juntas esconde qual junta explora.
        Os nomes são os da ação (`target_names`, a ordem do vetor de ação). O rsl_rl chama `log` só por nome.
        """
        original = self.logger.log

        def log(**kw):
            original(**kw)
            if self.logger.writer is None:
                return
            std = kw["action_std"]
            std = std.mean(dim=0) if std.dim() == 2 else std
            nomes = self.env.unwrapped.action_manager.get_term("joint_pos").target_names
            for nome, v in zip(nomes, std.tolist()):
                self.logger.writer.add_scalar(f"Policy/std/{nome}", v, kw["it"])

        self.logger.log = log

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
