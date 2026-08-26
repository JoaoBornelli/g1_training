"""O runner do PPO, com UMA adição: o estado do currículo vai para o checkpoint.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

POR QUE ISTO EXISTE. O Colab e o Kaggle MATAM SESSÃO, e um bloco de 5000 iterações
atravessa vários reinícios. O `mjlab` persiste apenas o `common_step_counter`
(`mjlab/rl/runner.py:73`) — nada do nosso currículo. Sem isto:

  · a rampa de ~400 iterações do balanço de forma é RE-PAGA a cada reinício, e o
    currículo fica não-monotônico: ele desce, a sessão morre, ele volta ao topo;
  · o nível por env volta a zero, jogando fora todo o avanço de dificuldade;
  · o contador de carência reinicia, o que ATRASA o portão em 200 iterações por sessão.

⚠ O caminho é o SUPORTADO, e não um monkeypatch: `register_mjlab_task` aceita
`runner_cls` (`mjlab/tasks/registry.py:27`), e o `launch_training` o resolve por
`load_runner_cls(task_id)`.

⚠ O QUE VAI, e por quê cada um:

    alvo, dur_loco, dur_manip, razao   as EMAs e a fatia — a rampa em si
    iters_balanco                      a carência, contada de quando o balanço começou
    abriu                              se o portão já abriu alguma vez
    nivel (por env)                    a dificuldade conquistada
    elo (por env)                      para as durações do 1º reset após o resume
                                       serem atribuídas ao lado certo

⚠ O QUE **NÃO** VAI: os buffers do termo de comando (cadeia, passo, sustain, σ). Eles
são estado de EPISÓDIO, não de currículo. Um resume começa com todos os envs em reset,
portanto eles nascem corretos de qualquer forma — e salvá-los criaria a chance de
restaurar um σ de uma pose que não existe mais.
"""
from __future__ import annotations

import torch

from mjlab.rl import MjlabOnPolicyRunner

__all__ = ["RunnerComEstadoDeCurriculo", "CHAVES_ESCALARES", "CHAVES_POR_ENV"]

# ⚠  e  VAO, e sao o que torna a rampa resume-safe: o
#  que o mjlab restaura e ABSOLUTO, logo sem o passo em que o
# balanco comecou a carencia seria recontada do zero a cada sessao.
CHAVES_ESCALARES = ("alvo", "dur_loco", "dur_manip", "razao",
                    "passo_inicial", "ultimo_degrau",
                    "iters_balanco", "abriu", "sorteio")
CHAVES_POR_ENV = ("limpo_nivel", "limpo_elo")


class RunnerComEstadoDeCurriculo(MjlabOnPolicyRunner):
    """O runner do mjlab, mais o estado do currículo no checkpoint."""

    def _env_cru(self):
        return self.env.unwrapped

    def save(self, path: str, infos=None) -> None:
        e = self._env_cru()
        estado: dict = {}
        forma = getattr(e, "limpo_forma", None)
        if isinstance(forma, dict):
            estado["forma"] = {c: float(forma[c]) for c in CHAVES_ESCALARES
                               if c in forma}
        for nome in CHAVES_POR_ENV:
            buf = getattr(e, nome, None)
            if buf is not None:
                estado[nome] = buf.detach().cpu().clone()
        infos = {**(infos or {}), "limpo_curriculo": estado}
        super().save(path, infos)

    def load(self, path: str, *a, **kw) -> dict:
        carregado = super().load(path, *a, **kw)
        estado = (carregado or {}).get("limpo_curriculo")
        if not estado:
            print("[g1_limpo] checkpoint SEM estado de currículo: a rampa recomeça "
                  "no piso. Esperado só ao retomar de um checkpoint anterior à F5.")
            return carregado

        e = self._env_cru()
        if "forma" in estado:
            from g1_limpo.curriculo import garante_forma
            f = e.cfg.curriculum["forma"].params["f"]
            st = garante_forma(e, f)
            st.update(estado["forma"])
            print(f"[g1_limpo] currículo restaurado: alvo={st['alvo']:.3f} "
                  f"razao={st['razao']:.3f} iters_balanco={st['iters_balanco']:.0f}")

        for nome in CHAVES_POR_ENV:
            if nome not in estado:
                continue
            buf = getattr(e, nome, None)
            salvo = estado[nome]
            if buf is None:
                continue
            # ⚠ O número de envs pode MUDAR entre sessões (o Kaggle dá 2 GPUs num dia e
            # 1 no outro). Copiar o que cabe é melhor que explodir ou que jogar tudo
            # fora, e o resto fica no default.
            k = min(buf.numel(), salvo.numel())
            buf[:k] = salvo[:k].to(buf.device, buf.dtype)
            if k < buf.numel():
                print(f"[g1_limpo] {nome}: checkpoint tem {salvo.numel()} envs e a "
                      f"sessão tem {buf.numel()}; os {buf.numel() - k} restantes "
                      f"ficam no default.")
        return carregado


def _ignora(*_a, **_k) -> None:      # pragma: no cover
    del _a, _k
    _ = torch
