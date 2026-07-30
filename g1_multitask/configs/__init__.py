"""Switchboard dos configs. `ACTIVE` aponta pro treino que está rodando agora.

Voltar a um treino antigo = trocar 1 linha aqui. Mesmo padrão do
`g1_training/skills/lift/configs/__init__.py`.

Cadeia: baseline (§14, herdado) -> [próximo: calibrado pela Tarefa 12]
"""
from .c2026_07_30_baseline import KNOBS as ACTIVE

__all__ = ["ACTIVE"]
