"""Movimento suave — #1 (action_scale estrutural) + #3 (anti-impacto na mesa), 2026-07-18.

WARM-STARTABLE (não muda shape de obs/ação): o action_scale é aplicado DEPOIS da policy
(mesmas 29 juntas) e o sensor/reward de impacto é treino-only (alimenta reward, nunca a
obs). Carrega o checkpoint das 4 alturas limpo; a LR 5e-4 reassenta a nova dinâmica.

Testa na MESMA distribuição de alturas que já estava treinando (h2_payload: [0.55,0.45,
0.35,0.25] + payload 0.5–5kg) — só ADICIONA os dois fixes de qualidade de movimento, sem
mexer no currículo (isolar o efeito na 0.25 que já estava rodando):

  #1 MOVIMENTO RÁPIDO -> `action_scale_mult=0.8`: encolhe 20% o deslocamento-alvo por
     passo (STRUCTURAL, não compete na soma de reward). É o próprio action_scale do G1, só
     menor. Start conservador; se ainda "correndo", baixa pra 0.7. Se reach/grasp caírem no
     começo, é o warm-start reassentando (LR 5e-4 amortece) — dá ~50-100 it antes de julgar.

  #3 BATE NA MESA -> `impact=-1e-4`: reusa o soft_landing TESTADO do mjlab no sensor novo
     body_table_impact (subtree torso = tronco/braço/punho/mão vs mesa). Pune a força no
     PRIMEIRO contato => aproximar gentil é de graça, só a PANCADA custa. WATCH
     `Metrics/landing_force_mean` (escala real da força) pra recalibrar: se a força média
     for ~centenas de N, −1e-4 pode ser forte (robô evita a mesa e não pega) → baixar; se
     ~dezenas e sem efeito visível → subir.

⚠ Amortecedores empilhados: herda arm_vel(−0.002) + joint_acc(−2.5e-7) do h2_payload e
  agora soma o action_scale. Se grasp/lift despencarem = over-damping — o 1º a tirar é o
  arm_vel (redundante com o action_scale global). Ver [[g1-lift-box-future-refinements]].

WARM-START: do melhor checkpoint que JÁ conhece a 0.25 (4 alturas), LR inicial 5e-4
([[warmstart-lower-lr]] — mudar action_scale é mudar a dinâmica ação→junta).
"""
from dataclasses import replace

from .c2026_07_17_plr_h2_payload import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    foundation=replace(_PREV.foundation, action_scale_mult=0.8),   # #1 estrutural
    reward=replace(_PREV.reward, impact=-1e-4),                    # #3 anti-impacto
)
