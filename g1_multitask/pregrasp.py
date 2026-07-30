"""A pose de pré-grasp: braços posicionados com as palmas TOCANDO a caixa.

Resolvida por IK numérica uma vez, offline, e congelada aqui. Provenance completa
abaixo — número sem justificativa é número copiado.

⚠️ **Ela NÃO segura a caixa, e isso é de propósito.** As palmas ficam em ±0.10 m,
exatamente na face de um cubo de meia-largura 0.10 -> contato com **força normal
zero** -> atrito zero -> a caixa escorrega. Medido: com ação nula ela cai 22 cm em
0.5 s e pousa na prateleira.

**Segurar é o que a tarefa ensina.** A condição de spawn monta a SITUAÇÃO ("a caixa
está nas suas mãos"), não entrega o resultado. Pré-apertar as palmas para dentro da
face resolveria de graça exatamente a habilidade que o `parado c/ caixa` existe para
treinar.

Consequência disso, e é o que faz o episódio ser ganhável: quem nasce segurando tem
comando pré-gatilho `parado c/ caixa`, não `parado` (§4 do doc). Com o atraso de
gatilho de até 2 s, a caixa já teria ido embora antes de a política receber objetivo.
"""
from __future__ import annotations

POSE_PRE_GRASP: dict[str, float] = {
    "left_shoulder_pitch_joint": +0.1963,
    "left_shoulder_roll_joint": +0.1171,
    "left_shoulder_yaw_joint": -0.3762,
    "left_elbow_joint": -0.0603,
    "left_wrist_roll_joint": +0.2398,
    "left_wrist_pitch_joint": -0.7740,
    "left_wrist_yaw_joint": +0.4654,
    "right_shoulder_pitch_joint": +0.1959,
    "right_shoulder_roll_joint": -0.1179,
    "right_shoulder_yaw_joint": +0.3768,
    "right_elbow_joint": -0.0601,
    "right_wrist_roll_joint": -0.2398,
    "right_wrist_pitch_joint": -0.7742,
    "right_wrist_yaw_joint": -0.4665,
}
"""14 juntas de braço. Como saiu, e como conferir de novo:

**Objetivo da IK, dois termos.** Posição: os sites `left_palm`/`right_palm` em
(0.20, ±0.10, +0.15) no frame da PELVE — os "pontos de pega" da §14, que são o
`alvo_peito_b` mais/menos meia-largura da caixa. Orientação: o eixo FINO do pad
(`y` local, `size=(0.035, 0.008, 0.045)`) paralelo ao eixo `y` do mundo, que é a
normal das faces laterais da caixa.

**Resíduo medido:** posição **0.0 mm** nas duas palmas, pad **0.0°** fora do plano
da face. A solução saiu simétrica esquerda/direita, o que indica que o otimizador
achou a pose natural e não um canto do espaço de juntas.

**A orientação é obrigatória no objetivo, não enfeite.** Resolvendo só posição, os
punhos saem arbitrários: na primeira tentativa o pad esquerdo apontava para `+y`
(para FORA, invertido) e o direito ficava 52° fora do plano. Contato de quina, não
de face — e aí nem apertando a preensão se estabelece.

**Refazer:** `scipy.optimize.least_squares` sobre as 14 juntas, com bounds em
`jnt_range`, resíduo = [erro de posição (3) , 0.10 × (eixo_fino.x, eixo_fino.z) (2)]
por palma. O peso 0.10 faz 1.0 de erro de normal custar como 10 cm de posição.
"""

CHAVE_JUNTAS = tuple(POSE_PRE_GRASP)
