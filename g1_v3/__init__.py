"""g1_v3 — loco-manipulação de caixa do G1, pacote novo treinado do zero.

Objetivo (S §1, enunciado em specs/g1-limpo-comportamento.md): o G1 anda até a caixa,
pega, carrega e bota no alvo. A v3 troca a renda por estado do g1_limpo por um potencial
de progresso em alvos macro de IK, e as quatro fases de cadeia por dois balanceadores,
com refino em degraus por tarefa.

Regra do pacote: zero import de código do projeto (g1_limpo, g1_poc, g1_multitask,
g1_training). O que fica é COPIADO para cá (S §5). Os módulos puros importam só math,
dataclasses, numpy, torch e a stdlib, nunca o mjlab nem o mujoco: knobs, alvos,
potencial, balanceador, estimador e percepcao_lidar. Eles também não importam uns aos
outros; recebem as seções do knobs como argumento. Motivo: o sim do robô real e o
robô carregam cada arquivo por caminho, sem o pacote (PD-43).

Este __init__ não importa nada. O registro da task do mjlab mora em g1_v3/tarefa.py
(PD-1): importar um módulo puro não pode executar o mjlab.
"""
