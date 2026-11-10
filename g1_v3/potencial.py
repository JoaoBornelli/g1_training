"""Potencial e termos puros da v3 (B1 do plano, §6.3). Sem mjlab, sem mujoco, sem o pacote.

Cada função recebe tensores (E,) do chamador e os escalares dos knobs como argumento. Nenhum
número de treino nasce aqui (plano §1.5). A forma do núcleo vem de g1_limpo/recompensas.py
(`_alcancar`, `exp(−(d/σ)²)`); o resto é novo, pela S §8.2, §9.1, §9.2 e §19.
"""
from __future__ import annotations

import functools
import operator

import torch


def nucleo(d: torch.Tensor, sigma) -> torch.Tensor:
    """exp(−(d/σ)²). Kernel de aproximação ao alvo corrente (S §9.1 [C20]).

    Em d = σ vale e⁻¹. σ é o número do início do alvo (distância naquele instante), por env.
    """
    return torch.exp(-((d / sigma) ** 2))


def rampa(x: torch.Tensor, x_lim, delta) -> torch.Tensor:
    """clip(1 − (x − x_lim)/Δ, 0, 1). Portão contínuo de grandeza (S §9.1, B04).

    ⚠ Não é 0 ou 1: com 0/1 a derivada é zero na região onde o robô está, e o termo não
    tem gradiente. Abaixo de x_lim vale 1; acima, cai linear até 0 em x_lim + Δ.
    """
    return torch.clamp(1.0 - (x - x_lim) / delta, 0.0, 1.0)


def phi(k_principal, portoes: list[torch.Tensor]) -> torch.Tensor:
    """Φ = k_principal · Π g (S §9.1). Proximidade ao alvo corrente, em [0, 1] se k ≤ 1.

    `portoes` são as rampas de invariante e de faixa já calculadas pelo chamador.
    """
    return k_principal * functools.reduce(operator.mul, portoes)


def fator_pouso(v_vertical: torch.Tensor, fundo_sobre_laje: torch.Tensor, sigma, altura,
                altura_inicio) -> torch.Tensor:
    """1 − s·(1 − exp(−(v/σ)²)), com s = clip((altura_inicio − fundo)/(altura_inicio − altura), 0, 1).

    S §9.1 (A10) e D3 da Parte 0 (dono, 29/10): o fator entra gradual em `altura_inicio` (0,10 m)
    e fica pleno em `altura` (0,05 m). Acima de `altura_inicio` vale 1; abaixo de `altura`, o
    exp(−(v/σ)²) inteiro. Sem degrau: com o degrau antigo em `altura`, Φ cortava no passo de
    entrada na janela.
    Multiplica o k_principal da caixa no B1 e no B2. Sem termo novo: é só o fator do pouso.
    ⚠ A velocidade entra ao quadrado: cair e subir pagam igual. O sinal não importa aqui.
    """
    s = torch.clamp((altura_inicio - fundo_sobre_laje) / (altura_inicio - altura), 0.0, 1.0)
    return 1.0 - s * (1.0 - torch.exp(-((v_vertical / sigma) ** 2)))


def custo_aperto(F_esq: torch.Tensor, F_dir: torch.Tensor, F_ref) -> torch.Tensor:
    """min(F_esq, F_dir)/F_ref. Custo do aperto, multiplicado por −w (S §8.2, B02).

    Linear (caso 9, decisão do dono 30/10): sem aperto custa 0 e a derivada é 1/F_ref em todo
    ponto. A forma 1 − exp(−F/F_ref) saturava: a 117,5 N a derivada era 3,7e−5 por N por s no
    degrau 5, canal morto. Hinge linear, como o FORGE e o legged_gym (limiar 0). Com F = F_ref vale 1.
    """
    return torch.minimum(F_esq, F_dir) / F_ref


def dobradica(x: torch.Tensor, limite) -> torch.Tensor:
    """relu(x − limite). Dobradiça das cercas de postura: zero dentro, custo fora (S §9.3)."""
    return torch.relu(x - limite)


def curso(q: torch.Tensor, centro: torch.Tensor, meio: torch.Tensor) -> torch.Tensor:
    """|q − c|/m (E, J): fração do curso de cada junta; 1 no batente do modelo (H1, P13).

    O `batente` (H3) e a métrica `fracao_do_curso` (H5b) leem esta mesma conta, para o curso ser um número só.
    """
    return torch.abs(q - centro) / meio


def excesso_de_junta(q: torch.Tensor, centro: torch.Tensor, meio: torch.Tensor, mole) -> torch.Tensor:
    """Σ_j relu(|q − c|/m − mole)/(1 − mole) (S §9.2): zero até o limite mole, 1 no batente.

    `q` é (E, J); `centro` e `meio` são (J,). Soma no último eixo. Com mole = 0,95 a rampa
    fica no último 5% do curso (a receita da Unitree cobra em radianos, e ela ficava rasa
    nas juntas de curso largo: ver a spec §9.2).
    """
    return torch.relu(curso(q, centro, meio) - mole).sum(dim=-1) / (1.0 - mole)


def rastreio_normalizado(erro2: torch.Tensor, c: torch.Tensor, v2: torch.Tensor,
                         sigma, sigma_zero, limiar, piso) -> torch.Tensor:
    """Rastreio normalizado pelo comando (S §19, conflito 5; PD-49).

    c ≥ limiar: (exp(−e²/σ²) − exp(−c²/σ²)) / (1 − exp(−c²/σ²)). Seguir o comando vale 1,
    parado vale 0. c < limiar ("sem comando"): exp(−v²/σ₀²) − 1, que só cobra o deslize.
    O piso corta o que a fórmula deixa abaixo de −1 (perto do limiar, a divisão explode).

    ⚠ NaN: `c` é trocado por `max(c, limiar)` no ramo normalizado. Sem isso, com c = 0 o
    denominador é 0 e o `torch.where` propaga NaN no gradiente mesmo no ramo não escolhido.
    ⚠ Precisão: 1 − exp(x) e exp(a) − exp(b) viram −expm1(x) e expm1(a) − expm1(b). Com
    float32 e c perto de 0,02, o `1 − exp` perde quase todos os dígitos.
    """
    c_n = torch.clamp(c, min=limiar)
    s2 = sigma * sigma
    cs = -(c_n * c_n) / s2
    normal = (torch.expm1(-erro2 / s2) - torch.expm1(cs)) / (-torch.expm1(cs))
    deslize = torch.exp(-v2 / (sigma_zero * sigma_zero)) - 1.0
    return torch.clamp(torch.where(c >= limiar, normal, deslize), min=piso)


def L_alvo(v_cmd: torch.Tensor, L_min, fator_T) -> torch.Tensor:
    """|v|·clamp(2·L_min/|v|, lo, hi)/2, com (lo, hi) = fator_T (S §19, conflito 6).

    Comprimento do passo esperado para a velocidade: 0,175 m a 0,25 m/s, 0,20 m a 0,5 m/s,
    0,40 m a 1,0 m/s, 0,60 m a 1,5 m/s (com L_min 0,20 e fator_T (0,8; 1,4)).
    ⚠ Divisão por |v| com `|v|` trocado por `max(|v|, 1e-6)` só no denominador: com v = 0 o
    gradiente daria NaN (0·∞). O multiplicador é |v| exato, então L_alvo(0) = 0.
    """
    lo, hi = fator_T
    m = torch.clamp(torch.abs(v_cmd), min=1e-6)
    return torch.abs(v_cmd) * torch.clamp(2.0 * L_min / m, lo, hi) / 2.0
