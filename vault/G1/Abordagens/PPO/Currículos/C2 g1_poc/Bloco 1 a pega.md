---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: validada
verificado: 2026-10-06
periodo: 18–20/08/26
resumo: "squeeze saturou e o robô prensava a caixa; unload deu o 1º sucesso do projeto (it 3080) e a caixa subiu 16,5 cm no model_5100."
---
## Hipótese
Com `squeeze` (força normal de palma) o gradiente chega à coordenada do aperto e a caixa sobe.

## Método
Diagnóstico herdado do C1: o `lift` pagava +0,34/s por cm de subida, mas `d(recompensa)/d(força de aperto) = 0`: antes de o atrito vencer o peso a caixa não se move. `squeeze` usa só a componente NORMAL ao pad (normal pela orientação do site; o campo `normal` do sensor perde sentido com `reduce=netforce`).

## Resultado
Bloco 1, 1884 iterações: mãos na caixa, sucesso ZERO. A sonda mostrou que ele PRENSAVA a caixa contra a prateleira (aperto 647% de `F_ref`, apoio mediano 138% do peso, subida 0,0 cm; `squeeze` saturado em `tanh(6,47)`, derivada 1e-5). Com o `unload` somado: primeiro sucesso do projeto na it 3080 (`episode_success` 0,0060); no `model_5100`, subida 16,5 cm, fecho em 63,4%.

## Por que parou
Resolvido; o problema passou a ser a locomoção e os freios.

## O que herda
- A projeção na normal impede que o empurrão para baixo seja PAGO como aperto, não que ele ACONTEÇA de graça.
- `unload = clamp(1 − F_apoio/m·g) × preensão × não_caiu × caixa_valida`, peso 2,0. Os dois gates fecham dois hacks: derrubar a caixa e empurrá-la para fora do tampo.
- Freio por passo global chega antes da fase: na it 3080 `joint_vel_hinge` + `action_rate` consumiam 99,1% das penalidades e 100% do positivo (líquido −0,03), e o `contato_ilegal` subiu de 6,4% para 17,5% — escorar economiza esforço.
