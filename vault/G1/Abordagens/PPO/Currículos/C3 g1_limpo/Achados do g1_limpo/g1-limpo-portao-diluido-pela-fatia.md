---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "O portao da forma no g1_limpo lia media diluida pela fatia de manipulacao, e isso tinha teto fechado"
---
O portao da `forma` lia `eficiencia_min.mean()` sobre TODOS os envs. Nos elos de
manipulacao o twist e forcado a zero, portanto nenhum segmento valido fecha e
`eficiencia_min` fica em ZERO EXATO. A rampa criava os proprios zeros que fechavam o
portao. Consertado em 2026-08-31, commit `0b7d925`, com mascara `segmentos > 0`.

O laco tem ponto fixo, e ele e um teto: `efic x (1 - fatia) = limiar`. Com efic 0,80 e
limiar 0,50, a fatia de manipulacao nao passava de 0,375 e a `forma` nao descia de 0,63.
O destino `alvo_loco_min = 0,30` era inalcancavel por construcao.

Prova: bloco 6 it 785, efic de quem anda 0,80, fatia 0,272, media prevista 0,582 contra
0,5844 medida.

⚠ `alvo_loco_min = 0,30` e 30% de LOCOMOCAO, logo 70% de manipulacao. Eu li ao contrario
uma vez e a conclusao mudou: no destino, a escada lendo o canal cru exigiria 1,67 de
eficiencia, acima do teto de 1,0.

**Why:** metrica gateada por comando ativo vale zero em quem nao recebeu o comando.
Media sobre a frota inteira transforma isso em sinal de incompetencia.

**How to apply:** para LER marcha no painel, use `seg_proj / seg_pedido`. Os dois sao
somas sobre a frota, diluidas pelo mesmo fator, portanto a razao cancela a diluicao e
nao precisa do `forma`. Ele e denso.

⚠ NAO use `eficiencia_min / Curriculum/forma` para julgar. Aquilo e subamostra de ~14%
dos resets, com media por evento nao ponderada (`command_manager.py:102`). Ele oscilou
0,758 -> 0,448 -> 0,738 em 300 iteracoes sem nada acontecer, e eu chamei regressao de
marcha em cima do ponto do meio. A razao mostrou 0,744 no mesmo ponto.

⚠ `Curriculum/elo` e a fracao de ENVS em locomocao; `Curriculum/forma` e a probabilidade
de sorteio por EPISODIO. Elas divergem porque episodio de locomocao dura mais. Com as
duas mais a duracao media da frota sai a duracao por tipo.

Relacionado: g1-poc-portao-forma-mede-sobrevivencia, g1-andar-como-rastreio-de-velocidade
