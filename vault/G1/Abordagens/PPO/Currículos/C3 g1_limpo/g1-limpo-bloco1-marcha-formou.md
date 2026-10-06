---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
---
# g1-limpo-bloco1-marcha-formou (original)

Corpo original da memória `g1-limpo-bloco1-marcha-formou`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

O g1_limpo andou no bloco 1. Verificado 2026-08-27 no log do Kaggle.

`razao_marcha` foi de −0,136 (it 254) para **0,468 (it 800)**. O erro caiu de 91% para
50,3% do comando, o que dá **0,380 m/s** de velocidade útil. Ao mesmo tempo `fell_over`
caiu para 0,083, a duração chegou a 984 de 1000 passos, e `tempo_de_voo`,
`pico_de_altura`, `landing_force` e `momento_angular` subiram juntos.

A marcha é de cadência alta e folga baixa: passo de ~3 cm, folga de **~18 mm** (18% do alvo
de 0,10 m), escorrego de 0,258 m/s. Não é bonita. É locomoção medida.

⚠ Os 18 mm vêm do `Metrics/peak_height_mean` do fabricante. O nosso `pico_de_altura` lia
5 mm porque é diluído por 3,6×. Ver g1-limpo-pico-altura-diluido.

**Why:** a fase 1 sem manipulação (fatia 0,95) reproduziu o resultado do multitask. Isto
confirma g1-locomocao-primeiro-fatia-100 num terceiro módulo.

**O portão abriu na it ~963.** Na it 1000: `razao_marcha` 0,553 (erro em 38,8% do comando,
0,468 m/s), duração no teto de 1000 passos, `fell_over` 0,0000, e `Curriculum/forma` caiu de
0,9500 para 0,8886. A marcha CONSOLIDOU: `tempo_de_voo` dobrou para 154 ms, folga para
23,8 mm, e ao mesmo tempo escorrego, força de pouso, momento angular e aceleração de ação
CAÍRAM. Velocidade subindo com esses quatro caindo distingue marcha de arrasto rápido.

**How to apply:** a rampa (0,02 a cada 12 iters) leva a fatia de 0,8886 ao piso de 0,30 por
volta da it 1353. A histerese de 0,80 devolve fatia se a razão cair abaixo de 0,40, e ela é
mais rígida que a minha linha de 0,35 — use a do código. Uma reversão é o mecanismo
funcionando; só intervir se a rampa OSCILAR, e o movimento é `limiar_portao` 0,50 -> 0,65.
NÃO mexer em peso de marcha: são idênticos ao módulo que andou.

## O preço do desenho simultâneo, MEDIDO na it 3658

A entropia SUBIU sozinha: `std` 0,43 -> 0,60 e entropia 14,7 -> 25,5, porque a fatia de
manipulação cresceu e exploração voltou a valer recompensa. Confirma que o `std` é
equilíbrio fixado pela tarefa atual, não cronograma. Ver g1-mjlab-api-armadilhas.

A manipulação acordou: `squeeze` 0,0002 -> 0,0664, `postura_ereta` saiu do ZERO exato para
0,0256 (exige rampa × preensão × descarga simultâneas — é pegada com erguimento), `unload`
0,14 -> 0,33.

E a marcha pagou: balanço 398 -> 146 ms, escorrego +31%, força de pouso +21%, rastreio
angular −65%, `action_rate` +68%, recompensa −30%, `razao_marcha` 0,514 -> 0,462.

O sistema ESTACIONOU na banda morta (0,462 entre 0,40 e 0,50) e a fatia congelou em 0,787.
O mecanismo está defendendo a marcha; a run converge para 79/21 em vez de 70/30.

`sucesso` segue 0,0000, e o motivo é legível: `avancos` (0,117) ≈ `fatia_cadeia` (0,150),
logo o 1º elo das cadeias de 2 elos completa e o 2º nunca. É a TRANSIÇÃO que não nasce.

**Consequência para o bloco 2:** consertar só o contador da rampa acelera o trade e piora a
marcha. As opções são baixar `limiar_portao` (aceitar marcha pior) ou separar os treinos com
encadeamento determinístico. Ver g1-anda-e-pega-separado-controle-deterministico.

## FECHAMENTO na it 4999

**A locomoção SE MANTEVE, e o "recuou 17%" que eu escrevi aqui estava ERRADO.** Duração 984
-> 988 e `fell_over` 0,000 -> 0,167: os desfechos não se moveram. O que caiu foi só métrica
de AÇÃO AMOSTRADA (`razao_marcha` 0,514 -> 0,426, escorrego, força de pouso, `action_rate`),
e o `std` subiu de 0,43 para 0,61 no mesmo intervalo — +42% de ruído nas 29 juntas faz a
MESMA política medir pior. O `play` determinístico do usuário no `model_3800` andou bem, e
ele é a régua. Ver a regra em g1-leitura-log-e-kaggle, que eu tinha escrito e não apliquei.

A rampa congelou: `forma` 0,7868 -> 0,7850 em 1341 iterações (UM degrau), porque 0,426 fica
na banda morta. Isso é estado de mecanismo, não afetado por ruído.

Manipulação com todos os componentes vivos e nenhum fechamento: `staged` 0,73, `unload` 0,49,
`squeeze` 0,14, `postura_ereta` 0,099 — ele agarra, tira a caixa da laje e fica de pé. E
`sucesso = 0,0000` com `sustentacao = 0,0002`. O `avancos` (0,138) bate com a população de
cadeias de 2 elos e a cadeia de 1 elo `(PEGAR,)` nunca fecha: **o REORIENTAR fecha, o PEGAR
não**. É o platô do grasp booleano. Ver g1-pegar-plato-do-grasp-booleano.

⚠ `tempo_de_voo` subiu 5,9× para 0,860 com a altura em 29 mm. Eu li isso como claudicação e
RETIREI: com `std` em 0,61 é contato de pé errático por ruído de ação. Métrica de contato não
se lê com `std` variando.

**O que a comparação com o g1_poc realmente prova.** Ela vale porque está nos desfechos:
g1_poc com 70% de manipulação tinha episódio de 102 passos e `fell_over` 34,6; g1_limpo com
21% tem 988 e 0,17. Mas a 21% a locomoção **não paga preço mensurável**, então o ponto de
ruptura está entre 21% e 70% e NINGUÉM MEDIU ONDE. Separar as políticas está PREMATURO — o
experimento que falta é uma política só, subindo a fatia por degraus, com o portão medindo
DESFECHO (duração e queda) em vez de estatística de ação.

⚠ Achado de método: o `smoke` ganhou a checagem "todo sensor tem consumidor" e ela achou 4
órfãos além do `corpo_prateleira`. `pes_chao` e `auto_colisao` são duplicatas deliberadas
documentadas; `dorso_E`/`dorso_D` são dívida real (um `back_penalty` nunca fiado).

⚠ Meus degraus de `pico_de_altura >= 0,010` e `tempo_de_voo >= 0,10` na it 800 FALHARAM
enquanto a marcha nascia. Eu os escolhi por múltiplo do valor de ontem, sem derivação
física. O `razao_marcha` pré-registrado era a única régua boa.

Relacionado: g1-limpo-desenho, g1-leitura-log-e-kaggle,
g1-poc-portao-forma-mede-sobrevivencia, g1-andar-como-rastreio-de-velocidade
