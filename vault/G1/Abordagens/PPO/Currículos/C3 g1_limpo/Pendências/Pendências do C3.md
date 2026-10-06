---
tipo: frente
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "As 12 pendências abertas do C3 em 05/10; cada uma é uma nota pendencia que sai da Home quando fecha."
cluster: c3
---
Pendências abertas do C3 em 05/10/2026, levantadas no `model_23250` e nos testes do 24350 e do 24800. Fonte: Estado do C3 em 05-10 e `docs/relatorios/2026-10-01-aprendizados-zero17-para-v3.md` §8.

Elas se agrupam em três famílias: a **pega** (PEGAR tomba a caixa, punho esquerdo gira, aperto no limite do motor), a **postura** (agachado na espera, perna virada no BOTAR, forma postural longe da IK, fecho do BOTAR sem de pé) e o **currículo** (freio em catraca, R4b sem prioridade, giro no lugar estranho). O CARREGAR lento e os abertos não medidos atravessam as três.

A maior parte delas é o motivo direto de uma decisão da v3. Ao resolver uma, mude `estado: fechada` e escreva onde e como foi resolvida; ela sai da lista de pendências da Home.

## Filhos
<!-- filhos:inicio -->
- **Abertos não medidos** (pendencia · aberta · 05/10/26) — Não se sabe em que elo acontecem o caixa_largada e o fell_over, nem quanto rende chegar agachado ao PEGAR.
- **Agachado na espera e no ANDAR parado** (pendencia · aberta · 05/10/26) — Na linha zero o robô espera agachado (joelho 1,4–1,55, pelve 0,61) contra 0,15 rad e 0,78 m pedidos; o 21850 não agacha.
- **Aperto no limite do motor** (pendencia · aberta · 05/10/26) — Força total robô → caixa fixa em ~200–250 N, sem modular por massa nem por μ; o preço do aperto (−0,05 e −0,15) não moveu a força.
- **CARREGAR lento, girando e agachado** (pendencia · aberta · 05/10/26) — O CARREGAR anda a ~0,35 m/s para 0,5, gira sem comando (−22° a −68° em 9 s) e vai agachado (pelve 0,69 m).
- **Fecho do BOTAR sem de pé nem caixa nivelada** (pendencia · aberta · 05/10/26) — O fecho do BOTAR não exige ficar de pé nem a caixa nivelada no apoio; o 6050 pousa a caixa numa aresta.
- **Forma postural longe da IK em laje baixa** (pendencia · aberta · 05/10/26) — Na laje 0,15 com 5 kg: tronco 109° contra 66° da IK, pés 0,10 m mais estreitos; a cabeça bate na caixa.
- **Freio em catraca sem descida** (pendencia · aberta · 05/10/26) — O freio por env sobe com o nível e nunca desce: freio sobe → quedas sobem → nível cai → freio fica.
- **Giro no lugar estranho** (pendencia · aberta · 05/10/26) — O giro no lugar forma de forma estranha, com e sem caixa; comando baixo não é respeitado (σ fixo do rastreio).
- **PEGAR tomba a caixa ao erguer** (pendencia · aberta · 05/10/26) — O PEGAR tomba a caixa ao erguer (38–47° no play do 6050); desvio do PEGAR ~30% no 23250.
- **Perna virada no BOTAR** (pendencia · aberta · 05/10/26) — hip_yaw de 22–30° no BOTAR, abaixo da rampa de 40°, então nada cobra.
- **Punho esquerdo gira na pega** (pendencia · aberta · 05/10/26) — O wrist_yaw E vai a 88% do curso na pega: COMANDO no elo PEGAR (alvo +0,6 rad) mais contato da caixa.
- **R4b sem prioridade para a laje difícil** (pendencia · aberta · 05/10/26) — Com s_C ≥ 0,50 a laje do BOTAR sai uniforme de 0,04 a 0,64 m e o R4b não dá mais fatia à laje que falha.
<!-- filhos:fim -->
