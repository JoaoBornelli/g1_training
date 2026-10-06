# G1 — Home

Raiz da árvore: [[G1]]. Anotação solta: [[inbox]].

## Pendências abertas
```dataview
TABLE pai, verificado FROM "" WHERE tipo = "pendencia" AND estado = "aberta" SORT verificado DESC
```

## Abordagens ativas e em desenho
```dataview
TABLE pai, estado, verificado FROM "" WHERE tipo = "abordagem" AND (estado = "ativa" OR estado = "em-desenho")
```

## Achados para verificar (mais de 3 semanas)
```dataview
TABLE pai, verificado FROM "" WHERE tipo = "achado" AND estado != "superada" AND date(verificado) < date(today) - dur(21 days) SORT verificado
```

<!-- estatico:inicio -->
## Lista estática (para agentes; gerada por indexa.py)

### Pendências abertas (15)

- **Abertos não medidos** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — Não se sabe em que elo acontecem o caixa_largada e o fell_over, nem quanto rende chegar agachado ao PEGAR.
- **Agachado na espera e no ANDAR parado** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — Na linha zero o robô espera agachado (joelho 1,4–1,55, pelve 0,61) contra 0,15 rad e 0,78 m pedidos; o 21850 não agacha.
- **Aperto no limite do motor** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — Força total robô → caixa fixa em ~200–250 N, sem modular por massa nem por μ; o preço do aperto (−0,05 e −0,15) não moveu a força.
- **CARREGAR lento, girando e agachado** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O CARREGAR anda a ~0,35 m/s para 0,5, gira sem comando (−22° a −68° em 9 s) e vai agachado (pelve 0,69 m).
- **Fecho do BOTAR sem de pé nem caixa nivelada** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O fecho do BOTAR não exige ficar de pé nem a caixa nivelada no apoio; o 6050 pousa a caixa numa aresta.
- **Forma postural longe da IK em laje baixa** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — Na laje 0,15 com 5 kg: tronco 109° contra 66° da IK, pés 0,10 m mais estreitos; a cabeça bate na caixa.
- **Freio em catraca sem descida** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O freio por env sobe com o nível e nunca desce: freio sobe → quedas sobem → nível cai → freio fica.
- **Giro no lugar estranho** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O giro no lugar forma de forma estranha, com e sem caixa; comando baixo não é respeitado (σ fixo do rastreio).
- **PEGAR tomba a caixa ao erguer** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O PEGAR tomba a caixa ao erguer (38–47° no play do 6050); desvio do PEGAR ~30% no 23250.
- **Perna virada no BOTAR** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — hip_yaw de 22–30° no BOTAR, abaixo da rampa de 40°, então nada cobra.
- **Punho esquerdo gira na pega** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — O wrist_yaw E vai a 88% do curso na pega: COMANDO no elo PEGAR (alvo +0,6 rad) mais contato da caixa.
- **R4b sem prioridade para a laje difícil** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências) — Com s_C ≥ 0,50 a laje do BOTAR sai uniforme de 0,04 a 0,64 m e o R4b não dá mais fatia à laje que falha.
- **Lacunas da linha bloco** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Linha bloco) — O que as fontes não confirmam: bloco 5, limites dos blocos 2, 6, 7, 10, 15, 16, nome bloco18 × bloco19, origem do ckp9100.
- **Lacunas da linha zero** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Linha zero) — O que as fontes não confirmam: fim da zero01 piramidal, zero02 × NoBatente, model_9100, linhagem do model_3100 e do 3450.
- **g1-v3-prateleira** (G1/Abordagens/PPO/Currículos/C4 v3) — FUTURO 02/10: prateleira fora da v3, arquitetura própria como o REORIENTAR; mãos pela frente também na laje; estante de vários níveis: só os braços entram no vão; pose dos níveis baixos em aberto

### Abordagens ativas e em desenho (6)

- **C3 g1_limpo** (G1/Abordagens/PPO/Currículos) — Máquina de elo com locomoção primeiro, em duas linhas: a linha bloco (resume encadeado) gerou o 21700, e o refino v2 dele deu o model_23250, MELHOR modelo; a linha zero (12 recomeços) não produziu política competente.
- **Refino v2** (G1/Abordagens/PPO/Currículos/C3 g1_limpo/Refino v2) — Resume do model_21700 com o código da zero17: model_23250 é o MELHOR modelo; as continuações e os resumes do 24350 (freio em catraca, preço do aperto) não melhoraram.
- **C4 v3** (G1/Abordagens/PPO/Currículos) — Quarta abordagem, em desenho: pacote g1_v3 treinado do zero só com as peças que funcionaram — potencial de progresso em alvos macro de IK, dois balanceadores, refino em degraus, locomoção da Unitree; mjlab primeiro, Isaac Lab depois.
- **Autônomo fase 1** (G1/Abordagens/Controle determinístico) — Controlador do depósito sem RL que chama a política como executor; missão A com oráculo; o 24350 pega em 1,8 s, carrega e pousa; comando baixo exige camada de rastreio.
- **Lidar e SLAM** (G1/Percepção) — MID-360 simulado com o padrão real da Livox e SLAM por KISS-ICP + ESKF com a IMU: σ de 1 cm com deskew; o 24999 não gira no lugar, roteiro usa curva.
- **Visão da caixa** (G1/Percepção) — Orientação da caixa por RGB: duas CNNs, memória das faces e HMM (85% com ORB); o RANSAC de planos no depth dá o frame da caixa sem a rede.

### Achados com mais de 3 semanas sem verificação (0)

- nenhuma

<!-- estatico:fim -->
