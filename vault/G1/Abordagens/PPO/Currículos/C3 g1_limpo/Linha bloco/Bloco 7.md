---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 31/08–02/09/26
resumo: "Mesa vira multa em rampa e vantagem normalizada por elo: postura_ereta sai do zero e a queda cai de 51,9% para 0,9%."
cluster: c3-bloco
---
- **Datas:** os dois patches "rodaram como célula no bloco 7" e entraram no repo em 02/09 08:22 [8c2c929]. Em 02/09 a run "segue rodando com o código antigo"; o dono decidiu não esperar o veredito [handoff]. Última iteração citada: ~2514 [contrato §2].
- **O que mudou antes (patches de célula, depois [8c2c929]):**
  1. contato com a mesa vira MULTA em rampa (zero abaixo de 50 N, linear até 100 N, saturada), peso −2,0 derivado do `postura_ereta` (pega ereta e escorada a quatro pontos; multa = 17% do teto de tarefa contra ~45× da terminação). `caixa_largada` segue terminação.
  2. `PPOPorElo`: vantagem normalizada por grupo de elo (subclasse do PPO; `rsl_rl/algorithms/ppo.py:188` normalizava sobre o lote inteiro).
- **Hipótese:** com terminação, 76% dos episódios de manipulação morriam na mesa e a aritmética favorecia ficar parado (90 de retorno contra 66 de tentar); o play mostrou que a ação MÉDIA nem se aproximava da mesa — os 76% eram ruído de exploração [8c2c929]. E a locomoção perdia sinal quando a manipulação destravava [8c2c929; `algoritmo.py`].
- **Medido:**
  - Depois da multa: `descarga` 0,0 → 0,994; `palmas_em_contato` 0,09 → 0,63; `caixa_largada` 62% → 28% das terminações; `postura_ereta` saiu de ZERO pela primeira vez no módulo [8c2c929].
  - `[ADV]` com ~32% dos envs em locomoção: it 1600 loco 0,112 / manip 0,446 (razão 0,251, 10,4% do gradiente); it 1650 0,162 / 1,042 (0,155, 6,7%); it 1800 0,164 / 1,174 (0,139, 5,6%) [`algoritmo.py`; 8c2c929].
  - Mesma iteração, mesmo nível de manipulação, sem → com patch: descarga 0,991 → 0,994; marcha 0,484 → 0,762; `fell_over` 51,9% → 0,9%; duração 425 → 888; Tl 97 passos → ~1000 [8c2c929].
  - it ~2514 [contrato §2]: andar `seg_proj/seg_pedido` 0,739; `Curriculum/elo` 0,296 (a rampa chegou aos 30%); `descarga = unload/squeeze` 0,965; `rampa = postura_ereta/unload` 0,978; `sustentacao` ≈ 0,0005; `precise_pos` "oscila em ~7–9 cm de `tol_pos` = 10 cm"; CARREGAR e BOTAR pouco exercitados; "Trazer ao alvo: não fecha".
- **Também na branch, 02/09 manhã (não está claro se entrou no bloco 7):** janela de espera 0,3–1,0 s antes da manipulação, com o fecho exigindo objetivo ativo (fechava o REORIENTAR no passo zero: `avancos` 0,43 com `sucesso` 0,0000) [acbf8f1]; o rastreio paga durante a janela (janela no PEGAR 2,00/s contra ANDAR 4,08 e PEGAR ativo 4,49; depois 3,816/s na janela e 0,000 após a borda) [5ec7f56]; modo de entrega no visualizador, depois REJEITADO (robô "sambava") [413f022; 78379ff; mem: g1-limpo-modo-entrega-viewer].
- **Por que foi trocado:** o contrato de troca de tarefa (v2) muda a observação; o checkpoint fica inválido e a próxima run treina do zero; a `exp/g1-limpo` fica como referência que treinou a `bloco7` [1f803e8; handoff; contrato §13].
- **Lições:** termina só o que não tem como ser pago (largar a caixa), encostar na mesa tem preço [8c2c929]; normalização de vantagem sobre lote misto encolhe a tarefa minoritária; agrupar por elo não separa os pesos [8c2c929]; segundo canal global aberto: a taxa de aprendizado, dirigida pelo KL agregado [8c2c929].

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
