---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
periodo: 05/10/26
resumo: "O que funciona (model_23250, 24350/24800), as 12 pendências abertas e o que motivou a v3."
cluster: c3
---
## O que funciona
- **`model_23250`** (21700 refinado, `39b8b82`): ciclo completo andar → pegar → carregar → botar em lajes de 0,55 a 0,15 m com 1 e 3 kg, levantando sem derrubar. Funil da C de 44–65% (V-CKPT; REL-APR §5).
- **`model_24350`/`24800`:** movimentação melhor segundo o dono. O 24800 anda mais com a caixa e bota de pé em 0,15, 0,35 e 0,55 com 3 kg e μ 0,6 (SESS-05/10).
- **Mecanismos validados:**
  - Currículo de cadeia por fases, com a SEGURA tornando o fecho sobrevivível (SPEC-CADEIA §10).
  - One-hot fora da normalização, com a dobra no `load` (MEM:g1-limpo-one-hot-normalizado-amplifica-slot-raro).
  - Currículo de dificuldade do BOTAR pelo `s_C` (R4/R4b): `s_C` de 0,004 a 0,20 na zero17 (REL-APR §2).
  - Métricas sem peso do funil da C (K12): "o que mais ajudou a decidir" (REL-APR §0 item 7).
  - Registrador com `giro_w`, `alvo`, `ckpt`, ação e forças (`git:025a9d6`, `git:7725f20`).

## Pendências abertas
1. PEGAR tomba a caixa ao erguer (38–47° no play do 6050, confundido pelo registrador antigo). Desvio do PEGAR ~30% no 23250.
2. CARREGAR da linha zero: lento, gira sem comando, agachado. Não diagnosticado (REL-APR §4, §8 item 5).
3. Agachado na espera e no ANDAR parado na linha zero. O 21850 não agacha.
4. Perna virada no BOTAR (`hip_yaw` 22–30°, abaixo da rampa de 40°; REL-APR §3).
5. Punho E gira na pega (comando + contato; REL-APR §3b; MEM).
6. Aperto no limite do motor (~200–250 N), sem modular por massa ou μ. O preço não moveu a força (REL-DR; MEM).
7. Freio em catraca: precisa de regra de descida (REL-APR §5c).
8. R4b sem prioridade para a laje difícil (REL-APR §8 item 6).
9. Giro no lugar estranho e comando baixo não respeitado (σ fixo; SESS-05/10; MEM:g1-limpo-giro-no-lugar-empata-no-estagio-0).
10. Fecho do BOTAR sem exigir de pé nem caixa nivelada no apoio (REL-PLATO §0; REL-APR §2).
11. Forma postural: tronco e pés longe da IK em laje baixa (`git:af97a5a`).
12. Abertos não medidos (REL-APR §8): em que elo acontecem o `caixa_largada` e o `fell_over`; a sonda do crítico sobre o ganho de chegar agachado ao PEGAR.

## O que motivou a v3 (C4)
- **Linha zero contra linha bloco** (REL-APR §5b): o caminho de aprendizado importa tanto quanto o ótimo da recompensa. Lições para a v3:
  - treino em degraus com resume;
  - mudar a recompensa sem recomeçar;
  - restrições (freio, limite de junta, aperto) em níveis que sobem com o fecho e podem descer.
- **Decisões do dono para a v3, em 05/10:**
  - "Palma de plástico", μ 0,3–0,5, massa máxima 5 kg. Pagar o RESULTADO da pega, e não a força lateral. Caixa de pé só no BOTAR (REL-DR §6; MEM:g1-v3-palma-de-plastico-5kg).
  - DR de atrito, massa e erros de sensor em níveis desde o começo (REL-APR §3c).
  - Soltar exige afrouxar: gate de força no B2, e o ator infere o aperto pelo torque do PD (SESS-05/10; MEM:g1-v3-soltar-exige-afrouxar).
  - Rastreio normalizado pelo comando (MEM:g1-limpo-giro-no-lugar-empata-no-estagio-0).
  - Sem cone elíptico, `nan_guard` ligado e impratio no knob (V-NAN; SESS-05/10).
- **Tentativas de consertar na v2 que falharam e foram adiadas para a v3:** freio de aperto, DR μ e forma ponderada, todos no resume do 24350 (SESS-05/10).
- **Outros documentos da v3 nessa janela** (não lidos em detalhe aqui): `docs/relatorios/2026-10-01-inventario-v3.md`, `2026-10-02-avaliacao-v3.md`, `2026-10-05-revisao-spec-v3.md`, `2026-10-05-dr-para-a-v3.md`, `vault/.../C4 v3/g1-v3-decisoes.md`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
