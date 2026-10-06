---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 04–08/09/26
resumo: "Gradientes v2.1 em warm-start: o PEGAR fecha pela primeira vez na it 2616 e o sucesso chega a 0,574 na it 5000."
---
### Início (relatório A)
- **Datas:** 04/09 [specs/g1-limpo-curriculo-de-cadeia.md §2].
- **O que mudou:** v2.1 [53f81e9]; warm-start do `model_1800`, LR 5e-4 [mem: g1-limpo-auditoria-gradientes]. A proposta dizia que o `assert` de pesos da célula de resume dispara por desenho e que zero × retomar de `model_2562.pt` era decisão do dono [proposta §224].
- **Medido** [mem: g1-limpo-auditoria-gradientes]:
  - it 2119 (+319 do resume): `precise_pos` por env de manipulação 0,7% → 26,3% (d = 0,208 m contra d₀ ≈ 0,45); `aproxima_caixa` 0,41; gradiente do par 14,6/m no ponto atual, 16,1/m no limiar (antes caía); renda de manipulação 8,1 → 9,8/s; queda zerou; `caixa_largada` 16,7% → 9,8%; `sucesso`, `sustentacao`, `renda_congelada` em zero.
  - it 2300: gangorra pela terceira vez — locomoção no melhor de sempre (eficiência 0,785, razão 0,663, voo 0,094 s, queda zero) e manipulação devolve um terço do ganho (d 0,208 → 0,221 m); `alcancar`, preensão e palmas em contato caem ~8% juntos, `descarga` 0,987, `self_collisions` dobrou, momento angular +7,6%.
  - **it 2616: o PEGAR fecha pela primeira vez.** O fecho era TERMINAL: o robô seguia segurando no mesmo alvo, e fechar pagava ~25/s contra ~12,5/s [specs/g1-limpo-curriculo-de-cadeia.md §2].
- **Fim:** fora da janela.

### Até a it 5000 (relatório B)
- **Datas:** desde ~04/09 até antes de 08/09 (o `play` do bloco9 é de 08/09).
- **O que mudou antes:** gradientes v2.1 (`53f81e9`, 04/09). P1 `precise_pos` σ 0,05 → 0,18 e peso
  2,0 → 3,0 (pagava `exp(-4) = 0,018` no limiar do fecho; passa a 0,73); P3 `renda_congelada`; P4
  rastreio gateado por `limpo_twist_zerado`; P5 twist fixo no CARREGAR-andando; P7 `caiu` sem a arma
  do `pegou`; P8 REORIENTAR inerte a 5% do sorteio. Warm-start do `model_1800`, LR 5e-4 (memória
  `g1-limpo-contrato-de-troca-de-tarefa`).
- **Medido (memória `g1-limpo-contrato-de-troca-de-tarefa`, números do log Kaggle):**
  - it 2616: a cadeia FECHA. `sucesso` 0,0000 → 0,362; `renda_congelada` 7,27/s por env de manipulação;
    `Curriculum/nivel` 0 → 1,68. A caixa foi de 0,385 m para 0,114 m do alvo em ~800 its.
  - Custo na 2616: `Mean value loss` 0,44 → 5,50; `action std` 0,61 → 0,66; `caixa_largada` 8,9% →
    44,1%; `fell_over` 0% → 9,0%; `upright` 0,88 → 0,58; eficiência da locomoção 0,785 → 0,444.
  - it 5000: `sucesso` 0,574, nível 2,88, preensão 0,946, descarga 0,998, caixa a 7,2 cm do alvo
    (aceite 10 cm), queda em 2,4% das terminações, retorno 261,3. `largou` = 0 o run inteiro: os
    sucessos são o fecho terminal do PEGAR, não o BOTAR.
  - Na última it o `velocity_stages` do molde avançou (`5000 × 24` passos): `lin_vel_x` ±1,0 →
    (−1,5; 2,0), `ang_vel_z` ±0,5 → ±0,7; eficiência da locomoção 0,915 → 0,364.
- **Medido depois, no `model_4999` (spec `g1-limpo-lento-e-estavel.md` §0, 08/09, 32 envs, 600 passos, CPU):**
  `|v|` de junta p90 — manipulação parada 3,13 rad/s (p99 8,19), locomoção parada 0,43 (p99 2,24),
  andando 2,00, correndo 2,50. "O robô se move mais rápido parado com a caixa do que correndo"; 7,3× a
  locomoção parada. Pernas correm: joelho 6,22, tornozelo pitch 7,05, quadril pitch 4,14, ombro pitch
  2,68, cotovelo 2,43.
- **Medido no `play` do bloco9 (spec `g1-limpo-espera-sigma-e-pose.md` §1, 08/09):** o robô assume
  pose de pré-pega durante a espera; σ fixado em 0,34 m no reset, mão a 0,20 m no fim da espera,
  `alcancar = 0,71` em vez de 0,368.
- **Lições:** o par macro/preciso (σ > tolerância) destrava o aceite; `renda_congelada` não observável
  e pagando dobrado no fecho terminal (memória contrato).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
