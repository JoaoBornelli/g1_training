# Revisão do plano da v3 (29/10)

Plano revisado: `docs/planos/2026-10-14-implementacao-g1-v3.md` (as linhas citadas são da versão
anterior à aplicação).

**Decisão do dono (29/10):** aplicar todas as correções menos a 12; a implementação segue com
agentes Haiku. Aplicadas no plano (§20, linha de 29/10), na I §2.4, no PERC §0 e na apresentação.
A correção 1 usa `z_ref` (o pé mais baixo no frame da odometria) e o filtro guarda o anterior; a 11
virou a lista dos CSV pela coluna `fase` (B5).
Os números dos testes conferem: refiz as contas da B0, D2b, D5, F4 e H2; a A6 bate com as tabelas
da IK (`pelve` 0,249 e 0,705, `tronco` 51,2°, `q_de_pe`).

## 1. Defeitos que quebram o treino

| # | onde | problema | correção |
|---|---|---|---|
| 1 | C1 `prediz` (:1046), PD-7, I §2.4 | O zero da altura é o pé mais baixo de agora, mas o mapa só se desloca em x, y e rumo. Na escada, o zero sobe 15 cm a cada degrau. As células da zona cega à frente ficam 15 cm erradas por degrau, com confiança alta. | `prediz` recebe `dz_ref` (variação do pé mais baixo no frame da odometria) e faz `h −= dz_ref`. Teste na C0: subir 15 cm sem leitura leva o degrau de +0,15 a 0. |
| 2 | D2 `Mapa` (:1166), E8, §6.4 | Ator e crítico têm `mapa` e `caixa`. Com cfg próprio, o mjlab cria duas instâncias (`manager_base.py:146-147`), e o filtro roda duas vezes por passo. | Trava e saída no env (`v3_percepcao_passo`, `v3_mapa_obs`, `v3_caixa_obs`). Os dois termos chamam `atualiza_percepcao(env)`: o primeiro atualiza, o outro lê. |
| 3 | PD-20 (:277), D7 (:1326) | O mjlab usa t = linha/(R−1). Com R = 10, o teto do CARREGAR fica em 13,3 cm e rampa 0,24, abaixo da S §11.2 e do aceite da S §15 (0,15 m; 0,27). | R = 13 e `linha_max_carregar` inteira por coluna: escadas e rampas 9, `discrete_obstacles` 8, `random_spread_boxes` 7 (a altura dele escala por 0,2 + 0,8·t, `primitive_terrains.py:794`). Teste contra a fórmula do gerador. |
| 4 | D2 (:1163), PD-28 | O atraso da observação nasce com a capacidade do cfg (`observation_manager.py:465-468`). Com 0–1, a E6 e a H4 não sobem para 0–2 e 0–3. | A D2 constrói com `delay_max_lag` 3; a D7 baixa para 1. Custo: 58 MB na GPU com 4 096 envs. |
| 5 | §6.3 (:529, :532), §6.6 | Alvos sem uso devolvem NaN, e a máscara multiplica. NaN × 0 = NaN na recompensa, na entrada do ator e nas métricas. | Alvos sem uso devolvem 0, com máscara de validade; a `mascara` devolve bool e o termo usa `torch.where`. |

## 2. Ambiguidades

| # | onde | problema | correção |
|---|---|---|---|
| 6 | §6.3 `Odometria`, D2b | O erro soma no mundo; o Δ do mapa pelo rumo estimado leva o erro absoluto de rumo a todo passo. O plano não diz como o `Mapa` tira o Δ. | A `Odometria` integra incrementos no frame da base (Thrun) e expõe `delta_b()` (dxy, dψ, dz). Só o rumo integra; roll e pitch vêm do IMU. `reinicia` zera o incremento. |
| 7 | §6.3 (:638-651), D2 | `filtra_corpo` pede pontos no mundo, e `pontos_no_frame_do_mapa` devolve o frame do mapa. O frame do `sensor_pos` não está definido. | `pontos_no_frame_do_mapa` devolve também os pontos no frame comum; parâmetros neutros no `filtra_corpo`; `atualiza` recebe `dist` para o d̄. |
| 8 | :198, :1103, :696, :1358 | PD-14 citada para o atraso (é o déficit); PD-15 citada para o `tarefa_critico` (é o canal zero). "(D3)", "(D4)" da viabilidade colidem com as tarefas. | PD-25 no lugar da PD-14; tirar a PD-15; prefixo "viab.". |
| 9 | §6.2 (:495) | `raio_caixa_na_mao` sobrou da PD-10 antiga. | Tirar. |
| 10 | §6.9 (:776), H5 | A §6.9 põe o estimador e a DR do refino no degrau por tarefa; a PD-41 diz global. | Citar o degrau global. |
| 11 | B5, PD-34 | "23250" casa com 2 CSV, "24800" com 5. | Nome exato por item (decisão do dono). |
| 12 | §1.2 e tarefas Haiku | Lanes de teste e 9 tarefas em Haiku, contra a regra "implementador em Sonnet". | Tudo em Sonnet; o Opus confere o teste de cada fase antes do código. |
| 13 | D4 (:1255) | `passos_por_iteracao` = 24 fora do `knobs`. | Campo no `knobs`; o `rl_cfg()` lê dele. |

## 3. Defeitos antigos achados na verificação

| # | onde | problema | correção |
|---|---|---|---|
| 14 | D4, B3 `fecha_iteracao(it)` | O `it` vem de `common_step_counter`, que volta a 0 no resume. Com a última subida salva na it 12 000, nenhum nível sobe na sessão retomada. O I6 (10 its) não pega. | O `Curriculo` conta as próprias iterações e salva no `estado()`. Teste: carregar no meio da carência espera só o resto. |
| 15 | D1, coluna `irregular` | `random_rough` tem `scale_with_difficulty` False (`heightfield_terrains.py:404`): todas as linhas têm a mesma rugosidade (2–10 cm). | `scale_with_difficulty=True`. |

## 4. Efeitos colaterais conferidos

- 1 precisa do `dz` da 6 e da passagem do `dz_ref` no `pilota_v3` (D11); muda a I §2.4, o PERC §0 e a apresentação.
- 3 custa 30% mais ladrilhos (o I5 mede) e 3 níveis a mais de terreno no ANDAR.
- 6 põe o erro de rumo na posição absoluta; sem efeito no alvo da caixa, porque o robô fica parado antes de abrir o PEGAR e o BOTAR. O teste de 1,5 cm em 4 m zera os outros termos.
- 10 faz a estimativa do PEGAR piorar só quando a tarefa mais atrasada refina (preço da PD-41).
- 8, 9 e 11 sem efeito.
