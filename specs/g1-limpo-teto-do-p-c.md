# G1 limpo — teto do `p_C` na fase 4

Estado: spec resumida, aguarda "implementa". Data: 30/09. Enunciado: §1 (progressão
ANDAR → PEGAR → CARREGAR → BOTAR, sem perder o que já foi aprendido).

## 1. Problema, medido na zero17

| Métrica | fim da fase 3 (it ~4650) | fase 4 (it 4990) |
|---|---|---|
| `p_C` | 0,05 | 0,70 |
| `s_C` | 0,001 | 0,000 (325 its em zero) |
| `Curriculum/nivel` | 2,9 | 1,34 (estabilizou) |
| `fell_over` | 0,8 | 1,3 (pico 2,8) |
| `s_B` | 0,55 | 0,56 |
| `passo_final` | 0,03 | 0,35 a 0,53 |

O balanceador faz `p_C = (1 − s_C)/((1 − s_B) + (1 − s_C))`, limitado a
`[piso, 1 − piso]` = [0,20; 0,80] (`curriculo.py:326`). Com `s_C = 0` ele manda ~70% das
cadeias para a C. O nível é por env: cada C que falha desce o nível do env, e a pega
desse env desce junto. O nível perdeu 1,5 de 2,9. O `passo_final` alto mostra que a C
chega ao BOTAR e não fecha.

## 2. Mudança

1. `knobs.Cadeia.balanceador_teto: float = 0.30` (novo). O `p_C` fica em
   `[balanceador_piso, balanceador_teto]`.
   - Hoje o teto é `1 − piso` = 0,80, implícito. Com o knob em 0,80 o comportamento é o
     de hoje, e o smoke confere essa identidade.
2. `curriculo.resolve_p_c(s_b, s_c, piso, teto)`: o `1.0 − piso` vira `teto`.
3. `comando.AlvoCaixaCmdCfg.balanceador_teto` e a chamada em `comando.py:1221`; o
   `env_cfg.py:595` passa o knob.
4. `smoke.py`: `resolve_p_c(0, 0, 0,20, 0,30) == 0,30` e, com teto 0,80, o valor de hoje.
5. Resume do `model_4650` (fim da fase 3, nível 2,8). `RUN` segue `zero17`.

Linhas estimadas: knobs +2, curriculo +0 (troca de argumento), comando +2, env_cfg +1,
smoke +4.

## 3. Por que 0,30

- 30% das cadeias em C dão ~2,3× a prática de BOTAR da fase 3 (0,05 × 0,77 → 0,30 × 0,77
  dos episódios), e deixam 70% em B/R, onde o nível sobe.
- Estimativa, não medida: com 30% de C falhando e 70% de B a ~0,55, o sucesso por cadeia
  fica ≈ 0,39, e o nível médio estabiliza perto de 2,3, contra 1,34 hoje (fórmula de
  `g1-limpo-piso-segura-o-s-b`).

## 4. Por que não quebra o treino

- Observação, rede e recompensa não mudam. O `model_4650` carrega como está.
- O `fase_cadeia` = 3 vem do checkpoint (`runner.CHAVES_ESCALARES`), e a troca 3 → 4
  acontece de novo em ~15 its com o `s_cauda` do checkpoint (0,85).
- O teto só vale na fase 4. Nas fases 1 a 3 o `p_C` segue fixo em 0,05.

## 5. Verificação

| O que ler | Esperado |
|---|---|
| `Curriculum/forma/p_C` depois da troca 3 → 4 | 0,30 |
| `Curriculum/nivel` | acima de 2,0 em 300 its |
| `s_C` | sai de zero em até 500 its |
| `s_B` | não cai abaixo de 0,45 |

Se o `s_C` ficar em zero por 500 its com o teto, o problema é o BOTAR, e não a fatia:
abre-se a medição do BOTAR (o que falta no fecho quando o `passo_final` chega a ele).

## 6. Decisões

- D1 (30/09, dono): teto no `p_C` e resume do `model_4650`.
- D1b (30/09, dono): antes de aplicar, deixar a zero17 seguir na fase 4. O fecho do
  BOTAR exige voltar à pose de pé, e isso levou tempo nos treinos anteriores.
- D2 aberta: implementar. Ela pode ir no mesmo commit da spec `g1-limpo-giro-no-lugar.md`.
