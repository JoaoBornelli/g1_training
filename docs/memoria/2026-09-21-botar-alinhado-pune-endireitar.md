# O `alinhado` do BOTAR pune a caixa ficar direita

Medido em 2026-09-21 no `model_9100` da run `zero02`, com `registra_juntas` em três
alturas de laje (0,55, 0,35 e 0,15) mais um registro de CARREGAR. Os CSV estão na raiz
do repo como `ckp9100_m2_t0{55,35,15}_i2.csv`.

## O que o BOTAR pede hoje

O fecho do BOTAR exige três condições ao mesmo tempo (`comando.py:1356`):

    perto (tol_pos = 0,10 m)  ∧  alinhado (tol_ang_deg = 25°)  ∧  apoiada (>= 0,5·m·g)

O `alinhado` compara a normal da face marcada com uma direção **congelada** na normal
que a caixa tinha no instante em que o elo abriu (`_congela_face`, `comando.py:1815`).
Ele mede "a caixa girou desde a abertura do BOTAR?".

## Medição 1 — a posição está certa

| laje | `caixa_z` final | alvo z | erro xy contra o alvo |
|---|---|---|---|
| 0,55 | 0,650 | 0,650 | 0,06 a 0,14 m |
| 0,35 | 0,450 | 0,450 | 0,08 a 0,17 m |

A altura é exata: a caixa assenta no topo mais a meia-aresta. O erro em xy fica na
fronteira do `tol_pos` = 0,10 m, e a faixa vem do jitter de ±0,05 m em
`botar_recuo_borda` (`knobs.py:252`), que o CSV não registra.

A 0,15 o robô cai: rumo gira 102°, pelve desce a 0,334 m, tronco a 129°, e a caixa
termina no chão a 0,46 m da laje. É perda de equilíbrio, não imprecisão de pouso.

## Medição 2 — o giro punido é o endireitamento

Decompondo a rotação da face entre a abertura do BOTAR e o pouso:

| laje | giro total da face | componente de guinada | tombamento da caixa |
|---|---|---|---|
| 0,55 | 33,2° | 17,4° | 29,7° → **0,0°** |
| 0,35 | 53,2° | 17,1° | 52,2° → **0,0°** |

O robô segura a caixa **tombada** e a endireita ao pousar; a laje nivela a caixa. O
congelamento captura a normal no instante tombado, logo o `alinhado` passa a exigir que
a caixa **mantenha o tombo**. Os 25° reprovam nas duas alturas.

## Medição 3 — o `sigma_ori` degenera no BOTAR

No avanço de elo as duas linhas rodam no mesmo passo (`comando.py:720-721`):

```python
self._aplica_elo(ids_avanca, so_pose=False)   # chama _congela_face -> ANG = 0
self._recalcula_sigmas(ids_avanca)            # le ANG e escreve sigma_ori
```

O congelamento zera o erro por construção, e
`sigma_ori = (ANG × sigma_fator).clamp(min = sigma_ori_min)` devolve **sempre o piso de
0,20 rad** (`comando.py:1808`). O `precise_ori` é
`alcançar × exp(−(Δθ/σ_ori)²)` (`recompensas.py:632`):

| Δθ | `precise_ori` | derivada por grau |
|---|---|---|
| 23,3° | 0,016 | −0,006 |
| 33,2° | 0,0002 | −0,0001 |
| 53,2° | 4e−10 | ~0 |

Nos 33° medidos o termo vale 2 em 10 000. É o canal morto que o `CLAUDE.md` do repo
manda medir antes de mexer no peso.

⚠ A regra "σ = a distância inicial" não se aplica a um alvo CONGELADO. Um alvo que nasce
da pose atual tem erro inicial zero por definição, então o σ dele cai no piso sempre. A
regra foi desenhada para o REORIENTAR, cujo alvo existe independente do robô.

## O que NÃO fazer

- **Mexer só no `sigma_ori`.** Dar σ vivo ao termo faz o gradiente funcionar apontando
  para MANTER O TOMBO. Ensinar mais alto a coisa errada é pior que não ensinar.
- **Mexer no peso do `precise_ori`.** Derivada de 1e−4 por grau não se conserta com peso.
- **Remover o `alinhado`.** Ver a restrição do dono abaixo.

## A restrição do dono (2026-09-21)

O `alinhado` existe porque **o robô real bota a caixa seguindo uma orientação
pré-determinada, não de qualquer jeito**.

⚠ **Mas esse requisito só vale quando o REORIENTAR entrar em treino.** Hoje
`reorientar_inerte = True` (`knobs.py:1049`): o elo é um atraso de 0,5 s e o fecho dele
ignora `alinhado`. Nada no currículo atual ensina orientação. Portanto a disciplina de
guinada é ALVO FUTURO, e não a exigência de hoje.

O que está errado não é exigir orientação — é a REFERÊNCIA ser a normal congelada da
caixa tombada nas mãos do robô. A referência precisa ser uma pose de destino, definida
pela laje e pela tarefa, e não pelo acaso de como o robô estava segurando.

O caminho em dois estágios usa o MESMO campo `_face_alvo_w`, sem mecanismo novo:

| estágio | referência do BOTAR | o que o fecho exige |
|---|---|---|
| hoje (REORIENTAR inerte) | a vertical do mundo | a caixa de pé |
| quando o REORIENTAR valer | a pose de destino da tarefa | de pé e com a face certa |

Com uma referência assim, três coisas se resolvem de uma vez, porque o `precise_ori` lê
o mesmo canal `ANG` do fecho:

1. o fecho passa a pedir o que a tarefa quer;
2. o `sigma_ori` deixa de degenerar, porque o erro na abertura passa a ser real
   (29,7° e 52,2° nos casos medidos) em vez do zero que o congelamento fabrica;
3. o gradiente aponta para a pose de destino em vez de apontar para o tombo.

## Ressalva de método

O `registra_juntas` troca de fase por TEMPO, não pelo fecho. Portanto estes registros
não provam que o PEGAR fecha com a caixa tombada 30° a 52° no treino — se fechasse, o
`alinhado` do próprio PEGAR já teria reprovado. O `s_B` = 0,32 diz que o PEGAR fecha.
Falta medir o tombamento no instante do fecho REAL do PEGAR: se a caixa chega direita ao
BOTAR no treino, o tamanho do problema muda.
