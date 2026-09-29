# A pega parada na mesa: o tombo é cinemático, e o punho é a única junta que o desfaz

Data: 2026-09-29. Run: `zero12` (commit `7d4be42`), platô da it ~1900 à 2135. Checkpoint
medido: `model_1750`, nível 0, 32 envs, 18 s, ator determinístico (`scratchpad/pega_parada.out`).
Sem simulação nova depois disso; o resto é leitura de código, de log e conta de modelo plano.

## 0. Conclusão

1. O robô aperta a caixa (força 4,9× `F_ref`), a mantém de pé (0,25°) e parada na mesa,
   a 0,29 m do alvo. Nenhum env sobe mais de 13 mm. Renda 12,7/s.
2. Levar a caixa da mesa à âncora do peito gira a mão de 20° a 35° POR CONSTRUÇÃO: ombro,
   cotovelo e `wrist_pitch` giram no mesmo eixo (`g1.xml`: `axis 0 1 0` nos três). A caixa
   segue a mão. Só o `wrist_pitch` desfaz o giro sem mover a caixa, e a escala da ação dele é
   6× menor (0,075 rad/unidade). O tronco também desfaz, mas três termos o seguram.
3. Até 28/09 erguer tombado pagava +5,5/s sobre ficar na mesa: o robô erguia e tombava
   (`zero11`, 20° a 48°). Desde a S3 (29/09) erguer tombado paga +2,9/s a 30° e nada a 40°.
   Erguer reto pagaria +7,4/s, mas exige o punho, que a política nunca girou para o lado certo.
4. O `precise_ori` ×4 NÃO é a trava (rejeitado em 29/09): ele paga o mesmo na mesa e no
   alvo com a caixa reta, e zerá-lo só barateia o tombo.

## 1. Medição no `model_1750` (PEGAR_COM, 26 208 passos)

| grandeza | p10 / p50 / p90 |
|---|---|
| força da palma mais fraca / `F_ref` | 3,8 / 4,9 / 6,7 |
| descarga da mesa | 0 / 0 / 1 (82 % dos passos na mesa) |
| subida máxima por env, mm | 2 / 8 / 13 |
| `ANG` | 0,1° / 0,25° / 1,2° |
| caixa ao alvo, m | 0,23 / 0,29 / 0,34 |

Renda por segundo: `staged` 4,20, `precise_ori` 3,66, `forma_postural` 2,32, `squeeze` 0,98,
`upright` 0,81, `pose` 0,43, `precise_pos` 0,29, `unload` 0,11; soma 12,7. Cerca de 11/s
não dependem de progresso. Freio −0,01, `faixa_de_pose` −0,09, `limite_de_junta` −0,01.

## 2. Cinemática plana (MJCF, sem simulação)

Braço: L1 = 0,198 m (ombro → cotovelo), L2 = 0,264 m (cotovelo → sítio da palma). Passo da
mão = `hip/waist_pitch + shoulder_pitch + elbow + wrist_pitch`. Pega na mesa de 0,55: pelve
0,62, tronco 12°, caixa a 0,30 m à frente → mão a +13° (ponta para baixo). Com a caixa a
0,40 m (o `d_alvo` p50 = 0,29 m sugere isso), +28°.

Tombo SEM punho ao levar a caixa ao alvo, robô de pé (pelve 0,78, tronco 0°), em graus,
referência mão +13° na pega (subtrair até 15° se a pega foi a +28°):

| z do alvo \ x à frente | 0,25 | 0,30 | 0,35 | 0,40 |
|---|---|---|---|---|
| 0,75 | +15 | +19 | fora | fora |
| 0,80 | +3 | +5 | +11 | fora |
| 0,85 | −7 | −7 | −3 | +9 |
| 0,90 | −18 | −18 | −14 | −6 |
| 0,95 | −30 | −28 | −25 | −17 |

A âncora de hoje é x = 0,25 (`peito_b`), z sorteado em (0,85; 0,95): tombo −7° a −30°, ou
−22° a −45° com a pega a +28°. O portão do fecho é 25°. O "1,6 rad/m" medido em 28/09 é
esta geometria (o modelo dá 1,2 a 1,4 rad/m) mais o punho empurrado para o lado errado.

## 3. Economia (peso × coluna, `alcançar` 0,96, σ_trazer 0,34, σ_pos 0,18, σ_ori 25°)

| situação | staged + precise_pos + precise_ori | + unload + postura | total | Δ sobre a mesa |
|---|---|---|---|---|
| mesa, de pé | 8,2 | 0,2 | 12,7 | — |
| alvo, tombada 30° (antes da S3) | 11,1 | 2,7 | 18,1 | +5,5 |
| alvo, tombada 30° (hoje) | 8,5 | 2,7 | 15,6 | +2,9 |
| alvo, tombada 40° (hoje) | 7,0 | 2,7 | 14,1 | +1,5 |
| alvo, reta (hoje) | 13,4 | 2,7 | 20,1 | +7,4 |

## 4. O que foi conferido e NÃO é a trava

- Freio `velocidade_por_regime`: dobradiça zero até 1,5 rad/s (punho 1,0); erguer em 1 s é grátis.
- `pose`: ombro e cotovelo saem da média com `pegou`; punho com σ 1,0 (−0,07/s a 1,2 rad).
- `faixa_de_pose`: `punho_pitch` tolera 0,6 rad no PEGAR_COM; `braco_pos` 1,5.
- `limite_de_junta`: a pose de carregar fica abaixo de 75 % do curso em todas as juntas do braço.
- `terminacao`: −200 × dt = −4 por evento; 15 % dos episódios terminam antes do tempo.
- `contato_tronco`: sensor tronco–laje, não tronco–caixa.
- `forma_postural`: referência presa em h = 0,68 (tronco 22°, pelve 0,72); ficar de pé custa ~0,2/s.
- Observação: o ator VÊ o tombo (`giro_b` = eixo × ângulo até a vertical, frame da base).

## 5. Caminhos

A. **Baixar a âncora** para onde a subida natural cai reta: `altura_carregar_faixa`
   (0,85; 0,95) → (0,75; 0,85) (`knobs.py:310`/`comando.py`). Tombo esperado −15° a +15°,
   dentro do portão, sem punho. Custo: a caixa viaja na altura do quadril (fundo a
   0,65–0,78 m), folga à coxa 5 a 8 cm ao andar. A S1–S5 fica como está.
B. **Manter a âncora** e afrouxar a janela nos termos de aproximação: `staged` e
   `precise_pos` usam só a reta `1 − Δθ/π` (`_alinha(aproximacao=True)`), e o híbrido com
   a gaussiana de 25° fica no `precise_ori`. Erguer tombado passa a pagar +4,3/s a 30° e
   +2,6/s a 60°; reto, +7,4/s. O fecho a 25° continua exigindo o punho.
   **ESCOLHIDO pelo dono em 29/09; RUN `zero13`.**
Rejeitados: escala do punho (dono); `precise_ori` 4 → 0 (não muda o caminho reto).
