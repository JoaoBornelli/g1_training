---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
---
# G1 limpo — registro de checkpoints que completam o ciclo

Registrado em 01/10 a pedido do dono, para reusar ou comparar. "Completa" = no play do
`registra_juntas.py` (roteiro `botar`, a caixa volta à MESMA laje da pega) a caixa termina
pousada na laje, de pé, e o robô sai andando de pé (~6 m). O registrador difere do treino
(`giro_w` = 0, mesma laje); o resultado é triagem, não veredito.

## Como o commit foi achado

O checkpoint `.pt` não grava o commit (só `iter` e o estado de currículo). O notebook do
Kaggle clona a branch `exp/g1-limpo-v2` no início da sessão. O commit abaixo é:
- **CERTO** quando a sessão foi acompanhada nesta conversa;
- **PROVÁVEL** quando sai da hora do download, do RUN da memória e do `git log`.

Alterações feitas só no notebook (fora do git) estão na coluna "fora do git".

## Linha zero (`RUN zero17`, treino do zero, currículo de cadeia)

| Checkpoint | sha256 (12) | Commit do treino | Fora do git | Play | Veredito do dono |
|---|---|---|---|---|---|
| `model_5000(1).pt` | `b99947bca7a3` | `2f83bfd` CERTO (K1b + R1 + K12; resume do 4650) | `LR_RETOMA = 5e-4`; assert `regime_parado` | FALHA: caixa cai a 90° | "pega, segura e larga no botar" |
| `model_5350.pt` | `6d89b4fc2e5c` | `2f83bfd` CERTO (mesma sessão) | idem | completa, laje 0,55; solta ao trocar para ANDAR (−0,32 m/s); quadril e coxa E na laje | — |
| `model_5800.pt` | `66bee41a688e` | `c4835ca` CERTO (R4, BOTAR raso; resume do 5350) | idem | completa, laje 0,55; pouso −0,02 m/s; caixa a 4 cm do alvo, tombo 4° | "pega e bota, mas fica meio agachado, as pernas tocam a laje por baixo, e não bota nas lajes mais baixas" (MEDIDO: coxa/canela E em 137 de 150 passos do BOTAR, ~1 mm) |

## Linha bloco (resumes encadeados, bloco21 e bloco22)

| Checkpoint | sha256 (12) | Download | Commit do treino | Fora do git | Play | Veredito do dono |
|---|---|---|---|---|---|---|
| `model_17999.pt` | `ba122fef72ee` | 17/09 08:24 | `a4fbd43` PROVÁVEL (bloco21: `forma_postural` contra a IK; resume do 15999) | — | completa em 0,25 e 0,55 (larga a −1,49 m/s na 0,55); robô cai na 0,05 | — |
| `model_19300.pt` | `1e4acafdc54c` | 17/09 11:37 | `9495e32` PROVÁVEL (bloco22: sola e rotação do quadril no `forma_postural`, peso 3,0; resume do 17999) | — | completa em 0,25 (−0,74 m/s) | — (deriva de rumo medida: −0,38 rad/s na pega) |
| `model_20500.pt` | `9e2f4cc35e4e` | 17/09 15:28 | `9495e32` PROVÁVEL (mesma sessão) | — | completa em 0,25 (−0,06 m/s) | "bota com a perna toda errada" |
| `model_20900.pt` | `0846d407b036` | 17/09 16:19 | `9495e32` PROVÁVEL | — | não gravado | ponto de partida do 21700 |
| `model_21800.pt` | `4a058a9b9c71` | 17/09 20:54 | entre `9495e32` e `a37300c` PROVÁVEL (continuação de 17/09) | — | FALHA na 0,15 com o clip de `hip_yaw` 0,7 (caixa a 90°) | — |
| `model_21700.pt` | `8cf48038cbf2` | 18/09 10:40 | `7b1b5ac` QUASE CERTO, branch `exp/g1-limpo-v2` (verificado 02/10: o código de treino é idêntico de `a37300c` a `7b1b5ac`, só o notebook muda em `7e4c662`; o `b226853`, com o clip do `hip_yaw` a ±0,7, foi revertido às 08:41 e o veredito fala de terminação, não de clip; resume do `model_20900`, outra sessão que a do `model_21800`) | terminação `|hip_yaw| > 1,0 rad` colada no notebook | completa em 0,15 (−0,22 m/s) | "todas as tarefas satisfatórias, menos o BOTAR: consegue, inclina o tronco demais sem esticar os braços, gira a perna"; 01/10: "pega e bota melhor, mas perna ainda torta" |
| `model_22300.pt` | `5695cb97f99f` | 18/09 11:50 | `7b1b5ac` PROVÁVEL (mesma sessão do 21700) | idem | não gravado no roteiro `botar` | — |
| `model_24999.pt` | `d6a8c2702475` | 18/09 21:17 | `7b1b5ac` PROVÁVEL (fim da bloco22) | idem | completa em 0,55 (larga a −1,16 m/s) | não gira no lugar (01/10); pega a 1,58 m/s e 2,5 g |
| `model_23250.pt` (= `model_23250(1).pt`) | `7967fa59d9b4` | 02/10 | `39b8b82` CERTO (`exp/g1-limpo-v2`, R4b; run `21700_zero_teste`: resume do `model_21700` da linha bloco, config do zero17, LR 5e-4; 21700→21950 com 4096 envs, 21950→23250 com 8192) | — | linha 22500–23000: bota em 0,55, 0,35 e 0,15 (1 e 3 kg) e se levanta; giro na pega caiu de ~50° a 5–7° | MELHOR MODELO até 02/10 (dono). Último checkpoint antes de o Colab encerrar por recursos. Log da it 23253: C 44% de sucesso, `s_C` 0,50, desvio do PEGAR 32% |
| `ckp9100` (arquivo não está em `~/Downloads`) | — | ~09/09 | `1dbfa06` PROVÁVEL (bloco13, v3.3) | — | completa em 0,35 e 0,55; falha na 0,15 (caixa a 90°) | — |

Todos os `.pt` estão em `~/Downloads`. Os CSVs: `5000_botar.csv`, `5350_botar.csv` e
`5800_botar.csv` na raiz do repo; os da linha bloco em `g1_limpo/registros/`.

## Para usar um checkpoint

- Resume no Kaggle: subir o `.pt` no dataset de entrada e confirmar o commit no print
  `clone =`. Checkpoint da linha bloco pede o commit dela (o conjunto de recompensas é outro)
  e a terminação fora do git do 21700.
- Comparar no registrador: `python registra_juntas.py --cena ~/g1_pilota/ --checkpoint
  ~/Downloads/<arquivo> --roteiro botar --topo <laje> --saida <nome>`.
