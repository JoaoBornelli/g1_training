# Postura de pé — spec v1

Data: 30/09. Enunciado: `specs/g1-limpo-comportamento.md` §3 ("Postura", "De pé, como um
humano") e §4.4. Auditoria: `docs/relatorios/2026-09-29-auditoria-de-recompensa.md`,
achado 31.

## 0. Resumo

Com a caixa nas mãos, o robô dobra no quadril: a pelve inclina 22° para a frente e a
cintura volta −0,33 rad para o tronco ficar reto. A causa medida é a referência da
`forma_postural`. Com a caixa acima da laje, a chave passa do fim da tabela de IK, e o
`clamp` devolve a linha de agachamento da laje de 0,55: pelve 0,72 m e tronco 22°. O
`upright` pede o tronco a 0°. Nenhum termo mede a pelve.

A fase 1 desta spec estende a tabela para cima com a pose de pé com caixa, que a IK de
30/09 mediu. Ela também acrescenta a inclinação da pelve como grandeza da forma, e liga a
forma no CARREGAR. Não entra termo novo. A fase 2, de pé sem caixa, fica aberta: ela pede
uma medida do ANDAR antes.

## 1. O problema, medido

Play do `model_3300` (`juntas.csv`, 30/09), pairando no alvo com a caixa:

| | Pelve z | Pelve incl. | Tronco | Quadril | Joelho | Cintura |
|---|---|---|---|---|---|---|
| Play, pairando | 0,717 m | +22° | −1° | −0,89 | 0,76 | −0,33 |
| Referência de hoje (clamp na linha h = 0,68) | 0,720 m | 4,7° | 22,1° | — | — | — |
| IK de pé, caixa a 0,90 m (§2) | 0,78–0,79 m | 0° | 0° | −0,07 a −0,24 | 0,20–0,42 | 0 |

A pelve do play bate com a linha presa: 0,717 contra 0,720. Três termos formam a pose:

| Termo | Peso | O que pede na pose do play |
|---|---|---|
| `forma_postural` | 3,0 × 1 (PEGAR_COM) | pelve 0,72 m e tronco 22°, a linha da laje de 0,55 |
| `upright` | 1,0 | `torso_link` na vertical |
| `postura_ereta` | 2,0 × 1 | pelve acima de 0,75 m; o termo é multiplicado pela reta de `_alinha`, e com a caixa a 60° paga um terço |

O robô atende a pelve da forma e o tronco do `upright` ao mesmo tempo. Ele baixa a pelve
dobrando o quadril, e a cintura compensa. A orientação da pelve é livre: a forma a deixa
de fora ("a terminação de 70° já cobra", `recompensas.py`, docstring da `FormaPostural`),
e o defeito está a 22°.

A pose não é física obrigatória. A IK sem as tarefas de postura (§2, "livre") inclina o
corpo para trás, nunca para a frente. O quadril do play segura 7,6 N·m contra 2,6 N·m da
pose de pé com 1 kg.

## 2. A referência de pé (IK, 30/09)

**Método.** IK com prioridade de tarefas sobre o `cena.mjb` exportado
(`~/g1_pilota_zero15`), reaproveitando `g1_limpo/ik/gera_botar.py`. Script da medida:
`pose_de_pe.py` do scratchpad desta sessão.

- Tarefas: pés planos no chão, paralelos e alinhados, largura entre 0,20 e 0,30 m; CoM
  combinado do robô e da caixa sobre o meio dos pés; palmas nas faces laterais, com a
  normal do pad na face; pelve nivelada e tronco vertical; sem rotação interna do quadril;
  folga entre peças; 10% de margem no curso de cada junta.
- Custo secundário, só no espaço nulo das tarefas: o torque estático de cada junta,
  normalizado pelo limite do motor, com peso 1 nas 15 de perna e cintura e 0,3 nas 14 do
  braço; mais um repouso fraco. O torque estático soma a gravidade, a reação dos dois pés
  (mínimo-norma) e o peso da caixa nas palmas.
- A caixa fica no alvo de transporte: `base + R_base · peito_b`, com o z absoluto.

**Resultado:**

| Pose | Pelve z | Pelve / tronco | Quadril | Joelho | Tornozelo | Pés à frente da pelve | τ joelho | τ quadril |
|---|---|---|---|---|---|---|---|---|
| Default de hoje (keyframe) | 0,756 m | 0° / 0° | −0,31 | 0,67 | −0,36 | 3,9 cm | 10,6 N·m | 0,5 N·m |
| De pé, sem caixa | 0,789 m | 0° / 0° | −0,01 | 0,15 | −0,13 | 0,5 cm | 0,3 | 0,1 |
| De pé, caixa 0,85–0,95 m, 1 kg | 0,788–0,789 m | 0° / 0° | −0,07 a −0,08 | 0,20–0,21 | −0,12 a −0,13 | 2,7 cm | 0,4–0,8 | 2,6–2,8 |
| De pé, caixa 0,85–0,95 m, 5 kg | 0,777–0,787 m | 0° / 0° | −0,15 a −0,24 | 0,24–0,42 | −0,09 a −0,18 | 6,2–6,6 cm | 0,8–2,3 | 8,2–8,5 |
| Livre (sem pelve e tronco), 0,90 m, 1 kg | 0,788 m | −6° / −12° | +0,12 | 0,13 | −0,14 | 0,9 cm | 0,1 | 1,0 |

A largura dos pés sai 0,231 m em todas as poses. O tornozelo segura ~6,5 N·m em todas: o
CoM cai no meio do pé, ~4 cm à frente do tornozelo. A altura da caixa, de 0,85 a 0,95 m,
só muda os braços.

**Carga no braço, parado.** Com 5 kg o punho usa de 46% a 56% do limite de 5 N·m, o
ombro de 32% a 40% e a cintura 17 N·m (34%). Com 1 kg: punho 20%, ombro 11%, cintura
6 N·m.

**Limites da medida.** É estática: sem dinâmica e sem marcha. A divisão da reação entre os
pés é o mínimo-norma. A caixa entra como peso vertical nas palmas.

## 3. Objetivo e fora de escopo

**Objetivo (fase 1).** Com a caixa nas mãos acima da laje, a referência de forma é a pose
de pé com caixa do §2: pelve nivelada, tronco vertical, pelve a 0,78–0,79 m. Isso vale no
PEGAR que ergue, na SEGURA e no CARREGAR. Enunciado §3 e §4.4.

**Fora de escopo:**
- a fase 2, de pé sem caixa (§8);
- a referência do `pose` e o keyframe default: a ação do mjlab é `default + escala × a`,
  e trocar o default desloca todas as ações da política treinada;
- os braços, que seguem a caixa;
- torque como recompensa (enunciado §6: "Torque e energia como regra explícita: por
  enquanto não").

## 4. A mudança (fase 1)

1. **Linhas novas na referência.** Um gerador próprio, `g1_limpo/ik/gera_de_pe.py`, com o
   solver de prioridade de tarefas e o custo de torque do §2. Ele importa a `Cena` e as
   constantes do `gera_botar.py`, que fica igual: as 15 poses aprovadas continuam
   reproduzíveis. Ele gera `g1_limpo/ik/ref_de_pe.npz` com as linhas h = 0,85, 0,90,
   0,95 e 1,00 m, caixa de 3 kg (o meio da faixa) e meia aresta 0,10, no esquema do
   `ref_botar.npz` (`topo = h − meia`).
1b. **Mais duas lajes no BOTAR.** A laje do BOTAR é a do PEGAR ±10 cm
   (`botar_delta_topo`), com teto no fundo da caixa menos 5 cm: ela vai até ~0,67 m, e o
   alvo até h ≈ 0,80. A tabela do BOTAR acaba em h = 0,68 (laje 0,55). Simulado por nível:

   | Nível | Laje do BOTAR (p5–p95) | BOTAR com o alvo acima da tabela |
   |---|---|---|
   | 0 | 0,46–0,65 m | 38% |
   | 1 | 0,39–0,61 m | 13% |
   | 2 | 0,27–0,58 m | 5% |
   | 3 e acima | até 0,56 m | 3% |

   `gera_botar.py --lajes 0.60 0.67 --pernas-de g1_limpo/ik/ref_botar.npz` gera só as 6
   poses novas, e elas são anexadas às 15 de hoje, que não mudam. As novas passam pela
   aprovação no viewer. O check do smoke
   "a tabela da IK cobre toda altura" (`smoke.py:1522`) passa a somar o
   `botar_delta_topo` e o jitter de 2 cm da laje: hoje ele olha só a laje do PEGAR.
2. **A tabela junta as duas referências.** `FormaPostural.__init__` lê as duas, concatena
   e ordena por h. Com o item 1b, as poses do BOTAR vão até h ≈ 0,80 (laje de 0,67); entre
   ali e h = 0,85 (de pé com caixa), a interpolação linear endireita a referência
   conforme a caixa sobe.
3. **Sétima grandeza: a inclinação da pelve.** `acos` do eixo z da pelve contra a vertical,
   contra o `pelve_incl` da tabela, escala 60°. No agachamento das lajes baixas a tabela
   pede até 55,8°; na pose de pé, 0°.
4. **Máscara de regime.** Com o comando de andar ativo (`limpo_twist_zerado < 0,5`, só no
   CARREGAR andando), a altura da pelve, a largura dos pés e a sola saem da média: elas
   seguem a marcha. Ficam a pelve, o tronco, o pad e o quadril. A média normaliza, e
   parado e andando têm o mesmo teto.
5. **A coluna CARREGAR da `forma_postural` vai de 0 para 1.** O `alcancar` continua
   multiplicando a forma: no CARREGAR ele vale ~0,6 com as mãos na caixa.
6. **Knobs.** `FormaPostural.referencia` vira tupla (`ik/ref_botar.npz`,
   `ik/ref_de_pe.npz`); `escala_pelve_incl_deg = 60`.
7. **RUN nova** (`zero18`), com resume do último checkpoint da `zero17`: a impressão
   digital da tabela muda.

**Derivada no ponto de operação** (a pose do play, `alcancar` 0,6, peso 3,0 × 1). A
`forma_postural` é a média das grandezas × `alcancar` × peso:

| Grandeza | Hoje (6 grandezas) | Depois (7 grandezas) |
|---|---|---|
| Altura da pelve | 1,0/s por m, puxando PARA 0,72 m | 0,86/s por m, puxando para 0,786 m |
| Tronco | 0,0033/s por grau, puxando para 22° (contra o `upright`) | 0,0029/s por grau, puxando para 0° (junto com o `upright`) |
| Inclinação da pelve | não existe | 0,0043/s por grau, puxando para 0° |

Da pose do play até a pose de pé, a forma passa a pagar +0,15/s. Hoje o mesmo movimento
custa −0,07/s nela. Somando os outros termos de postura:

| Termo | Da pose do play à pose de pé, hoje | Depois da fase 1 |
|---|---|---|
| `forma_postural` | −0,07/s | +0,15/s |
| `pose` (21 juntas; 0,62 → 0,88) | +0,26/s | +0,26/s |
| `postura_ereta` (× reta da caixa a 60°) | +0,07/s | +0,07/s |
| Soma | +0,26/s | +0,48/s |

Hoje ficar de pé já rende +0,26/s, e a política não vai para lá. A pose do play é, então,
em parte um mínimo local do aprendizado, e não só o ótimo da recompensa. A fase 1 tira o
único termo que puxa contra e soma um canal para a pelve. Ela não garante a mudança: é o
risco 1 do §6. Contra os ~24/s da SEGURA, +0,48/s é pouco.

**Pré-verificação da IK (30/09, fora do repo).** As 10 poses novas foram geradas e vistas
em render. Todas passam no portão do gerador (pé < 1 mm, mão < 1 mm, folga ≥ −0,9 mm,
|frac| ≤ 0,87):

| Poses | Pelve z | Pelve incl. | Tronco | Joelho |
|---|---|---|---|---|
| BOTAR, laje 0,60, caixas de 0,14 a 0,26 | 0,713–0,732 m | 2,4–4,4° | 23,0° → 15,2° | — |
| BOTAR, laje 0,67, caixas de 0,14 a 0,26 | 0,737–0,744 m | 1,6–3,6° | 15,5° → 7,0° | — |
| De pé, caixa a 0,85–1,00 m, 3 kg | 0,787 m | 0,0° | 0,0° | 0,24–0,26 |

As lajes novas continuam a tendência da laje de 0,55 (pelve 0,69–0,72 m, tronco 26° a
22°). Com as 21 + 4 poses, a tabela cobre h de 0,11 a 1,00 m.

**Linhas, estimativa:** `gera_de_pe.py` novo, ~230; `recompensas.py` +15 (duas
referências, grandeza 7, máscara); `knobs.py` +3; smoke +25.

## 5. Efeito em cada termo

| Termo | Efeito |
|---|---|
| `postura_ereta` | alinhado: a referência nova (0,786 m) fica acima do pleno da rampa (0,75 m) |
| `upright` | alinhado: tronco 0° nos dois |
| portão `de_pe` do fecho | a pose de pé fica a no máximo 0,47 rad do default no joelho; o portão aceita 0,69 |
| `pose` | já alinhado: 0,62 na pose do play (pesam `waist_yaw`, `hip_pitch` e `waist_roll`) contra 0,88 na pose de pé |
| `limite_de_pelve` (CARREGAR) | alinhado: a pelve sobe |
| BOTAR e PEGAR_SEM | sem mudança: a chave fica abaixo de 0,68 m, nas linhas de hoje |
| `renda_congelada` | sem mudança: a forma não entra em `TERMOS_CONGELAVEIS` |
| estátua | o CARREGAR só existe com a caixa na mão, e o `alcancar` segue no produto; parado e andando têm o mesmo teto |
| princípio da tabela | o CARREGAR é a cauda de B e R, sem tarefa depois dele; pagar mais ali torna o fecho do PEGAR mais valioso |

## 6. Riscos

1. **O ganho é pequeno, e a pose pode ser mínimo local.** A forma passa a pagar +0,15/s,
   e a soma dos termos de postura +0,48/s. Hoje a soma já é +0,26/s e o robô não se
   levanta. Se em 500 iterações a pelve não subir no play, o peso da forma é o botão, com
   medida nova.
2. **Os pés estreitam ao erguer.** A tabela de BOTAR pede 0,40 m de largura, e a de pé
   0,231 m. Entre h = 0,68 e 0,85 a referência pede para fechar os pés, e o robô pode dar
   um passo com a caixa.
3. **Transiente na troca de RUN.** A política da `zero17` segura a caixa a 0,72 m. A pose
   muda enquanto o `s_B` ainda sobe, e o `s_B` pode cair por algumas centenas de
   iterações.
4. **A referência é estática.** No andar valem só a pelve e o tronco, pela máscara.

## 7. Verificação

**Sonda na CPU, sem update:**
1. As linhas novas saem como no §2 (pelve ≥ 0,77 m, tronco ≤ 2°, inclinação da pelve
   ≤ 2°), sem peça dentro de peça.
2. `FormaPostural.referencia()` em h = 0,68, 0,76, 0,85, 0,95 e 1,05 devolve a linha do
   BOTAR, a interpolação, a pose de pé e o `clamp`.
3. Na pose do play, a forma paga mais depois de nivelar a pelve: o sinal da §4.

**Smoke** (roda o dono): a tabela cobre de 0,11 a 1,00 m, e o `altura_carregar_faixa`
cabe nela; a forma tem 7 grandezas; a coluna CARREGAR vale 1; a máscara de regime está no
código.

**No log:** `Episode_Metrics/altura_da_pelve` sobe de 0,73 para perto de 0,78;
`tronco_na_pega` não sobe; `s_B` sem queda além do transiente.

**Aceite no play** (`juntas.csv`): segurando a caixa parado, pelve acima de 0,76 m e
inclinação da pelve abaixo de 10°.

## 8. Fase 2, aberta: de pé sem caixa

Estados: ANDAR parado, ESPERA_SEM e CAUDA.

- **Conflito com o `pose`.** A referência do `pose` é o default (joelho 0,67), e a pose de
  pé da IK tem o joelho em 0,15. O `pose` puxaria de volta ao joelho dobrado.
- **O `alcancar` não serve.** Sem caixa, ele vale ~0 e zeraria a forma.
- **Opções:** (a) só a orientação (pelve nivelada e tronco vertical) nesses estados, sem o
  `alcancar`; (b) pernas retas, com a referência do `pose` no regime parado vinda da IK.
- **Primeiro passo:** medir a postura de hoje no ANDAR, parado e andando, com um
  `juntas.csv` de roteiro ANDAR. A inclinação da pelve sem caixa ainda não foi medida.

## 9. Decisões do dono

- **D1 (30/09).** Registrar a pose de pé da IK no enunciado: feito (§3).
- **D2, aberta.** Aprovar a fase 1 como está no §4.
- **D3, aberta.** Escolher a opção da fase 2 depois da medida do ANDAR.

## 10. Portabilidade

A referência é cinemática: IK sobre o MJCF do robô. No simulador novo, o mesmo gerador
roda sobre o modelo novo e refaz as linhas. As sete grandezas não dependem do simulador.
