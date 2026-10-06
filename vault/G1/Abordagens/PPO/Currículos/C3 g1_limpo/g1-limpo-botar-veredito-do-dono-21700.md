---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Veredito do dono no model_21700 (18/09): tudo satisfatório menos o BOTAR — tronco 65°, braço a 55% do alcance, hip_yaw E −34°; hipótese de hábito herdado do resume."
cluster: c3
---
# BOTAR: o veredito do dono no `model_21700` e a medição que o sustenta

Registrado em 2026-09-18. O checkpoint é o do resume `bloco22` (Kaggle), que partiu do
`model_20900` com a terminação `|hip_yaw| > 1,0 rad` colada só no notebook, fora do git.

## O veredito, nas palavras do dono

> Todas as tarefas, excluindo a de botar a caixa, independente da altura, estão
> satisfatórias. A tarefa de BOTAR a caixa não: ele consegue, mas não é uma movimentação
> eficiente. Inclina o tronco demais sem esticar os braços, o que pode estar interferindo
> no equilíbrio, e ainda está tentando girar a perna para fora.
>
> Esses são comportamentos que podem ser derivados do resume e não ocorram no
> treinamento do zero, mas preciso registrar.

Ponto de partida do dono: o PEGAR e o BOTAR deveriam ser a mesma movimentação, ou muito
próxima, só que inversa. Hoje não são.

## A medição (`21700_botar_resume.csv`, laje da cena de `juntas.csv`)

Cinemática reconstruída com o `cena.mjb` de `~/g1_pilota` a partir das 29 juntas e da
raiz do CSV (`mj_forward`). Medianas por fase; o BOTAR são os últimos 0,3 s.

| grandeza | PEGAR | BOTAR (fim) | referência da IK na altura do BOTAR |
|---|---|---|---|
| ombro → palma E / D | 0,23 / 0,25 m | 0,25 / 0,27 m | 0,41 a 0,46 m (todas as 15 poses) |
| cotovelo E / D | −38° / −26° | −16° / −18° | +30° a +63° |
| ombro pitch E / D | 16° / −1° | −23° / −64° | −57° / −57° |
| pelve z | 0,74 m | 0,53 m | 0,52 a 0,58 m |
| inclinação do tronco | 10° | 65° | 67° |
| hip_yaw E / D | 1° / −1° | −34° / −4° | ±9° |
| caixa − raiz, horizontal | 0,27 m | 0,46 m | 0,35 m |

Alcance do braço no modelo, ombro → palma, com ombro e punho em zero:

| cotovelo | −60° | 0° | +34° (keyframe) | +60° | +90° |
|---|---|---|---|---|---|
| alcance | 0,168 m | 0,345 m | 0,415 m | 0,445 m | 0,453 m |

Cotovelo positivo ESTENDE o braço neste modelo. O robô opera com o braço em 55 % do
alcance máximo, nos dois elos.

## O que a medição diz

1. **PEGAR e BOTAR já usam o mesmo braço.** Dobrado em 0,24 m nos dois. O que difere é o
   corpo: no PEGAR a caixa está a 0,96 m e o robô fica em pé; no BOTAR a laje está baixa
   e o robô desce a pelve e deita o tronco.
2. **O robô copia da referência só o que a `forma_postural` paga.** Pelve e tronco
   batem com a IK na mesma altura. O braço, que a referência estende a 0,41–0,46 m, não
   entra em nenhuma das seis grandezas. O `pose` desliga ombro e cotovelo enquanto o
   robô segura a caixa (`recompensas.PosturaPorElo`, máscara `limpo_pegou`), e a
   `faixa_de_pose` tolera 1,5 rad no braço em todo elo de manipulação.
3. **Na laje baixa a caixa fica alta.** Com o corpo na pose da referência e o braço
   0,18 m mais curto, a caixa para acima do alvo. O robô chega onde a laje é alta o
   bastante para o braço dobrado, e compensa o resto deitando mais o tronco.
4. **A torção do quadril caiu, e não sumiu.** O `hip_yaw` esquerdo foi de −96° (21800,
   antes da terminação) a −34°. A terminação em 1,0 rad = 57° tirou a torção grande;
   os 34° estão abaixo dela e 25° acima da referência.

## A hipótese do dono e como testá-la

Hipótese: os três hábitos (tronco deitado, braço dobrado, perna para fora) vêm do
resume, que herdou 20 900 iterações com a rampa do `limite_de_junta` a 85 % e sem a
terminação. O treino do zero (`g1_limpo_zero.ipynb`, Colab, `RUN = "zero01"`), com o
limiar de 40° e o rumo desde a primeira iteração, pode não os criar.

Teste: no primeiro checkpoint do zero com `s_C > 0,3`, rodar o `registra_juntas` no
BOTAR em laje baixa e alta, e ler as mesmas sete linhas da tabela acima. Três perguntas:

- ombro → palma no fim do BOTAR fica acima de 0,35 m?
- inclinação do tronco fica abaixo da referência na mesma altura, ou acima?
- `|hip_yaw|` fica abaixo de 20° sem terminação?

Se o zero repete o braço em 0,25 m, o hábito não é do resume: é a `forma_postural` sem
o braço, e o conserto é o braço entrar como grandeza, lido do `qpos` que o
`ref_botar.npz` já guarda. Se o zero estende o braço, o resume carrega o hábito e o
caminho é o zero.

## O que NÃO foi feito

Nenhuma mudança de código. A proposta do braço como grandeza da `forma_postural`, com
pesos por grandeza, fica em discussão. Memória do tópico:
`~/.claude/memory/g1-limpo-botar-deita-o-tronco.md` e `g1-limpo-forma-postural.md`.
