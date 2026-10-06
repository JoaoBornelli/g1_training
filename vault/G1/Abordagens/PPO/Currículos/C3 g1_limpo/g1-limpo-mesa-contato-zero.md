---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "IMPLEMENTADO em 2026-08-28, commit 6c6a5f8."
---
# g1-limpo-mesa-contato-zero (original)

Corpo original da memória `g1-limpo-mesa-contato-zero`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

**IMPLEMENTADO em 2026-08-28, commit `6c6a5f8`.** Tocar a mesa é TERMINAÇÃO
(`contato_ilegal`, limiar de 50 N medido no g1_poc), cobrindo o CORPO INTEIRO — os 33 geoms
de `.*_collision`, contra 6 antes.

A penalidade `contato_prateleira = -1.5` rodou 405 iterações do bloco 2 e caiu por medição: o
contato do tronco caiu monotonicamente (7,5% -> 3,8% -> 2,0% dos passos) E a manipulação caiu
com ele (`staged` 0,36 -> 0,17). O usuário parou o treino ao ver o movimento no `play`.

**Why:** multa que o robô pode pagar é multa que ele ORÇA. O princípio do g1_poc — terminar
em vez de penalizar — é o certo, e a reescrita o tinha perdido junto com os termos: até
28/08 o g1_limpo não tinha terminação própria nenhuma.

⚠ O que torna a lista de corpo inteiro segura é o LIMIAR DE FORÇA, não a lista: roçar o tampo
ao alcançar não termina, apoiar o peso termina. Com booleano, cobrir as 33 geoms tornaria
pose baixa inganhável.

⚠ `action_rate_l2 = -0.1` É o valor do fabricante (`velocity_env_cfg.py:318`), e o g1_limpo
já usa. O −1,0 é do g1_poc, que tem currículo nesse peso. Não confundir o PESO com o
`Episode_Reward` diluído.

**Why:** apoiar peso no móvel é o atalho que a marcha do bloco 1 tornou possível. A
justificativa no código para excluir o antebraço ("numa pega a 0,04 m ele passa perto do
tampo") cai se a mesa a 0,04 não deveria existir.

## Do ZERO, e não por resume (2026-08-28)

O resume do `model_4999.pt` com a terminação de 33 geoms FALHOU em 230 iterações:
`contato_ilegal` virou 28% das terminações, a duração caiu de 988 para 788, `eficiencia_min`
de 0,70 para 0,41, `upright` de 0,93 para 0,70, e `squeeze` e `postura_ereta` foram a ZERO
EXATO.

**Razão do usuário, e ela está certa:** a política da 4999 tinha o escoro como a estratégia
de manipulação de maior valor. A terminação não a corrige — ela apaga o único caminho que
aquela política conhecia. Desaprender custa mais que aprender certo. Do zero, escorar termina
desde a primeira tentativa e nunca vira hábito.

## A LISTA ESTÁ INVERTIDA (medido 2026-08-28, bloco 3 it 4251)

`add_pads_de_palma` APAGA `*_hand_collision` e cria `*_palm_pad` / `*_hand_back_pad`. Nenhum
`_pad` casa com `.*_collision`. Portanto a mão NÃO TEM GEOM no sensor `corpo_prateleira`: ela
é a única parte do corpo com custo zero para escorar.

O que a lista cobre no braço é `torso`, `shoulder_yaw`, `elbow_yaw`, `wrist` — exatamente as
peças que precisam chegar perto do tampo para pegar. O sinal está invertido: aproximar termina,
escorar é grátis.

MEDIDO: fatia de manipulação 24,7%, `contato_ilegal` 18,5% das terminações -> ~75% dos
episódios de manipulação morrem na mesa. `squeeze` = 0,0002 em 3200 iterações depois de a
fatia abrir; `staged` parado no valor de nascimento. O gradiente de aproximação existe e vale
+2,6/s, mas terminar custa −4,0 mais ~17 s de renda: o valor esperado de esticar o braço é
negativo.

O discriminador anterior ("se squeeze ficar zero, as mãos saem da lista") FECHOU com veredito
contrário: as mãos nunca entraram, e o resto do braço entrou sozinho. Conserto em três partes,
e as duas primeiras não funcionam separadas: (1) `CORPO_INTEIRO` ganha `.*_pad`; (2) punho e
cotovelo SAEM; (3) `unload` ganha porteiro de preensão — hoje derrubar a caixa paga 2,0/s sem
mão nenhuma.

**How to apply — RISCO LATENTE, ainda ABERTO.** Com `topo_min` em 0,04 m nos níveis 4 a 6 a
laje é um degrau de 4 cm na frente dos pés, e PISAR nela passa de 50 N: terminação. Nesses
níveis a mesa precisa deixar de existir, e isso NÃO está implementado. Falta: `topo_min`
terminar em 0,0 e `posiciona_cena` AFUNDAR a mesa quando `topo < caixa_meia_z` — não pode ser
`topo == 0`, porque o `topo` é sorteado CONTÍNUO em `[piso, teto]` e um lábio de 1 cm dispara
contato de pé sem parar. Os blocos 1 e 2 nunca saíram do nível 0 (topo >= 0,55 m), então é
risco adiado; se o nível subir, vira ativo.

**Retomar do `model_5000.pt` (fim do bloco 1)**, não do checkpoint do bloco 2 — aquele carrega
as 405 iterações em que a manipulação regrediu sob a penalidade.

Relacionado: g1-limpo-bloco1-marcha-formou, g1-limpo-desenho
