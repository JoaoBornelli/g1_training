---
tipo: achado
pai: "[[Princípios de recompensa]]"
estado: ativa
verificado: 2026-10-06
resumo: "Medido 2026-08-26 no g1_limpo, respondendo a pergunta do dono: 'a recompensa por nao se mover na manipulacao nao pode ser maior que a de executar as acoes com a caixa'."
---
# g1-sigma-e-a-distancia-inicial (original)

Corpo original da memória `g1-sigma-e-a-distancia-inicial`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Medido 2026-08-26 no `g1_limpo`, respondendo a pergunta do dono: "a recompensa por nao se mover na manipulacao nao pode ser maior que a de executar as acoes com a caixa".

A pergunta e' a certa, mas o risco NAO esta onde parecia. Tres condicoes, duas medidas:

**1. Teto supera piso — satisfeita com folga.** Ficar parado paga **5,81/s** (robo travado, elo `PEGAR`: `track_lin` 1,83 + `track_ang` 1,98 + `pose` 1,0 + `upright` 1,0). Fechar a tarefa paga isso mais os sete incentivos (~12,5/s). Razao ~3:1.

**2. Mover e' BARATO.** Com a base pinada, movendo as 14 juntas de braco: rampa de 2 s custa 0,103/s, de 1 s custa 0,176/s, de 0,5 s custa 0,249/s. Isso e' **2% a 4% do piso**. E 0,133 dos 0,15 vem de UM termo, o `track_angular_velocity`; os freios de movimento (`action_rate_l2`, `joint_acc`, `body_ang_vel`) somam 0,009 — irrelevantes. ⚠ Limite INFERIOR: base pinada nao conta a inclinacao que o braco causa nem a mudanca de velocidade do agachamento.

**3. O GRADIENTE NO REPOUSO — e esta e' a que matou o `g1_poc`.** A palma nasce a **0,339 m** da caixa (min 0,211, max 0,481). Kernel `exp(-d^2/sigma^2)` ali, com peso 3,0:

| sigma | valor | derivada | ganho do 1o cm |
|---|---|---|---|
| 0,05 | 1e-20 | 0,00 | 0,0000 |
| 0,10 | 1e-05 | 0,00 | 0,0000 |
| 0,20 | 0,056 | 0,95 | 0,029 |
| 0,30 | 0,278 | 2,10 | 0,063 |
| 0,50 | 0,631 | 1,71 | 0,051 |

Com sigma <= 0,10 o termo vale zero **e a derivada e' zero**. O robo move a mao 1 cm para perto e nada muda; 1 cm para longe e nada muda. Nao e' o robo PREFERINDO a estatua — e' o robo sem sinal nenhum que aponte para a caixa. Analogia: sino estreito = a luz so acende quando voce ja esta tocando o objeto.

**REGRA ADOTADA:** `sigma_env = ||palma - caixa|| no instante em que o elo abre`. Todo env nasce em `exp(-1) = 0,368` com derivada `2/d0 x 0,368` — 3,49 no env mais perto, 1,53 no mais longe. Vivo nos dois extremos, e sem numero magico.

**Sobra declarada:** o 1o centimetro e' negativo (ganha 0,063, custa 0,15); o equilibrio fica a 2-4 cm. Nao trava o PPO, porque ele otimiza RETORNO e nao recompensa de passo: fechar 0,34 m paga 3,0/s pelo resto do episodio contra perder 0,2/s por ~1 s. E o ruido de exploracao move a palma ~0,2 m contra um vale de 3 cm. **So uma run decide.**

**PRE-REGISTRADO:** se o alcance nao aparecer na F3, mover o **sigma** (mais largo), NUNCA o peso. Tornar o 1o cm positivo exigiria peso > 12, quatro vezes o da locomocao, e o robo pararia de andar.

**IMPLEMENTADO na F3 do g1_limpo, 2026-08-26.** MEDIDO em 48 envs: kernel 0,368 em todos, derivada minima acima de 1,0/m. O sigma e recalculado na passada do `_pendente`, o unico ponto do reset em que a pose da palma, da caixa e do alvo esta fresca.

**E um defeito meu que a medicao pegou: a distancia tem de ser ate a SUPERFICIE da caixa, nao o centro.** Ao centro, o minimo fisicamente alcancavel e 0,191 m (a caixa colide com a mao antes), logo o kernel saturava em **0,674** e o `staged` nunca passava de 3,35 de um teto de 6,0 -- o robo fazia a tarefa inteira e o termo dizia "faltam 33%". Subtraindo a meia-aresta, o kernel chega a 0,887 no contato; nao chega a 1,0 porque o sitio da palma fica DENTRO da mao, e isso e bom (sobra gradiente ate o contato, onde o `squeeze` assume).

**Preco declarado:** com `sigma = d0` o kernel nasce em `exp(-1)`, portanto o `staged` paga **1,95/s a uma estatua**. O piso do elo parado subiu de 5,81/s para **8,03/s**. Fechar a tarefa paga ~20/s, razao ~2,5:1 -- a decisao se mantem. Duas consequencias de leitura: `Episode_Reward/staged` NAO mede progresso (nasce em 1/3), e um env de manipulacao paga mais que um de locomocao (8,03 contra 4,44), o que o controlador de fatia da F5 tem de levar na conta.

Relacionado: g1-poc-auditoria-incentivos, g1-principio-incentivo-nao-penalizacao, g1-limpo-desenho, g1-pegar-plato-do-grasp-booleano
