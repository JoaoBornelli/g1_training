# ADR-0001: Currículo do `pegar` — rampa de altura de mundo, ponte de carga e bônus de objetivo

**Número do ADR:** 0001
**Título:** Currículo do `pegar` — rampa de altura de mundo, ponte de carga e bônus de objetivo
**Data:** 2026-08-17
**Responsável:** João Bornelli
**Status:** Aceito

---

## Contexto

O experimento multi-tarefa do G1 treina **uma política só**, condicionada a comando. A
tarefa não é um ambiente diferente: ela é um one-hot dentro do vetor de comando. São
cinco tarefas — `locomover`, `pegar`, `reorientar`, `locomover_carregando`, `botar` — e um
orquestrador abre cada uma quando a anterior atinge competência.

O `pegar` é o gargalo do grafo. Ele é pai do `locomover_carregando`, que é pai do `botar`.
Enquanto ele não fecha, duas tarefas nunca abrem.

Ele rodou **22 mil iterações sem pegar a caixa**. O comportamento observado no `play`: o
robô aproxima as duas palmas, agacha levemente, encosta na caixa, e não a ergue mais que
alguns centímetros.

Os números do log, convertidos pelos pesos efetivos da tarefa. O `contrib` publicado é
`peso × valor`, porque o `_step_reward` divide o `dt` de volta
(`reward_manager.py:132`):

| grandeza | valor |
|---|---:|
| `grasp` (preensão estabelecida) | 0,851 |
| `reaching` (palmas na face) | 0,80 |
| `lift` (progresso de altura) | **0,011** ⇒ a caixa sobe **4 mm** |
| `box_at_peito` | 0,000 |
| sinal de tarefa somado | **1,07 de 4,00** |
| `action_rate_l2` | **−0,88** (82% do sinal coletado) |
| penalidades somadas | −1,15 |
| `upright` (fundação, sem gate) | 0,97 |
| `posture` no `pegar` | 0,000 (o `locomover` coleta 0,276) |
| `cond_fisica` (a régua do currículo) | **0,0000** |

O robô tem as mãos na caixa em 85% dos passos e não a levanta. A régua do currículo mede 0
em todos os passos, portanto nenhum evento dispara, portanto nada destrava. O alarme de
estagnação do orquestrador ficou ligado por mais de 8 mil iterações.

Duas tentativas anteriores não resolveram e fazem parte do contexto:

- **10/08 — caixa fora de alcance.** O `box_xy` era 0,50 (centro da mesa), que é o valor
  que a skill Lift abandonou em 16/07 por ser fisicamente inalcançável. Corrigido para
  0,30. Isso destravou a aquisição — o `grasp` saiu de 0 para 0,25 em 3 600 iterações — mas
  não o erguer.
- **10/08 — âncora de mundo no alvo do peito.** O alvo do peito era `pelve_atual + 0,15`,
  ou seja **ele seguia o robô**. Agachar baixava o alvo, encurtava o percurso, e
  `progress = 1,0` ficava alcançável com a caixa mais baixa. O argmax era **levar o peito
  até a caixa**, e foi exatamente o que o robô aprendeu: `box_at_peito = 0,51` com
  `cond_fisica = 0,0000`. Ancorar o z no mundo matou esse argmax, mas o `lift` desabou para
  0,02 em vez de subir — e a âncora criou um problema novo, porque **não existe canal de
  altura de base na observação, nem no ator nem no crítico**. O alvo ficou parcialmente
  inobservável.

## Decisão

Seis mudanças no `pegar`, aplicadas juntas porque são interdependentes, mais um conserto de
bug que o congelamento de eixos expõe.

1. **Eixo novo `alvo`, em FRAÇÃO de uma altitude de mundo.** O eixo do `pegar` deixa de ser
   `altura` (a posição da prateleira) e passa a ser `alvo`:

   ```
   alvo_z = repouso_da_caixa + fração × (TOPO_RAMPA_Z − repouso_da_caixa)
   ```

   com cinco níveis de fração — `(0,2; 0,4; 0,6; 0,8; 1,0)`. **As duas pontas são do
   mundo.** O `repouso` é onde a caixa descansa, ou seja uma propriedade da PRATELEIRA. O
   `TOPO_RAMPA_Z` é uma **constante de 0,91 m**, lida uma vez do keyframe de pé. Nenhuma
   das duas consulta a pose do robô. Os outros três eixos — `velocidade`, `altura`, `giro`
   — congelam no nível 0 via `NIVEIS_ATIVOS`.

2. **Termo `unload`.** `preensão × clamp(1 − F_apoio / m·g) × [caixa acima do repouso]`. O
   sensor `box_support` passa a declarar `fields=("found", "force")`.

3. **Termo `sucesso_denso`.** 5,0 por segundo enquanto a condição física da tarefa vale,
   nas quatro tarefas com caixa, **fora de `TERMOS_DE_TAREFA`**.

4. **Critério de sucesso em PISO de z.** `box_z ≥ alvo_z − 0,02`, mais `preensão` e
   `de_pé`. A esfera 3D de 0,10 m sai. O `sustenta_pegar_s` cai de 5,0 s para 2,0 s.

5. **Episódio por tarefa.** 10 s na manipulação, 20 s na locomoção, via terminação própria
   com `time_out=True`.

6. **Custo de mover o braço.** `action_rate_l2` recebe fator 0,25 nos 14 canais de braço. O
   `box_shake` do `pegar` passa a cobrar só depois de a caixa chegar perto do alvo.

Além disso: o `box_at_peito` sai do gate do `pegar` e fica só no `locomover_carregando`, e
a âncora de mundo do `alvo_peito_w` volta ao frame da base. A amostragem de tarefa passa a
ser inversa à competência, com piso de 0,15. O `std` da política é reaquecido a 0,8 quando
uma tarefa abre e, sob a variável `G1_STD_RESUME`, na fronteira de bloco.

**Conserto de bug obrigatório:** a regra do evento ganha a condição
`or self.eventos_tarefa[t] == 0`.

## Justificativa

- **O alvo de erguer tem de ser do MUNDO, e o nome tem de dizer isso.** Esta é a decisão
  central, e ela é a lição do bloco 2. Um alvo que acompanha a pelve tem dois máximos: o
  robô pode subir a caixa, ou pode baixar o próprio peito. O segundo é mais barato, e é o
  que ele escolheu. Com o topo fixo em 0,91 m, agachar não move nada — a única forma de
  subir o `progress` é erguer a caixa. A constante chamava-se `_PEITO_DE_PE_Z` e foi
  renomeada para `_TOPO_RAMPA_Z`: ela não é "a altura do peito", ela é uma altitude que por
  acaso coincide com a altura do peito quando o robô está de pé. O `0,15` do
  `alvo_peito_b[2]` entra só para derivar o número em vez de digitá-lo, não para acoplar o
  alvo ao peito.

- **Faltava o eixo que gradua o que trava.** O `pegar_altura` movia a prateleira, ou seja
  graduava *de onde* pegar. Nada graduava *quanto* erguer, e é isso que trava. Com o topo
  fixo e sem rampa, o denominador do progresso era 0,26 m: erguer 5 cm dava
  `progress = 0,19` contra um limiar de 0,90. O robô tinha de erguer 23,4 cm **antes do
  primeiro evento**, e conseguia 0,4 cm. Com a rampa, o nível 0 fecha a tarefa com 5 cm.
  Baixar a prateleira aumentaria a distância a erguer — o eixo antigo empurrava na direção
  errada.

- **Fração e não centímetro.** A fração compõe com o eixo `altura` quando ele reabrir: o
  repouso desce com a prateleira, o topo continua em 0,91 m, e o percurso a vencer cresce
  — que é o correto. Um valor em metros ficaria descolado, e o nível topo pararia longe da
  altitude pretendida.

- **`_grasp` é booleano, e isso cria um platô pago.** Tocar paga 0,44 e libera os
  multiplicadores; **apertar paga zero** até a caixa se mover. A medição de 0,851 de
  preensão com 4 mm de subida é a assinatura exata desse platô: o robô mora no degrau que
  já paga. A força normal da prateleira contra a caixa é a única grandeza da cena que
  responde ao aperto de forma contínua — ela cai de `m·g` a zero **antes** de a caixa sair
  do lugar. Medido: 9,70 N apoiada, 0,00 N erguida, fração 0,011 → 1,000.

- **O horizonte do desconto ordena a lista.** `gamma = 0,99` a 50 Hz dá horizonte de
  **2,0 segundos**. Uma recompensa a 3 s de distância vale 0,22 do valor de face. Portanto
  degrau imediato atravessa o desconto e prêmio distante não. É por isso que o `unload` vem
  antes de tudo, e por isso o bônus de objetivo paga **por segundo enquanto a condição
  vale**, e não uma vez no fim.

- **O alvo não pagava nada.** O `cond_fisica` existia apenas como diagnóstico. O
  experimento rodou 22 mil iterações com ele em 0,0000 sem um único termo de recompensa
  olhando para ele.

- **O bônus tem de ficar fora do orçamento equalizado.** Dentro de `TERMOS_DE_TAREFA`, o
  orçamento do `pegar` iria de 4,5 para 9,5, o fator de equalização cairia de 0,889 para
  0,42, e o próprio bônus se diluiria para 2,1 — ele se anularia. Ele é bônus de
  **objetivo**, não sinal de aproximação. Como vale nas quatro tarefas com caixa, a
  paridade entre elas fica preservada. O `locomover` fica fora porque a condição dele é
  `ones_like`: ele coletaria 5,0/s de graça.

- **A esfera 3D reprovava o robô por fazer mais.** Com o alvo em +5 cm e a caixa a +26 cm,
  a distância dá 21 cm e o critério falha. E o robô **não observa** o nível do currículo,
  portanto não teria como parar na altura certa. O piso em z é monotônico: erguer mais nunca
  reprova. Foi essa monotonicidade que permitiu manter o alvo fora da observação em vez de
  alargá-la — alargar a observação é Categoria C, recomeçar do zero.

- **`preensão` e `de_pé` ficam no critério.** Sem `preensão`, o robô aninha a caixa no vão
  dos antebraços contra o tronco e ergue sem pegar; esse comportamento já foi medido e está
  documentado no `exige_grasp` do gate. O `_grasp` exige contato de palma e proíbe contato
  de dorso, portanto ele fecha esse caminho. Sem `de_pé`, o estado final do `pegar` deixa de
  ser o estado inicial canônico do `locomover_carregando` e do `botar`.

- **O episódio de 20 s desperdiçava amostra.** A preensão se estabelece por volta de 3 s —
  derivado de `grasp = 0,851` como média de um episódio de 20 s. Os outros 17 s eram
  repetição do mesmo estado. Com 10 s a taxa de episódios dobra e o limiar de 200 episódios
  do orquestrador enche na metade do tempo. Não 5 s: sobrariam 2 s para apertar, erguer e
  sustentar, e o `sustenta_pegar_s` não caberia.

- **O custo de mover o braço superava a tarefa.** O `action_rate_l2` cobrava 0,88 contra
  1,07 de todo o sinal de tarefa coletado — 82% — numa tarefa cujo conteúdo é mover os
  braços. E a saída da política para esse custo é encolher o `std`, o que reduz a exploração
  exatamente onde ela falta. O fator vale só nos braços: o termo precisa continuar contendo
  jitter de perna na marcha, onde ele funciona.

- **O `box_shake` cancelava o `lift`.** Ele subia junto com o `lift` no bloco 3. Erguer uma
  caixa por abraço gira a caixa: a rotação é parte da manobra, não hack. Gateado em "perto
  do alvo", erguer sai de graça e sacudir a caixa já erguida custa. É o mesmo desenho do
  `hold_still_bonus` da Lift, e pelo mesmo motivo declarado lá.

- **A âncora de mundo do peito perdeu a razão de existir.** Ela existia para matar o argmax
  agachado no `pegar`, e o `pegar` não usa mais esse termo. Sobrou o
  `locomover_carregando`, e ali a âncora de mundo é pior: a pelve oscila na marcha e não há
  canal de altura de base na observação. O frame da base é o único alvo que o robô consegue
  calcular de `box_pos_b`. O agachamento não compensa naquela tarefa porque ela tem de
  rastrear velocidade — 4,0 dos 5,0 de orçamento — e porque a terminação `largou` encerra o
  episódio se a caixa cair. **A distinção que fica: alvo de ERGUER é do mundo; alvo de
  CARREGAR é do corpo.** São objetivos diferentes e frames diferentes.

- **A amostragem uniforme desperdiçava envs.** Com três tarefas abertas, uma tarefa já
  resolvida consumia a mesma amostra que a travada. O `locomover` fechava `cond_fisica` em
  1,0 e levava um terço dos envs. O piso de 0,15 é anti-esquecimento — o mesmo papel do
  `rho` nos níveis: sem ele uma tarefa em `perf = 1,0` sairia do sorteio, a política a
  esqueceria, o `perf` cairia e a tarefa voltaria, num ciclo.

- **O congelamento de eixos expõe um bug latente, e ele é fatal em silêncio.** Com um nível
  só, o `locomover` não tem eixo a avançar e não está em `COM_DR_PESO`. Portanto a regra do
  evento caía no `continue` e ele **nunca** tinha evento. Sem o primeiro evento dele, o
  `pegar` e o `reorientar` nunca abriam: o treino rodaria só locomoção, para sempre, sem uma
  linha de erro no log. A condição nova é a semântica que o docstring do próprio campo
  `eventos_tarefa` já prometia.

## Alternativas Consideradas

### Termo `squeeze` por força de palma

`preensão × tanh(min(F_esq, F_dir)/F_ref)`, com `F_ref ≈ mg/2μ ≈ 4,9 N`. Ele preenche o
platô pela outra ponta: cobre a faixa de 0 a 5 N, antes de qualquer descarga acontecer.

**Não escolhido agora, mantido em reserva.** O `unload` cobre a faixa que importa — 5 N até
a carga inteira — e traz o anti-hack embutido. E `squeeze` sozinho é farmável: apertar a
caixa para baixo contra a prateleira gera força de palma. Os dois juntos se corrigem, porque
apertar para baixo aumenta a força de apoio e derruba o `unload`; mas a solução mínima é um
termo, não dois. Se o `unload` subir e o `lift` não seguir, este é o próximo degrau — e aí a
força de palma já será evidência medida em vez de hipótese.

### Baixar o `upright` para 0,2 no `pegar`

O `upright` paga 0,98 por passo, sem gate, contra 1,07 de todo o sinal de tarefa. É salário
por ficar em pé.

**Rejeitado.** A skill Lift tentou exatamente isso em 15/07 e reverteu: afrouxar
`upright`/`posture` para liberar o alcance degradou o treino inteiro. O aviso está escrito no
knob dela. E o conserto mínimo para "salário alto" é subir o numerador, não baixar o
denominador: o `unload` e o `sucesso_denso` levam o sinal de tarefa de 1,07 para bem acima de
5,0, e o salário passa a ser menos de 20% dele. Fica anotado como possibilidade se o
`upright` ainda dominar depois da subida.

### `table_contact` dez vezes maior

A proposta original citava 46,4 N de impacto na mesa custando 0,0376.

**Rejeitado por erro de atribuição, encontrado na verificação.** São dois termos diferentes:
o `table_contact` é booleano com peso −1,5 e custa 0,0376; os 46,4 N são medidos pelo
`soft_landing_table`, que tem peso −1e-4 e custa 0,0003. Multiplicar o termo booleano por
dez não toca no impacto. Dar autoridade ao impacto exigiria fator ~100, e essa decisão fica
para quando houver evidência de que o impacto atrapalha.

### Trocar o `std_type` de `scalar` para `log`

A intenção declarada era exploração por tarefa: abrir uma tarefa nova exige explorar.

**Rejeitado por não atender à intenção.** Existem duas classes no rsl_rl. O
`GaussianDistribution`, o atual, guarda o `std` num Parameter **global**: um valor por
junta, igual para toda tarefa. Trocar `scalar` por `log` muda só a parametrização, e ainda
renomeia o parâmetro de `std_param` para `log_std_param` — o `load` com `strict=True` quebra.
Exploração por tarefa exigiria o `HeteroscedasticGaussianDistribution`, em que a MLP emite o
`std`; ele dobra a última camada do ator, o que é Categoria C e invalida o checkpoint de 22
mil iterações.

**Escolhido em vez disso:** reaquecer o `std` para 0,8 com `clamp_(min=)`, que só sobe.
Custa nada, preserva os pesos, e ataca o problema real, que é o `std` já estar em 0,45
quando a distribuição muda.

### Segundo eixo no `pegar`, em vez de trocar

Manter `altura` e acrescentar `alvo`, serializando os dois.

**Rejeitado pelo custo de código.** O `eixo_de` faz `(nome,) = AXES[task]` — um unpack de um
elemento — e é usado em três pontos do orquestrador. Dois eixos exigiriam refazer a lógica de
célula, e o invariante "um eixo por tarefa" existe por um motivo medido: com dois eixos, a
célula de um mede marginalizada sobre o outro e o limiar trava, que foi o bug de 06/08. Como
o plano já serializa os dois, apenas um eixo fica vivo por vez — trocar entrega o mesmo
comportamento com uma linha. A altura volta a ser eixo do `pegar` num bloco futuro, quando o
`alvo` esgotar.

### Episódio de 5 s na manipulação

**Rejeitado com aritmética.** O `sustenta_pegar_s` era 5,0 s: num episódio de 5 s a condição
teria de valer desde o passo 0, e a caixa começa na prateleira. O sucesso ficaria
matematicamente impossível — o mesmo bloqueio de que estamos saindo. E a preensão só se
estabelece por volta de 3 s, o que deixaria 2 s para apertar, erguer e sustentar.

### Máquina de fases explícita

Um estado por etapa — aproximar, tocar, apertar, erguer, sustentar — com transições
gateadas.

**Rejeitado.** A cadeia já é gateada de forma multiplicativa: `reaching` → `grasp` → `lift`
(× preensão) → `box_at_peito` (× preensão). Não falta gate; faltava um degrau na escada, que
é apertar. E o princípio do projeto é explícito contra máquina de estado sem justificativa
forte. Gate multiplicativo é sem estado e já é o idioma do repositório.

### Alvo relativo ao site do torso

O robô observa as juntas de cintura, portanto a posição da caixa relativa ao torso é
calculável da observação — ao contrário da altura de mundo.

**Rejeitado, e por dois motivos.** O primeiro é conceitual: um alvo relativo ao torso
**volta a seguir o robô**, porque o torso desce com a pelve no agachamento. Ele reabriria
exatamente o argmax do bloco 2. O segundo é que se tornou desnecessário: com o alvo do
`pegar` sendo um piso em z monotônico, o robô não precisa observar o alvo, e com o
`box_at_peito` restrito ao `locomover_carregando` o frame da base já resolve.

## Consequências

### Positivas

- O primeiro sucesso do `pegar` fica alcançável: erguer 5 cm em pé, com preensão, por 2
  segundos.
- O alvo de erguer é uma altitude de mundo, e o nome e os docstrings agora declaram isso —
  o argmax "levar o peito até a caixa" fica fechado por construção, não por vigilância.
- O platô entre tocar e erguer ganha gradiente contínuo, verificado de 9,70 N a 0,00 N.
- O objetivo passa a pagar, e paga dentro do horizonte de desconto de 2 segundos.
- A fração mantém o eixo correto quando o eixo de altura reabrir.
- A taxa de episódios da manipulação dobra, e o limiar de 200 episódios enche na metade do
  tempo.
- A observabilidade do alvo deixa de ser um problema, sem alargar a observação.
- O bug que faria o treino rodar só locomoção em silêncio está fechado, com teste.
- O currículo cai de 24 para 12 destravamentos, o que concentra o bloco em abrir as cinco
  tarefas.

### Negativas

- **O checkpoint carrega, mas a função de valor não.** A escala de recompensa do `pegar`
  muda muito, e o crítico precisa reaprender. Mitigação: `learning_rate = 5e-4` no começo do
  bloco e reaquecimento do `std`.
- **O congelamento de três eixos é dívida declarada.** `velocidade`, `altura` e `giro` param
  no nível mais fácil. O robô não vai aprender a andar a 2 m/s, nem a pegar do chão, nem a
  girar 360° neste bloco. Descongelar é mudar um número em `NIVEIS_ATIVOS`, mas exige um
  bloco a mais.
- **O `unload` depende de um sensor a mais.** O campo `force` no `box_support` acrescenta
  custo, pequeno mas real, a todos os envs, inclusive nas tarefas que não o usam.
- **O bônus de 5,0/s quebra a paridade nominal com o `locomover`.** As quatro tarefas com
  caixa passam a poder coletar mais por passo que a de locomoção. A justificativa é que o
  sinal denso do `locomover` são os dois `track_*`, que ele já coleta; mas a simetria
  perfeita do orçamento deixa de existir.
- **O `sucesso_denso` é um degrau, não uma rampa.** Ele paga quando a condição vale e nada
  antes. O gradiente que leva até lá vem do `unload`, do `lift` e do `reaching`. Se algum
  desses falhar, o bônus não ajuda a encontrá-lo.
- **O piso em z aceita um erguer torto.** O critério mede altura e preensão; ele não mede a
  qualidade da pega. Uma pega de quina que erga 5 cm passa. As pressões que devem corrigir
  isso — `box_shake`, a DR de atrito, a DR de carga de 1 a 5 kg — são indiretas.
- **Duas convenções de frame convivem.** O alvo de erguer é do mundo, o alvo de carregar é
  do corpo. A razão é sólida, mas quem for mexer nessa área tem de saber qual é qual — e a
  confusão entre as duas já custou um bloco.

### Neutras

- O `sustenta_pegar_s` cai de 5,0 s para 2,0 s. O critério fica mais fácil nessa dimensão,
  em troca de ser possível.
- A `tol_w` continua em 0,70, com o `erro_vel_ang_filt` como vigia. Essa decisão é anterior e
  não muda aqui.
- O `G1_STD_RESUME` é variável de ambiente, não configuração. Ele tem de ser lembrado a cada
  bloco em que se quer o reaquecimento.
- O `squeeze` e a redução do `upright` ficam registrados como próximos passos condicionais,
  não como pendências.

## Referências

- `EXPERIMENTO.md` — §3 (as cinco tarefas), §8 (critério de sucesso), §9 (grafo do
  currículo), §14 (tabela de constantes)
- `g1_multitask/rewards.py` — `_TOPO_RAMPA_Z` (a distinção mundo × robô), `alvo_z_pegar`,
  `lift_altura`, `unload`, `box_shake_pegar`, `ActionRateJuntas`, `alvo_peito_w`
- `g1_multitask/tasks.py` — `LEVELS["alvo"]`, `NIVEIS_ATIVOS`, `AXES`, `TERMOS_DE_TAREFA`
- `g1_multitask/metrics.py` — `condicao_tarefa`, `sucesso_denso`
- `g1_multitask/curriculum.py` — `_dist_tarefas`, a regra do evento e o conserto
  `eventos_tarefa[t] == 0`
- `g1_multitask/terminations.py` — `time_out_por_tarefa`
- `g1_multitask/runner.py` — `STD_AO_ABRIR_TAREFA`, `STD_NO_RESUME_ENV`
- `g1_training/skills/lift/knobs.py:103` — o aviso de 15/07 sobre baixar o `upright`
- `g1_training/skills/lift/configs/c2026_07_16_box_edge.py` — a investigação que estabeleceu
  0,50 como inalcançável
- Commits: `8ca64a7` (as seis mudanças e o conserto do bug), `0943b56` e `af4b669` (o
  reaquecimento do `std`), `977823d` (a renomeação de `_TOPO_RAMPA_Z`), `4652b40` (o portão
  de resume)
