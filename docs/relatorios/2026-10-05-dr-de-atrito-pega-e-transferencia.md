# DR de atrito: ela resolve a pega? Ela transfere para o robô real?

Análise de 05/10, a pedido do dono, antes de mudar o treino. Pergunta do dono: "Preciso que
o robô aprenda a não apertar a caixa e também não deixe cair. Não quero corrigir um problema
que ocorre no simulador que não me ajuda em nada quando eu for para o real."

Fontes: medidas com o `model_24350` na cópia do registrador (`scratchpad/registra_mu.py`),
o código em `g1_limpo/`, e a memória. Tudo que é literatura ou valor do robô real está
marcado como NÃO VERIFICADO.

## 0. Resposta curta

1. **A DR de atrito sozinha não ensina a "não apertar".** Ela ensina só a "não deixar cair",
   e o jeito mais barato de não deixar cair é apertar mais. Hoje nada cobra o aperto. Com a
   DR sem preço do excesso, o robô vai apertar MAIS do que hoje, não menos.
2. **O aperto de hoje é o limite do motor.** O `shoulder_yaw` dos dois braços trabalha em
   ±24 a 27 N·m, contra o limite de 25. O `wrist_yaw` fica em ±5, o limite dele. O robô não
   escolheu uma força: ele empurra o alvo do PD para dentro da caixa até o motor saturar.
   Esse comportamento transfere para o real como "esmagar com toda a força do braço".
3. **Para aprender a não apertar e não deixar cair ao mesmo tempo, faltam duas coisas:**
   um preço para a força acima do necessário, e uma razão para o necessário variar (DR de
   atrito e de massa). Uma sem a outra não funciona: só o preço leva a soltar; só a DR leva a
   esmagar.
4. **O que mais ameaça a transferência não é o μ.** É, nesta ordem: a caixa rígida do
   simulador (a de papelão cede), o pulso do G1 com μ 1,0 no simulador (no real é plástico
   duro), e a pose exata da caixa que o ator vê.
5. **Antes de treinar, medir o μ real** do pad e do pulso do G1 contra papelão, com um plano
   inclinado. É meia hora de trabalho, e decide a faixa da DR.

## 1. Como o robô segura a caixa hoje (MEDIDO)

Roteiro `carregar`, laje 0,45, 1 kg, μ 1,0, impratio 2. Mediana no CARREGAR.

| Contato na caixa | Força lateral | Força vertical |
|---|---|---|
| `right_wrist_collision` (cápsula do `wrist_pitch_link`) | 73 N | +11 N |
| `left_wrist_collision` | 64 N | −9 N |
| `torso_collision` | 31 N | +5 N |
| `right_palm_pad` | 36 N | +3 N |
| `left_palm_pad` | 46 N | 0 N |
| **Total robô → caixa** | **~240 N** | 9,8 N (= peso) |

Torque do PD estimado por `kp·(alvo − q)` nas juntas do braço, mediana no CARREGAR:

| Junta | 1 kg, μ 1,0 | 5 kg, μ 1,0 | Limite |
|---|---|---|---|
| `shoulder_yaw` E / D | −21,9 / +23,9 | −26,7 / +26,4 | ±25 |
| `wrist_yaw` E / D | −5,0 / +4,6 | −4,0 / +3,0 | ±5 |
| `shoulder_pitch` E / D | −10,8 / −5,8 | −13,9 / −10,9 | ±25 |
| `elbow` E / D | −6,7 / −0,8 | −10,0 / −6,0 | ±25 |

Leitura:
- O aperto vem do `shoulder_yaw` no limite. A força total não muda com a massa nem com o μ
  (230–250 N em todos os 15 casos medidos), porque o motor já está no teto.
- O `squeeze` só vê os pads (36–46 N) e satura em `F/F_ref ≥ 3`. Para 1 kg, `F_ref` = 6 N.
  O termo vale 1,0000 e tem derivada zero. Ele não paga mais por apertar mais, e nada cobra.
- Não existe termo de torque nem de energia no conjunto (`knobs.py:451-466`: `action_rate_l2`
  −0,1, `joint_acc` −2,5e−7). Um aperto estático custa zero no `action_rate`, porque a ação
  não muda.
- Conclusão: apertar no máximo é grátis e é a estratégia mais segura contra cair. A política
  achou essa estratégia, e qualquer DR que só aumente o risco de cair a reforça.

## 2. A DR de atrito com `priority` (MEDIDO)

A caixa recebe prioridade acima do robô, e a mesa acima da caixa. Todo contato robô–caixa
usa o μ da caixa (conferido em `contact.friction`).

| μ da caixa | 1 kg | 3 kg | 5 kg |
|---|---|---|---|
| 1,0 | segura | segura | segura |
| 0,6 | segura | segura | segura |
| 0,45 | segura | segura | escorrega em 11,5 s |
| 0,3 | segura | escorrega em 7,0 s | escorrega na subida, 2,8 s |

- Com μ ≥ 0,6 a política de hoje não perde a caixa. Nessa faixa a DR não quebra o resume e
  também não cria gradiente nenhum.
- Com μ ≤ 0,45 e caixa pesada a caixa desce devagar entre as mãos (creep do contato mole),
  os pads perdem a face, e a caixa sai. O dono viu isso no viewer.
- O impratio 2 do treino atrasa o escorrego (5 kg, μ 0,45: 11,5 s contra 6,7 s com
  impratio 1) e não o evita.

### 2.1 O que a política pode aprender com a DR, dado o que ela observa

O ator não observa μ. Ele observa a pose exata da caixa no frame da base (10 canais, sem
ruído, `observacoes.caixa_no_frame_da_base`), as juntas, a ação anterior e a IMU. Com isso,
duas estratégias são aprendíveis:

| Estratégia | O que exige | Custo hoje | O que o PPO tende a achar |
|---|---|---|---|
| A. Conservadora: apertar sempre para o pior μ da faixa | nada | zero (sem preço do excesso) | esta, porque é a de hoje com μ 1,0 |
| B. Reativa: ver a caixa descer e apertar mais | ver o deslize a tempo e reagir em < 0,5 s | reação atrasada = episódio perdido | só se A for cara |

A estratégia B é a que o dono descreveu ("escorrega → aperta mais"). Ela só aparece se a
A custar caro. Isso é o preço do excesso. Sem ele, a DR produz a A, com mais força que hoje.

Observação: o ator já está no limite do motor. Com μ 0,3 e 5 kg, a conta `m·g/(2μ)` pede
82 N por palma e o braço não entrega mais que hoje. Esses envs não têm solução motora e viram
ruído na recompensa. A faixa da DR tem de respeitar o que o braço consegue: com 25 N·m no
`shoulder_yaw` e os pads como hoje, a borda de baixo útil fica perto de μ 0,45 para 5 kg.

## 3. O que ensina a "não apertar"

Precisa de um custo que cresça com a força acima do necessário. Três formas, da mais
transferível à menos:

| Forma | Como | Transfere? | Risco no treino |
|---|---|---|---|
| a. Custo de torque nos braços | `Σ τ²` nas 14 juntas do braço, peso pequeno | sim: o real mede corrente do motor, e esse custo é prática padrão de sim2real | taxa também o erguer; precisa gatear por elo ou aceitar |
| b. Dobradiça da força total na caixa | `relu(F_tot/(k·F_ref) − 1)²`, k = 2–3, com `F_ref` da massa e do μ do env | a recompensa pode ser privilegiada; o ator só vê o efeito | o ator controla a força pela posição do alvo do PD, e isso é o item 4.1 |
| c. Segunda parcela no `squeeze`, só nos pads | o plano de 21/09 | igual a b | NÃO basta: 110 N por lado ficam no pulso e no tronco, fora da régua |

Recomendação: **b sobre a força total** (todos os contatos do robô na caixa), e não c.
A forma b é o plano de 21/09 (`docs/planos/2026-09-21-cuidado-com-a-caixa-impacto-e-aperto.md`)
com a régua corrigida. A forma a pode entrar junto, com peso pequeno, como proxy transferível.

Com b e a DR juntas, o ponto de operação é `F ≈ k·m·g/(2·μ_min)` da faixa corrente. Ele é
conservador e limitado, e é isso que o dono pediu: não cair e não esmagar.

Detalhe que evita uma briga entre os dois termos: cobrar o excesso em relação ao `F_ref` do
**pior μ da faixa do nível**, e não ao μ sorteado do env. Se cobrar pelo μ do env, a
estratégia conservadora paga nos envs de μ alto e o PPO oscila entre apertar e soltar.
Cobrando pelo pior μ da faixa, apertar para o pior caso não custa nada, e só o que passa
disso custa.

## 4. O que ameaça a transferência (ordem de tamanho)

### 4.1 A caixa rígida (MAIOR)

No simulador a caixa é um corpo rígido com contato mole (`solref` 0,02 s). O robô produz a
força empurrando o alvo do PD para dentro da caixa: `F ≈ kp·(alvo − q)` até o limite do
motor. A caixa de papelão cede milímetros a centímetros sob 50–200 N. Com a mesma ação, a
junta anda mais, e a força sai menor que no simulador. Direção do erro: o real aperta MENOS
que o simulador para a mesma política. Isso aumenta o risco de cair, e não o de esmagar.

Hoje isso não aparece porque a política satura o motor: 25 N·m é 25 N·m em qualquer caixa.
No momento em que a política passar a dosar a força, a rigidez da caixa vira o maior erro
de transferência. Mitigação: DR da rigidez do contato da caixa (`solref`), e ruído no canal
`meia_aresta` (já previsto no estimador da v3). Um erro de 1 cm no tamanho da caixa é um
erro de 1 cm na penetração, e vira erro de força.

### 4.2 O pulso com μ 1,0 (erro de realismo, independente da DR)

No simulador a cápsula do `wrist_pitch_link` tem μ 1,0 e segura a caixa mais que o pad. No
G1 real essa superfície é plástico ou alumínio: contra papelão, μ na ordem de 0,3 a 0,4
(NÃO VERIFICADO; medir). A política de hoje confia num contato que no real escorrega.

Conserto: dar ao pulso o μ realista e deixar o pad com o μ do material que o robô real vai
ter (borracha: 0,7 a 0,9, NÃO VERIFICADO). Assim quem segura é o pad, e a régua do `squeeze`
passa a medir o que segura. Isso é mais transferível que a DR e custa uma linha.

### 4.3 A pose exata da caixa no ator

A estratégia reativa (B) depende de ver a caixa descer 1–2 cm. No simulador o ator vê isso
sem ruído e sem atraso. No real a caixa no peito está no campo da D435i (eixo 48° abaixo do
horizonte, FOV vertical 58°; caixa a ~0,4 m e ~37° abaixo), mas com oclusão pelos braços e
com o erro do estimador. A decisão da v3 (02/10) já é treinar com o estimador simulado. A
consequência aqui: a B é um bônus, não o plano. O plano é a A com teto (seção 3).

A massa, ao contrário do μ, é observável pela propriocepção: o torque para erguer e
sustentar a caixa aparece em `alvo − q` das juntas do ombro. "Caixa pesada → aperta mais" é
aprendível e transfere. "Escorrega → aperta mais" depende da visão.

### 4.4 O modelo de atrito do MuJoCo

- Um μ só, sem estático e cinético. No real, o estático é maior, a caixa segura firme até
  o limiar e depois escorrega rápido. No simulador ela desce devagar (creep). Uma política
  pode aprender a conviver com o creep (reajustar a pega aos poucos), e isso não transfere.
  Mitigação: impratio 2 já está; `noslip_iterations` está em 0 e pode subir (custo de
  solver; verificar se o MuJoCo Warp o implementa).
- Cone piramidal (o elíptico divergiu para NaN duas vezes). Em cone piramidal a força de
  atrito disponível depende da direção, e o deslize nas diagonais sai 41% maior.
- Atuador: o limite de torque do simulador (25 e 5 N·m) bate com o do G1 (NÃO VERIFICADO na
  ficha da Unitree; a memória registra ±5 no punho).

### 4.5 O que NÃO é problema

- A geometria do abraço (pads nas faces, pulso e tronco apoiando) é como uma pessoa carrega
  uma caixa. Ela transfere.
- A conta `F_ref = m·g/(2μ)` é física; ela vale no real.

## 5. Riscos para o treino

| Risco | Onde aparece | Como evitar |
|---|---|---|
| Preço do excesso derruba a pega | o robô solta para não pagar | `terminacao` −200 e `caixa_largada` já custam mais; começar com peso pequeno; k = 3 |
| DR larga de uma vez | a caixa pesada com μ baixo cai antes de ensinar | faixa [0,6; 1,0] e borda de baixo descendo por nível até 0,45; subir de volta se o sucesso cair |
| Envs sem solução motora | μ 0,3 com 5 kg pede 82 N por palma | não sortear abaixo do que o braço entrega; ou cortar a massa quando o μ é baixo |
| Dois termos brigando | preço pelo μ do env × aperto pelo pior μ | cobrar pelo pior μ da faixa (seção 3) |
| Treino aprende a explorar o creep | reajuste contínuo da pega | medir a taxa de descida com μ 0,6; se for material, `noslip` ou impratio maior |

## 6. O que fazer, em ordem

1. **μ dos materiais. DECISÃO DO DONO (05/10): opção A.** A palma do G1 fica como está
   (plástico duro × papelão, 0,3–0,5 por tabela, §7.5), e a massa máxima continua 5 kg. Com
   μ 0,3 a caixa de 5 kg pede 82 N por palma, e o braço entrega ~100 N no limite: o aperto
   lateral não basta, e o robô tem de aprender outro jeito de segurar (caixa inclinada contra
   o peito, apoio no antebraço). Consequências para a recompensa da v3:
   - pagar o RESULTADO da pega (a caixa fica com o robô e acompanha as mãos), e não a força
     normal nos pads; o `squeeze` de hoje (`_forca_das_palmas`, só normal ao pad) vale zero
     para qualquer pega que não seja lateral;
   - exigir a caixa de pé só no BOTAR; no CARREGAR a orientação de pé (×4 hoje) pune a
     caixa inclinada contra o peito;
   - alvo da caixa mais livre em altura e inclinação no CARREGAR;
   - ordem: recompensa do resultado primeiro, DR de μ baixo depois, senão os envs de 5 kg com
     μ 0,3 não têm solução e viram ruído.
   Memória: `g1-v3-palma-de-plastico-5kg`.
2. **Realismo primeiro:** μ do pulso como plástico; pad com o μ do material escolhido no item 1.
3. **Preço do excesso** (freio de aperto) sobre a força total robô → caixa, dobradiça acima
   de `k·F_ref`, k = 3, `F_ref` pelo pior μ da faixa do nível. Métrica da força total no log.
   Precedente: teto de força do HDMI em G1 real (§7.1).
4. **DR de atrito por `priority` da caixa**, faixa [0,6; 1,0], borda de baixo por nível até
   ~0,45. A `priority` é fixa na cena (`CollisionCfg.priority`; o campo não tem dimensão de
   mundo) e o μ por env sai do termo pronto `dr.geom_friction`. Memória `mjlab-warp-contato-e-dr`.
5. **DR da rigidez do contato da caixa** (`geom_solref` por env; sem termo pronto, pede evento
   próprio como o `carga_caixa`) e ruído no `meia_aresta`, junto com o estimador da v3. Faixa a
   medir no registrador antes: o valor em que 5 kg com μ 0,6 ainda fica parado. O creep não tem
   `noslip` no Warp; só impratio e `solref`.
6. Só depois, se sobrar: custo de torque nos braços como proxy transferível (1e−7 a 1e−4 nos
   papers, §7.1).

Prévia no resume: itens 3 e 4 juntos, com peso pequeno no 3. Só o item 4 reforça o esmagar.

## 7. O que a literatura faz (agente Sonnet, 05/10)

Aviso de método: o agente leu os papers pelo HTML do arXiv com um resumidor. Confirmar
cada número no paper antes de copiar.

### 7.1 Humanoide bimanual com caixa: como dosam o aperto

Ninguém dosa o aperto de forma explícita sem tátil. Três mecanismos aparecem:

| Mecanismo | Fonte | Detalhe |
|---|---|---|
| Teto de força na recompensa de contato | [HDMI](https://arxiv.org/abs/2509.16757) (G1 real) | `exp(−‖p_eef − p_alvo‖/σ_pos) · max(exp((‖F‖ − F_lim)/σ_frc), 1)`; o limite existe "para segurança no deploy"; peso do contato 5,0, torque L1 0,01 |
| Limite de força no alvo do PD (mola virtual) | [GentleHumanoid](https://arxiv.org/abs/2511.04679) (G1, sem sensor de força) | `F = Kp·(x_alvo − x)`; se passa do limite, escala a força; 5 N mínimo, 15 N máximo; abraço a ~10 N contra > 20 N no RL comum; balão a 5 N sem estourar, os dois baselines estouraram |
| Penalidade de torque pequena | todos | L2 de torque entre 1e−7 ([Embracing Bulky](https://arxiv.org/abs/2509.13534)) e 1e−4 ([PhysHSI](https://arxiv.org/abs/2510.11072)); excesso sobre o limite −0,5 e −500 ([WoCoCo](https://arxiv.org/abs/2406.06005)) |

Outros dados úteis:
- [Digit box (Dao, 2023)](https://arxiv.org/abs/2310.03191): termo de força mão–caixa com peso 0,05 (sinal não confirmado); o sim2real falhou por viés de orientação e caixa de 8 kg; corrigiram com ajuste manual do CoM.
- [ResMimic](https://arxiv.org/abs/2510.05070): G1 real com 4,5 kg (limite do punho ~2,5 kg, o corpo encosta); a recompensa de contato paga força maior sem teto.
- [PhysHSI](https://arxiv.org/abs/2510.11072): G1 real, 0,6 a 3,6 kg, mão de borracha; acima de 3,6 kg falha; falhas por superaquecimento do motor e deslize da caixa pesada.
- [SplitAdapter](https://arxiv.org/abs/2606.03297): G1 real com luvas de alto atrito, 0,5 a 5,0 kg; adaptação à carga: 26/27 contra 16/27.
- [Lin et al. 2025](https://arxiv.org/abs/2502.20396): GR-1, caixa bimanual; ajusta o sim ao real em menos de 4 min.

**Leitura para o nosso caso:** o teto de força do HDMI é o freio de aperto da seção 3, com
precedente publicado em G1 real. A mola virtual do GentleHumanoid é a mesma ideia aplicada
na ação, e limita `Kp·Δx` antes de saturar os 25 N·m. Todos os trabalhos reais com G1 usam
borracha ou luva de alto atrito na mão.

### 7.2 Deslize sem tátil nem F/T

- Em humanoide com braços, em tempo real: **nada publicado**. É lacuna.
- [RMA em mão (Qi et al., 2022)](https://arxiv.org/abs/2210.04887): adapta atrito e massa pelo
  histórico de propriocepção, sem detector de deslize. Em mão com dedos; não testado em abraço.
- [Blind Dexterity (2026)](https://arxiv.org/abs/2608.29487): G1 só com encoders; o estado do
  objeto fica legível no histórico curto quando o contato é ativo. Não trata deslize.
- Visão: optical flow com RealSense D435 mede deslize na garra, acurácia até 82%
  ([Applied Sciences 2023](https://www.mdpi.com/2076-3417/13/15/8620)); caixa lisa é difícil.
- Com F/T no punho ou mão SEA existe, e o G1 não tem nenhum dos dois.

Conclusão: o "escorrega → aperta mais" não tem precedente sem tátil. O "pesada → aperta mais"
tem, pelo histórico de propriocepção (RMA). Confirma a seção 4.3.

### 7.3 Faixas de DR nos papers

| Fonte | Atrito | Outras faixas |
|---|---|---|
| [Dactyl](https://arxiv.org/abs/1808.00177) | ×0,7 a ×1,3 | massa ×0,5–1,5; ganho P ×0,75–1,5 (log); dimensão ×0,95–1,05 |
| [Rubik / ADR](https://arxiv.org/abs/1910.07113) | faixa cresce sozinha | ADR 32,0±6,4 sucessos contra 2,7±1,1 com DR manual (confirmar) |
| WoCoCo (H1) | 0,2–1,1 | massa de elo ×0,7–1,3; PD ×0,75–1,25; atraso 0–20 ms |
| PhysHSI (G1) | caixa 0,5–1,2 | massa 0,6–3,6 kg; restituição 0–0,2 |
| SplitAdapter (G1) | caixa 0,5–1,0; plataforma 0,5–2,0 | massa 0,5–5,0 kg |
| Lin et al. (GR-1) | 0,5–1,5 | forma ±5% |
| Qi et al. (mão) | 0,3–3,0 | escala 0,70–0,86; massa 0,01–0,25 kg |
| Digit | chão −30% a +20% | caixa ±5 cm, ±0,5 kg |

- `solref`/`solimp`: quase ninguém randomiza; o robosuite expõe sem faixa; nenhuma faixa
  publicada para pega de caixa. A faixa da seção 6 (item 5) tem de ser medida aqui.
- Ninguém randomiza a profundidade do alvo do PD dentro do objeto, que é a variável que gera
  a força. Lacuna.

### 7.4 Caixa rígida no sim contra papelão

Nenhum paper mede esse gap em humanoide. PhysHSI e SplitAdapter treinam rígido e compensam com
atrito alto na mão e DR de atrito e massa. Deformáveis em geral:
[Ikemura et al. 2025](https://arxiv.org/abs/2510.25405) (currículo rígido → deformável, com
custo de estresse; tofu com garra). Leitura do agente: papelão fino com 1 a 5 kg deforma
pouco no aperto lateral; o risco é a caixa que amassa e perde a face plana. Hipótese.

### 7.5 Atrito de materiais

- Borracha × papelão: estático 0,5 a 0,8, seco e limpo
  ([Engineering Toolbox](https://engineeringtoolbox.com/friction-coefficients-d_778.html), tabela
  genérica sem método).
- Papelão × alumínio em folha: 0,43 (resultado de busca, sem fonte primária).
- Papelão × ABS, × PA, × chapa de alumínio: **sem fonte primária**.
- Normas: TAPPI T 815 e ASTM D4521 medem pelo plano inclinado (μ = tan do ângulo).

## 8. Não verificado e como verificar

| Afirmação | Como verificar |
|---|---|
| μ real pad/pulso × papelão | plano inclinado com o G1 (sem acesso ao robô hoje: usar faixa 0,3–0,8) |
| rigidez da caixa de papelão sob a palma | apertar a caixa com dinamômetro e régua |
| limite de torque do G1 bate com 25 / 5 N·m | ficha técnica Unitree |
| `noslip_iterations` no MuJoCo Warp | VERIFICADO 05/10: não existe (`NotImplementedError`); memória `mjlab-warp-contato-e-dr` |
| números dos papers da seção 7 | confirmar no PDF antes de copiar |
