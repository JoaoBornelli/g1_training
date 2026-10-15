# Viabilidade da v3: o resultado do treino contra a tarefa

Data: 2026-10-15. Plano julgado: `docs/planos/2026-10-14-implementacao-g1-v3.md` (commit `629adfe`).

Pergunta do dono: "meu objetivo não é só verificar se o plano tem furos de implementação, mas sim
verificar se o resultado que esse treinamento gerará é viável considerando a tarefa que eu quero que
o robo faça". O dono também pediu a dificuldade e a chance de um sim-to-real.

Método: o workflow `wf_a59ac6e6-143` usou 17 agentes Opus, só de leitura. Nenhum agente rodou
simulador. As etapas foram estas:

- Um agente extraiu o envelope da tarefa, com 50 requisitos (anexo A).
- Sete frentes previram o resultado e o compararam com o envelope.
- Um verificador adversarial por frente tentou refutar cada achado.
- Uma síntese integrou as frentes, e um crítico procurou as lacunas.

Os analistas propuseram 39 achados. Os verificadores refutaram 1 e acrescentaram 21. Conferi na
fonte, eu mesmo, as 12 afirmações que decidem o veredito (§9).

Regra de leitura: MEDIDO quer dizer log, CSV ou relatório com número. O resto é PREVISÃO dos
agentes, ancorada no histórico do C3.

## 0. Veredito

O treino, como está desenhado, não entrega a tarefa inteira. Ele entrega um núcleo.

- **No sim, o núcleo deve formar:** andar no plano até ~0,5 m/s, e pegar, carregar e botar cubos de
  1 a 3 kg em lajes de 0,30 a 0,55 m.
- **O envelope inteiro depende do currículo:** 5 kg, laje de 0,04 m, caixa de 14 cm, terreno e
  1,5 m/s só entram se o currículo único abrir os últimos níveis em 30 000 its. Três mecanismos do
  desenho atual travam ou inflam essa abertura (§2.1, itens 1, 2 e 4). O conserto dos três é barato.
- **No G1 real, o núcleo leve tem 30 a 70% de chance por cadeia, com quatro pré-condições:**
  - o G1 EDU no modo 5, com a cintura livre;
  - um efetuador igual ao pad do sim;
  - uma fonte real do estado da caixa (AprilTag no primeiro teste);
  - o LIO e o filtro do mapa rodando no robô.
  Sem qualquer uma delas, o resultado real é zero.
- **O ciclo do produto (enunciado §1) não fecha nem no sim, porque o passo 6 (REORIENTAR) fica fora
  da v3.** As emendas do ciclo não pedem treino próprio: o 24800 fez os ciclos sem elas (§2.1,
  item 5; dono, 15/10). Sem o passo 6 e com um controlador roteirizado, a previsão é esta:
  - no sim: 0,5 a 0,8 por ciclo com caixa fácil e 0,2 a 0,5 na média do envelope;
  - no real, com caixa leve e tag: 5 a 35% por ciclo.
- **Dificuldade:** 4/5 para a v3 como está escrita e 5/5 para a tarefa do dono.
- **Sim-to-real:** 2/5.

O resultado do treino tende a ser bimodal. Ou o currículo abre os últimos níveis, ou ele para no
meio. A previsão de 85 a 95% por cenário vale só no primeiro caso. Sem essa condição, o sucesso nos
cenários pesados fica em ~0,35 a 0,65 (conta do crítico).

| frente | veredito do verificador | dificuldade | sim-to-real |
|---|---|---|---|
| Física do G1 real | viável com ressalvas (o analista disse inviável) | 4 | 2 |
| Locomoção prevista | viável com ressalvas | 3 | 3 |
| Manipulação prevista | viável com ressalvas (análise otimista) | 4 | 2 |
| Confiabilidade do ciclo | inviável como desenhado, como produto | 5 | 2 |
| Sim-to-real: dinâmica | viável com ressalvas | 4 | 2 |
| Sim-to-real: o que a política recebe | inviável sem marcador; roda com tag | 4 | 2 |
| Estado da arte | incerto (análise otimista) | 5 | 2 |

## 1. O robô previsto ao fim dos 30 000 its

| tarefa | no sim (PREVISÃO, salvo MEDIDO) | no G1 real, com as quatro pré-condições |
|---|---|---|
| ANDAR | Segue 0,25 a 0,5 m/s no plano, com razão de 0,85 a 1,0. O nível de velocidade trava em 0,5 a 0,75 m/s, e o terreno quase não abre (§2.1, item 1). Parado, a pelve fica em 0,75 a 0,79 m. Com vx 0,10 pedido, anda a 0,06–0,12 m/s. | No plano, até 0,5 m/s, funciona no primeiro deploy com 50 a 80% de chance. |
| CARREGAR | Anda a 0,35–0,50 m/s para 0,5 pedidos. O C3 MEDIU 0,30 m/s de base. O passo fica miúdo: 0,05 a 0,13 m, a 4,5–6,5 passos/s. Para frear, leva 2 a 5 passos e anda 0,15 a 0,50 m. No giro de 180°, fecha 150 a 185°, e a base desloca 0,08 a 0,35 m. | Degrada. O ombro e o punho aquecem no transporte. A chegada lenta à mesa fica fora da distribuição do treino. |
| PEGAR | Aperta 150 a 260 N com qualquer massa. De 58 a 82% dessa força passa pela cápsula do pulso e pelo tronco, e não pelos pads (C3 MEDIDO). Fecha 85 a 95% nas lajes de 0,45 a 0,55 m. Na laje de 0,30 m com 4 a 5 kg, fecha 70 a 90%, com a cintura a 95–99%. Nas lajes de 0,04 a 0,15 m com 4 a 5 kg, fecha 50 a 80%. Com 5 kg, a caixa inclina 8 a 14° no p95. | 1 a 2 kg: 50 a 80%. 3 kg: 30 a 60%. 4 a 5 kg sem luva: menos de 20%. Com luva de μ ≥ 0,8, e esse μ posto no sim, 4 a 5 kg dão 40 a 70% em laje alta. Na laje de até 0,30 m com 4 a 5 kg, 0 a 20%, se a cintura real tiver 35 N·m. |
| BOTAR | Pousa a 0,1–0,3 m/s e a 2–6 cm do alvo. Solta e fica de pé em 70 a 90% das rodadas nas lajes altas, e em 50 a 75% nas lajes baixas com caixa pesada. O aceite da laje de 0,04 m com 5 kg, em 9 de 10, tem 30 a 50% de chance. | Sem marcador no destino, a caixa pousa a 0,1–0,2 m do alvo. A caixa arrasta na mesa ao soltar, porque o sim fixa o μ da laje em 1,0. |

Falhas típicas no real: o tronco cede na pega baixa; a caixa escorrega; o ombro e o punho aquecem;
a caixa arrasta ao soltar; a chegada lenta com a caixa sai da distribuição do treino.

## 2. Os mecanismos que mudam o resultado

Cada item diz o que acontece, a fonte e o conserto. Os ids levam ao anexo B.

### 2.1 Desenho do treino: mudam o resultado no sim e têm conserto barato

1. **O portão do parado trava o ANDAR (loco-3, loco-v3).**
   - O PD-15 mede o sucesso do canal zero pela deriva da base no segmento: menos de 0,10 m
     (plano:267).
   - O nível do ANDAR só sobe com todas as células no portão (PD-12, plano:264). O terreno só abre
     depois do último nível de velocidade (plano:578-582).
   - MEDIDO no 24800: frear de 0,5 m/s anda 0,126 m; de 1,0 m/s, 0,354 m; de 1,5 m/s, 0,623 m
     (`lidar_sim/cenarios/curva_model_24800.json`).
   - Consequência: a célula do parado reprova perto de 0,75 m/s. O terreno e o refino do ANDAR
     quase não abrem, e a DR de atuação do refino não chega ao ANDAR.
   - Conserto: medir a velocidade média no último 1 s do segmento e descontar 1,5 s depois de cada
     troca. O PD-15 está na sua lista de revisão.
2. **O termo do passo não impõe passo mínimo (loco-1).**
   - O termo paga `min(L/L_alvo, 1)` a cada toque (specs/g1-v3.md:1072-1077). Dois meios passos
     pagam o mesmo que um passo inteiro.
   - MEDIDO pelo crítico no `24800_mu10_m5`: com a caixa, 6,6 passos/s a 0,30 m/s, ou ~0,045 m
     por passo.
   - Conserto: guardar o valor do último toque e pagá-lo a cada passo de controle, vezes o
     rastreio. Muda a forma do termo; nenhum termo novo entra.
3. **Comando baixo e giro no lugar sem gradiente (loco-2, loco-v1, loco-v2).**
   - Com a caixa, o treino nunca sorteia comando abaixo de 0,25 m/s nem de 0,25 rad/s. A chegada
     à mesa passa por essa faixa.
   - MEDIDO no 24800: com vx de 0,3 m/s ou menos, a base não anda; com wz de 0,4 rad/s ou menos,
     a base quase não gira (mesmo JSON).
   - Conserto: mudar a forma do ramo `c < 0,02` do rastreio; sortear wz de 0,02 a 0,25 rad/s;
     dar comando baixo também ao CARREGAR.
4. **O sucesso do currículo conta a caixa caída (confiabilidade-2).**
   - Com a chave `CAIXA_PERDIDA_VIRA_ANDAR`, a perda não termina o episódio (plano:1469-1470). O
     sucesso do PEGAR é "P4 fechado e trecho seguinte sem terminação" (plano:1497-1498).
   - Assim, a caixa no chão durante o CARREGAR conta como sucesso do PEGAR, contra o enunciado
     :84-86. O nível sobe com a caixa no chão, e o log superestima o PEGAR em mais de 10 p.p.
   - Conserto: acrescentar "e não `v3_perdeu`" a toda regra de sucesso.
5. **Emendas do ciclo: REBAIXADO pelo dono (15/10), com medida.**
   - A validação previu que as emendas não treinadas derrubam o ciclo (loco-4, confiabilidade-4,
     sota-v1). As emendas são frear com a caixa, trocar o comando no CARREGAR e chegar andando ao
     BOTAR.
   - O dono testou isso com o 24800 no autônomo da fase 1. MEDIDO
     (`lidar_sim/runs/autonomo/*/resumo.json`): com o controlador ajustado, o 24800 fecha 10 de 10
     corridas datadas. As corridas H e I fecham 3 tarefas seguidas, com 1, 3 e 5 kg, do pallet e
     da mesa à mesa, com 18 a 21 s de transporte. As 2 falhas (L e h) são quedas no PEGAR do
     pallet baixo com a caixa de 16 cm: é a habilidade isolada, e não a emenda.
   - O C3 também não treinava as emendas. O comando não muda no meio do episódio
     (`resampling_time_range = 1e9`), e nenhuma cadeia vai do ANDAR ao PEGAR
     (`g1_limpo/comando.py:1068-1076`). Logo, a medida do 24800 vale para a v3.
   - Condição: o controlador faz rampa, distância de parada e uma camada de rastreio calibrada
     pela curva de cada checkpoint (`cenarios/curva_*.json`). A v3 vai pedir uma curva nova.
   - Uma diferença NÃO passa do 24800 para a v3. O C3 sorteava um deslocamento lateral do alvo do
     BOTAR em relação ao robô (`g1_limpo/comando.py:1909-1912`). O plano da v3 põe a laje reta à
     frente, com jitter só para longe (plano:1554-1556). No autônomo, o erro lateral de chegada
     mediu até 6,7 cm.
   - Fica sem medida: a troca para ANDAR com a caixa na mão, depois de um falso alarme da câmera.
     Precedente no C3 (zero16, 30/09): uma troca de one-hot nunca treinada largou a caixa em 28 de
     28 envs.
6. **O freio e o refino sobem em catraca e nunca descem (confiabilidade-v1).**
   - A regra está em specs/g1-v3.md:638-650. No C3, a catraca prendeu o sucesso abaixo do
     portão. É a pendência aberta "Freio em catraca sem descida".
7. **O estimador simulado é limpo demais (percepcao-2, -3, -v1, -v2, -v3).**
   - O alvo do PEGAR e a meia aresta chegam exatos ao ator.
   - O erro de orientação é independente por leitura, mas no real ele persiste.
   - A latência fica fixa em 33 ms (plano:465), e a spec manda subir até a medida real
     (specs/g1-v3.md:700).
   - Consequência: o treino ensina a política a confiar numa leitura fresca e perfeita da caixa
     de perto.

### 2.2 Física e hardware: decidem o envelope real

8. **O limite de torque da cintura é diferente no sim e no real (fisica-2, fisica-v1, sim2real-v1).**
   - O mjlab dá 50 N·m ao `waist_pitch` e 50 N·m ao `waist_roll`, cada um de forma independente
     (`mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`: o 5020 tem 25 N·m, vezes 2).
   - No real, dois motores 5020 movem o pitch e o roll juntos por um mecanismo de barras, e a soma
     tem limite.
   - As fontes da Unitree discordam: 35 N·m no URDF do `unitree_ros`, 25 no `unitree_rl_lab` e 50
     no `unitree_rl_gym`.
   - MEDIDO no sim (specs/g1-v3.md:886-893): com 5 kg, a cintura vai a 91–99% no chão e na laje de
     0,30 m. Se o real tem 35 N·m, ou a soma fica em 50, nenhuma pose da IK com 5 kg em laje de até
     0,45 m cabe.
   - Ressalva minha (§4): um G1 real pegou 6 kg do chão em 2 de 3 tentativas (SplitAdapter). Logo,
     o limite depende da pose. Um agachamento com o tronco mais de pé passa a carga ao joelho, que
     tem 139 N·m. A tabela de IK da v3 mistura a pose "agachado 00" (tronco a 45°) com a pose de
     pé, pela altura da caixa (inbox, 08/10). Nas lajes baixas, ela inclina o tronco.
9. **A pega usa o pulso e o tronco, com o atrito da caixa (fisica-4, manip-1, sim2real-2).**
   - MEDIDO no C3 (dr-de-atrito :32-41): cápsulas do pulso com 64–73 N, tronco com 31 N e pads
     com 36–46 N. Cerca de 2/3 dos 240 N ficam fora dos pads.
   - No sim, a priority da caixa dá o μ dela a todo contato com o robô (`g1_limpo/cena.py:173-176`).
     No real, o pulso e o tronco são carcaça dura, com outro μ.
10. **A mão do robô real não é a mão do treino (sim2real-4).**
    - A nota de 16/09 diz que o robô tem a Dex3 (`vault/G1/Transversais/g1-limpo-caminho-para-o-robo-real.md:12`).
    - A decisão de 14/10 rejeita a Dex3 na v3. Para o deploy, o robô precisa de um efetuador
      impresso igual ao pad.
11. **Aperto no limite e calor (fisica-1, fisica-3).**
    - 5 kg com μ 0,5 pedem 49 N por palma. O `wrist_yaw` (5 N·m) dá 40 N com a força no centro do
      pad (specs/g1-v3.md:404-410).
    - O aperto fica no limite por 4–7 s no treino e por 12–21 s no ciclo. O sim não modela
      temperatura.
    - Um relato de campo cita um G1 EDU com o enrolamento do punho aquecido e o braço desligando
      (fórum MYBOTSHOP, citado pelo crítico). O tempo até o corte é hipótese.
12. **Carga do braço.** A ficha dá 2 kg no G1 e 3 kg no EDU (inbox :10). Com 5 kg em duas mãos,
    cada braço leva 2,5 kg, que são 83% do EDU. A v3 exige o EDU, porque usa o `wrist_pitch`, o
    `wrist_yaw` e a cintura de 3 eixos.

### 2.3 Percepção: decide se a manipulação roda no real

13. **O bloco da caixa não tem fonte real (percepcao-1).**
    - O estimador real fica fora do escopo. Sem ele, a manipulação real não roda.
    - A ponte por AprilTag já existe no repo, e a spec a aceita.
    - No estado da arte, a maior parte das falhas reais vem da percepção e da localização, mesmo
      com marcador (Hanoi: 6 de 9).
14. **O mapa e a odometria do real têm outra distribuição (percepcao-4, -6).**
    - No real, a zona perto chega cheia, há 20× mais pontos, e o galpão em volta aparece no mapa.
    - O LIO em tempo real não existe no robô. O treino modela a odometria como passeio aleatório,
      sem salto.

### 2.4 Método e orçamento: decidem se o treino chega ao fim

15. **O método não tem precedente real (sota-2, sota-v1).**
    - Todo precedente real separa as habilidades, ou usa um controlador pré-treinado, ou usa um
      prior humano.
    - No sim do Hanoi, com DR, a tarefa encadeada fecha 16% no modo mais parecido com a v3 (Joint)
      e 0 de 100 no modo ponta a ponta.
16. **O C3 nunca formou política competente do zero (crítico).**
    - A linha zero fez ~12 recomeços, e nenhum formou política competente (vault C3 g1_limpo.md:17-24).
    - A única política competente veio de ~21 blocos de resume, com troca de recompensa entre eles.
    - A v3 é uma corrida só, do zero (plano:1677).
17. **O orçamento é o da locomoção sozinha (crítico).**
    - 30 000 its é o `max_iterations` da tarefa de locomoção do G1 no mjlab
      (`mjlab/tasks/velocity/config/g1/rl_cfg.py:45`).
    - A v3 põe nesse orçamento a locomoção, o terreno, três tarefas de manipulação e o refino.
      Depois que a B e a C abrem, o ANDAR soma ~8 a 11 mil its equivalentes.
    - Cada corrida custa ~4 semanas de cota da Kaggle. Em 3 meses cabem ~3 tentativas.
18. **O aceite pede mais do que o estado da arte mede (sota-v3, confiabilidade-5).**
    - Mais de 16 cenários a 9 de 10, num só checkpoint, pedem ~97% por cenário. A chance de passar
      tudo fica em 1 a 10%.
    - Um aceite de 9 de 10 não separa uma política de 0,75 de uma de 0,95.

### 2.5 Escopo do ciclo e segurança

19. **O ciclo pede o que a v3 não treina (manip-5).**
    - O passo 6 está fora.
    - A bancada vai de 0,40 a 0,70 m (enunciado :127-128), e a laje da v3 para em 0,55 m.
    - Também ficam fora o pallet com caixas vizinhas e a chegada andando ao BOTAR.
20. **A parada de emergência derruba o robô (crítico).**
    - A parada do G1 põe o robô em amortecimento, e ele cai. O G1 não tem botão de parada no corpo
      (docs da MYBOTSHOP e da NVIDIA, citados pelo crítico).
    - Com a caixa na mão, a parada derruba um robô de ~33–35 kg e uma caixa de até 5 kg.
    - Nos primeiros testes, isso pede zona de exclusão e pórtico.

## 3. Sim-to-real (nota 2/5)

**Quebram a transferência:**

- O limite da cintura e do tornozelo, sem a soma dos dois eixos (item 8).
- O μ da caixa no pulso e no tronco (item 9).
- A mão do real diferente do pad (item 10).
- O bloco da caixa sem fonte real, e o treino que confia numa leitura fresca de perto (itens 7 e 13).
- O alvo do PEGAR e a meia aresta exatos no treino. No real, eles saem da mesma percepção que a
  caixa.
- Um G1 que não esteja no modo 5 com a cintura livre. Nesse robô, a política não roda.

**Degradam a transferência:**

- O calor no aperto contínuo (item 11).
- A fluência do contato: o sim pede ~2× a força de Coulomb. No real, sobra força, e o aperto pode
  amassar o papelão.
- O μ fixo da laje, que faz a caixa arrastar ao soltar.
- O erro de orientação persistente e a latência real acima de 33 ms.
- O mapa e a odometria (item 14).
- A DR de atuação (PD, força do motor, solref), que só entra no refino e não chega ao ANDAR.
- Os ganhos do deploy da Unitree: kd 10 nos braços e kp 200 na cintura. O treino usa kd 1 e kp 40.
- Frear, girar e o comando baixo com a caixa, que já são frágeis no sim.
- A DR do CoM da caixa, que fica desligada.

**O real precisa ter pronto:**

1. O G1 EDU confirmado no modo 5, com a cintura livre, e a mão montada conhecida.
2. Um efetuador impresso igual ao pad (geometria, ~0,25 kg, borracha de μ ≥ 0,8) e uma capa no
   punho, com o mesmo μ no sim.
3. O limite medido da cintura e do tornozelo, com o acoplamento pitch+roll, no sim antes do treino.
4. O μ medido do pad, da carcaça e do tampo da mesa contra papelão.
5. Um teste térmico de 120 s com 3 e 5 kg.
6. AprilTag na caixa e no destino no primeiro teste.
7. O LIO e o filtro do mapa em tempo real no computador do G1, com a latência p95 medida.
8. Um driver de 50 Hz com menos de 20 ms, com os ganhos exportados junto do modelo.
9. No controlador: detectar o topo da mesa e do pallet, conferir a caixa depois do B4 e ter um
   laço de nova tentativa.
10. Um sim2sim no MuJoCo clássico com o driver, depois o robô suspenso, e só então o chão.
11. Um primeiro deploy limitado: plano, até 0,5 m/s, até 3 kg, lajes de 0,30 a 0,55 m.

## 4. Estado da arte

A posição da v3 depende do eixo:

- **Dentro do que já se mostrou:**
  - a palma sem dedos com caixa leve;
  - lajes de 0 a 60 cm;
  - o mapa de lidar no G1;
  - a receita de locomoção da Unitree, com deploy real.
- **Na fronteira:**
  - a caixa de 5 kg;
  - a colocação a até 0,10 m do alvo;
  - o estimador da caixa sem marcador.
- **Além do que já se mostrou:**
  - uma política PPO só, do zero, sem dado humano, para quatro tarefas e terreno;
  - o CARREGAR em escada com 5 kg;
  - 99% de sucesso por ciclo.

Conferi os dois precedentes mais próximos no G1 real:

| trabalho | mão | estado da caixa | resultado real |
|---|---|---|---|
| PhysHSI (arxiv.org/html/2510.11072) | mão de borracha da Unitree, por pressão | não conferi | 0,6 kg 2/3; 1,2 kg 3/3; 2,3 kg 2/3; 3,6 kg 1/3; 4,5 kg 0/3. Erro de colocação 0,19 ± 0,10 m. Ciclo de carregar caixa: 6 de 10. |
| SplitAdapter (arxiv.org/html/2606.03297) | luva de atrito alto | mocap, com marcador na pelve e no topo da caixa | 2, 4 e 6 kg × 0, 30 e 60 cm: 26 de 27 ciclos. 6 kg do chão: 2 de 3. A base PhysHSI fecha 16 de 27. A política estima a massa pelo histórico de observação e ação. |

Conclusão minha: num G1 real, pegar e carregar 5 kg de 0 a 60 cm já funcionou com três apoios —
luva de atrito alto, estado da caixa por mocap e uma política que estima a massa. O limite não é só
do hardware. Ele vem da mão nua, da falta de fonte da caixa e do método de treino.

Num envelope reduzido (1 a 3 kg, lajes de 0,30 a 0,55 m, plano, marcador), a v3 é um marco de
pesquisa na faixa desses trabalhos, com 30 a 70% por cadeia no real. No envelope inteiro, ela
depende de dois eixos sem precedente: o método e os 5 kg em uso contínuo. Ela depende também de
uma peça fora do escopo: o estimador real.

## 5. O que decide o veredito, em ordem

| # | incerteza | medida mais barata | custo | se der ruim |
|---|---|---|---|---|
| 1 | O currículo único chega aos últimos níveis e ao refino em 30 000 its? | Antes do treino: o agente aplica o núcleo de sucesso à frenagem medida do 24800, nível por nível, em CPU. No treino: você lê s_ANDAR_parado, os níveis e o `freio_degrau` a cada 1 000 its. Alarme: s < 0,60 por 300 its depois de um degrau. | horas de agente; 3–20 h de GPU até os marcos | O ANDAR para em 0,5–0,75 m/s, o terreno não abre e a manipulação para em 0,30 m com 3 kg. Conserte os itens 1 e 6 antes de treinar. Se ainda travar, passe a uma hierarquia. |
| 2 | Que robô existe: G1 ou EDU, que `mode_machine`, que mão? | Ler o `mode_machine` no LowState (5 = 29 DoF com cintura livre; 6 = cintura travada), contar os motores e ver a mão. | 1 a 10 min, com acesso ao robô | Com o modo 4 ou 6, ou com o G1 padrão, a política não roda. Com a Dex3 montada, troque por um efetuador igual ao pad. |
| 3 | Qual é o limite real da cintura e do tornozelo, com o acoplamento? | Perguntar à Unitree. No robô, em pórtico: tronco a 30–45° com 0, 3 e 5 kg, lendo o `tau_est` e o erro de q do pitch e do roll. | 0 a 1 h | Com 35 N·m, ou com a soma em 50, é preciso retreinar com o limite real. O canto de 4–5 kg em laje baixa sai do envelope real, ou a pose de referência muda para um agachamento de tronco mais de pé. |
| 4 | Qual é o μ do pad (ou da luva) e da carcaça contra papelão, e o calor no aperto contínuo? | Plano inclinado com papelão (30 min). Depois, com o robô suspenso, segurar 3 e 5 kg por 120 s, lendo a temperatura e o `tau_est`. | 30 a 60 min, com o robô | Teto de 2–3 kg no real, ou uma luva de μ ≥ 0,8. No sim, contato por material e um treino novo. |
| 5 | De onde vem o estado real da caixa e do topo da mesa? | Bancada com a D435i na pose da cabeça e caixas de 14, 20 e 26 cm em 20 poses, com e sem mãos. Medir a detecção, os erros e a latência p50 e p95. Primeiro com AprilTag. | algumas horas, sem treino | Sem fonte, a manipulação real não roda. Com tag, ela roda, e o modo sem marcador fica para depois. |
| 6 | Que altura têm a mesa e o pallet do galpão? Que caixas o produto move? | Medir no galpão. | zero | Com mesa acima de 0,55 m, os passos 2, 5 e 7 saem da distribuição. Pedem níveis 0,55 → 0,70 → 0,90 m. |
| 7 | Como a política reage às emendas? | RESPONDIDA pelo dono (15/10): o 24800 no autônomo fecha os ciclos sem treino das emendas (§2.1, item 5). | — | — |

As medidas 2, 3 e 4 pedem o robô. O vault registra que você não tem acesso a ele
(`C4 v3/g1-v3-palma-de-plastico-5kg.md:11`).

## 6. Mudanças que aumentam a viabilidade

Em 15/10, o dono aprovou as três mudanças da cena, e elas entraram no plano: a cerca de torque pela soma do par (PD-45), o atrito por material (PD-46) e a laje de destino de lado e girada ±10° (PD-47). A laje girada foi um pedido do dono. No mesmo dia entraram as quatro de currículo e recompensa: a âncora do parado (PD-15), o `passo` ao quadrado (PD-48), o σ do canal zero (PD-49) e o sucesso sem perda (PD-50). O estimador e o aceite esperam o veredito do dono.

### 6.1 Mudam o resultado no sim, e custam pouco antes do código

| mudança | onde | efeito previsto | achados |
|---|---|---|---|
| Medir o sucesso do canal zero pela velocidade média no último 1 s, descontando 1,5 s depois de cada troca e de cada empurrão. | PD-15; núcleo de sucesso (plano:1173) | O ANDAR pode chegar a 1,5 m/s, e o terreno e o refino abrem. | loco-3, loco-v3, loco-5 |
| Pagar o valor do último toque do `passo` a cada passo de controle, vezes o rastreio. | S §19; termo `passo` | O passo miúdo deixa de ser ótimo. | loco-1 |
| Mudar o ramo `c < 0,02` para −min((v/c_ref)², 1), com c_ref de 0,10 m/s e 0,15 rad/s. Sortear wz de 0,02 a 0,25 rad/s e dar comando baixo ao CARREGAR. | rastreio; sorteio de comando | Giro no lugar sem deslizar; a chegada lenta com a caixa passa a ser treinada. | loco-2, loco-v1, loco-v2 |
| Acrescentar "e não `v3_perdeu`" a toda regra de sucesso. | plano:1497-1498 | O nível não sobe com a caixa no chão, e o log mede o que o ciclo usa. | confiabilidade-2 |
| Sortear um deslocamento lateral do alvo do BOTAR em relação ao robô, como o C3 fazia (`g1_limpo/comando.py:1909-1912`). A variante C' e o comando zero no meio da B SAEM: o dono mediu, com o 24800, que o ciclo não precisa deles (15/10). | F2 (`posiciona_destino`, plano:1554-1556) | O BOTAR aceita o erro lateral de chegada, que mediu até 6,7 cm no autônomo. | confiabilidade-4, manip-5 |
| Pôr no atuador do sim o limite real da cintura e do tornozelo (ou 35 N·m, por segurança), com o limite da soma pitch+roll. | cena; knob do atuador | O que passa no sim passa no real. O canto de 4–5 kg em laje baixa pode sair do envelope. | fisica-2, fisica-v1, sim2real-v1 |
| Contato por material: o robô com priority acima da caixa; DR de μ só no pad (0,5–0,9); o pulso e o tronco com o μ medido; DR de μ na laje de 0,3 a 1,0. | cena e DR | A pega do sim passa a depender do que existe no real. O arrasto na soltura aparece no treino. | fisica-4, sim2real-2, manip-1 |
| Estimador mais honesto: FOV do RGB, alcance mínimo de 0,5 m, viés de orientação por episódio, ±2 cm no tamanho e no alvo do PEGAR, erro no alvo do BOTAR, latência medida, caixa tirada do mapa pela cinemática das palmas. | S §13; plano:465 | A política deixa de depender de uma leitura perfeita de perto. | percepcao-2, -3, -5, -v1, -v2, -v3 |
| Portão físico no aceite: saturação abaixo de 5%, \|τ\| médio de 10 s até 60%, soma da cintura até 40 N·m. | S §15 | Nada que só funcione no sim chega ao robô. | fisica-v2, fisica-3, sim2real-3 |
| Aceite estatístico: N ≥ 50 com o limite de Wilson, sucesso por faixa (altura × massa × tamanho), sorteio pelo déficit da faixa. | S §15; métricas | O portão para de reprovar por ruído, e o canto difícil para de se esconder na média. | confiabilidade-5, confiabilidade-v2, sota-v3, manip-2 |

### 6.2 Hardware e produto: decisão sua

- Um efetuador impresso com a geometria do pad e borracha de μ ≥ 0,8, com a mesma capa no punho.
  Com μ de 0,8 a 1,0, o pad segura 4,6 a 5,8 kg parado. O SplitAdapter mostra 4 kg em 9 de 9 com
  luva.
- Um primeiro deploy com envelope reduzido: até 3 kg, lajes de 0,30 a 0,55 m, plano, até 0,5 m/s e
  AprilTag na caixa e no destino.
- Um ciclo mais curto: ler o código na mesa A, ou com a caixa na mão. O ciclo cai de 10 para 6
  etapas, com 1 PEGAR, 1 BOTAR e 2 trechos de CARREGAR a menos.
- Níveis de mesa 0,55 → 0,70 → 0,90 m, se o galpão pedir.

### 6.3 Arquitetura: o plano B

Uma hierarquia: a locomoção da Unitree congelada como base, mais uma política de manipulação ou um
residual sobre ela. É o desenho de todo precedente real, e corta o maior risco de convergência. O
custo é rearquitetar e abrir mão de "um modelo só".

### 6.4 Minha recomendação

1. Antes do despacho, responda às perguntas 1, 2 e 3 da §7. Elas mudam a cena e o atuador, e
   mudar isso depois do código custa um treino de ~4 semanas.
2. Aplique ao plano as mudanças da §6.1, depois do seu OK. A maioria mexe em PD que você ainda
   revisa (PD-15) ou na forma de termos existentes. Nenhuma acrescenta termo de recompensa.
3. Deixe a hierarquia como plano B, com um gatilho escrito. Exemplo: o PEGAR não abre o nível 3 em
   10 000 its.

## 7. Perguntas que só você responde

1. Que robô existe hoje: G1 ou G1 EDU, em que `mode_machine`, com que mão? A nota de 16/09 diz
   Dex3, e a decisão de 14/10 diz mão de borracha. Quando você terá acesso ao robô para os testes
   de 30 a 60 min?
2. Que altura têm a mesa B e o pallet no galpão alvo? O enunciado põe a bancada a até 0,70 m, e a
   v3 para em 0,55 m.
3. Os 5 kg são um requisito do produto ou um teto de teste? O produto move cubos de 14 a 26 cm ou
   caixas de galpão maiores?
4. Você aceita um teto de 3 kg no primeiro deploy real, até medir o μ, o limite da cintura e o calor?
5. Você aceita AprilTag na caixa e no destino no primeiro teste real?
6. RESPONDIDA (dono, 15/10): as emendas não entram no treino. O 24800 fecha os ciclos quando
   aprende bem as tarefas isoladas.
7. O galpão pede escada com caixa, ou bastam soleira e rampa de doca? Se bastam, o terreno com
   caixa pode sair da v3.
8. Que taxa por ciclo, e quantas intervenções por hora, contam como sucesso do produto? O 9 de 10
   por cenário é marco de pesquisa.
9. Se o currículo único travar num marco, você aceita passar a uma hierarquia com a locomoção
   congelada?
10. Você quer um teto numérico para a velocidade da mão perto de pessoas? O enunciado diz só
    "devagar" (:176-178). O 0,25 m/s que a síntese usou vem da estimativa de tempo da spec, e não
    do enunciado.

## 8. Limites desta validação

- Nenhum número da §1 saiu de simulação nova. São faixas dos agentes, ancoradas no C3.
- A amostra do autônomo é selecionada. As contagens "11 de 13" e "12 de 14" omitem 6 falhas
  anteriores ao ajuste do controlador, em 05/10 (`lidar_sim/runs/autonomo/20261005-124232` a
  `-132852`).
- 7 dos 9 CSVs de BOTAR são anteriores ao commit `025a9d6` (giro_w no registrador). O erro de 5 a
  11 cm pode estar confundido (plato-do-botar, A9). Fica a conferir.
- A razão de velocidade com a caixa (0,60–0,72) depende do comando inferido do roteiro, porque o
  CSV não tem coluna de comando.
- Os pesos que decidem o aperto, as terminações e as cercas ainda são `None` até o portão da
  Parte 0 (plano:965-975). As previsões de aperto e de pouso supõem números que ainda não existem.
- O mecanismo da falha com 4–5 kg no real está em disputa: a caixa escorrega, ou a cintura e o
  calor cedem. A conclusão é a mesma nas duas leituras; só a medida da §5, item 4, decide.
- Duas lacunas ficaram sem resposta: o desgaste e a sujeira do pad com o uso, e a queda de tensão
  da bateria com o aperto saturado.

## 9. O que eu conferi na fonte

| afirmação | resultado |
|---|---|
| O sim dá 50 N·m independentes à cintura | CONFERE: o 5020 tem `effort_limit=25.0`, e a cintura usa `ACTUATOR_5020.effort_limit * 2` por eixo (`g1_constants.py`). |
| A cintura vai a 91–99% com 5 kg | CONFERE: specs/g1-v3.md:888-893. |
| 30 000 its é o orçamento da locomoção do mjlab | CONFERE: `rl_cfg.py:45`. |
| O PD-15 mede a deriva no segmento; o terreno abre depois do último nível | CONFERE: plano:264, :267, :578-582. |
| Frear de 0,5 m/s anda 0,126 m no 24800 | CONFERE: `curva_model_24800.json` (0,126; 0,354 a 1,0 m/s; 0,623 a 1,5 m/s). |
| O termo do passo paga por toque | CONFERE: specs/g1-v3.md:1072-1077. |
| A caixa perdida conta no sucesso do PEGAR | CONFERE: plano:1466-1470 e :1497-1498. |
| ~2/3 da força da pega fora dos pads | CONFERE: dr-de-atrito :30-42 (64 + 73 + 31 de ~240 N). |
| A bancada vai a 0,70 m | CONFERE: enunciado :127-128 (bancada do REORIENTAR, que é a mesa B do ciclo). |
| O robô tem a Dex3 (nota de 16/09) | CONFERE: `g1-limpo-caminho-para-o-robo-real.md:12`. A Dex3 foi rejeitada na v3 em 14/10 (inbox). |
| 28 de 28 largaram a caixa com troca não treinada | CONFERE: `g1-limpo-fechar-o-pegar-mata-o-episodio.md:10`. Medida do zero16, de 30/09. |
| PhysHSI e SplitAdapter | CONFERE nas páginas do arXiv (§4). A síntese disse "4 kg em 9/9 com luva e mocap", e isso confere. |

## Anexo A. Tarefa × resultado previsto, requisito a requisito

Fonte: síntese do workflow, com os ajustes dos verificadores. "sim" e "real" dizem se o requisito é atendido. Os ids dos achados estão no anexo B.

O R31 (emendas) foi rebaixado em 15/10 pela medida do dono com o 24800 (§2.1, item 5). A linha abaixo guarda a previsão original.

| req | requisito | pedido | previsto | sim | real | achados |
|---|---|---|---|---|---|---|
| R1 | Ciclo do galpão completo | 10 passos; 5 tarefas; BOTAR em 2 alturas; teste do dono | Não fecha nem no sim, porque o passo 6 está fora (specs/g1-v3.md:81-82). Sem ele, com controlador roteirizado: 0,5–0,8 por ciclo com caixa fácil e 0,2–0,5 na média do envelope. MEDIDO com o 24800 e oráculo: 11 de 13 corridas do autônomo fecharam; as 2 falhas foram quedas no PEGAR com caixa de 16 cm (lidar_sim/runs/autonomo/L_24800, h_24800). No real, hoje: 0. | nao | nao | confiabilidade-1, manip-5, sota-v1, percepcao-1 |
| R2 | Sucesso do ciclo: as 4 condições juntas | robô e caixa não caem; caixa a ≤ 0,10 m, estável 0,5 s; robô de pé | Em cenário isolado, as 4 condições fecham no sim. No ciclo, vale a conta do R1. O sinal do currículo conta como sucesso do PEGAR a caixa que cai no CARREGAR (plano:1469-1470, :1497-1498), contra specs/g1-limpo-comportamento.md:84-86. O aviso de BOTAR concluído olha só a pose do robô. | parcial | nao | confiabilidade-2, confiabilidade-3 |
| R3 | Tempo: episódio de no máximo 20 s; robô devagar | PEGAR 3–7 s, BOTAR 3,2–7,2 s; pior caso 18,2 s na C | PEGAR em 1,5–4 s e BOTAR em 2,5–6 s. O C3 MEDIU a caixa no peito 0,5–2,2 s depois da abertura (docs/relatorios/2026-10-09-viabilidade-v3.md:91). O pior caso cabe nos 20 s (specs/g1-v3.md:255-258). O 'devagar' falha (ver R42). | sim | incerto | cobertura resultado_manip R3, cobertura confiabilidade R3 |
| R4 | Cubo de 14–26 cm | meia aresta 0,07–0,13 m; IK em 3 tamanhos | Caixas de 20 e 26 cm saem como o previsto. A de 14 cm fica 5–15 p.p. pior. Cinco das 18 pegas da IK não fecham (specs/g1-v3.md:320-328). MEDIDO no autônomo: a caixa de 16 cm caiu 2 de 2 vezes, e a de 18–20 cm passou. | parcial | incerto | manip-2, confiabilidade-v2, sim2real-4 |
| R5 | Massa desconhecida, adaptação | 1–5 kg; níveis 0,55/1 → 0,04/5 | MEDIDO no C3: 211–275 N de aperto com 1 a 5 kg, sem modular. No sim: 1–3 kg passam de 85% nas lajes altas; 4–5 kg nas baixas ficam em 40–75%. No real: 1–2 kg 60–85%, 3 kg 30–60%, 4–5 kg abaixo de 20% sem luva e 40–70% com luva em laje alta. PhysHSI, G1 real: 2,3 kg 2/3, 3,6 kg 1/3, 4,5 kg 0/3. | parcial | parcial | manip-1, fisica-1, sim2real-1, sota-1, fisica-v3 |
| R6 | Atrito palma × caixa | μ ≥ 0,5; DR 0,6–1,0 → 0,5; luva se preciso | Pela priority 2 da caixa, o μ dela vale para todo contato robô → caixa, pulso e tronco incluídos (g1_limpo/cena.py:173-176, :201-203). A laje fica com μ 1,0 fixo (cena.py:232-234). Nada disso foi medido no real. | parcial | incerto | fisica-4, sim2real-2, manip-1 |
| R7 | 5 kg só pelas palmas | 49 N por palma; wrist_yaw a 80% dá 40 N (centro) e 62 N (borda) | No sim, a caixa fica, mas pelo pulso e pelo tronco (58–82% da força fora dos pads, MEDIDO no C3). Na marcha, o pico pede 64 N por palma: fz_rob p95 de 64 N no 24800_mu10_m5 (conta minha). No real, com μ 0,5, não segura no centro do pad. Com luva de μ ≥ 0,8, a conta fecha. | parcial | incerto | fisica-1, sim2real-1, manip-1, sota-1 |
| R8 | Carga de braço dentro da ficha | 2 kg (G1), 3 kg (EDU); 5 kg dão 2,5 kg por braço | A v3 exige 29 DoF, que só existe no EDU (README g1_description:15-18; unitree.com/g1). No EDU, 2,5 kg dão 83% parado e ~3,25 kg equivalentes no pico da marcha. O aperto lateral satura o shoulder_yaw. | nao_se_aplica | parcial | sim2real-v3, sota-v2, sota-1 |
| R9 | Todo atuador abaixo de 80% | cintura 66–97% na IK e 91–99% em física com 5 kg | Na física, com 5 kg, a cintura fica a 85–99% nas lajes de até 0,30 m (segura_pose.txt). A soma /τp/+/τr/ da cintura no PEGAR do C3 dá p95 de 47,7/58,2/69,9 N·m (conta minha). A cerca é custo mole, e o §15 não tem portão de torque. | nao | nao | fisica-2, fisica-v1, fisica-v2, manip-v1 |
| R10 | Lajes de 0,04–0,55 m | pega e destino independentes; C pega em ≥ 0,30 m | No sim: 85–95% em 0,45–0,55 m. Em 0,30 m com 4–5 kg: 70–90%, pagando a cerca. Em 0,04–0,15 m com 4–5 kg: 50–80%. O ponto mais duro é 0,30 m (95–99% de cintura, saturada 100% do tempo com 14 cm; segura_pose.txt:19). No real, 0,45–0,55 m cabem. Até 0,30 m com 3 kg ou mais, depende do limite da cintura. | parcial | parcial | manip-2, manip-v1, fisica-2, fisica-v1 |
| R11 | Envelope de posição na pega | base x −0,10–0, y ±0,10, rumo ±0,2 rad | Pega bem dentro da faixa. MEDIDO com oráculo: chegada longitudinal de −0,060 a −0,074 m e lateral de até 0,092 m, dentro da faixa. No treino, o alvo do PEGAR é a verdade (plano:1451). No real, ele sai do estimador, com ±2 cm. | sim | incerto | percepcao-v1, confiabilidade-4 |
| R12 | Alvo de soltura no centro da laje, no recuo da IK | recuo de 0,25–0,44 m; a laje só se afasta | A política aprende um alvo quase fixo. No ciclo, a chegada lateral ao BOTAR erra até 6,7 cm (MEDIDO com oráculo). O plano não sorteia a laje de destino de lado (plano:1554-1556). | sim | parcial | confiabilidade-4, manip-5, percepcao-5 |
| R13 | Caixa a até 0,10 m do alvo | ≤ 0,10 m | MEDIDO no C3 (conta minha): 5,0 a 11,2 cm em 9 plays, 2 acima de 10 cm. Na v3: 2–6 cm, com 5–15% acima de 10 cm. Só o B1 a 3 cm sustenta isso. No real, sem marcador no destino: 0,1–0,2 m (PhysHSI 0,19 ± 0,10 m). | parcial | incerto | manip-v2, percepcao-5, sota-4 |
| R14 | Face de cima normal ao solo | ≤ 15° no fim (30° no aprender) | MEDIDO no C3 com 5 kg: p95 15,6° e máximo 22°. Na v3: p95 8–14°, e o pico passa de 15° em 10–30% das rodadas com empurrão. | parcial | parcial | manip-4 |
| R15 | Caixa estável depois de apoiada | < 2 cm; ‖v‖ < 0,05 m/s por 0,5 s sem contato | Comportamento novo. O C3 nunca soltou dentro do BOTAR: fn_tot de 195 N no último passo do BOTAR do 24800_botar_t015 (conta minha). Na v3: 70–90% das solturas mexem a caixa menos de 2 cm. No real, a mesa com μ abaixo do 1,0 do sim deixa a caixa arrastar. | parcial | parcial | manip-3, sim2real-2 |
| R16 | Pouso sem impacto | vz ≤ 0,25 m/s no fim; aceite 0,04 m com 5 kg | Chegada a 0,1–0,3 m/s; o C3 MEDIU 0,77–1,18 m/s. O aceite de 0,04 m com 5 kg passa com 30–50% de chance. No real, o erro em z do alvo nunca aparece no treino. | parcial | incerto | manip-2, percepcao-5 |
| R17 | Menor aperto que segura | sem teto numérico; o C3 fez 110–125 N por face | 150–260 N no total e 60–125 N por face, quase sem modular pela massa. A fluência do sim pede ~2× a força de Coulomb (dr-de-atrito:54, :72-79). No real, a força sobra e o papelão pode marcar. | parcial | incerto | sim2real-v2, fisica-v3, manip-1 |
| R18 | Caixa não sacode | sem limiar; peso só no refino | MEDIDO no C3: /g+a/ da caixa no CARREGAR com p95 de 1,33 a 1,40 g. A v3 deve ficar parecida (1,25–1,40 g). Não há aceite. | incerto | incerto | fisica-1 |
| R19 | Faces laterais, duas mãos, sem antebraço | par de faces mais alinhado; P0–P2 com menos de meia aresta | Duas mãos nas faces laterais, mas o pulso faz a maior parte da força. A cápsula sobra 12 mm além do pad (g1.xml:222; cena.py:60-62). A regra 'antebraço proibido' não tem termo nem aceite. | parcial | incerto | fisica-4, manip-1 |
| R20 | Sem toque na laje; mãos fora do chão | ≤ 10 N; 9 de 10 por nível | Nas lajes altas, sem toque. Nas de 0,04–0,15 m, joelho ou canela acima de 10 N em 5–10% das rodadas depois do refino. A cobrança começa em 50 N e só chega a 10 N no refino (specs/g1-v3.md:490). | parcial | parcial | manip-2, cobertura resultado_manip R20 |
| R21 | Transporte no peito, mesma pose | caixa z 0,90 ± 0,05; pelve 0,78–0,79 m | Caixa a 0,85–0,92 m. Pelve a 0,72–0,77 m andando. MEDIDO no C3 com 5 kg: caixa a 0,843 m e pelve a 0,722 m (p50, conta minha), com rel_z caindo 7–9 cm em 5 s. | parcial | parcial | cobertura resultado_loco R21, cobertura resultado_manip R21 |
| R22 | Seguir velocidade em 4 direções e girar | vx −1,0 a +1,5; vy ±1,0; wz ±1,0; razão ≥ 0,50 | Sem caixa: razão de 0,85–1,0 até o teto que abrir, provavelmente 0,5–0,75 m/s; 1,0–1,5 m/s ficam sem treino. Com caixa, a 0,25–0,5 m/s: razão de 0,6–1,0. No real, no plano até 1,0 m/s, perde 5–15%. | parcial | parcial | loco-v3, loco-3, sim2real-5 |
| R23 | Comando baixo, sem derivar | 0,05–0,25; aceite vx 0,10 / wz 0,15; rumo < 3° com k = 2 | vx 0,10 vira 0,06–0,12 m/s; wz 0,15 vira 0,10–0,17 rad/s. O rumo com k = 2 assenta a 1–6°, e o aceite passa em 40–70%. O treino nunca sorteia wz pequeno com marcha (plano:1159-1167). | parcial | incerto | loco-2, loco-v1, percepcao-6 |
| R24 | Frear com caixa | ≤ 3 passos, 1,0 s, 0,30 m | 2 a 5 passos e 0,15–0,50 m. O aceite passa em 30–60%. Nada treina isso, e o freio do parado torna o passo caro. | incerto | incerto | loco-4, loco-v2 |
| R25 | Giro de 180° com caixa | 180 ± 15° em 4 s; base < 0,10 m | Gira 150–185°, e a base anda 0,08–0,35 m. O aceite passa em 25–45%. | parcial | parcial | loco-2, loco-v1 |
| R26 | Marcha sem passo miúdo | passo ≥ 0,16 m a 0,25 m/s e ≥ 0,18 m a 0,5 m/s | 0,05–0,13 m a 4,5–6,5 passos/s. O aceite falha em 90% ou mais das rodadas, porque o termo do passo é plano na cadência. MEDIDO no C3: 0,046–0,057 m a 6,2–6,7 passos/s com caixa. | nao | nao | loco-1 |
| R27 | Terreno no ANDAR e no CARREGAR | degrau 0,20/0,15 m; rampa 20/15°; obstáculo 0,15/0,10 m | O terreno só abre depois do nível de 1,5 m/s (plano:578-582), e o portão do parado trava antes disso. Chance de abrir em 30 000 its: ≤ 20%. Não há precedente de G1 com carga acima de ~1 kg em escada. | nao | nao | loco-v3, loco-5, sota-5 |
| R28 | Resistir a empurrão | ±0,5 m/s a cada 5 s; caixa volta a ≤ 15° em 1 s | Sem caixa, resiste. Com caixa, cai em 5–15% dos empurrões, e a caixa volta a ≤ 15° em 70–85% das vezes. | parcial | parcial | loco-4 |
| R29 | Nenhuma junta no batente | ≤ 95% do curso em 9 de 10 | Deve passar: a tabela do SDK é mais restrita que a página do produto (specs/g1-v3.md:197-220). O C3 levou o wrist_yaw E a 88% do curso (vault/_sistema/Home.md). A diferença de 9° no hip_pitch segue pendente. | sim | sim | cobertura sim2real_dinamica R29 |
| R30 | Transições sem tranco; BOTAR de um comando | cadeias B e C; detector de fim | As trocas da B e da C são treinadas. O BOTAR chega ao B4 em 70–90% nas lajes altas e em 50–75% nas baixas e pesadas. 'Sem tranco' não tem número. | parcial | parcial | manip-3, confiabilidade-3 |
| R31 | Emendas do ciclo não treinadas | risco aceito | O BOTAR depois de andar cai 10–30 p.p. Com caixa, comando abaixo de 0,25 m/s nunca é treinado. A chegada à mesa e ao pallet passa justo por essa faixa. | nao | nao | loco-v2, loco-4, confiabilidade-4, sota-v1, percepcao-4 |
| R32 | Estimador simulado com a forma do real | atraso, taxa, oclusão; erro do refino 5,3°/15,3° | A forma existe, mas mais generosa que o real. A visibilidade usa o FOV do depth (87°×58°) no lugar do RGB. A latência fica fixa em 33 ms, contra specs/g1-v3.md:700. O tamanho não tem viés, contra specs/g1-v3-comportamento.md:645. O alvo do PEGAR é exato. | parcial | parcial | percepcao-2, percepcao-3, percepcao-v1, percepcao-v2, percepcao-v3 |
| R33 | Estimador real da caixa | dentro do erro simulado | Não existe. O g1-box-vision só usa RGB, um cubo de 10 cm e simulador. Há uma ponte por AprilTag no repo (lidar_sim/percepcao/tags.py), aceita pela spec como fonte real. | nao_se_aplica | nao | percepcao-1, sota-3 |
| R34 | Mapa de elevação do MID-360 | 21 × 21 de 10 cm; mesmo filtro | A forma é a mesma nos dois lados. Com a caixa, a frente fica sem leitura. No treino a zona perto chega vazia; no real chega cheia. O real tem 20× mais pontos. O filtro e o LIO não existem em tempo real. | parcial | incerto | percepcao-4, percepcao-6, loco-5 |
| R35 | Laço de 50 Hz e entrada só com sensores reais | < 20 ms; 1 447 entradas; ganhos exportados | Os blocos 1–7 (510 entradas) transferem. Outras 937 entradas não têm fonte real medida. A cintura usa o kp 40 da configuração que limita a 25 N·m, mas o sim dá 50 N·m (specs/g1-v3-interface-real.md:198; g1_constants.py:165-170). | sim | parcial | sim2real-v1, sota-v2 |
| R36 | Controlador externo | tarefa, comando, alvo; status | Fora da v3. Faltam três coisas: detectar o topo da mesa e do pallet, conferir a caixa depois do B4 e um laço de nova tentativa. | nao_se_aplica | fora_do_escopo | confiabilidade-3, percepcao-5 |
| R37 | Mão real igual ao pad do treino | μ ≥ 0,5; mesma geometria | A peça não existe. A nota de 16/09 diz que o robô tem a Dex3 (g1-limpo-caminho-para-o-robo-real.md:12). Sem acesso ao robô, não há como medir o μ (C4 v3/g1-v3-palma-de-plastico-5kg.md:11). | nao_se_aplica | nao | sim2real-4, fisica-4 |
| R38 | Validação em dois simuladores | Isaac Sim 9 de 10; Isaac Lab a até 10 p.p. | É extensão, e não roda nesta máquina, que não tem GPU NVIDIA (specs/g1-v3.md:941-949). O sim2sim mais barato é o MuJoCo clássico com o driver. | nao | fora_do_escopo | cobertura sim2real_dinamica R38 |
| R39 | REORIENTAR | primitivas de 90°; erro < 10° | Fora da v3. Sem ele, o ciclo não fecha. | nao_se_aplica | fora_do_escopo | confiabilidade-1 |
| R40 | LEVANTAR | slot reservado | Fora da v3. Toda queda pede um humano. | nao_se_aplica | fora_do_escopo | confiabilidade-3 |
| R41 | Prateleira | folga de 7,5–15 cm | Fora da v3. | nao_se_aplica | fora_do_escopo |  |
| R42 | Devagar perto de pessoas | mão ≤ 0,25 m/s | A mão anda a 0,4–0,9 m/s no p95. O C3 MEDIU a caixa a 0,45–1,66 m/s no p95. A v3 não tem número de aceite. | nao | nao | cobertura resultado_manip R42, sim2real-6 |
| R43 | Caixas reais de galpão (hipótese) | 0,15–0,60 m; até 10–23 kg | Só treina cubo rígido e homogêneo. A DR do CoM fica desligada (PD-38). Os pads dão 3–6 kg. | nao_se_aplica | fora_do_escopo | sim2real-7, fisica-1 |
| R44 | Mesa de 0,72–0,90 m (hipótese) | 0,72–0,90 m | Fica fora da distribuição. O teto é 0,55 m, e o próprio enunciado põe a bancada a 0,40–0,70 m (specs/g1-limpo-comportamento.md:126-128). Em torque, seria o caso mais barato. | nao | fora_do_escopo | manip-5 |
| R45 | Pallet real (hipótese) | 1,0 × 1,2 m; camadas; folga de 2–3 cm | Bota no centro de uma laje isolada, sem caixas ao lado e sem empilhar. | nao | fora_do_escopo | manip-5 |
| R46 | Sucesso operacional (hipótese) | ≥ 99% por ciclo | No sim: 0,2–0,8 por ciclo sem o passo 6. No real: 5–35%. O aceite 9 de 10 não separa 0,75 de 0,95. | nao | nao | confiabilidade-1, confiabilidade-5, sota-4 |
| R47 | Tempo de ciclo (hipótese) | ≤ 60 s por caixa | 32–68 s, mais o REORIENTAR, que não existe (conta). No real, o tempo de recuperação domina. | incerto | incerto | cobertura estado_da_arte R47 |
| R48 | Segurança operacional (hipótese) | parada de emergência, zona, queda controlada | Nada disso está na v3. O robô pesa 33 kg. | nao_se_aplica | fora_do_escopo | confiabilidade-3 |
| R49 | Bateria e regime térmico (hipótese) | ~2 h; regime contínuo | O sim não modela temperatura. Com 3–5 kg, o shoulder_yaw passa de 95% em 77–93% dos passos do CARREGAR (C3 MEDIDO). O risco de calor existe, mas não foi medido. | nao_se_aplica | incerto | fisica-3, sim2real-3 |
| R50 | Chão do galpão | μ 0,3–0,8; soleira de 2 cm; doca de 6° | A DR do pé (0,3–1,0) cobre o atrito, mas o treino fica só no plano. A soleira e a doca com 5 kg não são treinadas, e filme stretch fica abaixo da DR. | parcial | parcial | loco-v3, cobertura resultado_loco R50 |

## Anexo B. Achados por frente

O analista propôs cada achado, e o verificador adversarial deu o veredito. Os refutados saíram. A tabela não traz os achados com o sufixo "v", que o verificador acrescentou sem uma segunda checagem.

### Física do G1 real

Veredito da frente: viavel com ressalvas (analista: inviavel como desenhado; viés da análise: equilibrada). Dificuldade 4/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| fisica-1 | alta | parcial | 5 kg com μ 0,5 não cabem no wrist_yaw só com os pads |
| fisica-2 | alta | parcial | O sim dá 50 N·m à cintura; as fontes da Unitree dão 25, 35 ou 50 |
| fisica-3 | media | parcial | O aperto no limite por dezenas de segundos aquece o punho e o ombro |
| fisica-4 | alta | confirmado | No sim o pulso tem o μ da caixa; no real é carcaça dura |
| fisica-5 | media | confirmado | O tornozelo tem a mesma dúvida de limite que a cintura |
| fisica-v1 | alta | novo | Cintura: pitch e roll dividem dois motores, e o sim os trata como independentes |
| fisica-v2 | alta | novo | O aceite da v3 não mede torque, saturação nem aperto |
| fisica-v3 | media | novo | A luva não baixa o torque: o ator aperta para μ 0,5 em qualquer caixa |

### Locomoção prevista

Veredito da frente: viavel com ressalvas (analista: viavel com ressalvas; viés da análise: equilibrada). Dificuldade 3/5. Sim-to-real 3/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| loco-1 | alta | confirmado | O termo do passo não impõe passo mínimo: é plano na cadência e vale 2,5% do rastreio |
| loco-2 | alta | confirmado | Sem comando, deslizar e girar quase não custam; o treino tira o laço de rumo |
| loco-3 | alta | confirmado | A régua de sucesso do canal zero mede posição; a frenagem e o empurrão a estouram, e o nível trava |
| loco-4 | alta | parcial | Frear e segurar o empurrão com a caixa: não treinado, e o freio do regime parado torna o passo caro |
| loco-5 | media | parcial | Terreno e mapa: último na fila do currículo, cego com a caixa e sem odometria real em tempo real |
| loco-v1 | media | novo | O laço de rumo do driver e as curvas suaves pedem um comando que o treino nunca sorteia |
| loco-v2 | alta | novo | Com a caixa, comando abaixo de 0,25 m/s e de 0,25 rad/s nunca é treinado; a chegada à mesa B e ao pallet passa por essa faixa |
| loco-v3 | alta | novo | O portão do parado tem piso físico: o ANDAR trava em 0,5–0,75 m/s e o terreno não abre |

### Manipulação prevista

Veredito da frente: viavel com ressalvas (analista: viavel com ressalvas; viés da análise: otimista). Dificuldade 4/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| manip-1 | alta | confirmado | A pega de 4–5 kg depende do pulso e do tronco, cujo atrito no sim é o da caixa e no real não foi medido |
| manip-2 | media | parcial | O canto difícil do envelope (laje baixa com 4–5 kg, caixa de 14 cm) fica sem pressão de treino |
| manip-3 | media | parcial | BOTAR de um comando (apoiar nivelado, soltar sem mexer, ficar de pé) é um comportamento que nenhum modelo anterior fez |
| manip-4 | media | confirmado | O limite de 15° em todo passo, com 5 kg e empurrão, fica no fio |
| manip-5 | alta | confirmado | O envelope treinado não cobre a mesa padrão, o BOTAR depois de andar nem o pallet com vizinhas |
| manip-6 | media | parcial | A cintura com kp 40 só faz o torque de segurar 5 kg com o alvo do PD ~1 rad além do batente |
| manip-v1 | media | novo | O ponto físico mais duro é a laje de 0,30 m com 4–5 kg, não o canto de 0,04–0,15 m, e ele entra em toda pega da cadeia C |
| manip-v2 | media | novo | O C3 bota a caixa a 5–11 cm do alvo; a previsão de 1–5 cm só se apoia no portão do B1 |
| manip-v3 | media | novo | O ator recebe a meia aresta sem erro no treino, e a pega da v3 passa a depender dela |

### Confiabilidade do ciclo

Veredito da frente: inviavel como desenhado (analista: inviavel como desenhado; viés da análise: equilibrada). Dificuldade 5/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| confiabilidade-1 | alta | parcial | Composição multiplicativa: o ciclo herda o portão de 0,75 e fecha em 6–68% no sim |
| confiabilidade-2 | alta | confirmado | Com a chave CAIXA_PERDIDA_VIRA_ANDAR ligada, o sucesso do currículo não conta a perda da caixa |
| confiabilidade-3 | alta | parcial | As falhas não têm recuperação na v3, e parte delas é silenciosa para o controlador |
| confiabilidade-4 | media | parcial | No ciclo, o PEGAR e o BOTAR começam fora do estado treinado |
| confiabilidade-5 | media | confirmado | O aceite 9 de 10 não separa uma política de 0,75 de uma de 0,95 |
| confiabilidade-v1 | alta | novo | A catraca do freio e do refino pode prender o sucesso final abaixo do portão |
| confiabilidade-v2 | alta | novo | O portão por nível esconde a cauda que o ciclo usa, e as falhas andam juntas pela caixa |
| confiabilidade-v3 | media | novo | Nenhum cenário de aceite mede a taxa de sucesso do PEGAR, o único elo que falhou no autônomo |

### Sim-to-real: dinâmica

Veredito da frente: viavel com ressalvas (analista: viavel com ressalvas; viés da análise: equilibrada). Dificuldade 4/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| sim2real-1 | alta | parcial | Segurar 4–5 kg pela pressão lateral passa do punho de 5 N·m no G1 real |
| sim2real-2 | media | parcial | O atrito por priority dá ao punho, ao tronco e à laje um μ que o real não tem |
| sim2real-3 | media | parcial | O sim não tem o regime do motor: aperto contínuo no teto, curva de torque e ganhos do deploy |
| sim2real-4 | alta | confirmado | A mão do robô real não é a mão do treino |
| sim2real-5 | media | parcial | A locomoção transfere no plano até 1,0 m/s; acima disso e no terreno, degrada |
| sim2real-6 | baixa | parcial | Se a caixa escapa no real, as mãos se fecham com torque cheio |
| sim2real-7 | baixa | parcial | Uma caixa real com conteúdo fora do centro gira na mão curva; a DR do CoM está desligada |
| sim2real-v1 | alta | novo | Cintura e tornozelo têm 50 N·m no sim; o URDF da Unitree dá 35 e a receita treina com 25 |
| sim2real-v2 | media | novo | A fluência do contato no sim pede ~2× a força de Coulomb: no real sobra atrito e falta o 'menor aperto' |
| sim2real-v3 | baixa | novo | A v3 exige o G1 EDU com 2 DoF de punho; o G1 padrão não tem wrist_pitch nem wrist_yaw |

### Sim-to-real: o que a política recebe

Veredito da frente: inviavel como desenhado (analista: inviavel como desenhado; viés da análise: equilibrada). Dificuldade 4/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| percepcao-1 | alta | parcial | O estimador real da caixa não existe, e o protótipo do repo não mede o que os 7 canais pedem |
| percepcao-2 | alta | parcial | No treino a caixa na mão é sempre 'vista'; no real não, e a falha passa para o mapa e para o status |
| percepcao-3 | media | parcial | O erro de orientação do treino é independente por leitura e menor que o medido; o real é persistente |
| percepcao-4 | media | confirmado | O mapa do real tem outra distribuição: zona perto cheia, 20× pontos e o galpão em volta |
| percepcao-5 | media | confirmado | O alvo do BOTAR não tem fonte real nem erro no treino |
| percepcao-6 | media | confirmado | A odometria em tempo real não existe no robô; o treino a modela como passeio aleatório sem falha |
| percepcao-v1 | media | novo | O alvo do PEGAR é a verdade no treino; no real ele sai da mesma percepção que a caixa |
| percepcao-v2 | media | novo | A meia aresta chega exata ao ator, contra a decisão do dono de ±2 cm no tamanho |
| percepcao-v3 | media | novo | A latência fixa em 33 ms, com a caixa presa ao mundo, vira erro de posição proporcional à velocidade |

### Estado da arte

Veredito da frente: incerto (analista: viavel com ressalvas; viés da análise: otimista). Dificuldade 5/5. Sim-to-real 2/5.

| id | gravidade | veredito | achado |
|---|---|---|---|
| sota-1 | alta | parcial | Caixa de 5 kg só nas palmas do G1 fica na fronteira física: aperto, torque e calor |
| sota-2 | alta | confirmado | O método da v3 não tem precedente real: uma política PPO do zero, sem dado humano, para 4 tarefas mais terreno |
| sota-3 | alta | confirmado | No real, a maior parte das falhas vem da percepção e da localização, mesmo com marcador. A v3 deixa o estimador real de fora |
| sota-4 | media | parcial | Confiabilidade: 9 de 10 por cenário está no nível dos papers, mas o ciclo composto fica em 38–74%, e 99% está fora do estado da arte |
| sota-5 | media | confirmado | CARREGAR em terreno com caixa de até 5 kg vai além do que se mostrou no G1 |
| sota-v1 | alta | novo | Encadear tarefas desaba mesmo no sim: o método mais parecido com a v3 fecha 0/100 |
| sota-v2 | alta | novo | A v3 só roda num G1 no modo de 29 DoF com a cintura livre, e o repo não registra o modo do robô |
| sota-v3 | media | novo | O portão de aceite pede uma confiabilidade por teste acima da que o estado da arte mede no sim |
