# Viabilidade da implementação do treino v3

**Origem:** o dono pediu esta análise em 09/10. O escopo é só o treino em simulação (mjlab, Isaac Sim, Isaac Lab). O robô real e o sim2real ficam fora.

**Método:** 3 leitores fizeram o dossiê da spec, do código e do vault. 4 analistas geraram achados: A (coerência), B (física e gradiente), C (implementação e custo) e D (terreno, lidar e Isaac). Um verificador adversarial julgou cada achado contra o texto e o vault. Um round de medida (round 2) fechou os achados incertos e as lacunas do verificador. Todas as medidas rodaram em CPU. Nenhum arquivo do repo mudou, exceto este relatório.

**Regra de leitura:** o resultado do round 2 substitui o julgamento do verificador para o mesmo id. Os ids L1 a L8 são as lacunas do verificador, medidas no round 2. "Spec" é `specs/g1-v3.md`. "CV" é `specs/g1-v3-comportamento.md`. "Enunciado" é `specs/g1-limpo-comportamento.md` (vault/G1/Transversais/Enunciado do comportamento.md). Os scripts ficam em `/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/a13fa48a-b637-4102-bab7-af8ab9d263c3/scratchpad/viabilidade/` (abaixo, `viabilidade/`).

## 0. Veredito

A v3 é viável no mjlab com condições. Nenhuma Parte de 0 a 4 tem bloqueio de física ou de cinemática sem conserto conhecido. A Parte 5 é inviável nesta máquina. Quatro bloqueantes pedem decisão ou conserto antes da Parte 1: o CV com regras antigas (L3), a área do filtro do mapa (D5), a fluência da caixa na pega (L2) e o termo de altura do pé com o z do mundo (L7).

| Parte | Veredito | Motivo |
|---|---|---|
| 0 | viável com condição | Tudo roda em CPU. Os itens 3 e 15 precisam do teto real de aperto (B01), da régua por face (B03), da forma do `r_aperto` (B02), das desigualdades da escada (L4) e da laje de 0,30 m (B07). |
| 1 | viável com condição | O rough do mjlab já tem terreno e scan (D15). O termo de altura do pé da Unitree usa o z do mundo e morre no terreno (L7). O filtro de elevação não existe (D9). O CV contradiz a spec (L3). |
| 2 | viável com condição | A caixa de 5 kg desce 4 a 8 cm em 6 s entre as palmas, por fluência da física (L2). Com μ 0,3, 4 e 5 kg passam do torque do `wrist_yaw` (B01). A laje de 0,30 m com 5 kg não fecha as cercas com custo zero (B07). |
| 3 | viável com condição | O limite de chegada da caixa no BOTAR está pendente do dono (A10). O tempo real do BOTAR não tem medida, e o pior caso da C diverge entre spec e CV (A9, L8). |
| 3b | viável com condição | A laje e as terminações leem z contra a origem do subterreno, que vai a ±1,4 m (D3, D4). O currículo de terreno do mjlab não é por tarefa (D1). O preset da plataforma não é de 2 m (L5). A caixa grande deixa o mapa cego à frente (D5). |
| 4 | viável com condição | O `r_aperto` paga positivo por passo, contra a §9.2 (B02). O freio como catraca e o atraso da ação seguem sem decisão (L4). |
| 5 | inviável (nesta máquina) | Sem GPU NVIDIA e sem Isaac (A13, C10, D11). O `RayCaster` do Isaac Lab 2.3 não vê malha dinâmica (D12). A T4 não está confirmada para o Isaac Sim 5.1 (L6). |
| percepção | viável com condição | Com a caixa de 26 cm, o chão mais perto visto à frente fica a 1,17 m, fora do mapa 2×2 m (D5). Com R = 256, o mapa fica esparso a 1 m/s (D6). O custo do raycast em CPU é de 3 a 14% do passo (D7). |

## A. Bloqueantes

| id | Parte | Achado | Evidência | Proposta | Seção do enunciado |
|---|---|---|---|---|---|
| L3 | geral | O CV tem autoridade maior que a spec e ainda tem as regras antigas. Quem segue a ordem de autoridade implementa a regra antiga. | Spec:9-10 (autoridade). CV:175, 199, 331-332, 523 (faixa F_ref a k·F_ref) contra spec:390. CV:193 (apoio 1·m·g) contra spec:342. CV:350, 510 (aperto 8× → 2,5×) contra spec:470. CV:367, 507 (air_time). CV:43 (fase 0 por razão > 0,50) contra spec:575-576. CV:35 (lidar desligado) contra spec:84-85. CV:322 (C 15,7 s) contra spec:243 (17,2 s). | O dono sincroniza o CV com a spec antes da Parte 1, ou inverte a ordem de autoridade na spec:9-10. | §6 Decisões fechadas; §7 Como usar |
| D5 | percepção | Com a caixa de 26 cm, a faixa à frente (±30°) de 0,9 a 1,1 m tem 0% de visibilidade. O chão mais perto visto fica a 1,17 m. A borda do mapa fica a 1,0 m. O degrau à frente nunca entra na memória do filtro. | `viabilidade/D/sombra_faixa.py` (reproduzido pelo verificador). Caixa de 20 cm: 7 a 16%. Degrau de −0,15 m: 0% até 1,2 m. Spec:82, 91-92. Plano de percepção §0 (os 0,89 m são o mínimo em todos os azimutes). | O filtro guarda um mapa interno de ~5×5 m e entrega à política o recorte 21×21 de 2×2 m. A entrada fica com 882 canais. Alternativa do dono: tirar a caixa de 26 cm do nível TERRENO do CARREGAR. Decidir antes de escrever a fase 1 (D9). | §2 Cenário físico (terreno); §4.4 CARREGAR |
| L2 | 2 | Com a física do mjlab, a caixa de 5 kg desce entre as palmas por fluência. Com margem de atrito ≥ 1, ela desce 4,2 a 7,9 cm em 6 s (0,7 a 1,3 cm/s). Com a oscilação da marcha, 62 N com μ 0,4 e 82 N com μ 0,3 deixam a caixa cair. | `viabilidade/medida-resto/l2_creep.py`, `l2_var.py`, `l2_imp.py`, `l2_csv.py`. MuJoCo clássico e Warp CPU dão o mesmo número. `noslip_iterations` 10: 5,8 cm (não resolve). Pyramidal com impratio 10: 0,57 cm. Elliptic com impratio 10: 0,24 cm, mas divergiu para NaN em 15/07 (knobs.py:101). | O dono escolhe um de dois caminhos: (a) subir o impratio no cone pyramidal e medir a estabilidade no Warp; (b) medir a descida em relação às palmas e aceitar que a política compensa. O aceite "≤ 2 cm em 7 s" (relatório de 06/10 §4.3.1) depende dessa escolha. | §3 Invariantes (pega que não escorrega); §4.3 PEGAR; §4.4 CARREGAR |
| L7 | 1, 3b | O termo "altura do pé" da Unitree (+1,0) usa o z do pé no mundo, sem sensor de terreno. Na grade da v3 as plataformas vão de −1,2 a +1,2 m. Com o chão a 0,5 m, o termo vale 0,004 e tem derivada −0,08/m: canal morto. Ele paga até 1,0/s por ficar perto de z = 0 no mundo e desincentiva subir. | `viabilidade/medida-resto/l7_pe.py`, `l7_terreno.py`. Chão 0 m: r 0,975. Chão 0,2 m: r 0,342. Chão 1,0 m: r 0. unitree_rl_lab `mdp/rewards.py` (`foot_clearance_reward`, body_pos_w z). mjlab rewards.py:241-320 (altura sobre o terreno). Spec:451. | Trocar o z do mundo pela altura do `TerrainHeightSensor`, como o `foot_clearance` do mjlab já faz. Troca o sensor de um termo que já existe; 0 termos novos. Opcional (B11): padrão do sensor com grade até ~15 cm à frente do pé. | §2 Cenário físico (terreno); §4.1 ANDAR |
| D12 | 5 | O `RayCaster` do Isaac Lab 2.3.0 só vê malha estática. O lidar da v3 vê o corpo do robô e a caixa na mão. A Parte 5(a) como escrita não reproduz a sombra da caixa. | Doc isaaclab.sensors v2.3.0: "only static meshes are supported". O `MultiMeshRayCaster` não está na página da 2.3.0. unitree_rl_lab usa Isaac Lab 2.3.0. Spec:939. | Fixar na Parte 5(a) uma versão do Isaac Lab com malha dinâmica (≥ 3.0, a confirmar) ou um raycast próprio em Warp. | §2 Cenário físico (simulador) |
| A13 / C10 / D11 | 5 | A máquina não tem GPU NVIDIA nem Isaac Sim, Isaac Lab, unitree_rl_lab ou USD do G1. O Isaac Sim pede RTX. Os aceites da Parte 5 não se medem aqui. | `nvidia-smi` ausente; lspci: Intel Raptor Lake-U; 14 GB livres no disco. Spec:938-943, 829-832. Requisitos do Isaac Sim 5.1 (RTX 4080 mínima). | Tratar a Parte 5 como projeto separado, fora do orçamento de 30 000 its do mjlab. O dono decide a máquina (RTX alugada ou T4 confirmada, ver L6). Daqui só sai a parte pura: paridade de obs e ação, filtro e estimador. | §2 Cenário físico (simulador) |

## B. Riscos

**Coerência de documentos**

- **A1:** a spec contradiz o enunciado em 4 pontos já decididos pelo dono. Lajes de 0 a 0,55 m (enunciado:126) contra o mínimo de 0,04 m (spec:262-263). CoM sobre o meio dos pés (enunciado:163) contra o piso de 3 cm da borda (spec:504). Torque sem regra (enunciado:423-424) contra a cerca de 80% (spec:507). Afrouxar para soltar (enunciado:318) contra o `F_solta` removido (spec:392-404). O dono edita o enunciado agora (git diff, +17 linhas).
- **A3 / C12:** o enunciado proíbe mão no chão e toque na laje (enunciado:192, 195). A spec só cobra preço (spec:460-461). A §15 não tem aceite com número para isso.
- **A5:** os aceites "a calibrar" (spec:334, 769, 775), "frear em poucos passos" (spec:763) e a troca de tarefa sem tranco (enunciado:210) não têm número. O portão da Parte 1 é numérico. Só a "espera sem agachar" do portão da Parte 2 fica vaga. Os CSV de play não servem para tirar limiares de tempo, porque as fases são de roteiro.
- **A9 / L8:** o enunciado rejeita teto de tempo por tarefa (enunciado:177, 428). A spec tem episódio de 20 s (spec:235). O pior caso da C é 17,2 s na spec (spec:243) e 15,7 s no CV (CV:322). No C3 a caixa chega ao peito 0,5 a 2,2 s depois da abertura do PEGAR, abaixo dos 3,0 a 7,0 s da spec. O BOTAR não tem medida.
- **A10:** o limite de chegada da caixa no BOTAR está pendente do dono (inbox 06/10). A spec usa 1,98 → 0,99 m/s (spec:551, 624). A medida no 24800 dá 0,88 a 1,02 m/s. O impacto não tem termo; a terminação sozinha não dá gradiente.
- **A11:** a Parte 3b não tem aceite de qualidade do mapa na zona da sombra. O D5 mostra que o problema é maior que a frase "a memória cobre a sombra" (spec:91-92).
- **L4:** dos 5 itens "Bloqueia" e "Alta" do relatório de 06/10, a spec absorveu 1 (hip_yaw, pela cerca do giro, spec:459, 495, 502). Ficam abertos: as duas desigualdades da escada (§4.3.4; spec:560-563), o freio como catraca (§4.3.5; spec:605-606), o atraso da ação (§4.3.6; spec:702 contra dr-para-a-v3.md:165, 194) e a caixa pequena desde a it 0 (§4.3.9; spec:592, 693).

**Recompensa e física**

- **B01:** o aperto lateral máximo por palma é 62 N a 100% do torque e 50 N com a cerca de 80%. Quem limita é o `wrist_yaw` (5 N·m, braço de 0,080 m). A frase "82 N por palma contra ~100 N do braço" (spec:388-389) usa o ombro. Com μ 0,3: 4 kg pede 105% e 5 kg pede 132% do `wrist_yaw`. Caminhos permitidos: apoio no tronco (enunciado:198-199) ou contato perto do eixo do punho. Medida: `viabilidade/B/aperto_capacidade.py`, `aperto_detalhe.py`, `niveis.py`.
- **B02:** o `r_aperto = w·exp(−aperto/F_ref)` é sempre positivo e vale do P3 ao B2 (spec:369-375, 465). A §9.2 proíbe termo positivo por passo no PEGAR e no BOTAR (spec:442-445). Proposta sem termo novo: `−w·(1 − exp(−aperto/F_ref))`, com a mesma derivada. O termo só entra na Parte 4.
- **B03:** a spec põe o ponto de operação do aperto em 250 N (spec:371-372). Esse número é a força total robô → caixa. A régua da spec é o mínimo por face (spec:365-368), que fica em ~110 a 125 N com o pulso. Ali o termo vale ~0,23·w, e não 0,05·w. Medida: `viabilidade/B/faixa_aperto.py`.
- **B07:** round 2. Nas lajes de 0,04 e 0,15 m com 5 kg, as cercas fecham com a cintura a 80%, a pelve e o recuo dentro de ±8 cm. Na laje de 0,30 m, nenhuma pose da vizinhança da IK fecha com custo zero. Prendem o torque, o CoM e a folga canela/mão contra a laje. Medida: `viabilidade/medida-fisica/b07_ik_cercas.py`.
- **L1:** com o terreno da v3, cada mundo leva 1 410 geoms. O Model vai de 11,2 para 29,0 MB (fixo). O Data por mundo vai de 0,28 para 0,35 MB (+25%), ~1,43 GB a 4096 mundos. A VRAM não parece ser o limite. Medida: `viabilidade/medida-resto/l1_terreno.py`.

**Terreno e percepção**

- **D1 / C5:** o currículo de terreno do mjlab sobe pela distância andada (curriculums.py:48-54). O tipo do env fica fixo no init (terrain_entity.py:340-360). O nível é um por env, não por tarefa (terrain_entity.py:216-233). Proposta: um evento de reset próprio, o primeiro do dict, que escreve `terrain_levels`, `terrain_types` e `env_origins` pela fatia e pelo nível da tarefa. Estimativa não medida: 40 a 60 linhas.
- **D2 / L5:** escada e rampa têm o centro plano. O `random_rough` não tem plataforma (z de +0,02 a +0,10 m no centro). Obstáculos invadem a borda dos 2×2 m (até 0,056 e 0,122 m). Os presets do mjlab usam plataforma de 3,0 m (escadas) e 2,0 m (rampa), e não 1 m como diz a spec:696 (config.py:48-65, 77, 90). Medida: `viabilidade/D/plataforma.py`, `viabilidade/medida-resto/l7_terreno.py`.
- **D3:** a origem do subterreno fica em z +1,40 m (escada), −1,40 m (invertida) e ±0,74 m (rampa). `posiciona_cena` põe a laje em z absoluto, sem a origem (g1_limpo/eventos.py:170-184, 235-250). Conserto de 2 a 3 linhas na cópia, obrigatório antes da Parte 3b.
- **D4:** `caixa_largada` mede z da caixa contra a origem (g1_limpo/terminacoes.py:85-93). Descer a escada a partir do topo dispara a perda da caixa depois de 0,75 a 0,81 m de descida (5 a 6 degraus de 0,15 m). A altura da pelve e o joelho também leem contra a origem (terminacoes.py:150-155; recompensas.py:860, 1053, 1174). Proposta: altura contra o terreno pelo `TerrainHeightSensor`.
- **D6:** com R = 256 raios por leitura, a fração de células da faixa à frente com ≥ 1 leitura é 64%, 41% e 21% a 0,25, 0,5 e 1,0 m/s. Com o mapa interno de 5×5 m (D5): 99,5%, 97,5% e 84,5%. Com R = 1024 e 5×5 m: 100%. Medida: `viabilidade/D/densidade_mapa.py`, `densidade_5x5.py`.
- **D7:** em CPU, o `sense()` com 2025 raios custa ~10% do passo (16 envs) e 12 a 14% (64 envs). Ele roda a cada passo da política, sem `update_period` (mjlab sim/sim.py:522-560). O filtro de Kalman 21×21 custa 26 ms por passo com 4096 envs e P = 256 (torch CPU). A razão na T4 não está medida. Medida: `viabilidade/D/custo_ray.py`, `custo_kalman.py`.
- **D9:** o filtro de elevação (fase 1 do plano de percepção) não existe em código. `lidar_sim/percepcao/` tem só `eskf.py`, `fase.py`, `mapa.py` (voxel do Open3D) e `odometria_lidar.py`. A spec põe as fases 1 a 4 antes da Parte 1 (spec:918).
- **D10:** a borda da plataforma fica a 1,0 m do centro. O CARREGAR a 0,25 m/s por 4 a 7 s anda 1 a 1,75 m e entra 0 a 0,75 m no terreno. O giro no lugar nunca sai da plataforma. Proposta: o reset põe o robô a ~0,5 m da borda, de frente para o terreno.

**Custo e implementação**

- **C7:** round 2. Em CPU, o terreno da v3 custa 1,8 a 2,1 vezes o plano (env.step 178 a 180 ms contra 84 a 100 ms, 16 envs). O terreno é um modelo único e cobra os 1 410 geoms em todo env, inclusive no PEGAR e no BOTAR. Se a razão valer na GPU, os 8,54 s/it do Colab vão para 15 a 18 s/it. Não verificado em GPU. Os its por Parte são palpite. Medida: `viabilidade/medida-resto/l1_terreno.py`.
- **D13:** o unitree_rl_lab só tem terreno plano. O `height_scan` está comentado no ator e no crítico. Ele dá o USD, o PD e o esqueleto. A armadura de 0,01 por junta diverge da armadura refletida do mjlab (g1_constants.py:45-123); não verificado.
- **D14:** o aceite "até 10 pp do mjlab" (spec:829-832) não define nível, protocolo, número de episódios nem DR. Com sucesso por média móvel e balanceador (spec:582-583), dois treinos não são comparáveis. Proposta: protocolo de avaliação fixo na §15.
- **L6:** a T4 tem RT cores, mas não aparece na lista do Isaac Sim 5.1. Fontes não oficiais citam a T4 no GCP. O teste que decide é `isaac-sim.compatibility_check.sh --no-window` na instância. Não verificado.

## C. Medidas feitas nesta análise

Todas em CPU, com o Python do projeto, cada uma abaixo de 10 min. Caminhos relativos a `viabilidade/`.

| Medida | Script | Resultado |
|---|---|---|
| Capacidade de aperto por palma (B01) | `B/aperto_capacidade.py`, `B/aperto_detalhe.py`, `B/niveis.py` | `wrist_yaw` dτ/dN 0,080; 62 N a 100%, 50 N a 80%. `shoulder_yaw` satura a 96 a 103 N. Com μ 0,3: 4 kg 105%, 5 kg 132%. Com μ 0,5: 5 kg 79%. |
| Ponto de operação do aperto e faixa (B03, B04) | `B/faixa_aperto.py` | Força total p50 220 a 276 N; pads 25 a 56 N; aperto dos pads p50 24 a 38 N. `g_faixa = 0` em 74 a 100% dos passos de pegar e botar do C3. |
| Torque da cintura por nível (B07) | `B/niveis.py` | Cintura estática 42% (0,55 m; 1 kg) a 85 a 87% (0,04 m; 5 kg); 0,30 m com 5 kg chega a 94 a 97%. |
| IK com cercas e cintura ≤ 80% (B07, round 2) | `medida-fisica/b07_ik_cercas.py` | 0,04 e 0,15 m fecham (pelve 0 a 4,5 cm abaixo, recuo 0,8 a 3,0 cm menor). 0,30 m não fecha com custo zero: CoM 0,5 a 2,4 cm abaixo do piso ou mão a 1,0 a 1,3 cm. Newton local, 3 sementes. |
| Passo com a caixa (B10) | `B/passo_caixa.py` | Caixa de 26 cm: pé até 0,312 m sem toque; primeiro toque a 0,281 m (quadril −81°). Sem caixa: 0,592 m. O degrau de 0,15 m cabe. |
| Sensor de altura do pé na quina (B11, round 2) | `medida-fisica/b11_pe_geom.py`, `medida-fisica/b11_degrau.py` | Ponta do pé a 0,102 m do site. Com o anel de 3 cm há janela de colisão de 7 cm. O custo da trajetória segura é 0,013 a 0,022 por passo e pé, ~3 a 5% do rastreio. Mecanismo real, força pequena. |
| Termo de altura do pé da Unitree no terreno (L7) | `medida-resto/l7_pe.py`, `medida-resto/l7_terreno.py` | r = 0,975 (chão 0 m), 0,342 (0,2 m), 0,004 (0,5 m), 0 (1,0 m). Origens da grade de −1,2 a +1,2 m. |
| Plataforma e origem dos subterrenos (D2, D3, L5) | `D/plataforma.py`, `medida-resto/l7_terreno.py` | Escada e rampa planas nos 2×2 m. `random_rough` +0,02 a +0,10 m. Obstáculos até 0,056 m e caixas até 0,122 m na borda. Origem z ±1,40 m (escada), ±0,74 m (rampa). |
| Visibilidade à frente com a caixa (D5) | `D/sombra_faixa.py` | Caixa de 26 cm: 0% em 0,9 a 1,1 m; chão mais perto visto a 1,17 m; 99% só a partir de 2,1 m. Caixa de 20 cm: 7 a 16% em 0,9 a 1,1 m. |
| Densidade do mapa por R (D6) | `D/densidade_mapa.py`, `D/densidade_5x5.py` | Raios no mapa por leitura: 8,5 (R 128), 16,7 (R 256), 64,6 (R 1024). Cobertura da faixa com R 256: 64/41/21% a 0,25/0,5/1,0 m/s; com 5×5 m: 99,5/97,5/84,5%. |
| Custo do raycast e do Kalman (D7, D15) | `D/custo_ray.py`, `D/custo_kalman.py` | 256 raios: 3% do passo; 2025: 10%; 10000: 31% (16 envs). 64 envs, 2025 raios: 12 a 14%. Kalman 26 ms (P 256) e 88 ms (P 2048) com 4096 envs. Build do env com terreno: 2,4 a 3,2 s. |
| Custo do terreno e memória por mundo (C7, L1) | `medida-resto/l1_terreno.py` | Física 35,3 ms contra 16,7 a 19,7 ms por subpasso; env.step 178 a 180 ms contra 84 a 100 ms. Model 11,2 → 29,0 MB; Data 0,28 → 0,35 MB por mundo. |
| Fluência da caixa entre as palmas (L2) | `medida-resto/l2_creep.py`, `l2_var.py`, `l2_imp.py`, `l2_csv.py` | Margem ≥ 1: desce 4,2 a 7,9 cm em 6 s. Noslip e iterations 100 não resolvem. Impratio 10: 0,24 a 0,57 cm. Plays do C3: rel_z ≤ 1,3 cm em 7 s (compensação em malha fechada). |
| Tempo do PEGAR nos plays (A9, L8) | `medida-resto/a9_csv.py` | Caixa no peito 0,5 a 2,2 s depois da abertura do PEGAR. 24800_topo004 cai em 3,6 s. BOTAR sem medida: nenhum CSV fecha de pé na janela de 2 s. |
| Priority de contato do terreno (C4, verificador) | script não registrado | Geoms do terreno do mjlab com priority 0 e grupo 0. Os pés (priority 1) mandam no μ do par pé × terreno. |
| Grupos de geom do G1 (D8, verificador) | `D/custo_ray.py` (inspeção do modelo) | Grupo 2: 35 geoms visuais sem colisão. Grupo 3: 33 de colisão. `grupo_mobilia = 2` (g1_limpo/knobs.py:80). |

## D. Hipóteses ainda abertas

| id | O que medir | Quem roda |
|---|---|---|
| L8 / A9 | Tempo de fecho do P4 e do B4 com o model_24800, nas lajes de 0,04 a 0,55 m, com 1 e 5 kg. Cópia do `registra_juntas.py` que troca o trecho pelo fecho, não pelo relógio. Decide o teto de 20 s. | CPU, MuJoCo clássico. Pode rodar antes da Parte 3. |
| B07 | Busca global (mais sementes ou outro otimizador) na laje de 0,30 m com 5 kg. Decide se relaxar ali a cerca de torque ou a de CoM. | CPU, item 13 da Parte 0. |
| B01 | A pega com 4 e 5 kg e μ 0,3 fecha com a caixa apoiada no tronco? Medir o escorrego, o `wrist_yaw` em %, a força do tronco e os torques de ombro e cotovelo. | CPU, arnês do `segura_pose`. |
| B02 / L4 | Conta pura do valor descontado de pairar no B1 contra B_alvo e o potencial do B2 ao B4. As duas desigualdades da escada (§4.3.4 de 06/10). | CPU, item 3 da Parte 0. |
| L2 | Estabilidade do cone pyramidal com impratio 10 no Warp com muitos envs, e a descida com a política no laço. | Warp CPU dá o primeiro sinal; a escala e a política são do dono (GPU). |
| D4 | Expressão de `terminacoes.py:93` com a caixa na altura do peito sobre cada degrau, a partir do topo. | CPU. Barato; o conserto já é conhecido. |
| D7 / C7 / L1 | Razão de custo do terreno, do raycast (R 256 e 1024) e do Kalman no passo do treino; se a T4 aguenta 4096 envs com terreno e mapa. Fase 5c do plano de percepção. | Dono, GPU (Colab ou Kaggle). |
| C7 | s/it real da v3 e its por Parte. | Dono, treino. |
| D10 | Fração de passos de CARREGAR fora da plataforma no nível TERRENO. | Dono, smoke em GPU. |
| B10 | Contato coxa ou joelho × caixa na escada com a caixa de 26 cm, e a dinâmica do apoio simples com 5 kg. | Dono, play da Parte 3b. |
| D12 | Versão do Isaac Lab que tem o `MultiMeshRayCaster` com malha dinâmica. | Dono, máquina RTX. |
| L6 | `isaac-sim.compatibility_check.sh --no-window` na T4 do Colab ou da Kaggle; custo de uma RTX alugada para 30 000 its. | Dono. |
| D13 | Efeito da armadura de 0,01 contra a refletida do mjlab na resposta do PD. | Dono, máquina RTX. |

## E. Achados refutados

- **A2:** a rampa a 95% já penaliza o limite desde o começo; o `NoBatente` no refino é decisão do dono de 02/10 (spec:552); o item 7 já mediu a IK a ≤ 90%.
- **A7:** a spec já liga o mapa com o chão plano na fase 0; o terreno é nível posterior do ANDAR (spec:588).
- **A8:** as cercas em toda tarefa são decisão do dono de 08/10 (spec:495); o item 14 mediu custo zero no ANDAR (spec:901-904).
- **A12:** a pose do refino foi substituída pelas cercas em 08/10 (spec:627-633); só o peso da Sacudida segue aberto.
- **B04:** não é fato novo; o relatório de 06/10 §4.3.2 já propôs o gate mole. Os plays do C3 pegam de pé e não servem de proxy da v3. A pendência de 06/10 segue aberta.
- **B05:** o índice do alvo está no crítico (spec:152-154); o PBRS não muda as vantagens. Já refutado no relatório de 06/10 §7.
- **B06:** a spec já exige alive e altura do pé a zero depois da caixa perdida (spec:442-445, 556-558, 1008-1009). Vira teste no smoke da Parte 2.
- **B12:** refutado pela medida do D5; os 0,89 m do plano são o mínimo em todos os azimutes, não à frente.
- **C4:** o terreno do mjlab tem priority 0; o pé (priority 1) manda no μ (medida do verificador).
- **C6:** o vault tem s/it medido na Kaggle com 4096 envs (4,8 a 12,7 s/it); 30 000 its dão ~40 a 105 h. O orçamento é restrição conhecida.
- **C8:** a hipótese "Parte 1 com mapa perfeito" contraria a decisão do dono de 06/10 (o mesmo filtro no treino e no real, spec:80-82).
- **C2, C3, C9, C11:** sem fato novo; já registrados na revisão de 08/10 (§D, E7) e no relatório de 06/10 §4.4.1. As pendências seguem abertas. Para o C11, a spec ainda não tem o pré-aquecimento do mapa.
- **C1, C13:** a §17 exclui o smoke e as ferramentas por definição (spec:950-951) e será refeita na Parte 1.

## F. Ordem sugerida

1. **Documentos (dono):** sincronizar o CV com a spec (L3). Fechar as divergências com o enunciado (A1, A3, C12). Decidir o limite de chegada do BOTAR (A10).
2. **Física da pega (dono decide, CPU mede):** escolher o caminho da fluência (L2: impratio ou aceite relativo às palmas). Escolher o caminho do aperto com 4 a 5 kg (B01: tronco, piso de μ ou massa depois do μ). Corrigir spec:388-389.
3. **Parte 0 em CPU:** itens 10, 3, 1, 2, 4 e 15. Incluir a régua por face (B03), a forma negativa do `r_aperto` (B02), as desigualdades da escada (L4), a laje de 0,30 m (B07) e os limiares que faltam (A5).
4. **Percepção, fase 1:** decidir a área interna do filtro (D5) e escrever o filtro de elevação com ela (D9). Fixar R pela cobertura (D6). Seguir com as fases 2 a 4.
5. **Parte 1:** partir do `unitree_g1_rough_env_cfg` (D15). Trocar o z do mundo do termo de altura do pé pela altura do sensor de terreno (L7). Mudar o grupo da mobília para o raycast (D8). Medir a fase 5c em GPU: terreno, raycast e Kalman (D7, C7, L1).
6. **Parte 2 e Parte 3:** cadeia B e cadeia C no plano. Medir antes o tempo do BOTAR em CPU (L8).
7. **Parte 3b:** somar a origem ao z da laje e da caixa (D3). Ler alturas contra o terreno (D4). Evento de reset por tarefa (D1). Plataforma de 2 m posta à mão e colunas sem plataforma fora da fatia B (D2, L5). Reset perto da borda (D10).
8. **Parte 4:** freio e atraso da ação conforme a decisão do L4. `r_aperto` com o peso da Parte 0.
9. **Parte 5 (projeto separado):** confirmar a máquina (L6), a versão do Isaac Lab com malha dinâmica (D12), a armadura (D13) e o protocolo de avaliação fixo (D14).
