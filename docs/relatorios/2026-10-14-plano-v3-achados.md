# Plano de implementação da v3: achados e mudanças

Data: 2026-10-14. Plano: `docs/planos/2026-10-14-implementacao-g1-v3.md` (1 827 linhas).

## 0. Resumo

O plano está pronto para o despacho. Ele tem 9 fases (A a I), 70 tarefas, um contrato de
interfaces (§6) e 44 decisões de implementação (PD-1 a PD-44, §5) que a spec não fixa. Escrever e
revisar o plano expôs 31 problemas. O plano já resolve todos eles. Uma decisão do dono ainda fica em
aberto (§3 abaixo): a revisão das PD antes do despacho.

| fase | Parte da spec | tarefas | portão |
|---|---|---|---|
| A | preparação (knobs, tabelas de dados) | 6 | testes |
| B | Parte 0 (funções puras, itens 1, 2, 3, 4, 6, 8, 15) | 7 | o dono aceita os números |
| C | percepção, fase 1 (filtro de elevação) | 2 | testes |
| D | Parte 1 (pacote, ANDAR, lidar no treino, driver, notebook) | 16 | testes + revisão |
| E a H | Partes 2, 3, 3b, 4 | 39 | testes + revisão |
| I | smoke do dono, exportação, treino na Kaggle | — | S §15, nos marcos do currículo |

## 1. Decisões do dono nesta sessão

Todas estão na inbox do vault (entradas de 14/10, ainda sem commit, porque a inbox é WIP do dono):

1. O plano cobre as Partes 0 a 4 em detalhe de tarefa.
2. A Parte 0 roda inteira antes da Parte 1.
3. A percepção roda as fases 1 e 5 antes da Parte 1. As fases 2 a 4 e a 6 correm em paralelo.
4. A IK é fonte de referências macro, e não rastreio de pose.
5. O item 10 da Parte 0 sai (sem IK nova). P0, P1, P3, B1, B2 e B3 são alvos geométricos.
6. O smoke roda uma vez, no fim. Commits locais estão liberados.

Consequência da decisão 6: os portões de play das Partes 1 a 4 (S §16) passam para o treino. O
currículo abre o ANDAR, a cadeia B, a cadeia C e o refino nessa ordem, e o plano confere cada portão
da S §15 no marco em que o currículo abre a tarefa (plano §13.3).

## 2. Achados

### 2.1 Entre documentos

| # | achado | o que o plano faz |
|---|---|---|
| 1 | O bloqueante L3 da viabilidade (CV com regras antigas e autoridade maior que a spec) | conferido: o CV foi sincronizado em 14/10; nenhuma ação |
| 2 | "A única pose usada inteira é o `q_de_pé`" contradiz a S §9.2: o desvio dos braços, da cintura e do quadril no CARREGAR tem como alvo a pose de pé com caixa (dono, 05/10) | o plano segue a spec: as duas poses de pé são usadas inteiras; a entrada da inbox foi corrigida |
| 3 | A S §16 põe as fases 1 a 4 da percepção antes da Parte 1 | mudou pela decisão 3; registrado na inbox |
| 4 | O DR §3 tem valores que a S §12 trocou (atrito do pé 0,3–1,2, empurrão a cada 3–8 s, impratio 2, faixa que desce com sucesso < 0,50) | o plano usa a S (0,3–1,0; 5 s; impratio 1; nunca desce, PD-24) |

### 2.2 Técnicos, resolvidos no plano

| # | achado | evidência | solução |
|---|---|---|---|
| 5 | As tabelas de IK guardam o `hash_mjcf`, que é o sha256 do `cena.mjb` inteiro. A cena da v3 muda o PD, os sites e o grupo da mobília, e o hash muda junto | `g1_limpo/exporta_cena.py:93-94` | a v3 lê uma tabela macro derivada uma vez (`referencias.npz`); a guarda é a tolerância nas grandezas macro (tarefa A3) |
| 6 | O atraso de 0–1 passo "sorteado por episódio" (S §12) existe no mjlab sem API privada | `delay_buffer.py:247-258`: `update_period` enorme e fase igual sorteiam o lag no 1º passo depois do reset | PD-14 |
| 7 | O atraso maior por nível (0–2, 0–3) exige mudar o teto do buffer em tempo de execução | o `DelayBuffer` lê `max_lag` a cada sorteio | PD-28; o smoke confere (seções 8 e 24) |
| 8 | O padrão rotativo do MID-360 (fase 5a) parece viável sem fork do mjlab | `raycast_sensor.py:691-747`: `prepare_rays` roda em PyTorch fora do grafo e lê `_local_directions` a cada passo | seção 5 do smoke confirma |
| 9 | 1 024 raios num passo a cada 0,1 s custam o mesmo raycast em todo passo, porque o mjlab não tem `update_period` no sensor | `sensor/sim.py` (viabilidade D7) | lidar em fluxo: 205 raios por passo, 1 025 por 0,1 s, ~1/5 do custo (PD-5) |
| 10 | A deriva da odometria em "cm por m" mistura a escala do passo; ela cresce com a raiz da distância | — | `deriva` em m/√m; 0,0075 dá 1,5 cm em 4 m, o ATE do ESKF (PD-18, D2b) |
| 11 | O máximo dos z por célula sobe o chão em ~+2 cm com 5 pontos e ruído de 2 cm | conta: E[máx de 5 normais] = 1,16 σ | média por célula, com variância σ²/n (PD-42) |
| 12 | O lidar da cabeça não vê o chão a menos de ~0,9 m. O teste do mapa "parado, ≥ 60% da frente vista" ia falhar | I §2.4 | o smoke move a base 1,0 m e pede ≥ 90% (I §2.4 mediu 205 de 210) |
| 13 | Queda livre de 20 cm dá 1,98 m/s, e não 0,99 | √(2·9,81·0,20) | o teste da chegada usa 5 cm (0,99 m/s, não termina) e 10 cm (1,40 m/s, termina) |
| 14 | A cerca da pelve com a referência da pega cobraria o robô por ficar de pé no P4 | S §9.3 (pelve ±8 cm da IK da pega) | referência macro por alvo: P0–P3 a pega, P4 a pose de pé com caixa, B1–B3 a pega de destino, B4 de pé (E4) |
| 15 | O potencial perde `−Φ` no passo em que a tarefa troca, na hora em que o controlador decide, e não o robô | PBRS com Φ → 0 no fim | r = 0 nesse passo (PD-36) |
| 16 | A `CarregarComQualidade` multiplicando o rastreio negativo apaga o custo da deriva com a caixa caída | — | a qualidade multiplica só a parte positiva (PD-37) |
| 17 | A cerca do CoM no apoio simples da marcha cobra sempre (o CoM sai do pé de apoio) | a medida da Parte 0 foi só com os dois pés: `g1_limpo/ik/cercas_nos_plays.py:76` | só com os dois pés no chão (PD-35) |
| 18 | O passo de lado receberia a largura da base (0,23 m) de graça no termo do passo | S §19, conflito 6 | desconto da projeção na pose de pé (PD-16) |
| 19 | Com uma célula por (tipo, nível), o piso 0,10 do balanceador estoura com mais de 10 células | n·piso > 1 na fórmula da S §11.1 | uma célula por tipo; o nível sobe por tarefa (PD-12) |
| 20 | `dr.body_com_offset` corrompeu a heap no `g1_limpo` | `g1_limpo/env_cfg.py:386`; commit `af77cf4` | o CoM da caixa fica atrás de uma chave desligada; a seção 37 do smoke decide (PD-38) |
| 21 | O mjlab 1.5.3 não tem DR de gravidade nem de `solref` | `envs/mdp/dr/` | gravidade sai (o viés do IMU cobre o efeito, PD-40); `solref` por evento próprio (H4) |
| 22 | O termo de altura do pé da Unitree mede a origem do `ankle_roll_link`, e não o site do pé | `foot_clearance_reward` do unitree_rl_lab | o sensor de terreno fica no corpo do tornozelo, alvo 0,10 m (D1) |
| 23 | O sim do real e o robô carregam os módulos puros por caminho, e um import entre eles quebra a carga | PERC §2 | os módulos puros não importam uns aos outros (PD-43) |
| 24 | O `Mapa` e a `Caixa` dariam cada um o passo da odometria, e a deriva dobraria | — | o `CadeiaCmd` é o dono do passo (PD-44) |
| 25 | O `pytest` não está no venv | `ModuleNotFoundError` | testes com `unittest` (PD-2) |
| 26 | `afasta_cena` não tem chamador no `g1_limpo`, mas a v3 precisa tirar a laje e a caixa da cena na fatia ANDAR | `g1_limpo/eventos.py:214-253` | copiada como `estaciona_cena`, 20 m acima da origem (E1) |
| 27 | `NoBatente` nunca é construído no `g1_limpo` | `g1_limpo/env_cfg.py` não o referencia | a v3 o liga no refino (H3) |
| 28 | O `g1_limpo` herda do molde DR com valores fora da S §12 (`push_robot` 1–3 s, atrito 0,3–1,2) | `velocity_env_cfg.py:217-258` | a v3 escreve cada evento com o valor da S (D7) |
| 29 | O viés por episódio do IMU (DR §5, no portão do ANDAR) e os eventos do refino não tinham tarefa nem lugar no contrato | — | evento `vies_imu` (E1), termos `giro_base` e `gravidade` (E8), eventos `pd`, `forca_motor`, `solref`, `com_caixa`, `rfi` (H4) na §6.5 |
| 30 | A barra vertical do valor absoluto dentro de células quebrava 9 linhas de tabela em markdown | — | trocada por `abs(x)` |
| 31 | Duas referências cruzadas apontavam para seções que não existem (§13.1, §10.2) | — | corrigidas |

## 3. O que o dono revisa antes do despacho

As 44 decisões da §5 do plano. As que mudam número de treino ou o modo de trabalhar:

| PD | ponto | por que revisar |
|---|---|---|
| PD-3 | o agente roda FK sem física (MuJoCo clássico, segundos) nos geradores de dados e na Parte 0 | a regra "sem simulador na sua máquina" (memória `feedback-quem-executa`); se você vetar, roda três comandos |
| PD-5 | lidar em fluxo, 205 raios por passo | troca a forma da D6 (1 024 por leitura) mantendo a densidade |
| PD-12 | células do balanceador por tipo, nível por tarefa | interpreta a S §11.2 |
| PD-15 | limiares do sucesso com canal zero (0,10 m; 0,25 rad) | número novo |
| PD-22 | o notebook clona a branch do GitHub | exige o seu push antes do run |
| PD-30 a PD-34 | regras que transformam as medidas da Parte 0 em pesos (cercas a 50% do ganho do Φ; `B_alvo` = 1,5 × reguladores; peso do progresso; perda = 1,2 × a desigualdade; sacudida a 10% do rastreio) | a S manda "medir", sem a regra; os números finais passam pelo seu portão na fase B |
| PD-38 | CoM da caixa desligado até o smoke | DR §3.3 pede essa DR |
| PD-40 | gravidade fora da DR | DR §3.7 pede essa DR |
| PD-41 | atraso global, DR do refino pela tarefa principal da fatia | o refino é por tarefa, e o buffer de atraso é global |

## 4. Riscos que ficam

| risco | tamanho | onde o plano trata |
|---|---|---|
| a cota de GPU da Kaggle (cerca de 30 h por semana; conferir) leva os 30 000 its a ~4 semanas, a 12–15 s/it | alto para o prazo | plano §15; o Colab roda 8 192 envs |
| o smoke só no fim junta os defeitos de integração de cinco Partes | médio | `test_cfg` em toda fase; uma rodada de correção reservada na fase I |
| a lógica da cadeia concentra no `comando.py` (lição da F4 do `g1_limpo`: sete defeitos, cinco silenciosos) | médio | núcleos puros testados e revisão do Opus por Parte |
| duas APIs privadas do mjlab (`_local_directions`, `max_lag`) | baixo | seções do smoke; plano B na PERC §5 |
| a contagem fica em ~6 550 linhas, acima da meta de ~5 000 da S §17 | baixo | a meta é de 02/10 e não conta o filtro, o estimador, as cercas e a observação nova (plano §16) |

## 5. Arquivos tocados nesta sessão

| arquivo | mudança | commit |
|---|---|---|
| `docs/planos/2026-10-14-implementacao-g1-v3.md` | novo | sim |
| `docs/relatorios/2026-10-14-plano-v3-achados.md` | novo (este) | sim |
| `vault/_sistema/inbox.md` | 4 entradas de 14/10 (escopo, IK macro, item 10, correção da pose de pé com caixa) | não: o arquivo já era WIP seu no começo da sessão |
| `~/.claude/memory/g1-v3-plano-de-implementacao.md` | novo | fora do repo |

## 6. Próximo passo

1. Você revisa as PD da §3 e responde só as que quer trocar.
2. O despacho começa pela fase A (A1, depois A6, depois A2 a A5 em paralelo).
3. A fase B termina no seu portão: os números da Parte 0 entram no `knobs.py` só com o seu OK.
