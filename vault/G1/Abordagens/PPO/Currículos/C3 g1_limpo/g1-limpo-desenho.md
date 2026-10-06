---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1_limpo é o módulo ativo do G1 — o desenho vive na spec e no plano do repo; aqui ficam as âncoras que não se derivam do código"
---
# g1_limpo: âncoras de desenho e estado das fases F1 a F6

Corpo movido sem edição da memória `g1-limpo-desenho` em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

`g1_limpo` é o módulo ativo do treino do G1 (branch `exp/g1-limpo`, verificado 2026-08-26). O desenho vive no repo, não aqui:

- `g1_training/specs/g1-limpo.md`
- `g1_training/docs/planos/2026-08-25-g1-limpo.md`

Âncoras do desenho que não se derivam de ler o código:

- Máquina de ELO, teto de dois elos por cadeia. As tarefas trocam DENTRO do episódio para treinar as transições. São cinco estados de um objetivo, não cinco tarefas.
- Um alvo, duas metades de referencial: `peito_b = (0,25, 0,00, 0,15)` no frame da base para x e y, e `altura_carregar = 0,95` ABSOLUTO em mundo para z. O z absoluto impede o robô de satisfazer o alvo agachando.
- `pegar` e `carregar` pedem o MESMO alvo. O que impede andar no `pegar` é o comando de velocidade em zero, não a forma do alvo.
- No `reorientar` a face alvo é CONSTANTE e marcada por uma placa inerte (`contype = conaffinity = density = 0`, massa bit-idêntica). A dificuldade está na orientação de nascimento da caixa.
- Nenhuma fase de treino começa antes de `inspeciona --tabela` da fase anterior passar sem falha.

Estado 2026-08-26: **as SETE fases (F0 a F6) implementadas**, e um code review depois achou 8 defeitos (dois deles quebravam o treino). `smoke` **301 ok / 0**, `paridade` 85 / 0, `inspeciona --tabela` 0 falhas, `leitura --demo` ok). Nenhuma run rodada ainda; o portao de treino e' remoto.

F1 (locomoção pura) em quatro peças, e a fundação do fabricante ficou intacta:

- `ELO_DE_TREINO = ANDAR` em `env_cfg.py` — constante de FASE, não knob. A F3 troca para `PEGAR`.
- A tabela de pesos divergE do molde em **dois nomes só**: `terminacao = −200` e `joint_acc = −2,5e−7`. Todo o resto é o valor do fabricante, escrito explícito no `knobs.Recompensa` como trava contra deriva de upgrade.
- `metricas.py`: as cinco métricas de marcha saíram de dentro dos termos de recompensa e viraram `MetricsTermCfg`, porque `reward_manager.py:122` pula termo com peso 0 — e o `air_time` do molde já vem 0,0, logo o `Metrics/air_time_mean` do fabricante NÃO EXISTE no painel de quem roda a receita dele.
- `razao_marcha` mora no próprio termo de twist (`TwistComRazaoDeMarcha`), com as somas em `self.metrics` para o `reset` do comando as zerar depois do currículo.

F2 (one-hot e gates), e ela mudou duas decisoes do plano por medicao:

- O **gate dos `track_*` NAO entrou.** Com o twist forcado a zero nos elos parados, gatear removeria a unica coisa que paga ficar parado. O preco passou a ser medido: **3,82/s** de piso num elo parado (robo travado), 5,82/s no env, contra ~12,5/s de teto dos sete incentivos da F3. Piso, nao concorrente.
- O **4o regime de sigma da postura NAO existe.** Ver g1-postura-penhasco-std-standing para a tabela que o derrubou. O termo ficou neutro.
- `fatia_loco = 0,95` e nunca 1,00, e o motivo e' o normalizador sem clamp, nao o treino.
- O **sorteio de elo mora no CURRICULO**, por env, porque evento e comando os dois o leem.
- `CARREGAR` e `BOTAR` nao sao sorteaveis: comecam com a caixa nas maos e so existem como 2o elo de cadeia (F4). Os slots 3 e 4 do one-hot ficam constantes ate la -- declarado, com mitigacao pre-registrada.

F3 (os sete incentivos), e tres formas que sao desenho e nao gosto:

- `staged = alcancar x (1 + trazer)`, PRODUTO e nao soma. Com soma, o robo ganharia por EMPURRAR a caixa ate o alvo com o pe.
- `squeeze` usa `min(F_esq, F_dir)`, nao soma: uma palma sozinha EMPURRA a caixa, nao a segura.
- `postura_ereta` multiplica rampa x preensao x descarga. Somada, o robo colheria a rampa so por ficar de pe sem tocar a caixa. A rampa tem clamp nos DOIS lados -- sem o superior, esticar-se alem do alvo pagaria mais e ele ficaria na ponta dos pes.

O `precise_pos` e o unico com sigma FIXO, de proposito: ele responde "a caixa esta NO alvo" (aceite), nao "estou me aproximando" (rampa). Duas perguntas, dois termos. Ver g1-sigma-e-a-distancia-inicial.

Os 8 canais da caixa entram na obs no frame da BASE. Mundo carregaria a origem do env (4096 deslocamentos diferentes) e o rumo.

⚠ Armadilha de medicao: o `trava_robo` pina as juntas em `default_joint_pos` a cada passo, portanto **nao se mede curva de recompensa movendo o braco** -- ele nao acumula deslocamento (2 cm em qualquer amplitude). Move-se a CAIXA; o kernel depende de ||palma - caixa||, nao de quem se moveu.

F4 (a maquina de elo), e o invariante que quase se perdeu:

- **Quem decide loco x manipulacao e o CURRICULO (a fatia da F2). A cadeia so escolhe QUAL transicao praticar, e ela tem de COMECAR no elo que o curriculo sorteou.** A primeira implementacao inverteu: sorteava a cadeia e sobrescrevia o elo com o 1o dela. Como tres das quatro cadeias comecam no PEGAR, TODOS os envs viravam PEGAR e a fatia de 95% era apagada -- sem log. Medido depois do conserto: 0,9414.
- `ANDAR` recebe `CADEIA_NENHUMA = -1`, e isso precisa de guarda explicito: `CADEIAS[-1]` em Python le a ULTIMA cadeia.
- O avanco roda no `_update_command` (pose fresca), nunca no reset. Os sigma sao recalculados a cada avanco.
- O invariante da laje do BOTAR **tem de ser medido no MESMO instante**: no pos-avanco nada segura a caixa e ela cai, entao topo em t contra fundo em t+10 acusa violacao que nao existe.
- Ler `.data` depois de `write_mocap_pose_to_sim` sem dar um passo devolve o buffer VELHO -- a mesma armadilha do `_pendente`, agora na LEITURA. Foi o que fez o inspetor acusar "a laje nao subiu" com o topo antigo.

F5 (o balanco de forma) -- o mecanismo central, e ele tem duas partes que nao se confundem:

- **`alvo_loco` e fatia de TRANSICOES; `sorteio` e probabilidade por EPISODIO.** A conversao e `f = alvo*Tm / (Tl*(1-alvo) + alvo*Tm)`, com Tl/Tm = EMAs das duracoes medidas. Com Tl=24 e Tm=961 um sorteio de 0,30 entrega **1,06%** do gradiente -- erro de 40x, e foi o que o g1_poc fez. A funcao `resolve_sorteio` e PURA para poder ser testada contra a tabela da spec sem GPU.
- **Estado inicial ASSIMETRICO:** duracoes NEUTRAS (governam a fatia; erro ali desafina o sorteio por ~tau), `razao_marcha` PESSIMISTA em 0,0 (governa o PORTAO; um portao que nasce aprovando entrega a locomocao antes de existir marcha).
- Um sinal so no portao. Histerese assimetrica. Carencia contada de quando o BALANCO comecou.

⚠ **A ORDEM DO CURRICULO e `command_vel -> forma -> nivel -> elo`,** e isso foi conserto de correcao no meu proprio codigo da F2. O `forma` e o `nivel` medem o episodio que ACABOU e leem `limpo_elo`; o `elo` escreve o do episodio que COMECA. Na F2 o `elo` vinha primeiro e nao havia problema porque ninguem lia o elo antigo.

O estado do curriculo vai para o checkpoint via `runner_cls` (caminho suportado do `register_mjlab_task`), porque Colab/Kaggle matam sessao e a rampa de ~400 iteracoes seria re-paga. Vao as EMAs, a fatia, a carencia, e o nivel e o elo por env. NAO vao os buffers de episodio (cadeia, passo, sustain, sigma).

F6 (o passeio de nivel): sobe com sucesso de cadeia, desce sem. O ponto fixo e p=0,5 por CONSTRUCAO -- nenhum limiar a mao (o g1_multitask tinha `limiar_competencia = 0,90`). **So episodio de CADEIA move o nivel:** um de locomocao nao tem o que fechar e "fracassaria" sempre, empurrando o nivel ao piso com 95% dos envs.

⚠ Detalhe de teste: com p=0,5 o passeio +-1 e uma caminhada SEM VIES -- ele DIFUNDE, nao converge. Testar "estacionaria em X" falha por difusao e acusa o codigo. O invariante e nao colar nos extremos.

Relacionado: g1-sigma-e-a-distancia-inicial, feedback-subagentes, g1-leitura-log-e-kaggle, g1-mjlab-api-armadilhas, g1-mjlab-reset-buffers-obsoletos, g1-air-time-o-incentivo-desligado, g1-multitask-curriculum-design

⚠ **Os dois defeitos que o code review achou e que quebravam o treino** (2026-08-26):

1. **A carencia e a rampa da F5 contavam PASSOS, nao iteracoes** — erro de 24x, e ele fazia o modulo cometer a falha que ele existe para evitar. Ver g1-mjlab-api-armadilhas.
2. **O `CARREGAR` fechava com o robo PARADO.** O `pegar` e o `carregar` pedem o MESMO alvo (decisao da spec), e as condicoes eram `perto & alinhado & de_pe` contra **`perto` sozinho** — subconjunto sobre alvo que nao muda. No instante em que o pegar fechava, o carregar ja estava satisfeito: 1,5 s parado e a cadeia marcada como SUCESSO, movendo o curriculo de nivel. Agora exige `perto & ANDOU`, com 0,50 m derivados do portao da F1.

E o cronometro do termo `sustentacao` nao zerava no avanco de elo, entao com alvos identicos ele chegava saturado ao carregar — o elo nascia pago.

**Licao de metodo:** condicao de fechamento de um elo tem de ser testada contra a do elo ANTERIOR. Se ela e subconjunto, a transicao e automatica e o curriculo avanca sobre nada.
