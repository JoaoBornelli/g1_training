# Problemas da implementação v3: o que foi achado, pesquisado e resolvido

Registro aberto em 30/10, por pedido do dono: cada problema achado na implementação vai a um agente que pesquisa na
internet a melhor solução; a solução é julgada contra o objetivo do treino (`specs/g1-limpo-comportamento.md`) e
aplicada. Este arquivo cresce a cada caso; o resumo final sai dele.

Formato de cada caso: o problema (onde e o efeito), o que foi pesquisado (fontes), o que foi considerado (opções e o
julgamento contra o objetivo), a solução aplicada (commit) e o que fica para a fase I.

## Casos

<!-- um bloco por caso, na ordem em que aparecem -->

### 1. Métricas de evento logam a média do passo, e não a do episódio (resolvido)

- **Problema:** `g1_v3/metricas.py` (`tempo_no_alvo`, `phi_medio_do_alvo`, `velocidade_de_chegada`, `fracao_fora_da_plataforma`). O mjlab tira a média do valor no último passo sobre os envs que resetam; a correção da F9 devolve a média transversal dos envs com o evento NO PASSO. A revisão G7 mediu 0,054 logado contra 0,259 por episódio. As métricas da fase I (D10, pouso, tempo por alvo) sairiam erradas.
- **Pesquisa:** agente de pesquisa, na internet e no código instalado. Leu o `MetricsManager` do mjlab 1.5.3 e o do `main` (que só acrescenta `reduce="sum"`), o logger do rsl_rl 5.4.0, o `CommandTerm.reset` e o `TerminationManager` do Isaac Lab (issue #2977, PR #3107), e o `reset_idx` do legged_gym e do unitree_rl_gym. Nenhum exemplo oficial loga métrica condicional por máscara: todos tiram a média sobre os envs que resetam. A lição do Isaac Lab (#2977) é reportar o valor do episódio no reset. Conferido por mim no código: o rsl_rl concatena os tensores de cada chave e tira a média dos elementos (`rsl_rl/utils/logger.py:185-196`); o mjlab lê o `last` antes do `reset` do termo (`metrics_manager.py:122`, `:132`).
- **Considerado:**
  - (a) acumular soma e contagem e devolver a média acumulada: inexata, porque o termo não sabe quando a iteração fecha.
  - (b) soma e contagem separadas: exata só fora do manager, e o log mostra dois números crus.
  - (c) NaN com reduce que ignora NaN: o `torch.mean` do mjlab e o do rsl_rl propagam o NaN.
  - (d) o `reset` do termo escreve um vetor 1-D com os episódios que fecham e tiveram o evento em `extras["log"]["Episodio/<nome>"]`: exata, sem estado novo, e a chave some sem evento. O próprio mjlab já escreve direto em `extras["log"]` em `tasks/velocity/mdp/rewards.py`.
  - Contra o objetivo do treino: as métricas não entram na recompensa nem no currículo (o currículo usa o registro de sucesso da cadeia). A mudança corrige só o que a fase I mede: a D10, o pouso (P3 da F9) e o tempo por alvo.
- **Solução:** a opção (d): contrato no plano (`09688ff`), código em `8f4409d`. O `_media_dos_com_evento` saiu; entrou `_loga_episodios`. As cinco métricas de evento devolvem a flag no `__call__` e logam o valor do episódio no `reset`. Sete testes falharam antes do conserto e passam depois (397 testes, 1 pulado).
- **Fase I:** `Episode_Metrics/<nome>` passa a ser a fração de episódios com o evento; o valor condicional fica em `Episodio/<nome>`. Lacuna no TensorBoard numa iteração sem evento é o esperado. Risco medido só por estimativa: a máscara força uma sincronização GPU→CPU por termo e por reset (≤ 1–2 % do passo).

### 2. A s32 do smoke pode passar sem testar (resolvido)

- **Problema:** `g1_v3/smoke.py`, s32. Com `caixa_perdida_vira_andar` ligada, o termo `caixa_perdida` nunca dispara (`terminacoes.py:67`); e se o robô cai no passo, o env reinicia e o `v3_perdeu_agora` lido é o do estado novo.
- **Pesquisa:** só código (lógica interna do smoke; a internet não acrescenta). O `step` do mjlab (`envs/manager_based_rl_env.py:416-520`) roda física, terminações, recompensas, métricas, reset e só então o comando, e devolve `terminated` e `truncated` por mundo.
- **Considerado:** tirar o check do termo; ou exigir o passo sem reset e conferir o `v3_perdeu_agora`, que é a regra da altura contra o terreno local (viab. D4).
- **Solução:** `90f994b`. A s32 falha, com a mensagem do porquê, se algum mundo terminou ou truncou no passo. Ela confere o `v3_perdeu_agora`. O termo `caixa_perdida` só é conferido com a chave `caixa_perdida_vira_andar` desligada; ligada, a seção imprime que ele não se aplica.
- **Fase I:** nada a fazer além de rodar o smoke. A lane achou o caso 4.

### 3. O registro estaciona a laje e a caixa de outro jeito que o treino (resolvido)

- **Problema:** `registra_juntas_v3.py`, `limpa_a_cena` e `--afasta`. No ANDAR, o registro sobe a laje 10 m e anda a caixa 10 m em +x. O treino (`eventos.estaciona_cena`) sobe as duas `Cena.estaciona_z` = 20 m, com a caixa apoiada na laje. O mapa e a câmera do driver veem uma cena que o treino nunca mostrou à política.
- **Pesquisa:** só código: o contrato do driver (D11) é espelhar o treino, e o treino está em `eventos.estaciona_cena`.
- **Considerado:** manter os 10 m (sem efeito na marcha de 2 m, como a G7 M3 notou), ou espelhar o treino. Contra o objetivo: o driver mede a política no regime do treino; uma diferença de cena é uma variável a mais na fase I.
- **Solução:** `a12bdf3` (a laje em z) e `7fe6bd7` (a caixa em z e o padrão de 20 m). O `--afasta` tem `Cena.estaciona_z` (20 m) como padrão, e no ANDAR a caixa sobe em z com a laje, apoiada nela, como o `eventos.estaciona_cena`. Teste: a caixa ia +20 m em x antes do conserto.
- **Fase I:** nada.

### 4. A perda da caixa é cobrada e termina um passo depois (resolvido)

- **Problema:** a terminação `caixa_perdida` e a recompensa `perda_caixa` leem o `v3_perdeu_agora`, que a cadeia escreve no `compute` do comando. O mjlab roda o comando depois das terminações e das recompensas: as duas veem a perda do passo anterior (0,02 s de atraso). A docstring de `terminacoes.py:65` diz o contrário. Achado pela lane da s32.
- **Pesquisa:** agente de pesquisa. Fontes: o `manager_based_rl_env.py` do Isaac Lab (a mesma ordem do mjlab: terminação, recompensa, reset, comando, observação; "done after reset to get the correct observations"), o FAQ e a visão de arquitetura do mjlab (atraso consistente, tradeoff deliberado), a issue #998 do mjlab (raycast atrasado). Nenhuma fonte oficial trata termo que depende de estado escrito pelo comando, nem recomenda evitar o atraso.
- **Considerado:** (A1) aceitar e corrigir os textos; (A2) detectar a perda dentro da terminação e o comando ler o resultado (+10 linhas, efeito colateral num termo de terminação, zero ganho no ramo `caiu` por causa do raycast); (A3) evento de intervalo ou de passo (rodam depois do comando: pior). Contra o objetivo: a penalidade não depende da ação do passo do atraso; ela chega à ação causal pelo crítico com peso γ ≈ 0,99 (GAE, arXiv 1506.02438). A terminação está inerte no padrão (`caixa_perdida_vira_andar` = True).
- **Solução:** A1. Textos corrigidos em `18952aa` (`terminacoes.py`, `recompensas.py`) e no commit do caso 6 (`comando.py`). O texto de saída do `parte0/item3_escada.py` fica como registro histórico da Parte 0.
- **Fase I:** nada; a pesquisa achou o caso 6.

### 5. O `cena.mjb` do exportador pode não levar a DR da caixa (resolvido)

- **Problema:** `exporta_cena.py` grava `env.sim.mj_model` (CPU). Os eventos `tamanho_caixa` e `carga` escrevem no modelo do Warp. A caixa do driver pode sair com a massa e o tamanho nominais. Achado pela lane 2 da G7.
- **Pesquisa:** o mesmo agente, no código do mjlab: `sim.py:263-268` (o `mj_model` é o molde; `put_model` copia para o Warp), `sim.py:454-461` ("mj_model ... before any randomization"), `viewer/model_sync.py` (o mjlab copia campos do Warp para o `MjModel` só nos viewers), `rl/exporter_utils.py:61-62` (o exportador do mjlab lê o nominal). A doc de randomização confirma que a DR escreve só no `sim.model`.
- **Considerado:** (B1/B3) gravar o nominal e pôr o reset exportado na caixa nominal (+1 linha); (B2) copiar o sorteado para o `mjb` (~10 linhas, `geom_size`, `geom_rbound`, `geom_aabb`, `body_mass`, `body_inertia` e `mj_setConst`; o `mjb` passa a depender da semente). O defeito real: o `qpos_inicial` punha a caixa em `topo + a_sorteado` num `mjb` com a caixa de 0,10: no driver ela nascia enterrada ou flutuando até 3 cm. Contra o objetivo: o driver mede a política; uma caixa coerente com o próprio `mjb` é o que importa, e 0,10 está dentro da faixa do treino.
- **Solução:** B1/B3, `d4f97cd`: `env.v3_meia` = a meia nominal antes do reset exportado; o contrato do arquivo diz que o `mjb` é nominal, sem a DR do treino.
- **Fase I:** se o driver precisar da caixa pesada (nível > 0), gravar só a massa no `mjb`.

### 6. Perda falsa da caixa no primeiro passo do episódio (resolvido; achado pela pesquisa do caso 4)

- **Problema:** o raycast `altura_caixa` só recalcula no `sim.sense()`, que roda depois do comando (`raycast_sensor.py:771-816`; antes do primeiro `sense()` os dados são zero, `:549-556`). No primeiro `compute` de um episódio novo, a cadeia lê a altura da caixa do fim do episódio anterior. O ramo `caiu` da perda (`comando.py:55-61`) não exige a caixa pega e vale em toda tarefa que não é ANDAR. Um episódio que nasce em PEGAR, CARREGAR ou BOTAR pode marcar perda no passo 0: com `caixa_perdida_vira_andar` ligada, ele vira ANDAR e paga a `perda_caixa` (0,56 de retorno). Isso corromperia o treino da manipulação.
- **Pesquisa:** o código do mjlab (a ordem do `step` e o ciclo do raycast) e a issue #998 do mjlab, que trata o mesmo atraso do raycast.
- **Considerado:** (a) a perda não vale no passo do reset (`episode_length_buf == 0`), uma linha; (b) medir a altura da caixa por conta, sem raycast (perde a altura contra o terreno local da viab. D4); (c) chamar o `sense()` antes do comando (muda a ordem do mjlab). Contra o objetivo: nada se perde de verdade no passo do reset (a cena acabou de ser posta); (a) é exata e mínima.
- **Solução:** (a), no commit seguinte a `d4f97cd`: `_perda` ganhou `& (env.episode_length_buf > 0)`. Teste: com `episode_length_buf` 0 e a altura stale da caixa no chão, o código de antes marcava a perda; o de agora não (399 testes, 1 pulado).
- **Fase I:** no smoke, conferir no primeiro passo de um env reiniciado que `v3_perdeu` fica False; no log, a fração de episódios B e C que viram ANDAR no início deve cair a zero.

### 7. O contrato da fase H tem 24 problemas antes de existir código (em pesquisa)

- **Problema:** o rascunho do contrato da fase H (`docs/relatorios/2026-10-30-v3-fase-h-rascunho-do-contrato.md`, feito por um agente Opus só de leitura) achou 24 problemas entre o plano, a spec, o código e o mjlab. Exemplos: o `aperto` e a `sacudida` com peso positivo pagariam por apertar e por sacudir (P2); o `apply_body_impulse` do mjlab sorteia a força por componente, sem máscara, e derrubaria a caixa estacionada (P3); o `solref` na geom de menor prioridade não chega ao contato (P17); nenhuma gaussiana por eixo dá a mediana 5,3° e o p90 15,3° da orientação medida (P21).
- **Resolvidos sem pesquisa (organização):** P1, o `valor_do_degrau` da H1 é o `recompensas.por_degrau` que já existe: uma função só; P6, o `env_cfg.py` e o `observacoes.py` saem de lanes paralelas e ganham uma lane própria.
- **Pesquisa:** três agentes, por tema: recompensa e terminação (P2, P8–P14, P23); DR do refino (P3–P5, P15–P20, P24); estimador do refino (P7, P21, P22).
- **Considerado / solução / fase I:** a preencher, problema a problema.
