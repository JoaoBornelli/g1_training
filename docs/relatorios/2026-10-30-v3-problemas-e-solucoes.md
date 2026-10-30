# Problemas da implementação v3: o que foi achado, pesquisado e resolvido

Registro aberto em 30/10, por pedido do dono: cada problema achado na implementação vai a um agente que pesquisa na
internet a melhor solução; a solução é julgada contra o objetivo do treino (`specs/g1-limpo-comportamento.md`) e
aplicada. Este arquivo cresce a cada caso; o resumo final sai dele.

Formato de cada caso: o problema (onde e o efeito), o que foi pesquisado (fontes), o que foi considerado (opções e o
julgamento contra o objetivo), a solução aplicada (commit) e o que fica para a fase I.

## Casos

<!-- um bloco por caso, na ordem em que aparecem -->

### 1. Métricas de evento logam a média do passo, e não a do episódio (aberto)

- **Problema:** `g1_v3/metricas.py` (`tempo_no_alvo`, `phi_medio_do_alvo`, `velocidade_de_chegada`, `fracao_fora_da_plataforma`). O mjlab tira a média do valor no último passo sobre os envs que resetam; a correção da F9 devolve a média transversal dos envs com o evento NO PASSO. A revisão G7 mediu 0,054 logado contra 0,259 por episódio. As métricas da fase I (D10, pouso, tempo por alvo) sairiam erradas.
- **Pesquisa:** agente de pesquisa, na internet e no código instalado. Leu o `MetricsManager` do mjlab 1.5.3 e o do `main` (que só acrescenta `reduce="sum"`), o logger do rsl_rl 5.4.0, o `CommandTerm.reset` e o `TerminationManager` do Isaac Lab (issue #2977, PR #3107), e o `reset_idx` do legged_gym e do unitree_rl_gym. Nenhum exemplo oficial loga métrica condicional por máscara: todos tiram a média sobre os envs que resetam. A lição do Isaac Lab (#2977) é reportar o valor do episódio no reset. Conferido por mim no código: o rsl_rl concatena os tensores de cada chave e tira a média dos elementos (`rsl_rl/utils/logger.py:185-196`); o mjlab lê o `last` antes do `reset` do termo (`metrics_manager.py:122`, `:132`).
- **Considerado:**
  - (a) acumular soma e contagem e devolver a média acumulada: inexata, porque o termo não sabe quando a iteração fecha.
  - (b) soma e contagem separadas: exata só fora do manager, e o log mostra dois números crus.
  - (c) NaN com reduce que ignora NaN: o `torch.mean` do mjlab e o do rsl_rl propagam o NaN.
  - (d) o `reset` do termo escreve um vetor 1-D com os episódios que fecham e tiveram o evento em `extras["log"]["Episodio/<nome>"]`: exata, sem estado novo, e a chave some sem evento. O próprio mjlab já escreve direto em `extras["log"]` em `tasks/velocity/mdp/rewards.py`.
  - Contra o objetivo do treino: as métricas não entram na recompensa nem no currículo (o currículo usa o registro de sucesso da cadeia). A mudança corrige só o que a fase I mede: a D10, o pouso (P3 da F9) e o tempo por alvo.
- **Solução:** a opção (d), contrato no plano (`09688ff`). O código vem numa lane Haiku depois da lane 1 da G7.
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
- **Solução:** `a12bdf3` e o commit seguinte. O `--afasta` tem `Cena.estaciona_z` (20 m) como padrão, e no ANDAR a caixa sobe em z com a laje, apoiada nela, como o `eventos.estaciona_cena`. Teste: a caixa ia +20 m em x antes do conserto.
- **Fase I:** nada.

### 4. A perda da caixa é cobrada e termina um passo depois (aberto)

- **Problema:** a terminação `caixa_perdida` e a recompensa `perda_caixa` leem o `v3_perdeu_agora`, que a cadeia escreve no `compute` do comando. O mjlab roda o comando depois das terminações e das recompensas: as duas veem a perda do passo anterior (0,02 s de atraso). A docstring de `terminacoes.py:65` diz o contrário. Achado pela lane da s32.
- **Pesquisa:** agente de pesquisa (o padrão do Isaac Lab e do mjlab para termos que dependem do comando; o efeito de um passo de atraso no PPO).
- **Considerado / solução / fase I:** a preencher.

### 5. O `cena.mjb` do exportador pode não levar a DR da caixa (aberto)

- **Problema:** `exporta_cena.py` grava `env.sim.mj_model` (CPU). Os eventos `tamanho_caixa` e `carga` escrevem no modelo do Warp. A caixa do driver pode sair com a massa e o tamanho nominais. Achado pela lane 2 da G7.
- **Pesquisa:** o mesmo agente (o código do mjlab sobre `mj_model` × modelo do Warp).
- **Considerado / solução / fase I:** a preencher.
