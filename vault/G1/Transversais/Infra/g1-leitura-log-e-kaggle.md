---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "Como ler o log do mjlab sem inverter o diagnóstico, e as três armadilhas operacionais do treino no Kaggle"
cluster: transversais
---
# g1-leitura-log-e-kaggle (original)

Corpo original da memória `g1-leitura-log-e-kaggle`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Normalize antes de comparar qualquer número do log do mjlab.

- `Episode_Reward/x` divide por `max_episode_length_s`, não por passos. Então `func = Episode_Reward × 20 / (duração_real_s × peso)`. Errar dá fator 20 e inverte o diagnóstico.
- `Mean reward` não tem essa divisão: `Mean reward = líquido × 20`.
- `Episode_Termination/*` é CONTAGEM de envs, não fração. Divida pelo total.
- Somas por episódio crescem com a duração. Compare a razão contra o crescimento do episódio, nunca o valor cru.
- `Episode_Reward/*` é PONDERADO (peso × média do indicador), não a taxa crua. Divida pelo peso do termo para ter o indicador. Isso já me fez diagnosticar "preensão presa em 47%" quando ela estava em 94%.
- Um critério de sucesso que uma perturbação externa pode zerar não mede competência. O `push_robot` resetava o gatilho e produziu `perf = 0` por 8000 iterações com marcha existente.
- Não julgue um resume antes de +200 iterações. O transiente dura 120 a 150, e o fundo do poço em +90 engana como morte. A profundidade é proporcional ao número de mudanças simultâneas.
- Não meça suavidade pelas métricas de ação do TREINO quando o `std` varia: elas são sobre ações AMOSTRADAS, então um `std` maior as infla sem a política média ficar mais brusca. A régua de suavidade é o `play` determinístico. Por isso `entropy_coef` é o knob errado num run de suavidade.
- Métrica de treino com entropia alta não mede a capacidade da política determinística.
- **Régua de rastreio não informa antes de a duração crescer.** O `razao_marcha` só é número de rastreio depois de o episódio passar de ~150 passos (3 s). Com 21 passos o robô não tem tempo físico de acelerar, e a velocidade xy é dominada pela queda. A ordem de leitura é DURAÇÃO primeiro, razão depois. Verificado 2026-08-27 no bloco 1 do g1_limpo: razão −0,08 com episódio de 0,43 s.
- **Recompensa multiplicada por contato loga zero exato, e isso não é defeito.** `postura_ereta`, `squeeze` e `sustentacao` do g1_limpo multiplicam por `tanh(força_da_palma/12)`. Sem toque o produto é 0,0000. O `unload` fica diferente de zero porque lê o sensor de apoio.
- **`tempo_de_voo` > 0 é degrau INÚTIL, e `razao_marcha` precisa virar m/s antes de julgar.** O `tempo_de_voo` é média CONDICIONAL (só dos pés no ar): perder contato por UM passo de controle já o torna positivo. Medido na it 500 do g1_limpo: 0,0227 s = 1,1 passo, contra 200 a 400 ms de um balanço real. Degrau útil é ≥ 0,10 s. E converta a razão: com comando médio de 0,765 m/s, razão 0,085 são 0,071 m/s de progresso útil — o portão de 0,50 exige 0,38 m/s.
- **`razao_marcha` cruzando zero NÃO prova passo.** Arrastar cruza também: o corpo desliza a ~0,24 m/s sem o pé sair do chão. O discriminador é `Episode_Metrics/pico_de_altura` com `Metrics/slip_velocity_mean`. Medido 2026-08-27 na it 380 do g1_limpo: razão +0,004, pico 0,0097 m (CAIU de 0,0165), escorrego 0,240, `tempo_de_voo` plano. Zero significa só "tão bom quanto estátua" — erro em 97,5% do comando.
- **Os termos da pega do g1_limpo PARTILHAM fatores, e isso deixa resolver as componentes.** `squeeze = preensao`, `unload = descarga × preensao`, `postura_ereta = rampa × descarga × preensao` (todos com os mesmos gates). Portanto `descarga = unload/squeeze` e `rampa = postura_ereta/unload`, em unidades de indicador. As razões são médias CONDICIONAIS (ponderadas pelo fator partilhado), não médias simples — é a leitura certa mesmo assim, porque a pergunta é sobre os passos em que ele está agarrando. Medido na 1503 da bloco8: preensao 0,843, descarga 0,933, rampa 0,718.
- **`unload` lê FORÇA de apoio, não altura — mas o contato é rígido, logo ele é altura na prática.** `descarga = clamp(1 − F_apoio/m·g)`, e medido em 2026-08-27 ela salta de 0 a 1 em **2 mm** de erguida. Portanto `descarga` alta PROVA caixa fora da mesa; ela não pode ser colhida apertando contra a laje. Eu afirmei o contrário em 2026-09-04 e estava errado: o `unload` não financia o não-erguer. Verificado 2026-09-04.
- **Três canais da manipulação do g1_limpo são FALSOS enquanto `reorientar_inerte = True`.** O alvo publicado no `REORIENTAR` é a própria caixa (`comando.py:53`), portanto `d_caixa_alvo = 0` e o elo fecha em `sustenta_outros_s = 0,3 s`. (a) `Metrics/alvo_caixa/avancos` e `fatia_cadeia` contam só essa janela: previsto `0,5 × fatia_manip` e medido bate em 5% em três iterações. (b) `Episode_Reward/precise_pos` lê 1,0 exato durante os 0,3 s, o que dá um PISO de `0,5 × 0,3/dur_episodio` — 0,8% num episódio de 18,8 s. Ele leria 0,8% com a caixa colada na mesa; não tem resolução abaixo disso. (c) `staged = alcancar × (1 + trazer)` é o único canal com a distância caixa-alvo, e o log não publica o `alcancar` separado: dois desconhecidos num número. A âncora é `σ_trazer = d₀` (`sigma_fator = 1,0`), logo na abertura do elo `trazer = e⁻¹ = 0,368` e o `staged` com as mãos na caixa vale 1,368 — acima disso a caixa está mais perto que na abertura, mas quanto NÃO se lê. Verificado 2026-09-04.
- **A régua da caixa que falta é um `MetricsTermCfg`.** `d_alvo/σ_trazer` com `reduce="last"` sobre um mínimo corrente dá a melhor aproximação do episódio, exata. `reduce="mean"` sobre extremo monótono enviesa — é o defeito que o `impacto_da_caixa` tem hoje.
- **Fatia de locomoção: `Curriculum/elo` é a probabilidade POR EPISÓDIO, não a fatia de transições.** Os dois divergem quando as durações divergem. `Curriculum/elo` de 0,63 com alvo 0,95 é o `resolve_sorteio` funcionando, e exige episódio de loco ~11× o de manipulação.

Ferramenta (2026-08-26): `python g1_limpo/leitura.py` desfaz a diluição e imprime a escada de corte. Rode como ARQUIVO, não com `-m` — o `-m` importa o `__init__` e exige mjlab. `--demo` é o autoteste da aritmética, sem log e sem GPU.

⚠ O `Episode_Metrics/*` (o `MetricsManager`) NÃO tem a diluição: ele divide por `step_count`, então métrica em [0,1] fica em [0,1]. A correção é só da recompensa; aplicá-la à métrica erra na outra direção.

**How to apply — Kaggle:**

- **A versão que roda é `mjlab==1.5.3` com `rsl-rl-lib==5.4.0`** (verificado 2026-09-04). Nunca pine 1.5.1: o `RslRlModelCfg` dela manda `cnn_cfg`/`rnn_type` que o `MLPModel` do rsl-rl rejeita. O `g1_multitask/kaggle/requirements.txt` ainda pede 1.5.1 e 3.10.0.2 — ele está VELHO, não o copie. Pins de hoje: mjlab 1.5.3, mujoco 3.10.0, mujoco-warp 3.10.0.3, warp-lang 1.15.0, tensordict 0.13.0, tyro 1.0.15, tensorboard 2.21.0. `torch` nunca entra na lista.
- **O venv local não LANÇA treino:** `mjlab/scripts/train.py` chama `select_gpus` e estoura com `IndexError` sem placa. Local só `smoke`, `paridade`, `inspeciona` e `leitura --demo`.
- `LOG_ROOT` fica FORA do diretório do clone. A célula do clone faz `rm -rf` e apagaria o checkpoint.
- **Resume na Kaggle exige SEMEAR a árvore de log.** O `get_checkpoint_path` resolve `<log_root>/<experiment_name>/<run_dir>/model_N.pt`, e o dataset da Kaggle é PLANO e read-only. Copie o checkpoint para uma pasta de run com data no PASSADO (`1900-01-01_00-00-00_<run>`): ela ordena abaixo de qualquer pasta real que o mjlab criar, portanto um segundo resume pega o checkpoint novo e não a semente.
- **`max_iterations` é ADITIVO no resume.** `rsl_rl/runners/on_policy_runner.py:78` faz `total_it = current_learning_iteration + num_learning_iterations`. Derive-o de uma META absoluta menos o `iter` do checkpoint, e limite pelo teto de 12 h da Kaggle — versão commitada que estoura FALHA sem salvar a saída.
- **`!pip install` com pin de comparação é ARMADILHA DE SHELL.** `scipy>=1.15` e `numpy<2.5` sem quotes viram REDIRECIONAMENTO: o shell tenta abrir um arquivo `2.5`, aborta com exit 2, e **o pip nunca roda**. Com `-q` e sem conferir o código de saída, a célula não reclama e o erro aparece 2 células depois como `No module named 'mjlab'`. Instale por LISTA: `subprocess.run([sys.executable, "-m", "pip", "install", ...])`, e faça `assert returncode == 0`. Verificado 2026-09-04.
- **Instale só `mjlab==1.5.3`.** Ele já pina a árvore (`mujoco-warp~=3.10.0,>=3.10.0.3`, `mujoco~=3.10.0`, `rsl-rl-lib==5.4.0`, `numpy<2.5`, `tensorboard>=2.20.0`, `scipy>=1.15`). Repetir os pins transitivos à mão não dá segurança — uma versão indisponível derruba a resolução toda.
- **Sessão SEM commit perde `/kaggle/working` inteiro, e código nenhum salva sessão MORTA.** `try/finally` em volta do `launch_training` cobre fim normal, exceção e interrupção do kernel — não cobre kill de 12 h nem aba fechada. A única persistência que roda sem você presente é `kaggle datasets version` de dentro do notebook, com `KAGGLE_USERNAME`/`KAGGLE_KEY` em Add-ons -> Secrets. Suba o `.pt` CRU, não zip: a célula de resume acha `ARQ_CKPT` por `rglob`.
- Achar o dataset com `rglob`, nunca caminho fixo: a Kaggle monta ora em `/kaggle/input/<slug>/`, ora em `/kaggle/input/datasets/<usuario>/<slug>/`.
- Depois de `git push`, RE-RODE a célula do clone. O Kaggle já rodou código antigo sem avisar; o tell foi um peso de knob antigo no primeiro print.
- O learning rate do resume vai na célula. Um knob de `lr_warm_start` no config pode não ter leitor nenhum.
- A task é registrada por efeito colateral do `import`, então editar o arquivo não basta: o kernel usa a versão em cache. Reinicie o runtime e confira `print(list(load_env_cfg(TASK).rewards))` antes de treinar.
- Multi-GPU só compensa em run LONGO e headless. A primeira iteração é warmup e engana como se multi-GPU fosse lento; e o log dos dois ranks inunda a célula e trava o navegador.

Relacionado: g1-mjlab-warmstart-obs-armadilhas, g1-de-pe-e-gratis-no-mjlab, warmstart-lower-lr
