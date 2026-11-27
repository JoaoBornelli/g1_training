# Currículo e balanceador

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `balanceador.py` (inteiro), `curriculo.py` (inteiro), `knobs.py` (inteiro), `runner.py` (inteiro).
- `comando.py` (trechos: `CadeiaCmd`, `_alvos`, `_registra`, `_publica`, `_abre_trecho`, `compute`, `_resample`, `TwistV3`, `sorteia_velocidade`, `sucesso_do_segmento`, `sucesso_do_pegar`, `sucesso_do_botar`).
- `eventos.py` (trechos: `sorteia_fatia`, `escolhe_celula_terreno`, `sorteia_topo_e_massa`, `posiciona_cena`, `sorteia_destino`, `posiciona_destino`, `_base_da_pega`, `atraso`, e os eventos de DR).
- `recompensas.py` (trechos: `_rastreio`, `por_degrau`, `por_restricao`, `contato_laje`, `sacudida`, `aperto`, `freio`, `n_qualidade`), `observacoes.py` (trechos: `_magnitude`, `_calcula_caixa`, `privilegiados`, `niveis`, `degrau`), `terminacoes.py` (trechos: `fim_da_cadeia`, `n_batente`, `batente`), `alvos.py` (trecho: `avanca`), `env_cfg.py` (trechos 95 a 217), `tarefa.py` (trecho: `rl_cfg`).
- Biblioteca: `mjlab/rl/runner.py`, `mjlab/rl/vecenv_wrapper.py`, `mjlab/managers/curriculum_manager.py`, `mjlab/managers/command_manager.py`, `mjlab/envs/manager_based_rl_env.py`, `mjlab/envs/mdp/dr/geom.py`, `mjlab/envs/mdp/dr/_types.py`, `mjlab/tasks/velocity/config/g1/rl_cfg.py` (só a linha de `save_interval`), `rsl_rl/runners/on_policy_runner.py`.

Todas as referências `arquivo.py:linha` abaixo são relativas a `g1_v3/`. As referências de biblioteca são relativas a `site-packages/`.

## 1. Vocabulário

| Termo | O que é |
|---|---|
| Tarefa | Um dos quatro comportamentos do currículo: `ANDAR`, `PEGAR`, `CARREGAR`, `BOTAR` (`balanceador.py:20`). |
| Fatia | O grupo de tarefas que um env recebe no reset do episódio. Há três: `ANDAR`, `B` e `C` (`knobs.py:43`). |
| Célula | Uma unidade de medida do currículo: um par (tarefa, tipo). Cada célula guarda uma nota `s` (`balanceador.py:54-60`). |
| Nota `s` | A média móvel exponencial do sucesso da célula, atualizada uma vez por iteração (`balanceador.py:163-168`). |
| Portão | O limiar que a nota `s` de uma célula precisa alcançar (`balanceador.py:122-123`). |
| Nível | Um inteiro por tarefa. Ele fixa a dificuldade dos sorteios da tarefa (`balanceador.py:83`). |
| Subnível | Um degrau do nível 0 de `PEGAR` e de `BOTAR` (`balanceador.py:91`). |
| Linha de terreno | Um número que fixa a dificuldade do terreno. No `CARREGAR` é um inteiro por tarefa (`balanceador.py:84`). No `ANDAR` é um número real por coluna de terreno (`balanceador.py:87`). O `nivel_terreno("ANDAR")` é o piso da média das 8 colunas (`balanceador.py:329-333`). |
| Visita de terreno | Um episódio de `ANDAR` em terreno (coluna ≠ 0) que teve passos (`episode_length_buf` > 0) e que pediu `Terreno.dist_travessia`, acabou em queda ou passou dessa distância (seção 6.3). |
| Travessia | Uma visita em que a base passou de `Terreno.dist_travessia` = 3,5 m da origem da célula de terreno (seção 6.3). |
| Degrau de refino | Um inteiro por tarefa, de 0 a `Balanceador.degraus`. Ele leva as perturbações do valor inicial ao final (`balanceador.py:99`). |
| Etapa | Um alvo do `PEGAR` (`P0`, `P1`, `P2`, `P3`) ou do `BOTAR` (`B1`, `B2`, `B3`) (`knobs.py:44`). |
| Zona `z` | Um número em [0, 1] por etapa. Com `z` = 1 o critério de fecho da etapa é largo. Com `z` = 0 o critério é o final (`balanceador.py:105`). |
| Fecho da iteração | O método `fecha_iteracao`. Ele atualiza todo o estado do currículo uma vez por iteração (`balanceador.py:153-195`). |
| Iteração | O bloco de `Balanceador.passos_por_iteracao` = 24 passos de ambiente (`knobs.py:402`). `tarefa.py:49` copia este valor para `num_steps_per_env` do PPO. |
| Degrau global | O menor degrau de refino entre as quatro tarefas (`balanceador.py:341-343`). |

## 2. Estrutura do estado

### 2.1 Tarefas, fatias e células

`FATIA_DE` liga cada tarefa a uma fatia: `ANDAR` à fatia 0, `PEGAR` e `CARREGAR` à fatia 1 (`B`), `BOTAR` à fatia 2 (`C`) (`balanceador.py:34`).

`LAYOUT` fixa a ordem das células. O `celula_id` de uma célula é o índice dela em `LAYOUT` (`balanceador.py:22-25`, `balanceador.py:318-320`). A lista de células nunca muda de ordem (`balanceador.py:79`).

| `celula_id` | Tarefa | Tipo | Fatia | Portão da célula | Nasce aberta |
|---|---|---|---|---|---|
| 0 | `ANDAR` | `andar` | 0 | `Balanceador.portao_andar` = 0,75 (`knobs.py:392`) | sim |
| 1 | `ANDAR` | `terreno` | 0 | `Balanceador.portao_andar_terreno` = 0,50 (`knobs.py:393`) | não; abre no nível `Balanceador.nivel_abre_terreno` = 2 do `ANDAR` (seção 5.6) |
| 2 | `CARREGAR` | `reto` | 1 | `Balanceador.portao` = 0,75 (`knobs.py:391`) | não |
| 3 | `CARREGAR` | `giro_lugar` | 1 | `Balanceador.portao` = 0,75 | não |
| 4 | `CARREGAR` | `arco` | 1 | `Balanceador.portao` = 0,75 | não |
| 5 | `CARREGAR` | `terreno` | 1 | `Balanceador.portao` = 0,75 | não |
| 6 | `PEGAR` | `pegar` | 1 | `Balanceador.portao` = 0,75 | não |
| 7 | `BOTAR` | `botar` | 2 | `Balanceador.portao` = 0,75 | não |

Só a célula `(ANDAR, andar)` nasce aberta: a constante `ABERTAS_NO_COMECO` lista só ela, e a construção marca `aberta` quando o par (tarefa, tipo) está nessa lista (`balanceador.py:27`, `balanceador.py:79`). O portão de cada célula entra na lista `_portao` na construção (`balanceador.py:97-98`). A célula `(ANDAR, terreno)` abre pelo nível do `ANDAR`, e não pela construção (seção 5.6).

Nenhum código de `balanceador.py` fecha uma célula depois de aberta. As atribuições a `aberta` são a construção (`balanceador.py:79`), `_abre_cel` (`balanceador.py:120`) e `carrega` (`balanceador.py:410`).

As células de velocidade de cada tarefa ficam em `VELOCIDADE` (`balanceador.py:29-30`): `ANDAR` tem `andar`; `CARREGAR` tem `reto`, `giro_lugar`, `arco`; `PEGAR` tem `pegar`; `BOTAR` tem `botar`. As tarefas de `CADEIA` = (`ANDAR`, `CARREGAR`) têm também a célula `terreno` (`balanceador.py:31`).

### 2.2 Variáveis de estado de `Curriculo`

| Variável | Valor inicial | Significado | Linha |
|---|---|---|---|
| `k`, `k_dr` | `Knobs.balanceador`, `Knobs.dr` | Os knobs lidos pelo currículo. Com `k_dr` = `None`, `dr()` devolve `{}` | `balanceador.py:68-69`, `curriculo.py:47` |
| `max_nivel` | ver abaixo | O nível máximo de cada tarefa | `balanceador.py:72-74` |
| `_max_ter` | ver abaixo | A linha de terreno máxima de `ANDAR` e de `CARREGAR` | `balanceador.py:77-78` |
| `celulas` | 8 `Celula` com `s` = 0,0 e `n_medidas` = 0 | As células, na ordem de `LAYOUT` | `balanceador.py:79` |
| `it` | 0 | Número de fechos já feitos | `balanceador.py:80` |
| `_acum_n`, `_acum_suc` | 0 por célula | A contagem de segmentos e a soma das notas da iteração corrente | `balanceador.py:81-82` |
| `_nivel` | 0 por tarefa | O nível de cada tarefa | `balanceador.py:83` |
| `_nivel_ter` | 0 por tarefa | A linha de terreno do `CARREGAR`. O `ANDAR` não a usa: a linha dele é `linha`. `PEGAR` e `BOTAR` ficam em 0 | `balanceador.py:84` |
| `linha` | `[0,0] * 9` | A linha de terreno do `ANDAR` por coluna, em número real, na ordem de `Terreno.colunas`. A coluna 0 (`plano`) não conta | `balanceador.py:87` |
| `vis_ter` | `[0] * 9` | As visitas de terreno do `ANDAR` por coluna, na iteração corrente | `balanceador.py:88` |
| `trav_ter` | `[0] * 9` | As travessias de terreno do `ANDAR` por coluna, na iteração corrente | `balanceador.py:89` |
| `_sub` | `PEGAR` 0, `BOTAR` 0 | O subnível | `balanceador.py:91` |
| `_sub_max` | `len(Niveis.subnivel_extra)` − 1 = 1 | O último subnível (`knobs.py:422`) | `balanceador.py:92` |
| `_seguidas` | 0 por tarefa | Os fechos seguidos em que a condição de subida vale | `balanceador.py:95` |
| `_portao` | ver 2.1 | Um portão por célula | `balanceador.py:97-98` |
| `_degrau` | 0 por tarefa | O degrau de refino | `balanceador.py:99` |
| `_cont` | 0 por tarefa | Os fechos seguidos no portão do último nível, para o refino | `balanceador.py:100` |
| `_dr` | `portao0`, `vies`, `subida` = `False` | As três travas da randomização de domínio. Elas só ligam | `balanceador.py:103` |
| `z` | `[1,0] * 7` | A zona de cada etapa | `balanceador.py:105` |
| `vis_acum` | `[0] * 7` | As visitas acumuladas de cada etapa | `balanceador.py:106` |
| `fech_acum` | `[0] * 7` | Os fechos acumulados de cada etapa | `balanceador.py:107` |
| `taxa` | `[-1,0] * 7` | A última taxa de fecho por visita calculada. −1 = sem medida | `balanceador.py:108` |
| `_it_abertura` | `[0, None, None]` | O fecho em que cada fatia abriu. `None` = fechada. A fatia `ANDAR` abre no fecho 0 | `balanceador.py:110` |
| `_rampa` | `[0, 0, 0]` | Os degraus da rampa de cada fatia | `balanceador.py:112` |

`N_ETAPAS` = 7 é o número de etapas (`balanceador.py:33`). Ele iguala o tamanho de `knobs.ALVOS` (`knobs.py:44`).

Os máximos saem dos knobs na construção:

| Máximo | Conta | Valor |
|---|---|---|
| `max_nivel["ANDAR"]` | `round((Comando.frente_tetos[-1] − Comando.andar_inicio) / Comando.andar_passo)` | round((1,0 − 0,3) / 0,1) = 7 (`knobs.py:122`, `knobs.py:129`, `knobs.py:134`, `balanceador.py:72`) |
| `max_nivel["CARREGAR"]` | `len(Comando.frente_tetos) − 1` | 4 (`knobs.py:122`, `balanceador.py:73`) |
| `max_nivel["PEGAR"]` | `len(Niveis.pegar) − 1` | 4 (`knobs.py:417`, `balanceador.py:74`) |
| `max_nivel["BOTAR"]` | `len(Niveis.botar) − 1` | 4 (`knobs.py:418`, `balanceador.py:74`) |
| `_max_ter["ANDAR"]` | `Terreno.linhas − 1` | 12 (`knobs.py:429`, `balanceador.py:77`) |
| `_max_ter["CARREGAR"]` | o maior valor de `Terreno.linha_max_carregar` | 9 (`knobs.py:439-440`, `balanceador.py:78`) |

`Celula.n_medidas` conta as atualizações de nota da célula desde a última zeragem. O fecho da iteração (`balanceador.py:166-167`) e `nivel_passou` (`balanceador.py:355`) o leem. `registra` não o lê. Uma subida de subnível ou de nível zera `n_medidas` de todas as células da tarefa (`balanceador.py:277-278`). Uma subida de linha de terreno do `CARREGAR` zera só o `n_medidas` da célula `terreno` dele (`balanceador.py:270-271`). A linha do `ANDAR` não zera `n_medidas`. Nenhuma subida zera `s`.

### 2.3 Trechos de cada fatia

Um trecho é um intervalo do episódio com uma tarefa publicada. No reset, o env abre o primeiro trecho da cadeia da fatia sorteada (`comando.py:660`). A duração de um trecho por tempo, ou o atraso de um trecho por aviso, é uniforme em [lo; hi] pelo `v3_gerador`, sorteada na abertura do trecho (`comando.py:578-579`). Um trecho por aviso termina quando o fecho do `P3` (no `PEGAR`) ou do `B3` (no `BOTAR`) vale e o atraso passou (`comando.py:201-216`, `comando.py:606-609`).

| Fatia | Trechos, em ordem (tarefa: duração ou atraso em s) | Referência |
|---|---|---|
| `ANDAR` (0) | `ANDAR`: `Cadeia.episodio_s` = 20,0 s fixo; em terreno o episódio acaba também na travessia (abaixo) | `comando.py:176`, `knobs.py:160` |
| `B` (1) | `ANDAR`: `Cadeia.espera_s` = (0,5; 1,0); `PEGAR`: por aviso, atraso `Cadeia.atraso_pegar_s` = (0,0; 1,0); `CARREGAR`: `Cadeia.carregar_b_s` = (4,0; 7,0) | `comando.py:177`, `knobs.py:154-155`, `knobs.py:158` |
| `C` (2) | `ANDAR`: `espera_s`; `PEGAR`: por aviso, `atraso_pegar_s`; `CARREGAR`: `Cadeia.carregar_c_s` = (0,5; 1,0); `BOTAR`: por aviso, `Cadeia.atraso_botar_s` = (0,0; 1,5); `ANDAR`: `Cadeia.andar_final_s` = 1,5 s fixo | `comando.py:178-179`, `knobs.py:154`, `knobs.py:156-159` |

O termo `time_out` vale quando `episode_length_buf` ≥ `max_episode_length` (de `Cadeia.episodio_s` = 20,0 s) ou quando `v3_fim` vale (o último trecho da cadeia acabou) (`terminacoes.py:58-72`, `env_cfg.py:105`, `env_cfg.py:138`, `knobs.py:160`). No `ANDAR` em terreno o `time_out` vale também quando a base passa de `Terreno.dist_travessia` = 3,5 m da origem da célula, no plano xy. A distância sai do `twist.pose_ult` e a conta exige `episode_length_buf` > 2 (`terminacoes.py:69-72`; `knobs.py:442`). Com a caixa perdida, o trecho congela e a tarefa vira `ANDAR` até o fim do episódio (`comando.py:610-613`, `comando.py:412-420`).

### 2.4 Funções internas de consulta

| Função | O que devolve | Linha |
|---|---|---|
| `_aberta(t, tp)` | `aberta` da célula (`t`, `tp`) | `balanceador.py:116-117` |
| `_abre_cel(t, tp)` | Liga `aberta` da célula | `balanceador.py:119-120` |
| `_passa(i)` | `s` da célula `i` ≥ `_portao[i]` | `balanceador.py:122-123` |
| `_passa_media(ids)` | Verdadeiro se `ids` não é vazio e a média de `s` das células é ≥ a média dos portões delas | `balanceador.py:125-128` |
| `_de(t)` | Os `celula_id` da tarefa `t` | `balanceador.py:130-131` |
| `_abertos(t)` | Os `celula_id` abertos da tarefa `t` | `balanceador.py:133-134` |
| `_deficit(ids)` | `max(portão − s, 0)` por célula, em float64. Na célula `terreno` do `ANDAR` o valor é `1 − média(linha[1:]) / _max_ter["ANDAR"]`, a distância das linhas ao topo: a nota `s` dela não entra | `balanceador.py:136-144`, `balanceador.py:44-45` |

## 3. Ligação do currículo ao env

| Fato | Referência |
|---|---|
| O env cria `v3_cur` = `balanceador.Curriculo(...)` em `garante_v3`. A função é idempotente (só cria o que falta). | `curriculo.py:39-47` |
| O construtor de `CadeiaCmd` chama `garante_v3`. Por isso `v3_cur` existe na criação do env, antes do primeiro reset e antes de qualquer carga de checkpoint. | `comando.py:293` |
| `garante_v3` cria também `v3_alvos` = `Knobs.alvos` (o critério final de fecho das etapas) e `v3_iter` = −1. | `curriculo.py:48-49`, `curriculo.py:58-59` |
| O cfg de treino registra um termo de currículo só: `v3` = `curriculo.v3`. No cfg de `play` não há termo: o estado do currículo fica congelado. | `env_cfg.py:216` |
| O mjlab chama o termo no início de `_reset_idx`, antes dos eventos de reset e antes do `command_manager.reset`. Isso ocorre em todo passo em que algum env reinicia. | `mjlab/envs/manager_based_rl_env.py:553-554`, `mjlab/managers/curriculum_manager.py:112-117` |
| Os eventos de reset, na ordem do cfg: `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, `pd`, `forca_motor`, `solref` (estes três só fora do `play`), `com_caixa` (só com `DR.com_caixa`), `filtros`. | `env_cfg.py:153-184` |
| Os eventos de startup (uma vez, na criação do env): `atrito_pe`, `massa_tronco`, `vies_encoder`, `tamanho_caixa` e `atraso`. O `atraso` põe o teto de atraso `DR.atraso_inicio[1]` = 0 passos (`knobs.py:511`) com `Contrato.decimacao` = 4 (`knobs.py:55`). | `env_cfg.py:185-198` |
| Os eventos de passo e de intervalo, só fora do `play`: `empurrao` (intervalo, a cada `DR.periodo_empurrao_s` = (3,0; 7,0) s, `knobs.py:520`), `forca_caixa` (passo) e `rfi` (passo). | `env_cfg.py:200-210` |
| `garante_v3` cria só o que não existe, com `hasattr`. Cria `v3_cur`, `v3_alvos`, `v3_ref` (`alvos.Referencias`), `v3_filtro`, `v3_padrao`, `v3_iter` (−1), `v3_percepcao_passo` (−1), `v3_gerador` (um `torch.Generator` no device do env, semeado com `cfg.seed`, ou 0 se a seed é `None` ou 0), `v3_odo`, `v3_estimador` e os buffers por env `v3_*` de `_LONGOS`, `_BOOLS`, `_ESCALARES`, `_VETORES`, mais `v3_alvo_idx` e `v3_celula` (−1), `v3_t_empurrao` (−∞), `v3_phi` (−1), `v3_h` (1), `v3_v_chegada` (−1), `v3_caixa_xy` e `v3_incl_imu` (identidade). | `curriculo.py:24-30`, `curriculo.py:33-90` |
| Os sorteios de fatia, de terreno, de cena, de base, de velocidade, de duração de trecho (`comando.py:579`), de viés do alvo do `BOTAR` (`comando.py:556`) e da DR do refino (`pd`, `forca_motor`, `solref`, `com_caixa`, `rfi`) leem `v3_gerador` (uniforme em [0, 1) por `torch.rand` com o gerador). O intervalo de reamostra do comando usa `uniform_` sem gerador explícito (`mjlab/managers/command_manager.py:119-121`). O μ dos pads e da laje usa o `dr.geom_friction` do mjlab com `distribution` = `"uniform"` por padrão. O empurrão e o impulso `forca_caixa` usam `sample_uniform` do mjlab, sem gerador explícito (`mjlab/envs/mdp/events.py:336`, `mjlab/envs/mdp/events.py:525`). | `eventos.py:79`, `eventos.py:464-489`, `mjlab/envs/mdp/dr/geom.py:123-124` |
| O reset inicial do env roda na criação do wrapper. O `train` do mjlab cria o wrapper, depois o runner, depois chama `runner.load` (se há checkpoint para retomar) e depois `runner.learn`. O reset inicial roda, portanto, antes da carga do currículo. | `mjlab/rl/vecenv_wrapper.py:24-25`, `mjlab/scripts/train.py:149`, `mjlab/scripts/train.py:167`, `mjlab/scripts/train.py:173` |
| O reset inicial chama `command_manager.compute(dt=0.0)` depois do `_reset_idx`. Por isso `CadeiaCmd._alvos` roda no reset inicial (seção 6.1), com `cur.it` = 0. | `mjlab/envs/manager_based_rl_env.py:372` |

## 4. O caminho de uma medida

`registra(celula_id, n, soma)` soma `n` em `_acum_n[celula_id]` e `soma` em `_acum_suc[celula_id]` (`balanceador.py:148-151`). `soma` é a soma das notas de `n` segmentos. Ele não confere se a célula está aberta. Duas classes de `comando.py` chamam `registra`: `CadeiaCmd` (seção 4.1) e `TwistV3` (seção 4.2). Nenhum outro arquivo do treino o chama.

### 4.1 `CadeiaCmd._registra`: células `PEGAR` e `BOTAR`

Chamada em `comando.py:539`, com `celula` = `id_celula("PEGAR", "pegar")` (6) ou `id_celula("BOTAR", "botar")` (7), `n` = `ids.numel()` e `soma` = `suc.sum()`.

Um episódio vira pendente de registro assim:
- `self.pega` fica `True` quando o `P0` abre, em `_abre_p0` (`comando.py:384`).
- `self.bota` fica `True` quando o `B1` abre, em `_abre_botar` (`comando.py:566`).
- `_registra` limpa o pendente dos ids que registra (`comando.py:526-540`).

`_registra` roda em quatro lugares:

| Quando | Chamada | `falhou` | Linha |
|---|---|---|---|
| Na abertura de um trecho `BOTAR` (registra o `PEGAR` pendente) | `_registra(ids[botar], zeros)` | `False` | `comando.py:589` |
| No fim da cadeia (`fim` verdadeiro em `compute`; registra o `BOTAR` pendente) | `_registra(fim_ids, zeros, botar=True)` | `False` | `comando.py:614-616` |
| No reset do episódio, `PEGAR` | `_registra(env_ids, falhou)` | `~time_outs[env_ids]` | `comando.py:643-644` |
| No reset do episódio, `BOTAR` | `_registra(env_ids, falhou, botar=True)` | `~time_outs[env_ids]` | `comando.py:643`, `comando.py:645` |

O valor registrado por env:
- `PEGAR`: `suc` = `pega_fechada & ~falhou & ~perdeu` (`comando.py:68-70`) e `max(v3_cur.z[:4]) == 0,0` (`comando.py:537`). `pega_fechada` fica `True` no fecho do `P3` (`comando.py:492`, `comando.py:496`).
- `BOTAR`: `suc` = `bota_fechada & ~falhou & ~perdeu` (`comando.py:91-93`) e `max(v3_cur.z[4:]) == 0,0` (`comando.py:534`). `bota_fechada` fica `True` no fecho do `B3` (`comando.py:498`).

Então o sucesso conta só quando as zonas de todas as etapas da tarefa valem exatamente 0,0. Antes disso o episódio entra como falha. `perdeu` é a caixa perdida (`comando.py:412-414`).

### 4.2 `TwistV3._fecha`: células `ANDAR` e `CARREGAR`

Chamada em `comando.py:794`: uma chamada por célula, com `n` = `m.sum()` e `soma` = `x[m].sum()`, onde `m` marca os segmentos de `reg` dessa célula.

Um segmento é o trecho de tempo com um comando de velocidade fixo. `_abre` guarda o comando, a célula (`seg_cel`) e a marca "tarefa com caixa" (`seg_carrega` = tarefa diferente de `ANDAR`) no instante da abertura (`comando.py:735-751`). `_fecha` fecha o segmento em dois momentos:

| Quando | Chamada | `falha` | Linha |
|---|---|---|---|
| Troca de segmento: o `command_counter` mudou desde o passo anterior | `_fecha(mudou, ...)` | `v3_perdeu & seg_carrega` | `comando.py:816-821` |
| Reset do episódio | `_fecha(env_ids, ...)` | `~time_outs[env_ids]` | `comando.py:879-883` |

O `command_counter` muda quando o `twist` reamostra (`mjlab/managers/command_manager.py:123`). Isso ocorre por três vias:
- no reset do episódio, que chama `_resample` (`mjlab/managers/command_manager.py:99-106`);
- por tempo: a cada `Comando.reamostra_s` = (3,0; 8,0) s (`knobs.py:137`, `env_cfg.py:122-123`), sorteio uniforme por env em `mjlab/managers/command_manager.py:117-123`;
- por troca de trecho: `_publica` chama `_twist()._resample(ids)` (`comando.py:519`).

No `CARREGAR` da fatia `B`, `_resample_command` põe `time_left` = ∞. O comando fica fixo até o fim do trecho (`comando.py:939-941`).

A célula do segmento vem de `v3_celula`, escrita em `_resample_command` do `twist` (`comando.py:897-925`):
- `v3_celula` = −1 por padrão (sem célula: nenhum registro). Isso vale para `PEGAR`, `BOTAR`, `CARREGAR` na fatia `C` e `ANDAR` nas fatias `B` e `C` (`comando.py:901-905`).
- Anda só `ANDAR` na fatia `ANDAR` e `CARREGAR` na fatia `B` (`comando.py:905`).
- Se `terrain_types` do env é diferente de 0 (episódio de terreno), a célula é a `terreno` da tarefa (`comando.py:914-916`).
- Senão `sorteia_celulas(nome, n, g, sem_tipo="terreno")` sorteia entre as células abertas, com a fração da célula `terreno` valendo 0 (`comando.py:917-918`, `balanceador.py:306-316`).

O valor `x` do segmento (`comando.py:759-794`):
- Tarefa `ANDAR` (`seg_carrega` falso): `x` = `nota` × `(~falha)`.
  - `v_med` = `seg_nota / seg_dur` (a velocidade média da base depois da âncora; `seg_dur` com mínimo de 1e-6).
  - `den` = [‖c_xy‖, |c_z|], onde `c` é o comando do segmento. `ativo` = `den >= Comando.limiar_canal` = 0,02 (`knobs.py:138`). Depois `den` recebe o mínimo `limiar_canal`.
  - Canal ativo: `razao` = clamp(proj / den, 0, 1), com proj linear = (v̄_xy · c_xy) / ‖c_xy‖ e proj de giro = v̄_z · sinal(c_z).
  - Canal sem comando (nos envs de rumo, o `wz` de `seg_cmd` é 0 e o canal de giro cai aqui: `comando.py:742`): `deriva` = exp(−v̄² / σ²), com σ = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`). Para o linear, v̄² = ‖v̄_xy‖². Para o giro, v̄² = v̄_z².
  - `nota` = a média dos dois canais (linear e giro), cada um com `razao` se ativo, ou `deriva` se não.
- Tarefa com caixa (`CARREGAR`): `x` = `sucesso_do_segmento(...)` como 0/1, × `(~falha)`.
  - `ok_lin` = razão linear ≥ `Comando.razao_portao` = 0,50 (`knobs.py:146`), se o canal é ativo; senão deriva de posição < limite.
  - `ok_giro` = razão de giro ≥ 0,50, se ativo; senão deriva de rumo < `Comando.deriva_rumo_rad` = 0,25 (`knobs.py:144`).
  - O limite da deriva de posição é `no_degrau(Comando.deriva_pos_inicio, Comando.deriva_pos_m, restricao, 1)`, com valores 0,25 m e 0,10 m (`knobs.py:142-143`, `comando.py:773`).
  - O sucesso exige também `~(carregar & perdeu)`, e, na célula `terreno`, `incl_max_seg <= Comando.incl_max_terreno_graus` = 15,0 graus (`knobs.py:149`, `comando.py:239-240`).
- Registra (`reg`): para `ANDAR`, `seg_dur > 0`; para `CARREGAR`, `t_fecho >= t_ancora` (o segmento só entra depois da âncora). Em ambos, `| falha` força o registro (`comando.py:241`, `comando.py:790`).
- A âncora é o instante em que o código grava a pose da base para medir a deriva. Ela fica em `max(t_abertura, t_empurrao) + Comando.ancora_s`, com `ancora_s` = 1,5 s (`knobs.py:145`, `comando.py:221-224`, `comando.py:833-842`). `t_abertura` é `_agora` na abertura do segmento (`comando.py:737`). `t_empurrao` é −∞ no reset do episódio (`eventos.py:163`) e recebe `episode_length_buf × step_dt` quando o empurrão ocorre (`eventos.py:269`). A pose da âncora é gravada no primeiro passo com `_agora` ≥ instante, e é regravada quando o instante muda (`comando.py:834-842`).
- Só as células com `v3_celula` ≥ 0 e `reg` verdadeiro registram (`comando.py:792`).

## 5. O fecho da iteração

### 5.1 Quando o fecho roda

`curriculo.v3` calcula `it = common_step_counter // Balanceador.passos_por_iteracao` (`curriculo.py:119`, `knobs.py:402`). Se `v3_iter >= 0` e `it != v3_iter`, chama `fecha_iteracao()` uma vez e grava `v3_iter` = `it` (`curriculo.py:120-122`).

Consequências, todas no código:
- A primeira chamada só grava `v3_iter`, porque `v3_iter` começa em −1 (`curriculo.py:58-59`, `curriculo.py:120`).
- Cada chamada fecha no máximo uma iteração, mesmo que `it` tenha avançado mais de 1 (`curriculo.py:120-122`).
- O termo só roda em passos com algum reset (ver seção 3). O fecho acontece no primeiro reset depois da virada de `it`.
- Depois de uma carga de checkpoint, `common_step_counter` recebe o valor salvo (`mjlab/rl/runner.py:138-140`). `v3_iter` guarda o valor do reset inicial (0). Se o contador restaurado dividido por 24 não for 0, o primeiro reset depois da carga fecha uma iteração.

### 5.2 Os passos, na ordem do código (`balanceador.py:153-195`)

| # | Passo | Linhas |
|---|---|---|
| 1 | `it` += 1 | `balanceador.py:162` |
| 2 | Média móvel da nota de cada célula | `balanceador.py:163-168` |
| 3 | Zona `z` de cada etapa e acumuladores | `balanceador.py:169-174` |
| 4 | Restrição de cadeia entre etapas | `balanceador.py:175-176` |
| 4b | Linha de terreno de cada coluna do `ANDAR` | `balanceador.py:177-182` |
| 5 | Contador de refino de cada tarefa (`_conta_refino`) | `balanceador.py:183-184` |
| 6 | Abertura de células (`_abre`) | `balanceador.py:185` |
| 7 | Abertura de fatias | `balanceador.py:186-188` |
| 8 | Rampa das fatias | `balanceador.py:189-193` |
| 9 | Subida de nível, de subnível e de linha de terreno (`_sobe`) | `balanceador.py:194` |
| 10 | Travas do DR (`_trava_dr`) | `balanceador.py:195` |

### 5.3 Média móvel da nota (passo 2)

Para cada célula `i` com `_acum_n[i]` > 0 (`balanceador.py:164-167`):
- `x` = `_acum_suc[i] / _acum_n[i]`.
- Se `n_medidas` = 0: `s` = `x`. Senão: `s` += `Balanceador.alfa` × (`x` − `s`), com `alfa` = 0,05 (`knobs.py:394`).
- `n_medidas` += 1.

Com `_acum_n[i]` = 0, a nota `s` não muda. Todas as células zeram `_acum_n` e `_acum_suc` (`balanceador.py:168`).

### 5.4 Zona adaptativa (passos 3 e 4)

A zona `z[e]` de cada etapa se move assim (`balanceador.py:170-174`):
- A condição de atualização: `vis_acum[e] >= Balanceador.zona_n_min` = 200 (`knobs.py:406`).
- Nesse caso: `taxa[e]` = `fech_acum[e] / vis_acum[e]`.
- `z[e]` = min(1,0; max(0,0; `z[e]` − `Balanceador.zona_ganho` × (`taxa[e]` − `Balanceador.zona_t_alvo`))), com `zona_ganho` = 0,05 e `zona_t_alvo` = 0,6 (`knobs.py:404-405`).
- Depois `vis_acum[e]` e `fech_acum[e]` voltam a 0.
- Se `vis_acum[e]` < 200, nada muda: `z[e]`, `taxa[e]` e os acumuladores ficam, e a contagem segue na iteração seguinte.

Taxa acima de 0,6 reduz `z` (critério mais apertado). Taxa abaixo de 0,6 aumenta `z` (critério mais largo).

Depois do loop, a restrição de cadeia (`balanceador.py:175-176`): para `e` em (1, 2, 3, 5, 6), nessa ordem, `z[e]` = max(`z[e]`, `z[e−1]` − `Balanceador.zona_folga_cadeia`), com `zona_folga_cadeia` = 0,2 (`knobs.py:408`). Os índices 1, 2, 3 são `P1`, `P2`, `P3`. Os índices 5, 6 são `B2`, `B3`. O `P0` (índice 0) e o `B1` (índice 4) não sofrem restrição. Cada passo do loop lê o `z[e−1]` já atualizado.

Os acumuladores `vis_acum` e `fech_acum` recebem valores da cadeia de comando (`comando.py:488-490`, `comando.py:646-650`). Ver a seção 6.

### 5.4b Linha de terreno do ANDAR (passo 4b, `balanceador.py:177-182`)

Cada coluna de terreno do `ANDAR` tem a própria linha (`linha`, seção 2.2). O passo roda para as colunas 1 a 8, na ordem, depois da restrição de cadeia e antes de `_conta_refino`:
- A condição de atualização: `vis_ter[c]` ≥ `Balanceador.linha_n_min` = 50 (`knobs.py:410`).
- Nesse caso: `taxa` = `trav_ter[c]` / `vis_ter[c]`.
- `linha[c]` = min(12; max(0; `linha[c]` + `Balanceador.linha_ganho` × (`taxa` − `Balanceador.zona_t_alvo`))), com `linha_ganho` = 2,5 e `zona_t_alvo` = 0,6 (`knobs.py:411`, `knobs.py:404`). O teto 12 é `_max_ter["ANDAR"]`.
- Depois `vis_ter[c]` e `trav_ter[c]` voltam a 0.
- Se `vis_ter[c]` < 50, nada muda, e a contagem segue na iteração seguinte.

Cada coluna se move sozinha. A taxa 1,0 sobe a linha em 1,0, a taxa 0 a desce em 1,5, e a taxa 0,6 a mantém (derivado: 2,5 × (1,0 − 0,6) e 2,5 × (0 − 0,6)). Uma coluna vai da linha 0 à linha 12 em no mínimo 12 atualizações, todas com taxa 1,0 (derivado: 12 / 1,0).

O `ANDAR` no nível máximo tem `ok` falso em `_sobe` (seção 5.8): só este passo move a linha dele. O `sorteia_fatia` preenche `vis_ter` e `trav_ter` no reset (seção 6.3).

### 5.5 Refino (passo 5, `_conta_refino`, `balanceador.py:234-248`)

Roda para cada tarefa, na ordem `ANDAR`, `PEGAR`, `CARREGAR`, `BOTAR` (`balanceador.py:183-184`), com as notas já atualizadas e antes de `_abre` e `_sobe`.

1. Se `_degrau[t]` ≥ `Balanceador.degraus` = 5 (`knobs.py:401`): retorna sem mudar `_cont`.
2. `ultimo` é verdadeiro quando `_nivel[t]` = `max_nivel[t]` e, para `ANDAR` e `CARREGAR`, a célula `terreno` está aberta e `nivel_terreno(t)` = `_max_ter[t]`. No `ANDAR`, `nivel_terreno` é o piso da média das 8 linhas: vale 12 só com as 8 linhas em 12 (`balanceador.py:239-240`, `balanceador.py:329-333`).
3. `abertas` = as células abertas da tarefa. `_passa_media(abertas)` vale quando a média de `s` das abertas é ≥ a média dos portões das mesmas células (`balanceador.py:125-128`).
4. Se `ultimo` e `_passa_media(abertas)`: `_cont[t]` += 1. Senão: `_cont[t]` = 0.
5. Se `_cont[t]` ≥ `Balanceador.iters_refino` = 100 (`knobs.py:400`): `_degrau[t]` += 1 e `_cont[t]` = 0.

O degrau nunca desce. `_conta_refino` não lê `n_medidas`.

⚠ Divergência: o comentário de `fecha_iteracao` diz que o contador de refino roda antes das subidas "para que a célula recém-subida (s velho, n_medidas = 0) não conte no nível novo". O código só garante que a subida do mesmo fecho ainda não vale quando o contador roda (`balanceador.py:183-194`). Nos fechos seguintes, `_conta_refino` usa o `s` guardado da célula, sem olhar `n_medidas` (`balanceador.py:234-248`).

### 5.6 Abertura de células (passo 6, `_abre`, `balanceador.py:204-232`)

Cada condição abaixo abre no máximo uma célula por cadeia a cada fecho. `_abre_cel` só liga `aberta`. Uma célula nova tem `s` = 0,0, que fica abaixo de qualquer portão do layout: ela não abre a célula seguinte no mesmo fecho.

| Célula aberta | Condição exata | Linha |
|---|---|---|
| `(CARREGAR, reto)` e `(PEGAR, pegar)`, juntas | `(CARREGAR, reto)` fechada e `_nivel["ANDAR"]` ≥ `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`) | `balanceador.py:209-211` |
| A primeira célula de velocidade fechada de `CARREGAR`, na ordem `reto`, `giro_lugar`, `arco` | `abertas` (células abertas da tarefa) não vazia, existe célula de velocidade fechada, e `_passa_media(abertas)` | `balanceador.py:212-217` |
| `(ANDAR, terreno)` | `_nivel["ANDAR"]` ≥ `Balanceador.nivel_abre_terreno` = 2 (`knobs.py:412`), célula fechada, e `_passa_media(células abertas de ANDAR)`: a nota de `(ANDAR, andar)` ≥ 0,75, porque a célula `terreno` ainda está fechada | `balanceador.py:222-228` |
| `(CARREGAR, terreno)` | `_nivel["CARREGAR"]` = 4, célula fechada, `_passa_media(células abertas de CARREGAR)`, e `nivel_terreno("ANDAR")` ≥ 1 (o piso da média das 8 linhas do `ANDAR`). A nota da célula `(ANDAR, terreno)` não entra mais | `balanceador.py:222-228` |
| `(BOTAR, botar)` | `(BOTAR, botar)` fechada e `s` de `(PEGAR, pegar)` ≥ `Balanceador.portao` = 0,75 | `balanceador.py:231-232` |

Quando a célula `(ANDAR, terreno)` abre, no mínimo (derivado: conta sobre `_abre` e `_sobe`, com a nota de `(ANDAR, andar)` ≥ 0,75 desde o fecho 1):
- O fecho 50 sobe o `ANDAR` do nível 0 ao 1, e o fecho 100 sobe do 1 ao 2 (`iters_nivel` = 50 fechos seguidos, `knobs.py:395`).
- O fecho 101 abre a célula, porque `_abre` roda antes de `_sobe` e vê o nível 2 só no fecho seguinte à subida (`balanceador.py:185`, `balanceador.py:194`).
- O fecho 101 equivale a 101 × 24 = 2 424 passos de env (`knobs.py:402`).

A célula nasce com `s` = 0 e as linhas em 0, então o déficit dela é 1,0. Com só a fatia `ANDAR` aberta (a fatia `B` abre no nível 5), a fração de episódios de terreno logo na abertura vale 0,885 com a nota de `(ANDAR, andar)` no portão, e 0,786 com a nota em 0,60 (derivado: fórmula da seção 7.2).

O loop de células de velocidade percorre `CADEIA`, mas `ANDAR` não tem célula de velocidade fechada (`VELOCIDADE["ANDAR"]` = (`andar`,) e ela nasce aberta). Só `CARREGAR` abre ali (`balanceador.py:29-30`, `balanceador.py:212-217`).

### 5.7 Abertura de fatias e rampa (passos 7 e 8)

- Passo 7: para a fatia `f` em (1, 2), se `_it_abertura[f]` é `None` e alguma célula aberta pertence à fatia `f` (por `FATIA_DE`): `_it_abertura[f]` = `it` (`balanceador.py:186-188`). A fatia 0 tem `_it_abertura[0]` = 0 desde a construção.
- Passo 8: para `f` em (1, 2): `d` = 0 se `_it_abertura[f]` é `None`, senão `it − _it_abertura[f]`. Se `d` > 0 e `d` múltiplo de `Balanceador.rampa_fatia_its` = 12 (`knobs.py:399`): `_rampa[f]` += 1 (`balanceador.py:189-193`). O código não limita `_rampa`. A fatia 0 não tem rampa.

### 5.8 Subida de nível, de subnível e de linha de terreno (passo 9, `_sobe`, `balanceador.py:250-278`)

Para cada tarefa `t`, na ordem `ANDAR`, `PEGAR`, `CARREGAR`, `BOTAR`:

1. `ids` = os `celula_id` das células de velocidade de `t`. `ter` = o `celula_id` da célula `terreno`, ou `None` em `PEGAR` e `BOTAR`.
2. A condição `ok`:
   - Se `_nivel[t]` < `max_nivel[t]`: `ok` = todas as células de `ids` abertas e `_passa_media(ids)`.
   - Senão, se `t` = `CARREGAR` e `_nivel_ter[t]` < `_max_ter[t]`: `ok` = célula `terreno` aberta e `s` dela ≥ seu portão (`_passa(ter)`).
   - Senão: `ok` = `False`.
3. `_seguidas[t]` = `_seguidas[t]` + 1 se `ok`, senão 0.
4. Se `_seguidas[t]` < `Balanceador.iters_nivel` = 50 (`knobs.py:395`): passa para a próxima tarefa.
5. Com `_seguidas[t]` ≥ 50: `_seguidas[t]` = 0. Depois:
   - Se `_nivel[t]` = `max_nivel[t]`: `_nivel_ter[t]` += 1 e `n_medidas` da célula `terreno` = 0. Só o `CARREGAR` chega aqui: o `ANDAR` no nível máximo tem `ok` falso (passo 4b, seção 5.4b).
   - Senão, se `subnivel(t)` < `_sub_max` (só `PEGAR` e `BOTAR` no subnível 0): `_sub[t]` += 1.
   - Senão: `_nivel[t]` += 1.
   - Nos dois últimos casos, `n_medidas` de todas as células da tarefa = 0.

`subnivel(t)` devolve `_sub[t]` para `PEGAR` e `BOTAR` e `_sub_max` para as outras tarefas (`balanceador.py:325-327`). Então `ANDAR` e `CARREGAR` sobem de nível direto.

O `s` das células não zera na subida. Só `n_medidas` zera. A primeira medida depois da subida substitui `s` por `x` (`balanceador.py:166`).

### 5.9 Travas do DR (passo 10, `_trava_dr`, `balanceador.py:197-202`)

Cada trava só liga. Uma vez `True`, fica `True`.

| Trava | Liga quando |
|---|---|
| `portao0` | `nivel_passou("PEGAR")` ≥ 0 |
| `vies` | `s` de `(ANDAR, andar)` ≥ `Balanceador.portao_andar` = 0,75 |
| `subida` | `_nivel["PEGAR"]` + `_nivel["CARREGAR"]` > 0 |

`nivel_passou(t)` (`balanceador.py:350-356`): `i` é a primeira célula de `t`. `ok` = `s[i]` ≥ portão e `subnivel(t)` = `_sub_max` e (`_nivel[t]` > 0 ou `n_medidas[i]` > 0). Devolve `_nivel[t]` se `ok`, senão `_nivel[t]` − 1.

## 6. Visitas e zona no env

Esta seção descreve como os acumuladores `vis_acum` e `fech_acum` recebem valores e como o `z` age no env.

### 6.1 O critério efetivo das etapas

`CadeiaCmd._alvos` guarda `it_zona` (início −1, `comando.py:303`). Quando `it_zona` ≠ `cur.it`, ele refaz o critério efetivo uma vez por iteração (`comando.py:428-443`):

1. `palma` = `Alvos.tol_palma` repetido para as 7 etapas.
2. Para cada `(nome, campo, largo)` em `Alvos.zona` (`knobs.py:211-220`): `e` = índice da etapa, `v` = `no_degrau(valor final do campo, largo, cur.z[e], 1)` = final + `z[e]` × (largo − final).
3. Se `campo` = `tol_palma`: `palma[e]` = `v`. Senão: `efetivo[campo]` = `v`.
4. `v3_alvos` = `replace(A, tol_palma=tensor(palma), **efetivo)`.
5. `tempos` = [0, 0, `tempo_p2`, `tempo_p3`, `tempo_b1`, `tempo_b2`, `tempo_b3`] efetivos, usados por `alvos.avanca` (`comando.py:442-443`, `comando.py:482`).

Como `z` começa em 1,0, o primeiro critério efetivo é o largo. O critério se refaz só quando `cur.it` difere de `it_zona` (`comando.py:428-432`). O reset inicial já roda `_alvos` uma vez (`mjlab/envs/manager_based_rl_env.py:372`), então `it_zona` vale 0 antes da carga do checkpoint. Depois da carga, `cur.it` vale o valor salvo. Se o valor salvo é diferente de 0, o primeiro passo refaz o critério com o `z` carregado. Se é 0, o critério do reset inicial (feito com `z` = 1,0) fica até o primeiro fecho.

Valores da zona (`knobs.py:174-220`):

| Etapa | Campo | Final (`z` = 0) | Largo (`z` = 1) |
|---|---|---|---|
| `P0` | `tol_palma` | 0,03 m (`knobs.py:183`) | 0,05 m |
| `P1` | `tol_palma` | 0,03 m | 0,05 m |
| `P2` | `tempo_p2` | 0,3 s (`knobs.py:182`) | 0,2 s |
| `P3` | `subida_min_p3` | 0,03 m (`knobs.py:186`) | 0,01 m |
| `P3` | `tol_transporte_p3` | 0,05 m (`knobs.py:187`) | 0,20 m |
| `P3` | `tol_pelve_p3` | 0,03 m (`knobs.py:188`) | 0,10 m |
| `P3` | `tempo_p3` | 0,5 s (`knobs.py:189`) | 0,2 s |
| `B1` | `raio_b1` | 0,10 m (`knobs.py:191`) | 0,20 m |
| `B1` | `tempo_b1` | 0,3 s (`knobs.py:194`) | 0,1 s |
| `B2` | `b2_afasta` | 0,05 m (`knobs.py:196`) | 0,02 m |
| `B2` | `tempo_b2` | 0,5 s (`knobs.py:197`) | 0,2 s |
| `B3` | `tol_pelve_b3` | 0,03 m (`knobs.py:198`) | 0,06 m |
| `B3` | `tol_tronco_b3_graus` | 5,0 graus (`knobs.py:199`) | 10,0 graus |
| `B3` | `tol_palmas_b3` | 0,05 m (`knobs.py:200`) | 0,10 m |
| `B3` | `tempo_b3` | 0,5 s (`knobs.py:201`) | 0,2 s |

Quem lê o critério efetivo `env.v3_alvos`: o fecho do `PEGAR` (`ok_agora_pegar`, `comando.py:41-56`, chamado em `comando.py:472`), o fecho do `BOTAR` (`ok_agora_botar`, `comando.py:73-88`, chamado em `comando.py:479`) e a qualidade da caixa no `CARREGAR`, que lê `tol_transporte_p3` efetivo (`recompensas.py:587`). A distância das palmas ao alvo usa o `Alvos` final `A` (`comando.py:468`). O fecho do `P0` e do `P1` lê `tol_palma[idx]` (`comando.py:50-52`). O fecho do `P2` exige o contato das duas palmas (`comando.py:53`) e usa o tempo efetivo `tempo_p2` por `tempos`.

### 6.2 A visita de uma etapa

Uma visita é o período em que a etapa está aberta. `_alvos` mantém `vis_e` (a etapa da visita corrente) e `t_vis` (o tempo desde a abertura, −1 sem visita) (`comando.py:444-452`):

1. `etapa` = `v3_alvo_idx` se o índice é ≥ 0, não é (`P3` com `pega_fechada`) e não há `v3_perdeu`; senão −1 (`comando.py:447`).
2. `mudou` = `etapa != vis_e`. `acaba` = `vis_e` onde `t_vis` ≥ 0 e `mudou`, senão −1.
3. `vis_e` = `etapa`. Onde `mudou`, `t_vis` = 0,0 se `etapa` ≥ 0, senão −1,0. Onde `t_vis` ≥ 0, `t_vis` += `dt`.
4. `alvos.avanca(idx, ok, relogio, dt, tempos)` devolve `av` (`alvos.py:195-210`). `ok` é a condição de fecho da etapa corrente. Com `ok` falso, o relógio da etapa volta a 0. Com `ok` verdadeiro, ele soma `dt`. `av` = `ok` & `idx` ≥ 0 & relógio ≥ `tempos[idx]` − 1e-9 (`_EPS`, `alvos.py:19`). Com `av`, o índice sobe 1, e depois de `B3` vira −1. Em `comando.py:492-493`, o índice fica em `P3` depois do fecho do `P3`.
5. `fecho` = `av & (t_vis >= 0)`. `venceu` = `~av & (t_vis >= Balanceador.zona_t_visita)`, com `zona_t_visita` = 8,0 s (`knobs.py:407`, `comando.py:484`). `encerra` = `fecho | venceu`.
6. `visitas` = contagem por etapa de `acaba` (≥ 0) junto com `vis_e[encerra]`. `fechos` = contagem por etapa de `vis_e[fecho]` (`comando.py:486-487`).
7. `cur.vis_acum[e]` += visitas e `cur.fech_acum[e]` += fechos (`comando.py:488-490`).
8. `t_vis` = −1 onde `encerra`. A visita só reabre quando a etapa muda (`comando.py:491`).

No reset do episódio, a visita aberta (`t_vis` ≥ 0) conta uma visita sem fecho para a etapa `vis_e`. Depois `vis_e` = −1 e `t_vis` = −1,0 (`comando.py:646-650`).

### 6.3 Visita e travessia de terreno do ANDAR

O `eventos.sorteia_fatia` conta, no reset, o episódio de `ANDAR` em terreno que acabou: ele soma as visitas em `vis_ter` e as travessias em `trav_ter`, por coluna (`eventos.py:143-160`). O fecho da iteração lê esses dois vetores no passo 4b (seção 5.4b). O arquivo `03-reset-terreno-e-perturbacoes.md`, seção 4.6, descreve a regra, a ordem de leitura da posição e os números.

## 7. Consultas do env ao currículo

| Consulta | Devolve | Quem chama | Quando |
|---|---|---|---|
| `fracoes_fatias()` | Tensor float64 (3,) com a fração de cada fatia | `sorteia_fatias`; `curriculo.v3` (log) | Reset (`fatia`); cada chamada de `v3` |
| `sorteia_fatias(n, g)` | `n` índices de fatia por `torch.multinomial(fracoes_fatias, n, replacement=True, generator=g)` | `eventos.sorteia_fatia` (`eventos.py:161`) | Reset do episódio, por lote de envs |
| `fracoes_celulas(tarefa)` | (`ids` das células abertas, frações float64). Sem abertas: (`[]`, vazio) | `sorteia_celulas`; `eventos._fracao_terreno` | Reset |
| `sorteia_celulas(tarefa, n, g, sem_tipo)` | `n` `celula_id` sorteados entre as abertas | `TwistV3._resample_command` (`comando.py:918`) | Cada reamostra do comando: reset do episódio, intervalo `Comando.reamostra_s` (não vale para o `CARREGAR` da fatia `B`) e abertura de trecho |
| `id_celula(tarefa, tipo)` | O índice em `LAYOUT` | `comando.py:535`, `comando.py:538`, `comando.py:768`, `comando.py:915`; `recompensas.py:582` | Conforme o chamador |
| `nivel(tarefa)` | `_nivel[tarefa]` | `comando.py:920`; `eventos.py:344`, `eventos.py:358`, `eventos.py:403`; `observacoes.py:333` | Conforme o chamador |
| `subnivel(tarefa)` | `_sub[tarefa]` em `PEGAR` e `BOTAR`; `_sub_max` nas outras | `eventos.py:193`, `eventos.py:346`, `eventos.py:403` | Reset; abertura do `BOTAR` |
| `nivel_passou(tarefa)` | Ver 5.9 | `eventos.py:344`; `_trava_dr` | Reset; fecho |
| `nivel_terreno(tarefa)` | `CARREGAR`: `_nivel_ter`. `ANDAR`: o piso da média de `linha[1:]` | `eventos.py:168` (só o do `CARREGAR`); `balanceador.py:225`, `balanceador.py:240`; `curriculo.py:138` (log); `estado()` | Reset; fecho; log |
| `niveis_abertos(tarefa)` | `_nivel[tarefa]` + 1 | `CadeiaCmd._publica` (`comando.py:515`) | Abertura de cada trecho |
| `degrau(tarefa)` | `_degrau[tarefa]` | `CadeiaCmd._publica` (`comando.py:516`); `curriculo.v3` | Abertura de cada trecho; `v3` |
| `degrau_global()` | `min(degrau(t))` das 4 tarefas | `comando.py:553`, `comando.py:623`; `observacoes.py:189`; `dr()` | Abertura do `BOTAR`; todo passo; todo passo de percepção; `dr()` |
| `restricao()` | Ver seção 9 | `dr()`; `comando.py:773`; `recompensas.py:394`, `recompensas.py:456`; `v3` | Conforme o chamador |
| `dr()` | O dict da seção 10 | `curriculo.v3` e os eventos de DR | Reset; passo (`forca_caixa`, `rfi`) |
| `z`, `vis_acum`, `fech_acum` | As listas de zona | `comando.py:427-490`, `comando.py:534`, `comando.py:537`, `comando.py:649`; `observacoes.py:317` (o canal `z` do termo `privilegiados` vale `z[v3_alvo_idx]`, e 0 sem etapa; o termo entra só no grupo `critic`, `observacoes.py:416`) | Passo; reset |
| `celulas[i].s`, `.aberta`, `.tipo` | Os campos da célula | `recompensas.py:916`; `comando.py:923` | Passo; troca de segmento |
| `it` | Contagem de fechos | `comando.py:428` | Passo |
| `max_nivel[t]` | O máximo da tarefa | `observacoes.py:333` (o termo `niveis` devolve `nivel(t) / max_nivel[t]` das 4 tarefas de `balanceador.TAREFAS`, importada como `CADEIA` em `observacoes.py:33`; o termo `niveis` e o termo `degrau` entram só no grupo `critic`, `observacoes.py:412-413`) | Passo de observação |

### 7.1 Frações de fatia (`fracoes_fatias`, `balanceador.py:282-293`)

1. `abertas` = as fatias com alguma célula aberta (por `FATIA_DE`).
2. `teto[f]` = 1,0 para a fatia 0. Para as outras: `Balanceador.rampa_fatia_inicio` + `Balanceador.rampa_fatia_passo` × `_rampa[f]` = 0,05 + 0,02 × `_rampa[f]` (`knobs.py:397-398`).
3. `resto` = 1,0. As fatias abertas entram em ordem crescente de `teto` (ordenação estável). A fatia `j`-ésima recebe `min(teto[f], resto / (len(abertas) − j))`, e `resto` perde essa fração.
4. Fatia fechada recebe 0.

Com só `ANDAR` aberta, a fração dela é 1. Com todas abertas e tetos grandes, as frações são iguais (1/3 cada). O teto de uma fatia nova corta a fração dela, e o excesso vai para as fatias de teto maior.

### 7.2 Frações de célula (`fracoes`, `deficit`, `balanceador.py:37-45`, `balanceador.py:295-299`)

Para as `n` células abertas da tarefa, `D[i]` = max(portão[i] − `s[i]`, 0). A fração é:

`F[i]` = `piso_sub` + (1 − `n` × `piso_sub`) × (`D[i]` + `eps`) / Σ(`D` + `eps`)

com `Balanceador.piso_sub` = 0,10 e `Balanceador.eps` = 0,02 (`knobs.py:389-390`). Se todos os déficits são iguais, `F[i]` = 1/`n`.

Na célula `terreno` do `ANDAR` o déficit `D` não é a distância da nota ao portão. É `1 − média(linha[1:]) / 12`, a distância das 8 linhas ao topo (`balanceador.py:141-143`). A nota `s` dessa célula não entra: o robô parado na plataforma tira cerca de 0,5 e zerava o déficit (comentário do código). A nota segue no refino (`_passa_media`).

Fração da célula `terreno` do `ANDAR`, com as duas células abertas (`andar` e `terreno`, `n` = 2), em função da média `m` das linhas e da nota `s` de `andar` (derivado: a fórmula acima, com `piso_sub` = 0,10 e `eps` = 0,02; `D` de `andar` = max(0,75 − `s`; 0); `D` de `terreno` = 1 − `m` / 12):

| `m` (média das linhas) | `D` de `terreno` | `s` ≥ 0,75 | `s` = 0,60 | `s` = 0,40 |
|---|---|---|---|---|
| 0 | 1,000 | 0,885 | 0,786 | 0,687 |
| 3 | 0,750 | 0,880 | 0,755 | 0,640 |
| 6 | 0,500 | 0,870 | 0,703 | 0,567 |
| 9 | 0,250 | 0,845 | 0,591 | 0,438 |
| 12 | 0,000 | 0,500 | 0,184 | 0,141 |

Enquanto só a fatia `ANDAR` está aberta, essa fração é a fração de episódios de terreno do treino inteiro. Com os dois déficits em 0, cada célula recebe 0,50 (derivado: (0 + `eps`) / (0 + 2 × `eps`)).

`sorteia_celulas` com `sem_tipo` zera a fração das células desse tipo (`terreno`) sem renormalizar, e `torch.multinomial` normaliza o resto (`balanceador.py:310-316`).

### 7.3 Terreno do episódio (`eventos.sorteia_fatia`, `eventos.py:133-175`)

No reset, depois de sortear a fatia:
- `p` = a fração da célula `terreno` de `ANDAR` (se a fatia é `ANDAR`), senão a de `CARREGAR`. `_fracao_terreno` devolve 0 se a célula está fechada (`eventos.py:36-40`, `eventos.py:165-166`).
- `terreno_ep` = uniforme[0, 1) do `v3_gerador` < `p` (`eventos.py:167`).
- `nivel` = `nivel_terreno("CARREGAR")`, o mesmo para todos os envs (`eventos.py:168`). O `ANDAR` não usa `nivel`: ele lê a linha por coluna, `Curriculo.linha` (`eventos.py:169`).
- `escolhe_celula_terreno` (`eventos.py:43-74`): o terreno vale só na fatia `ANDAR` ou `B` com `terreno_ep`. Quatro uniformes saem do `v3_gerador` por episódio (`eventos.py:60`).
  - Coluna do `ANDAR`: o `u[0]` passa pela inversa da soma acumulada de pesos por coluna, e a coluna vai de 1 a 8 (`eventos.py:63-65`). O peso é `fracoes(atraso, Terreno.piso_coluna, Balanceador.eps)` com `atraso` = 1 − `linha[c]` / 12, `piso_coluna` = 0,05 e `eps` = 0,02 (`knobs.py:443`). A coluna atrasada recebe mais; a coluna vencida desce até perto de `piso_coluna`. Com as linhas iguais, cada coluna vale 1/8.
  - Linha do `ANDAR`: `⌊linha[coluna] + u[1]⌋`, com teto em 12 (`eventos.py:66`). É arredondamento estocástico: a média das linhas sorteadas de uma coluna é a `linha` dela. A linha 2,4 sai 2 com chance 0,6 e 3 com chance 0,4 (derivado: conta).
  - Coluna da `B`: uniforme entre as colunas de `Terreno.linha_max_carregar`. Linha da `B`: uniforme inteira em 0..min(`nivel_terreno("CARREGAR")`, teto da coluna). Sem terreno, coluna 0 e linha 0.
  - `sorteia_fatia` escreve a linha em `terrain_levels`, a coluna em `terrain_types` e a origem do env em `scene.env_origins` (`eventos.py:173-175`). O `twist` lê `terrain_types` ≠ 0 como episódio de terreno (`comando.py:914`).

Pesos das colunas do `ANDAR` (derivado: fórmula acima; a coluna "freq." vem de `escolhe_celula_terreno` com `torch.Generator().manual_seed(0)` e 200 000 sorteios, todos de `ANDAR` com `terreno_ep` verdadeiro). Linhas do exemplo: `escada` 3, `escada_inv` 2, `rampa` 6, `rampa_inv` 5, `escada_aleatoria` 1, `irregular` 8, `obstaculos` 6, `caixas` 10.

| Coluna | Linha | Peso | Freq. (200 000) |
|---|---|---|---|
| `escada` | 3 | 0,147 | 0,147 |
| `escada_inv` | 2 | 0,158 | 0,158 |
| `rampa` | 6 | 0,116 | 0,116 |
| `rampa_inv` | 5 | 0,126 | 0,126 |
| `escada_aleatoria` | 1 | 0,168 | 0,169 |
| `irregular` | 8 | 0,095 | 0,095 |
| `obstaculos` | 6 | 0,116 | 0,116 |
| `caixas` | 10 | 0,074 | 0,074 |

- As três colunas de escada (`escada`, `escada_inv`, `escada_aleatoria`) somam 0,474 no exemplo e 0,375 com as linhas iguais (3 × 1/8). As duas de rampa somam 0,242 no exemplo e 0,250 com as linhas iguais.
- O mínimo por coluna é `piso_coluna` = 0,05 mais um resto. Com sete colunas na linha 12 e `caixas` na linha 0, cada coluna vencida pesa 0,060 (`0,05 + 0,60 × 0,02 / 1,16`) e `caixas` pesa 0,578. O peso nunca chega a 0,05 (derivado: conta e simulação de 200 000 sorteios, frequências 0,060 e 0,578).
- Com as 8 linhas em 12, os pesos voltam a 1/8: o atraso é igual em todas as colunas.

### 7.4 Níveis, subníveis e a cena

- `ANDAR` (`comando.py:125-147`): `lado` = min(`andar_lado_inicio` + `andar_passo` × nível, `lado_teto_max`); `frente` = min(`andar_inicio` + `andar_passo` × nível, `frente_tetos[-1]`); `tras` = min(`andar_inicio` + `andar_passo` × nível, `tras_teto_max`); `giro` = min(`andar_giro_inicio` + `andar_passo` × nível, `giro_tetos[-1]`). Os valores são `Comando.andar_lado_inicio` = 0,2; `andar_inicio` = 0,3; `andar_passo` = 0,1; `andar_giro_inicio` = 0,3; `lado_teto_max` = 0,3; `tras_teto_max` = 0,5; `frente_tetos[-1]` = 1,0; `giro_tetos[-1]` = 0,5 (`knobs.py:122-134`).
- `CARREGAR` (`comando.py:148-150`): `frente` = `Comando.frente_tetos[nível]` = (0,4; 0,55; 0,7; 0,85; 1,0); `tras` = min(`frente`, `Comando.tras_teto_max` = 0,5); `lado` = min(`frente`, `Comando.lado_teto_max` = 0,3); `giro` = `Comando.giro_tetos[min(nível, 3)]` = (0,3; 0,4; 0,5; 0,5) (`knobs.py:122-125`).
- Sorteio da velocidade de um segmento (`sorteia_velocidade`, `comando.py:118-158`). Acontece a cada reamostra do `twist` (reset do episódio, intervalo `Comando.reamostra_s` ou abertura de trecho) para os envs que andam (`comando.py:897-925`). O `CARREGAR` da fatia `B` não reamostra por intervalo: `time_left` = ∞ (`comando.py:939-941`).
  - `ANDAR`, tipo `andar`: `r` = raiz de uniforme[0, 1); `ang` = π × (uniforme − 0,5); o semieixo x é `frente` com probabilidade `frente`/(`frente` + `tras`), senão `−tras`. `vx` = semieixo × `r` × cos(`ang`); `vy` = `lado` × `r` × sen(`ang`) (`comando.py:136-142`). `wz` = `giro` × (2 × uniforme − 1) (`comando.py:143`). Com ‖(vx, vy)‖ < `Comando.linear_min` = 0,1 m/s, o linear vira 0. Com |`wz`| < `Comando.giro_min` = 0,1 rad/s, o giro vira 0 (`knobs.py:132-133`, `comando.py:144-145`). Com probabilidade `Comando.andar_parado` = 0,10, o comando inteiro vira 0 (`knobs.py:135`, `comando.py:146`).
  - `ANDAR`, tipo `terreno` (27/11, T19): só `vx` = `frente` × `r` (`comando.py:137-138`); `vy` e `wz` valem 0 no sorteio. O `frente` recebe o teto `Comando.terreno_v_max` = 1,0 m/s (`knobs.py:148`, `comando.py:131-132`). O `wz` vem do rumo: a `TwistV3` o recalcula a cada passo nos envs de terreno com comando linear (`comando.py:943-948`). O arquivo `04-comando-de-velocidade.md` descreve o rumo e traz os números do comando.
  - `CARREGAR`, tipos `reto`, `terreno`, `arco`: direção uniforme entre 4 (frente, trás, lado esquerdo, lado direito); módulo uniforme em [piso; teto], com piso `Comando.piso_velocidade` = 0,25 m/s para frente e trás e `Comando.piso_lado` = 0,2 m/s para os lados (`knobs.py:126-127`, `comando.py:102-110`, `comando.py:154-155`). Tipos `arco` e `giro_lugar`: `wz` = ±uniforme[`piso_velocidade`; `giro`], sinal uniforme (`comando.py:113-115`, `comando.py:156-157`). No tipo `terreno`, `frente` e `tras` recebem o teto `terreno_v_max` (`comando.py:151-152`). O tipo `giro_lugar` não sorteia linear (fica 0).
- `PEGAR` e fatias `B` e `C` (`eventos.sorteia_topo_e_massa`, `eventos.py:82-109`): `(h_k, m_k)` = `Niveis.pegar[nivel("PEGAR")]`; `h_0` = `Niveis.pegar[0][0]`; `extra` = `Niveis.subnivel_extra[subnivel("PEGAR")]`. Fatia `B`: topo uniforme em [`h_k` + `extra`; `h_0` + `extra`], massa uniforme em [`Cena.caixa_massa_base`; `m_k`]. Fatia `C`: topo uniforme em [max(`h_k`, `Niveis.topo_min_c`) + `extra`; `h_0` + `extra`], massa uniforme em [`caixa_massa_base`; `m_j`], com `m_j` = `Niveis.pegar[nivel_passou("PEGAR")][1]`, ou `caixa_massa_base` se `nivel_passou` < 0. Fatia `ANDAR`: topo 0 e massa `caixa_massa_base`. Valores: `Niveis.pegar` = ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)), `topo_min_c` = 0,30, `subnivel_extra` = (0,10; 0,0), `Cena.caixa_massa_base` = 1,0 (`knobs.py:417`, `knobs.py:419`, `knobs.py:422`, `knobs.py:80`).
- A base da pega (`eventos._base_da_pega`, `eventos.py:192-194`): a distância ao recuo recebe `(2u − 1) × Alvos.faixa_recuo × Niveis.subnivel_jitter[subnivel("PEGAR")]`, com `faixa_recuo` = 0,08 m e `subnivel_jitter` = (0,0; 1,0) (`knobs.py:227`, `knobs.py:424`).
- A caixa em x, nas fatias `B` e `C`, no reset do episódio (`eventos.py:358`): `Cena.jitter_x_max[nivel("PEGAR")]` × uniforme, com `jitter_x_max` = (0,20; 0,20; 0,20; 0,15; 0,08) (`knobs.py:99`).
- `BOTAR` (`eventos.sorteia_destino`, `eventos.py:373-390`): `longe` = uniforme × `Cena.jitter_x_max[nivel("BOTAR")]`; `topo` = `Niveis.botar[nivel]` + (`botar[0]` − `botar[nivel]`) × uniforme. Se `subnivel("BOTAR")` < `_sub_max`: `longe` = uniforme × `Niveis.subnivel_longe[subnivel]` e `topo` = `botar[0]` + `subnivel_extra[subnivel]`. Valores: `Niveis.botar` = (0,55; 0,45; 0,30; 0,15; 0,04), `subnivel_longe` = (0,10; 0,20) (`knobs.py:418`, `knobs.py:423`).

### 7.5 Freio e degrau publicados

`_publica` escreve, por env e por trecho, `v3_freio` = `niveis_abertos(tarefa)` e `v3_degrau` = `degrau(tarefa)`. Para tarefas fora das quatro primeiras, ambos valem 0 (`comando.py:508-519`). Os valores ficam no env até o trecho seguinte.

Consumidores de `v3_degrau`: `por_degrau` (`recompensas.py:450`), `observacoes.degrau` (valor ÷ 5, `observacoes.py:340`) e `terminacoes.batente` (liga quando `v3_degrau` ≥ `Refino.degrau_batente` = 1, `knobs.py:547`, `terminacoes.py:164`, `terminacoes.py:177-181`). Consumidor de `v3_freio`: `recompensas.freio` (`recompensas.py:919`).

A recompensa `freio` usa também a nota das células: `rampa` = min(1, média de `s` das células abertas da tarefa ÷ `Balanceador.portao`), com 0 para tarefa sem célula aberta (`recompensas.py:916-918`).

## 8. A função `no_degrau`

`no_degrau(ini, fim, d, degraus)` devolve `ini + (fim − ini) × d / degraus` (`balanceador.py:48-51`). `d` é inteiro ou tensor (E,). É uma interpolação linear de `ini` (em `d` = 0) a `fim` (em `d` = `degraus`).

Todas as chamadas no código lido:

| Grandeza | `ini` → `fim` | `d` | `degraus` | Linha |
|---|---|---|---|---|
| `pd` (ganho do PD, par de faixa) | 1 → `DR.pd` = (0,9; 1,1) | degrau global | `Balanceador.degraus` = 5 | `balanceador.py:368-369`, `balanceador.py:385` |
| `forca_motor` | 1 → `DR.forca_motor` = (0,9; 1,1) | degrau global | 5 | `balanceador.py:386` |
| `forca_motor_bracos` | 1 → `DR.forca_motor_bracos` = (0,8; 1,1) | degrau global | 5 | `balanceador.py:387` |
| `solref` | `DR.solref_base` = 0,02 → `DR.solref` = (0,015; 0,03) | degrau global | 5 | `balanceador.py:388` |
| `rfi` | `DR.rfi[0]` = 0,05 → `DR.rfi[1]` = 0,1, só com a trava `subida` | degrau global | 5 | `balanceador.py:384` |
| Critério efetivo da zona | valor final do campo → largo | `z[e]` | 1 | `comando.py:436` |
| Viés do alvo do `BOTAR` (xy e z) | `Estimador.vies_alvo_botar_xy` = (0,01; 0,05) e `vies_alvo_botar_z` = (0,005; 0,03) | degrau global | 5 | `comando.py:553-555` |
| Deriva da odometria: `deriva_odo`, `deriva_rumo`, `deriva_pos_giro`, `deriva_rumo_giro` | Os pares de `Estimador`: (0,002; 0,0075), (0,002; 0,005), (0,004; 0,008), (0,0035; 0,014) | degrau global | 5 | `comando.py:247-250`, `comando.py:623-625` |
| Limite da deriva de posição do `CARREGAR` | `Comando.deriva_pos_inicio` = 0,25 → `deriva_pos_m` = 0,10 | `restricao` | 1 | `comando.py:773` |
| `sigma_zero` angular | `Recompensa.sigma_zero_ang_inicio` = 0,5 → `sigma_zero_ang` = 0,15 | `restricao` | 1 | `recompensas.py:394` |
| `por_restricao` | `ini` → 1,0 | `restricao` | 1 | `recompensas.py:456` |
| `por_degrau` | `ini` → `fim` | `v3_degrau` por env | 5 | `recompensas.py:450` |
| Magnitudes do estimador | Os pares de `Estimador` (ver baixo) | degrau global | 5 | `observacoes.py:84-90`, `observacoes.py:189` |

Chamadores de `por_restricao` (`recompensas.py:453-458`): o parâmetro `so` (nome de uma tarefa; `None` vale para todas) limita o fator aos envs dessa tarefa, e nos outros envs o fator vale 1. Os termos de estilo `giro_xy`, `vel_junta`, `acel_junta` e `energia` não usam `so`, com `ini` = `Recompensa.estilo_inicio` = 0,25 (`knobs.py:260`, `recompensas.py:672-697`). Os desvios `desvio_bracos`, `desvio_cintura` e `desvio_quadril` usam `so="CARREGAR"` (`recompensas.py:707-722`): no `ANDAR` o fator vale 1 e o peso é cheio desde a iteração 0. As cercas (`cerca_giro`, `cerca_tronco`, `cerca_com`, `cerca_torque`, `cerca_pelve`) usam `ini` = 0,0 e `so="ANDAR"` (`recompensas.py:744-788`): nas outras tarefas o fator vale 1.

Chamadores de `por_degrau`: `incl_lim` com `Alvos.incl_inicio_graus` = 30,0 → `incl_fim_graus` = 15,0 (`recompensas.py:464`, `knobs.py:203-204`); o fator de pouso com `Recompensa.sigma_pouso` = (1,0; 0,35) (`recompensas.py:549`, `knobs.py:368`); `contato_laje` com `Recompensa.laje_joelho_N` = (50,0; 10,0) (`recompensas.py:890`, `knobs.py:302`); `sacudida` e `aperto` com 0,0 → 1,0 (`recompensas.py:944`, `recompensas.py:955`); `terminacoes.chegada` com `Terminacao.chegada` = (1,0; 0,25) (`terminacoes.py:149`, `knobs.py:381`).

Magnitudes do estimador (`observacoes.py:84-90`): `sigma_pos` (`Estimador.sigma_pos[0]` → 0,0), `sigma_pos_rel` (0,0 → `sigma_pos[1]`), `ori_mediana_graus` e `ori_p90_graus` (os dois pares de `Estimador.ori_graus`), `ori_leitura_graus`, `perda`, `vies_meia`, `vies_pos` e `rajada_taxa`, cada um com o par de `Estimador` de mesmo nome (`knobs.py:459-476`).

## 9. O valor `restricao`

`restricao()` = o mínimo, sobre as quatro tarefas, de `_nivel[t] / max_nivel[t]` (`balanceador.py:345-348`). Os máximos são `ANDAR` 7, `CARREGAR` 4, `PEGAR` 4, `BOTAR` 4 (seção 2.2). Ele usa só o nível. Subnível, linha de terreno e degrau não entram. Ele vale 0 enquanto alguma tarefa está no nível 0, e nunca desce, porque os níveis nunca descem.

Consumidores: o teto do atraso em `dr()` (`balanceador.py:381-382`), o limite da deriva de posição (`comando.py:773`), o `sigma_zero` angular (`recompensas.py:394`), `por_restricao` (`recompensas.py:456`) e o log `restricao` (`curriculo.py:141-142`).

## 10. A randomização de domínio (`dr()`)

`dr()` devolve um dict com tipos do Python. Sem `k_dr`, devolve `{}` (`balanceador.py:363-364`). Com `g` = degrau global e `n` = `Balanceador.degraus` = 5 (`balanceador.py:365-366`):

| Chave | Valor | Referência |
|---|---|---|
| `vies_imu` | `_dr["vies"]` | `balanceador.py:376` |
| `atrito_pad` | `_dr["portao0"]` | `balanceador.py:377` |
| `atrito_laje` | `_dr["portao0"]` | `balanceador.py:378` |
| `mu_min_pad` | Sem `portao0`: `DR.atrito_pad[0]` = 0,6. Com `portao0`: max(`DR.mu_pad_min` = 0,5; 0,6 − `DR.passo_mu_pad` × (`_nivel["PEGAR"]` + `_nivel["CARREGAR"]`)), com `passo_mu_pad` = 0,025 | `balanceador.py:371-373`, `knobs.py:523`, `knobs.py:525-526` |
| `com_caixa` | `[0,0; 0,0]` se `DR.com_caixa` é falso (valor `False`, `knobs.py:527`). Se verdadeiro: `DR.com_caixa_frac_max` = (0,3; 0,2) com a trava `subida`, senão `com_caixa_frac0` = (0,2; 0,2) com `portao0`, senão (0; 0) | `balanceador.py:374`, `balanceador.py:380`, `knobs.py:528-529` |
| `atraso_max` | `DR.atraso_refino[1]` = 3 se `g` ≥ `DR.degrau_atraso_refino` = 3; senão `ceil(DR.atraso_portao[1] × restricao − 1e-9)` com `atraso_portao[1]` = 2. Então vale 0 com `restricao` = 0, 1 com `restricao` em (0; 0,5], 2 com `restricao` em (0,5; 1] | `balanceador.py:381-382`, `knobs.py:512-514` |
| `forca_caixa` | `_dr["subida"]` | `balanceador.py:383` |
| `rfi` | `no_degrau(DR.rfi[0], DR.rfi[1], g, n)` com `subida`, senão 0,0 | `balanceador.py:384` |
| `pd`, `forca_motor`, `forca_motor_bracos`, `solref` | Um par [`no_degrau(ini, fim[0], g, n)`, `no_degrau(ini, fim[1], g, n)`]. Com `g` = 0 o par vale (`ini`, `ini`) | `balanceador.py:368-369`, `balanceador.py:385-388` |

Como os eventos leem o dict:

| Evento | Modo | O que faz com o dict | Linha |
|---|---|---|---|
| `atrito_pad` | reset | Com `atrito_pad` verdadeiro: μ dos pads uniforme em [`mu_min_pad`; `DR.atrito_pad[1]` = 1,0]. Senão: fixo em `Cena.pad_atrito[0]` = 1,0. Usa `dr.geom_friction` com distribuição `uniform` e operação `abs` | `eventos.py:464-475`, `mjlab/envs/mdp/dr/geom.py:123-124`, `knobs.py:81` |
| `atrito_laje` | reset | Com `atrito_laje` verdadeiro: μ da laje uniforme em `DR.atrito_laje` = (0,3; 1,0). Senão: fixo em `Cena.prateleira_atrito[0]` = 1,0 | `eventos.py:478-489`, `knobs.py:91`, `knobs.py:524` |
| `vies_imu` | reset | Com `vies_imu`: giro uniforme em ±`DR.vies_imu_giro` = 0,05 rad/s por eixo e inclinação do IMU uniforme em ±`DR.vies_imu_incl_graus` = 1,0 grau (x e y; z = 0). Senão: giro 0 e inclinação identidade | `eventos.py:492-509`, `knobs.py:521-522` |
| `pd` | reset (fora do `play`) | `dr.pd_gains`, `kp` e `kd` escalados pela faixa `pd`, uniforme pelo `v3_gerador` | `eventos.py:524-531`, `env_cfg.py:169-170`, `env_cfg.py:181` |
| `forca_motor` | reset (fora do `play`) | `dr.effort_limits`, escala pelas faixas `forca_motor` (pernas e cintura) e `forca_motor_bracos` (braços) | `eventos.py:535-545`, `env_cfg.py:173-175` |
| `solref` | reset (fora do `play`) | `solref[0]` das geoms de pés, pads, laje e caixa = uniforme em `solref`, um valor por env | `eventos.py:549-565`, `env_cfg.py:177-180` |
| `com_caixa` | reset | Existe no cfg só se `DR.com_caixa` é verdadeiro. Com `fx, fz` = `dr()["com_caixa"]`: o deslocamento do CoM da caixa é uniforme em ±`fx` nos eixos x e y e em ±`fz` em z, multiplicado pela meia aresta `v3_meia` do env | `eventos.py:569-581`, `env_cfg.py:182-183` |
| `forca_caixa` | passo (fora do `play`) | Sem `forca_caixa` verdadeiro: não age. Com ele: aplica o impulso do mjlab e zera o wrench fora de (`CARREGAR` & `pegou` & ~`perdeu`). O impulso sorteia cada eixo em `DR.forca_caixa_N` = (−11,5; 11,5) N, com duração `DR.forca_caixa_dur_s` = (0,1; 0,3) s e intervalo `DR.forca_caixa_periodo_s` = (3,0; 8,0) s (`knobs.py:531-533`). A força não escala pelo degrau | `eventos.py:584-602`, `env_cfg.py:207-209` |
| `rfi` | passo (fora do `play`) | `qfrc_applied` por junta = (2u − 1) × `rfi` × `TAU_MAX` da junta, com u uniforme em [0, 1) pelo `v3_gerador` | `eventos.py:605-615`, `env_cfg.py:210` |

O teto do atraso chega aos buffers por `curriculo.v3` (seção 11).

## 11. O termo de currículo `v3`

`curriculo.v3(env, env_ids)` roda a cada chamada do mjlab no início de `_reset_idx` (seção 3). Ele faz, em ordem (`curriculo.py:117-143`):

1. `garante_v3(env)` e `cur = env.v3_cur`.
2. O fecho da iteração, se `it != v3_iter` (seção 5.1).
3. `dr = cur.dr()` e `teto = dr["atraso_max"]`.
4. Se `teto` ≠ `DR.atraso_inicio[1]` (= 0, `knobs.py:511`): `eventos.atraso(env, env_ids, teto, Contrato.decimacao)`. A função põe `max_lag` = `teto × decimacao` (`decimacao` = 4, `knobs.py:55`) nos buffers de atraso dos atuadores e `max_lag` = `teto` nos buffers de atraso das observações (`eventos.py:244-258`). A chamada roda em todo reset em que `teto` ≠ 0. Ela ignora `env_ids`: os buffers de atraso são os do env inteiro (`eventos.py:252-258`).
5. O dict de log.

Além do que o termo escreve, `v3` roda `garante_v3` (seção 3) e grava `env.v3_iter` = `it` (`curriculo.py:122`). `v3_freio` e `v3_degrau` não são escritos aqui. Eles são de `CadeiaCmd._publica` (`comando.py:517-518`).

`curriculo.mascara(env, tarefas)` devolve um tensor (E,) booleano: verdadeiro onde `v3_tarefa` está em `tarefas` (`curriculo.py:93-100`). Recompensas e terminações o importam e aplicam com `torch.where` (`recompensas.py:365`, `terminacoes.py:112`, `terminacoes.py:125`, `terminacoes.py:151`, `terminacoes.py:183`).

Chaves do log (`curriculo.py:127-142`), publicadas pelo mjlab como `Curriculum/v3/<chave>` em `extras["log"]` (`mjlab/managers/curriculum_manager.py:97-107`, `mjlab/envs/manager_based_rl_env.py:577-579`):

| Chave | Valor |
|---|---|
| `frac_ANDAR`, `frac_B`, `frac_C` | `fracoes_fatias()` |
| `s_<tarefa>_<tipo>` (8 chaves, uma por célula) | A nota `s` da célula |
| `nivel_<tarefa>` (4) | `nivel(t)` |
| `degrau_<tarefa>` (4) | `degrau(t)` |
| `subnivel_PEGAR`, `subnivel_BOTAR` | `subnivel(t)` |
| `z_<etapa>` e `taxa_<etapa>` (7 + 7), com `<etapa>` em `P0`..`P3`, `B1`..`B3` | `z[e]` e `taxa[e]` (−1 = sem medida) |
| `nivel_terreno_ANDAR`, `nivel_terreno_CARREGAR` | `nivel_terreno(t)` (`ANDAR`: o piso da média das 8 linhas) |
| `linha_<coluna>` (8 chaves), com `<coluna>` em `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `irregular`, `obstaculos`, `caixas` | `linha[c]` do `ANDAR`, em número real (`curriculo.py:139-140`) |
| `atraso_max` | `teto` |
| `mu_min_pad` | `dr["mu_min_pad"]` |
| `degrau_global` | `degrau_global()` |
| `restricao` | `restricao()` |

O log não traz `_seguidas`, `_cont`, `_rampa` nem `it`. `terminacoes`, `recompensas` e `observacoes` lêem o estado direto de `v3_cur`.

O mjlab guarda o dict devolvido por `v3` em `_curriculum_state` e o publica no reset de cada grupo de envs (`mjlab/managers/curriculum_manager.py:95-117`).

## 12. Estado no checkpoint e carga

### 12.1 Salvar

`RunnerV3.save` acrescenta `v3_curriculo` = `env.unwrapped.v3_cur.estado()` ao `infos` e chama o `save` do mjlab (`runner.py:88-90`). O `save` do mjlab acrescenta `env_state.common_step_counter` e grava `infos` no checkpoint (`mjlab/rl/runner.py:73-80`). O `save` roda quando `it % save_interval == 0`, com `save_interval` = 50 iterações do PPO (`mjlab/tasks/velocity/config/g1/rl_cfg.py:43`, `rsl_rl/runners/on_policy_runner.py:128-129`), e uma vez no fim do treino (`rsl_rl/runners/on_policy_runner.py:133`). As duas chamadas exigem `logger.writer` diferente de `None` (`rsl_rl/runners/on_policy_runner.py:128`, `rsl_rl/runners/on_policy_runner.py:132`). `tarefa.py:46-50` não muda `save_interval`.

`estado()` (`balanceador.py:392-405`) devolve:

| Chave | Conteúdo |
|---|---|
| `it` | `it` |
| `celulas` | Lista de {`aberta`, `s`, `n`} das 8 células |
| `acum` | Lista de [`_acum_n`, `_acum_suc`] por célula |
| `tarefas` | Por tarefa: `nivel`, `nivel_ter`, `seguidas`, `degrau`, `cont`, `sub` (`subnivel(t)`). O `nivel_ter` do `ANDAR` é `nivel_terreno("ANDAR")`, o piso da média das linhas (`balanceador.py:398`) |
| `dr` | As três travas |
| `it_abertura` | `_it_abertura` |
| `rampa` | `_rampa` |
| `zona` | {`z`, `visitas` (`vis_acum`), `fechos` (`fech_acum`)} |
| `terreno` | {`linha` (`linha`), `visitas` (`vis_ter`), `travessias` (`trav_ter`)}, uma lista de 9 posições cada (`balanceador.py:404`) |

O estado não leva `taxa`, `_portao`, `max_nivel`, `_max_ter`, `v3_iter`, `it_zona` nem nenhum estado de episódio (`runner.py:13-15`).

### 12.2 Carregar

`RunnerV3.load` (`runner.py:92-112`):
1. Se `load_cfg` é `None` e alguma chave do crítico do checkpoint falta no crítico atual ou tem forma diferente: carrega só ator e iteração (`load_cfg` = `{"actor": True, "critic": False, "optimizer": False, "iteration": True}`) e imprime um aviso (`runner.py:96-101`). Com `load_cfg` dado pelo chamador, não há essa troca.
2. Chama o `load` do mjlab, que restaura `common_step_counter` de `infos["env_state"]` (`mjlab/rl/runner.py:138-140`).
3. Lê `infos["v3_curriculo"]`. Se está ausente ou vazio: imprime que o currículo recomeça do zero e retorna sem carregar (`runner.py:103-109`).
4. Senão chama `v3_cur.carrega(estado)` e imprime `it` (`runner.py:110-111`).

`carrega` (`balanceador.py:407-446`):
- Restaura `it`, `aberta`, `s` e `n_medidas` de cada célula, `_acum_n` e `_acum_suc` (`balanceador.py:408-412`).
- Por tarefa: `_nivel[t]` = min(`nivel` salvo, `max_nivel[t]`); `_nivel_ter[t]` = valor salvo, sem teto, para todas as tarefas menos o `ANDAR` (o `nivel_ter` salvo do `ANDAR` não volta: a linha dele é a de `terreno`) (`balanceador.py:417-418`); `_seguidas[t]` = valor salvo ou 0; `_degrau[t]` e `_cont[t]` = valores salvos (`balanceador.py:413-419`).
- `_sub[t]` para `PEGAR` e `BOTAR` = `sub` salvo, ou `_sub_max` se a chave falta (`balanceador.py:420-421`).
- `_dr` = cópia do dict salvo. `_it_abertura` = valor salvo ou `[0, None, None]`. `_rampa` = valor salvo ou `[0, 0, 0]` (`balanceador.py:422-424`).
- `z`, `vis_acum` e `fech_acum` = valores de `zona`, ou `[1,0]*7` e `[0]*7` se a chave falta (`balanceador.py:425-429`).
- Se `len(z)` ≠ `N_ETAPAS`: mantém os 3 primeiros valores de `z`, `vis_acum` e `fech_acum`, completa o resto com `z` = 1,0 e 0, e põe `taxa` = `[-1,0] * 7` (`balanceador.py:436-443`).
- `linha`, `vis_ter` e `trav_ter` = valores de `terreno`, ou 0 em todas as colunas se a chave falta (checkpoint antigo) (`balanceador.py:430-435`). Os acumuladores `vis_ter` e `trav_ter` que os resets somaram antes da carga perdem o valor. O reset inicial não soma nada neles: a guarda `episode_length_buf` > 0 o exclui (seção 4.6 do arquivo 03).
- `taxa` não é restaurada na carga normal: fica em −1,0 até a próxima atualização da zona.

## 13. Knobs

### 13.1 `Balanceador` (`knobs.py:387-412`)

| Knob | Valor | Efeito | Linha de uso |
|---|---|---|---|
| `piso_tarefa` | 0,20 | Nenhum arquivo lido o usa | — |
| `piso_sub` | 0,10 | Piso da fração de cada célula aberta (7.2) | `balanceador.py:299` |
| `eps` | 0,02 | Soma ao déficit na fração da célula (7.2) | `balanceador.py:299` |
| `portao` | 0,75 | Portão das células fora de `ANDAR`; teto da rampa do `freio` | `balanceador.py:98`, `recompensas.py:918` |
| `portao_andar` | 0,75 | Portão da célula `(ANDAR, andar)`; liga a trava `vies` | `balanceador.py:97-98` |
| `portao_andar_terreno` | 0,50 | Portão da célula `(ANDAR, terreno)`. Ele entra no refino (`_passa_media`); não abre a célula e não move a linha | `balanceador.py:97` |
| `alfa` | 0,05 | Taxa da média móvel da nota | `balanceador.py:166` |
| `iters_nivel` | 50 | Fechos seguidos no portão para subir (subnível, nível ou linha de terreno do `CARREGAR`) | `balanceador.py:266` |
| `nivel_abre_b` | 5 | Nível do `ANDAR` que abre a fatia `B` (reto e `pegar`) | `balanceador.py:209` |
| `rampa_fatia_inicio` | 0,05 | Teto inicial da fração de uma fatia nova | `balanceador.py:288` |
| `rampa_fatia_passo` | 0,02 | Quanto o teto sobe por degrau de rampa | `balanceador.py:288` |
| `rampa_fatia_its` | 12 | Fechos por degrau da rampa | `balanceador.py:192` |
| `iters_refino` | 100 | Fechos seguidos no portão para subir um degrau de refino | `balanceador.py:246` |
| `degraus` | 5 | Número de degraus de refino e denominador de `no_degrau` no refino | `balanceador.py:237`, `balanceador.py:366` |
| `passos_por_iteracao` | 24 | Passos de ambiente por iteração; define o fecho e `num_steps_per_env` | `curriculo.py:119`, `tarefa.py:49` |
| `zona_t_alvo` | 0,6 | Taxa de fecho por visita em que `z` fica parado; também a taxa de travessia em que a linha de uma coluna do `ANDAR` fica parada | `balanceador.py:173`, `balanceador.py:181` |
| `linha_n_min` | 50 | Visitas de uma coluna de terreno do `ANDAR` para atualizar a linha dela | `balanceador.py:179` |
| `linha_ganho` | 2,5 | Ganho da linha por atualização: taxa 1,0 sobe 1,0; taxa 0 desce 1,5 | `balanceador.py:181` |
| `nivel_abre_terreno` | 2 | Nível do `ANDAR` em que a célula `(ANDAR, terreno)` abre | `balanceador.py:222` |
| `zona_ganho` | 0,05 | Ganho de `z` por atualização | `balanceador.py:173` |
| `zona_n_min` | 200 | Visitas mínimas para atualizar `z` | `balanceador.py:171` |
| `zona_t_visita` | 8,0 s | Tempo desde a abertura da etapa; sem fecho até lá, a visita conta falha | `comando.py:484` |
| `zona_folga_cadeia` | 0,2 | Diferença máxima de `z` para baixo entre etapas consecutivas | `balanceador.py:176` |

⚠ Divergência: o comentário de `Balanceador.piso_tarefa` diz "piso de fração por tarefa". `fracoes_fatias` não aplica piso, e nenhum arquivo lido lê `piso_tarefa` (`balanceador.py:282-293`).

### 13.2 `Niveis` (`knobs.py:415-424`)

| Knob | Valor | Efeito | Linha de uso |
|---|---|---|---|
| `pegar` | ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)) | (topo em m; massa em kg) por nível do `PEGAR`. O tamanho dá `max_nivel["PEGAR"]` | `balanceador.py:74`, `eventos.py:99-102` |
| `botar` | (0,55; 0,45; 0,30; 0,15; 0,04) | Topo da laje de destino por nível do `BOTAR`. O tamanho dá `max_nivel["BOTAR"]` | `balanceador.py:74`, `eventos.py:386` |
| `topo_min_c` | 0,30 m | Piso do topo sorteado na fatia `C` | `eventos.py:105` |
| `subnivel_extra` | (0,10; 0,0) | Metros somados ao topo no subnível; o tamanho dá `_sub_max` | `balanceador.py:92`, `eventos.py:346`, `eventos.py:389` |
| `subnivel_longe` | (0,10; 0,20) | Teto do `longe` do destino por subnível | `eventos.py:388` |
| `subnivel_jitter` | (0,0; 1,0) | Fração da `faixa_recuo` na base da pega | `eventos.py:193-194` |

### 13.3 `Refino` (`knobs.py:543-547`)

| Knob | Valor | Efeito | Linha de uso |
|---|---|---|---|
| `degrau_batente` | 1 | A terminação `batente` liga quando o degrau da tarefa do env é ≥ 1 | `terminacoes.py:164`, `terminacoes.py:177-181` |

### 13.4 Knobs de outras classes que o currículo lê

| Knob | Valor | Efeito | Linha de uso |
|---|---|---|---|
| `Comando.frente_tetos` | (0,4; 0,55; 0,7; 0,85; 1,0) | Dá `max_nivel["CARREGAR"]` (4) e o último valor entra em `max_nivel["ANDAR"]` | `balanceador.py:72-73` |
| `Comando.andar_inicio` | 0,3 | Entra em `max_nivel["ANDAR"]` | `balanceador.py:72` |
| `Comando.andar_passo` | 0,1 | Entra em `max_nivel["ANDAR"]` | `balanceador.py:72` |
| `Terreno.linhas` | 13 | Dá `_max_ter["ANDAR"]` = 12; é também o divisor do `atraso` das colunas | `balanceador.py:77`, `eventos.py:63` |
| `Terreno.piso_coluna` | 0,05 | Fração mínima de uma coluna nos episódios de terreno do `ANDAR` | `eventos.py:64` |
| `Terreno.dist_travessia` | 3,5 m | Distância da origem da célula que conta como travessia e acaba o episódio de terreno do `ANDAR`. Fica no meio da borda plana da célula, que vai de 3,0 a 4,0 m do centro (seção 9.2 do arquivo 03) | `eventos.py:156-158`, `terminacoes.py:71` |
| `Terreno.linha_max_carregar` | (`escada`, 9), (`escada_inv`, 9), (`rampa`, 9), (`rampa_inv`, 9), (`escada_aleatoria`, 6), (`obstaculos`, 8), (`caixas`, 7) | O maior valor (9) dá `_max_ter["CARREGAR"]`; cada valor limita a linha da coluna | `balanceador.py:78`, `eventos.py:56-69` |
| `DR` (`atrito_pad`, `atrito_laje`, `passo_mu_pad`, `mu_pad_min`, `com_caixa`, `com_caixa_frac0`, `com_caixa_frac_max`, `rfi`, `pd`, `forca_motor`, `forca_motor_bracos`, `solref`, `solref_base`, `atraso_portao`, `atraso_refino`, `degrau_atraso_refino`, `atraso_inicio`) | ver seção 10 | Entram em `dr()` | `balanceador.py:365-389`, `curriculo.py:125` |

## 14. Resumo do que não está determinado pelo código lido

- Não determinado pelo código lido: o uso do currículo por `coleta.py` e `mede_deriva.py` além dos pontos achados por busca. `coleta.py` guarda `v3_cur.estado()` em `meta` (`coleta.py:56`, `coleta.py:148`) e troca `fracoes_fatias` por (0,5; 0,5; 0,0) (`coleta.py:59`). `mede_deriva.py` lê `cur.celulas`, `cur.restricao()` e `cur.k.alfa` (`mede_deriva.py:56`, `mede_deriva.py:75`, `mede_deriva.py:111`). Os dois arquivos não fazem parte do treino e não foram lidos por inteiro.
- Não determinado pelo código lido: se algum script de treino sobrescreve `num_steps_per_env` depois de `rl_cfg()`. `tarefa.py:49` o iguala a `Balanceador.passos_por_iteracao`, mas o script que chama o treino não foi lido.
- Não determinado pelo código lido: qualquer leitor de `Balanceador.piso_tarefa`. Nenhum arquivo lido o usa.
