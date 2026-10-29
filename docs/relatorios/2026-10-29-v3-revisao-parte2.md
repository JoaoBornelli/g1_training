# Revisão da Parte 2 da v3 (E11)

**Veredito: NÃO PRONTO para a fase I. Um nome indefinido derruba o env no primeiro reset; doze defeitos silenciosos mudam o treino da cadeia B.**

| severidade | achados |
|---|---|
| bloqueia o treino | 1 |
| defeito silencioso | 12 |
| regra do plano | 6 |
| menor | 12 |

Escopo: `git diff a31fa7d..5bb27e2 -- g1_v3 pilota_v3.py registra_juntas_v3.py` (5bb27e2 é o último commit da fase E).
Durante a revisão o HEAD andou para 7d926d3 (testes da F0 e o contrato "Fase F: núcleos puros" na §6.3). O código fora
dos testes não mudou desde 5bb27e2. A API foi lida no mjlab instalado em `.venv`. Nenhum env foi construído. Fora do
env, só: `make_env_cfg()`, `alvos.Referencias` e `balanceador.Curriculo`.

Suíte (`unittest discover`): 275 testes, todos ok, 1 skip. Nenhum teste chama um invólucro com env: o B1 passa verde.

Duas checagens estáticas, com scripts fora do repo: (1) nome global lido por função e ausente no módulo (achou só o
B1); (2) todo campo do `knobs` lido por alias de seção (`A.`, `D.`, `T.`, `CENA.`, `R.`, `ATIVO.x.`, `K.x.`) existe
(nenhum achado).

Conferido sem achado: zero import do `g1_limpo`, do `g1_poc` e do `g1_multitask`; módulos puros só com stdlib, numpy e
torch; z da laje e da caixa com a origem do env (`eventos.py:199-216`); todo sorteio no device do `v3_gerador`
(`eventos.py:67-69`, `comando.py:70-71`, `estimador.py:223-229`); formas dos sensores (`palma_E` e `apoio_caixa`
found (E,1); `caixa_robo` force (E,12,3), `contact_sensor.py:193-197`; mesas (E,P,3); `altura_*` heights (E,F));
`write_mocap_pose_to_sim` e `write_root_link_*` existem (`entity/entity.py:892-921`, `:1156-1169`);
`requires_model_fields` no `tamanho_caixa` e no `carga`; o `CadeiaCmd` nasce antes dos eventos de startup
(`manager_based_rl_env.py:306-354`); a ordem dos termos da §6.5; o `float64` do `Referencias` entra nos buffers
`float32` sem erro; o `perda_caixa` pulsa um passo só; a terminação e a recompensa leem o mesmo pulso; o reset zera
todo buffer da cadeia (`comando.py:441-454`); nenhum termo paga positivo no PEGAR com o robô parado (rastreio ≤ 0 com
comando zero; progresso ≤ 0 parado, entrada paga 0).

Ordem do `env.step` (`manager_based_rl_env.py:436-456`): terminação e recompensa leem o que a cadeia escreveu no fim do
passo anterior; as métricas rodam logo depois das recompensas, no mesmo passo. A soma telescópica do `progresso` fecha
no índice, mas não fecha no σ (S3) nem no fecho do P4 (S2).

---

## Escada da S §10, pela conta

Unidade de retorno: o mjlab multiplica cada termo por dt = 0,02.

| item | conta | valor |
|---|---|---|
| queda | 200 × 0,02 | 4,00, mais o resto do episódio |
| perda da caixa (chave ligada, `devolve_b_alvo` False) | 28,0515 × 0,02 | 0,561 |
| ganho da volta à pose de pé (desenho) | (B_alvo[B4] 1,8144 + w_p 1,8167 × ΔΦ ≤ 1) × 0,02 | ≤ 0,073 |
| lado direito da desigualdade (2) | (B_alvo P0–P3 3,3775 + Φ máx 1,8167) × 0,02 + aperto evitado ≤ 0,02 × 0,02 × 350 passos | ≤ 0,104 + 0,140 = 0,244 |

Vale: 4,00 > 0,561 > 0,073, e 0,561 > 0,244 (folga de 2,3×). Os defeitos S2 e S3 sobem o lado direito de (2) em no
máximo ~0,03; a escada continua valendo. No código de hoje o ganho da volta é zero (S8): a escada vale por vacuidade,
e a volta à pose de pé não tem gradiente positivo.

---

## Bloqueia o treino

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| B1 | `g1_v3/recompensas.py:405`, `:433`, `:445`, `:469`, `:480`, `:490` | `K` não existe no módulo. Só `R` e `C` estão definidos (`:30-31`). | `env.reset()` chama `command_manager.compute` (`manager_based_rl_env.py:372`); a cadeia chama `recompensas.incl_lim` (`comando.py:340`) → `NameError`. Nada roda: treino, smoke 2 a 24, play. O `progresso` e a `qualidade` caem do mesmo jeito em todo passo. | `recompensas.py:31`: `K = ATIVO.alvos`. Testes: `incl_lim` e `progresso` com env falso; checagem estática de nomes globais em todos os módulos (M9). | E4; §1.5 |

## Defeito silencioso

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| S1 | `g1_v3/eventos.py:327` (e `:338`) | O `SceneEntityCfg("robot", geom_names=PALM_PAD_GEOMS)` nasce dentro da função, sem `resolve`. O `geom_ids` fica `slice(None)` (`scene_entity_config.py:81`), e o `dr.geom_friction` usa `indexing.geom_ids[asset_cfg.geom_ids]` (`dr/_core.py:214-215`): todas as geoms do robô. | Todo reset põe μ = 1,0 em todas as geoms do robô (pés, pulsos, tronco) antes do portão; depois, U[μ_min; 1,0] por geom. A DR do pé (`atrito_pe`, 0,3–1,0 por pé) some no primeiro reset. A carcaça sai de 0,3 para 1,0 contra a caixa (prioridade igual: vale o maior μ), e o pulso e o tronco voltam a segurar a caixa, o que a PD-46 tirou. A laje (`:338`) tem uma geom só: sem efeito hoje. | `eventos.py`: `atrito_pad(env, env_ids, asset_cfg)` e `atrito_laje(env, env_ids, asset_cfg)`. `env_cfg.py:144-145`: `params={"asset_cfg": SceneEntityCfg(...)}` (o manager resolve, `manager_base.py:144-145`). Teste: o evento só muda os índices dos pads. | E1 item 6; PD-46; §6.8 |
| S2 | `g1_v3/recompensas.py:708`; `g1_v3/comando.py:348-353` | No fecho do P4 a cadeia põe `v3_avancou` = True e mantém o índice em P4. O progresso supõe "alvo velho = índice − 1" e paga como P3. | No passo do fecho: Φ(s′, P3) ≈ 0 (a faixa do tronco da pega é 27–54°; de pé, ~0°) e B_alvo[P3] = 0,836. r ≈ 1,8167·(0 − 1) + 0,836 ≈ −0,98 antes do dt, no lugar de +1,65 (B_alvo[P4] = 1,6721). O maior bônus nunca paga; o B[P3] paga duas vezes e entra duas vezes no `v3_b_pagos`. O marco do PEGAR custa. | `recompensas.py:708`: `i_ef = where(av & (idx >= 0) & ~((idx == P4) & env.v3_p4_fechado), idx − 1, idx)`. Teste do invólucro com env falso. | E4 (D1 da Parte 0); S §9.1 "Ao alcançar um alvo: bônus fixo" |
| S3 | `g1_v3/recompensas.py:709`; `g1_v3/comando.py:359-360` | No avanço a cadeia troca `v3_sigma` para o σ do alvo NOVO antes da recompensa. O progresso calcula Φ(s′, i_velho) com o σ novo; o `v3_phi`, Φ(s, i_velho), foi calculado com o σ velho. | Pagamento espúrio no passo do avanço. P1→P2: σ cai de ~0,15 para ~0,02 (P1 e P2 têm o mesmo alvo de palma, `alvos.py:145-151`): ΔΦ ≈ −0,6, r ≈ −1,1 contra B[P1] = 0,836. P2→P3: σ 0,02 → 0,05, r ≈ +0,9. P3→P4: σ 0,05 → 0,3–0,6, r até +1,1. A conta dos B_alvo da Parte 0 deixa de valer. | Buffer novo `v3_sigma_velho` (E,): `curriculo.py:27` (`_ESCALARES`); `comando.py:359` grava o σ corrente antes da troca; `recompensas.py`: `_phi_env(env, indice, sigma)`, com `sigma = where(av, v3_sigma_velho, v3_sigma)` para o `i_ef` e `v3_sigma` para o `phi_novo`. | E4 (D1: "Φ do alvo VELHO no estado novo"); §6.4 |
| S4 | `g1_v3/eventos.py:264-268` com `g1_v3/recompensas.py:479` | A partida do PEGAR (caixa em x 0,32 + U[0; 0,20], base em x U[−0,10; 0] e ±y) põe o recuo inicial entre 0,32 e 0,72 m. A IK pede 0,33–0,44. A faixa do recuo zera o Φ com \|desvio\| ≥ 0,16 m. | Monte Carlo com os knobs: no nível 0, 45% das aberturas do PEGAR têm Φ(P0) ≡ 0 e derivada zero nas palmas (37% no nível 1, 23% no 2). O único sinal para chegar perto é a `cerca_recuo` (custo); dar passo no PEGAR custa no `rastreio` (comando zero) e no `freio`. Canal morto no ponto de operação. | Decisão do dono. Opção A (E §4.3: "O controlador já levou o robô até a posição de pega"): sortear a base pelo `ref.recuo(topo, a)` ± `faixa_recuo`. Opção B: a faixa do recuo só de P1 em diante. | S §8.2 "faixa"; E §4.3; CLAUDE.md (gradiente) |
| S5 | `g1_v3/comando.py:304-305` | (a) `saida_b` dá a posição no frame de rumo da base estimada (`estimador.py:298-300`); a volta usa `quat_apply` com o quat inteiro, roll e pitch inclusos. (b) Caixa nunca vista: `saida_b` dá zero (`estimador.py:303`) e a pega vira a posição da base. | (a) Base a 3° de pitch e caixa a ~0,5 m: a pega erra ~2,5 cm em z, e o alvo do P3 junto. (b) Pega na pelve: o portão desloc do P0–P2 vale 0 (caixa a ~0,45 m da "pega", `comando.py:339`), o invariante nunca vale 1 e o Φ é zero: o episódio do PEGAR morre. Conta de geometria (pelve 0,789 m, câmera em torso + `d435_pos`, pitch 47,6°, FOV vertical ±21°): o centro da caixa fica fora do FOV na espera em pé em 4% das aberturas no nível 2, 17% no 3 e 43% no 4. | `estimador.py`: `do_frame_da_base(p_b, odo)`, o inverso de `alvo_no_frame_da_base` (+4 linhas); `comando.py:305` usa. Caixa nunca vista: decisão do dono (adiar a abertura até a primeira leitura, ou o centro verdadeiro). A E §4.2 pede a caixa visível na espera. | E3 item 1; S §6.4; E §4.2 |
| S6 | `g1_v3/comando.py:356-358` | No P3 e no P4 a cadeia sobrescreve `v3_alvo_w` com o alvo macro (pega + 5 cm; transporte, que anda com a base). O ator lê `v3_alvo_w` no canal do alvo (`observacoes.py:184`). | No treino o ator vê o alvo macro trocar e seguir a base. No robô o controlador manda um ponto fixo, o centro da caixa escolhida (I §2.3 tabela do alvo; I §4.1). Entrada fora da distribuição no real; e o canal entrega o índice macro, privilegiado (S §6.4: o gerador macro roda só no treino). | `comando.py`: apagar `:356-358`. O Φ já refaz P3 e P4 a partir da pega e da base (`recompensas.py:428-433`); o `_perda` só lê `v3_alvo_w` no BOTAR. | E3 item 1; S §6.1, §6.4; I §2.3 |
| S7 | `g1_v3/recompensas.py:684-686` | No PEGAR e no BOTAR a `cerca_recuo` usa `ref.recuo(topo, a)` em todo alvo, P4 inclusive. | No P4 a caixa está no transporte (recuo 0,25 m); a referência é 0,33–0,44. Custo de até 0,108 × 0,2492 por passo no P4 e na espera do detector (~75 passos, ~2,0 antes do dt, ~1,2 B[P4]). A cerca empurra a caixa para longe; a faixa do Φ(P4) puxa para 0,25. | `recompensas.py:684-686`: no PEGAR e no BOTAR, `ik = _recuo_ref(env, env.v3_alvo_idx)`; no CARREGAR, `R.recuo_carregar`. | E4 (parágrafo da referência macro: "Sem isto, a cerca ... cobraria o robô por ficar de pé no P4") |
| S8 | `g1_v3/env_cfg.py:46`; `g1_v3/recompensas.py:718`, `:477`, `:707-713`; `g1_v3/comando.py:320-326` | A volta à pose de pé depois da perda não paga. (1) A máscara do `progresso` não tem ANDAR, e a tarefa vira ANDAR na perda. (2) `paga=True` zera o progresso com `v3_perdeu`; a §6.6 lista vivo, altura_pe, passo e rastreios, não o progresso. (3) O portão desloc do B3–B4 lê `v3_ponto_apoio` = 0 (sem escritor na E): com a caixa a mais de 4 cm da origem xy do mundo, o portão é 0. (4) O fecho do B4 leva o índice a −1, e a `troca` zera o B_alvo[B4]. A perda também não põe `v3_phi` = −1: corrigidos (1) e (2), o passo da perda cobraria −w_p·Φ(alvo velho). | Depois da perda o robô só tem custo (os desvios do ANDAR). O "paga uma vez, como potencial" da S §10 não existe: canal morto. | `env_cfg.py:46`: progresso em (ANDAR, PEGAR, BOTAR) (S §9.2 e §10; o ANDAR com índice −1 paga 0). `recompensas.py:718`: sem `paga=True`. `recompensas.py:477`: `desloc_apoio` = 0 com `v3_perdeu`. `comando.py:322`: `env.v3_phi[ids] = −1`. O B_alvo[B4] fica com a R3. Aviso à F1: o `ok_agora_botar` do B4 da perda não pode usar o portão desloc. | S §9.2, §10; §6.6; E3 item 4 |
| S9 | `g1_v3/terminacoes.py:106`; `g1_v3/comando.py:320` | A chave `caixa_perdida_vira_andar` sai de dois lugares: a terminação lê o `ATIVO`; a cadeia lê `cfg.k`. | `make_env_cfg(k)` com a chave desligada em `k` (o teste "com volta atrás" da S §10): a cadeia não troca para ANDAR e a terminação (`ATIVO`, ligada) não encerra. Com a caixa no chão, `caiu` vale a todo passo: o `perda_caixa` cobra −28 por passo até o fim do episódio. | `terminacoes.py`: `caixa_perdida(env, tarefas, vira_andar)`; `env_cfg.py:132` passa `k.cadeia.caixa_perdida_vira_andar`. Smoke 19 sem o `mock.patch`. | E5; S §10; PD-27 |
| S10 | `g1_v3/recompensas.py:432` (com `g1_v3/comando.py:266`, `:284`) | O alvo do P4 no progresso e na `qualidade` usa o chão = `env_origins` z. A cadeia usa a pelve menos a `altura_pelve` (o chão local). | No CARREGAR de terreno (escada, rampa) o chão local muda até ~1 m; o alvo do transporte fica na altura da origem; d ≫ 5 cm e a `qualidade` vai a 0: o rastreio positivo do CARREGAR no terreno some. Na plataforma de 2 m as duas contas batem. | Uma função só: `recompensas.alvo_da_caixa_w(env, indice)` com o chão local (z da raiz − `altura_pelve`); `comando._alvo_da_caixa` chama ela. | §6.3 `caixa_alvo_w`; PD-21 |
| S11 | `g1_v3/comando.py:562-565` | O fecho de segmento por troca de comando passa `falha` = zeros. A perda da caixa no CARREGAR troca o comando (`_publica`). Antes da âncora (1,5 s), o segmento não registra. | Perder a caixa no começo do CARREGAR (o momento típico: começar a andar) não conta como falha. O sucesso do CARREGAR sobe, e o portão abre com uma política que larga a caixa. | `comando.py:564`: `falha = env.v3_perdeu[mudou] & self.seg_carrega[mudou]`. | PD-50; S §11.3 |
| S12 | `pilota_v3.py:294-314` | O `Visao.passo` não zera os 11 canais no ANDAR (o treino zera, `observacoes.py:187`) e passa as 33 cápsulas na oclusão (o treino passa as 8 dos braços, `observacoes.py:177-178`). | No play, em ANDAR o ator recebe a leitura da caixa e a idade 2,0, que nunca viu no treino. A cabeça e o tronco tapam a câmera no driver e não no treino: a cápsula da cabeça passa a 0,2 mm da câmera (0,0602 m contra raio 0,06). | `pilota_v3.py`: 11 zeros no ANDAR; só as cápsulas com `braco` do `capsulas.npz`. | E8, E9; PD-11 |

## Regra do plano

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| R1 | `g1_v3/env_cfg.py:46`; plano §12.1, tabela da E4 | Três textos para a máscara do `progresso`: o código usa PEGAR, CARREGAR, BOTAR; a S §9.2 diz PEGAR e BOTAR; o plano diz `v3_alvo_idx ≥ 0`. | No CARREGAR o índice fica no P4 e o progresso cobra (1 − γ)·w_p·Φ(P4) por passo de quem segura bem. O ANDAR da perda fica de fora (S8). | O plano escreve (ANDAR, PEGAR, BOTAR), sem `paga`. Regra nova na §6.4: todo salto de índice fora do `avanca` põe `v3_phi` = −1 (perda → B4; P4 → B1 na abertura do BOTAR, fase F). | S §9.2, §10; §6.4 |
| R2 | `g1_v3/recompensas.py:480` | Portão "com": `desvio_com = relu(lim_com − margem)` com `x_lim = faixa_com`. O portão fica em 1 até a margem 0 e zera em −3 cm. A S §8.2 pede o CoM a 3 cm ou mais da borda. O plano não define `desvio_com`. | A faixa do CoM aceita 3 cm a mais que a spec. | O plano define. Proposta: `rampa(relu(faixa_com − margem), 0, delta_com)`: 1 com margem ≥ 3 cm, 0 com margem ≤ 0. | §6.3 `portoes_do_alvo`; S §8.2 |
| R3 | `g1_v3/recompensas.py:707-715` | O fecho do B4 leva o índice a −1; a PD-36 zera o passo, B_alvo inclusive. O 9º valor do `b_alvo` (1,8144) nunca paga. | O BOTAR da F nunca recebe o bônus do B4. | Decisão do dono: pagar o B_alvo[B4] na troca (só o ΔΦ vai a zero) ou tirar o valor. | PD-36; S §9.1 |
| R4 | minimização | Contas repetidas: alvo do P4 (`comando.py:279-284` e `recompensas.py:428-433`, com chão diferente, S10); `palmas_c` (`comando.py:267` e `recompensas.py:463-465`); contatos das palmas (`comando.py:269` e `recompensas.py:423-425`); `_ref_alvo` e `_ref_do_alvo` (`recompensas.py:334-338`, `:441-445`); `recompensas._rumo` = `estimador.yaw` (`recompensas.py:392-395`, `estimador.py:13-15`), e a cadeia usa `heading_w`, conta diferente com a base inclinada (o `frame_da_caixa` pode escolher outra face em cada lado); a pelve sobre o pé no driver (`pilota_v3.py:331`) e `observacoes._z_pe_min`. Knobs mortos: `Cadeia.pega_c_topo_min` (`knobs.py:146`, repete `Niveis.topo_min_c`, `:336`), `prateleira_jitter_z`, `prateleira_topo_piso`, `jitter_x_max[5:7]`. | Duas contas da mesma grandeza divergem sem aviso. | Uma função por conta; apagar os campos mortos. | §1.5; CLAUDE.md |
| R5 | `g1_v3/eventos.py:326`, `:337` | μ = (1,0; 1,0) literal antes do portão. | Mudar o μ do pad no `knobs` não muda o evento. | `Cena.pad_atrito[0]` e `Cena.prateleira_atrito[0]`. | §1.5 |
| R6 | plano §6.3, §6.4; `g1_v3/comando.py:366` | Contrato desatualizado: `EstimadorCaixa.passo(..., dist)` e a chave `ori_persistente` do dict; o `v3_alvo_w` do PEGAR fixo na pega (S6); `v3_sigma_velho` (S3); `do_frame_da_base` (S5); `alvo_da_caixa_w` (S10). O detector do treino lê a pelve sobre o terreno (`altura_pelve`); a I §4.2 diz "sobre o pé mais baixo" (igual no plano, onde PEGAR e BOTAR rodam). | Uma lane nova lê o contrato e não acha os nomes. | O orquestrador escreve antes do despacho; o detector passa a usar `observacoes._z_pe_min`. | §6.3, §6.4; I §4.2 |

## Menor

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| M1 | `g1_v3/smoke.py:640-673`, `:676-709` | s21 e s22 caem. Na s21, depois do aviso, o atraso U[0; 1] s vence em ~99% dos casos de 4 mundos e o trecho abre o CARREGAR; a s22 abre o CARREGAR direto. O `twist` sorteia célula do CARREGAR, fechada no começo do currículo: `torch.multinomial` sobre vetor vazio levanta `RuntimeError` (`balanceador.py:229-230`; medido). | O smoke da fase I não fecha. | s21: `cad.alvo_s[:] = inf` depois do `_abre_trecho_de`. s21 e s22: abrir as células do CARREGAR no `v3_cur` antes. | §1.7; E10 |
| M2 | `g1_v3/smoke.py:603` | s20: a caixa em (0,45; 0; 0,3) da base fica ~24° abaixo do horizonte da câmera; o FOV começa em 26,6° (47,6° − 21°). | A primeira checagem falha pela geometria, não pelo estimador. | `ofs` z ≈ 0,0 (~50° abaixo). | E10 item 20 |
| M3 | `g1_v3/smoke.py:502-521`, `:524-542` | s17 e s18 rodam com a cena estacionada: o reset sorteia a fatia ANDAR e `_abre_trecho_de` só troca `v3_fatia`. A s17 passa com a pega na base (S5b) e não mede nada. A s18 pode ver ΔΦ > 0 pequeno do PD assentando. | Falso verde na s17; falso vermelho possível na s18. | s17: `posiciona_cena`, `forward` e 10 passos parados antes de abrir; checar ‖pega − caixa‖ < 2 cm. s18: tolerância ou partida com Φ = 0. | E10 itens 17, 18 |
| M4 | `g1_v3/metricas.py:295`, `:304-315` | `deslocamento_caixa_laje` usa a norma 3D (o portão usa xy). `idade_leituras` soma os envs do ANDAR (idade 2,0). `phi_*` e `tempo_*` somam zeros dos envs que nunca tiveram o alvo. | Médias diluídas no log. | xy; máscara pelas tarefas com caixa. | E7; S §14 |
| M5 | `g1_v3/eventos.py:351-352` | `vies_imu` sorteia ±1° em x, y e z (Euler). O giro em z não é inclinação (DR §3.4) e gira o mapa ±1° em volta do sensor (PD-42). | Erro de rumo no mapa que o real não tem desse jeito. | Decisão do dono; proposta: z = 0. | §6.8; PD-42 |
| M6 | `g1_v3/comando.py:419` | `anda` é sempre verdadeiro no CARREGAR de B: as 4 células têm comando ≠ 0. | Condição morta. | Simplificar, ou manter com comentário. | E3 item 3 |
| M7 | `g1_v3/recompensas.py:696` | A parte da caixa no `contato_fora` conta pontos de contato (até 12 fatias); a do chão conta corpos. | Um antebraço com 4 pontos paga 4. | Decisão: contar corpos ou um por passo. | S §9.2 |
| M8 | `pilota_v3.py:290-292`; `registra_juntas_v3.py:77-80` | O alvo do PEGAR no driver é o centro verdadeiro; no treino é a leitura do estimador (S §6.4, "a caixa escolhida"). O roteiro B troca por tempo, não pelo aviso. | Pequena diferença de paridade. | `abre_pegar` pela última leitura (`do_frame_da_base`); troca no aviso + atraso, opcional. | E9; I §4.2 |
| M9 | testes | Falta: a idade cresce 0,02 por passo depois de uma leitura válida; invólucros com env falso (o B1, o S2 e o S3 passam verdes hoje); checagem estática de nomes globais. | Bloqueio que só aparece no smoke. | Lane de testes. | §14 |
| M10 | `g1_v3/recompensas.py:709-710`, `:411`, `:463`, `:522` | `_phi_env` roda duas vezes por passo (o `torch.where` avalia os dois ramos). `find_sites` e `find_bodies` resolvem regex a cada passo. | Custo de CPU por passo. | Segundo Φ só com `av.any()`; índices guardados no módulo. | §15 |
| M11 | `g1_v3/comando.py:326` | Risco a medir: a perda numa pega funda troca a tarefa para ANDAR no mesmo passo, e a queda passa a 0,8 rad e 0,20 m (`terminacoes.py:44`). | Se a pelve passar de 46° na pega, a perda vira queda (−200). | Medir no smoke da fase I. | S §10 |
| M12 | `g1_v3/comando.py:294` | `sigma_min` (m) é piso também do B4, que é adimensional. | Sem efeito prático. | Piso só nos alvos métricos, como `parte0/item1_phi.py:39`. | E3 |

---

## Pendências

Veredito de cada item de `e11_pendencias.md`.

| item | veredito |
|---|---|
| E5: `caixa_perdida` em PEGAR, CARREGAR, BOTAR | OK. A máscara bate com a S §10. O `caiu` não é armado pela pega, mas antes dela a caixa só chega ao chão saindo da laje, e isso já conta (fundo na laje ≥ 0,04 > 0,02). |
| E7: `caixa_robo` (E,12,3) com `maxforce` | Confirmado: `found` [B, N], `force` [B, N, 3], N = P·num_slots, P = 1 (`contact_sensor.py:193-197`, `:405`). |
| E7: `idade_leituras` e `saida_b` pura | Confirmado: `saida_b` só lê (`estimador.py:295-304`). |
| E7: `phi_*` com defasagem de 1 passo | Não há defasagem: o `progresso` escreve `v3_phi` no mesmo passo e com o mesmo `v3_alvo_idx` que a métrica lê (`manager_based_rl_env.py:440-441`). |
| E7: deslocamento 3D × portão xy | Alinhar ao xy (M4). |
| E7: velocidade de chegada e aceleração (S §14) | OK na H6; a E7 não as lista. |
| E6: `atraso_max` só acima do inicial; `com_caixa` [0, 0] | OK (PD-28, PD-38). |
| E2: kwarg `dist` | OK no código; falta na §6.3 (R6). |
| E2: viés da meia N(0,1)·`vies_meia`; orientação persistente | OK; leitura razoável de "pequeno" (S §13). O dono confirma se quer uniforme ±. |
| E2: teste da idade parado em `idade_max` | Lacuna real (M9). A lógica está certa por leitura (`estimador.py:297`). |
| E2: `visivel` com o centro no FOV e 9 pontos | OK: PD-11 e o contrato do detector (I §2.3, "Contrato do detector"). |
| E1: `orientacao_de_nascimento` não portada | Decisão do dono. A do `g1_limpo` é a do REORIENTAR (face marcada, quartos de volta). No PEGAR o desalinho vem só do rumo da base (±0,2 rad). Recomendo manter a identidade nesta fase. |
| E1: `jitter_x_max` 7 valores × 5 níveis | OK no uso; cortar para 5 (R4). |
| E1: `prateleira_jitter_z` fora | OK: o topo já é U[h_k; h_0]. Apagar o campo (R4). |
| E1: `vies_imu` Euler ±1° por eixo | Decisão do dono (M5). |
| E1: `v3_topo` = 0 no ANDAR | OK: só termos de PEGAR, CARREGAR e BOTAR o leem. |
| E1: ordem do mjlab (cadeia antes do startup) | Confirmado (`manager_based_rl_env.py:306-354`). |
| E4: `desvio_com` | Regra do plano (R2). |
| E4: tronco de referência 0 no P4 | OK: a pose de pé com caixa tem o tronco quase vertical, e a faixa é 15°. |
| E4/F: B1–B3 com `v3_topo_destino` × cercas com `v3_topo` | Resolvido pelo contrato da fase F (762af01): as cercas de B1–B3 usam `v3_topo_destino`. Apagar `_ref_alvo` (R4). |
| E4/G: `sigma_pouso` pelo degrau | Resolvido pelo contrato da F4 (`por_degrau(*sigma_pouso)`). |
| E4: a cadeia roda depois das recompensas; a soma fecha | Fecha no índice. Não fecha no σ (S3) nem no fecho do P4 (S2). |
| E3: `v3_ponto_apoio` sem escritor | Defeito no B4 da perda (S8). No BOTAR, a F escreve. |
| E3: B4 da perda inline | Resolvido pela F1 (`ok_agora_botar`). Atenção: o invariante desloc do B3–B4 zera o B4 da perda; a F1 tem de ignorá-lo com `v3_perdeu` (S8). |
| E3: `laje_fora` com giro no lugar | OK: a S §7 diz "pede movimento", e girar é movimento. A condição é sempre verdadeira hoje (M6). |
| E3: perda congela o trecho; fim no `time_out` | OK (S §10, "até o fim"). |
| E3: `v3_sigma` = 1,0 no reset | OK. |
| E3: `forca_de_apoio` sem uso | OK; é do B2 (fase F). |
| E-fix: `incl_lim` no portão | Resolvido em 95d5ed5 (`comando.py:340`, `:343`). |
| E9: alvo do driver = centro verdadeiro | Alinhar (M8). |
| E9: 9 pontos do `visivel` no driver | Confirmado: mesmos pontos e mesma ordem (`observacoes.py:47-48`, `pilota_v3.py:280`, `:297`). O driver passa as 33 cápsulas (S12). |
| E9: `alvo_x/y/z` do gravador = ponto do PEGAR | OK: é o ponto que o controlador manda (I §4.1). |
| E9: roteiro B por tempo | Menor (M8). |
| E8: `caixa_verdade` direto do `scene["box"]` | OK: canal privilegiado do crítico. |
| E8: magnitude no índice 0 | OK nesta fase; o refino é da G/H. |
| E8: `t` global, grade desde o `reinicia` | OK (`estimador.py:246-248`). |
| E8: esperas = ANDAR | OK (E §4.2). |
| E-fix 2 | Feito em 95d5ed5: `reinicia` no `filtros` (`eventos.py:161`), alvo zero sem índice (`observacoes.py:185`), `incl_lim` no portão (`comando.py:340`), teste da guarda no `test_observacoes`. |
| E10: chave com duas fontes | Defeito (S9). |

## DÚVIDAS no código

| marca | veredito |
|---|---|
| `g1_v3/eventos.py:238` ("A caixa NÃO se move aqui") | O plano responde: E1 item 5 e S §7 tiram só a laje. Apagar a referência. |
| `g1_v3/smoke.py:492` (leitura do mocap pelo corpo) | O mjlab responde: `write_mocap_pose` grava `mocap_pos` (`entity/data.py:205-213`); a pose do corpo sai do `xpos` (`entity/data.py:247`), que o `forward` preenche a partir do mocap. A seção já chama `sim.forward()`. Apagar a marca. |
| `g1_v3/smoke.py:553` (duas fontes da chave) | Defeito S9. Corrigir no `terminacoes` e no `env_cfg`; depois, tirar o `mock.patch`. |
| `g1_v3/smoke.py:617` (nada chama `passo` nem `visivel`) | Desatualizada: a E8 chama os dois (`observacoes.py:178-181`). Apagar. |
| `g1_v3/smoke.py:632` (oclusão pelos braços) | O dono mede na fase I. Antes, trocar o `ofs` (M2): hoje a caixa já fica fora do FOV sem os braços. |
| `g1_v3/smoke.py:651` (detector só lê as palmas) | O plano responde: I §4.2, "Limite conhecido" (a pose sozinha não garante a caixa nas mãos). O PD em `q_com_caixa` e a pelve em 3 cm: medir na fase I. Corrigir o atraso da s21 (M1). |
| `registra_juntas_v3.py:309` (alvo = ponto do PEGAR) | O plano responde: E9 e I §4.1 (o controlador manda um ponto no mundo). Apagar a marca. |

---

## Lanes de correção

Arquivos disjuntos dentro de cada onda. Contagem antes → depois (estimada). ⚠ As lanes F1 a F4 tocam
`comando.py`, `eventos.py`, `recompensas.py` e `terminacoes.py`: despachar estas correções antes delas, ou dentro delas.

**Onda 1 (em paralelo)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 1 — recompensas | `g1_v3/recompensas.py` | B1; S2; S3 (leitor de `v3_sigma_velho`); S7; S8 (sem `paga` no progresso; desloc do B4 = 0 com `v3_perdeu`); S10 (`alvo_da_caixa_w(env, indice)` pública, chão local); R4 (`_ref_alvo` sai; `_rumo` → `estimador.yaw`); M10 opcional | 758 → ~755 |
| 2 — eventos, cfg e terminação | `g1_v3/eventos.py`, `g1_v3/env_cfg.py`, `g1_v3/terminacoes.py`, `g1_v3/curriculo.py`, `g1_v3/testes/test_cfg.py` | S1 (`asset_cfg` pelos params); R5; S8 (progresso em ANDAR, PEGAR, BOTAR; `test_cfg.py:58`); S9 (`vira_andar` no param); `v3_sigma_velho` no `_ESCALARES`; M5 se o dono decidir | eventos 357 → ~358; env_cfg 177 → ~180; terminacoes 108 → ~106; curriculo 122 → 122 |
| 3 — estimador e métricas | `g1_v3/estimador.py`, `g1_v3/metricas.py` | `do_frame_da_base(p_b, odo)` (S5a); M4 | estimador 304 → ~308; metricas 370 → ~372 |
| 4 — testes (TDD, Haiku) | `g1_v3/testes/test_recompensas_e.py`, `test_eventos_e.py`, `test_estimador_e.py`, `test_knobs.py` | invólucros com env falso: `incl_lim` (B1), fecho do P4 paga B[P4] (S2), Φ velho com σ velho (S3), `cerca_recuo` no P4 (S7); `atrito_pad` muda só os pads (S1); idade cresce depois de leitura válida; `do_frame_da_base` ∘ `alvo_no_frame_da_base` = identidade; checagem estática de nomes globais em todos os módulos (pega o B1) | +~60 linhas de teste |

**Onda 2 (depois das lanes 1 a 3)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 5 — cadeia | `g1_v3/comando.py`, `g1_v3/testes/test_comando_e.py` | S3 (grava `v3_sigma_velho`); S5 (`do_frame_da_base`; caixa nunca vista pela decisão do dono); S6 (apagar `:356-358`); S8 (`v3_phi` = −1 na perda); S10 (`_alvo_da_caixa` chama `recompensas.alvo_da_caixa_w`); S11; R6 (detector com `_z_pe_min`); M12 | 646 → ~640 |
| 6 — driver | `pilota_v3.py`, `registra_juntas_v3.py` | S12; M8; apagar a marca de `registra_juntas_v3.py:309` | pilota 784 → ~788; registra 404 → ~402 |
| 7 — smoke | `g1_v3/smoke.py` | M1, M2, M3; s19 sem `mock.patch` (depois da lane 2); apagar as marcas de `:492` e `:617`; nova checagem na s17 (pega a < 2 cm da caixa) | 794 → ~800 |

**Onda 3 (orquestrador, Opus)**

| lane | arquivos | correções |
|---|---|---|
| 8 — plano e knobs | `docs/planos/2026-10-14-implementacao-g1-v3.md`, `g1_v3/knobs.py` | §6.3: `dist` e `ori_persistente` no `EstimadorCaixa.passo`; `do_frame_da_base`; `alvo_da_caixa_w`. §6.4: `v3_sigma_velho`; `v3_alvo_w` fixo na pega durante o PEGAR; `v3_phi` = −1 em todo salto de índice fora do `avanca`. §12.1 E4: máscara do progresso (ANDAR, PEGAR, BOTAR) sem `paga`; B4 da perda sem o portão desloc. Decisões do dono: S4 (recuo × partida), S5b (caixa nunca vista), R2 (portão com), R3 (B_alvo[B4]), M5 (`vies_imu` em z), orientação de nascimento. `knobs.py`: apagar `Cadeia.pega_c_topo_min`, `prateleira_jitter_z`, `prateleira_topo_piso`; `jitter_x_max` com 5 valores (a F usa o mesmo índice de nível). §20: linha desta revisão. |

Ordem: a lane 4 escreve os testes que falham antes das lanes 1, 2, 3 e 5. O smoke da fase I só vale depois das lanes
1, 2, 5 e 7. As decisões do dono (S4, S5b, R2, R3, M5) entram na lane 8 antes da lane 5.
