# Revisão da Parte 3 da v3 (F9)

**Veredito: PRONTO PARA O SMOKE, NÃO PARA O TREINO DA FATIA C. Nada derruba o env; quatro defeitos silenciosos mudam o que o BOTAR aprende, e o B1 e o B4 abrem com Φ ≡ 0.**

| severidade | achados |
|---|---|
| bloqueia o treino | 0 |
| defeito silencioso | 4 |
| regra do plano | 5 |
| menor | 15 |

Escopo: `git diff 364988d..a44aebb -- g1_v3 pilota_v3.py registra_juntas_v3.py` (13 arquivos, +549 −103) e os testes
da F0 (`7d926d3`, 4 arquivos, +391). Li inteiros: `comando.py`, `eventos.py`, `terminacoes.py`, `recompensas.py`,
`metricas.py`, `env_cfg.py`, `alvos.py`; `cena.py` (sensores e laje), `observacoes.py` (caixa, referência, crítico),
`curriculo.py` (`garante_v3`), `knobs.py`, `smoke.py` (25 a 29 e os auxiliares), `pilota_v3.py` (`Visao`),
`registra_juntas_v3.py`. A API foi lida no mjlab do `.venv`. Nenhum env foi construído. Fora do env, só:
`make_env_cfg()`, `alvos.Referencias` e os núcleos puros. As contas estão em
`/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/5d540910-028a-4291-8d9f-065788e279ae/scratchpad/f9_contas.py`.

Suíte (`unittest discover`): 313 testes, todos ok, 1 skip. Nenhum teste chama um invólucro da F com env: S1 a S4
passam verdes hoje.

Checagem estática de nome global ausente (script fora do repo, todos os módulos): só closures, nenhum achado.

Conferido sem achado:
- `make_env_cfg()` monta; terminações na ordem da §6.5 (`time_out`, `queda`, `caixa_perdida`, `chegada`); o
  `apoio_caixa` tem `track_air_time=True`.
- Formas (ponto 5): `compute_first_contact` devolve (E,P) e `apoio_caixa` tem P = 1 (`contact_sensor.py:374-384`);
  a `chegada` e a métrica leem `[:, 0]` e cruzam com (E,). `found.amax(-1)` (E,); `forca_de_apoio` (E,1,3) → (E,);
  `afasta` (E,2,3) → (E,2); `v_caixa.norm` (E,). `posiciona_destino`: sorteio no device do gerador, depois
  `.to(dev, f64)`; a pose da laje em float32 aceita o float64; a soltura volta como (n,3).
- `v3_vz_caixa_ant`: a cadeia escreve no fim do `compute`, para todo env, depois do reset (`comando.py:525`). A
  terminação e a métrica do passo seguinte leem o mesmo número, a vz do passo anterior. As suspeitas da F3 (reset) e da
  F6 (métrica depois da terminação) caíram em `035d9b1`.
- `_abre_botar` grava `v3_alvo_w` antes do σ; `v3_phi` = −1 na abertura; o `_perda` do BOTAR lê a soltura
  (`comando.py:347`). Item da E11 lane 5 resolvido.
- Progresso do B1 ao B4: a entrada paga 0; o velho é `idx − 1`; o B4 paga na troca, salvo com perda; o σ do velho
  vem do `v3_sigma_velho`. O B4 da perda usa `ok_agora_botar` com invariante 1.
- Registro: o PEGAR da C registra na abertura do BOTAR; o BOTAR no fim da cadeia; o reset com BOTAR pendente conta
  falha; o CARREGAR parado da C não registra célula (`comando.py:705`, `:708`).
- `Recompensa.recuo_carregar` saiu do código (sobra em docstring, M8). `metricas` usa `TAREFAS.index`.
  `tem_leitura()` e `z_pe_min` são públicos. Itens da E11 lanes 1, 3, 5 e 6 resolvidos.
- `heading_w` do mjlab é `atan2` do eixo x girado (`entity/data.py:589-592`, `forward_vec_b` = (1,0,0)); o
  `estimador.yaw` é a mesma conta. As duas não divergem com a base inclinada: o R4 da E11 errou nisso. Fica só a
  duplicação (R3).

Ordem do `env.step` (`manager_based_rl_env.py:419-464`): física, terminação, recompensa, métricas, reset (eventos,
depois `command_manager.reset`), `sim.forward()`, `command_manager.compute` (a cadeia), eventos de intervalo, `sense`,
observação. A laje de destino é escrita no `compute`, depois do `forward`: o lidar daquele passo ainda a vê no lugar
velho (um passo; sem efeito).

---

## 1. Escada do BOTAR, pela conta

Unidade: antes do dt; o retorno do mjlab é × 0,02.

| item | antes do dt | retorno |
|---|---|---|
| B_alvo B1; B2; B3; B4 | 1,8144; 0,9072; 0,9072; 1,8144 | 0,0363; 0,0181; 0,0181; 0,0363 |
| w_p·ΔΦ por alvo, de e⁻¹ a 1 (máximo) | 1,1484 | 0,0230 |
| BOTAR ideal, B1 a B4 | ≤ 10,04 | ≤ 0,201 |
| `perda_caixa` | 28,0515 | 0,561 |
| `chegada` (terminação) | 200 | 4,00, mais o resto do episódio |
| ANDAR parado final, 1,5 s (`vivo` 0,15/s + `altura_pe` 1,0/s com o pé parado) | — | 1,725 |
| `cerca_pelve` depois do B4 (índice −1, referência 0, S1) | 0,0757 por passo | 0,0757 por s; 0,057 no atraso médio |

Valem: perda (0,561) > BOTAR inteiro (0,201), 2,8×; chegada (≥ 5,7) ≫ tudo. O que pesa no BOTAR é chegar ao ANDAR
final (1,725), e o detector leva a ele sem o B4. Largar a caixa de até 7 cm sobre a soltura e ficar de pé perde só o
B2 a B4: ≤ 0,073 de bônus + ≤ 0,069 de ΔΦ = ≤ 0,14. Pousar com cuidado ganha de largar por 0,08 a 0,14 (P4).

Canais mortos e canais sem efeito líquido:

| alvo | ponto de operação | Φ e derivada | achado |
|---|---|---|---|
| B1, abertura | robô em pé na pose de transporte (tronco ~0°, recuo 0,25) | portão do tronco |incl − IK|: IK do destino 27°–54°. Nível 0 (0,55): 0,00; 0,02; 0,19 (a = 0,07; 0,10; 0,13). Níveis ≥ 1 com topo ≤ 0,45: 0. Recuo: 0,183–0,188 de desvio no topo 0,30 → 0. Φ ≡ 0, derivada 0 até o tronco descer 2° a 24° | R1, P2 |
| B2, pouso | caixa descendo a 0,1–0,3 m/s, fundo < 5 cm | `fator_pouso` σ 1,0: fator 0,99–0,91; derivada −0,20 a −0,55 por m/s. Vivo, mas o Φ volta a 1 com a caixa parada no fecho do B2: a soma telescopa e o retorno não depende da velocidade de pouso. Só o penhasco da `chegada` decide | P3 |
| B3–B4 | caixa deslocada do `v3_ponto_apoio` | rampa(x; 0,02; 0,02): derivada −50/m entre 2 e 4 cm; 0 acima de 4 cm: Φ ≡ 0 e o invariante trava o avanço até o fim do episódio | S2 |
| B4, começo | tronco na IK do destino (27°–54°), referência 0 | portão do tronco 0 para IK ≥ 30° (todo destino abaixo de 0,55). Φ(B4) ≡ 0 até o tronco passar de 30°; só as cercas puxam | R1, P2 |
| depois do B4 | em pé, índice −1, ainda em BOTAR | `cerca_pelve` contra 0: 0,0757 ret/s, com derivada para agachar | S1 |

## 2. A `chegada` com a vz do passo anterior

Euler semi-implícito do MuJoCo: v_n = g·dt_f·n; queda z_n = g·dt_f²·n(n+1)/2; dt_f = 0,005, decimação 4.

| queda | toque | vz_ant (passo anterior) | v no toque | √(2gh) | degrau 0 (1,0 m/s) |
|---|---|---|---|---|---|
| 5 cm | subpasso 20, passo 4 | 0,785 | 0,981 | 0,990 | não termina |
| 10 cm | subpasso 29, passo 7 | 1,373 | 1,422 | 1,401 | termina no passo 7 |

O veredito bate com o que a F8 espera. O `k_toque` da s27 dá 5 e 7: o de 10 cm é exato; o de 5 cm erra um passo para
mais, sem efeito no check. O critério julga a vz até g·dt = 0,196 m/s abaixo da do impacto. Degrau 0: passa queda de
até 7,3 cm, termina de 7,4 cm (o limite em √(2gh) seria 5,1 cm). Degrau 5 (0,25 m/s): passa até 0,8 cm, termina de
0,9 cm (seria 0,3 cm). É a regra da S §10 ("passo antes do 1º contato"); o número fica para o dono saber.

## 3. A laje de destino na abertura do BOTAR

Laje de 4 cm (`prateleira_meia_z` 0,02), topo ≤ 0,55 m. Pose de transporte (`ref.caixa_transporte_b`): centro em
x = 0,25, z = pelve + 0,111 = 0,90 m; fundo a 0,77–0,83 m. Borda perto em x = recuo(topo, a) − 0,15 + longe:
0,181–0,288 m. Folga vertical caixa × laje: ≥ 0,22 m (topo 0,55, a = 0,13). As palmas ficam na altura do centro da
caixa (0,90 m). O bico do pé fica em x = 0,152 m (pé 0,010 + canto da sola 0,142); a laje de 0,04 m ocupa z de 0 a
0,04 com a borda em x ≥ 0,230: folga ≥ 7,8 cm. Coxa e joelho ficam perto do x da pelve; a laje de 0,41–0,55 m tem a
borda em x ≥ 0,181. Veredito: na pose de transporte nada corta a caixa, as mãos nem as pernas. O θ de ±10° não aproxima
o canto (teste da F0).

## 4. O frame do `v3_alvo_w`

A suspeita parte de um erro. No PEGAR, `do_frame_da_base(saida_b(odo), odo)` desfaz a mesma conta com a mesma pose da
odometria: dá `_r_pos`, a caixa no mundo VERDADEIRO com o ruído do estimador (`estimador.py:307-316`, `:193-198`).
No BOTAR, `posiciona_destino` dá a soltura no mundo verdadeiro. Os dois alvos estão no mesmo frame. A observação
(`observacoes.py:184`) lê os dois pela `odo`: os dois erram pela deriva desde o reset. As recompensas leem os dois no
mundo verdadeiro: as duas acertam. No real o controlador guarda o ponto pela odometria no instante da escolha, e o erro
é a deriva desde a escolha. Na fatia C o robô não anda antes do BOTAR: a deriva é de milímetros. Sem achado na fase F;
vale rever se a fase G puser marcha antes do BOTAR.

---

## Defeito silencioso

| # | arquivo:linha | o que está errado | cenário | correção | plano |
|---|---|---|---|---|---|
| S1 | `g1_v3/recompensas.py:670-671` com `g1_v3/observacoes.py:120-121` | No PEGAR e no BOTAR, a `cerca_pelve` usa a referência do alvo corrente. Com índice −1 o `referencia_do_alvo` dá pelve 0. | Em pé, 0,789 m: relu(0,789 − 0,08) × 0,10672 = 0,0757 por s no retorno. Age na espera do PEGAR (S5b, índice −1) e depois do B4 (índice −1, ainda em BOTAR, detector + atraso de até 1,5 s). Fechar o B4 LIGA esse custo: 0,057 no atraso médio, mais que o B_alvo[B4] (0,036) e perto de B[B4] + ΔΦ (0,059). E a derivada manda agachar, contra o detector. No PEGAR que nunca vê a caixa: 19 s × 0,0757 = 1,4. | `recompensas.py:670`: `manip = (... PEGAR) | (... BOTAR)) & (env.v3_alvo_idx >= 0)`; com −1 vale a pose de pé (`pelve_de_pe`). Teste do invólucro com env falso. | §6.6; S §9.3 |
| S2 | `g1_v3/comando.py:372-373` | O `v3_ponto_apoio` é a posição da caixa no PRIMEIRO `found` do `apoio_caixa`, em qualquer B. O item 1 da Parte 0 calibrou com o ponto "no fim do B2" (`parte0/saida/item1_phi.txt`, premissas). A S §8.2 diz "depois de apoiada". | (a) A caixa raspa a borda perto no B1 (a borda fica 0,15 m antes da soltura) ou encosta a 3 cm do alvo; o Φ(B2), sem portão desloc, paga para arrastar até a soltura (3 cm: Φ de 0,74 a 1, +0,47 antes do dt). O B3 abre com desloc 3 cm: portão 0,5 e o invariante trava o B3. (b) Arraste ≥ 4 cm: Φ(B3) = Φ(B4) ≡ 0, derivada 0, B3 nunca fecha: o BOTAR acaba no `time_out` como falha. | Gravar o ponto no fecho do B2: em `_alvos_pegar`, depois do `avanca`, `fecha_b2 = av & (idx == B2)`; `v3_ponto_apoio[fecha_b2] = m["caixa_p"][fecha_b2]`; sai o `primeiro` e a convenção "soma zero". Decisão do dono (P1): o texto da F1 diz "no primeiro apoio". | §12.2 F1, F4; S §8.2; Parte 0 item 1 |
| S3 | `g1_v3/comando.py:390` | `afasta = max(|c| − a)` mede o sítio da palma até o plano da face. Com o pad na face, o sítio já está a `palma_face` = 0,035 (medido: 0,0354–0,0368 em `palmas_com_caixa_b`). O alvo do Φ(B3) põe o sítio a a + 0,085 (`alvos.py:147`). | O B3 fecha com o pad a 1,5 cm da face, e não a 5 cm (S §8.2). No fecho, Φ(B3) = exp(−(0,035/0,05)²) = 0,61: o alvo do Φ fica 3,5 cm além do ponto que fecha. Ao levantar no B4, a palma a 1,5 cm pode arrastar a caixa e travar o invariante. | `comando.py:390`: `afasta = (m["palmas_c"].abs() − (a + A.palma_face)[:, None, None]).amax(-1)`. Teste: palma em contato dá afasta 0. | S §8.2; §6.3 `ok_agora_botar` |
| S4 | `g1_v3/observacoes.py:279` | A observação `referencia` do crítico usa `v3_topo` (a laje da pega) em todo alvo. A F trocou as cercas e o Φ do B1 ao B3 para `v3_topo_destino` (`recompensas.py:430-439`); o crítico ficou com a pega. | BOTAR do nível ≥ 1: pega em 0,30–0,55, destino em 0,04–0,55. A pelve da referência erra até 0,42 m e o tronco até 27°: o crítico recebe a referência de uma pose que a recompensa não paga. | `observacoes.referencia` chama `recompensas.ref_do_alvo(env, env.v3_alvo_idx)` (hoje `_ref_do_alvo`, público, import dentro da função): uma fonte. §6.3 da `referencia_do_alvo` passa a dizer o topo por alvo. | §6.3, §6.5 |

## Regra do plano

| # | arquivo:linha | o que está errado | cenário | correção | plano |
|---|---|---|---|---|---|
| R1 | `g1_v3/recompensas.py:478`; plano §6.3 `portoes_do_alvo` | O portão do tronco usa |incl − IK|. A S §8.2 diz "tronco até IK + 15° (só para um lado: a vertical é sempre livre)". O plano diz "quem chama tira o módulo", e o item 1 da Parte 0 também. A `cerca_tronco` já é de um lado (`recompensas.py:172`). | Seção 1: na abertura do B1, Φ ≡ 0 com derivada 0 até o robô inclinar 2° (nível 0) a 24° (topo 0,30). O mesmo vale para o P0 do PEGAR. No começo do B4 (e do P4) o tronco está na IK e a referência é 0: Φ ≡ 0 até o tronco passar de 30°, mesmo com um lado só; o núcleo do B4 já mede o tronco. | Decisão do dono (P2). Proposta: `torch.relu(incl − IK)` em `recompensas.py:478`; portão do tronco = 1 no B4 (`portoes_do_alvo`, como o do recuo). | S §8.2, §9.1; §6.3; CLAUDE.md (gradiente) |
| R2 | `g1_v3/comando.py:425`, `:510-511` | A §6.3 diz que o invólucro guarda o latch do aviso ("o detector já avisou neste trecho"). O código passa o aviso do instante: o relógio do atraso só anda com a pose estável. | Depois do aviso o robô agacha (S1 empurra): o atraso para e o trecho não acaba. O atraso efetivo deixa de ser U[0; 1,5] s. No real o controlador troca depois do aviso, sem esperar a pose. | `self.latch |= aviso`, zerado na abertura do trecho; o `avanca_trecho` recebe o latch. Ou o dono tira o latch da §6.3 (P5). | §6.3 `avanca_trecho`; I §4.2 |
| R3 | `g1_v3/eventos.py:328-331`; `g1_v3/alvos.py:164-165`; `pilota_v3.py:257-260`; `knobs.py:147-148`, `:171-172` | Contas repetidas: base → mundo em três lugares (torch, inline, numpy); `heading_w` e `estimador.yaw` são a mesma conta com dois nomes; `Detector.tol_pelve`, `tol_tronco_graus` repetem `Alvos.tol_pelve_b4`, `tol_tronco_b4_graus` (a I §4.2 diz que são as do B4). | Duas fontes do mesmo número divergem sem aviso. | `alvos.base_para_mundo(base_xy, rumo, p)` puro (o driver já carrega o `alvos` por caminho); o `eventos`, o `caixa_alvo_w` e o driver chamam. Um yaw só. O detector lê as tolerâncias do B4. | §1.5; CLAUDE.md |
| R4 | plano §6.3, §6.4, §12.2 | Texto: a §6.4 diz que o `v3_vz_caixa_ant` é escrito "no fim do `_update_command`" (é no `compute`); a §12.2 F1 diz "no primeiro apoio" e a F4 "invariante B2–B4", a §6.3 diz B3–B4 (coerente só com o ponto no fecho do B2, S2); a §6.3 não define `afasta_palmas` (S3); a `referencia_do_alvo` diz `h = v3_topo` (S4); o contrato do `exporta_cena` não fixa o z do chão (M12). | Uma lane nova lê o contrato e erra. | O orquestrador escreve antes do despacho. | §6.3, §6.4, §12.2 |
| R5 | `g1_v3/observacoes.py:185`; `pilota_v3.py:366`, `:397-398` | O canal do alvo do ator zera no BOTAR com índice −1, isto é, no fecho do B4: o ator lê o índice macro (S §6.4: o gerador macro roda só no treino). O driver zera no primeiro aviso do detector. | Entre o fecho do B4 e o aviso o treino mostra alvo zero e o driver mostra a soltura. O detector pede palmas e juntas paradas além da pose do B4: o aviso vem junto ou depois. | Treino como o driver: no BOTAR, zera com o latch do aviso (R2), e não com o índice. Ou aceitar (janela curta). | E8; S §6.4; I §2.3 |

## Menor

| # | arquivo:linha | o que está errado | cenário | correção |
|---|---|---|---|---|
| M1 | `g1_v3/smoke.py:971` | s28: o limite da troca é 0,5 s + atraso máximo + margem, contado do começo, e não do aviso medido. O robô do reset assenta antes de o detector contar. | Aviso em 0,7 s: falha com atraso > 1,32 s; 12% por mundo, 40% em 4 mundos. O smoke da fase I não fecha. | Checar `t_troca − t_aviso ≤ atraso_max + margem`. |
| M2 | `g1_v3/smoke.py:476-485` | s25 a s28 abrem o BOTAR sobre o reset da fatia ANDAR. A caixa estacionada a 20 m não conta como caída (raio sem acerto: `heights` = clamp(z, 0, 10 m), `terrain_height_sensor.py:74-80`), mas perde a laje e cai 20 m. | Na s28 ela cai perto do robô em ~2,0 s (origem + (0,32; 0)), no fim da janela de 2,04 s: pode bater no robô ou na laje (chegada a ~20 m/s). | `_abre_botar_c` põe a caixa parada sobre a soltura (`v3_alvo_w`, v = 0), ou a reestaciona. |
| M3 | `g1_v3/smoke.py:916` | s27 confere o toque do robô na caixa só pelos pads (`palma_E`, `palma_D`). | Pulso ou antebraço na caixa muda a queda livre sem o check ver. | `caixa_robo.found` (caixa × subárvore do robô). |
| M4 | `g1_v3/metricas.py:347-355` | `deslocamento_depois_do_apoio` vale de B2 a B4 com o ponto 0 antes do apoio, e no B4 da perda com o ponto velho ou 0. | Quase todo BOTAR passa pelo B2 antes do toque: a métrica dá a norma xy da caixa no mundo (~58 m no ladrilho (0,0)), e o `max` a guarda. | Só B3–B4, com o ponto do fecho do B2 (S2) e sem `v3_perdeu`. |
| M5 | `g1_v3/metricas.py:319-355`, `phi_*`, `tempo_*` | O mjlab tira a média sobre todos os envs que resetam (`metrics_manager.py:113-124`). As métricas da F somam zeros dos envs que nunca abriram o BOTAR. | A velocidade de chegada aparece diluída pela fração da fatia C. | Devolver, para todo env, a média corrente dos eventos (a média de uma constante é ela), ou logar soma e contagem. |
| M6 | `g1_v3/comando.py:542`, `:330` | `_resample` mede `_mede()` a cada reset, antes do `forward`, só para o `_abre_botar`, que nunca abre no reset. `_abre_p0` roda `saida_b` em todo env. | Um `_mede` a mais por passo com reset. | `m=None` no `_abre_trecho`; o `compute` passa o seu. |
| M7 | `g1_v3/comando.py:361` | `_alvos_pegar` cobre PEGAR e BOTAR. | Nome engana. | `_alvos`; `test_comando_e.py:114`. |
| M8 | `g1_v3/recompensas.py:444`; `g1_v3/eventos.py:274` | Docstring cita o `recuo_carregar`; a DÚVIDA da E1 que a E11 mandou apagar ficou. | Texto. | Apagar. |
| M9 | `g1_v3/comando.py:532` | O reset registra o BOTAR pendente como falha também quando o `time_out` de 20 s corta o ANDAR final. A PD-50 pede "ANDAR final sem terminação"; o `time_out` não é terminação (S §10). | BOTAR lento (aviso depois de ~18,5 s) com B4 fechado conta falha. | P7. |
| M10 | `registra_juntas_v3.py:104-114` | `limpa_laje` tira a laje no CARREGAR parado; o treino só tira no CARREGAR que anda (`comando.py:499`). | Roteiro `c`: 1,0 s de mapa sem a laje da pega, que o treino nunca viu. | `limpa_laje` só com marcha no CARREGAR. |
| M11 | `registra_juntas_v3.py:76` | BOTAR de 7,0 s fixo; o roteiro não corta no aviso. | Depois do aviso o one-hot fica em BOTAR até ~5 s; no treino, no máximo 1,5 s. | Cortar em aviso + atraso, ou conferir `aviso_fim` na fase I. |
| M12 | `pilota_v3.py:323` | O driver põe o chão em z = 0. | Certo na coluna `plano` (a fatia C e o reset do exportador). | Fixar no contrato do `exporta_cena`. |
| M13 | `g1_v3/comando.py:385-392`, `:408-427` | Detector e B4 correm relógios próprios. O B3 fecha sem exigir postura; se o robô fica de pé antes de o B3 fechar, o aviso vem antes do B4 e o atraso pode ser curto. Depois da troca o B4 não fecha mais. | Aviso Δ s antes do fecho do B4: falha com probabilidade Δ/1,5. Os portões do Φ(B3) (tronco e recuo da IK) seguram o robô agachado, e o caso deve ser raro. | Medir na fase I a fração de BOTAR com aviso antes do B4. |
| M14 | `g1_v3/recompensas.py:449`, `:683` | No BOTAR, com B4 e −1, a `cerca_recuo` mede contra `ref.recuo(v3_topo)`, a pega. A caixa já está solta. | Até 0,027 × 0,2492 por passo. | Cerca do recuo zero no BOTAR com B4 e −1. |
| M15 | testes | Nenhum teste de invólucro da F: `posiciona_destino`, `_abre_botar`, `chegada`, ponto de apoio, `afasta`, `referencia`. A checagem estática de nomes (E11 M9) não entrou. | S1 a S4 passam verdes. | Lane de testes. |

---

## Pendências

Veredito de cada item de `pendencias`.

| item | veredito |
|---|---|
| E11 lane 3: literais em `metricas._com_caixa` | Resolvido (`metricas.py:219`). |
| E11 lane 3: `phi_*` e `tempo_*` diluídos | Confirmado; vale também para as métricas da F (M5). |
| E11 lane 1: recuo do transporte com duas fontes | Resolvido: `_recuo_ref` usa `ref.caixa_transporte_b`; o knob saiu. |
| E11 lane 6: `_r_t` no driver | Resolvido (`tem_leitura()`, `pilota_v3.py:360`). O smoke ainda escreve `_r_t` (estado forçado, aceitável). |
| E11 lane 5: `_z_pe_min` privado | Resolvido (`z_pe_min`). |
| E11 lane 5: PEGAR sem leitura acaba no `time_out` e registra falha | Confirmado. Consequência: pelos números da E11 (centro fora do FOV em 4%, 17% e 43% das aberturas nos níveis 2, 3 e 4), o nível 4 do PEGAR fica com teto de ~57% de sucesso, abaixo do portão de 0,75 (P6). Esse PEGAR ainda paga a S1 por 19 s. |
| E11 lane 5: `longe` lê `v3_alvo_w` | Resolvido (`comando.py:468-470`). |
| E11 lane 5: `saida_b` em todo env | Confirmado; custo pequeno (M6). |
| F3: primeiro contato sem trava | Confirmado e sem dano: a máscara BOTAR já é "depois da abertura"; um quique ou um novo pouso é julgado de novo, o que é a regra. |
| F3: `v3_vz_caixa_ant` no reset | Resolvido (`035d9b1`): a cadeia escreve depois do reset. |
| F6: métrica depois da terminação | Resolvido (`035d9b1`): as duas leem a vz do passo anterior. |
| F6: "depois do apoio" de B2 a B4 | Defeito na métrica (M4); com o ponto no fecho do B2 (S2) vira B3–B4. |
| F2: `_base_para_mundo` repetido; `heading_w` × `estimador.yaw` | Duplicação (R3). As duas contas de rumo são idênticas. |
| F2: laje teleportada com a caixa na mão | Medido (seção 3): folga ≥ 0,22 m da caixa; ≥ 7,8 cm do bico do pé. Sem colisão. |
| F2: soltura em float64, z = topo | Confirmado e certo: `comando.py:468-469` converte e soma a. |
| F4: `recuo_carregar` sem uso; mock da S7 | Resolvido: sobra só em docstring (M8). |
| F4: fundo do pouso × chão pelo pé | Sem divergência: o fundo mede contra o topo da laje, que o `_laje` pôs em `env_origins` z + topo; o chão pelo pé é só do P4. A fatia C é sempre plano. |
| F4: `desloc` B3–B4 × "B2–B4" da §12.2 | A §12.2 dá o nome do invariante da S §8.2. Coerente com a §6.3 só se o ponto sair do fecho do B2 (S2, R4). |
| F4: `_recuo_ref` no CARREGAR | Sem efeito: o progresso não vale no CARREGAR e o índice fica no P4, cuja referência já é o transporte. |
| F1: `_alvos_pegar` com o nome velho | M7. |
| F1: "sem apoio" = soma zero; primeiro apoio em qualquer B | Defeito (S2). Na abertura a laje não toca a caixa (seção 3). |
| F1: `afasta` = max(|c| − a) | Defeito (S3): falta o `palma_face`. |
| F1: B4 por `m["h"]`, detector por `z_pe_min` | No plano coincidem em milímetros (a fatia C é plano). O B4 da perda no CARREGAR de terreno (fase G) diverge: unificar lá. |
| F1: BOTAR registra no fim; reset conta falha | Confirmado; o caso do `time_out` no ANDAR final é M9. |
| F1: `_mede()` no reset | Confirmado (M6). |
| F1: `peso_N` pela gravidade do modelo | OK. |
| Frame do `v3_alvo_w` | A premissa falha (seção 4): os dois alvos estão no mundo verdadeiro. |
| `recompensas.py:444` | M8. |
| F7: chão em z = 0 | OK no plano (M12). |
| F7: `_base_para_mundo` numpy × torch | R3. |
| F7: driver zera o alvo no aviso | As tolerâncias de pelve e tronco são iguais nos dois knobs (R3); o detector pede ainda palmas e juntas paradas, e o B4 o invariante. A diferença é de janela e de origem: R5. |
| F7: BOTAR de 7,0 s | M11. |
| F7: `limpa_laje` no CARREGAR parado | Confirmado (M10). |
| F8: s28/s29 sobre o reset de outra fatia | A caixa estacionada não conta como caída (`heights` = 10 m pelo fallback), mas cai 20 m quando a laje sai (M2). A s29 não abre o BOTAR: sem efeito. |
| F8: margens | s27: exata (passo 7 = 7). s29: +1 passo. s28: a margem não é o problema; a base de 0,5 s é (M1). |
| F8: palmas só pelos pads | M3. |

## Pendências do dono

| # | pergunta | opções | recomendação |
|---|---|---|---|
| P1 | Onde nasce o ponto de apoio do invariante B3–B4 (S2)? | A: primeiro contato (código, texto da F1). B: fecho do B2, a caixa "apoiada" da tabela da S §8.2 (premissa da Parte 0). | B. Tira o conflito com o Φ(B2) e o beco do toque raspado. comando.py −3/+2. |
| P2 | Portão do tronco do Φ (R1). | A: |incl − IK| (hoje). B: um lado só, como a S §8.2, e portão 1 no B4. C: só um lado. | B. Abre o gradiente no B1 e no P0; no B4 o núcleo já mede o tronco. recompensas.py +1. A Parte 0 (item 1) usou A: o B_alvo não muda, o Φ de partida sobe. |
| P3 | Pouso devagar. O `fator_pouso` dentro do Φ telescopa: no fecho do B2 a caixa está parada e o fator é 1. O retorno não muda com a velocidade de pouso; só a `chegada` (penhasco) decide. | A: aceitar. B: o B_alvo[B2] multiplicado por exp(−(v_chegada/σ)²), com σ do degrau e a vz do primeiro contato num buffer `v3_v_chegada` (nenhum termo novo; muda a forma do bônus). | Medir na fase I a `velocidade_de_chegada` (sem diluição, M5). Se ficar perto do limite do degrau, B (~+5 linhas). |
| P4 | O ANDAR final paga 1,725 e é alcançado pelo detector; a escada B1–B4 inteira paga ≤ 0,20; largar de ≤ 7 cm perde ≤ 0,14. | A: aceitar e medir. B: rever a `altura_pe` no ANDAR parado (paga 1,0/s com o pé parado), fora da F. | A. O sucesso do BOTAR exige o B4, e o portão do currículo pega o atalho. Medir a fração de B2 fechado com as duas palmas. |
| P5 | Latch do aviso (R2). | A: latch, como a §6.3. B: sem latch; a §6.3 muda. | A (+3 linhas). Casa com o real e com o R5. |
| P6 | PEGAR que nunca vê a caixa (E11 lane 5). | A: não registrar o PEGAR que não abriu o P0 (a falha é da percepção). B: falha, como hoje. | A. Com B o nível 4 do PEGAR não chega ao portão. |
| P7 | BOTAR com B4 fechado e o `time_out` de 20 s no ANDAR final (M9). | A: falha (hoje). B: sucesso. | B: o `time_out` não é terminação na S §10. |

## DÚVIDAS no código

| marca | veredito |
|---|---|
| `g1_v3/smoke.py:933` (± 2 passos no disparo de 10 cm) | A conta dá o passo exato (7 = 7). A margem fica; apagar a marca. |
| `g1_v3/smoke.py:949` (margem de 2 passos na s28) | A margem serve; o defeito é a base de 0,5 s (M1). |
| `pilota_v3.py:323` (chão em z = 0) | OK na coluna `plano`. Fixar no contrato do exportador (M12) e apagar a marca. |
| `pilota_v3.py:395` (tolerâncias do B4 × detector) | Os números são iguais; os dois knobs viram um (R3). A janela é o R5. |
| `registra_juntas_v3.py:75` (7,0 s) | M11: conferir `aviso_fim` na fase I. |
| `registra_juntas_v3.py:108` (laje no CARREGAR parado) | Confirmado: alinhar ao treino (M10). |
| `g1_v3/eventos.py:274` (da E1) | A E11 mandou apagar; segue lá (M8). |

---

## Lanes de correção

Arquivos disjuntos dentro de cada onda. Contagem antes → depois (estimada).

**Onda 0 (orquestrador, Opus):** as decisões P1 a P7 e a lane 8 escrevem o plano antes dos testes.

**Onda 1 (testes, TDD, Haiku)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 5 — testes | `g1_v3/testes/test_comando_f.py`, `test_recompensas_f.py`, `test_observacoes.py`, `test_eventos_f.py`, `test_alvos.py` | invólucros com env falso: `cerca_pelve` com −1 vale 0 em pé (S1); ponto no fecho do B2 (S2, P1); `afasta` 0 com o pad na face (S3); `referencia` usa o topo de destino no B1–B3 (S4); latch (R2, P5); portão do tronco de um lado e 1 no B4 (P2); `base_para_mundo` (R3); checagem estática de nomes | +~90 linhas de teste |

**Onda 2 (código, em paralelo)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 1 — cadeia | `g1_v3/comando.py`, `g1_v3/testes/test_comando_e.py` | S2 (P1); S3; R2 (P5); M6; M7; M9 (P7); P6 | 735 → ~738 |
| 2 — recompensas | `g1_v3/recompensas.py` | S1; R1 (P2); `ref_do_alvo` público; M8 (docstring); M14; P3 se B | 761 → ~763 |
| 3 — observação e métricas | `g1_v3/observacoes.py`, `g1_v3/metricas.py` | S4 (chama `recompensas.ref_do_alvo`); R5; M4; M5 | observacoes 373 → ~372; metricas 428 → ~430 |
| 4 — base → mundo | `g1_v3/alvos.py`, `g1_v3/eventos.py`, `pilota_v3.py` | R3 (`alvos.base_para_mundo`; um yaw; o detector com as tolerâncias do B4); M8 (marca da E1); M12 | alvos 224 → ~227; eventos 450 → ~445; pilota 843 → ~839 |

⚠ A lane 3 chama o `ref_do_alvo` que a lane 2 publica: na mesma onda, o nome entra no contrato da lane 8 antes.

**Onda 3**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 6 — smoke | `g1_v3/smoke.py` | M1; M2; M3; apagar as marcas de `:933` e `:949` | 1054 → ~1057 |
| 7 — roteiro | `registra_juntas_v3.py` | M10; M11 | 410 → ~410 |

**Lane 8 (orquestrador, Opus; antes da onda 1)**

| lane | arquivos | correções |
|---|---|---|
| 8 — plano e knobs | `docs/planos/2026-10-14-implementacao-g1-v3.md`, `g1_v3/knobs.py`, `g1_v3/exporta_cena.py` (só o texto do contrato) | R4: §6.3 (`afasta_palmas` com `palma_face`; ponto do fecho do B2; portão do tronco; latch; `ref_do_alvo` e o topo por alvo; `base_para_mundo`), §6.4 (`v3_vz_caixa_ant` no `compute`; `v3_ponto_apoio` no fecho do B2), §12.2 F1 e F4. `knobs.py`: sai `Detector.tol_pelve` e `tol_tronco_graus` (R3). Contrato do exportador: chão em z = 0 na coluna `plano`. §20: linha desta revisão. |

Ordem: lane 8, lane 5 (os testes falham), lanes 1 a 4, lanes 6 e 7. O smoke da fase I só vale depois das lanes 1, 2 e 6.
