---
tipo: frente
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Um achado por memória g1-limpo-*: o fato medido, o porquê e como aplicar. Complementa as notas por run das três linhas."
---
Corpo original de cada memória `g1-limpo-*`, copiado em 06/10/2026. A memória do Claude guarda só a linha do fato e aponta para cá.

## Filhos
<!-- filhos:inicio -->
- **g1-limpo-aborto-por-perda-de-preensao** (achado · ativa) — o robô mergulha atrás da caixa caída porque perder a caixa custa o mesmo que cair; v3 02/10: a perda troca para ANDAR parado (teste com volta atrás), trecho pobre
- **g1-limpo-alvo-da-pega-era-circular** (achado · ativa) — altura_carregar 1,02 vinha de medir onde a caixa JÁ estava; o piso real do alvo é 0,80 e vem do tol_pos, não da anatomia
- **g1-limpo-anda-agachado-com-a-caixa** (achado · ativa) — o robo pega de pe e anda agachado com a caixa porque o postura_ereta esta zerado na coluna CARREGAR e a justificativa do zero caducou
- **g1-limpo-aperto-fixo-nao-acompanha-massa** (achado · ativa) — MEDIDO 05/10 (model_24350): o aperto é o LIMITE DO MOTOR (shoulder_yaw ±25 N·m, wrist_yaw ±5), força total 230–250 N com qualquer μ e massa; pads 37–40 N; nada cobra o excesso; DR de atrito sozinha reforça o esmagar
- **g1-limpo-auditoria-gradientes** (achado · ativa) — Auditoria de todos os gradientes do g1_limpo (2026-09-04): 10 buracos, ranqueados; o relatório está em docs/planos
- **g1-limpo-batente-e-restricao-mole** (achado · ativa) — o limite de junta do MJCF é restrição mole e o ankle_roll sai 50% do curso; a rampa que cobra tem de acabar ALÉM de onde a junta vive
- **g1-limpo-botar-alinhado-pune-endireitar** (achado · ativa) — o alinhado do BOTAR congela a normal da caixa TOMBADA nas mãos e passa a exigir que ela mantenha o tombo; o sigma_ori trava no piso e o gradiente morre
- **g1-limpo-botar-deita-o-tronco** (achado · ativa) — BOTAR é a única tarefa reprovada pelo dono (18/09): tronco deitado demais, braço dobrado em 55% do alcance, perna ainda virando; hipótese de hábito do resume, a testar no treino do zero
- **g1-limpo-botar-e-bang-bang-nao-pressa** (achado · ativa) — o BOTAR do model_13999 e bang-bang; o alvo e alcancavel ate a laje no chao, e agachar troca inclinacao de tronco por 0,05 de margem
- **g1-limpo-botar-esquecido-na-fase-4** (achado · ativa) — MEDIDO por sonda de rede (30/09 e 01/10): o crítico desvaloriza descer e fechar no BOTAR e o ator manda subir a caixa; o resume K1b + R1 levou 48% das C ao BOTAR sem nenhum fecho; o fecho não exige de pé
- **g1-limpo-botar-paga-nao-botar** (achado · ativa) — no g1_limpo o BOTAR nunca fecha; a politica achou duas fugas em sequencia — arremessar e segurar para sempre — e as duas vinham do _alcancar identico a 1
- **g1-limpo-cadeia-da-pega** (achado · ativa) — A cadeia mao->laterais->preensao->unload->alvo do g1_limpo: onde ela quebrava e o que a consertou
- **g1-limpo-caixa-esmagada-squeeze-cego** (achado · ativa) — O robo aperta a caixa com 8x a forca necessaria e nenhum termo cobra por isso: squeeze satura em 1/8 da forca aplicada
- **g1-limpo-caixa-presa-pelo-pulso** (achado · ativa) — MEDIDO 05/10 (model_24350): a caixa é presa pelas cápsulas do pulso e pelo tronco, além dos pads; força total ~230–250 N fixa; com `priority` na caixa, μ ≥ 0,6 segura 1–5 kg e μ ≤ 0,45 escorrega a caixa pesada (dono viu no viewer)
- **g1-limpo-carregar-parado-e-estatua** (achado · ativa) — No CARREGAR parado o rastreio ×3,5 pagava 14/s por ficar; o R2 criou desvio pago e multiplicador inobservável; o K1b (×1 no regime parado, em toda cadeia) o substitui, commitado e em treino
- **g1-limpo-cauda-andando-herda-twist-zero** (achado · ativa) — CÓDIGO 01/10: a cauda CARREGAR que anda herda vx = vy = 0 do regime parado por ~2,9 s, e o rastreio ×3,5 paga a estátua nesse trecho
- **g1-limpo-cauda-carregar-piso-sem-derivada** (achado · ativa) — 60% da renda viva da cauda CARREGAR e piso sem derivada, e a cauda nao tem destino no mundo: o alvo e ancorado no peito
- **g1-limpo-checkpoints-que-completam-o-ciclo** (achado · ativa) — MEDIDO 01–02/10 nos plays + veredito do dono: quais checkpoints pegam, botam e saem andando; MELHOR até 02/10 = model_23250 (21700 refinado no v2)
- **g1-limpo-curriculo-de-cadeia** (decisao · ativa) — DECISÃO 30/09 (zero17): a manipulação abre por fase — PEGAR (90% SEGURA), CARREGAR parado, CARREGAR andando, BOTAR — com troca por s_B e s_cauda; C = (PEGAR, CARREGAR parado, BOTAR); sucesso = fecho E time_out; o dono retoma do model_3100
- **g1-limpo-deriva-de-rumo-com-a-caixa** (achado · ativa) — deriva de rumo sem comando; 17/09 laço de rumo aplicado; MEDIDO 02/10: com k = 0,5 a marcha reta assenta 6–13° fora do rumo (erro = viés/k); kernel de rumo efetivo 81°; proposta k = 2
- **g1-limpo-escala-da-acao-do-punho** (achado · ativa) — A escala da ação do wrist_pitch/yaw é 0,075 rad por unidade (6x menor que o braço): a política não explora nem aprende o giro do punho que mantém a caixa reta ao erguer
- **g1-limpo-escorar-compra-o-fecho** (achado · ativa) — o robô se apoia na caixa porque isso dispara o `apoiada` e abre a CAUDA, que paga 58/s contra 35/s do BOTAR
- **g1-limpo-escoro-e-o-tronco** (achado · ativa) — Quem escora na mesa no g1_limpo e o que a particao das tres terminacoes mediu
- **g1-limpo-espera-sem-agacha-o-andar-parado** (achado · ativa) — MEDIDO 01/10: o agachamento nasce nos 0,5 s após o reset, dentro do de_pe (Δ 0,67 vs 0,69 rad); custa 0,3/s de pose e adianta o fecho ~0,5 s (~7 de renda); a espera inicial é indistinguível da locomoção parada para o ator
- **g1-limpo-faixa-de-pose** (achado · ativa) — braço direito no batente em TODA fase, inclusive antes da pega; o pose não corrige porque é exp(−média) de 21 juntas
- **g1-limpo-fechar-o-pegar-mata-o-episodio** (achado · ativa) — MEDIDO 30/09: 28 de 28 fechos forçados do PEGAR terminam em ≤1,6 s (24 caixa_largada, 4 queda); o one-hot CARREGAR faz a política largar a caixa, mesmo com twist 0 — por isso o PEGAR nunca fecha; o currículo de cadeia (zero17) resolve
- **g1-limpo-fechar-paga-menos-que-pairar** (achado · ativa) — Fechar o PEGAR rende menos do que pairar no alvo (16,5 vs 25/s), e 'fechou mas não avançou' paga o dobro (32/s): explica os fechos que aparecem e somem
- **g1-limpo-forma-postural** (achado · ativa) — o termo `forma_postural` — incentivo de forma contra a referência da IK, seis grandezas, média simples; a média simples dilui o quadril a 6% da renda do BOTAR e a torção da perna piora a cada bloco
- **g1-limpo-forma-postural-agacha-no-alvo** (achado · rejeitada) — REFUTADO como causa (30/09): o clamp da forma_postural em h 0,68 puxa só 0,12/s para o agachamento; pose+postura_ereta puxam ~0,7/s para ficar de pé
- **g1-limpo-freio-nao-taxa-a-marcha** (achado · ativa) — o velocidade_por_regime a −15 cobra ~0 de uma marcha treinada (0,0003/s a 1 m/s), mas custa −14,8/s com o ruído do começo do treino (σ 0,96)
- **g1-limpo-giro-canal-e-envelope** (achado · ativa) — O robo nao girava por dois motivos somados: o termo de guinada media o gingado nao comandado, e o envelope era lento demais; o sigma tem de acompanhar o envelope
- **g1-limpo-giro-no-lugar-empata-no-estagio-0** (achado · ativa) — Comando baixo não é respeitado: parado, o robô colhe exp(−c²/σ²) do rastreio (96% a 0,1 com σ 0,5; 98% com o piso 0,707 do g1_limpo); todos os modelos giram estranho ou pedem comando mais alto; v3 normaliza o rastreio pelo comando
- **g1-limpo-ik-referencia-botar** (achado · ativa) — IK de referência da postura; DONO 01/10 aprovou as tabelas da v3 (ref_v3_manipulacao.npz e ref_v3_de_pe.npz) com folga de 10 mm na laje — a ref_botar.npz antiga punha a perna dentro da laje em 12 de 15 poses
- **g1-limpo-impacto-da-caixa-sem-preco** (achado · ativa) — o impacto da caixa nunca foi cobrado; MEDIDO 01/10: a métrica impacto_da_caixa (pico de força por passo de controle) é ALIASADA e não diz a altura da queda; v3 mede a velocidade de chegada (limiar 0,99 m/s = queda de 5 cm)
- **g1-limpo-inercia-real-da-caixa** (achado · ativa) — a inércia real da caixa é gravável nos dois caminhos: mj_setConst no MuJoCo clássico, e o dr de massa do mjlab parou de corromper no 1.5.3
- **g1-limpo-janela-de-espera** (achado · ativa) — g1_limpo tem a janela de espera do g1_poc; decisão de 2026-09-02: ela passa a publicar ANDAR e o VALIDA sai da observação
- **g1-limpo-limite-de-junta-nao-morde-no-curso-largo** (achado · ativa) — fração do curso não é limite em junta de curso largo (hip_yaw 316°): o hip_yaw foi a 151° no BOTAR; a rampa em 40° virou preço pago; do zero só o clip da ação impede
- **g1-limpo-modo-entrega-viewer** (achado · rejeitada) — entrega de tarefa por evento no viewer foi REJEITADA; a task de entrega é PEGAR com espera alongada
- **g1-limpo-nao-anda-com-a-caixa** (achado · ativa) — O robo nao anda carregando a caixa por INCENTIVO, nao por capacidade: andar acrescenta 5,2% a uma renda que ja e 94% ligada a caixa
- **g1-limpo-one-hot-normalizado-amplifica-slot-raro** (achado · ativa) — MEDIDO 30/09: a EmpiricalNormalization do rsl_rl põe o slot raro do one-hot a ~95× (CARREGAR) e 25× (REORIENTAR); desde a zero17 o PPOPorElo fixa média 0 e desvio 1, e o load dobra a normalização antiga na 1ª camada
- **g1-limpo-pega-parada-na-mesa** (achado · ativa) — zero12 parou na pega: tira o peso da mesa e não leva ao peito; o tombo é cinemático (eixos coaxiais) e só o punho desfaz; zero14 = unload × _trazer + pose pelo estado
- **g1-limpo-pico-altura-diluido** (achado · ativa) — Episode_Metrics/pico_de_altura do g1_limpo é diluído ~3,6×; o número físico é o Metrics/peak_height_mean do fabricante
- **g1-limpo-piso-da-estatua-travava-a-pega** (achado · ativa) — No g1_limpo, ficar parado no elo de manipulacao pagava mais que tentar (31/08, consertado); em 23/09 o forma_postural paga 0,81/s a estatua
- **g1-limpo-piso-segura-o-s-b** (achado · ativa) — MEDIDO 30/09 por simulação: o piso uniforme de 20% do nível segura o sucesso abaixo de 0,5 até o nível médio chegar perto de 3; o portão s_B ≥ 0,50 da fase 2 exige a pega no nível 3
- **g1-limpo-plano-tres-tarefas** (achado · ativa) — decisao do dono em 2026-09-08 para o g1_limpo: fechar o BOTAR primeiro; depois reestruturar em A/B/C/R com um so comportamento ANDAR e balanceamento por sucesso; gatilhos definidos
- **g1-limpo-portao-diluido-pela-fatia** (achado · ativa) — O portao da forma no g1_limpo lia media diluida pela fatia de manipulacao, e isso tinha teto fechado
- **g1-limpo-postura-com-caixa-medida** (achado · ativa) — como a politica model_6999 do g1_limpo segura a caixa, medido em CPU em 2026-09-08 — altura 1,02 m, x 0,25, postura assimetrica na perna esquerda; e a sonda de nivel forcado que nao agiu
- **g1-limpo-postura-de-pe** (achado · ativa) — MEDIDO 30/09: a 'bunda empinada' (pelve +22°, cintura −0,33, pelve 0,72 m) vem da forma_postural presa na linha da laje de 0,55 com a caixa no peito; IK dá a pose de pé (pelve 0,79, nivelada, joelho 0,15–0,42); spec postura-de-pe
- **g1-limpo-punho-esquerdo-empurrado-pela-caixa** (achado · ativa) — MEDIDO 05/10: o giro do wrist_yaw E na pega é COMANDO + CONTATO; a política manda girar só no elo PEGAR (ação +8,5, alvo +0,60 rad no 24800; +0,34 no 24350) e a caixa empurra mais 0,6 rad; no CARREGAR a política manda ~0 e a mão volta
- **g1-limpo-punho-vai-ao-batente** (achado · ativa) — no g1_limpo apertar o std do punho no `pose` matou o termo em vez de discipliná-lo; revertido para 1,00 em 2026-09-11
- **g1-limpo-rampa-estreita-nao-ensina** (achado · ativa) — no comando PARADO (v=0) a ação MÉDIA leva o waist_pitch ao batente em 7 passos; com NoBatente todo episódio de manipulação morre ali e PEGAR nunca treina; a terminação (−4) esconde a rampa (−53,6/s)
- **g1-limpo-rampa-queima-janelas** (achado · ativa) — A rampa de fatia do g1_limpo avança 22x mais devagar que o desenho: ela queima as janelas puladas quando o sinal fica na banda morta
- **g1-limpo-reducao-e-limiar-andam-juntos** (achado · ativa) — o amax no velocidade_por_regime matou a movimentacao do G1 e foi REMOVIDO; o codigo roda mean nos tres regimes e quem protege o batente e o limite_de_junta
- **g1-limpo-regime-vem-do-estado** (achado · ativa) — o laço de rumo tirava a manipulação do regime standing e afrouxava o freio de velocidade em 2,7x; corrigido em 21/09 mas ainda não treinado
- **g1-limpo-registrador-cego-ao-giro** (achado · ativa) — MEDIDO 30/09: o registra_juntas.py monta a observação com giro_w = 0, e o ator fica cego ao tombo da caixa; todo tombo medido em play está confundido; o 5000_c.csv é do model_5200
- **g1-limpo-renda-congelada-paga-10x-o-deficit** (achado · ativa) — A renda_congelada ACUMULA por fecho; o desperdicio e o empilhamento, nao o fecho isolado. Sobrepaga 10,3x na cadeia B e so 1,2x por fecho na C
- **g1-limpo-sigma-do-alvo-congelado-degenera** (achado · ativa) — σ = erro inicial é incompatível com alvo = atitude atual; no PEGAR isso deixava tombar a caixa de graça, medido em 53 a 59 graus
- **g1-limpo-std-cai-cedo** (achado · ativa) — o std cai rápido no começo em toda run desta pilha (fábrica 1,0→0,61 na it 200); na g1_limpo o freio −15 aprofunda a queda até ~0,45, onde a dobradiça desliga; o LR da config não importa
- **g1-limpo-tabela-por-estado** (achado · ativa) — Desenho da recompensa sequencial do g1_limpo: teto do presente >= piso do passado; tabela peso[estado][termo] em 10 estados; renda_congelada fica como piso; a orientação paga em toda tarefa de manipulação
- **g1-limpo-tag-estavel-smoke-vermelho** (achado · ativa) — a tag estavel-bloco17 nasce com 2 falhas no smoke; as duas são contadores que não subiram com o faixa_de_pose, não defeito de código
- **g1-limpo-teleporte-da-laje-no-corpo** (decisao · ativa) — MEDIDO 30/09: a abertura do BOTAR teleporta a laje para dentro da perna em 31–72% das aberturas; DECISÃO 01/10: fica a DR ±0,10 em xyz, com o sorteio em x só afastando a laje (toque 0,6% de pé)
- **g1-limpo-velocidade-de-junta-medida** (achado · ativa) — medição de velocidade de junta no bloco9: a manipulação parada é mais rápida que correr, e a causa é o rastreio desligado (P4)
- **g1-limpo-velocidade-por-regime-formula-obsoleta** (achado · ativa) — o knobs.py documentava 1-exp para velocidade_por_regime mas o codigo roda um clamp; comentario corrigido em 2026-09-11, peso -2,0 mantido, clamp segue como licenca acima do teto
- **g1-limpo-yam-lento-por-desenho** (achado · ativa) — o exemplo YAM do mjlab (levantar cubo) é lento por servo de 2 Hz, freio em currículo e recompensa sem prêmio de tempo; na g1_limpo a pressa é paga pela renda_congelada (6,9 por 0,5 s)
<!-- filhos:fim -->
