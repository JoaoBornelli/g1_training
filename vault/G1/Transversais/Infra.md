---
tipo: frente
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "mjlab, Warp, Kaggle/Colab, gargalo de banda."
cluster: transversais
---
mjlab, Warp, Kaggle/Colab, gargalo de banda.

## Filhos
<!-- filhos:inicio -->
- **g1-auditoria-mjlab-bloco2** (achado · ativa) — Bloco 2 da auditoria do g1_multitask contra o mjlab — 2 dos 4 itens rejeitados com razão medida, A11 (pé no table_contact) era real
- **g1-auditoria-mjlab-bloco3** (achado · ativa) — Bloco 3 da auditoria do g1_multitask — condim, histórico de obs, sensores e terreno TODOS rejeitados com medição; só a colisão de chave de log era real
- **g1-colab-8192-kaggle-4096** (achado · ativa) — O Colab roda 8192 envs do g1_limpo numa GPU de 16 GB; a Kaggle, tambem com 16 GB, so roda 4096. O discriminador e o hospedeiro, nao a VRAM
- **g1-leitura-log-e-kaggle** (achado · ativa) — Como ler o log do mjlab sem inverter o diagnóstico, e as três armadilhas operacionais do treino no Kaggle
- **g1-limpo-cone-eliptico-diverge-para-nan** (achado · ativa) — CAUSA ACHADA 05/10: elliptic+impratio 10 divergiu 2× (15/07, 17/09) por 3 defeitos upstream — Hessiano do cone indefinido perto do ápice (mujoco_warp #1657, fix PR #1660 de 23/09, AUSENTE no 3.10.0.3 instalado), busca de linha esgotada com impratio 10 e ls 20 (mujoco #3628, PR aberto), warm-start do Warp propaga o NaN e o reset não limpa
- **g1-mjlab-api-armadilhas** (achado · ativa) — Fatos de API do mjlab que enganam — root_com_pos_w não é o CoM, entidade fixed-base vira mocap; a corrupção de heap do dr.body_mass ACABOU no 1.5.3
- **g1-mjlab-mutacao-em-treino** (achado · ativa) — mjlab: o que pode mudar em tempo de treino/episódio — deepcopy assimétrico, currículo só no reset, sem duração por env, sem bootstrap em falha
- **g1-mjlab-reset-buffers-obsoletos** (achado · ativa) — Duas armadilhas de medição no mjlab — Entity.data.* está obsoleto dentro do reset, e limiar de check abaixo de 5 mm acusa a própria gravidade
- **g1-mjlab-velocity-anatomia** (achado · ativa) — Anatomia da tarefa de caminhada do mjlab (tasks/velocity, G1) — só 2 eixos de currículo, 1 falha, peso×dt
- **g1-mjlab-warmstart-obs-armadilhas** (achado · ativa) — Cinco armadilhas medidas de warm-start e observação no mjlab/rsl_rl — enxerto de entrada, normalizador de canal constante, register_buffer, seed e índice de obs
- **mjlab-curriculo-de-comando-sobrescreve-a-faixa** (achado · ativa) — o commands_vel do mjlab escreve cfg.ranges a cada tique, entao ler a faixa do twist antes do launch mostra um numero que o robo nao recebe no passo 0
- **mjlab-pose-escrita-persiste** (achado · ativa) — no mjlab uma pose escrita em corpo livre no meio do episodio e re-aplicada a cada passo; o corpo nao cai e a forca de contato vem so da penetracao
- **mjlab-warp-contato-e-dr** (achado · ativa) — VERIFICADO 05/10 (mujoco_warp 3.10.0.3): noslip NÃO existe no Warp (NotImplementedError); geom_priority é estático (sem dim. de mundo) mas decide atrito, condim e solref do par; geom_friction tem DR pronto (dr.geom_friction, coluna 0); geom_solref/solimp por env só com evento próprio; impratio 2 só no notebook, knob diz 1,0
- **warmstart-lower-lr** (achado · ativa) — Ao warm-startar um treino RL num degrau/distribuição mais difícil, BAIXAR a learning_rate inicial (5e-4 vs 1e-3) pra não destruir o checkpoint nos primeiros updates
<!-- filhos:fim -->
