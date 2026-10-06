---
tipo: frente
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "mjlab, Warp, Kaggle/Colab, gargalo de banda."
---
mjlab, Warp, Kaggle/Colab, gargalo de banda.

## Filhos
<!-- filhos:inicio -->
- **g1-auditoria-mjlab-bloco2** (achado · ativa) — Bloco 2 da auditoria do g1_multitask contra o mjlab — 2 dos 4 itens rejeitados com razão medida, A11 (pé no table_contact) era real
- **g1-auditoria-mjlab-bloco3** (achado · ativa) — Bloco 3 da auditoria do g1_multitask — condim, histórico de obs, sensores e terreno TODOS rejeitados com medição; só a colisão de chave de log era real
- **g1-leitura-log-e-kaggle** (achado · ativa) — Como ler o log do mjlab sem inverter o diagnóstico, e as três armadilhas operacionais do treino no Kaggle
- **g1-mjlab-api-armadilhas** (achado · ativa) — Fatos de API do mjlab que enganam — root_com_pos_w não é o CoM, entidade fixed-base vira mocap; a corrupção de heap do dr.body_mass ACABOU no 1.5.3
- **g1-mjlab-mutacao-em-treino** (achado · ativa) — mjlab: o que pode mudar em tempo de treino/episódio — deepcopy assimétrico, currículo só no reset, sem duração por env, sem bootstrap em falha
- **g1-mjlab-velocity-anatomia** (achado · ativa) — Anatomia da tarefa de caminhada do mjlab (tasks/velocity, G1) — só 2 eixos de currículo, 1 falha, peso×dt
- **warmstart-lower-lr** (achado · ativa) — Ao warm-startar um treino RL num degrau/distribuição mais difícil, BAIXAR a learning_rate inicial (5e-4 vs 1e-3) pra não destruir o checkpoint nos primeiros updates
<!-- filhos:fim -->
