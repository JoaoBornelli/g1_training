---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO 01/10: treino v3, refeito simples só com o que funcionou; um modelo só é DURO; IK como alvo macro (sem mímica); cadeias com transição ficam (o ganho do g1_limpo sobre o multitask); nenhum episódio nasce com a caixa; cada tarefa com seu nível e fatia parecida de envs, giro no lugar incluso"
cluster: c4
---
DECISÃO DO DONO (01/10): v3 refeito simples, só com o que funcionou. "Um modelo só" é duro. Poses de IK viram alvos macro, sem mímica. Ficam as cadeias B (PEGAR → CARREGAR) e C (PEGAR → CARREGAR parado → BOTAR), após ANDAR parado. Nenhum episódio nasce com a caixa. Refino (02/10): primeiro pega e bota em todas as alturas, depois as restrições apertam aos poucos.

Pagar o progresso (potencial `γΦ(s') − Φ(s)`), não o estado. RISCO (05/10, R13): pagar a pose por passo no refino reabre a renda por estado.

**Why:** 15 de 26 hacks do catálogo são economia de renda por estado.
**How to apply:** spec revisada com o dono em 05/10 (`specs/g1-v3.md`); no refino o robô recebe por se aproximar da pose da IK do alvo, sem zona morta, com a forma ainda em aberto; a seguir, a Parte 0 sem GPU.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md

Relacionado: g1-limpo-enunciado-do-comportamento g1-locomocao-primeiro-fatia-100 g1-limpo-giro-no-lugar-empata-no-estagio-0
