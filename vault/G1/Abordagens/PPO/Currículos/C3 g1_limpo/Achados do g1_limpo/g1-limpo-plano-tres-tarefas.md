---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "decisao do dono em 2026-09-08 para o g1_limpo: fechar o BOTAR primeiro; depois reestruturar em A/B/C/R com um so comportamento ANDAR e balanceamento por sucesso; gatilhos definidos"
---
**Estado (2026-09-11):** v3.4 fechou o BOTAR (`s_C` 0 → 0,34). A bloco15 mostrou três defeitos com a mesma raiz — o passado paga mais que o presente: não anda com a caixa, não levanta depois de botar, corre na pega. Resposta: a TABELA POR ESTADO (`peso[estado][termo]`, 10 estados), `renda_congelada` fica como piso, dobradiça de velocidade sem clamp, punho no `pose` com std 1,00 (0,30 matava o canal). Ver g1-limpo-tabela-por-estado. A treinar como `bloco16` no COLAB com 8192 envs, META 12500, LR 5e-4 (warm-start: o crítico refaz a escala). Smoke: 627 ok / 0 falhas (12 checks velhos consertados, nenhum defeito real). O dono fecha a arquitetura aqui para o treino final em GPU boa.

**Princípio:** um comportamento ANDAR, em dois contextos: só andar (A) e andar-com-manipulação (espera, espera com caixa, locomoção com caixa). Dois elos que mudam a cena: PEGAR e BOTAR. REORIENTAR fica em fração fixa pequena.

**Sequências (aprovadas 2026-09-08):** regra única — todo elo tem espera antes; a cadeia termina em espera → ANDAR. A = ANDAR(normal). B = espera → PEGAR → espera(caixa) → ANDAR(caixa). C = espera → PEGAR → espera(caixa) → BOTAR → espera → ANDAR. R = espera → REORIENTAR → espera → PEGAR → espera(caixa) → ANDAR(caixa). O estado CARREGAR fica como rótulo e recompensa (VALIDA 1, âncora no peito); saem só o fecho e a faixa P5. `renda_congelada` perde o `× _valida`.

**Gatilhos:** →ᵗ timer de duração sorteada, gateado em `perto` antes do BOTAR. →ᶠ fecho sustentado. `de_pe` = pose padrão de pernas e tronco; vale nos fechos de PEGAR e BOTAR. Na espera a pose só paga. Cauda pós-BOTAR: sorteio normal com giros.

**Cena e alvo (2 bits, aprovado 2026-09-08):** twist ≠ 0 ⟹ laje a +5 m (caixa junto se não está na mão) e alvo referenciado no robô, todo passo. Twist = 0 ⟹ laje presente e alvo CONGELADO em mundo no ponto `_pos_no_elo + R·peito_b`. Isso torna a deriva custosa e paga a recuperação do push (1–3 s, 0,5 m/s, ativo em todo env). Do PEGAR ao BOTAR a laje muda pouco de altura e ponto, ancorada em `_pos_no_elo`, no fim da espera. Penalidades de pé NÃO mudam: cobrariam a recuperação do push.

**Postura baixa (decidido 2026-09-08):** `fell_over` fica em 70°. Entram uma penalidade de `hip_yaw`/`hip_roll` fora do padrão (idioma do `pose_de_braco`, 4 juntas) e contato indesejado joelho/canela–chão. Motivo: as faixas do MJCF (yaw ±158°, roll −30°/+170°) tornam o joelho de lado gratuito.

**Why:** simplificar, reutilizar a espera e o ramo do ANDAR, treinar só transições essenciais.
**How to apply:** o timer gateado herda o relógio do `_segura_parado`, que fica preso a 99% em 10% dos episódios — investigar antes de reutilizar.

Relacionado: g1-limpo-botar-paga-nao-botar, g1-limpo-contrato-de-troca-de-tarefa
