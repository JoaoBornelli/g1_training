---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 05/10: o giro do wrist_yaw E na pega é COMANDO + CONTATO; a política manda girar só no elo PEGAR (ação +8,5, alvo +0,60 rad no 24800; +0,34 no 24350) e a caixa empurra mais 0,6 rad; no CARREGAR a política manda ~0 e a mão volta"
---
MEDIDO 05/10 (`24800_mu06_m3.csv`, roteiro `carregar`, 3 kg, μ 0,6). No PEGAR, a ação do `wrist_yaw` E sobe a +8,5 (escala 0,075 rad/unidade, alvo do PD +0,60 rad) e a junta vai a +1,18 rad. Os 0,58 rad além do alvo pedem ~9,7 N·m contra ±5 N·m do motor: é o contato. Ao trocar para CARREGAR, a ação cai a −1, o alvo a ≈ 0 e a mão volta a +0,2 rad. No `model_24350` o alvo era +0,34: o comando dobrou entre os dois. O registrador segura o PEGAR 7 s; no treino ele fecha logo depois da subida.

**Why:** a postura de mão é por elo (one-hot); "só contato" (afirmado antes, 05/10) estava ERRADO pela metade.
**How to apply:** o giro é aprendido no PEGAR; atacar comando E momento no punho; julgar o giro só na janela de subida.

Relacionado: g1-limpo-punho-vai-ao-batente g1-limpo-escala-da-acao-do-punho g1-limpo-aperto-fixo-nao-acompanha-massa
