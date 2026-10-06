---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 05/10 (model_24350): a caixa é presa pelas cápsulas do pulso e pelo tronco, além dos pads; força total ~230–250 N fixa; com `priority` na caixa, μ ≥ 0,6 segura 1–5 kg e μ ≤ 0,45 escorrega a caixa pesada (dono viu no viewer)"
cluster: c3
---
MEDIDO 05/10 (cópia do registrador, laje 0,45). A caixa é presa pela cápsula `*_wrist_collision` (`wrist_pitch_link`, 64–100 N) e pelo tronco, além dos pads; o `squeeze` só mede os pads (`recompensas.py:579-616`). A força total do robô na caixa fica em 230–250 N com qualquer μ e massa. Com `priority` na caixa (todos os contatos usam o μ dela; mesa acima), impratio 2: μ ≥ 0,6 segura 1–5 kg; 0,45 solta 5 kg; 0,3 solta 3 e 5 kg. O dono confirmou no viewer: a caixa escorrega.

**Why:** o MuJoCo usa o maior μ do par; DR só em pad e caixa fica mascarada.
**How to apply:** DR de atrito pela `priority` da caixa, faixa inicial [0,6; 1,0] e borda descendo por nível; métrica do aperto sobre a força total.
**Detalhe:** docs/relatorios/2026-10-01-aprendizados-zero17-para-v3.md §3b

Relacionado: g1-limpo-aperto-fixo-nao-acompanha-massa g1-limpo-punho-esquerdo-empurrado-pela-caixa
