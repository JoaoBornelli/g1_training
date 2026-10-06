---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "BOTAR é a única tarefa reprovada pelo dono (18/09): tronco deitado demais, braço dobrado em 55% do alcance, perna ainda virando; hipótese de hábito do resume, a testar no treino do zero"
---
VEREDITO DO DONO, 2026-09-18, no `model_21700` (resume com terminação): todas as tarefas
satisfatórias, em toda altura, MENOS o BOTAR. Ele bota, mas "inclina o tronco demais sem
esticar os braços" e "ainda está tentando girar a perna para fora". PEGAR e BOTAR
deveriam ser a mesma movimentação inversa.

MEDIDO no BOTAR: tronco 65° (referência 67°), pelve 0,53 (ref 0,52–0,58), ombro→palma
0,25 m (ref 0,41–0,46; máximo 0,45), `hip_yaw` −34° (ref ±9°; era −96° antes da
terminação). O robô copia o corpo da referência e não o braço.

HIPÓTESE DO DONO: hábitos herdados do resume; o treino do zero pode não os criar. Teste:
`registra_juntas` no BOTAR do primeiro checkpoint do zero com `s_C > 0,3`.

**Why:** toda sonda de forma lê `raiz_*` e `caixa_*` do CSV; supor pelve vertical errava 3×.
**How to apply:** só decidir o conserto do braço depois do teste no zero.
**Detalhe:** `g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-botar-veredito-do-dono-21700.md`

Relacionado: g1-limpo-forma-postural, g1-limpo-limite-de-junta-nao-morde-no-curso-largo
