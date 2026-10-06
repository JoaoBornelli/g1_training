---
tipo: abordagem
pai: "[[Linha zero]]"
estado: parada
verificado: 2026-10-06
periodo: 29/09/26
resumo: "Caminho B, staged e precise_pos veem o tombo só pela reta; sem registro de treino."
---
- **Commit:** `7a23ea9`. **Caminho B, escolhido pelo dono:** `staged` e `precise_pos` veem o tombo só pela reta `1 − Δθ/π` (0,83 a 30°, 0,67 a 60°). O `precise_ori` mantém o híbrido com a gaussiana de 25°. Erguer tombado passa a pagar +4,3/s a 30° e +2,6/s a 60°; reto, +7,4/s (`git:7a23ea9`; REL-MESA §5).
- **Caminho A, rejeitado nessa data:** baixar a âncora para 0,75–0,85 (REL-MESA §5). Ele voltou na zero15.
- **Rejeitado:** `precise_ori` 4 → 0. Ele paga o mesmo na mesa e no alvo com a caixa reta, e zerá-lo só barateia o tombo (REL-MESA §0).
- **Treino:** nenhuma fonte registra resultado da zero13. A auditoria de recompensa foi feita sobre o HEAD dela (REL-AUD), e a zero14 "o leva junto" (REL-AUD §0).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
