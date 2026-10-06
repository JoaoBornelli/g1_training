---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 28/09/26
resumo: "precise_ori híbrido e ANG pelo eixo de cima: erguia a caixa tombando com o cotovelo (17° → 60°)."
---
- **Commits:** `ff5ac85` e `04e99ca` (RUN zero10).
- **Hipótese:** a orientação não tinha gradiente longe do alvo, e o `ANG` não via o tombo.
  - `precise_ori` vira híbrido `½(1 − Δθ/π) + ½·exp(−(Δθ/σ)²)`. A metade linear dá 0,16/rad em qualquer orientação, e a gaussiana 1,0/rad na tolerância do fecho (`git:ff5ac85`).
  - O `ANG` mede o eixo de cima (capturado na mesa por `_captura_cima`) contra a vertical. O regime `FACE_CONGELADA` sai. O CARREGAR e o BOTAR herdam o eixo do PEGAR. σ de pé = max(erro inicial × fator, `tol_ang_deg`) (`git:04e99ca`).
  - Requisito do dono: a face de baixo fica para baixo; a guinada é livre (V-ANG).
- **Medida:** com o `precise_ori` em peso 1, a zero10 erguia a caixa tombando-a com o cotovelo: `caixa_na_pega` 17° → 60°, enquanto o `precise_pos` subia de 0,29 a 0,71. A 60° a derivada era 0,18/rad (`git:829ee03`).
- **Avaliação por 3 agentes (28/09):** a mudança conserta o portão, mas não destrava o PEGAR sozinha. Todo termo de segurar paga igual com a caixa tombada, e o `precise_ori` é ~2% da renda do PEGAR_COM (V-ANG).
- **Por que terminou:** o tombo ao erguer não tinha preço suficiente.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
