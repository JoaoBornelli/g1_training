---
tipo: decisao
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO 30/09 (zero17): a manipulação abre por fase — PEGAR (90% SEGURA), CARREGAR parado, CARREGAR andando, BOTAR — com troca por s_B e s_cauda; C = (PEGAR, CARREGAR parado, BOTAR); sucesso = fecho E time_out; o dono retoma do model_3100"
---
DECISÃO 30/09 (zero17): dentro da manipulação, as tarefas abrem por fase. Fase 1: 90% dos fechos B/R SEGURAM no PEGAR até o fim, 10% vão à cauda CARREGAR parada. Fase 2: 100% à cauda, 90% parada. Fase 3: a cauda anda. Fase 4: o balanceador de hoje; `p_C` = 0,05 antes. A cadeia C é PEGAR → CARREGAR parado (fecho pela régua do PEGAR, 0,5 s) → BOTAR. O sucesso que move o nível e `s_B`/`s_C` é `concluiu ∧ time_out`.

**Why:** fechar levava ao CARREGAR não treinado e matava o episódio. Sonda: a SEGURA paga 23,95/s contra 11,63/s de pairar; a C inteira fecha por sustentação em 4 de 4 envs.
**How to apply:** no log, ler `Curriculum/forma/fase_cadeia`, `s_B`, `s_cauda` e `Curriculum/nivel`. A troca 1 → 2 segue o nível médio: ver g1-limpo-piso-segura-o-s-b.
**Detalhe:** specs/g1-limpo-curriculo-de-cadeia.md

Relacionado: g1-limpo-fechar-o-pegar-mata-o-episodio g1-limpo-one-hot-normalizado-amplifica-slot-raro g1-limpo-tabela-por-estado
