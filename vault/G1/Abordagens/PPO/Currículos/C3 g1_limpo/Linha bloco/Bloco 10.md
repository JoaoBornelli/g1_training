---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: parada
verificado: 2026-10-06
periodo: 05–08/09/26
resumo: "Continuação até a it 6999; os checkpoints 5950 e 6999 alimentam as sondas: 0 de 190 BOTAR fecham, caixa no hold a 1,025 m."
---
- **Datas:** entre ~05/09 e 08/09. Nenhum commit em `g1_limpo/` entre 05/09 e 07/09 (git log).
- **O que mudou antes:** nada confirmado em git. O código provável é o v2.1 (a sonda de 08/09 roda
  "worktree no commit `8f7eca1`", spec `lento-e-estavel` §0). PROVÁVEL, não confirmado.
- **Medido:** os checkpoints dela alimentam as sondas de 08/09.
  - `model_5950` (spec `g1-limpo-dois-bits.md` §0.1, 64 envs, 20 s, CPU): BOTAR alcançado por 46,3%
    e **0 de 190 fecham**. Sem espera entre CARREGAR e BOTAR o alvo salta 0,83 m (p50) a 2,29 m (p90);
    `action_rate_l2` a −25/s; RMS de junta 0,99 → 6,88 rad/s; 82% morrem em 1 s. Base deriva 0,29 m
    → 0,85 m (p90 4,65 m) com twist zero. Relógio do hold zera a 99% (`perto` oscila no limiar de
    0,10 m em 71 de 71 resets). G2 inerte por 1022 its (`1 − exp(−x)` com derivada 0,27 em x = 1,3).
    Joelho de lado: `hip_yaw` ±158°, `hip_roll` −30°/+170°, `std_standing = {".*": 0,05}` vale ~0.
  - `model_6999` (memória `g1-limpo-postura-com-caixa-medida`; commit `34a356f`): `caixa_b.z` no hold
    p50 1,025 (p10 0,913; p90 1,076), n = 44 988 → `altura_carregar` 0,95 → 1,02 e `peito_b.z`
    0,15 → 0,222. `dq` de pé com a caixa p90 0,69 → `de_pe_tol_rad` 0,35 → 0,69 (`b2cba3e`).
- **Por que terminou:** fim de bloco (META). Notebook `c1801ea`: "META=9000 sobre a 6999 da bloco10".

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
