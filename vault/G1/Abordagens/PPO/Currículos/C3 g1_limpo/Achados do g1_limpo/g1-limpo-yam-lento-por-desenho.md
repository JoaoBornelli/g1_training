---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o exemplo YAM do mjlab (levantar cubo) é lento por servo de 2 Hz, freio em currículo e recompensa sem prêmio de tempo; na g1_limpo a pressa é paga pela renda_congelada (6,9 por 0,5 s)"
---
LIDO 23/09 no mjlab 1.5.3 (`tasks/manipulation/lift_cube_env_cfg.py`, `asset_zoo/robots/i2rt_yam/yam_constants.py`):
- Servo PD a 2 Hz no braço e 1 Hz na garra, ζ = 2: constante de tempo ~0,30 s. O G1 usa 10 Hz.
- Dobradiça de velocidade com limite 0,5 rad/s e peso em `reward_curriculum`: −0,01, −0,1 na it 500, −1,0 na 1000.
- Recompensa por kernels gaussianos largos a cada passo; sem bônus de fecho nem término por sucesso.

Na g1_limpo, fechar a pega 0,5 s antes compra ~6,9 de `renda_congelada` (`knobs.py:833`). O freio −15 anula um prêmio por velocidade que o desenho criou.

**Why:** a pressa é incentivo da recompensa, e não falta de freio.
**How to apply:** antes de subir o freio, reduza o prêmio de tempo; use `reward_curriculum` para o peso.

Relacionado: g1-limpo-std-cai-cedo, g1-limpo-freio-nao-taxa-a-marcha, g1-limpo-renda-congelada-paga-10x-o-deficit
