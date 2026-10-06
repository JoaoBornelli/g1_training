---
tipo: pendencia
pai: "[[Pendências do C3]]"
estado: aberta
verificado: 2026-10-06
periodo: 05/10/26
resumo: "hip_yaw de 22–30° no BOTAR, abaixo da rampa de 40°, então nada cobra."
cluster: c3
---
hip_yaw de 22–30° no BOTAR, abaixo da rampa de 40°, então nada cobra.

REL-APR §3; ver g1-limpo-botar-veredito-do-dono-21700.

**Medido:** `hip_yaw` E/D de −0,69 a −0,91 / −0,44 a −0,54 rad no `model_21700`; de −0,38 a −0,76 / −0,20 a −0,37 no `model_21850` (refino v2 melhorou, não resolveu). No `model_20500` o `hip_yaw` chegava a 151° (curso 316°), e a rampa a partir de 40° foi paga 900 its sem mover a pose.

**Por que nada cobra:** 22–30° fica abaixo do início da rampa do `limite_de_junta` (40°); a `forma_postural` não tem o quadril com peso suficiente.

**Próximo passo:** na v3, os alvos macro de IK incluem a pose do quadril; medir o `hip_yaw` no BOTAR do primeiro checkpoint.
