---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "zero12 parou na pega: tira o peso da mesa e não leva ao peito; o tombo é cinemático (eixos coaxiais) e só o punho desfaz; zero14 = unload × _trazer + pose pelo estado"
---
`model_1750` (zero12, nível 0): força 4,9× `F_ref`, subida 8 mm p50, caixa a 0,29 m do alvo. O robô tira o peso da mesa, mas nenhum PEGAR fecha: `unload` e `postura_ereta` saturavam em 2 mm. Da mesa à âncora a mão gira −7° a −45° (eixos coaxiais), e só o punho desfaz.

**Why:** cada correção do tombo encareceu o único erguer que a política sabe.
**How to apply:** zero14 (`unload × _trazer`, `pose` pelo estado) fez erguer, mas a caixa DEITAVA 90°. zero15 (29/09): reta `max(0, 1 − Δθ/90°)` FICA; âncora z 0,75–0,85 REVERTIDA (zero16). Julgar tombo pelo play, não pelo log. Penalidade de tombo ANALISADA 01/10 (zero em 25° dentro do `precise_ori`): cobra −2,4/s a 30° e derruba a margem de erguer sobre ficar na mesa de +4,3 a ~+1,9/s, perto do atrator da zero12; não adotada. `precise_ori` 4 → 0 REJEITADO.
**Detalhe:** `g1_training/docs/relatorios/2026-09-29-pega-parada-na-mesa-cinematica-do-tombo.md`

Relacionado: g1-limpo-escala-da-acao-do-punho g1-limpo-fechar-paga-menos-que-pairar g1-limpo-enunciado-do-comportamento
