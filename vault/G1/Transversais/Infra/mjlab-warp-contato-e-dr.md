---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "VERIFICADO 05/10 (mujoco_warp 3.10.0.3): noslip NÃO existe no Warp (NotImplementedError); geom_priority é estático (sem dim. de mundo) mas decide atrito, condim e solref do par; geom_friction tem DR pronto (dr.geom_friction, coluna 0); geom_solref/solimp por env só com evento próprio; impratio 2 só no notebook, knob diz 1,0"
cluster: transversais
---
VERIFICADO 05/10 por agente, somente leitura. `noslip_iterations > 0` dispara `NotImplementedError` em `mujoco_warp/_src/io.py:350`. `geom_priority` tem shape `(ngeom,)`, sem dimensão de mundo: não randomiza por env, mas o geom de prioridade maior fornece atrito, condim, solref e solimp ao par (`collision_core.py:356-373`); em empate, atrito = max. `dr.geom_friction` existe (`mjlab/envs/mdp/dr/geom.py:117`, coluna 0 por padrão). Não existe DR de `geom_solref`/`geom_solimp`; os campos têm dimensão de mundo e aceitam evento próprio com `@requires_model_fields("geom_solref")`, como o `carga_caixa` faz com a massa. O repo treina com impratio 2 só pela célula 8 do notebook; `knobs.py:105` diz 1,0 e o smoke assegura 1,0.

**Why:** a DR de atrito da caixa funciona com `priority` fixa na caixa + `dr.geom_friction` por env; o creep só se reduz por impratio ou solref, nunca por noslip.
**How to apply:** prioridade na cena (`CollisionCfg.priority`), μ por env pelo termo pronto; solref por env pede evento novo.

Relacionado: g1-limpo-caixa-presa-pelo-pulso g1-mjlab-api-armadilhas g1-limpo-cone-eliptico-diverge-para-nan
