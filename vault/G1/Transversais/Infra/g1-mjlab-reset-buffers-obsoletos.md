---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "Duas armadilhas de medição no mjlab — Entity.data.* está obsoleto dentro do reset, e limiar de check abaixo de 5 mm acusa a própria gravidade"
---
`Entity.data.*` está OBSOLETO dentro de um evento de reset e no resample do comando. A escrita vai para a simulação, mas os buffers de `data` só recompõem no forward seguinte. Nenhuma leitura errada levanta erro.

Sintomas medidos: a caixa foi para a mesma coordenada nos três envs sem a origem do env; um clamp leu fundo negativo e nunca aconteceu; o alvo saiu 0,000.

Solução: marque o env como pendente no resample e conclua a parte dependente de pose no primeiro `_update_command`, quando a pose está fresca.

Limiar de check abaixo de 5 mm acusa a própria gravidade. O corpo cai `½·g·dt²` = 2,0 mm entre a escrita da pose e a leitura. Isso me enganou quatro vezes em checagens diferentes.

Outras duas armadilhas do reset, as duas medidas:

- **NaN na observação um passo depois de um reset PARCIAL.** O estado oculto do solver não é limpo, e o cone `elliptic` com `impratio=10` é frágil a esse carryover. `pyramidal` com `impratio=1.0` roda milhares de passos limpo no mesmo harness. Ponha também a terminação `nonfinite`: um mundo com NaN passa a custar um reset, não o treino.
- **`reset_joints_by_offset` do mjlab assume `soft_joint_pos_limits` com shape `(B, nj, 2)`.** Sem nenhum evento de DR de startup o mjlab não expande o campo por mundo, ele fica `(1, nj, 2)`, e indexar com ids acima de zero dá device-side assert. As tasks do fabricante nunca veem isso porque todas têm DR.

**How to apply:** compare elemento a elemento, nunca `max()` de um env contra `min()` de outro — o clamp é por env. Para medir "está de lado", use a componente y no frame do robô, não o desvio-padrão da distância: a âncora é `quat_apply(base_q, peito_b)`, então a projeção horizontal muda quando a base pende.

Relacionado: g1-mjlab-mutacao-em-treino, g1-limpo-desenho
