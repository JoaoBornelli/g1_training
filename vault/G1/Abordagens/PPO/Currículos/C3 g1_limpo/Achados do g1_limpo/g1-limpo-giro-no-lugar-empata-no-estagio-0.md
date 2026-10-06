---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Comando baixo não é respeitado: parado, o robô colhe exp(−c²/σ²) do rastreio (96% a 0,1 com σ 0,5; 98% com o piso 0,707 do g1_limpo); todos os modelos giram estranho ou pedem comando mais alto; v3 normaliza o rastreio pelo comando"
---
Relato do dono (05/10): todos os modelos treinados giram de forma estranha, ou só executam com comando mais alto que o necessário. Causa: com σ fixo, a estátua colhe `exp(−c²/σ²)` do rastreio: 96% a 0,1 e 78% a 0,25 com o σ 0,5 da Unitree; com o piso 0,707 do `g1_limpo`, 0,61–0,92 do termo no estágio 0 (verificado 01/10). A marcha por fase da Unitree paga marchar no lugar acima de 0,1.

DECISÃO DO DONO (05/10): v3 com rastreio normalizado pelo comando (parado 0, perfeito 1), sobre a velocidade média de 0,8 s; marcha por fase × rastreio; comando baixo como subtarefa.

**Why:** enunciado §3 "Sem estátua"; o balanceador sozinho não cria gradiente.
**How to apply:** desvio registrado da receita pura; a Parte 0 mede ganho contra custo de marcha por comando.
**Detalhe:** g1_training/specs/g1-v3.md §19, conflito 5

Relacionado: g1-v3-locomocao-unitree g1-limpo-giro-canal-e-envelope g1-track-lin-parado-colhe-27pct
