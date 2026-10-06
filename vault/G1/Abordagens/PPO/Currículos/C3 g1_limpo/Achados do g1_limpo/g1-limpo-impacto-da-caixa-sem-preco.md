---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o impacto da caixa nunca foi cobrado; MEDIDO 01/10: a métrica impacto_da_caixa (pico de força por passo de controle) é ALIASADA e não diz a altura da queda; v3 mede a velocidade de chegada (limiar 0,99 m/s = queda de 5 cm)"
cluster: c3
---
`metricas.impacto_da_caixa` publica o pico de `|F_apoio_z| / (m·g)` por episódio e nunca foi cobrada; ficou em 6–9 por 25 mil its.

MEDIDO 01/10 (teste de queda na cena, MuJoCo clássico): o choque dura menos que um passo de controle (20 ms), e a força lida uma vez por passo pega o pico ou não, conforme a fase. Queda de 5 cm lê 2,5; de 8 cm, 9,0; de 10 cm, 5,5. O pico real por subpasso é 11,8, 16,1 e 17,8. Não depende da massa nem do tamanho.

**Why:** a regra do dono é "soltar de 5 cm pode, de mais alto não"; a força aliasada não separa os dois.
**How to apply:** na v3, medir a velocidade vertical da caixa no passo antes do primeiro contato; limiar 0,99 m/s.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md (item 20)

Relacionado: g1-limpo-caixa-esmagada-squeeze-cego g1-limpo-v3-alvos-macro
