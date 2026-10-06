---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o termo `forma_postural` — incentivo de forma contra a referência da IK, seis grandezas, média simples; a média simples dilui o quadril a 6% da renda do BOTAR e a torção da perna piora a cada bloco"
---
`recompensas.FormaPostural` paga `média_i(1 − min(erro_i/escala_i, 1)) × alcancar` sobre SEIS grandezas: pelve, tronco, separação dos pés, pad, sola e `|hip_yaw|`. Referência `ik/ref_botar.npz` pela altura do alvo; `PesoPorEstado` 1 na pega, 2 no BOTAR.

⚠ A MÉDIA SIMPLES É O BURACO. MEDIDO: endireitar os quadris rende 6% da renda do BOTAR. O robô não corrige por 6%.

⚠ O BRAÇO NÃO ESTÁ NA CONTA. MEDIDO 2026-09-18 (`21700_botar_resume.csv`): no BOTAR o robô copia pelve e tronco da referência e mantém ombro→palma em 0,25 m; a referência tem 0,41–0,46 m nas 15 poses. PEGAR usa o mesmo braço dobrado. Na laje baixa a caixa fica alta. O `qpos` do npz já traz o cotovelo.

**Why:** grandeza fora da média não tem gradiente, por mais alto que seja o peso do termo.
**How to apply:** pesar as grandezas vence subir o peso; o braço entra como grandeza lida do `qpos`.

Relacionado: g1-limpo-ik-referencia-botar, g1-limpo-deriva-de-rumo-com-a-caixa
