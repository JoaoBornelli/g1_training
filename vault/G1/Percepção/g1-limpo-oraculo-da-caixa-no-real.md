---
tipo: decisao
pai: "[[Percepção]]"
estado: ativa
verificado: 2026-10-06
resumo: "o ator recebe a caixa exata (oráculo); DECISÃO 02/10: a v3 treina com estimador simulado (forma desde o começo, erro em quase tudo crescendo no refino); caixa no real por câmera RGB sem marcador + depth"
cluster: percepcao
---
O ator recebe a pose exata da caixa em todo passo fora do ANDAR (`observacoes.caixa_no_frame_da_base`, 10 canais, sem ruído nem atraso). No real a câmera da cabeça perde a caixa na aproximação, na pega (mãos) e no carregar (peito). Fonte real (dono): 2 CNNs leves fundidas com depth ou lidar; o treino mantém os canais e PIORA o sinal. Caminho anotado: estimador puro (câmera → odometria → cinemática das mãos com `pegou`), visibilidade simulada no treino, canal de idade, crítico com a verdade.

**Why:** política treinada com a caixa sempre visível recebe no real leitura que congela ou pula.
**How to apply:** DECISÃO DO DONO 02/10 (substitui a de 24/09): estimador simulado na v3; a forma desde o começo, o erro em praticamente tudo crescendo no refino. Caixa no real: câmera RGB sem marcador + depth.
**Detalhe:** `g1_training/specs/g1-v3-comportamento.md` §13; `docs/planos/2026-09-24-grade-de-ocupacao-lidar.md` §9

Relacionado: g1-limpo-lidar-sim-desenho, g1-camera-profundidade-d435i, g1-limpo-caminho-para-o-robo-real
