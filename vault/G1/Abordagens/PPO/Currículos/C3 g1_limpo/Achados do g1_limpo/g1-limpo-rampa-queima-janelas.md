---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "A rampa de fatia do g1_limpo avança 22x mais devagar que o desenho: ela queima as janelas puladas quando o sinal fica na banda morta"
cluster: c3
---
O controlador de forma do `g1_limpo/curriculo.py` avança a rampa **uma vez por travessia do
limiar**, não uma vez por `iters_entre_degraus`. Medido 2026-08-27: `Curriculum/forma` foi de
0,8886 (it 1000) para 0,8483 (it 1525) — **2 degraus em 525 iterações**, contra 44 no
desenho.

Duas coisas se somam. Existe banda morta entre `histerese × limiar` (0,40) e `limiar` (0,50),
onde nenhum ramo do `if` age — isso é histerese correta. E `st["ultimo_degrau"] = janela`
crava o contador na janela atual, portanto toda janela que passou durante a banda morta é
perdida. O `iters_balanco` conta iterações de RELÓGIO, não de portão aberto.

**Why:** com a `razao_marcha` oscilando em torno de 0,50 a rampa quase não anda. Do 0,8483
até o piso de 0,30 faltariam ~7000 iterações, e o bloco acaba na 5000 — logo a manipulação
NÃO recebe a fatia dela. Isto reinterpreta `Metrics/alvo_caixa/sucesso = 0,0000`: não é
falha de aprendizado, é falta de dado.

**How to apply:** o conserto é um contador próprio que só avança quando
`razao >= limiar_portao`, usado no lugar de `iters_balanco`. NÃO trocar por
`ultimo_degrau += 1`: o termo roda ~24× por iteração e isso desceria 24 degraus por
iteração. O comentário `# => >= 396 iterações de rampa` no knob engana — é limite inferior.

Relacionado: g1-limpo-bloco1-marcha-formou, g1-limpo-desenho,
g1-mjlab-mutacao-em-treino
