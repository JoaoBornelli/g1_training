---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "no mjlab uma pose escrita em corpo livre no meio do episodio e re-aplicada a cada passo; o corpo nao cai e a forca de contato vem so da penetracao"
---
No mjlab, `write_root_link_pose_to_sim` num corpo LIVRE, chamada no meio do episódio, **persiste**: a pose volta a ser aplicada a cada passo. Medido 2026-09-03 no `g1_limpo`: a caixa solta 20 cm acima da laje fica 25 passos a 20 cm, e a força de apoio é zero. Em queda livre ela cairia 1,2 m.

Consequência para testes: um corpo pinado não gera força de contato por peso, só por **penetração**. Pinar na altura exata de repouso dá força ZERO. Medido, uniforme nos envs:

```
dz     +0,020   0,000   −0,002   −0,005   −0,020
F/mg    0,00     0,00    0,98     0,94     0,75
```

A força CAI com penetração maior, porque o re-pino limita o impulso do solver.

**Why:** Dois testes do `g1_limpo` mediram força zero e leram isso como defeito de física. O defeito era o método.
**How to apply:** Para medir contato num teste, escreva a pose a CADA passo, 2 mm abaixo da altura de repouso. Uma pose escrita no reset (dentro do `event_manager`) não congela o corpo — ela assenta normalmente.

Relacionado: g1-mjlab-reset-buffers-obsoletos, g1-limpo-botar-paga-nao-botar
