---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Quem escora na mesa no g1_limpo e o que a particao das tres terminacoes mediu"
cluster: c3
---
A particao da terminacao da mesa em tres (commit `ac93fce`) respondeu. Medido no bloco 6,
iteracao 493 com 8192 envs (988 equivalentes de 4096), com `fell_over` em 1% das
terminacoes — portanto contato de mesa NAO e mais queda.

    contato_tronco   94,7% dos contatos com a mesa
    contato_palma     5,3%
    contato_dorso      0,0%   exato

O tronco na mesa e ~50% dos episodios de manipulacao. As duas suspeitas anteriores
MEDIRAM FALSAS: por o pad da palma na lista nao estreitou o corredor de aproximacao
(5,3%), e o freio geometrico do dorso funciona (zero exato).

⚠ `dorso_em_contato` e o pad de dorso contra a CAIXA. `contato_dorso` e contra a MESA.
Sao sensores diferentes. Eu li a razao 41% do primeiro como falha do segundo.

**Why:** a terminacao ja cobra o preco maximo (−4,0 mais o episodio) e o robo continua
escorando. Isso pesa para "reach falhado", nao para "escoro orcado".

**How to apply:** o discriminador que falta e a ALTURA DA PELVE no passo da terminacao.
Pelve alta = tronco veio por escolha. Pelve baixa = agachou para alcancar e caiu.

Relacionado: g1-limpo-cadeia-da-pega, g1-limpo-mesa-contato-zero
