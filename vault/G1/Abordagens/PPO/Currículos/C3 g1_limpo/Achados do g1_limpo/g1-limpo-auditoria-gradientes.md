---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Auditoria de todos os gradientes do g1_limpo (2026-09-04): 10 buracos, ranqueados; o relatório está em docs/planos"
---
Relatório: `g1_training/docs/planos/2026-09-04-auditoria-gradientes-g1-limpo.md`. Só leitura. Nenhum código mudou.

Regra do dono: em todo instante existe incentivo da posição atual até o alvo do elo, e do fecho até o elo seguinte. Par macro/preciso: mesma forma, macro com raio grande e peso menor, preciso com raio pequeno e peso maior.

Veredito: o `trazer` do `staged` é o único gradiente vivo entre a caixa e o alvo. Dez buracos, por efeito:

- H1 `precise_pos_sigma` 0,05 < `tol_pos` 0,10. Paga 0,018 no limiar.
- H2 `sustentacao` morre após qualquer avanço. `avancou` é pegajoso (`comando.py:1007`, zera só em `:681`).
- H3 fechar o PEGAR custa ≈ −2/s (σ do `trazer` recomputado, `pose` sai do neutro).
- H4 fechar o CARREGAR custa −2 a −6/s.
- H5 sem gradiente de volta à pose após largar (`pose` morto com braços fora).
- H6 a H10 no relatório.

Peso do par caixa→alvo está invertido: macro 3,0, preciso 2,0.

Proposta v2: `docs/planos/2026-09-04-proposta-gradientes-g1-limpo.md`. Um bloco. Filtro de simplicidade: 1 termo entra (`renda_congelada`: no passo de cada fecho congela a soma dos termos dependentes de elo e paga como constante; sem número mágico; decisão do dono 2026-09-04) e 1 sai (`load`); `sustentacao` passa a ler `_sust` do comando (H2 some por remoção); rastreio gateado por `limpo_twist_zerado` (um mecanismo, cobre esperas e segurar-parado); twist fixo no CARREGAR-andando; `caiu` desarmado; REORIENTAR a 5% do sorteio enquanto inerte (decisão do dono 2026-09-04, opção b: o canal do one-hot não pode ficar constante — regra do normalizador, a mesma do `fatia_loco = 0,95`). Verificado NÃO ser buraco: alvo do REORIENTAR é reescrito todo passo.

IMPLEMENTADA em 2026-09-04, três commits na `exp/g1-limpo-v2` (feat 53f81e9, test 6051a60, docs 8f7eca1), sem push. Smoke 524 -> 561 ok. Contrato: `g1_training/specs/g1-limpo-gradientes-v2.md`. Pendências: (1) smoke sem seed — as seções 19 (topo do BOTAR vs fundo lido após assentar) e 28 falham de forma intermitente, antes e depois da v2.1; (2) H5 (volta dos braços após largar) adiado até a espera final ser atingida; (3) run nova (`bloco9`): o assert de pesos da célula de resume dispara por desenho, e a decisão zero × retomar de `model_2562.pt` é do dono.

EFEITO MEDIDO na `bloco9` (warm-start do `model_1800`, LR 5e-4, iteração 2119, +319 do resume). A caixa ANDA para o alvo: `precise_pos` por env de manipulação passou de 0,7% (canal morto, só a janela do REORIENTAR) para 26,3%, o que dá d = 0,208 m contra d₀ ≈ 0,45. O `aproxima_caixa` confirma sozinho: 0,41 por env de manipulação. O gradiente do par no ponto atual é 14,6 por metro, e SOBE para 16,1 no limiar de fecho — antes ele caía. Renda de manipulação 8,1 -> 9,8/s. Queda zerou; `caixa_largada` 16,7% -> 9,8%. O elo ainda não fecha: falta de 21 cm para 10 cm, logo `sucesso`, `sustentacao` e `renda_congelada` seguem em zero — o congelamento está no ar e nunca disparou. A GANGORRA se repetiu na 2300, e é a terceira medição: a locomoção foi ao melhor de sempre (eficiência 0,785, razão 0,663, voo 0,094 s, queda zero) e a manipulação devolveu um terço do ganho (d 0,208 -> 0,221 m). O que cai é SÓ a mão: `alcancar`, `preensao` e palmas em contato caem ~8% juntos, enquanto `descarga` fica em 0,987 e a rampa da pelve fica plana. `self_collisions` DOBROU e o momento angular subiu 7,6%. Hipótese: perturbação mecânica da marcha sobre os braços, e não troca econômica de orçamento de `action_rate`. O discriminador é um `play` ou uma métrica de ação por elo.

**How to apply:** Recompensa é decisão do dono. Antes de mudar um peso, ler a proposta §3 e o smoke `--- v2.1: gradientes`.

Relacionado: g1-limpo-contrato-de-troca-de-tarefa, g1-principio-incentivo-nao-penalizacao, g1-sigma-e-a-distancia-inicial
