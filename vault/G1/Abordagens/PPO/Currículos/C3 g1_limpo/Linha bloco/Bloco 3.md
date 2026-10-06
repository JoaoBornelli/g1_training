---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 28/08/26
resumo: "Do zero com terminação de corpo inteiro: ~75% da manipulação morre na mesa; a auditoria achou três anti-hacks de geometria perdidos na reescrita."
cluster: c3-bloco
---
- **Datas:** 28/08; medido na it 4251 em 28/08 [vault:g1-limpo-mesa-contato-zero; 917bf38].
- **O que mudou:** treino do zero com `contato_ilegal` de corpo inteiro (50 N) [vault:g1-limpo-mesa-contato-zero].
- **Hipótese:** do zero, escorar termina desde a primeira tentativa e nunca vira hábito.
- **Medido (it 4251):** `staged` parado no valor de nascimento e `squeeze` 0,0002 depois de 3200 iterações com a fatia aberta [917bf38]. Fatia de manipulação 24,7%, `contato_ilegal` 18,5% das terminações → ~75% dos episódios de manipulação morrem na mesa [917bf38; vault:g1-limpo-mesa-contato-zero]. O gradiente de aproximação valia +2,6/s, mas terminar custava −4,0 mais ~17 s de renda: esticar o braço tinha valor esperado negativo [vault:g1-limpo-mesa-contato-zero].
- **Por que terminou:** auditoria contra o g1_poc achou três anti-hacks perdidos na reescrita, todos geometria [917bf38]:
  1. alcance era `min` das palmas contra uma esfera no centro → agora cada palma mira a SUA face lateral e o kernel usa a MÉDIA;
  2. lista da mesa INVERTIDA: `.*_collision` cobria punho e cotovelo e NÃO cobria os pads (`add_pads_de_palma` apaga `*_hand_collision`); aproximar terminava, escorar com a palma era grátis → pads entram, punho e cotovelo saem;
  3. `precise_ori` inerte (direção viva que se movia com o robô) → só o REORIENTAR pede direção viva; os outros congelam a normal da abertura.
  Mais: `squeeze` projeta a força na normal da palma; `F_ref = m·g/(2μ) = 6,13 N` (era 12,0 sem derivação); `unload` ganha porteiro de preensão (derrubar a caixa pagava 0,0995 com `squeeze` zero); terminação `caixa_largada` armada pela primeira preensão; métricas `palmas_em_contato` e `dorso_em_contato`. Smoke 348 ok [917bf38].
- **Lição:** o gradiente existia em cada termo isolado; faltava a LIGAÇÃO entre eles. Ao ler pega travada, comparar a cadeia termo a termo com o g1_poc antes de mexer em peso [mem: g1-limpo-cadeia-da-pega].

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
