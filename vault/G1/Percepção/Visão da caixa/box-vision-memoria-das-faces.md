---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1-box-vision etapa 3: vetor de 6 faces com 4 estados; ver a tag encerra a busca, dedução vira provável"
---
Vetor de 6 faces: `-1` não vi, `0` vi sem tag, `1` tag, `2` provável. Marca com
ângulo abaixo de 65° (era 60°). Cinco `0` deduzem a sexta como `2`. Ver a tag
encerra a busca.

**O bloco difícil é a ASSOCIAÇÃO, não a contagem.** As 24 simetrias do cubo dão a
mesma imagem, então a rede troca de representante e renomeia as faces em
silêncio. Não conserte na rede — representante canônico gera alvo descontínuo.

**Why:** O `2` e o alinhamento fecham a mesma falha: conclusão errada sem aviso.

**How to apply:** Alinhe contra a ESTIMATIVA ACUMULADA (moda/mediana do buffer),
nunca um quadro só. Feito para qual das 24 usar e para a rejeição; rejeição caiu
de 9,8% para 0% em 500 quadros. Não compense frame-rate baixo simulando física
por tempo real — TESTADO E PIOROU.

**Detalhe:** `g1-box-vision/ESTUDO-MEMORIA-DAS-FACES.md`

Relacionado: box-vision-desenho-face-da-tag
