---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1-box-vision: continuidade da orientação entre quadros — HMM sobre as 24 respostas é o melhor (70% só transição, 85% com ORB/SIFT, 78% com histograma); rejeição de 45° é a pior peça (medido 2026-09-28)"
---
Medido 2026-09-28 numa sonda de tombos por roteiro, caixa decorada: associação de hoje 45% dos quadros com nomes certos; HMM só transição 70% (0,6 ms); HMM + NCC de histograma 78% (1 ms); HMM + ORB/SIFT 85% (11 a 17 ms). NCC de pixels REJEITADO (exige recorte exato). No papelão liso todos ficam em 79% a 82%; com decisão rígida a aparência piorava. A rejeição de 45° é a pior peça: o buffer envelhece.

**Why:** o user não quer outra CNN no robô; a continuidade tem de sair de método clássico.
**How to apply:** primeiro passo é o HMM só com transição no `ao_vivo.py`; depois somar histograma ou ORB. Compare métodos só dentro da mesma execução (skin sem semente).
**Detalhe:** `g1-box-vision/docs/relatorios/2026-09-28-continuidade-por-aparencia.md`

Relacionado: box-vision-skin-procedural box-vision-memoria-das-faces
