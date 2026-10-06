---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 24/07/26
resumo: "tracking_inference sobre o rollout da política RL do Lift Box. Rejeitado: o BFM não reproduz o agachamento e a pega."
cluster: bfm
---
## Hipótese
Em vez de um estado alvo (um `z` fixo), entregar uma SEQUÊNCIA de estados: `tracking_inference` dá um `z` por passo, o modo do BFM feito para seguir referência. A referência é um movimento real que funciona: a política RL do Lift Box agacha e pega a caixa sem cair.

## Método
`imitate_rl.py`: rollout da política RL do Lift Box → `tracking_inference` (média de B numa janela `seq_length`) → o BFM segue. Compatibilidade verificada: ordem das 29 juntas do mjlab = `POLICY_JOINT_NAMES`, nq/nv 36/35, dt de controle 0,02 nos dois; conversão 1:1. Tecla G compara com `goal_inference` por frame.

## Resultado
Não funcionou para o Lift Box: o BFM não reproduz o agachamento e a pega da política RL.

## Por que parou
Mesma causa da frente: seguir a referência exige passar e parar em poses que o LaFAN não tem.

## O que herda

## Filhos
<!-- filhos:inicio -->
- **Tracking do 24800** (abordagem · rejeitada · 06/10/26) — O BFM não abaixou até a pega na laje a 0,04 m (relato do dono, 06/10).
<!-- filhos:fim -->
