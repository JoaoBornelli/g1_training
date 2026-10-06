---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 09/09/26
resumo: "Soltar fora do alvo termina: o arremesso some e o robô segura a caixa até o time_out; fechar não ganha nada."
cluster: c3-bloco
---
- **Datas:** 09/09 (notebook `17ae885`).
- **O que mudou antes:** v3.2 (`88fe9b5`, spec `soltar-termina`): soltar a caixa fora do alvo é
  terminação (`soltou_fora`: velocidade relativa > `v_solta` fora de `raio_solta = precise_pos_sigma`).
  `v_solta = 1,2 m/s` medido: caixa na mão p90 0,85; queda livre do peito 0,71/0,85/1,31 nos passos
  1/3/5. `_alcancar = 0` na cauda. `caixa_dist_max` sai. 28 termos, 3 terminações.
- **Medido (spec `g1-limpo-mao-no-alcancar.md` §0; commit `206070b`):**

  | | it 7150 | it 7499 |
  |---|---|---|
  | `caixa_largada` | 54,6% | 29,4% |
  | `time_out` | 32,6% | 66,4% |
  | vida média (passos) | 537,9 | 824,3 |
  | `load` | 0,0000 | 0,0001 |
  | `s_C` | 0,0005 | 0,0000 |
  | Mean reward | 48,05 | 152,60 |

  O robô aprendeu a **segurar a caixa perto do alvo até o fim do episódio**. Segurar no BOTAR rende
  3,99/s sem mão (`staged` 2,93 + `precise_pos` 0,82 + `precise_ori` 0,24); fechar congela a mesma
  soma, portanto o fecho ganha ~zero e arrisca morte.
- **Por que terminou:** fim da META; troca para v3.3. O `model_7500.pt` não estava na máquina; a
  bloco13 retomou da 6999 (spec `mao-no-alcancar` §3, commit `1dbfa06`).
- **Lição:** a v3.2 removeu o arremesso e não criou o apoiar; `caixa_largada` só pega `v_rel > 1,2`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
