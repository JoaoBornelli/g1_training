---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Ficar parado rendia 145 contra 102 de explorar no PEGAR (rastreio pagava comando nulo); rastreio por elo destravou a pega. VALIDADO."
---
# Piso da estátua no elo de manipulação

Memória: `~/.claude/memory/g1-limpo-piso-da-estatua-travava-a-pega.md`.

## 31/08: o defeito e os dois consertos

Commits `369af74` e `0b7d925`. Os dois defeitos eram de renda, e não de gradiente.

1. **O piso da estátua.** No elo `PEGAR` o twist é forçado a zero, portanto ficar imóvel
   era resposta perfeita para a locomoção, e os dois `track_*` pagavam 4,0/s por rastrear
   nada. Piso medido: PEGAR 8,265/s contra ANDAR 3,863/s. Com episódio de 17,6 s e 60% de
   morte na mesa, ficar parado rendia 145 contra 102 de explorar. O `play` confirmou: na
   ação média o robô ficava na pose default. Conserto: `rastreio_por_elo` zera os dois nos
   elos que não andam. O piso caiu para ~4,5/s.
2. **O portão diluído.** Ver a memória `g1-limpo-portao-diluido-pela-fatia`.

**Resultado, 577 iterações depois do resume:** `forma` 0,7277 → 0,1773 (a rampa chegou ao
piso). Por env de manipulação, `palmas_em_contato` 3,7×, `staged` 3,2×, e o `squeeze` saiu
de zero. `caixa_largada` subiu 24×, e ela é armada só pela primeira preensão: o robô pega e
perde. A locomoção segurou: eficiência des-diluída 0,803 → 0,758, quedas em 0,28%.

**Gargalo novo:** `contato_tronco` = 56% das terminações, e `upright` 0,83 → 0,56. Sem renda
por ficar parado, a exploração vai para frente e para baixo.

## 23/09: o `forma_postural` paga a estátua

Medido na CPU, 32 envs, robô travado (`inspecao=True`), depois da janela de espera:

| estátua | total | termos acima de 0,005/s |
|---|---|---|
| PEGAR | 4,863/s | staged 1,504 · upright 1,0 · pose 1,0 · `forma_postural` 0,814 · precise_ori 0,367 · precise_pos 0,179 |
| ANDAR | 4,416/s | track_ang 1,596 · upright 1,0 · pose 1,0 · track_lin 0,823 |

- O `forma_postural` (16/09, `46dea30`) paga 0,81/s a um robô parado na pose default.
- O sorteio do comando move a estátua do ANDAR entre 4,24 e 4,42/s. Razão PEGAR/ANDAR: 1,10
  a 1,15×, contra os 2,1× de 31/08.
- O `velocidade_por_regime` custa menos de 0,005/s nas duas: o freio não entra no piso.
- A checagem do smoke passou de `parado < anda + 0,5` para `+ 1,0`. Com 1,0, o defeito de
  31/08 ainda reprova.
- Em aberto: o `forma_postural` pagar só o GANHO sobre a pose da abertura, e nada pela pose
  default. É mudança de recompensa, e pede um plano.
