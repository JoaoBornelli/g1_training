---
tipo: abordagem
pai: "[[C1 multitask]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 10–18/08/26
resumo: "Rampa de altura, unload e sucesso denso: lift cru 0,06 → 0,39, mas ergue 3,0 cm de 3,5 exigidos."
---
## Hipótese
O `pegar` travou 22 mil iterações por seis defeitos estruturais; consertá-los (rampa de altura, ponte de carga, bônus denso) fecha a pega.

## Método
ADR-0001. (1) Eixo `alvo`: rampa em FRAÇÃO da distância repouso → peito (0,2 a 1,0); nível 0 pede 5,5 cm. (2) `unload`: fração do peso que saiu da prateleira (9,70 N apoiada → 0,00 N erguida), gateada em preensão E caixa acima do repouso. (3) `sucesso_denso` 5,0/s fora do orçamento. (4) Critério em piso de z, não esfera 3D. (5) Episódio de 10 s na manipulação. (6) `action_rate` ×0,25 nos braços. Antes: caixa na borda (0,30) e `alvo_peito` com z de mundo em 0,91 m. Reaquecer o `std` para 0,8 quando abre tarefa.

## Resultado
`lift` cru 0,062 → 0,394; `unload` funciona (`pegar/_total` +0,32 líquido). Mas o robô ergue 3,0 cm e o critério pede 3,5: `cond_fisica` 0,0000 exato. Os braços batem no limite de junta ao erguer (`dof_pos_limits` −0,07). A pega é bamba (|ω_caixa| ≈ 1,45 rad/s).

## Por que parou
A fuga do `std` (ver Std foge) zerou todas as tarefas. Fim do módulo em 18/08.

## O que herda
- `_grasp` booleano: tocar paga, apertar paga zero até a caixa mover — platô. O `unload` é a ponte contínua.
- Prêmio distante não atravessa o desconto (`gamma` 0,99 a 50 Hz = 2 s); pagamento contínuo atravessa.
- Sem preensão no critério ele aninha a caixa no vão dos antebraços.
- `squeeze` por força de palma com `min(f_esq, f_dir)`, `F_ref` ≈ 4,9 N, ficou como reserva.
