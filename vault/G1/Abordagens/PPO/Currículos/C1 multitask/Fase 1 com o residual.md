---
tipo: abordagem
pai: "[[C1 multitask]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 30/07–03/08/26
---
## Hipótese
O env multitask com 7 tarefas serve de base para o Residual PPO sobre o BFM e para a política monolítica.

## Método
Mesmo env para o `g1_residual` (ver o cluster BFM-Zero) e para a monolítica. 30/07–31/07: dezenas de consertos de cena e de critério — prateleira afastada de quem não a usa (para CIMA, idempotente), caixa sobre a prateleira pelo mesmo delta, `base_com` desligado (`dr.body_com_offset` corrompe CPU e GPU), EMA do currículo começando em 0,0. Critérios: sentado conta como queda, `pegar` exige preensão, `parado` exige DE PÉ, `reorientar` mede a rotação da CAIXA, pré-gatilho não fecha sucesso. 03/08: as recompensas param de punir o agachamento.

## Resultado
A monolítica aprendeu a ficar de pé em ~250 iterações explorando ±18°. Vale das runs de 30/07: monolítica 57 → 11 → 164 → 851 passos de episódio.

## Por que parou
Tirar o `fell_over` abriu o hack de SENTAR (sucesso 0,955 com o robô no chão) e foi revertido. O residual parou em 03/08.

## O que herda
- EMA iniciada em 0,5 virava o pico antes de qualquer medida e congelava a célula na 8ª atualização sem regressão real.
- `posture_manip` cobrava 0,5 de quem agacha (12% do orçamento): termo que zera fora da pose padrão pune a tarefa.
- `com_balance` −3,0 cancelava o orçamento inteiro numa excursão de 1 m: alcançar o chão joga o CoM à frente por definição.
- Critério do `parado`: `de pé E ‖v_xy‖ < 0,30 m/s` em 80% dos passos — por FRAÇÃO, senão andar 20 s e parar no último passo aprovaria.
