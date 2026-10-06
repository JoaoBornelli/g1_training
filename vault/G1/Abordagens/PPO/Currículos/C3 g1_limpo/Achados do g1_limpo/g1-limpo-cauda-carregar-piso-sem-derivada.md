---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "60% da renda viva da cauda CARREGAR e piso sem derivada, e a cauda nao tem destino no mundo: o alvo e ancorado no peito"
---
MEDIDO 2026-09-10, `model_10200`, n = 25 852. Renda viva da cauda `CARREGAR`, 13,20/s:

| | /s |
|---|---|
| constantes ESTRUTURAIS | 3,96 |
| saturados | 3,96 |
| com gradiente | **5,28** |

**As constantes nao sao saturacao — sao incapacidade de variar.** `unload ≡ 1,0` porque o comando manda a laje para **+5 m** no `CARREGAR`, logo `F_apoio ≡ 0 N`. `postura_ereta = rampa × unload`, com a rampa saturada (pelve 0,777 > `pelve_alvo` 0,75). `load ≡ 0` fora do `BOTAR`.

Saturados: `precise_pos` (caixa a 1,35 cm do alvo; derivada 0,025/s por cm contra 0,559 do `staged`) e `squeeze`.

⚠⚠ **E a cauda nao tem DESTINO.** O alvo e `_alvo_ancorado_na_base`, no peito: ele anda com o robo. A renda de manipulacao e invariante a posicao no mundo. Nada paga por chegar a lugar nenhum.

**How to apply:** apagar os 3,96/s estruturais custa zero gradiente; os saturados ainda tem derivada.

Relacionado: g1-limpo-nao-anda-com-a-caixa, g1-limpo-renda-congelada-paga-10x-o-deficit, g1-limpo-piso-da-estatua-travava-a-pega
