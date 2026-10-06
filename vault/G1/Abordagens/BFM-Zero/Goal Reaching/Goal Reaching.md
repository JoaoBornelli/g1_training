---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23/07–06/08/26
---
## Hipótese
Entregar a pose alvo (ângulos de junta) em vez de uma recompensa: `goal_inference` mapeia a pose direto para `z = project_z(B(pose))`, sem buffer, e o BFM cuida do equilíbrio.

## Método
Seis experimentos em sequência, cada um respondendo a falha do anterior: poses à mão (Teste de goal), pose gerada por IK (IK gera pose), viabilidade estática da pose (Viabilidade da pose), mapa comandado → alcançado (Mapa comandado-alcançado), malha fechada com overdrive (Goal servo) e a ferramenta por bloco (Goal track).

## Resultado
O BFM chega perto de poses em pé e de poses de braço. Nas poses profundas ele passa pela pose e não a segura, ou cai. A mão não fica abaixo de ~0,54 m de forma estável.

## Por que parou
`goal_inference` é um forward do encoder B mais normalização: não há otimização rumo ao alvo, não há realimentação, e a magnitude é descartada. A única alavanca é onde se põe o goal, e mover o goal gira o `z` em vez de empurrar mais. Causa de fundo: a pose não está no LaFAN.

## O que herda
- Uma pose de IK são 29 ângulos MAIS o quaternion da base. Passar só os ângulos dá outra pose.
- Antes de culpar o controlador, verificar se a pose é estaticamente viável.
- Medir o erro por bloco (pernas, cintura, braços), nunca a média dos 29.

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
