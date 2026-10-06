---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
---
Macro das abordagens de controle do G1: BFM-Zero, PPO, controle determinístico.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
