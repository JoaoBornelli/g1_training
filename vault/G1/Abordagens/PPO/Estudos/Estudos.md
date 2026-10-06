---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
---
Estudos em paralelo ao treino.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
