---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
---
Uma tarefa por política.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
