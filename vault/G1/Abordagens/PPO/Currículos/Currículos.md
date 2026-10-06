---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
---
Um modelo para várias tarefas, com currículo.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
