---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
---
Ver a caixa e o ambiente.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
