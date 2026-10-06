---
tipo: frente
pai: "[[Abordagens]]"
estado: parada
verificado: 2026-10-06
---
Usar o BFM-Zero pré-treinado (../BFM-Zero) como base de controle.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
