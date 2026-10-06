---
tipo: frente
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
---
mjlab, Warp, Kaggle/Colab, gargalo de banda.

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
