# G1 — Home

Raiz da árvore: [[G1]]. Anotação solta: [[inbox]].

## Pendências abertas
```dataview
TABLE pai, verificado FROM "" WHERE tipo = "pendencia" AND estado = "aberta" SORT verificado DESC
```

## Abordagens ativas e em desenho
```dataview
TABLE pai, estado, verificado FROM "" WHERE tipo = "abordagem" AND (estado = "ativa" OR estado = "em-desenho")
```

## Achados para verificar (mais de 3 semanas)
```dataview
TABLE pai, verificado FROM "" WHERE tipo = "achado" AND estado != "superada" AND date(verificado) < date(today) - dur(21 days) SORT verificado
```
