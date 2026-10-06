---
tipo: frente
pai: "[[PPO]]"
estado: superada
superada_por: Currículos
verificado: 2026-10-06
periodo: 15/07–24/07/26
---
Uma política por skill, no módulo `g1_training/` (pasta `skills/`: `stand`, `lift`, `place`). Cada skill tem `knobs.py` versionado e configs datados em `configs/`, com warm-start de uma para a outra. `place` ficou só como stub (fase 1 do comando reservada).

A locomoção nunca foi treinada como tarefa isolada. Ela entrou já dentro dos currículos (C1 multitask, C2 g1_poc).

## Veredito

O Lift Box funcionou como skill isolada: pegou, ergueu e sustentou a caixa até prateleiras baixas. Mas skills separadas não encadeiam: o fine-tune do Lift fez o robô ESQUECER o Stand (esquecimento catastrófico), e andar não emerge de "alvo longe" como agachar emergiu. Isso motivou um modelo só, com currículo.

## Filhos

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
