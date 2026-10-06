---
tipo: frente
pai: "[[PPO]]"
estado: superada
superada_por: Currículos
verificado: 2026-10-06
periodo: 15/07–24/07/26
resumo: "Uma política por skill (Stand, Lift Box). Funcionou isolado, mas não encadeia: esqueceu o Stand e não anda. Superada pelos currículos."
cluster: tarefas-isoladas
---
Uma política por skill, no módulo `g1_training/` (pasta `skills/`: `stand`, `lift`, `place`). Cada skill tem `knobs.py` versionado e configs datados em `configs/`, com warm-start de uma para a outra. `place` ficou só como stub (fase 1 do comando reservada).

A locomoção nunca foi treinada como tarefa isolada. Ela entrou já dentro dos currículos (C1 multitask, C2 g1_poc).

## Veredito

O Lift Box funcionou como skill isolada: pegou, ergueu e sustentou a caixa até prateleiras baixas. Mas skills separadas não encadeiam: o fine-tune do Lift fez o robô ESQUECER o Stand (esquecimento catastrófico), e andar não emerge de "alvo longe" como agachar emergiu. Isso motivou um modelo só, com currículo.

## Filhos
<!-- filhos:inicio -->
- **Lift Box** (abordagem · validada · 15–24/07/26) — Pegar e sustentar a caixa parado, agachar emergente. Validado até prateleira baixa e 5 kg; superado porque não encadeia.
- **Stand** (abordagem · validada · 15–16/07/26) — Ficar de pé e recuperar de empurrão (Stand e Stand-Step); fundação para o Lift.
<!-- filhos:fim -->
