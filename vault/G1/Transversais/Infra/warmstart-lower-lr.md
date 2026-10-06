---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "Ao warm-startar um treino RL num degrau/distribuição mais difícil, BAIXAR a learning_rate inicial (5e-4 vs 1e-3) pra não destruir o checkpoint nos primeiros updates"
cluster: transversais
---
# warmstart-lower-lr (original, com evidência do g1 lift)

Corpo original da memória `warmstart-lower-lr`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

**REGRA: ao warm-startar um treino (RL/PPO) numa distribuição NOVA/mais difícil**
(novo degrau de currículo — altura adicionada, payload, etc.), **baixar a
`learning_rate` inicial** de `1e-3` (default) pra **`5e-4`** (ou `1e-4/2e-4` se ainda
mergulhar):

```python
cfg.agent.algorithm.learning_rate = 5e-4   # schedule="adaptive" segue ligado;
                                            # é só o valor INICIAL, o KL-adaptive sobe depois
```

**Why:** a value function do checkpoint está "velha" pra a nova distribuição →
advantages grandes → os **primeiros updates dão passos destrutivos** e **quebram a
skill/equilíbrio** que o warm-start trouxe. LR baixa = passos gentis no começo (o
momento crítico); o `desired_kl` adaptive sobe a LR de volta sozinho depois.

**Evidência (g1 lift, 2026-07-18, adicionar altura 0.35 ao run de payload):**
- LR **1e-3**: colapso catastrófico — `fell_over` 84, equilíbrio zerado, ~1000 iters
  só pra reconstruir o ficar-de-pé, grasp rastejando, **não fechou** a task.
- LR **5e-4**: dip raso — `fell_over` 19→7, e aos ~100 iters já tinha
  `grasp/lift/sustain > 0` (re-adaptando, não reaprendendo do zero).

**Sinal de que funcionou (1º print):** `fell_over` NÃO dispara pra dezenas (fica baixo
desde o começo) e o `grasp` começa ALTO (herdado, ~0.3+) em vez de zerar. Se ainda
mergulhar fundo, baixa mais a LR.

**Combina com:** `seed_newest_high=False` no PLR (a altura nova não toma ~48% da massa
de cara → menos gradiente dominado por falha). Ver g1-lift-box-task.
