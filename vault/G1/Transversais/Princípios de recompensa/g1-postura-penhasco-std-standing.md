---
tipo: achado
pai: "[[Princípios de recompensa]]"
estado: ativa
verificado: 2026-10-06
---
# g1-postura-penhasco-std-standing (original)

Corpo original da memória `g1-postura-penhasco-std-standing`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Medido/derivado em **2026-08-20** no `g1_poc`, lendo o código (não é hipótese de log).

O G1 do mjlab usa `std_standing = {".*": 0.05}` — 0,05 rad para **toda** junta, braços incluídos (`mjlab/tasks/velocity/config/g1/env_cfgs.py:107`). O `postura_manipulacao` do `g1_poc` escolhe entre esse dicionário e o `std_manipulando` por um LIMIAR de demanda residual da caixa:

```
demanda = 10 · ‖caixa − alvo‖ + 6 · Δθ      troca quando demanda < 1,5
```

Com a caixa segurada a 0,82 m, quatro juntas de braço ficam a ≈ 0,7 rad do default:

| regime | erro²/σ² do braço | `pose` (peso 1,0) |
|---|---|---|
| `std_manipulando` (ombro/cotovelo σ = 1,00) | 4 × 0,49/1,00 = 1,96 sobre 29 juntas | **≈ 0,93** |
| `standing` (σ = 0,05) | 4 × 0,49/0,0025 = 784 sobre 29 juntas | **≈ 0** |

**Portanto a troca de regime não é um gradiente que levanta o robô — é um penhasco que cobra ≈ 0,93/s por terminar a tarefa.** A §9.4 da ESPECIFICACAO afirmava o contrário e foi retificada.

E o penhasco explica um número que estava travado. Na it 5217: `erro_posicao = 0,0106` e `erro_angulo_deg = 14,2°` dão `demanda = 0,11 + 1,49 = 1,60`, contra limiar 1,5. A troca exigiria `Δθ < 13,3°`; a régua de sucesso aceita `Δθ < 20°`. **Existe uma banda de 13,3° a 20° em que o elo fecha e o penhasco não é pisado**, e o ângulo medido mora dentro dela.

Verificação sem treinar: `Metrics/postura_frac_manipulando` ≈ 1,0 significa que o robô praticamente nunca entra em `standing`.

**Conserto (não é alinhar a demanda com a régua — isso poria o penhasco EM CIMA do sucesso):**

1. o 4º regime passa a ser escolhido pela FORMA do episódio (`caixa_valida`), não pela demanda residual. `peso_dist`, `peso_ang` e `limiar` morrem.
2. termo novo, contínuo, para a altura da pelve, gateado por preensão bimanual — o idioma que o g1-pegar-plato-do-grasp-booleano validou com o `unload`.

As duas juntas, no mesmo bloco: sem (1) o termo novo luta contra −0,93/s; sem (2) nada levanta o robô.

Relacionado: g1-poc-bloco1-treino, g1-lift-progress-alvo-e-a-caixa.

**MEDIDO 2026-08-26, e corrige o numero acima.** O custo nao e' 0,93/s: o termo vale **exatamente zero**, com **gradiente zero**, ja a 10% da faixa de junta. E nenhum sigma conserta -- `exp(-media(erro^2/sigma^2))` sobre 29 juntas e' um produto de 17 gaussianas quando os bracos saem do default, e colapsa para qualquer sigma:

| fracao da faixa | standing | walking | running | run x3 | run x5 |
|---|---|---|---|---|---|
| 0,10 | 0,000 | 0,014 | 0,184 | 0,829 | 0,935 |
| 0,20 | 0,000 | 0,000 | 0,001 | 0,471 | 0,763 |
| 0,40 | 0,000 | 0,000 | 0,000 | 0,049 | 0,338 |

Dois fatos de calibracao do G1 que explicam a certeza: `std_standing` e' **uma entrada so** (`.*` = 0,05) para as 29 juntas, e o `walking_threshold` do G1 e' **0,05**, nao 0,5 -- logo com o twist forcado a zero o regime `standing` e' CERTO, nao provavel.

Excluir os bracos tambem nao basta: o alcance passa a custar 0,000, mas o **agachamento** com as pernas em `running` da' 0,128 a 10% da faixa e 0,000 a 20%, e o nivel 4+ poe a laje a 0,04 m, o que exige agachar.

**Solucao adotada no g1_limpo (F2):** o termo fica NEUTRO (devolve 1,0) nos elos de manipulacao. Nao 0,0 -- zero seria penalidade por sorteio de elo. Quem segura o robo de pe passa a ser o `upright` do fabricante mais a condicao de fechamento do elo, que exige "de pe". Zero knobs novos. Ver g1-principio-incentivo-nao-penalizacao.
