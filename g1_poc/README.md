# g1_poc

POC de loco-manipulação de caixa do Unitree G1, escrita no idioma do mjlab.

Especificação completa: **`ESPECIFICACAO-g1_poc.md`** na raiz do repositório.

---

## O desenho em uma frase

**Uma tarefa, dois comandos, e o alvo da caixa define o comportamento.**

```
twist(3)        para onde andar.  Zero = ficar parado.
caixa_alvo(10)  onde a caixa deve ficar, e se existe caixa.
```

"Pegar", "carregar", "botar" e "reorientar" **não são tarefas**. São posições de alvo.

| alvo em… | o robô faz | seu item |
|---|---|---|
| `caixa_valida = 0` | ignora a caixa: fica de pé, anda, gira | 1 e 3 |
| a altura de carregar, 0,78–0,85 m em MUNDO | pega e levanta | 5 |
| no peito, frame da base | carrega | 2 e 4 |
| na prateleira | põe | 6 |
| mesma posição, girado | reorienta | 7 |

Portanto: **não existe one-hot de tarefa, não existe orçamento equalizado, e não
existe gate por tarefa.**

---

## O que sai do treino atual

| item de hoje | linhas | aqui |
|---|---:|---|
| `curriculum.py` + `tasks.py` | 786 | ~90 (`curriculo.py`) |
| 43 knobs de recompensa | — | **18 termos**, 13 com pesos do mjlab |
| `T.gated`, `TERMOS_DE_TAREFA`, equalização | — | auto-gate pelo **comando** |
| piso 0,15 + amostragem inversa | — | o **rebaixamento** do nível |
| `entre_blocos.py`, `calibra.py` | 741 | TensorBoard |

---

## Como rodar

### Passo 0 — a geometria, sem GPU. É o portão.

```bash
python -m g1_poc.cena          # compila as entidades e imprime a geometria
python -m g1_poc.play --geometria
```

O repositório perdeu um bloco em 16/07 por um erro de geometria. A lição escrita lá
é: **ataque a geometria, e não o sintoma.** Esta verificação custa minutos.

### Passo 1 — o smoke, na CPU

```bash
python -m g1_poc.smoke
```

Ele confere os contratos e os invariantes. O teste mais importante é o do §7: com
`caixa_valida = 0`, os quatro canais de caixa **e** os quatro termos de tarefa têm de
ser zero. Um vetor zerado dá `exp(0) = 1`; sem a multiplicação pelo bit, "não existe
caixa" pagaria o valor máximo.

### Passo 2 — o primeiro bloco, na GPU

```bash
python -m g1_poc.train --task Mjlab-G1-Poc \
    --env.scene.num-envs 4096 \
    --agent.algorithm.learning-rate 5e-4
```

**Portão de aceite:** `Contrib/squeeze` sai de zero **antes** de
`Contrib/precise_pos`. Depois a taxa de sucesso passa de 0,50, em até 1 000
iterações.

Se o `squeeze` não subir, o problema é a força de palma, e não a subida da caixa.

---

## O termo que decide o resultado

`recompensas.squeeze`. Leia o docstring dele antes de mexer em qualquer peso.

O diagnóstico curto: o `lift` do treino atual paga **+0,34/s por centímetro** de
subida no nível 0. O gradiente existe e é grande. O robô não subiu 1 cm em 22 mil
iterações, porque o gradiente está na **coordenada errada**:

```
d(recompensa)/d(altura da caixa)  = grande
d(recompensa)/d(força de aperto)  = ZERO
```

A política só age em alvos de junta. Antes de a força de atrito vencer o peso, a
caixa não se move e nenhuma recompensa muda. É um degrau, não uma rampa.

O `squeeze` é o único termo com derivada positiva na coordenada do aperto.

---

## Estado dos arquivos

| arquivo | estado |
|---|---|
| `knobs.py`, `cena.py` | completo |
| `observacoes.py`, `recompensas.py`, `postura.py`, `terminacoes.py` | completo |
| `eventos.py` | completo, menos o movimento da prateleira no fecho do `pegar` |
| `comando.py` | **esqueleto**: 1 elo (`pegar`), sem cadeia |
| `curriculo.py` | `sorteia_forma` completo; `nivel_caixa` mínimo, sem células |
| `env_cfg.py`, `smoke.py`, `train.py`, `play.py` | completo |

As 4 cadeias, a troca de elo e a tabela de células entram no **passo 4** da §17 — e
só depois do portão do passo 1. O motivo é o histórico deste repositório: o treino
atual construiu o orquestrador antes de o `pegar` funcionar, e 22 mil iterações
depois o gargalo era a recompensa.

---

## Regras do pacote

1. **Não editar `g1_training/`.** Este pacote só consome.
2. **Nenhum número solto no código.** Tudo em `knobs.py`. Um treino tem de ser
   reproduzível por `git diff` de um arquivo.
3. **Terminar em vez de penalizar.** Uma trajetória inválida acaba.
4. **Uma mudança por bloco.** Warm-start sempre com `learning_rate = 5e-4`.
5. **Se aparecer um hack, volte UM termo** (a §8.5 da especificação), e não seis.
