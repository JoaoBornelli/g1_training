# Plano — o `limite_de_pelve` no CARREGAR

Data: 2026-09-15. Branch: `exp/g1-limpo-alvo-e-inercia`. Checkpoint medido: `model_15200`
do `bloco19`. Nenhum código foi alterado ainda.

O robô pega a caixa de pé e **anda agachado com ela**. Este plano acrescenta UM termo, um
preço, gateado num estado só. Não remove nada.

## 1. O defeito, medido

Medições em `carregar_15200_055.csv` (950 passos), `botar_15200_ckp.csv`,
`botar_15200_025.csv` e `botar_15200_055.csv`.

Altura da pelve por fase (cinemática direta, pés planos, pelve vertical):

| fase | pelve z | desvio | joelho | hip_pitch |
|---|---|---|---|---|
| espera | 0,747 | 0,064 | +0,633 | −0,404 |
| pegar | 0,748 | 0,083 | +0,440 | −0,380 |
| **carregar** | **0,564** | **0,012** | **+1,718** | **−1,016** |
| andar SEM caixa | 0,726 | 0,015 | +1,077 | −0,515 |

Três fatos que a tabela entrega:

- Ele **pega de pé** (0,748) e **cai 18 cm** ao carregar.
- A marcha vazia a 0,726 tem o **mesmo desvio** (1,3 cm) da agachada. Andar em pé já está
  no repertório; o agachamento não é exigência da marcha.
- O agachamento é **estável**, não oscilação: desvio de 1,2 cm em 500 passos.

## 2. O alvo não explica

O alvo do CARREGAR é absoluto em z (`altura_carregar`, 0,75 a 1,0). Medido, a caixa fica
no lugar certo **mesmo agachado**:

```
palma z = 0,871 m        contra altura_carregar = 0,85
```

Ele abaixa o corpo e mantém a caixa erguendo o braço. O `precise_pos` é pago agachado.

E andar não sacode a caixa: desvio da palma p50 = 0,0078 m e p90 = 0,0161 m, contra σ fixo
de 0,18 m. **Perda do `precise_pos` no p90: 0,8%.**

## 3. A causa, no código

`knobs.py:1143`, coluna 7 (CARREGAR) da tabela por estado:

```
postura_ereta: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 8.0)
                                                    ^^^ CARREGAR
```

É o termo cujo docstring (`recompensas.py:707-709`) diz:

> É o termo que impede o robô de satisfazer o alvo DESCENDO até a caixa. O alvo já tem z
> absoluto, o que remove o atalho de baixar o alvo; este termo remove o atalho de baixar o
> CORPO.

A justificativa do zero, no docstring da tabela:

> Medido: `unload ≡ 1`, `load ≡ 0` e **`postura_ereta` saturada ali** — três dos seis são
> constantes, apagá-los custa zero gradiente.

**A justificativa caducou.** A rampa é `clamp((z − 0,45)/(0,75 − 0,45), 0, 1)`
(`knobs.py:880`, `pelve_alvo = 0,75`). Com a pelve em 0,564 ela vale **0,379** — no meio,
com derivada cheia. Deixou de ser constante e ninguém releu o número.

No CARREGAR sobram dois termos ativos, e o próprio docstring diz que os dois são satisfeitos
agachado:

| termo | coluna CARREGAR |
|---|---|
| `precise_pos` | 1,0 |
| `track_linear_velocity` / `track_angular_velocity` | 3,5 |
| `staged`, `unload`, `load`, `postura_ereta` | **0,0** |

## 4. Preço, e não renda

Reativar o `postura_ereta` no CARREGAR resolveria o gradiente, mas abre três frentes:

1. **Enriquece a estátua.** Ele soma `2c` à renda de ficar PARADO com a caixa. A tolerância
   a risco de andar — `ganho/(parado + ganho)` — cai de 45% para 38% com `c = 3`. É o
   defeito `não anda com a caixa`, que o rastreio ×3,5 foi calibrado para consertar.
2. **Entra na `renda_congelada`** (`env_cfg.py:94`), portanto sobe o piso da CAUDA, cuja
   coluna 8 foi calibrada com o CARREGAR valendo zero.
3. Obriga a subir o rastreio junto, ~+0,4 de coluna por +1 de `postura_ereta`.

Um preço não faz nenhuma das três. Ficar de pé custa zero, agachar custa. A renda parada
continua 16,8/s, a tolerância continua 45%, e a CAUDA não muda.

A regra 2 do projeto (`g1-principio-incentivo-nao-penalizacao`) não proíbe. A cláusula 4
dela autoriza:

> Penalidade não ensina o que fazer. Ela só limita COMO fazer, e só vale **depois de o
> comportamento existir**.

Andar em pé existe: 0,726 m com 1,3 cm de desvio, andando vazio.

## 5. A forma: dobradiça com QUADRADO, sem teto

```
excesso = relu(h_lim − z_pelve)
custo   = peso × (excesso / d_ref)²
```

**Por que quadrado e não exponencial.** A exponencial exige teto, e teto cria zona morta —
a lição que o `limite_de_junta` já custou duas tabelas reprovadas (95,7% e 59,1% dos passos
no teto, derivada zero). O quadrado cresce sem teto, e o precedente é do próprio código
(`velocidade_por_regime`):

> ⚠ QUADRADO do excesso, e NÃO exponencial: `e^{5,7}` num único passo dominaria o lote. O
> quadrado já cresce sem teto.

A pelve é fisicamente limitada em [0,18; 0,78], portanto o excesso nunca passa de ~0,56 e
nenhum passo domina o lote.

**Por que não linear.** Decisão do dono: um centímetro abaixo da linha é irrelevante, cinco
centímetros já comprometem a pose. O quadrado entrega exatamente essa curva; o linear cobra
igual em 1 cm e em 11 cm.

**Consequência declarada.** O quadrado tem derivada ZERO na linha, portanto o equilíbrio
assenta ABAIXO dela. O `h_lim` fica acima do alvo de propósito.

## 6. O termo

| knob | valor | razão |
|---|---|---|
| `h_lim` | **0,74** | 1 cm acima da marcha vazia medida (0,726 ± 0,013). A quadrática assenta abaixo da linha, então a linha é ambiciosa por desenho |
| `d_ref` | **0,10** | o knob se lê sozinho: 10 cm abaixo da linha custa o peso |
| `peso` | **−1,0** | hoje custa 3,1/s, 18% da renda parada de 16,8/s — abaixo do maior incentivo (rastreio, 14/s). Passa na regra 5 |
| teto | **nenhum** | o quadrado não precisa; ver §5 |
| gate | **só CARREGAR** | ver §7 |

A curva resultante:

| quanto abaixo de 0,74 | pelve | custo | derivada |
|---|---|---|---|
| 1 cm | 0,730 | 0,01/s | 0,02/s por cm |
| 5 cm | 0,690 | 0,25/s | 0,10/s por cm |
| 11 cm | 0,630 | 1,21/s | 0,22/s por cm |
| **17,6 cm (hoje)** | **0,564** | **3,10/s** | **0,35/s por cm** |

A 5 cm o custo empata com o que o tornozelo já paga hoje (0,5/s). A 17,6 cm ele é 6×.

## 7. O gate

**Só o estado CARREGAR.** As outras fases precisam agachar, e o dado prova:

| fase | pelve medida |
|---|---|
| BOTAR, laje 0,05 | **0,181** |
| BOTAR, laje 0,25 | 0,259 |
| BOTAR, laje 0,55 | 0,505 |
| PEGAR | 0,748 |
| CAUDA | já tem `postura_ereta` = 8 na coluna 9 |

Sem o gate, o BOTAR na laje baixa pagaria `1,0 × (0,559/0,10)² = 31/s` e a tarefa morreria.

**Gate por dentro, lendo `env.limpo_estado`** — como o `_fora_do_botar` já faz. NÃO entra na
`PesoPorEstado`. Razão, declarada em `env_cfg.py:770`:

> FORA DA TABELA, de propósito: `upright`, `terminacao`, `contato_*`, `joint_acc`,
> `action_rate_l2`, `velocidade_por_regime` e `renda_congelada`.

Preço fica fora da tabela; renda fica dentro. O `velocidade_por_regime` e o
`limite_de_junta` são os precedentes.

## 8. Pontos de impacto

| arquivo | ponto | mudança |
|---|---|---|
| `knobs.py:1260` | ao lado de `LimiteDeJunta` | dataclass `LimiteDePelve(h_lim, d_ref)` |
| `knobs.py:824` | `Tarefa` | `limite_de_pelve: float = -1.0` |
| `knobs.py:1338` | `Knobs` | o `field(default_factory=...)` |
| `recompensas.py:22` | `__all__` | nome novo |
| `recompensas.py` | função nova | lê `root_link_pos_w[:,2] − env_origins[:,2]`, idêntico ao `postura_ereta` (`:722-723`) |
| `env_cfg.py` | registro | **antes** do `renda_congelada` (`:756`), que o smoke prova ser o último de `cfg.rewards` |
| `metricas.py:100` | métrica nova | ver §9 |
| `smoke.py` | checks novos | ver §10 |
| notebooks Kaggle/Colab | asserts de paridade | precedente: commit `af34cab`, que fez o mesmo pelo `limite_de_junta` |

O que **não** muda, e é o ganho de usar preço em vez de renda:

- `TERMOS_CONGELAVEIS` (`env_cfg.py:94`) — o piso da CAUDA fica intacto.
- A tabela por estado — o `smoke.py:1411` compara `set(_DEZ)` contra dez nomes literais e
  quebraria com um décimo primeiro.
- As colunas de rastreio — não precisam subir junto.

## 9. A métrica é obrigatória

Sem ela o teste é cego. O custo satura em informação: duas poses diferentes abaixo da linha
lêem diferente, mas o número não diz a ALTURA. É a mesma razão pela qual o `fracao_do_curso`
teve de nascer ao lado do `limite_de_junta` (`metricas.py:378`):

> Aquele é o CUSTO, e o custo satura no teto da rampa: duas poses igualmente proibidas lêem
> igual. Esta lê a POSIÇÃO, e continua subindo depois do teto.

Métrica `altura_da_pelve`, com uma decisão pendente sobre o `reduce`:

Resolvido na implementação, com uma limitação aberta (15/09):

- `reduce="min"` **não existe** no `MetricsTermCfg`. Os três valores são `mean`, `last` e `max`.
- O `"mean"` do manager é `soma / step_count` sobre TODOS os passos do episódio, portanto um
  valor gateado por fora sai diluído pela fração de passos em CARREGAR, e não pela altura.
- A rota implementada é acumulador por dentro (soma e contagem só nos passos de CARREGAR)
  com `reduce="last"` — o idioma do `impacto_da_caixa`.
- **LIMITAÇÃO ABERTA:** o manager tira a média sobre os ENVS e não há máscara por env. Um
  env que nunca entra no CARREGAR entra na média com ZERO, portanto o logado vale
  `fração × altura`. O alvo de 0,70 da §11 **não se lê direto** enquanto a fatia se move.

## 10. Checks do smoke

1. `h_lim` está acima da marcha vazia medida: `0,74 > 0,726`.
2. O gate zera fora do CARREGAR: montar `limpo_estado` nos dez estados e provar que só a
   coluna 7 devolve valor.
3. O custo a `z = h_lim` é exatamente 0, e a derivada ali é 0 (a forma é quadrática).
4. O custo a `z = 0,564` é 3,1/s ± 1%, contra a tabela da §6.
5. O termo NÃO está em `TERMOS_CONGELAVEIS`.
6. O termo NÃO é o último de `cfg.rewards` — o `renda_congelada` continua sendo.
7. O termo NÃO é campo de `PesoPorEstado` (o `_DEZ` continua com dez).

## 11. O teste e o contra-teste

**Teste:** `altura_da_pelve` no CARREGAR passa de **0,70**. Hoje está em 0,564 com desvio de
1,2 cm.

**Contra-teste:** `Metrics/twist/eficiencia_min` **não pode cair**. Se cair, ele parou de
andar para ficar de pé, e o preço está grande demais ou o rastreio precisa subir junto.

**Terceiro canal, esperado subir e depois descer:** `Mean value loss`. O `limite_de_junta`
fez ir de 1,3 para 7,4 neste mesmo bloco. É a distribuição de retorno mudando, não defeito.

**Quarto canal, esperado MELHORAR:** `Episode_Reward/limite_de_junta`. Levantar troca custo
de tornozelo por custo de joelho, e o joelho é mais barato:

| junta, p50 de \|frac\| | agachado | em pé | limiar da rampa |
|---|---|---|---|
| `right_ankle_pitch` | **0,899** | 0,295 | 0,85 |
| `left_ankle_pitch` | **0,828** | 0,110 | 0,85 |
| `left_knee` | 0,195 | 0,703 | 0,85 |

Em pé o joelho fica em 0,70, abaixo do limiar, custo zero. Os dois termos concordam.

## 12. Ordem de execução

1. Implementar o termo, a métrica e os sete checks.
2. Rodar o `smoke`. Ele é quem confere a configuração; o dono o roda.
3. **Resume do `model_15200`** com o termo ligado. É a CALIBRAÇÃO, não o ensaio: ele mede
   quanto o agachamento compra — número que hoje só tem piso (0,5/s do tornozelo).
4. Ler o teste e o contra-teste. Ajustar o `peso` se a pelve não passar de 0,70.
5. Levar o peso calibrado para o treino do zero.

O peso que funciona no resume é um **teto** do que o zero precisa: no resume o termo tem de
vencer um hábito entrincheirado; do zero ele só molda.

## 13. O que foi rejeitado, e por quê

| opção | veredito |
|---|---|
| reativar `postura_ereta` no CARREGAR | **levantado e recusado por decisão do dono, 15/09.** Custa três números contra ~60 linhas, e o gradiente já existe (rampa 0,379). Mas enriquece a estátua (tolerância a risco 45% → 38%), entra na `renda_congelada` e obriga a recalibrar a CAUDA e o rastreio. O dono quer o comportamento do CARREGAR governado por um termo próprio, com a forma quadrática, e não por um efeito colateral de um incentivo compartilhado |
| rampa exponencial | exige teto, e o teto ficaria a 2,6 cm de onde a pelve vive — zona morta |
| rampa linear | cobra igual em 1 cm e em 11 cm; decisão do dono é que 1 cm é irrelevante |
| linha na `PesoPorEstado` | preço não vai na tabela (`env_cfg.py:770`), e quebra o `smoke.py:1411` |
| referência de pose por IK no CARREGAR | o defeito está nas PERNAS, e elas estão andando; referência estática não se aplica a marcha. A tabela já declina subir o `pose` ali pela mesma razão: *"NÃO em CARREGAR (é marcha; `std_walking`)"*. A referência por IK fica para o BOTAR, que é quase-estático |
| deixar para um bloco de refinamento depois | paga o vale do reaprendizado. O `limite_de_junta` cobrou `s_C` de 0,10 a **zero absoluto** por ~380 iterações, ~500 até voltar |

## 14. Economia de linhas

A diretriz do projeto é **remover antes de alterar, e alterar antes de acrescentar**. Este
plano acrescenta, por decisão do dono (§13). O que ele ainda deve é acrescentar o MÍNIMO.

Fiação mínima:

| onde | linhas | o quê |
|---|---|---|
| `knobs.Tarefa` | 1 | `limite_de_pelve: float = -1.0` |
| `knobs.Tarefa` | 2 | `pelve_limiar = 0.74`, `pelve_ref = 0.10` — **corrigido 15/09**: em `Recompensa` o `aplica_pesos` (`env_cfg.py:98-115`) itera `asdict` e afirma que cada nome existe em `cfg.rewards`; knob órfão ali explode na montagem. `pelve_alvo`, `pelve_piso` e `tol_pos` já moram em `Tarefa` pelo mesmo motivo |
| `recompensas.py` | ~8 + docstring | a função, com o gate lendo `env.limpo_estado` |
| `env_cfg.py` | 4 | o registro, antes do `renda_congelada` |
| `metricas.py` | ~6 | `altura_da_pelve`, gateada no CARREGAR |
| `smoke.py` | ~15 | os sete checks da §10 |

Três coisas que o plano **não** faz, e cada uma economiza um bloco:

- **Sem dataclass própria.** O `LimiteDeJunta` é dataclass porque tem `k` e teto POR FAMÍLIA,
  em quatorze padrões. Aqui são dois escalares, e dois escalares moram em `Tarefa` como
  os outros da pelve.
- **Sem knob de teto.** A §5 decidiu que a forma quadrática não precisa de teto.
- **Sem linha na `PesoPorEstado`.** O gate é interno, como o `_fora_do_botar`. Preço fica
  fora da tabela (`env_cfg.py:770`), e o `smoke.py:1411` não quebra.

O docstring é a parte mais longa do que entra, e isso é o estilo da casa: ele carrega o
porquê de cada número e a medição que o gerou. Ele não é candidato a corte.
