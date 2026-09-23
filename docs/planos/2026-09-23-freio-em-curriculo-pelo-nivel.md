# Plano — o freio de velocidade sobe com o nível, por env

Data: 2026-09-23. Branch: `exp/g1-limpo-v2`. Runs de referência: `zero05` (freio −15) e
`zero06` (freio −6). Nenhum código foi alterado ainda.

O `velocidade_por_regime` hoje tem UM peso para o treino inteiro. Este plano troca a
agregação do termo pela SOMA e faz o peso subir por env, em degraus, conforme o `nivel`
daquele env bate recordes. A locomoção fica sempre no piso. Nada novo é pago; o termo e o
currículo que já existem são estendidos.

## 1. O problema

**A manipulação é rápida demais para trabalhar perto de pessoas.** Medido:

| medição | o que diz |
|---|---|
| 08/09, `model_4999` | parado com a caixa o robô move as juntas mais rápido que correndo: p90 3,13 contra 2,50 rad/s |
| 10/09, `model_10200`, freio a −2 | só o PEGAR ainda corre: p90 2,32 rad/s; `hip_pitch` 3,52, `knee` 3,17, `shoulder_pitch` 3,01, `wrist` 2,73 |
| 21/09, `model_24999` | a mão chega a 2,58 m/s na pega, num pulso de 0,62 s; a ISO/TS 15066 usa 0,25 m/s |

**A pressa é paga pela cadeia, e a cadeia fica.** Fechar a pega 0,5 s antes compra ~6,9 de
`renda_congelada` (`knobs.py:833`). A renda é o que faz o robô seguir as tarefas em
sequência; ela não muda. O freio só precisa tornar caro o EXCESSO acima do vmax.

**Nenhum peso fixo serve:**

| peso | efeito medido |
|---|---|
| −2 | a pressa na pega ainda compensa |
| −15 (`zero05`) | com o std perto de 1 o freio custa −14,8/s, 64% de todos os custos ligados ao ruído; o std cai a 0,45 contra 0,61 da task de fábrica; com o robô de pé a taxa líquida fica em ~0/s e a marcha atrasa |
| −6 (`zero06`) | a marcha avança bem; na pega, pela conta de 21/09, a pressa volta a compensar (empate em ~−12) |

**Referência:** o exemplo YAM do mjlab (`tasks/manipulation/lift_cube_env_cfg.py:199-211`)
explora com o freio quase desligado e o endurece 100× depois que a tarefa se forma. Ver a
memória `g1-limpo-yam-lento-por-desenho`.

## 2. As decisões do dono (23/09)

- O freio vale para as duas cadeias, a do PEGAR (`s_B`) e a do BOTAR (`s_C`).
- O peso é POR ENV, como o `nivel`.
- Ele sobe um degrau por vez, para a política se adaptar.
- Ele NÃO recua: depois que sobe, fica.
- O estado vai no checkpoint.
- O custo NÃO é a média das juntas: uma junta sozinha acima do limite tem de pagar de verdade.

## 3. O desenho

### 3.1 O custo: a soma das dobradiças

Para cada uma das 29 juntas, `r = |v| / vmax`, com o vmax da tabela do regime
(`knobs.py:752-781`, sem mudança). O custo do passo:

    custo = Σ_j relu(r_j − 1)²

| situação | custo do passo |
|---|---|
| todas abaixo do limite | 0 |
| 1 junta a 2× o limite | 1 |
| 1 junta a 3× | 4 |
| 2 juntas a 2× | 2 |

Com a soma, o peso é o PREÇO POR JUNTA: uma junta a 2× custa exatamente o peso, por segundo.

Formas descartadas, com a conta de 23/09 (cada peso calibrado para o pulso de 21/09 custar o
mesmo que custa hoje):

| forma | defeito |
|---|---|
| média das 29 (hoje) | com os pesos que usamos, 1 junta paga 1/29 do peso: a −6, uma junta a 2× custa 0,21/s |
| média das 5 maiores, dobradiça antes | acima de 5 juntas rápidas, as outras passam de graça: 20 juntas a 2,5× custam 5,8 contra 23,3 |
| média das 5 maiores, dobradiça depois | as juntas lentas escondem a rápida: `[3,0; 0,6; 0,5; 0,5; 0,4]` custa zero |

### 3.2 O preço por env

    preço = piso × fator^degrau     fora do ANDAR
    preço = piso                    no ANDAR

O `RewardTermCfg` leva `weight = −piso`, e o termo multiplica o custo por `fator^degrau` nos
envs fora do ANDAR (`env.limpo_estado != ESTADO_ANDAR`).

Com piso 0,207 e fator 1,5:

| degrau | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| preço por junta | 0,207 | 0,31 | 0,47 | 0,70 | 1,05 | 1,57 | 2,36 |

**Por que 0,207:** é 6/29, o equivalente EXATO do −6 da `zero06` na média de hoje. No degrau 0
a recompensa nova é idêntica à da `zero06`.

**Por que fator constante, e não −1 por degrau:** −2 para −3 é +50%, e −14 para −15 é +7%. A
política sente a mudança relativa; o fator fixo dá o mesmo impacto a cada degrau.

**O balanço de 21/09 na nova escala.** A rajada da pega custava 8,64 a −15 na média, ou seja a
0,517 por junta. O empate com os 6,9 da pressa fica em **0,41 por junta** (≈ −12 na média). O
degrau 2, com 0,47, já torna a pressa um mau negócio; o degrau 6 fica 5,7× acima do empate.

### 3.3 O degrau: recorde de nível

No fim de cada episódio de cadeia, o `nivel` do env anda ±1, como hoje
(`curriculo.py:94-122`). Se o novo valor passar do maior nível que aquele env já alcançou, o
degrau sobe 1. O degrau nunca desce.

    recorde = nivel > nivel_max
    sobe    = recorde & (episodios_desde_o_ultimo >= M) & (degrau < degraus_max)

**Por que recorde, e não "sucesso acima de um alvo por n episódios":** o passeio ±1 equilibra
cada env em ~50% de sucesso, por construção. Uma condição de sucesso no mesmo env mediria
sequências de sorte, e não competência. O nível máximo só sobe e mede competência.

**A sequência que isso produz, sem código de coordenação:** o robô aprende a cadeia no piso e
bate o nível 1: degrau 1. O freio mais forte faz o env errar mais, e o nível recua. O robô se
adapta, passa do nível 1 e bate o 2: degrau 2. A descida do `nivel` é a válvula de alívio: um
degrau forte demais torna a tarefa mais difícil, e o nível recua até o sucesso voltar. O
freio não prende o env.

**O espaçamento `M`:** sem ele, uma sequência de 6 sucessos seguidos subiria o freio 6 degraus
em 6 episódios, sem tempo de adaptação. `M` conta episódios de cadeia daquele env desde o
último degrau. Sugestão: 10.

**O teto:** o `nivel` tem 7 valores (`knobs.Nivel.n_niveis = 7`), portanto há no máximo 6
recordes. `degraus_max = 6` é o teto natural, com preço 2,36 por junta.

⚠ **O recorde é lido ANTES do piso de nível.** O piso (`curriculo.py:124-142`) sorteia níveis
ao acaso para uma fração dos envs; um nível sorteado alto seria um recorde falso.

### 3.4 A parada pela velocidade — DECISÃO PENDENTE

**Versão simples, a deste plano:** sem parada. O degrau segue os recordes até o teto. O dono
registrou que subir além não atrapalha depois que o robô aprende a não correr: abaixo do vmax o
custo é zero em qualquer degrau.

**Se a parada entrar:** o critério NÃO pode ser "alguma junta passou do limite no episódio".
Com o ruído da exploração algum passo sempre passa, e a parada nunca dispararia. O critério
robusto é a fração dos passos de PEGAR e BOTAR com alguma junta acima do vmax, por exemplo
acima de 5%. Isso pede dois contadores por env.

### 3.5 O checkpoint

`runner.CHAVES_POR_ENV` ganha `limpo_freio` e `limpo_nivel_max`. O contador de espaçamento
não entra: perdido numa retomada, ele só atrasa um degrau.

⚠ **Retomada de um checkpoint sem esse estado:** o `limpo_nivel_max` tem de nascer IGUAL ao
`limpo_nivel` restaurado, e não zero. Nascido em zero, qualquer env acima do nível 0 registraria
um recorde no primeiro episódio, e o freio saltaria.

## 4. As mudanças por arquivo

### 4.1 `g1_limpo/recompensas.py` — `velocidade_por_regime`

O `__init__` recebe `fator`. O `__call__` troca a média pela soma e aplica o preço do env:

```python
excesso = torch.relu(v.abs() / vmax - 1.0) ** 2
custo = excesso.sum(dim=1)                      # o peso é o preço por junta
degrau = getattr(env, "limpo_freio", None)
if degrau is not None:
    from g1_limpo.comando import ESTADO_ANDAR
    manip = env.limpo_estado != ESTADO_ANDAR
    custo = custo * torch.where(manip, self.fator ** degrau.float(),
                                torch.ones_like(custo))
return custo                                    # o RewardTermCfg multiplica por −piso
```

A docstring e o comentário do knob (`knobs.py:733-848`) são reescritos: o balanço de 21/09
passa para a escala da soma.

### 4.2 `g1_limpo/curriculo.py` — `nivel`

Uma função `garante_freio(env)` cria os três buffers por env (`limpo_freio`,
`limpo_nivel_max`, `limpo_freio_espera`), no mesmo idioma do `garante_nivel`. Dentro do
`nivel`, depois do passo ±1 e ANTES do piso:

```python
freio, nivel_max, espera = garante_freio(env)
espera[env_ids] += de_cadeia.long()
recorde = buf[env_ids] > nivel_max[env_ids]
sobe = recorde & (espera[env_ids] >= espacamento) & (freio[env_ids] < degraus_max)
freio[env_ids] += sobe.long()
espera[env_ids] = torch.where(sobe, 0, espera[env_ids])
nivel_max[env_ids] = torch.maximum(nivel_max[env_ids], buf[env_ids])
```

O `nivel` continua devolvendo a média do nível: a chave `Curriculum/nivel` fica igual, e o
`leitura.py:100` e o smoke (`smoke.py:1264`) não mudam.

### 4.3 `g1_limpo/runner.py`

`CHAVES_POR_ENV = ("limpo_nivel", "limpo_elo", "limpo_freio", "limpo_nivel_max")`. No `load`,
se o checkpoint não trouxer `limpo_nivel_max`, ele recebe o `limpo_nivel` restaurado (§3.5).

### 4.4 `g1_limpo/knobs.py`

- `Tarefa.velocidade_por_regime`: −15,0 → −0,207, o piso.
- Novos campos ao lado dele: `freio_fator = 1.5`, `freio_degraus = 6`, `freio_espacamento = 10`.

### 4.5 `g1_limpo/env_cfg.py`

- `fator` entra nos params do `velocidade_por_regime` (`env_cfg.py:686`).
- `degraus_max` e `espacamento` entram nos params do `nivel` (`env_cfg.py:526`).

### 4.6 `g1_limpo/metricas.py` — duas réguas para o log

- `freio_degrau`: o degrau médio dos envs, `reduce="last"`. Sai como
  `Episode_Metrics/freio_degrau`.
- `mao_na_pega`: a velocidade linear da mão mais rápida, em m/s, nos estados PEGAR_SEM,
  PEGAR_COM e BOTAR, pela `media_por_estado` com uma grandeza nova. É a grandeza da ISO. ⚠ Ela
  é uma MÉDIA sobre os passos; o pulso de 0,62 s aparece diluído. Ela lê a tendência, e não o
  pico.

### 4.7 `g1_limpo/smoke.py` — cinco checagens

1. Uma junta a 2× o limite custa exatamente o preço.
2. No ANDAR o preço é o piso, qualquer que seja o degrau.
3. O degrau sobe só com recorde E espaçamento cumprido, e nunca desce.
4. O piso de nível não conta como recorde.
5. Numa retomada sem `limpo_nivel_max`, ele nasce igual ao `limpo_nivel`.

### 4.8 Os notebooks do zero

⚠⚠ **O override `weight = -6.0` SAI dos dois notebooks do zero NO MESMO COMMIT.** Com a soma,
−6,0 seria o preço de 6 por junta: um freio 29× mais forte que o da `zero06`.

## 5. Como aplicar

**Na `zero06`, sem warm-start.** No degrau 0 a recompensa nova é idêntica à da `zero06`, porque
0,207 na soma é o −6 na média. O `nivel` da `zero06` está em 0 em todos os envs, portanto nenhum
degrau sobe até a cadeia fechar. A LR fica em 1e-3.

⚠ A impressão digital de pesos compara o NÚMERO do peso: −6,0 no `zero06.pesos.json` contra
−0,207 no código novo. O assert da retomada dispara, embora a recompensa seja a mesma. Na sessão
da troca, o json antigo tem de ser apagado, com a razão registrada no commit.

## 6. Riscos

- **O crítico não vê o degrau.** Entre o degrau 0 e o 6 o preço muda 11×, e a mesma observação
  passa a ter retornos diferentes. A perda de valor sobe na manipulação. Mitigação possível: um
  canal só do crítico com o degrau, acrescentado por último (contrato de `observacoes.py`). Isso
  muda a entrada do crítico de 131 para 132 e exige migrar o checkpoint por append. **DECISÃO
  PENDENTE.**
- **O robô perde o jeito rápido antes de achar o lento.** O fator de 1,5, o espaçamento `M` e a
  descida do `nivel` limitam isso.
- **A pega aprendida no piso é rápida.** No piso, 0,207 está abaixo do empate de 0,41: o robô
  aprende primeiro a pega rápida, e o currículo a corrige depois. É o desenho do YAM, e é
  intencional.
- **O CARREGAR fica fora do ANDAR** e recebe o preço do degrau, com a tabela de andar. Carregar a
  caixa depressa também é risco para pessoas.

## 7. O que olhar no log

| canal | o que responde |
|---|---|
| `Episode_Metrics/freio_degrau` | o freio está subindo? ele só sai de 0 depois que o `Curriculum/nivel` sair de 0 |
| `Episode_Metrics/mao_na_pega` | a mão desacelera na pega? a meta é 0,25 m/s |
| `Curriculum/nivel` | caiu depois de um degrau e voltou? é a válvula funcionando |
| `Curriculum/forma/s_B` e `s_C` | o fecho sobrevive ao freio? queda sem volta em ~1000 iterações quer dizer degrau forte demais |
| `Mean value loss` | subiu com o degrau? é o risco do crítico cego |

## 8. Tamanho

Estimativa: ~40 linhas de código, fora comentários e as checagens do smoke. A contagem antes →
depois de cada arquivo sai no commit.

## 9. Decisões pendentes

| decisão | sugestão |
|---|---|
| piso | 0,207, o −6 da `zero06` |
| fator | 1,5 |
| teto | 6 degraus, o máximo que o `nivel` permite |
| espaçamento `M` | 10 episódios de cadeia |
| parada pela velocidade | fora, na primeira versão |
| canal do degrau no crítico | medir a perda de valor primeiro |
| onde aplicar | na `zero06`, que no degrau 0 não muda |

## 10. Contrato de implementação (23/09) — prevalece sobre as seções 3 e 4

Decidido pelo dono em 23/09: SEM parada pela velocidade; os números da seção 9 valem como
sugeridos (piso 6/29, fator 1,5, teto 6, espaçamento 10); o canal do degrau no crítico fica
fora.

**Duas mudanças no desenho, feitas ao fechar o contrato:**

- **Um buffer só para o freio.** O nível máximo e o degrau subiam sempre juntos, portanto o
  degrau É o nível que o freio já alcançou. A regra vira: o freio persegue o `nivel` do env
  para cima, um degrau a cada `espacamento` episódios de cadeia, e nunca desce. Numa retomada
  sem esse estado, o freio nasce em 0 e alcança o nível restaurado aos poucos, um degrau a
  cada `espacamento` episódios, sem salto. O `limpo_nivel_max` sai do plano.
- **O nível sorteado pelo piso não conta.** Ler o recorde antes do piso não bastava: o nível
  sorteado vale para o episódio SEGUINTE, e o fim desse episódio contaria como recorde. Uma
  marca por env, `limpo_nivel_sorteado`, tira da regra o episódio que começou num nível
  sorteado.

### Lote A — `recompensas.py`, `knobs.py`, `env_cfg.py`

- `knobs.Tarefa.velocidade_por_regime: float = -6.0 / 29.0` (o piso, preço por junta).
- Campos novos em `knobs.Tarefa`, logo depois dele: `freio_fator: float = 1.5`,
  `freio_degraus: int = 6`, `freio_espacamento: int = 10`.
- O comentário que antecede o peso é reescrito na escala da soma: o balanço de 21/09 vira
  "0,517 por junta pagava 8,64 pela rajada; o empate com a pressa é 0,41 por junta; o degrau 2,
  0,47, já passa do empate".
- `env_cfg.py`, params do `velocidade_por_regime`: acrescenta `"fator": tr.freio_fator`.
- `env_cfg.py`, params do currículo `nivel`: acrescenta `"degraus_max": <Tarefa>.freio_degraus`
  e `"espacamento": <Tarefa>.freio_espacamento`, lidos da MESMA instância de `Tarefa` que o
  termo usa.
- `recompensas.velocidade_por_regime`:
  - `__init__` guarda `self.fator = float(cfg.params.get("fator", 1.0))`.
  - `__call__` ganha o parâmetro `fator: float = 1.0` e o descarta com `del`, como os vmax.
  - O custo passa a ser a SOMA. O texto `torch.relu(v.abs() / vmax - 1.0) ** 2` tem de
    continuar LITERAL no código: dois notebooks o procuram com `inspect.getsource`. A palavra
    `max=4.0` não pode aparecer.
  - Com `env.limpo_freio` presente, o custo é multiplicado por `self.fator ** freio` nos envs
    com `env.limpo_estado != ESTADO_ANDAR`, e por 1 nos outros. Sem `env.limpo_freio`, o custo
    sai sem multiplicador. `ESTADO_ANDAR` vem de `g1_limpo.comando` por import tardio, como as
    outras funções do módulo fazem.

```python
excesso = torch.relu(v.abs() / vmax - 1.0) ** 2
custo = excesso.sum(dim=1)
freio = getattr(env, "limpo_freio", None)
if freio is not None:
    from g1_limpo.comando import ESTADO_ANDAR
    manip = env.limpo_estado != ESTADO_ANDAR
    custo = custo * torch.where(manip, self.fator ** freio.float(), torch.ones_like(custo))
return custo
```

### Lote B — `curriculo.py`, `runner.py`, `metricas.py`

- `curriculo.garante_freio(env) -> tuple[Tensor, Tensor, Tensor]`: cria, se faltarem,
  `env.limpo_freio` (long, zeros), `env.limpo_freio_espera` (long, zeros) e
  `env.limpo_nivel_sorteado` (bool, False), todos com `num_envs` entradas no `env.device`, e
  devolve os três nessa ordem. Mesmo idioma do `garante_nivel`.
- `curriculo.nivel` ganha `degraus_max: int = 0` e `espacamento: int = 0`. Com
  `degraus_max == 0` nada muda. Com `degraus_max > 0`, dentro do bloco do passeio, logo depois
  de `buf[env_ids] = (buf[env_ids] + passo).clamp(...)`:

```python
freio, espera, sorteado = garante_freio(env)
elegivel = de_cadeia & ~sorteado[env_ids]
espera[env_ids] += de_cadeia.long()
sobe = (elegivel & (buf[env_ids] > freio[env_ids])
        & (espera[env_ids] >= espacamento) & (freio[env_ids] < degraus_max))
freio[env_ids] += sobe.long()
espera[env_ids] = torch.where(sobe, torch.zeros_like(espera[env_ids]), espera[env_ids])
```

  E no bloco do piso de nível, a marca: `sorteado[env_ids] = False` para todos os `env_ids`,
  e `True` para os que o piso sorteou (`env_ids[sorteia]`). Com `degraus_max == 0` a marca
  não é tocada. O `nivel` continua devolvendo `float(buf.float().mean())`.
- `runner.CHAVES_POR_ENV = ("limpo_nivel", "limpo_elo", "limpo_freio")`. No `load`, antes do
  laço que copia os buffers por env, `garante_freio(e)` é chamado, para o buffer existir
  quando o laço o procurar. `limpo_freio_espera` e `limpo_nivel_sorteado` NÃO vão no
  checkpoint.
- `metricas.py`:
  - Função nova `freio_degrau(env) -> Tensor`: `env.limpo_freio.float()`, ou zeros com
    `num_envs` entradas se o buffer não existir.
  - `media_por_estado` ganha a grandeza `"mao_vel"`: a maior das duas normas de
    `robot.site_lin_vel_w` nos sítios `("left_palm", "right_palm")`, com os ids resolvidos no
    `__init__` por `find_sites`.
  - Em `termos()`, duas entradas novas: `"freio_degrau"` (`func=freio_degrau`,
    `reduce="last"`) e `"mao_na_pega"` (`func=media_por_estado`, `reduce="last"`,
    `params={"grandeza": "mao_vel", "estados": (ESTADO_PEGAR_SEM, ESTADO_PEGAR_COM,
    ESTADO_BOTAR)}`). As chaves de log saem `Episode_Metrics/freio_degrau` e
    `Episode_Metrics/mao_na_pega`.

### Lote do main — `smoke.py` e os notebooks do zero

- A seção 3 do smoke passa a esperar a SOMA: com as 29 juntas a 2×, custo 29; a 3×, 116; a
  5×, 464. O check de texto troca `torch.mean(...)` pelo `excesso.sum(dim=1)`.
- Checagens novas: o preço no ANDAR não muda com o degrau; o degrau sobe só com nível acima
  do freio, espaçamento cumprido e episódio não sorteado; ele nunca desce.
- O override `weight = -6.0` sai dos dois notebooks do zero.

### Mudanças do code-review (23/09) — prevalecem sobre o resto da seção 10

- **O degrau só sobe com SUCESSO no episódio.** Sem isso, um env que falha subia o freio: o
  passeio o leva de um nível sorteado alto para baixo, e os episódios seguintes já não têm a
  marca. Com 20% de piso, todo env que nunca fecha a cadeia chegava ao nível máximo da
  população.
- **`freio_degraus = None` é "até o maior nível"; `0` desliga.** O teto sai de
  `env_cfg.degraus_do_freio(k)`, e não de um 6 copiado à mão. O teto 0 é o ÚNICO
  interruptor: o `nivel` zera o `limpo_freio`, que o runner restaura de qualquer
  checkpoint. O fator do termo fica sempre em `freio_fator` (2ª rodada).
- **A linhagem dos blocos fica com o freio antigo.** O `g1_limpo_kaggle.ipynb` sobrescreve
  o peso para −15/29 (o −15 na média), com teto 0.
- **A impressão digital aceita a troca de escala** só no `g1_limpo_kaggle.ipynb`: um peso `w`
  na média antiga contra `w/29` na soma passa com um aviso. Qualquer outra diferença
  continua barrando.
- **A retomada da Kaggle pega a pasta mais nova QUE TEM checkpoint**, e não a mais nova de
  todas. Uma pasta vazia de uma queda nesta sessão não trava mais a retomada.
- O termo resolve `ESTADO_ANDAR` uma vez no `__init__` e multiplica por
  `fator ** (degrau × manip)`. O `nivel` desempacota os buffers do freio uma vez só.

### 2ª rodada do code-review (23/09)

- **A marca de sorteio só cai quando um episódio de CADEIA termina.** O reset de um episódio
  de locomoção apagava a marca, e o nível sorteado chegava sem ela ao episódio de cadeia
  seguinte. O smoke cobre o caso com um episódio de locomoção no meio.
- **A `zero07` começa do zero**, e não retoma a `zero06` (decisão do dono). Isto substitui o
  rollout da §5. O `g1_limpo_zero_kaggle.ipynb` vem com `RETOMA = False`, sem a exceção de
  escala, e com `freio_fator`, `freio_degraus` e `freio_espacamento` na impressão digital.
- **O `g1_limpo_zero.ipynb` saiu**, substituído pelo `g1_limpo_zero_kaggle.ipynb`.
- **A `mao_vel` lê os sítios de `cena.PALM_SITES`**, a mesma fonte do comando.
