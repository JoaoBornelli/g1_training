# g1_limpo — a mão volta a ser exigida no BOTAR (v3.3)

Lote de conserto sobre a v3.2 (`g1-limpo-soltar-termina.md`). Uma função muda.

## §0 O problema MEDIDO

A bloco12 retomou da 6999 com a v3.2 e rodou 501 iterações. Ela removeu o arremesso
e **não** criou o apoiar.

| | it 7150 | it 7499 |
|---|---|---|
| `Episode_Termination/caixa_largada` | 54,6% | **29,4%** |
| `Episode_Termination/time_out` | 32,6% | **66,4%** |
| vida média (passos) | 537,9 | **824,3** |
| `Episode_Reward/load` | 0,0000 | **0,0001** |
| `Curriculum/forma/s_C` | 0,0005 | **0,0000** |
| Mean reward | 48,05 | **152,60** |

A renda triplicou e nada foi apoiado. `time_out` dobrou. O robô aprendeu a **segurar
a caixa perto do alvo até o fim do episódio**.

### A aritmética que cria o ótimo

Taxas reais em 7499 (`painel × 1000 / 824,28`):

| canal | /s | portão de mão? |
|---|---|---|
| `staged` | 2,93 | `_alcancar` — **hoje ≡ 1 no BOTAR** |
| `precise_pos` | 0,82 | **nenhum** (`recompensas.py:440`) |
| `precise_ori` | 0,24 | `_alcancar` — **hoje ≡ 1 no BOTAR** |
| `load` | 0,000 | `perto ∧ força de apoio` |

Segurar dentro do BOTAR rende **3,99/s** sem exigir a mão. Fechar o BOTAR congela
aproximadamente essa mesma soma. **O fecho ganha ~zero e arrisca morte** (`apoiada`
exige transferir o peso; soltar fora do `raio_solta` termina). A invariante do dono —
"o reward só sobe na troca de um elo" — está quebrada no BOTAR.

### O segundo exploit, ainda dormente

`caixa_largada` só dispara com `v_rel > v_solta = 1,2`. Depositar a caixa **na laje**,
devagar, dentro do σ do `precise_pos` mas fora do `tol_pos`, não mata. E `_soltou` só
acende no fecho, portanto `_alcancar` continua valendo 1: caixa na laje, mãos livres,
3,99/s. Hoje ele ainda segura (`palmas_em_contato` 0,533 → 0,673). Com 1500 iterações
a mais, ele acha.

## §1 A MUDANÇA

`recompensas.py::_alcancar` perde os DOIS ramos que devolvem `1` no BOTAR.

Estado atual:

```python
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is None:
        um = t._elo == BOTAR
        return torch.where(um, torch.ones_like(kernel), kernel)
    no_botar = (t._elo == BOTAR) & (soltou < 0.5)
    depois = soltou > 0.5
    return torch.where(depois, torch.zeros_like(kernel),
                       torch.where(no_botar, torch.ones_like(kernel), kernel))
```

Estado pedido:

```python
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is None:
        return kernel
    return torch.where(soltou > 0.5, torch.zeros_like(kernel), kernel)
```

Consequências, e são o objetivo:

- `staged` e `precise_ori` passam a medir a mão DENTRO do BOTAR. Mão fora da caixa,
  os dois caem juntos.
- A cauda continua em zero (v3.2 preservada): só `renda_congelada` paga lá.
- O `from g1_limpo.comando import BOTAR` dentro de `_alcancar` fica sem uso. **Remova.**
  Confira antes que nenhuma outra linha da função use `BOTAR`.

A docstring tem de registrar o que a v3.3 faz e por quê, no estilo do arquivo (o bloco
`⚠⚠ MUDANÇA v3.2` é o molde). O texto da v3.2 que afirma "a terminação `caixa_largada`
agora GARANTE que a mão está na caixa enquanto o BOTAR paga" **está refutado pela
medição do §0** e tem de sair: `v_solta = 1,2` deixa o depósito lento passar.

## §2 O que NÃO muda, e por quê

`precise_pos` continua sem portão de mão. É desenho, e a docstring dele já diz:
`staged` faz a rampa de aproximação, `precise_pos` responde "a caixa está NO alvo?".
Um aceite não exige a mão — quando a caixa está no alvo, mão livre é o estado
desejado, não um hack.

Resta um resíduo: depositar na laje dentro do σ do `precise_pos` (0,18 m) mas fora do
`tol_pos` rende **0,82/s** com as mãos livres. É 1/5 dos 3,99/s de hoje, e exige a
caixa genuinamente sobre a laje perto do alvo. Fica REGISTRADO e NÃO tratado neste
lote: tratar exigiria um segundo portão, e o dono proibiu mudança aditiva. Se o
`load` continuar em zero depois deste lote, este é o próximo suspeito.

## §3 A MEDIÇÃO — go/no-go, roda UMA vez

O risco é o degrau de renda no resume. Ele depende do kernel da mão na pose de
segurar TREINADA.

### O que a corrida de 2026-09-09 achou (`model_6999`, 32 envs, CPU, 1200 passos)

**Achado estrutural**, lido em `comando.py:1648` e confirmado na medição:

```python
sigma_alcance[ids] = (d_palma * c.sigma_fator).clamp(min=c.sigma_min)  # sigma_min = 0,08 m
```

Quando o `VALIDA` do BOTAR acende, as palmas já estão na caixa (`d` p50 = 0,060 m),
portanto o produto cai ABAIXO do piso e **σ trava em 0,08 em todo env**. No BOTAR o
kernel é sempre `exp(−(d/0,08)²)`:

| `d` (m) | kernel |
|---|---|
| 0,060 | 0,570 |
| 0,080 | 0,368 |
| 0,120 | 0,105 |
| 0,152 | 0,027 |

Isto REFUTA a invariante que a docstring do `_alcancar` afirma (kernel `= exp(−1) =
0,368` no passo do `VALIDA`, porque `σ = d₀`). No BOTAR `d₀ < σ_min`, logo o kernel
nasce em 0,570. A docstring tem de registrar a exceção.

**A amostra de `_elo == BOTAR ∧ pegou ∧ ¬soltou` deu 16 passos em 38.400 env-passos.**
O piso é 200, portanto os percentis não são veredito. Com a cadeia C forçada em 100%
dos envs, o `model_6999` quase nunca abre o BOTAR com a caixa na mão — coerente com
`s_C = 0,0005` do §0. O gargalo é o fecho do PEGAR, não o BOTAR.

`model_7500.pt` NÃO está nesta máquina; o último de `g1_limpo` em `~/Downloads` é o
`model_6999.pt`.

### A medição que vale

Mede o `d` na cauda `CARREGAR` da cadeia **B** (índice 0 de `CMD.CADEIAS`). É a MESMA
pose de segurar, e ela é densa no 6999 porque a cadeia B conclui (`s_B = 0,84`).

- `make_env_cfg(play=True, elo=CMD.PEGAR, cadeia=0)`, `episode_length_s = 20.0`
  explícito, 32 envs, CPU, `model_6999.pt`.
- Amostra os passos com `_elo == CARREGAR ∧ pegou ∧ ¬soltou`.
- Reporta `dist_palma_caixa` p10/p50/p90, a contagem, e o kernel reconstruído
  `exp(−(d/0,08)²)` nos mesmos percentis.
- **σ não precisa ser medido.** Ele é o piso 0,08 por construção, e a corrida anterior
  confirmou. Reporte-o só como conferência.

⚠ `cadeia=N` sozinho NÃO força a cadeia: o `reset_base_por_elo` lê o elo que o
currículo sorteou, e o comando só o sobrescreve depois. Sem `elo=` junto, os envs
resetam na faixa de locomoção (±0,50 m, qualquer rumo) e o robô nasce longe da
mobília.

Risco declarado: no BOTAR o robô BAIXA a caixa, e o `d` pode subir. O erro vai para o
lado conservador — o kernel real fica MENOR que o reconstruído. E o 6999 segura menos
que o 7500 (`palmas_em_contato` 0,533 → 0,673), o que empurra na mesma direção.

### O RESULTADO (2026-09-09, `model_6999`, 32 envs, CPU, 900 passos)

**4284 amostras** de `_elo == CARREGAR ∧ pegou ∧ ¬soltou` em 28.800 env-passos.

| | p10 | p50 | p90 |
|---|---|---|---|
| `dist_palma_caixa` (m) | 0,0409 | **0,0576** | 0,0941 |
| kernel `exp(−(d/0,08)²)` | 0,2509 | **0,5953** | 0,7701 |
| σ real da cauda (m) | 0,0800 | 0,0800 | 0,1000 |

O σ MEDIDO na cauda deu o piso 0,08 no p50, igual ao do BOTAR: a reconstrução deixa de
ser só argumento de código e passa a ter medição. A causa é a mesma — a abertura do
elo de cauda recalcula o σ com a caixa JÁ na mão.

**Veredito: kernel p50 = 0,595, banda do meio. SOBE**, com transiente de 120 a 150
iterações.

### Leitura

| kernel p50 | veredito |
|---|---|
| ≥ 0,85 | a mudança é quase grátis segurando; todo o efeito recai na mão fora. **Sobe.** |
| 0,15 a 0,85 | corta a renda de segurar na proporção; é o efeito pretendido. **Sobe**, e o transiente de warm-start é de 120 a 150 iterações. |
| ≤ 0,15 | o BOTAR pagaria quase nada mesmo segurando CERTO: o piso do σ está mal escalado ali. **NÃO sobe.** Relate e pare. |

Se a amostra vier com menos de 200 passos, diga isso em vez de reportar percentis.

## §4 O notebook

`g1_limpo/kaggle/g1_limpo_kaggle.ipynb`:

1. `RUN = "bloco13"` — as duas células que carregam o nome.
2. `META = 9500`. Ele retoma da 7500 (ou da 6999, se o resgate falhar); 9500 dá 2000
   iterações, ~2,8 h a 5,0 s/iter, dentro da sessão. O relógio continua sendo o
   segundo teto e a célula já o imprime.
3. Um discriminador de v3.3, junto ao bloco `--- OS DISCRIMINADORES DE v3.2 ---`:

```python
import inspect
from g1_limpo import recompensas as RC
assert "ones_like" not in inspect.getsource(RC._alcancar), \
    "clone anterior à v3.3: o `_alcancar` ainda devolve 1 constante no BOTAR — " \
    "segurar a caixa paga 3,99/s sem exigir a mão (spec §0)"
```

A impressão digital NÃO pega esta mudança: ela compara só `weight`, e nenhum peso
muda. O discriminador é a única trava. `IT_MINIMA` fica em 4000 — subir travaria o
retomar da 6999 se o resgate do 7500 falhar.

Não mexa na LR. A célula já baixa para 5e-4, e isto é warm-start.

## §5 Contagem — a mudança não é aditiva

| | antes | depois |
|---|---|---|
| ramos em `_alcancar` | 3 | **1** |
| termos de recompensa | 28 | 28 |
| portões de mão no BOTAR | 0 | **2** (`staged`, `precise_ori`) |
| imports em `_alcancar` | 1 | **0** |

Dois ramos saem, nenhum entra. O notebook ganha um assert; um guarda de clone não é
termo de recompensa.

A linha dos portões TEM NÚMERO, e é o que faltava. No BOTAR o σ é sempre o piso 0,08
(§3), portanto o portão vale `exp(−(d/0,08)²)`: mover a palma de 0,12 m para 0,06 m
multiplica o pagamento de `staged` e `precise_ori` por **5,4** (0,105 → 0,570). Antes
os dois eram constantes ali, com derivada ZERO em relação à mão. É o gradiente que
faltava, e ele é o objetivo do lote.

## §6 O que este lote NÃO toca

`terminacoes.py`, `comando.py`, `knobs.py`, `curriculo.py`, `env_cfg.py`, os pesos,
`precise_pos`, `renda_congelada`, o balanceador, o piso de 30% da locomoção.

`smoke.py` **não roda** — decisão do dono. Ele já falha por semântica velha da v3.2
(`smoke.py:319` lê `k.terminacao.caixa_dist_max`, removido). Não conserte isso aqui.

## §7 Commits

Atômicos, Conventional Commits, mensagem em português, **sem trailer de atribuição**
(`git -c core.hooksPath=/dev/null`).

1. `fix(limpo): a mão volta a gatear staged e precise_ori no BOTAR`
2. `chore(limpo): notebook — RUN bloco13, META 9500 e o discriminador de v3.3`
3. `docs(specs): g1-limpo mão no alcancar v3.3`

Nunca commite: `docs/handoff/`, `docs/memoria/`, `record_rl.py`, `rl_rollout.npz`,
`g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`,
`reference_checkpoints/*.pt`, `ver_play.py`. Nunca `git push`.
