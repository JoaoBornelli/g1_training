---
tipo: achado
pai: "[[Locomoção]]"
estado: ativa
verificado: 2026-10-06
---
# g1-locomocao-primeiro-fatia-100 (original)

Corpo original da memória `g1-locomocao-primeiro-fatia-100`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Estabelecido em **2026-08-25**, conferido por mim no codigo. E a explicacao causal do
"andou / nao andou", e ela **valida a hipotese original do usuario** de 2026-08-24:

> "eu suspeito da arquitetura ou a divisao dos treinos. treinar tudo junto pode estar
>  impactando negativamente ambas as partes"

⚠ Eu havia respondido que "a divisao em duas formas NAO e o defeito". Isso estava
incompleto. O defeito nao e DIVIDIR — e treinar as duas **ao mesmo tempo, cedo demais**.

## O mecanismo, nos dois modulos

**`g1_multitask` — sequencial por construcao (talvez por acidente):**

```
g1_multitask/curriculum.py:118   self.abertas: list[int] = [T.LOCOMOVER]
g1_multitask/curriculum.py:389   if float(self.perf[cel][self._topo(cel)]) < self.limiar: continue
g1_multitask/knobs.py:756        limiar_competencia: float = 0.90
```

Só `LOCOMOVER` nasce aberta. Nenhuma outra tarefa abre antes de `perf[locomover] >= 0,90`.
Portanto a locomocao recebeu **100% dos dados** durante **22.296 iteracoes**.

⚠ **Corrigido em 25/08 pelo usuario:** ela nao levou 22.296 iteracoes para ANDAR. Ela
andava "ja a algum tempo"; o `perf` so descolou em 22.296 porque o **`push_robot` resetava
o gatilho de sucesso**. Ou seja: a fase sequencial durou MAIS do que o desenho pretendia,
por causa de um bug de medicao. O efeito sobre a fatia de dados e o mesmo, e mais forte.

**`g1_poc` — simultaneo, e cedo:** o portao da forma media SOBREVIVENCIA
(`dur_loco_ema >= 600`), um robo imovel marcava 1000 passos, o portao abria, e a
manipulacao levava 70% das transicoes em ~420 iteracoes.
Ver g1-poc-portao-forma-mede-sobrevivencia.

## A aritmetica que torna isso fatal, e que o proprio poc ja documentou

`ESPECIFICACAO-g1_poc.md:1020-1035` tem a tabela. A fatia de dados de locomocao e
governada pelo **tempo de vida do episodio**, nao pelo sorteio:

| duracao do episodio de andar | transicoes de locomocao |
|---|---:|
| 24 passos (0,5 s — MEDIDO no inicio) | **1,1%** |
| 100 passos | 4,3% |
| 400 passos | 15,1% |
| 961 (igual a manipulacao) | 30,0% |

`frac_locomocao = 0,30` e por EPISODIO; o PPO aprende de PASSO. Com 7 episodios de
manipulacao de 961 passos contra 3 de locomocao de 24, andar recebe **1,1% do gradiente**.

E no `g1_multitask` a distorcao e estrutural e **nao ha correcao**: ele sorteia por
episodio (`curriculum.py:330`) e nunca corrige por duracao. Com o piso de amostragem
`rho = 0,30` (anti-esquecimento) o sorteio maximo das tarefas de andar fica em ~0,55, e a
fatia de transicoes resultante e ~5,5%; com todas competentes, ~3,1%.

**Portanto o multitask so andou porque a FASE 1 dele nao tinha manipulacao nenhuma.**
Depois que a manipulacao abre, a locomocao cai para 3-5% — e e por isso que ele "nunca
aprendeu a girar".

## A consequencia de desenho para o g1_limpo

1. **Currículo SEQUENCIAL na entrada.** A locomocao recebe 100% dos dados ate passar por
   uma regua ALCANCAVEL. Só depois a manipulacao abre. E o unico mecanismo com sucesso
   MEDIDO para o andar neste repo.
2. Quando as duas coexistirem, a fatia tem de ser corrigida por DURACAO, nao por sorteio.
   O `sorteia_forma`/`_alvo_locomocao` do poc ja faz isso
   (`f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)`) e o multitask NAO faz. Levar o do poc.
3. O portao de saida da fase 1 nao pode medir sobrevivencia. `razao_marcha >= 0,50`
   (conserto de 24/08, commit `3ff4847`) e a grandeza certa.
4. O piso anti-esquecimento (`rho`) e o que torna o alvo de fatia inalcancavel depois.
   Ele e uma escolha, nao um default.

## Nao confundir com o que NAO foi a causa

- **Nao foi o one-hot.** O poc tinha o equivalente funcional: bit `caixa_valida` em
  `command[:,9]` (`g1_poc/env_cfg.py:243-246`) + twist forcado a zero
  (`g1_poc/comando.py:826`) — o mesmo mecanismo de `g1_multitask/tasks.py:63`.
- **Nao foram (principalmente) os tres knobs.** `escala_acao_mult`, `upright_std`,
  `postura` viram suspeitos, nao causa. Ver g1-modulo-novo-heranca-comportamental.
- **NAO foi a escada de push nem a fase `PARADO`** (verificado no codigo 2026-08-27). Elas
  existiram: regime `PARADO`, regime `ANDAR`, escada de 5 niveis, `push_fator` por env,
  forca sustentada, celula `(PARADO, PUSH)`. `push.py:4-6` e `curriculum.py:43` registram
  que TODAS saíram na reforma de 07/08. O run que andou veio DEPOIS, com o `push_robot` do
  fabricante fixo em todos os envs, mais uma janela livre de 0,5 s. Portanto a estrutura de
  fases nao e evidencia pre-medida a favor dela.

**Candidato herdavel que o g1_limpo NAO tem** (verificado 2026-08-27): o multitask parte os
10% de `rel_standing_envs` ao meio — `frac_giro_no_standing = 0.5` e
`piso_giro_rad_s = 0.15` dao 5% parado puro e 5% giro parado (`vx = vy = 0`, `|wz| >= 0,15`).
Isso corta pela metade a populacao que colhe `exp(0) = 1,0` de graca no
`track_linear_velocity`. Ver g1-mjlab-velocity-anatomia.

Relacionado: g1-poc-portao-forma-mede-sobrevivencia,
g1-principio-incentivo-nao-penalizacao, g1-modulo-novo-heranca-comportamental,
g1-multitask-orcamento-igual-por-tarefa, g1-curriculo-congelamento-escala.


## A COMPARACAO FICOU LIMPA — e ela isola a configuracao de locomocao

Com a correcao de 25/08, os dois modulos passam a ser comparaveis num ponto so:

| | fatia de locomocao | resultado |
|---|---|---|
| `g1_multitask`, fase 1 | **100%** (abertas = [LOCOMOVER]) | **andou cedo** |
| `g1_poc`, bloco5 pos-conserto | **100%** (`frac_loco_sorteio = 1,0`, plano) | **estatua na it 997** |

Os DOIS rodaram locomocao pura. Um andou, o outro nao. **Portanto a diferenca NAO e a
fatia de dados, nem o orcamento, nem o one-hot.** Ela esta na CONFIGURACAO DE LOCOMOCAO.

Isto reabilita a bisseccao de 24/08 como a pista principal:

| item | multitask (andou) | g1_poc (nao andou) |
|---|---|---|
| `escala_acao_mult` | **0,8** | 1,0 |
| `upright_std` | **0,1** | 0,447 |
| `postura` (peso) | **0,5** | 1,0 |
| `terminacao` (queda) | **−200** (= −4,0/passo) | ausente |
| `joint_acc` | **−2,5e−7** | ausente |
| `dof_pos_limits` | −0,5 | −1,0 |

⚠ E ha veredito OPOSTO documentado sobre a escala de acao: o commit `3fa588a` do poc chama
o 0,8 de "o desvio que eu levei mais tempo para achar, e possivelmente o mais caro". Nao se
herdam os dois — exige escolha declarada ou A/B.

**O que a fatia de dados explica, e continua valendo:** por que a locomocao DEGRADA depois
que a manipulacao abre (cai a 3-5% das transicoes), e por que o giro nunca veio.
