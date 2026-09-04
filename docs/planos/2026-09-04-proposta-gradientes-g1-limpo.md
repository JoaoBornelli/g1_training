# Proposta de mudança — gradientes e trocas de objetivo do g1_limpo (v2.1)

Data: 2026-09-04. Base: `docs/planos/2026-09-04-auditoria-gradientes-g1-limpo.md`. Nenhum código foi alterado. Toda mudança é decisão do dono.

A v1 desta proposta acrescentava cinco termos. A v2 substitui a v1. Ela acrescenta UM termo e remove UM. O saldo de linhas é negativo. A §8 lista o que saiu da v1 e por quê.

## 0. As regras

1. Em todo instante existe incentivo da posição atual até o alvo do elo. Do fecho de um elo até o elo seguinte a renda não cai.
2. Par macro/preciso: mesma forma. Macro com raio grande e peso menor. Preciso com raio pequeno e peso maior. O raio preciso cobre o raio de aceite.
3. Simplicidade: antes de acrescentar um termo, procurar o que remover. Um mecanismo por função. Nenhum relógio duplicado.

## 1. O mecanismo central: troca de objetivo e hack de fecho

Os termos de posição pagam pelo ESTADO "caixa perto do alvo", todo passo. Quando um elo fecha e o alvo muda, o mesmo estado físico passa a pagar pouco. O fecho reduz a renda.

A política compara "fechar" com "ficar a 1 cm de fechar". O segundo mantém a renda alta. Ela escolhe o segundo. Este é o hack de fecho. Ele existe em toda transição em que a renda cai.

Medido na proposta, cadeia 3, com H1 aplicado:

| transição | renda antes | renda depois | degrau |
|---|---|---|---|
| PEGAR → CARREGAR | 12,5 | 9,4 | −3,1 |
| CARREGAR → BOTAR | 10,6 | 5,1 | −5,5 |
| BOTAR → espera final | 10,0 | 10,0 + largou | ≥ 0 |

O gradiente recomeçar no alvo novo é correto. A renda cair é o defeito. A correção CONGELA a renda do elo no passo do fecho, em número, e paga esse número como nível constante até o fim do episódio. Os termos do elo novo somam por cima. `renda depois = congelado + termos novos ≥ congelado = renda antes`. Um nível constante não muda derivada. Ele só muda a ordem entre "antes do fecho" e "depois do fecho" — e essa é a ordem que estava errada. A cada progresso, o termo soma o que já foi ganho com o que está sendo executado.

O módulo já usa esse mecanismo uma vez: `load` e `largou` foram acrescentados em 03/09 para o fecho do BOTAR pagar mais que pairar. A v2 generaliza esse mecanismo para todos os fechos, sem número escolhido à mão, e remove `load`.

## 2. Tabela da proposta

| # | buraco ou hack | mudança | tipo | onde |
|---|---|---|---|---|
| P1 | preciso morto no aceite (H1) | `precise_pos_sigma 0,05 → 0,18`; `precise_pos 2,0 → 3,0` | 2 números | `knobs.Tarefa` |
| P2 | `sustentacao` tem relógio próprio, com condição diferente do fecho, e morre após avanço (H2) | `sustentacao` lê `_sust` e `_sustain_alvo` do comando. O relógio próprio, o `avancou` e os params `tol_pos`, `tol_ang`, `sustenta_s` do termo saem | remoção | `recompensas.sustentacao`, `comando` |
| P3 | fecho de elo reduz a renda (H3, H4) | termo `renda_congelada`: no passo de cada fecho ganho, congela a soma dos termos dependentes de elo e a paga como constante. REORIENTAR inerte não conta. `load` sai. `largou` perde o fator `× load` | 1 termo entra, 1 sai | `recompensas`, `comando`, `env_cfg` |
| P4 | rastreio paga 4/s por velocidade zero forçada (H7 e as duas esperas) | `rastreio_por_elo` lê `env.limpo_twist_zerado`, publicado por `_zera_twist_nos_parados`. O gate por `ELOS_QUE_ANDAM` e o helper `_anda_neste_elo` saem | substituição, menos linhas | `recompensas`, `comando` |
| P5 | `andou ≥ 0,50 m` inalcançável em standing e desfeito por resample (H8) | no CARREGAR-andando o twist é sorteado UMA vez na abertura, com `‖v‖ ≥ 0,3`, e mantido até o fecho | poucas linhas | `comando._zera_twist_nos_parados` |
| P6 | rampa da pelve satura no limiar de `de_pe` (H9) | `knobs.Tarefa.pelve_margem = 0,03`; topo da rampa = `pelve_alvo + margem` | 1 número | `knobs`, `env_cfg` |
| P7 | NOVO: caixa derrubada da mesa ANTES da primeira preensão não termina; o env fica morto até o time_out | `caiu` deixa de exigir `pegou`. `escapou` continua exigindo | 1 edição | `terminacoes.caixa_largada` |
| P8 | NOVO: metade da manipulação nasce num elo inerte de 0,3 s, e a cadeia 1 é só um PEGAR atrasado | REORIENTAR sorteado a **5%** enquanto inerte (`Cadeia.prob_reorientar_inerte`); sorteio com peso em `sorteia_elo`. Não sai do sorteio: o canal do one-hot não pode ficar constante (normalizador). Decisão do dono, opção b | 1 knob, sorteio com peso | `knobs`, `curriculo`, `env_cfg` |
| P9 | `impacto_da_caixa` publica média de um pico | `reduce="max"` | 1 palavra | `metricas` |
| P10 | não existe régua da caixa no log | `aproxima_caixa` (mín. corrente de `d_alvo/σ_trazer`, `reduce="last"`) e `renda_por_elo` | 2 métricas, sem peso | `metricas` |

Não bloqueiam progressão e ficam para depois: H5 (volta dos braços após largar), H10 (ANDAR sem macro).

## 3. Detalhe

### P1 — `precise_pos`

```python
precise_pos: float = 3.0          # era 2,0
precise_pos_sigma: float = 0.18   # era 0,05
```

Derivada total caixa → alvo, recompensa por metro: 5,0 → 5,7 em d = 0,385; 2,5 → 16,1 em d = 0,10. Peso preciso 3,0 > macro efetivo 2,7. Raio 0,18 > aceite 0,10. Razão de raios 2,5: sem vale.

### P2 — `sustentacao` lê o relógio do comando

O comando já tem o relógio: `_sust` acumula enquanto a condição de fecho vale, zera quando não vale, zera no avanço e no reset (`comando.py:888-891, 971, 680`). O alvo de sustain por elo é calculado em `_avanca_elo` e não é guardado. A mudança:

- `comando`: guardar `self._sustain_alvo` por env, escrito na abertura do elo (reset, avanço). É o valor que `_avanca_elo` já calcula.
- `recompensas.sustentacao` vira função:

```python
def sustentacao(env, nome_do_comando):
    t = _t(env, nome_do_comando)
    return (t._sust / t._sustain_alvo.clamp(min=1e-6)).clamp(0.0, 1.0) * _valida(env, nome_do_comando)
```

O que sai: a classe com `self.t`, o `reset`, a leitura de `avancou`, os params `tol_pos`, `tol_ang`, `sustenta_s`; o buffer `self.avancou` e as três escritas dele no comando.

O que muda de comportamento: a `sustentacao` passa a pagar pela condição de fecho DO ELO (no PEGAR inclui `de_pe`; no CARREGAR-andando inclui `andou`). Hoje ela paga por `perto ∧ alinhado`, que é mais frouxo que o fecho — ela pagava por um estado que não fecha. E H2 desaparece por remoção: `_sust` já zera corretamente no avanço.

### P3 — `renda_congelada` entra, `load` sai

**O que congelar.** A SOMA dos termos que dependem do elo, em número, no passo do fecho. Não os termos vivos contra o alvo velho: depois do CARREGAR fechar, a caixa tem de sair do peito, e termos vivos contra o peito puniriam o elo novo.

**Onde o número mora.** O `RewardManager` do mjlab guarda `peso × valor` por termo em `_step_reward[:, i]`, em unidade por segundo, sobrescrito termo a termo na ordem do dict. A ordem do passo é recompensas → comando. Portanto o fecho do passo `t` acontece DEPOIS das recompensas de `t`, e as recompensas de `t+1` já leem o elo novo. O termo guarda a soma do passo anterior e a congela quando o contador de fechos sobe.

No comando, um contador por env:

```python
# em _avanca_elo_force, antes de mudar self._elo:
origem = self._elo[ids]
ganho = origem != REORIENTAR          # o inerte não conta
self._fechos[ids] += (ganho & (pode | fecha_terminal)).long()
```

Zera no reset. Incrementa em TODO fecho: avanço e terminal. Uma regra, sem caso especial.

O termo, inserido por ÚLTIMO em `cfg.rewards`:

```python
class renda_congelada:
    def __init__(self, cfg, env):
        rm = env.reward_manager
        self.idx = [rm.active_terms.index(n) for n in cfg.params["termos"]]
        z = torch.zeros(env.num_envs, device=env.device)
        self.soma_anterior, self.congelado = z.clone(), z.clone()
        self.fechos_anterior = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, nome_do_comando, termos):
        t = _t(env, nome_do_comando)
        fechou_agora = t._fechos > self.fechos_anterior                 # o fecho foi no passo t
        self.congelado += torch.where(fechou_agora, self.soma_anterior, 0.0)   # soma do passo t
        self.fechos_anterior = t._fechos.clone()
        self.soma_anterior = env.reward_manager._step_reward[:, self.idx].sum(dim=-1)
        return self.congelado * _valida(env, nome_do_comando)

    def reset(self, env_ids=None):
        # zera os três buffers em env_ids
```

Peso 1,0. `termos` = `staged`, `precise_pos`, `precise_ori`, `squeeze`, `unload`, `postura_ereta`, `sustentacao`, `track_linear_velocity`, `track_angular_velocity`. Os termos de locomoção pura não mudam na troca de elo e seguem ao vivo; congelá-los pagaria duas vezes.

**Dependência declarada.** O termo lê `_step_reward`, atributo privado do mjlab, e exige ser o último do dict. O smoke fixa os dois (§5). Um upgrade que renomeie o buffer falha no smoke, e não no treino.

**`load` sai.** O trabalho dele era pagar o fecho do BOTAR mais que pairar. O fecho terminal do BOTAR congela ≈ 10/s. Na laje, antes do fecho: pairar a 1 cm paga 10,0; tocar paga 10,0; 0,3 s depois o fecho congela +10,0. Sem degrau negativo. `load` fica redundante.

`largou` vira `soltou × (1 − exp(−(d_palma/0,10)²))`. O `× load` saía como gate de "caixa apoiada no alvo". Na espera final o BOTAR JÁ fechou com `apoiada`; se a caixa cair depois, `caiu` termina. O gate é redundante.

**Teste de hack.** O valor congelado é limitado pelo teto dos termos do elo. Fechar mais tarde para inflar custa tempo sem receber o congelado. O fecho é automático quando a condição vale o sustain. Não há alavanca.

Renda por segundo, cadeia 3, com P1–P4:

| ponto | congelado | ao vivo | total |
|---|---|---|---|
| PEGAR a 1 cm do fecho | 0 | 12,5 | 12,5 |
| CARREGAR aberto | 12,5 | 9,4 | 21,9 |
| CARREGAR a 1 cm do fecho | 12,5 | 10,6 | 23,1 |
| BOTAR aberto | 23,1 | 5,1 | 28,2 |
| caixa tocando a laje | 23,1 | 10,0 | 33,1 |
| BOTAR fechado | 33,1 | 10,0 | 43,1 |
| espera final, mãos fora | 33,1 | 11,0 | 44,1 |

Monótona. Cadeia 0: o fecho do PEGAR congela ≈ 12,5/s pelo resto do episódio; é o sinal de "tarefa feita".

O que o congelamento NÃO afeta: o controlador de fatia lê duração. O `PPOPorElo` normaliza a vantagem por elo interno, e uma constante por elo sai da normalização. O `Mean reward` sobe; é cosmético.

### P4 — um gate só para o rastreio

Hoje o rastreio é zerado por elo publicado (`ELOS_QUE_ANDAM`). Isso deixa três estados pagando 4,0/s por velocidade zero FORÇADA: a espera inicial, a espera final e o CARREGAR de segurar parado. Nenhum dos três é rastreio.

A regra nova: se a TAREFA zerou o twist, o rastreio não paga. `_zera_twist_nos_parados` já calcula a máscara `parados`; ela passa a ser publicada:

```python
self._env.limpo_twist_zerado = parados.float()
```

e `rastreio_por_elo` vira:

```python
def rastreio_por_elo(env, *, func, **kwargs):
    return func(env, **kwargs) * (1.0 - env.limpo_twist_zerado)
```

`_anda_neste_elo` sai. Os params `canal_do_elo`, `nome_do_comando`, `elos_que_andam` do termo saem. `ELOS_QUE_ANDAM` continua no `PosturaPorElo` e no `reset_base_por_elo`.

Envs de standing SORTEADO (10% da locomoção) não são "parados pela tarefa"; eles continuam rastreando zero e pagando. Correto.

Efeito nas esperas: a espera inicial passa de ≈ 5,8/s para ≈ 2,0/s, e a abertura do PEGAR paga ≈ 1,9/s. O degrau de −3,3 na abertura da tarefa vira ≈ 0. Risco declarado (já aceito no PEGAR): nada paga por ficar parado durante 1 s de espera. `action_rate` e o alvo ancorado na base seguram.

### P5 — twist fixo no CARREGAR-andando

Em `_zera_twist_nos_parados`, para `(_elo == CARREGAR) ∧ ¬segura_parado`: na abertura do elo, sortear uma vez `v_x ∈ [0,3; 1,0]`, `v_y = 0`, `w_z = 0`, guardar em `self._twist_carregar`, e escrever esse valor no `vel_command_b` todo passo até o fecho. Poucas linhas, ao lado da escrita de zero que já existe.

Sem isto: 10% dos envs recebem "fique parado" e "ande 0,50 m" ao mesmo tempo, e o resample de 3 a 8 s pode inverter o sentido antes de `andou` valer. Com o twist fixo, o rastreio integrado É deslocamento, e nenhum termo de deslocamento é necessário.

### P6 — margem da pelve

`Tarefa.pelve_margem = 0,03`. `postura_ereta` recebe `pelve_alvo + pelve_margem = 0,78` como topo. O fecho `de_pe` continua em 0,75. A rampa tem derivada em 0,75 e a política não para exatamente na borda do fecho.

### P7 — `caiu` desarmado

```python
return caiu | (escapou & (pegou > 0.5))
```

Hoje: `(caiu | escapou) & pegou`. Uma caixa derrubada da mesa ANTES da primeira preensão fica no chão até o time_out. O env paga ≈ 2/s e não aprende nada por até 18 s. `caiu` no reset é falso: o fundo da caixa está a `topo ≥ 0,04 > folga 0,02` do chão. `escapou` continua armado, pelo motivo do docstring.

### P8 — REORIENTAR inerte a 5%

A v2.0 propunha tirar o REORIENTAR do sorteio. O C1 achou a decisão registrada em `knobs.py`: o REORIENTAR continua sorteado porque o canal dele no one-hot não pode ficar constante. O normalizador do rsl_rl é `(x − μ)/(σ + 0,01)` sem clamp; um canal constante tem `σ = 0` e, ao acender, entra como ×100. É a regra que fixa `fatia_loco = 0,95`.

Opções: (a) 0%, canal constante, armadilha ao religar; (b) 5%, canal vivo, custo de 0,08% do tempo; (c) 50%, hoje. **Decisão do dono em 2026-09-04: (b).**

```python
# knobs.Cadeia
prob_reorientar_inerte: float = 0.05
```

`sorteia_elo` sorteia o elo de manipulação com peso `(p, 1 − p)` em vez de uniforme. Cadeias 2 e 3 passam de 2,8% para 5,3% dos envs de manipulação. `_fechos` e `renda_congelada` continuam excluindo o fecho inerte.

### P9 e P10 — medição

`impacto_da_caixa` com `reduce="max"`. `aproxima_caixa`: mínimo corrente de `d_alvo/σ_trazer` por env, `reduce="last"`; 1,0 = não saiu de onde o elo abriu, 0 = no alvo. `renda_por_elo`: soma dos termos de manipulação por env; é a régua da regra 1 no treino.

## 4. O que NÃO muda

`staged` (3,0), `squeeze` (1,0), `unload` (2,0), `largou` (1,0), `precise_ori` (1,0), `postura_ereta` (2,0). `σ = d₀` em toda abertura de elo. `_fora_do_botar` booleano. `tol_pos`, `tol_ang`, `sustenta_*`. Toda a locomoção. `alvo_loco_min`. `reorientar_inerte`.

## 5. Verificação (smoke)

1. `precise_pos(d = tol_pos) ≥ 0,5`; derivada do par em `d = tol_pos` maior que em `d = d₀`.
2. `sustentacao` = `_sust / _sustain_alvo` em cada elo; zero no passo após um avanço forçado; 1,0 após `fechou`.
3. Renda no passo após cada fecho ≥ renda no passo antes: PEGAR → CARREGAR (cadeias 2 e 3), CARREGAR → BOTAR, BOTAR → espera final, fecho terminal da cadeia 0. Env sintético no estado de fecho.
4. `renda_congelada` é o ÚLTIMO termo de `cfg.rewards`; vale 0 durante REORIENTAR e PEGAR; após um fecho forçado vale a soma dos `termos` do passo anterior, e não a do passo corrente; `_fechos` não sobe no fecho do REORIENTAR inerte.
5. `rastreio_por_elo` = 0 nas duas esperas, no PEGAR, no BOTAR e no segurar-parado; ≠ 0 em ANDAR e no CARREGAR-andando; ≠ 0 em standing sorteado.
6. No CARREGAR-andando, `‖cmd‖ ≥ 0,3` e constante entre abertura e fecho.
7. `caixa_largada` verdadeiro com a caixa no chão e `pegou = 0`; falso no reset.
8. Com `reorientar_inerte = True`, nenhum env nasce em REORIENTAR.
9. Monotonia do BOTAR: pairar ≤ tocar < fechado < espera final.

## 6. Ordem de aplicação

Um bloco. Todas as mudanças são verificáveis pelo smoke antes do treino, exceto a força do shaping de P1, que é o único número que o treino avalia. Critérios do bloco: `sucesso > 0` na cadeia 0; `aproxima_caixa` caindo; `sustentacao > 0`; `renda_por_elo` sem degrau negativo em nenhuma transição observada.

H5 (volta dos braços) entra num bloco seguinte, só se a espera final for atingida e os braços não voltarem.

## 7. Checkpoint

O contrato de observação não muda (ator 114, crítico 131). Retomar de `model_2562.pt` é possível; o crítico reaprende a escala. O `assert` de pesos na célula de resume dispara por desenho. Run nova (`bloco9`). Treinar do zero é a alternativa.

## 8. O que saiu da v1, e por quê

| v1 | por quê saiu |
|---|---|
| H3, manter `σ_trazer` no avanço | o banco cobre o degrau; e sem H3 o estado terminal do CARREGAR é menos rico, o que REDUZ o degrau para o BOTAR de 7,1 para 5,5 |
| H4a, gate contínuo no BOTAR | o banco cobre; o booleano fica |
| H4b, `postura_ereta` no BOTAR | conflita com laje baixa (0,30 m exige agachar) |
| H5, termo `retorno_bracos` | não bloqueia progressão; a espera final é o FIM da cadeia. Adiado |
| H6, `pose` neutro no CARREGAR | canal morto e canal neutro têm a mesma derivada (zero); o degrau de nível o banco cobre |
| H8b, termo `deslocamento` | com o twist fixo (P5) o rastreio integrado já é deslocamento |
| banco `elo_fechado` com `w = 8,0` (v2 inicial) | substituído por `renda_congelada`: o valor congelado é a renda medida, e não um número que precisa ser refeito a cada mudança de peso |
| três blocos | tudo menos P1 é verificável pelo smoke; um bloco basta |

## 9. Verificado e NÃO é buraco

- O alvo do REORIENTAR é reescrito todo passo (`_update_command`, ramo `segue`). Empurrar a caixa na espera não trava o elo.
- `alinhado`: 8,2° medido contra 25°.
- `contato_tronco` é multa da mesa, não da caixa.
- `unload` lê força; o contato é rígido; `descarga = 0,933` prova caixa fora da mesa.
- Alvo do BOTAR ancorado na origem do env, e não na base: só a cadeia 3 chega ao BOTAR, e nela o robô não anda. Fragilidade registrada, sem ação.
