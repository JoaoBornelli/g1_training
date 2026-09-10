# g1_limpo — a cauda pós-BOTAR é espera; levantar e o punho (v3.5)

Sobre a v3.4 (`g1-limpo-botar-fecha-e-para.md`), ainda não treinada. Cinco mudanças:
duas fecham furos da v3.4, uma paga o levantar, duas contêm o punho. **Uma adição**
(um `where` em `postura_ereta`), autorizada pelo dono em 2026-09-10 para este caso.

## §0 Os problemas, MEDIDOS ou lidos em código

### 0.1 A v3.4 paga `precise_pos` e `load` DUAS vezes na cauda

`VALIDA = (elo_interno ≠ ANDAR) × ¬aguardando` (`comando.py:677`). Na cauda pós-BOTAR o
`_elo` fica BOTAR e a espera acabou, logo `VALIDA = 1`. Com a cena intacta (v3.4 §2.4):

| termo | peso | na cauda |
|---|---|---|
| `precise_pos` | 3,0 | caixa no alvo → **paga cheio, ao vivo** |
| `load` | 2,0 | caixa apoiada, `perto` → **paga cheio, ao vivo** |
| `renda_congelada` | 1,0 | soma congelada no fecho — **que já inclui os dois** |

O módulo tem a regra escrita em `env_cfg.py:88-93`: "congelar isso e pagar de novo ao
vivo na cauda contaria o mesmo canal duas vezes". Os sete `TERMOS_CONGELAVEIS` foram
escolhidos porque PARAVAM de pagar na cauda — e paravam porque a caixa ia a +5 m. A
v3.4 tirou o teleporte e quebrou a regra.

### 0.2 Depois do fecho, tirar a caixa do alvo é grátis

`caixa_largada` desarma `soltou_fora` com `& (soltou < 0.5)` (`terminacoes.py:105`).
Bater na caixa e deslocá-la 30 cm sobre a laje não termina. Só `caiu` (chão) termina.
O dono pediu: "se o robô bater e tirar do alvo/derrubar a caixa ele termina".

### 0.3 Levantar depois do fecho é pago em ~5% do episódio

Na cauda, com o `VALIDA` consertado:

| canal | /s | depende da postura? |
|---|---|---|
| `renda_congelada` | ~7,6 | não |
| `rastreio` lin + ang de v=0 | até 4,0 | não — mede a BASE |
| `pose` + `upright` | 0 a 2,0 | sim |
| `dof_pos_limits` | −0,5 a 0 | sim |

Delta agachado → de pé ≈ +1,5/s × ~7 s = +10, contra um transiente de −2 a −4 e um
episódio de ~194. A pose agachada SOBREVIVE (`time_out` 78,0% na bloco13), portanto a
proteção da anuidade vale pouco. O dono e o PM concordam: o robô tende a ficar agachado.

`postura_ereta` não ajuda como está: `rampa_pelve × descarga`, e `descarga → 0` com a
caixa apoiada. Ele paga por erguer a caixa DA laje — o inverso.

### 0.4 O punho gira livre

O dono viu no `play`: "a posição das mãos na pega está meio estranha, ele tá com a mão
virada". Duas causas, as duas em código:

1. `PosturaPorElo` mascara TODAS as `JUNTAS_BRACO` durante `pegou ∧ ¬soltou`
   (`recompensas.py:102`), e `JUNTAS_BRACO` inclui `wrist_roll/pitch/yaw`. Girar o
   punho é grátis enquanto segura.
2. Fora da pega, `std_standing[".*wrist.*"] = 1,00` (`knobs.py:381`). Um punho a 1 rad
   custa `err² = 1`, dividido por 29 juntas: o `pose` cai 3,4%. Livre também.

Nenhum outro termo olha o punho: `squeeze` é força normal, `_alcancar` é distância.

## §1 A DECISÃO DO DONO (2026-09-10)

> nesse caso específico pode ser feita uma adição. (…) quando estiver tudo refinado
> certinho, vou treinar novamente em uma GPU boa (…) a arquitetura do treinamento deve
> estar fechada e terá que rodar sem alterações no meio do treino.

> bota como recompensa a mão estar sempre, ou sempre que possível, alinhada com o
> antebraço.

"Sempre que possível" = kernel macio, não portão. O `pose` já é esse kernel.

## §2 AS MUDANÇAS

### 2.1 `VALIDA` lê o elo PUBLICADO — `comando.py:673-677`

```python
        publica_andar = self._soltou | (aguardando & ~self._pegou)
        self._command[:, ELO] = torch.where(
            publica_andar, torch.full_like(self._elo, ANDAR), self._elo).float()
        base = (self._elo != ANDAR).float()
        self._command[:, VALIDA] = base * (~aguardando).float()
```
vira
```python
        publica_andar = self._soltou | (aguardando & ~self._pegou)
        self._command[:, ELO] = torch.where(
            publica_andar, torch.full_like(self._elo, ANDAR), self._elo).float()
        # ⚠ v3.5: o elo PUBLICADO, não o interno. Depois do fecho do BOTAR o interno
        # fica BOTAR (para o crítico) mas a tarefa ACABOU — e `VALIDA` significa "há
        # tarefa ativa". Com o interno, `precise_pos` e `load` pagavam ao vivo na
        # cauda POR CIMA da renda congelada (spec v3.5 §0.1).
        base = (self._command[:, ELO] != ANDAR).float()
        self._command[:, VALIDA] = base * (~aguardando).float()
```

Nenhum operando novo. O outro caso de `publica_andar` (`aguardando ∧ ¬pegou`) já tinha
`VALIDA = 0` pelo `¬aguardando`; só o caso `soltou` muda.

Efeito: os sete congeláveis zeram ao vivo na cauda de uma vez (`staged`,
`precise_pos`, `load`, `precise_ori` por `× VALIDA`; `squeeze`, `unload`,
`postura_ereta` já zeravam por `_fora_do_botar`). Sobra `renda_congelada`, `rastreio`,
`pose`, `upright` e os freios — o que o dono descreveu.

⚠ CONSEQUÊNCIA AUDITADA (coder, 2026-09-10) — o PM tinha escrito o contrário. Entrar
na cauda EXIGE `_sigma_pendente = True` (`ja_em_cauda = ~_sigma_pendente`,
`comando.py:647`). Hoje o `liga = _sigma_pendente & (VALIDA > 0.5)` (linha 683) dispara
no passo em que a cauda abre e limpa a flag. Com a §2.1, `VALIDA = 0` nesse passo e o
`liga` **nunca** dispara na cauda C. Rastreado:

- `_sigma_pendente` fica `True` até o fim → o bloco da cauda REEXECUTA todo passo. Para
  envs `soltou` é inócuo: `vira_carregar` e `fica` são filtrados por `~_soltou` e ficam
  vazios; sobra `_forcado[ids_cauda] = False`, idempotente. Nenhuma escrita em cena,
  alvo ou elo.
- `_recalcula_sigmas` não roda na abertura da cauda C. Os σ ficam nos da abertura do
  BOTAR; todos os leitores são `× _valida`, zerados. `raio_solta` é knob estático.
- `_pos_no_elo` não é atualizado na cauda C. É write-only no pacote.
- `aproxima_caixa` (`metricas.py:243`, gate `VALIDA > 0.5`) para de atualizar na cauda C.
  É um mínimo corrente, já atingido antes do fecho.
- Cauda de B e R: `publica_andar` falso, `VALIDA` 1, `liga` dispara como hoje.

Registre isto na docstring de `_aplica_espera` ou de `_zera_twist_nos_parados`, onde
couber melhor — uma frase, apontando para esta spec.

### 2.2 Fora do alvo termina depois do fecho — `terminacoes.py:98,103-106`

```python
    soltou_fora = (v_rel > v_solta) & ~no_alvo
    ...
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        soltou_fora = soltou_fora & (soltou < 0.5)
    return caiu | (soltou_fora & (pegou > 0.5) & apos_pegar)
```
vira
```python
    # ⚠ v3.5: ANTES do fecho, soltar fora do alvo exige velocidade (arremesso). DEPOIS
    # do fecho, fora do alvo BASTA: a caixa está onde o robô a pôs, e se saiu do raio
    # foi ele que bateu nela. Pedido do dono: "se o robô bater e tirar do alvo/derrubar
    # a caixa ele termina".
    soltou = getattr(env, "limpo_soltou", None)
    depois = soltou > 0.5 if soltou is not None else torch.zeros_like(no_alvo)
    soltou_fora = ((v_rel > v_solta) | depois) & ~no_alvo
    return caiu | (soltou_fora & (pegou > 0.5) & apos_pegar)
```

Um operando entra (`| depois`), um sai (`& (soltou < 0.5)`). A docstring do termo
("DESARMA na espera final: depois do fecho do BOTAR as mãos TÊM de sair, e isso não
pode terminar o episódio") está REFUTADA: as mãos saírem não move a caixa, e `~no_alvo`
já protege o fecho legítimo. Reescreva.

Folga: o fecho exige `perto ≤ tol_pos = 0,10 m`; a terminação dispara fora de
`raio_solta = precise_pos_sigma = 0,18 m`. **8 cm.**

⚠ O ALVO CONGELA depois do fecho — conferido: `_alvo_ancorado_na_base` só roda para
`vira_carregar`, que exclui `soltou`; `_aplica_elo` não roda de novo na cauda. Sem isso
o alvo derivaria e a terminação dispararia sozinha. Se você achar QUALQUER escrita em
`_command[:, ALVO]` que alcance envs `soltou`, PARE e me diga.

### 2.3 `postura_ereta` paga a pelve SOZINHA depois do fecho — `recompensas.py:559-587`

⚠⚠ REVISADO após a medição do §3 (2026-09-10): a pelve no fecho é p10 **0,379** / p50
**0,443** / p90 0,571 m. O `pelve_piso = 0,45` está ACIMA da mediana — mais da metade
das amostras teria `rampa = 0` com derivada 0. A rampa da cauda ganha PISO PRÓPRIO, e o
`pelve_piso` do PEGAR **não muda**: ele calibra um termo que fecha em 89%, e baixá-lo
suavizaria a inclinação da subida do PEGAR em 29%.

`knobs.py:731`, ao lado de `pelve_piso`:

```python
    pelve_piso: float = 0.45       # abaixo disto ela paga zero
    # ⚠ v3.5: o piso da rampa DEPOIS do fecho do BOTAR. MEDIDO no `model_9499`, estado
    # do fecho (`perto ∧ apoiada`), 8762 amostras: pelve p10 0,379 / p50 0,443 / p90
    # 0,571 / mín 0,334. O `pelve_piso` de 0,45 fica acima da MEDIANA — metade dos
    # fechos teria rampa 0 com derivada 0, onde a subida tem de nascer. 0,32 fica
    # 1,4 cm abaixo do mínimo observado; no p50 a rampa vale 0,29 e sobe até 1,0.
    pelve_piso_cauda: float = 0.32
```

`env_cfg.py` (~linha 609): passe `pelve_piso_cauda` nos params de `postura_ereta`, ao
lado de `pelve_piso`.

`recompensas.py`: a assinatura ganha `pelve_piso_cauda: float`, e o corpo:

```python
    z = (env.scene["robot"].data.root_link_pos_w[:, 2]
         - env.scene.env_origins[:, 2])
    rampa = ((z - pelve_piso) / max(pelve_alvo - pelve_piso, 1e-6)).clamp(0.0, 1.0)
    descarga = unload(env, nome_do_comando, sensor_apoio,
                      sensores_palma, mu, asset_cfg)
    # ⚠ v3.5 — A ÚNICA ADIÇÃO DO LOTE, autorizada pelo dono. Depois do fecho do BOTAR
    # a caixa está apoiada, `descarga → 0`, e este termo valia zero faça o que fizer a
    # pelve. Ali ele passa a pagar UMA RAMPA SOZINHA: erguer o corpo depois de largar.
    # Sem isto, levantar rendia ~+1,5/s contra ~11,6/s de anuidade — 5% do episódio —
    # e o robô ficava agachado (spec v3.5 §0.3). Com peso 2,0 a rampa dobra o delta.
    # ⚠ PISO PRÓPRIO (`pelve_piso_cauda`): a pelve no fecho está em 0,33–0,62 m
    # (MEDIDO, §3), abaixo do `pelve_piso` do PEGAR. Com o piso do PEGAR a rampa
    # nasceria em zero com derivada zero em metade dos fechos.
    # ⚠ SEM PENHASCO: dentro do BOTAR o termo é 0 (`_fora_do_botar` dentro de
    # `descarga`); depois do fecho vira `rampa_cauda ≥ 0`. Monótono no fecho.
    # ⚠ `clamp(0, 1)` em cima: ficar na ponta do pé ou subir na caixa não paga mais.
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is None:
        return rampa * descarga
    rampa_cauda = ((z - pelve_piso_cauda)
                   / max(pelve_alvo - pelve_piso_cauda, 1e-6)).clamp(0.0, 1.0)
    return torch.where(soltou > 0.5, rampa_cauda, rampa * descarga)
```

Na cauda de **B e R** (`CARREGAR`, `soltou = False`) nada muda.

`postura_ereta` está em `TERMOS_CONGELAVEIS`: no fecho ele congela em 0 (valia 0 dentro
do BOTAR) e depois paga ao vivo. Sem contagem dupla.

⚠ `g1_limpo/ARQUITETURA.md:1709` escreve a rampa do PEGAR com 0,45 e 0,75. Ela continua
verdadeira para o PEGAR. **Não toque no ARQUITETURA.md** — docs do G1 só sob pedido.

### 2.4 O punho FICA na média do `pose` durante a pega — `recompensas.py:102`

```python
        ids_braco, _ = asset.find_joints(list(JUNTAS_BRACO), joint_subset=joint_names)
```
vira
```python
        # ⚠ v3.5: o PUNHO não sai da máscara. Ombro e cotovelo trabalham para alcançar
        # e saem da média enquanto seguram; o punho não precisa girar para segurar
        # uma caixa, e girava (dono, `play` da bloco13: "a mão virada"). Pedido:
        # "a mão sempre, ou sempre que possível, alinhada com o antebraço".
        # `JUNTAS_BRACO` NÃO muda — `_ids_de_pe` é o complemento dela, e pôr o punho
        # no `de_pe` do PEGAR exigiria punho neutro no fecho, que hoje passa a 89%.
        braco_sem_punho = [p for p in JUNTAS_BRACO if "wrist" not in p]
        ids_braco, _ = asset.find_joints(braco_sem_punho, joint_subset=joint_names)
```

A máscara ENCOLHE de 14 para 8 juntas. Subtrativo. Durante a pega a média passa a ter
15 + 6 = 21 juntas em vez de 15; atualize os QUATRO comentários que dizem que sobram 15 juntas: `knobs.py:369`,
`recompensas.py:65` ("zerar 14 juntas"), `recompensas.py:88` e `env_cfg.py:305`.

### 2.5 `std_standing` do punho: 1,00 → 0,30 — `knobs.py:381`

```python
        r".*shoulder.*": 1.00, r".*elbow.*": 1.00, r".*wrist.*": 1.00,
```
vira
```python
        r".*shoulder.*": 1.00, r".*elbow.*": 1.00,
        # ⚠ v3.5: o MESMO 0,30 que o fabricante dá ao punho no `std_walking`
        # (`mjlab/.../g1/env_cfgs.py`), e o mesmo das outras juntas de rotação que
        # devem ficar neutras (hip_yaw, hip_roll, ankle_roll, waist). Com 1,00 um punho
        # a 57° custava 3,4% do `pose`; com 0,30 custa 41% (divisor de 21 juntas).
        r".*wrist.*": 0.30,
```

`std_walking` e `std_running` NÃO mudam: são do fabricante, e a locomoção é paridade.

## §3 A MEDIÇÃO — rodou UMA vez, 2026-09-10

`model_9499.pt`, 32 envs, CPU, `episode_length_s = 20.0`, `make_env_cfg(play=True,
elo=CMD.PEGAR, cadeia=2)`, 1500 passos. Estado: `_elo == BOTAR ∧ VALIDA ∧ perto ∧
apoiada`. **8762** amostras em 1360 passos.

**Pelve** (`root_link_pos_w[:, 2] − env_origins[:, 2]`, m):

| p10 | p50 | p90 | mín | máx |
|---|---|---|---|---|
| **0,379** | **0,443** | 0,571 | 0,334 | 0,620 |

Veredito: `< 0,40` → o PM decidiu **piso próprio para a cauda**, `pelve_piso_cauda =
0,32` (§2.3). O `pelve_piso = 0,45` do PEGAR fica.

**Punhos**, `|q − q_default|` em rad — o tamanho do giro que o dono viu:

| junta | p50 | p90 |
|---|---|---|
| left_wrist_roll | 0,53 | 0,93 |
| left_wrist_pitch | 0,45 | 1,12 |
| left_wrist_yaw | 0,93 | 1,63 |
| **right_wrist_roll** | **1,25** | **2,08** |
| right_wrist_pitch | 0,90 | 1,11 |
| right_wrist_yaw | 0,68 | 1,44 |

O punho direito rola **71°** na mediana e 119° no p90. A assimetria esquerda/direita
diz que não é geometria da pega — é hábito sem custo. Com `std = 0,30` e divisor de 21
juntas, 1,25 rad custa `err² = 17,4 → /21 = 0,83 → exp(−0,83) = 0,44`: o `pose` cai a
44% só por esse punho. O 0,30 morde.

Caveat registrado pelo coder: a fração `perto ∧ apoiada` deu 26,6% dos passos de BOTAR
nesta corrida contra 74,6% na sonda da manhã; nenhuma das duas tem seed. O veredito da
pelve não depende disso — as duas medições concordam que o robô está agachado no fecho.

## §4 O notebook — `g1_limpo/kaggle/g1_limpo_kaggle.ipynb`

`RUN = "bloco14"` e `META = 11500` **ficam**: a v3.4 nunca rodou, e a v3.5 entra na
mesma run. Acrescente os discriminadores de v3.5 logo após o bloco de v3.4:

```python
# --- OS DISCRIMINADORES DE v3.5 (cauda parada de pé, spec `g1-limpo-cauda-parada-de-pe.md`)
import inspect
from g1_limpo import recompensas as RC, terminacoes as TM
assert "limpo_soltou" in inspect.getsource(RC.postura_ereta), \
    "clone anterior à v3.5: `postura_ereta` não paga a pelve depois do fecho — o robô " \
    "fica agachado colhendo a anuidade (spec §0.3)"
assert "wrist" in inspect.getsource(RC.PosturaPorElo.__init__), \
    "clone anterior à v3.5: o punho ainda sai da média do `pose` durante a pega"
_std = rw["pose"].params["std_standing"]
assert any("wrist" in k and abs(v - 0.30) < 1e-9 for k, v in _std.items()), \
    f"clone anterior à v3.5: std do punho não é 0,30 — {_std}"
assert "self._command[:, ELO] != ANDAR" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera), \
    "clone anterior à v3.5: `VALIDA` lê o elo INTERNO — `precise_pos` e `load` pagam " \
    "duas vezes na cauda (spec §0.1)"
assert "(soltou < 0.5)" not in inspect.getsource(TM.caixa_largada), \
    "clone anterior à v3.5: tirar a caixa do alvo depois do fecho não termina"
```

Conferido pelo coder: `VALIDA` é escrito em `_aplica_espera` (`comando.py:677`), e a
chave do dict é `r".*wrist.*"`.

## §5 A CONTAGEM

| | antes | depois |
|---|---|---|
| operandos em `VALIDA` | 2 | 2 (um trocado) |
| operandos em `soltou_fora` | 3 | 3 (um trocado) |
| `where` em `postura_ereta` | 0 | **1** — a adição |
| juntas na máscara do braço | 14 | **8** |
| termos de recompensa | 28 | 28 |
| terminações | 3 | 3 |
| knobs | — | 2 números: o std do punho e `pelve_piso_cauda` |
| params de `postura_ereta` | 8 | 9 (`pelve_piso_cauda`) |

## §6 O que NÃO muda

`JUNTAS_BRACO` (o `de_pe` do PEGAR é o complemento dela), `std_walking`,
`std_running`, `renda_congelada`, `rastreio_por_elo`, `TERMOS_CONGELAVEIS`, os pesos, o
fecho do BOTAR (v3.4), a cauda parada (v3.4), a cena intacta (v3.4), o balanceador, o
piso de 30%. `smoke.py` não roda.

**Não entra**: portão de `de_pe` na renda congelada. Cria penhasco de 60% no fecho
(11,6 → 4,7/s) e a política volta a não fechar.

## §7 A SANIDADE — uma vez, no fim

16 envs, 50 passos, CPU, `make_env_cfg()` padrão, script no scratchpad. Sem exceção;
`env.limpo_twist_zerado` shape `(16,)`; e `cfg.rewards["postura_ereta"]` e `cfg.rewards["pose"]`
construídos sem erro (`make_env_cfg()` devolve o env cfg direto; `cfg.env.` é a forma
do notebook). Nada além.

## §8 O que ler no primeiro log da bloco14

| canal | o que decide |
|---|---|
| `Curriculum/forma/s_C` | **sair de 0,0000** — é a v3.4 funcionando |
| `Episode_Reward/postura_ereta` | subir: era 0,42/s; a cauda passa a pagar a pelve |
| `Episode_Reward/pose` | subir: era 0,37 no painel; o punho neutro e o corpo de pé |
| `Episode_Termination/caixa_largada` | pode subir um pouco (2.2 dispara na cauda); tem de estabilizar |
| `Episode_Termination/time_out` | cair de 78% só se a terminação nova morder |
| `Curriculum/forma/s_B` | não pode cair de 0,89 |
| `Episode_Reward/precise_pos` | **cair** ~metade — ele deixa de pagar na cauda. Esperado |

Se `s_C` sair de zero e `postura_ereta` não subir, a pelve não está cruzando
`pelve_piso`: a medição do §3 estava errada ou o knob ficou alto.

## §9 Commits

Atômicos, Conventional Commits, mensagem em português, **sem trailer de atribuição**
(`git -c core.hooksPath=/dev/null`). Seis:

1. `fix(limpo): VALIDA lê o elo publicado — a cauda pós-BOTAR não paga manipulação ao vivo`
2. `fix(limpo): depois do fecho, tirar a caixa do alvo termina`
3. `feat(limpo): postura_ereta paga a pelve sozinha depois do fecho do BOTAR`
4. `feat(limpo): o punho fica na média do pose, com std 0,30`
5. `chore(limpo): notebook — os discriminadores de v3.5`
6. `docs(specs): g1-limpo cauda parada de pé v3.5`

`pelve_piso_cauda` (knob + param) entra no commit 3.

Nunca commite: `docs/handoff/`, `docs/memoria/`, `record_rl.py`, `rl_rollout.npz`,
`g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`,
`reference_checkpoints/*.pt`, `ver_play.py`. Nunca `git push`.
