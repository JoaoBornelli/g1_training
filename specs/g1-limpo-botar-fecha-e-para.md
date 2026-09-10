# g1_limpo — o BOTAR fecha, e a cauda fica parada de pé (v3.4)

Três mudanças. Duas subtrativas, uma de knob. Nenhum código novo.

## §0 O problema MEDIDO

A `bloco13` (v3.3, 6999 → 9500) destravou o apoiar e não fechou o BOTAR:
`Episode_Reward/load` 0 → 0,441/s, `precise_pos` 0,82 → 1,66/s, e
`Curriculum/forma/s_C = 0,0000` em 2501 iterações.

Sonda de 2026-09-10, `model_9499`, 32 envs, CPU, **62 109 passos** com
`_elo == BOTAR ∧ VALIDA`:

| portão | fração verdadeira |
|---|---|
| `alinhado` | 0,9480 |
| `perto` | 0,8475 |
| `apoiada` | 0,7712 |
| **`de_pe`** | **0,0958** |

Entre os 44 336 passos com TRÊS portões verdadeiros, o quarto que falha é o `de_pe`
em **99,975%**. Os quatro fecharam juntos em **2 passos de 62 109**, não consecutivos;
o fecho exige `sustenta_outros_s` seguidos.

A junta que reprova é o **JOELHO em 100%** das reprovações (esquerdo 55,9%, direito
44,1%; nenhuma outra aparece). `amax|q − q_default|`: p10 0,803, p50 1,457, p90 1,977,
contra o limiar 0,69 — **o p10 já está acima**.

⚠ A hipótese do tornozelo está REFUTADA: os dois `ankle_pitch` são a pior junta em
0,89% dos passos e em ZERO das reprovações.

**A causa é geométrica.** A laje fica entre 0,30 e 0,55 m (`knobs.topo_min` por nível,
teto `prateleira_topo_teto = 0,55`). Apoiar uma caixa nessa altura exige agachar, e
`de_pe` proíbe agachar. O PEGAR fecha (`s_B` 0,89) porque o alvo dele é o PEITO: o
robô ergue a caixa e se levanta ANTES de fechar. O BOTAR não tem essa saída.

## §1 A DECISÃO DO DONO (2026-09-10)

Literal, e ela manda sobre a v3.1:

> com a caixa no alvo estável >0.5s, congelar a recompensa e mudar a tarefa para
> Andar (sem a caixa) novamente. (…) o robo não tem que aprender botar -> andando.
> quero só que depois do BOTAR ele receba um comando de ficar parado de pé, só

Duas reversões explícitas de decisões anteriores, e elas são intencionais:

1. **O `de_pe` sai do fecho do BOTAR.** Ele foi pedido em 08/09 ("fecho do botar deve
   ser perto ^ alinhado ^ apoiada ^ de_pe"). A medição do §0 mostra que, como escrito,
   ele torna o fecho impossível. Ele CONTINUA no PEGAR, onde funciona.
2. **A cauda pós-BOTAR passa a ser PARADA.** Em 08/09 ficou "a cauda pós-BOTAR: sorteio
   normal com giros". O dono retirou: `botar → andando` sai da distribuição.

O levantar NÃO ganha portão nem termo. Depois do fecho a tarefa publicada é ANDAR com
twist zero, e ANDAR já paga por estar de pé.

## §2 AS MUDANÇAS

### 2.1 `de_pe` sai do ramo BOTAR — `comando.py:1209`

```python
            elif elo_tipo == BOTAR:
                fecha[m] = (perto[m] & alinhado[m] & apoiada[m] & de_pe[m])
```
vira
```python
            elif elo_tipo == BOTAR:
                fecha[m] = (perto[m] & alinhado[m] & apoiada[m])
```

O ramo PEGAR (`comando.py:1207`) **não muda**: ele continua exigindo `de_pe`, e fecha
em 89% dos episódios. O cálculo de `de_pe` acima permanece — só o ramo BOTAR deixa de
lê-lo.

⚠ Se o `de_pe` ficar sem NENHUM leitor, isso é sinal de erro: o PEGAR tem de continuar
usando. Confira antes de remover qualquer coisa a mais.

### 2.2 `& ~soltou` sai do `parados` — `comando.py:1130`

```python
        parados = (torch.isin(self._elo, torch.tensor(
            self.cfg.elos_parados, device=self.device)) & ~self._soltou
        ) | (self._espera > 0.0)
```
vira
```python
        parados = torch.isin(self._elo, torch.tensor(
            self.cfg.elos_parados, device=self.device)) | (self._espera > 0.0)
```

Na cauda pós-BOTAR o `_elo` FICA `BOTAR` (`comando.py:654`), e `BOTAR ∈ elos_parados`.
Sem o `& ~soltou`, a cauda passa a ter twist zero até o fim do episódio.

⚠ Isto afeta SÓ a cauda pós-BOTAR. `_soltou` só liga no fecho do BOTAR; a cauda de B e
de R é `CARREGAR`, que NÃO está em `elos_parados`, portanto ela continua andando.

⚠ O que isso LIGA, e é o objetivo: `rastreio_por_elo` usa
`fator = 1 − zerado × (1 − engajado)` com `engajado = limpo_pegou`, monotônico e
sobrevivente ao soltar. Na cauda, `zerado = 1 ∧ pegou = 1` ⟹ fator 1. O robô passa a
ser PAGO por rastrear velocidade zero, de pé. É o incentivo a levantar.

A docstring de `_zera_twist_nos_parados` afirma hoje "na cauda pós-BOTAR o `_elo` fica
BOTAR e o twist tem de fluir". Isso está REVERTIDO pela decisão do §1 — reescreva a
linha, não a deixe contradizendo o código.

### 2.3 `sustenta_outros_s` — `knobs.py:792` e `comando.py:328`

`0.3` → `0.5`. Pedido do dono: a caixa tem de ficar estável no alvo por mais de 0,5 s.
Ele aceita que o robô decore esse tempo.

⚠ SÃO DOIS LUGARES com o mesmo número (`knobs.Alvo` e `AlvoCaixaCmdCfg`). Mude os dois,
ou eles derivam em silêncio. `sustenta_pegar_s` fica em 0,5 e NÃO muda.

⚠ O `REORIENTAR` também lê `sustenta_outros_s` (`comando.py:1218`). Subir para 0,5 s
alonga o atraso inerte dele de 0,3 para 0,5 s. É efeito colateral aceito: o REORIENTAR
é fração pequena e inerte por desenho.

### 2.4 A cauda pós-BOTAR NÃO mexe na cena — `comando.py:655-656`

```python
            sobe_caixa = (~self._pegou[ids_cauda]) | self._soltou[ids_cauda]
            self._laje_para(ids_cauda, self.cfg.afasta_z, sobe_caixa=sobe_caixa)
```
vira
```python
            # ⚠ O PÓS-BOTAR NÃO MEXE NA CENA (decisão do dono, 2026-09-10). O robô se
            # APOIA na caixa: a sonda de 10/09 mediu `forca_de_apoio` p50 = 1,24·m·g,
            # isto é MAIS que o peso da caixa — parte do corpo dele está ali. Tirar a
            # laje no instante do fecho o derruba, e o crédito da queda vai para o
            # fecho: seria desincentivo a botar.
            # ⚠ E a regra dos dois bits CONCORDA: a laje só saía porque a cauda ia ter
            # twist ≠ 0. A §2.2 zerou esse twist, logo `twist = 0 ⟹ laje presente`.
            fica = ids_cauda[~self._soltou[ids_cauda]]
            if len(fica):
                self._laje_para(fica, self.cfg.afasta_z,
                                sobe_caixa=~self._pegou[fica])
```

⚠ A cauda de **B e R** (`CARREGAR`, `soltou = False`) continua afastando a laje: ali o
robô SAI andando com a caixa na mão, e a laje na frente é obstáculo. Só o caso
`soltou` deixa de mexer.

⚠ Consequência pretendida: depois do BOTAR a cena fica intacta — laje no lugar, caixa
apoiada no alvo, robô parado de pé ao lado. É o mesmo estado da espera ANTES do PEGAR,
que é o que o dono pediu.

## §3 O que já existe e passa a acontecer de graça

Nada abaixo é código novo. Tudo já está implementado e hoje é inalcançável porque o
fecho não dispara:

| comportamento | onde já está |
|---|---|
| congelar a recompensa no fecho | `renda_congelada`, em `_fechos++` |
| a tarefa publicada vira ANDAR | `publica_andar = _soltou \| ...` (`comando.py:658`) |
| a mão deixa de ser exigida | `_alcancar` devolve 0 com `soltou > 0.5` |
| janela parada de 0,5 a 1,5 s antes da cauda | a espera armada pelo fecho (`knobs.py:259`) |
| soltar ali não termina o episódio | `caixa_largada` só dispara fora do `raio_solta` |

## §4 O notebook — `g1_limpo/kaggle/g1_limpo_kaggle.ipynb`

1. `RUN = "bloco14"` nas DUAS células que carregam o nome (a do clone e a do
   `empacota`). A semente é `1900-01-01_00-00-00_{RUN}` e o `load_run` é `.*_{RUN}$`;
   se só uma mudar, o checkpoint não é achado.
2. `META = 11500`. A bloco13 fechou na 9499; isso dá 2001 iterações, ~2,8 h a 5,0 s/iter.
3. Um discriminador de v3.4, junto ao bloco de v3.3. A impressão digital compara só
   `weight`, e nenhum peso muda — o assert é a única trava:

```python
# --- O DISCRIMINADOR DE v3.4 (o BOTAR fecha sem de_pe, spec `g1-limpo-botar-fecha-e-para.md`)
assert cfg.env.commands["alvo_caixa"].sustenta_outros_s == 0.5, \
    "clone anterior à v3.4: `sustenta_outros_s` ainda é 0,3 s"
import inspect
_src = inspect.getsource(CMD.AlvoCaixaCmd._fecha_elo_corrente)
assert "apoiada[m] & de_pe[m]" not in _src, \
    "clone anterior à v3.4: o fecho do BOTAR ainda exige `de_pe` — MEDIDO, ele falha " \
    "em 99,975% dos passos que já satisfazem os outros três (spec §0)"
```

Não mexa na LR: a célula já baixa para 5e-4, e isto é warm-start.

## §5 A contagem

| | antes | depois |
|---|---|---|
| predicados no fecho do BOTAR | 4 | **3** |
| operandos em `parados` | 3 | **2** |
| operandos em `sobe_caixa` | 2 | **1** |
| termos de recompensa | 28 | 28 |
| funções novas | — | **0** |

Duas coisas saem, nenhuma entra. O knob é um número.

## §6 O que este lote NÃO toca

`recompensas.py` (nenhum termo muda), `terminacoes.py`, `curriculo.py`, `env_cfg.py`,
`observacoes.py`, os pesos, o cálculo de `de_pe`, o ramo PEGAR do fecho, o balanceador,
o piso de 30% da locomoção, `postura_ereta` e o gate `_fora_do_botar`.

⚠ **`postura_ereta` NÃO entra.** A fórmula dele é `rampa_pelve × descarga`, e
`descarga = 1 − F_apoio/(m·g)`: com a caixa apoiada a descarga vai a zero e o termo vale
zero, faça o que fizer a pelve. Ele paga por erguer a caixa DA laje — é o inverso da
tarefa. Não o religue "para pagar o levantar"; ele não pode fazer isso.

`smoke.py` **não roda** — decisão do dono. Ele já tem 3 falhas de semântica velha.

## §7 A verificação, e ela é curta

Nenhuma medição prévia. A mudança é subtrativa e a medição que a justifica já está no
§0. Rode UMA vez um script de sanidade no scratchpad: 16 envs, 50 passos, CPU,
`make_env_cfg` padrão — sem exceção, e `env.limpo_twist_zerado` com shape `(16,)` sem
NaN. Nada além disso.

## §8 O risco a observar no primeiro log

`rastreio_por_elo` passa a pagar cheio na cauda pós-BOTAR, e `renda_congelada` já paga
ali. É renda terminal grande. Ela é o incentivo pretendido — concluir a tarefa —, mas se
`Episode_Termination/time_out` subir E `s_C` continuar baixo, o robô pode ter achado um
jeito de entrar na cauda sem concluir. Canais: `Curriculum/forma/s_C` (tem de sair de
0,0000), `Episode_Reward/load`, `Episode_Termination/time_out`, `Curriculum/forma/s_B`
(não pode cair de 0,89).

## §9 Commits

Atômicos, Conventional Commits, mensagem em português, **sem trailer de atribuição**
(`git -c core.hooksPath=/dev/null`).

1. `fix(limpo): o fecho do BOTAR larga o de_pe; a cauda fica parada de pé`
2. `chore(limpo): notebook — RUN bloco14, META 11500 e o discriminador de v3.4`
3. `docs(specs): g1-limpo o BOTAR fecha e para v3.4`

Nunca commite: `docs/handoff/`, `docs/memoria/`, `record_rl.py`, `rl_rollout.npz`,
`g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`,
`reference_checkpoints/*.pt`, `ver_play.py`. Nunca `git push`.
