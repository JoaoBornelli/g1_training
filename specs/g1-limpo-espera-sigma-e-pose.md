# Spec — F1: σ medido no fim da espera. F2: pose de braço com raio largo

Data: 2026-09-08. Repo `g1_training`, branch `exp/g1-limpo-v2`. Só a pasta `g1_limpo/`.

Aprovado pelo dono em 2026-09-08. Duas mudanças, independentes entre si.

## 0. Regras para quem implementa

- Python é `.venv/bin/python`. O portão é `cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -30`, timeout 600000 ms.
- **O smoke completo leva ~4,5 min. Rode-o UMA VEZ, no fim.** Durante o desenvolvimento valide com script pontual no scratchpad, montando UM env sintético de 16 envs. Base conhecida: 561 ok, com 1 falha intermitente pré-existente na seção 19 ("topo do BOTAR"), que depende de sorteio sem seed e não é sua.
- `g1_limpo/` NUNCA importa `g1_training`, `g1_poc` nem `g1_multitask`. Import entre módulos do próprio `g1_limpo` é permitido.
- Estilo idêntico ao do módulo: docstring em português, um `⚠` por fato medido ou armadilha, sem comentário decorativo, sem emoji.
- TDD: escreva o check no smoke primeiro, veja falhar, implemente, veja passar.
- Simplicidade: EXATAMENTE o que esta spec lista. Nenhum termo, knob ou buffer fora dela.
- Git: NÃO commitar, NÃO stash/reset/checkout. Não tocar em untracked (`docs/handoff/`, `record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`, `*.ipynb`, `reference_checkpoints/`, `ver_play.py`).
- Se algo for impossível ou contradizer o código real, não invente: implemente o resto e descreva o impasse com arquivo:linha.

## 1. F1 — o σ é a distância inicial DA TAREFA, não do reset

### O defeito

`comando.py:731-742`, na passada do `_pendente`, chama `_recalcula_sigmas(pend)` no PRIMEIRO passo depois do reset. A janela de espera (`espera_s = (0,5; 1,5)`) só termina 0,5 a 1,5 s depois, e é ela que acende o `VALIDA`.

O docstring de `recompensas._alcancar` afirma:

> "No passo em que o elo abre ele vale `exp(−1) = 0,368` por construção, porque `σ = d₀`."

A afirmação é falsa hoje. O robô aproxima as mãos durante a espera, de graça: o `VALIDA` é zero, portanto os oito termos de manipulação pagam zero, e o `pose` do molde é canal morto com os braços fora. No instante em que a tarefa liga, `d_palma < σ` e o `alcancar` nasce acima de 0,368.

MEDIDO no `play` do `bloco9` em 2026-09-08 (observação do dono): o robô assume pose de pré-pega durante a espera. Exemplo numérico: σ fixado em 0,34 m no reset, mão a 0,20 m no fim da espera, `alcancar = exp(−(0,20/0,34)²) = 0,71` em vez de 0,368.

### A mudança

`comando.py`:

- Novo buffer `self._sigma_pendente` (bool, por env), alocado no `__init__` ao lado do `_pendente`.
- `_resample_command`: marcar `self._sigma_pendente[env_ids] = True`, junto do `self._pendente[env_ids] = True`.
- `_update_command`, passada do `_pendente`: **remover** de lá o `_recalcula_sigmas(pend)` e o `self._pos_no_elo[pend] = ...`. O `_aplica_elo(pend, so_pose=True)` FICA — a pose e o alvo continuam sendo refeitos no primeiro passo, que é o defeito que o `_pendente` existe para consertar.
- `_aplica_espera`, no FIM (depois de escrever `_command[:, VALIDA]`): para os envs com `_sigma_pendente` E `VALIDA > 0,5`, chamar `_recalcula_sigmas(ids)`, escrever `_pos_no_elo[ids]`, e baixar `_sigma_pendente[ids]`.

Ordem: `_update_command` chama `_aplica_espera()` ANTES de `_avanca_elo()`. Portanto o σ está fresco quando a lógica de avanço roda, no mesmo passo. Confirmado em `comando.py:726-745`.

O `⚠` do docstring do `_recalcula_sigmas` e o do `_alcancar` passam a descrever o comportamento novo, com o número medido acima.

### O que NÃO muda, e por quê

- **`_congela_face` fica onde está** (dentro do `_aplica_elo`, no reset). A caixa está em repouso na laje durante a espera e o robô não a toca — `VALIDA` é zero e nenhum termo paga por tocá-la. Portanto a normal da face é a mesma no reset e no fim da espera. O check 4 do smoke instrumenta isso: se o robô passar a empurrar a caixa na espera, ele acusa.
- O `σ` do avanço de elo (`_avanca_elo_force`) fica como está: ali não existe espera.
- O `ANDAR` puro (locomoção) tem `espera = 0`, portanto o `VALIDA` nunca acende e o `_sigma_pendente` fica pendente o episódio todo. Isso é correto e inofensivo: nenhum termo de manipulação lê σ com `VALIDA = 0`. Não acrescente ramo para isso.

### Smoke do F1

Seção nova `--- F1: o σ é a distância da TAREFA`, num env sintético `elo=PEGAR`, 16 envs:

1. No primeiro passo depois do reset, com a espera correndo, `_sigma_pendente` é verdadeiro em todos os envs e `_pendente` é falso.
2. Teleporte a caixa para MAIS PERTO das palmas durante a espera (escreva a pose com `write_root_link_pose_to_sim`), passe a janela, e afirme que `sigma_alcance` bate com `dist_palma_caixa` medido no passo em que o `VALIDA` acendeu, com tolerância de 1e−3. Sem o conserto ele bateria com a distância do reset.
3. No passo em que o `VALIDA` acende, `_alcancar` vale `exp(−1) = 0,3679`, com tolerância de 0,01 — a invariante do docstring, agora verdadeira mesmo com a caixa movida.
4. No fim da espera, o `ANG` publicado é `< 0,05 rad` em todos os envs: a caixa não gira sozinha na espera, portanto o `_congela_face` no reset segue válido. (Se este check falhar no futuro, o `_congela_face` tem de mudar junto.)
5. `_pos_no_elo` no fim da espera bate com a pose da base daquele instante, e não com a do reset.

## 2. F2 — `pose_de_braco`: o macro que falta

### O defeito

Durante as DUAS janelas de espera (a inicial e a final) o robô deve ficar na pose padrão. O único termo que pede isso é o `pose` do molde, peso 1,0, e ele é **canal morto**.

A tabela está medida dentro do docstring de `recompensas.PosturaPorElo`: com os braços a 10% da faixa de junta o termo vale `0,000`, com **derivada zero**. Aproximar ou afastar os braços não muda nada. Nenhuma força os traz de volta.

Isso é o buraco H5 da auditoria (`docs/planos/2026-09-04-auditoria-gradientes-g1-limpo.md`), que ficou aberto, mais a observação do dono de 2026-09-08 sobre a espera inicial. **Um termo fecha os dois.**

### A mudança

`cena.py`: constante nova, no `__all__`.

```python
# ⚠ AS 14 JUNTAS DE BRAÇO, por padrão de nome. A fonte é o `G1_ACTION_SCALE` do
# fabricante: sete padrões por lado. O tronco (`waist_*`) NÃO entra — ele participa da
# marcha e da pega, e a pose dele já é assunto do `pose` do molde.
JUNTAS_BRACO = (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint", ".*_elbow_joint",
                ".*_wrist_roll_joint", ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint")
```

`recompensas.py`: classe nova, e ela entra no `__all__`.

```python
class pose_de_braco:
    """`exp(−(rms(Δq_braços)/σ)²)`, ativo só onde o `VALIDA` é ZERO.

    ⚠ ELE É O MACRO DE UM PAR, e o `pose` do molde é o preciso. A tabela medida no
    `PosturaPorElo` deste arquivo mostra que o `pose` vale 0,000 com derivada ZERO a
    10% da faixa de junta: com os braços fora ele não puxa nada. Este termo tem σ
    LARGO e puxa em toda a faixa.

    ⚠ AS DUAS JANELAS, e é isso que o `VALIDA = 0` seleciona: a espera inicial, antes
    de a tarefa existir, e a espera final, depois de largar a caixa. Nas duas o robô
    deve estar na pose padrão. Com a tarefa ativa ele TEM de sair dela, portanto o
    termo cala. No `ANDAR` de locomoção o `VALIDA` também é zero, e ali a pose padrão
    de braço é a certa.

    ⚠ MÉDIA sobre as juntas, e não produto por junta. O produto de 17 gaussianas
    colapsa para qualquer σ — é a medição que aposentou o `variable_posture` na
    manipulação. Ver a tabela no `PosturaPorElo`.
    """
```

Corpo: RMS de `(joint_pos − default_joint_pos)` sobre as juntas resolvidas, kernel gaussiano com `sigma`, multiplicado por `(1 − _valida(env, nome_do_comando))`.

Resolva os ids das juntas no `__init__` com `find_joints`, e guarde o `default_joint_pos` daquelas colunas. O `SceneEntityCfg` **tem de viver em `params`** (`manager_base.py:141-145` só resolve os que estão lá) — siga o idioma do `squeeze` e do `unload` deste arquivo.

`knobs.py`, classe `Tarefa`:

```python
# ⚠ σ LARGO, e o número sai da tabela medida no `PosturaPorElo`: a faixa média das 17
# juntas de manipulação é 3,77 rad, e nem `running×5` sobrevive a 40% dela. Com
# σ = 1,0 rad o termo vale 0,37 a 1 rad de excursão e 0,02 a 2 rad — vivo nos dois.
pose_de_braco: float = 1.0
pose_de_braco_sigma: float = 1.0
```

`env_cfg.py`, no fim da seção 3g, **antes** do `renda_congelada`:

```python
cfg.rewards["pose_de_braco"] = RewardTermCfg(
    func=RC.pose_de_braco, weight=tr.pose_de_braco,
    params={"nome_do_comando": _cmd, "sigma": tr.pose_de_braco_sigma,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(C.JUNTAS_BRACO))})
```

⚠ **Ele NÃO entra em `TERMOS_CONGELAVEIS`.** Ele não depende do elo; ele depende do `VALIDA`. Congelá-lo pagaria por uma coisa que não muda na troca de elo.

⚠ O `renda_congelada` continua sendo o ÚLTIMO termo do dict. O smoke já afirma isso; não quebre.

`ARQUITETURA.md`: acrescentar o termo à tabela dos incentivos (perto da linha 1556) e à seção 3i (perto da linha 310). Edição mínima.

### Smoke do F2

Na mesma seção nova:

6. `list(cfg.rewards)[-1] == "renda_congelada"` continua verdadeiro, e `"pose_de_braco"` está em `cfg.rewards`.
7. `"pose_de_braco" not in TERMOS_CONGELAVEIS`.
8. O termo resolve 14 juntas (`len(asset_cfg.joint_ids) == 14`), e nenhuma delas é `waist_*`.
9. Fórmula, sem simulador: `exp(−(1,0/σ)²)` com o knob dá 0,3679, e `exp(−(2,0/σ)²)` dá 0,0183. Afirme que o termo é `> 0,3` a 1 rad de excursão — o `pose` do molde vale 0,000 ali.
10. Num env sintético `elo=PEGAR`, 16 envs: durante a espera (`VALIDA = 0`) o termo é `> 0`; depois de `_passa_janela` (`VALIDA = 1`) ele é `0` exato.
11. Num env `elo=ANDAR` de locomoção: o termo é `> 0` (o `VALIDA` é zero ali, e a pose padrão de braço é a certa).

## 3. Lote de revisão

Depois do coder, revisar o `git diff` contra esta spec:

- O `_recalcula_sigmas` e o `_pos_no_elo` saíram da passada do `_pendente` e só rodam quando o `VALIDA` acende. O `_aplica_elo(pend, so_pose=True)` FICOU.
- O `_sigma_pendente` é marcado no `_resample_command` e baixado uma única vez por episódio.
- O `_congela_face` não se moveu, e o check 4 o instrumenta.
- `pose_de_braco` usa média (não produto), tem `SceneEntityCfg` em `params`, resolve 14 juntas sem `waist_*`, e é gateado por `1 − VALIDA`.
- `pose_de_braco` não está em `TERMOS_CONGELAVEIS`; `renda_congelada` é o último termo.
- Smoke verde, fora a falha intermitente conhecida da seção 19. Nenhum check antigo apagado sem substituto.
- Docstrings no estilo do módulo. Nenhum comentário decorativo.

Saída: APROVADO, ou lista de correções com arquivo e linha.
