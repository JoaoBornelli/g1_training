# Prévia no resume: DR de atrito da caixa + freio de aperto

Spec de 05/10/2026. Aprovada pelo dono para implementação ("sim", 05/10). Pontuação PM: 11/18
(média): coder → revisão. Branch `exp/g1-limpo-v2`. Base: `docs/relatorios/2026-10-05-dr-de-atrito-pega-e-transferencia.md` §1–§3, §6.

## Objetivo

Testar, num resume do `model_24350` (ou `model_23250`), se um preço sobre a força total do
robô na caixa faz a política apertar menos, sem largar a caixa, com o atrito da caixa sorteado
por env. Enunciado atendido: `specs/g1-limpo-comportamento.md` §6.6 (pega que não danifica a
carga) e a transferência ao real (§13).

Medido hoje (`model_24350`): força total robô → caixa 230–250 N com qualquer μ e massa; o
`shoulder_yaw` dos dois braços em ±24–27 N·m (limite 25). O `squeeze` satura e nada cobra o
excesso.

## Mudanças (seis, todas pequenas)

### 1. `priority` na caixa e na mesa — `g1_limpo/cena.py`

`_spec_box` ganha o parâmetro `priority: int = 0` e o passa a `add_geom`. `spec_caixa` chama
com `priority=2`; `spec_mesa` (a laje) com `priority=3`. Efeito: todo contato robô–caixa usa o
μ, o condim e o solref da caixa (o robô tem prioridade 0, os pés 1); o contato caixa–mesa usa
os da mesa. Nada no robô muda.

Verificar: o `exporta_cena.py` reproduz a prioridade no `.mjb` (ele monta a cena pelo mesmo
cfg; confirmar lendo `geom_priority` do modelo exportado no smoke ou num assert).

### 2. DR do μ da caixa por env — `g1_limpo/env_cfg.py`

Evento novo `cfg.events["atrito_caixa"]`, `mode="reset"`, com o termo pronto do mjlab
`mjlab.envs.mdp.dr.geom_friction` (`mjlab/envs/mdp/dr/geom.py:117`): `ranges=(0.6, 1.0)`,
`operation="abs"`, `distribution="uniform"`, `asset_cfg` apontando para a entidade `box` e a
geom `BOX_GEOM`. Coluna 0 só (padrão do termo; a caixa tem condim 3). Os limites da faixa
entram como knob em `knobs.Tarefa` (`atrito_caixa_faixa: tuple[float, float] = (0.6, 1.0)`).

Não publicar μ por env para a recompensa: a referência usa o pior μ da faixa (item 3).

### 3. `squeeze_mu` 0,8 → 0,6 — `g1_limpo/knobs.py:964`

`F_ref = m·g/(2·0,6)` passa a ser a do pior μ da faixa. Atualizar o comentário do knob: o
"μ pessimista" agora é a borda de baixo de `atrito_caixa_faixa`. Se o coder preferir, derivar:
`squeeze_mu` = `atrito_caixa_faixa[0]`, uma fonte de verdade só.

### 4. Sensor caixa × robô por fatias — `g1_limpo/cena.py` (`sensores()`)

```python
ContactSensorCfg(
    name=SENSOR_CAIXA_ROBO,            # constante nova em cena.py
    primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("force", "found"),
    reduce="maxforce",
    num_slots=12,
)
```

`force` no frame do contato: a componente 0 é a normal. A força total é
`Σ_slots |force[..., 0]|`. Não usar `netforce`: as duas palmas apertam em sentidos opostos e
a soma vetorial se cancela. Medido no play: até 8 contatos robô–caixa ao mesmo tempo; 12
fatias dão folga. Verificar na API do mjlab se `mode="geom"` com `entity="box"` resolve a geom
da caixa e se `mode="subtree"` em `pelvis` cobre o robô inteiro (o sensor `auto` já usa isso).

### 5. Freio de aperto — `g1_limpo/recompensas.py` + `env_cfg.py` + `knobs.py`

Termo **separado** `aperto_excessivo`, e não segunda parcela do `squeeze`. Motivo: o
`squeeze` é termo congelável (`renda_congelada` fixa a renda no fecho; `knobs.py:957-962`
avisa que mudar a forma dele muda o valor do fecho às vésperas de um resume). O termo novo
NÃO entra na lista de congeláveis.

```
F_tot  = Σ_slots |force_normal|  do sensor do item 4          [B]
F_ref  = m·g / (2·squeeze_mu)   (reusar `_forca_ref`)         [B]
teto   = 2 · k · F_ref           (dois lados; k = knob `aperto_k` = 3.0)
custo  = relu(F_tot / teto − 1)²  × _fora_do_botar(env, cmd)
```

Peso: knob `Tarefa.aperto_excessivo: float = -0.05`, escalar (sem tabela por estado), gateado
por `_fora_do_botar` como o `squeeze`. Conta no ponto de operação de hoje, 1 kg e μ 0,6:
`F_ref` 8,2 N, teto 49 N, `F_tot` 240 N → razão 4,9 → custo 15,2 → −0,76/s com o peso. Com
5 kg: teto 245 N → custo ≈ 0. O termo não briga com o `terminacao` (−200) nem com o
`caixa_largada`.

Reaproveitar: `_forca_ref`, `_fora_do_botar`. Se existir um helper que lê `env.scene[s].data.force`
com slots (`_forca_das_palmas` lê `[B, slots, 3]`), seguir o mesmo padrão.

### 6. Métrica `forca_total_na_caixa` — `g1_limpo/metricas.py`

Mesma soma `F_tot`, sem peso, pelo padrão de `metricas.termos()`. Média por env nos passos
com `found > 0`. É o número a acompanhar no TensorBoard (esperado: de ~240 N para 50–100 N).

## Smoke (`g1_limpo/smoke.py`) — o dono roda

- §4 sensores: o sensor novo existe, `num_slots == 12`, `reduce == "maxforce"`, campos
  `force` e `found`.
- §5 física: `geom_priority` da caixa == 2, da mesa == 3, do robô == 0 (menos pés == 1).
  Manter o check `impratio == 1.0` (o notebook põe 2,0 em runtime; registrar isso no check).
- §7 eventos: `atrito_caixa` existe, `mode == "reset"`, `ranges == (0.6, 1.0)`.
- §11 recompensa: o termo `aperto_excessivo` existe com o peso do knob; atualizar a contagem
  do título ("mais dezesseis" → "mais dezessete") se ela for afirmada.
- Contrato do freio: com `F_tot` sintético = 240 N, m = 1 kg, μ = 0,6, o custo vale
  `relu(240/49 − 1)²` ≈ 15,2 (teste puro da função, sem env, se o padrão do smoke permitir).

## Notebook — `g1_limpo/kaggle/g1_limpo_zero_colab.ipynb`, célula 8

Acrescentar dois asserts no bloco "o clone é o que eu penso que é?": `"atrito_caixa" in
cfg.env.events` e `cfg.env.rewards["aperto_excessivo"].weight < 0`. Nada mais muda na célula.

## Restrições

- `CLAUDE.md` do repo: contagem de linhas antes → depois por arquivo no relatório final.
  Antes: cena 464, env_cfg 975, knobs 1611, recompensas 1369, metricas 574, smoke 6296.
- Sem acoplamento novo entre módulos; o sensor e as constantes seguem o padrão de `cena.py`.
- Nenhum termo existente muda de forma além do `squeeze_mu`.
- O checkpoint `model_24350` precisa continuar carregável: a observação não muda (nenhum
  canal novo), e o `runner.py` não precisa de chave nova.
- Não rodar treino, env nem smoke: o dono roda. Pode rodar `uv run python -c` para checar
  assinaturas do mjlab.
- Sem commit: o dono comita.

## Fora de escopo

Recompensa do resultado da pega, orientação de pé só no BOTAR, DR da rigidez da caixa, faixa
de μ abaixo de 0,6. Tudo isso é v3 (`g1-v3-palma-de-plastico-5kg`).

## Entrega esperada do coder

Lista de arquivos tocados com linhas antes → depois; o `diff` resumido; o que foi verificado
na API do mjlab (assinaturas de `geom_friction`, `ContactMatch`, prioridade no `add_geom`); o
que ficou para o dono testar no smoke.
