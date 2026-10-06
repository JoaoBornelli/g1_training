---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "a inércia real da caixa é gravável nos dois caminhos: mj_setConst no MuJoCo clássico, e o dr de massa do mjlab parou de corromper no 1.5.3"
cluster: c3
---
**MuJoCo clássico** (`pilota.py`, `registra_juntas.py`): `m.body_mass` e
`m.body_inertia` são arrays numpy graváveis, e `mujoco.mj_setConst(m, d)` depois é
obrigatório — ele refaz `body_invweight0` e `body_subtreemass`. Entrou a flag
`--massa` no `registra_juntas.py`.

**mjlab**: o `dr.pseudo_inertia` em modo reset **parou de corromper a heap** no 1.5.3
(A/B em CPU, 2026-09-14, 8 envs, 3 voltas de reset, saída 0). O mjlab trocou
`torch.linalg.cholesky` e `eigh` por versões próprias, para não carregar o cuSOLVER.
GPU não verificada.

No `g1_limpo` o `carga_caixa` passa a escrever massa e inércia em vez de aplicar força
externa. A escrita é PRÓPRIA: o `alpha_range` do termo do fabricante é estático, e o
teto da carga vem da célula do NÍVEL. Cubo homogêneo: `I = (2/3)·m·a²`.

**Why:** a força externa endurecia só a estática, e impunha um torque espúrio fora do CoM.
**How to apply:** massa sem inércia junto reproduz o defeito. Escreva as duas.

Relacionado: g1-mjlab-api-armadilhas, g1-limpo-tag-estavel-smoke-vermelho
