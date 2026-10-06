---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
---
# g1-multitask-caixa-fora-de-alcance (original)

Corpo original da memória `g1-multitask-caixa-fora-de-alcance`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

**Confirmado 2026-08-10 (play do model_7299 + evidência em código).** A caixa do
multitask nasce fora do alcance dos braços na maioria dos episódios.

**A recombinação errada** (`g1_multitask/knobs.py::Scene`):
- `box_xy = (0.50, 0.0)` — é o DEFAULT de `skills/lift/knobs.py`, que a Lift
  **abandonou em 16/07** (`c2026_07_16_box_edge.py`): mesmo sintoma reportado pelo
  user ("espacato" pra alcançar), investigação concluiu que com a caixa no centro da
  mesa "a pose correta fisicamente NÃO alcança lá" → moveu pra borda, `box_xy=0.30`.
- `box_jitter_x = (0.0, 0.20)` — herdado do `c2026_07_16_generalize.py`, que foi
  calibrado EM CIMA do 0.30: "+0.20 é o limite prático de alcance (0.50 era o 'longe
  demais' original)". O multitask herdou o jitter mas não o nominal.
- S12 somou `table_jitter_xy = ±0.15` (delta compartilhado mesa+caixa).

**Distribuições:** Lift comprovada: caixa x ∈ [0.30, 0.50]. Multitask: x ∈ [0.35,
0.85] — o sample mais próximo do nominal (0.50) é o ponto documentado como
inalcançável. Robô nasce em x ∈ (−0.10, 0).

**O clique quantitativo:** P(caixa ≤ 0.50) ≈ 0.19 com esses uniformes; o
`cond_fisica` do `reorientar` no bloco 1 mediu **0.17** — o teto dele era a fração
alcançável. `pegar` (grasp bimanual sustentado, mais exigente) deu 0.0000.
O `erro_angulo_deg = 4.26` global era diluído por share de tarefa (~⅓); real ≈ 12–13°.

**Fix APLICADO 10/08** (`knobs.py`: `box_xy = (0.30, 0.0)`, `table_jitter_xy` 0.15 →
0.05; hotfix por célula de patch no Kaggle enquanto o push não chega no branch
`exp/locomocao-velocidade`). **Validado no bloco 2 em tempo real:** grasp 0 → 0.25,
lift 0 → 0.32, box_at_peito 0 → 0.19 (caixa chegando NO PEITO), `contrib/pegar/_total`
cruzou o zero. Bloqueio restante do currículo é a régua `tol_w = 0.35` (erro angular
global 0.89, sucesso ≡ 0 nas 3 tarefas); decisão de régua pendente dos números por
tarefa (`contrib/<tarefa>/erro_vel_*`, implementados 10/08). Efeito colateral da
borda: `caixa_caiu` ~0.4-0.5 no reorientar (derruba a caixa da beirada — sinal de
aprendizado, não bug).

**Segundo defeito da mesma classe (10/08, confirmado no play do model_12299):** o
robô segurava a caixa AGACHADO — `box_at_peito = 0.51` com `cond_fisica = 0.0000`
exato. Causa: `alvo_peito_w` era 100% frame da base → o alvo descia com a pelve e
agachar ENCURTAVA o percurso do `lift_ao_peito` (a linha de chegada seguia a pelve
pra baixo). Nada no reward pedia ficar de pé; só a régua (`de_pé`) pedia, sem
gradiente. Conserto APLICADO 10/08: `alvo_peito_w` com **xy na base, z ancorado no
mundo** em `pelve_de_pé(KNEES_BENT 0.76) + 0.15 = 0.91 m`; o `no_peito` da régua
usa a MESMA função (mudança de régua na janela grátis: perf ≡ 0). Vale pro
`locomover_carregando` também (decisão do user: não pode agachar com caixa em
loco-manipulação). Premissa declarada: chão plano.

**Janela de régua FECHADA 11/08 (fim do bloco 3):** `tol_w` 0.35 → **0.70** — piso
real medido por tarefa 0.62–0.73, uniforme com/sem comando (oscilação de marcha +
push dentro da integral; 0.35 selecionava andar arrastado). 0.70 = 1× o std do
kernel do fabricante; é guarda de sanidade, não teste de rastreio — quem vigia
rastreio é o `erro_vel_ang_filt` novo (EMA 0.5 s no erro assinado, só-log). User
temia afrouxar; resposta: tol não tem gradiente, movimento é moldado pelo reward.
Aceitação: perf_n0 do locomover descola em ~200 iters, senão degrau 0.80.
**VALIDADO no fim da run 22296:** `sucesso` 0 → **0,2884** (primeira vez fora do zero em
todo o experimento), `locomover_velocidade/perf_n0` 0 → **0,9143**, nível 1 aberto com
perf 0,68. E o `erro_vel_ang_filt` provou que 0,70 NÃO é régua frouxa: filtrado dá
locomover 0,230 · pegar 0,111 · reorientar 0,076, contra o cru de 0,56–0,61. Ou seja o
piso cru era ~80% oscilação de guinada da marcha + push, não falha de rastreio.
**+ `box_at_peito` dupla escala (0.30/0.05):** com a âncora em mundo o std único
valia e⁻²⁵ a 25 cm → `pegar` acomodou em "mãos sem erguer" (lift 0.02,
std_vantagem 0.18→0.075). Mesmo conserto do `botar_std_grosso`.

Ver g1-multitask-spec-validacao-2026-08, g1-lift-box-task.
