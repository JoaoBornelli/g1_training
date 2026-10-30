# Contrato da fase H (Parte 4, refino): rascunho de 30/10

Worktree: `/home/joaobornelli/Documents/g1_v3_impl`, branch `g1_v3`, HEAD `1451ded`. Rascunho só de leitura:
nenhum arquivo do repo mudou. "Plano" = `docs/planos/2026-10-14-implementacao-g1-v3.md`; "S" = `specs/g1-v3.md`.

Fontes lidas: plano §6.2 a §6.9, §12.4, PD-24 a PD-50, §16, §20 (todas as linhas de 29 e 30/10); S §6.5, §8.2,
§9.2, §10, §11.5, §12, §13, §14; o código de `g1_v3/` (potencial, recompensas, terminacoes, eventos, estimador,
curriculo, balanceador, knobs, metricas, env_cfg, cena, observacoes, comando, runner); `g1_limpo/terminacoes.py:158-196`
e `g1_limpo/eventos.py:256-301`; o mjlab 1.5.3 (`envs/mdp/dr/actuator.py`, `dr/body.py`, `envs/mdp/events.py`,
`managers/event_manager.py`, `sensor/contact_sensor.py`, `entity/entity.py`, `envs/manager_based_rl_env.py`); o
rsl_rl 5.4.0 (`runners/on_policy_runner.py`, `utils/logger.py`); o relatório da Parte 0 (`2026-10-29-v3-parte0.md`
D6–D8, §6) e o `item15_aperto.py`; `docs/relatorios/2026-10-05-dr-para-a-v3.md` §3.5–§5; no vault,
`g1-limpo-batente-e-restricao-mole` e `g1-limpo-std-da-acao-por-junta`.

As decisões abertas estão na seção 4 (problemas P1 a P24). Onde um núcleo depende de uma delas, o bloco diz
"conforme P<n>".

---

## 1. Fase H: núcleos puros

Assinaturas que a lane H0 testa e as lanes H1 a H6 implementam. Índices de alvo: P0..P4 = 0..4, B1..B4 = 5..8,
−1 = nenhum (`knobs.ALVOS`). Todo núcleo devolve 0 ou False com o índice −1 ou fora da máscara, nunca NaN (§6.6).
Os invólucros aplicam a máscara com `torch.where` (`recompensas._fim`, `curriculo.mascara`), nunca multiplicando.
Dois degraus distintos (§6.9):

- **degrau da tarefa** `d` = `env.v3_degrau` (E,) long. O `comando.cadeia` escreve na abertura do trecho
  (`comando.py:446-448`). As regras de termo e a terminação `batente` usam esse degrau.
- **degrau global** `g` = o menor degrau entre as 4 tarefas da cadeia (PD-41), um int para o env inteiro. Toda a DR do
  refino e o estimador usam esse degrau. Hoje ele não existe em lugar nenhum (P5).

```python
# balanceador.py (H1). Módulo puro; a fórmula da §6.9 fica num lugar só
def no_degrau(ini, fim, d, degraus: int):  # float ou tensor (E,), na unidade de ini/fim
    # ini + (fim − ini)·d/degraus. d é int (o degrau global) ou tensor float (E,) (o `v3_degrau`)
class Curriculo:
    def degrau_global(self) -> int: ...    # min(self._degrau[t] for t in TAREFAS) (PD-41); 0..5
    def dr(self) -> dict: ...              # o de hoje (balanceador.py:252-272), mais as chaves do refino:
        # "degrau_global": int = g;
        # "pd": [lo, hi] = [no_degrau(1, DR.pd[0], g, 5), no_degrau(1, DR.pd[1], g, 5)]    (escala; g=0 → [1, 1])
        # "forca_motor": idem com DR.forca_motor; "forca_motor_bracos": idem com DR.forca_motor_bracos
        # "solref": [no_degrau(solref_base, solref[0], g, 5), no_degrau(solref_base, solref[1], g, 5)]   (s, solref[0])
        # "rfi": float = 0 antes da trava "subida"; depois no_degrau(rfi[0], rfi[1], g, 5)  (fração de tau_max)
        # "atraso_max": int = atraso_refino[1] se g ≥ degrau_atraso_refino; senão o de hoje (portão 2, início 1)
        # "estimador": float = g/5   (a fração da magnitude "início → refino", H5)
        # Só tipos do Python. Nada entra no `estado()`: tudo deriva dos degraus e das travas, que já entram.
        # "Nunca desce" (PD-24) vale porque os degraus nunca descem.

# curriculo.py (H1): o `valor_do_degrau` do §12.4 JÁ EXISTE como `recompensas.por_degrau(env, ini, fim)`
# (recompensas.py:391-394), com 5 chamadas: `incl_lim` (:399), o pouso do `_phi_env` (:471), o fator do B2 no
# `progresso` (:736), o `contato_laje` (:749) e a `terminacoes.chegada` (:133). Duas saídas (P1):
#  (A) fica o `por_degrau`, passando a chamar `balanceador.no_degrau`; não nasce `valor_do_degrau`; saem os campos
#      mortos `Refino.regras`, `fator_por_degrau` e `tarefas_fator` (hoje só o `test_knobs` os lê).
#  (B) `curriculo.valor_do_degrau(env, regra: str) -> (E,)` resolve `Refino.regras` (a mesma conta do
#      `test_knobs.test_dr_e_refino`); com a regra em `Refino.fator_por_degrau`, o par vale (0, 1), isto é d/5;
#      devolve no_degrau(ini, fim, env.v3_degrau.float(), degraus). O `por_degrau` SAI, e as 5 chamadas passam a
#      usar o nome da regra (`valor_do_degrau(env, "sigma_pouso")`). Fora das tarefas da regra vale o mesmo número:
#      quem chama já mascara. Sem NaN: o `v3_degrau` é long ≥ 0.
```

```python
# recompensas.py (H2)
def n_sacudida(dv_lin, dw, dt, norma_lin, norma_ang) -> torch.Tensor: ...
    # (E,) adimensional. dv_lin, dw (E,3) = velocidade linear (m/s) e angular (rad/s) da caixa no mundo, menos as do
    # passo anterior; dt = env.step_dt (s). a = ‖dv_lin‖/dt; α = ‖dw‖/dt; custo = (a/norma_lin)² + (α/norma_ang)²
    # (PD-34). norma = (2,9; 17,0), que hoje não está no knobs (P10). ⚠ Sem raiz, sem divisão por tensor: sem NaN.
def forcas_por_face(pos_w, normal_w, forca_w, achado, caixa_w, R_caixa) -> tuple[torch.Tensor, torch.Tensor]: ...
    # (F_esq (E,), F_dir (E,)) em N. Entradas do sensor `caixa_faces` (cena.py:399-404; reduce="none", 12 fatias,
    # global_frame): pos_w, normal_w, forca_w (E,S,3); achado (E,S) bool (= found > 0); caixa_w (E,3); R_caixa (E,3,3)
    # = alvos.frame_da_caixa(quat, rumo) (PD-29, colunas x, y, z). p_c = R_caixaᵀ·(pos_w − caixa_w); eixo =
    # argmax_i |p_c_i| (no cubo, dividir por `a` não muda o argmax); F_n = |⟨forca_w, normal_w⟩| (a componente normal:
    # o sensor gira o vetor do frame do contato, cujo 1º eixo é a normal); F_esq = Σ F_n nas fatias com achado,
    # eixo == 1 e p_c_y > 0; F_dir: idem com p_c_y < 0. Faces ±x (a frente e o fundo, onde o tronco apoia) e ±z
    # não entram (S §8.2). Fatia vazia: torch.where com 0.
# O aperto é `potencial.custo_aperto(F_esq, F_dir, R.F_ref)` (potencial.py:69-75): o núcleo já existe, sem cópia.

# invólucros (H2). "Caixa presa" = `_em(v3_alvo_idx, _CAIXA_ALVO)`: o grupo (P3, P4, B1, B2) já existe
# (recompensas.py:56). No CARREGAR o índice fica no P4 (E11), então o mesmo grupo cobre o CARREGAR. Conforme P11,
# talvez também `& ~apoio`.
def sacudida(env, tarefas): ...
    # dv = box.root_link_vel_w (E,6) − a velocidade do passo anterior (P9: de onde ela vem); valor =
    # n_sacudida(dv[:, :3], dv[:, 3:], env.step_dt, *R.sacudida_norma) · fator; fator = d/5 (o `por_degrau(env, 0, 1)`
    # de hoje, ou `valor_do_degrau(env, "sacudida")`); torch.where(presa, valor, 0); _fim(env, tarefas, ·).
    # Sem `paga`: o termo é custo. O peso do cfg vem conforme P2.
def aperto(env, tarefas): ...
    # s = env.scene["caixa_faces"].data; R = alvos.frame_da_caixa(box quat, estimador.yaw(base quat)) (a mesma conta
    # de `_phi_env`, recompensas.py:463); F_esq, F_dir = forcas_por_face(s.pos, s.normal, s.force, s.found > 0, caixa_w,
    # R); valor = potencial.custo_aperto(F_esq, F_dir, R.F_ref) · fator (d/5); torch.where(presa, valor, 0); _fim(·).
```

```python
# terminacoes.py (H3). Sai a classe `NoBatente` (g1_limpo/terminacoes.py:158-196); entram núcleo e invólucro, como os
# outros termos da v3. A régua é a do NoBatente: frac = |q − centro|/meio, com os limites duros do modelo
# (`robo.data.joint_pos_limits`, os mesmos do `limite_junta`, recompensas.py:600-604).
def n_batente(q, centro, meio, frac_max, q_hip_yaw, hip_yaw_max, degrau, degrau_min) -> torch.Tensor: ...
    # (E,) bool. q (E,J) rad, só as juntas vigiadas (conforme P13); centro, meio (J,); q_hip_yaw (E,2) rad;
    # degrau (E,) long = v3_degrau. = ((|q − centro|/meio > frac_max).any(−1) | (|q_hip_yaw| > hip_yaw_max).any(−1))
    #   & (degrau ≥ degrau_min). O "e" da S §10 ("junta no batente e hip_yaw acima de 1 rad") é lido como dois gatilhos
    #   (OU). `>` estrito, como o NoBatente; conforme P14.
def batente(env, tarefas): ...
    # Knobs: Terminacao.batente_frac (1,0), Terminacao.hip_yaw_max (1,0 rad), Refino.degrau_batente (1). Os índices
    # das juntas saem de JUNTAS no import (o padrão de `_YAW_E`, recompensas.py:39). torch.where(mascara(env, tarefas),
    # ·, False). Vale nas 4 tarefas (§6.9: ✓ em todas).
```

```python
# eventos.py (H4). A faixa de cada evento sai de `env.v3_cur.dr()` (H1): o evento só aplica. É o padrão do `atrito_pad`
# (eventos.py:430-441), que já embrulha um `dr.*` do mjlab com a faixa do currículo.
# ⚠ O invólucro que chama um `dr.*` tem de levar o MESMO `@requires_model_fields` da função do mjlab. O EventManager só
#   expande por mundo os campos do `term_cfg.func` (event_manager.py:383-386). O `atrito_pad` funciona sem o decorador
#   porque o `atrito_pe` (startup, `dr.geom_friction` direto no cfg) já registra o `geom_friction`. Para os campos dos
#   atuadores, nenhum outro termo registra: sem o decorador, o índice de mundo estoura.
@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd(env, env_ids, *, asset_cfg): ...
    # lo, hi = dr()["pd"]; dr.pd_gains(env, env_ids, kp_range=(lo, hi), kd_range=(lo, hi), asset_cfg=asset_cfg,
    # operation="scale"). Escala sobre o default do modelo C (`get_default_field`), kp e kd sorteados à parte, por junta.
@requires_model_fields("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
def forca_motor(env, env_ids, *, asset_cfg_pernas, asset_cfg_bracos): ...
    # duas chamadas de dr.effort_limits(..., operation="scale"): pernas e cintura com dr()["forca_motor"], braços com
    # dr()["forca_motor_bracos"]. ⚠ Seleção dos braços: P18.
@requires_model_fields("geom_solref")
def solref(env, env_ids, *, asset_cfg_caixa, asset_cfg_chao): ...
    # u = U[lo, hi] = dr()["solref"], UM valor por env no v3_gerador, escrito em
    # env.sim.model.geom_solref[env_grid, geom_grid, 0] das geoms dos dois (o padrão do `tamanho_caixa` da v3,
    # eventos.py:382-406; o `g1_limpo/eventos.py:256-301` citado no plano é o mesmo `tamanho_caixa`, e não DR de
    # solref). O mjlab não tem DR de `geom_solref`. O campo é por mundo no mujoco_warp (`types.py:1593`,
    # array("*", "ngeom")). ⚠ Quais geoms de fato mudam o contato: P17.
@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def com_caixa(env, env_ids, *, asset_cfg): ...
    # fx, fz = dr()["com_caixa"] (fração da meia aresta); off = (U±fx·a, U±fx·a, U±fz·a) com a = v3_meia[env_ids], no
    # v3_gerador; body_ipos[env, caixa] = default + off. Entra no cfg só com `DR.com_caixa` = True (P19).
class forca_caixa(mdp.apply_body_impulse): ...   # mode="step" (P4); conforme P3
    # Depois do super().__call__ (que sorteia e escreve o xfrc da caixa), zera o wrench dos envs sem
    # (v3_tarefa == CARREGAR) & v3_pegou & ~v3_perdeu & dr()["forca_caixa"]. ⚠ Sem a máscara, a força empurra a caixa
    # estacionada a 20 m (ANDAR) para fora da laje e derruba a caixa da laje no P0–P2 (perda por `fora_da_laje`).
def rfi(env, env_ids): ...   # mode="step" (P4)
    # s = dr()["rfi"]; τ = (2u − 1)·s·tau_max (E,29) N·m, u no v3_gerador; env.sim.data.qfrc_applied[:,
    # robo.indexing.joint_v_adr] = τ (entity.py:1216-1232). Com s = 0 escreve zero. O mjlab não tem RFI
    # (`qfrc_applied` só aparece no viewer). tau_max: P20.
# O atraso 0–3 não pede código novo no eventos: o `curriculo.v3` já chama `eventos.atraso` quando
# `dr()["atraso_max"]` sai do teto inicial (curriculo.py:116-118). Basta o H1 pôr o 3 no dr(). A gravidade sai (PD-40).
```

```python
# H5. Três arquivos: estimador.py (puro), observacoes.py e comando.py (invólucros)
# observacoes.py: `_magnitude(k, i)` (observacoes.py:72-79, índice 0/1) vira `_magnitude(k, f)` com f = dr()["estimador"]
# = g/5 ∈ [0, 1], interpolado por campo (§6.9): sigma_pos = k.sigma_pos[0]·(1 − f) (m); sigma_pos_rel =
# k.sigma_pos[1]·f (fração da distância); sigma_ori_graus, perda, vies_meia, vies_pos (P21) = no_degrau(par[0],
# par[1], f·5, 5). Sai a constante `_MAGNITUDE_INICIO` (:82); `_calcula_caixa` monta a magnitude a cada passo.
# O canal `forca_caixa` do crítico (observacoes.py:305-311, hoje zero) lê
# ‖env.scene["box"].data.body_external_wrench[:, 0, :3]‖/100: sem buffer novo.
# comando.py: `CadeiaCmd.deriva` (comando.py:266-268, fixo no índice 0) passa a sair a cada passo de dr()["estimador"]:
# {"odo", "rumo", "pos_giro", "rumo_giro"} = no_degrau(k.deriva_*[0], k.deriva_*[1], f·5, 5). `_abre_botar`
# (:470-485): soltura += viés por episódio (xy U±vies_alvo_botar_xy, z U±vies_alvo_botar_z, interpolados por f;
# conforme P22), antes de gravar `v3_alvo_w`.
class EstimadorCaixa:   # estimador.py (puro)
    # reinicia(ids): sorteia também _z_pos (E,3) e o viés de orientação do episódio (eixo e ângulo, conforme P21).
    # passo(...): pos += _z_pos·vies_pos (viés) + σ·N(0,1) (tremor, como hoje); ori = rotvec(viés do episódio) ∘
    #   rotvec(N(0, σ_leitura)) (viés E tremor; hoje é OU, pelo `ori_persistente`); perda isolada (como hoje) OU
    #   rajada: estado `_rajada_ate` (E,) s; conforme P21, uma rajada abre com taxa λ por s e dura U[0,3; 1,0] s; amostra
    #   com t < _rajada_ate é inválida. Novas chaves de `magnitude`: "vies_pos" (m), "rajada_taxa" (1/s); ausente vale 0.
```

```python
# H6. runner.py e curriculo.py
# runner.py: `RunnerV3` grava o std por junta. `self.alg.get_policy().output_std` é (B,29) (rsl_rl
# `mlp_model.py:139`) e o logger do rsl_rl só grava a média (`logger.py:211`, `Policy/mean_std`). O RunnerV3 embrulha
# o `self.logger.log` (ou repete a chamada) e grava `Policy/std/<junta>` = output_std.mean(0)[j], com os nomes de
# `env.unwrapped.scene["robot"].joint_names` (o runner não importa código do projeto, runner.py:3).
# curriculo.py, termo `v3`: as chaves da §6.7 que faltam: `atraso_max`, `mu_min_pad` e `degrau_global`. O "degrau
# por tarefa" do H6 JÁ ESTÁ no log (`degrau_<tarefa>`, curriculo.py:125).
# metricas.py (opcional, P23): `aceleracao_da_caixa`, o custo da sacudida SEM o fator, nos passos com a caixa presa.
```

---

## 2. Linhas novas das §6.4 e §6.5

**§6.4 (buffers).** Nenhum buffer novo é obrigatório. O degrau global sai de `env.v3_cur` (H1); o viés do alvo do BOTAR
é sorteado uma vez, na abertura do BOTAR; os vieses do estimador moram no objeto `v3_estimador`; a força na caixa sai
do `body_external_wrench`. A única exceção é a da sacudida (P9):

| buffer | forma | escrito por | lido por |
|---|---|---|---|
| `v3_v_caixa_ant` (substitui o `v3_vz_caixa_ant`, só na opção P9-a) | (E,6) [lin xyz, ang xyz] da caixa no fim do `compute` da cadeia, todo passo, todo env | `comando.cadeia` | `recompensas.sacudida`, `terminacoes.chegada` e `metricas.velocidade_de_chegada` (coluna 2) |

**§6.5 (termos), na ordem do dict.** A lista de nomes da §6.5 e o `test_cfg` (test_cfg.py:40-50) já trazem todos.
O que falta é a linha de cada um no cfg:

| bloco | nome | onde entra | tarefas | peso ou modo | fonte |
|---|---|---|---|---|---|
| recompensa | `sacudida` | depois do `progresso`, antes do `perda_caixa` | PEGAR, CARREGAR, BOTAR | −0,05 no degrau 5 (o fator d/5 está no termo); a forma de pôr o sinal fica para P2 | PD-34; §6.9 |
| recompensa | `aperto` | depois da `sacudida` | PEGAR, CARREGAR, BOTAR | −0,02 no degrau 5; P2 | S §8.2; Parte 0 D6 |
| terminação | `batente` | depois da `chegada` | as 4 | `time_out=False` | S §10; §6.9 |
| evento reset | `pd`, `forca_motor`, `solref`, `com_caixa` | depois do `vies_imu`, antes do `filtros` | — | reset; `com_caixa` só com `DR.com_caixa` (P19) | §6.8; PD-41 |
| evento | `forca_caixa`, `rfi` | depois do `empurrao` | — | **"step"**, e não "interval" (P4); no play, P24 | §6.8 |

---

## 3. Testes de cada tarefa (os números conferidos pela conta)

**H1** (`test_balanceador_h.py`; `test_curriculo_h.py` só na opção P1-B, com um env falso `SimpleNamespace(v3_degrau=…)`):
- `no_degrau(ini, fim, d, 5)`: d = 0 → ini; d = 5 → fim; d = 3 → ini + 0,6·(fim − ini). Com os pares do knobs, d = 3 dá:
  `laje_joelho_N` 50 − 40·0,6 = **26,0 N**; `incl_lim` 30 − 15·0,6 = **21,0°**; `chegada` 1,0 − 0,75·0,6 = **0,55 m/s**;
  `sigma_pouso` 1,0 − 0,65·0,6 = **0,61 m/s**; o fator (0 → 1) **0,6**. O "degrau 3 → 60%" do plano fecha.
- `degrau_global()`: com os degraus (ANDAR 5, PEGAR 3, CARREGAR 4, BOTAR 0) → 0; com (5, 3, 4, 2) → 2.
- `dr()` no degrau global g:
  - g = 0: pd [1, 1], solref [0,02; 0,02], rfi 0,05 (com a trava "subida"; 0 sem ela), atraso_max 2 (com "portao0").
  - g = 3: pd [0,94; 1,06], forca_motor_bracos [0,88; 1,06], solref [0,017; 0,026], rfi 0,08, atraso_max **3**.
  - g = 5: pd [0,9; 1,1], forca_motor [0,9; 1,1], forca_motor_bracos [0,8; 1,1], solref [0,015; 0,03], rfi 0,1.
  - Monotonia (PD-24): lo de g + 1 ≤ lo de g e hi de g + 1 ≥ hi de g; com g = 2, atraso_max ainda 2.

**H2** (`test_recompensas_h.py`):
- `n_sacudida(dv = 2,9·dt, dw = 17·dt, dt = 0,02, 2,9, 17)` = (2,9/2,9)² + (17/17)² = **2,0**. ⚠ É o p50 da SPEC. O p50
  MEDIDO no 23250_topo004 por diferença finita é ‖a‖ 5,04 e ‖α‖ 31,56 (item 15, linha 32), e ali o custo é
  (5,04/2,9)² + (31,56/17)² = 3,020 + 3,447 = **6,47** (P8). O "p50 do 23250 = 2" do plano testa a fórmula, e não o
  23250.
- `custo_aperto(117,5; 117,5; 49)` = 1 − exp(−117,5/49) = 1 − e^−2,398 = 1 − 0,0909 = **0,909** ✓. Com 49 N nas duas
  faces, 0,632; com uma face só (a outra 0), 0; no degrau 3 o termo vale 0,909·0,6 = 0,545; no degrau 0, 0.
- `forcas_por_face`: um contato de 200 N na face −x (frente) e outro na +z não mudam F_esq nem F_dir (**contato na face
  da frente não entra** ✓). 117,5 N em +y e 117,5 N em −y dão (117,5; 117,5). Uma fatia com achado = False e força
  ≠ 0 não entra. A caixa girada 90° em yaw dá o mesmo resultado (o frame da PD-29 acompanha).
- Máscara: com o índice −1, P0..P2, B3 ou B4, o valor é 0. Num env fora da tarefa com NaN na força, a saída é 0 e
  finita (§6.6).

**H3** (`test_terminacoes_h.py`):
- Uma junta vigiada em q = centro + 1,01·meio, no degrau 1 → True; no degrau 0 → False. ⚠ O teste do plano, "uma
  junta a **100%** do meio curso no degrau 1 termina", NÃO passa com o `>` estrito copiado do NoBatente: frac = 1,0
  não é > 1,0 (P14).
- |hip_yaw| = 1,01 rad no degrau 1 → True; 0,99 rad → False; 1,01 rad no degrau 0 → False.
- O tornozelo a 1,2 do meio curso (o medido no g1_limpo é 1,23 a 1,49) → False se ele não é vigiado (P13).

**H4** (`test_eventos_h.py`): os núcleos de escala por degrau são o `dr()` do H1, já testados lá. Aqui fica só o que
é puro no eventos: `rfi` com s = 0 dá torque 0; com s = 0,1, |τ_j| ≤ 0,1·tau_max_j, com o mesmo gerador (repetível).
Os invólucros vão ao smoke (36).

**H5** (`test_estimador_h.py`):
- Magnitude com f = 0,6 (g = 3): sigma_pos 0,005·0,4 = 0,002 m; sigma_pos_rel 0,012; perda 0,005 + 0,017·0,6 = 0,0152;
  vies_meia 0,005 + 0,015·0,6 = 0,014; deriva_odo 0,002 + 0,0055·0,6 = 0,0053 m/√m.
- Gerador None: sem viés nem rajada, com a saída de hoje (os testes da E continuam verdes).
- Viés de posição constante no episódio: duas leituras com a caixa parada diferem só pelo tremor, e a média de 200
  leituras fica a ≤ 3σ/√200 do viés sorteado.
- Rajada: com a taxa forçada a ∞, a primeira amostra abre uma rajada, e a idade cresce sem leitura por ≥ 0,3 s.
- Orientação: com 10 000 episódios, a mediana e o p90 do ângulo do viés batem com a distribuição escolhida em P21
  (5,3° e 15,3° se for a medida).

**H6**: o log tem `degrau_global`, `atraso_max` e `mu_min_pad`; `Policy/std/<junta>` tem 29 chaves (smoke 38 ou teste
do runner com um logger falso).

**H7** (smoke 34 a 38), conferido:
- 34: degrau forçado a 5 → `laje_joelho_N` = 10, `chegada` 0,25, σ do pouso 0,35 (os "fim" dos pares) ✓.
- 35: `batente` só com degrau ≥ 1 ✓.
- 36: `pd_gains` dentro de 0,9–1,1 no degrau **global** 5. As 4 tarefas têm de estar no degrau 5, porque o global é o
  menor. Ler `actuator_gainprm[:, ctrl, 0] / default` e `biasprm[..., 2]`.
- 37: `com_caixa` ligado por 200 passos em 16 mundos, sem erro de heap (decide a PD-38). Exige o cfg montado com
  `DR.com_caixa` = True (P19).
- 38: `save` → `load` com os degraus e as travas do DR. Os campos derivados do `dr()` saem iguais depois do `carrega`.

---

## 4. Problemas achados

**P1. `valor_do_degrau` (H1) duplica `recompensas.por_degrau`.** Onde: plano §12.4 H1 contra `recompensas.py:391-394`
(5 chamadas). Efeito: duas funções para a mesma fórmula da §6.9 (CLAUDE.md, minimização). Além disso, o
`Refino.regras`, o `fator_por_degrau` e o `tarefas_fator` (knobs.py:461-471) não têm leitor no pacote: só o `test_knobs`
os lê. Opções: (A) fica o `por_degrau`, apoiado em `balanceador.no_degrau`, e saem os três campos mortos (−7 no knobs,
−15 no test_knobs); (B) `curriculo.valor_do_degrau(env, regra)` sobre o `Refino.regras`, sai o `por_degrau`, e as 5
chamadas mudam (≈ +6 linhas). Nas duas, os degraus global e da tarefa usam a mesma `no_degrau`.

**P2. Sinal do peso da `sacudida` e do `aperto`.** Onde: `knobs.py:291-292` (+0,05 e +0,02) contra o `env_cfg._RECOMPENSAS`,
que passa o knob cru como peso (env_cfg.py:100-103). Os custos de hoje têm o knob negativo (`escorrego` −0,2, cercas
negativas); o `perda_caixa` é a exceção, com `_PESO_FIXO` −1. Efeito: posto como os outros, o termo PAGA por sacudir e
por apertar. O item 15 diz "o termo paga −w·custo". Opções: (a) o knob vira −0,05 e −0,02 (o `test_pesos_iguais_ao_knobs`
segue valendo); (b) `_PESO_FIXO` −1, com w e d/5 no núcleo (o padrão do `perda_caixa`); (c) o `env_cfg` nega os dois.

**P3. A força na caixa pela `mdp.apply_body_impulse` não cumpre a §6.8.** Onde: plano H4; `mjlab/envs/mdp/events.py:386-555`.
(a) O termo é de modo "step" (o docstring dele) e não "interval" (§6.5). (b) O `force_range` é sorteado por
componente: (5, 20) dá as três componentes positivas, |F| de 8,7 a 34,6 N, sempre no octante (+x, +y, +z) do mundo;
(−20, 20) dá |F| de 0 a 34,6 N. "5–20 N" em direção qualquer não sai dessa API. (c) Ele age em todo env e todo passo,
sem máscara. Sem máscara, a caixa estacionada a 20 m do ANDAR é empurrada para fora da laje e cai no terreno do robô,
e a caixa do P0–P2 sai da laje e vira perda (`fora_da_laje`). (d) Usa o RNG global, e não o `v3_gerador`. Opções:
subclasse que zera o wrench fora da máscara, ficando o defeito (b); ou uma classe própria (≈ 35 linhas), com |F| U[5; 20],
direção sorteada (no plano ou no espaço: decisão), duração e intervalo do knobs, máscara e `v3_gerador`.

**P4. Modos de evento.** Onde: §6.5 "intervalo: `empurrao`, `forca_caixa`, `rfi`"; comentário do test_cfg.py:49. A
`forca_caixa` (P3) e o RFI "por passo" pedem `mode="step"`, que o mjlab 1.5.3 tem (`event_manager.py:289-291`; roda
depois do `command_manager.compute`, `manager_based_rl_env.py:456-461`). Efeito: com "interval", o RFI não seria por
passo. Opção: corrigir a §6.5 para "step" (a ordem do teste não muda).

**P5. O degrau global (PD-41) não tem dono no plano.** Onde: §12.4 (o H4 e o H5 leem, mas nenhuma tarefa o cria);
`balanceador.py` não tem a função; o `dr()` (balanceador.py:252-272) não tem as chaves do refino, e o `atraso_max`
dele para no portão (comentário: "o refino é da fase G", e a G não fez). Opção: o H1 cria `Curriculo.degrau_global()` e
as chaves do refino no `dr()` (seção 1).

**P6. O `env_cfg.py` está em três lanes da mesma onda.** Onde: §12.4 ondas {H2, H3, H4, H5, H6}. O H2 (recompensas), o
H3 (terminação) e o H4 (eventos) acrescentam linhas no mesmo arquivo; o `observacoes.py` também está no H4
(`forca_caixa` do crítico) e no H5 (magnitude). Efeito: quebra a regra da §1.4. Opção: a seção 5 (o `env_cfg` numa
onda própria; o `observacoes` só no H5).

**P7. O canal `forca_caixa` do crítico segue zero.** Onde: observacoes.py:305-311 ("entra na fase H"); a §12.4 não o
cita. Efeito: o crítico não vê o impulso. Opção: ler o `body_external_wrench` da caixa (1 linha, sem buffer), na lane H5.

**P8. O p50 da sacudida não fecha com o método do termo.** Onde: PD-34 e S §11.5 (2,9 m/s² e 17 rad/s²) contra o item 15
(medido por diferença finita da POSIÇÃO no 23250_topo004: 5,04 e 31,56; D7 da Parte 0, "conferir o método antes de
fixar"). O H2 mede Δv/dt da velocidade do simulador a 50 Hz, um terceiro método, que no contato costuma sair mais ruidoso.
Efeito: no ponto de operação medido, o custo é 6,47 por w, e não 2. Com w = 0,05 no degrau 5, dá 0,32/s, contra os
0,10/s da regra da PD-34 (3,2 vezes). Opções: manter 0,05 (D7) e medir o p50 do próprio termo na fase I; w = 0,0155 (o
medido); trocar a norma para o medido (5,04; 31,56). A D7 segue aberta com o dono.

**P9. A sacudida precisa da velocidade da caixa do passo anterior, e hoje só a vz existe.** Onde: `v3_vz_caixa_ant`
(§6.4; comando.py:529); o mjlab não dá a aceleração do corpo (`entity/data.py` só tem `joint_acc`). Opções: (a) o
buffer vira `v3_v_caixa_ant` (E,6), com o mesmo escritor (cadeia) e leitores (chegada e métrica na coluna 2): toca 4
arquivos (comando, terminacoes, metricas, curriculo) e entra na onda do H1; (b) termo de recompensa por classe, com o
estado (E,6) dentro e um `reset`, sem buffer no env (a §6.4 proíbe buffer criado por termo no ENV, não estado na
classe), só no recompensas.py.

**P10. As normas da sacudida não moram no knobs.** Onde: `parte0/item15_aperto.py:34` (`NORM_LIN, NORM_ANG = 2.9, 17.0`);
o `knobs.Recompensa` não tem campo. Efeito: um número de treino nasceria no termo (§1.5). Opção:
`Recompensa.sacudida_norma = (2,9; 17,0)`.

**P11. Máscara do aperto: índice só, ou "fora do apoio".** Onde: S §8.2 "do P3 ao B2 (nas mãos e fora do apoio)" contra
o plano H2 "caixa presa (P3 a B2 e o CARREGAR)". Com o índice só, o aperto cobra no P3 com a caixa ainda na laje (o
aperto antes de erguer) e no B2 depois do toque, antes do fecho. Com `& ~apoio`, o custo liga em degrau no passo em
que a caixa sai da laje: mais um custo de erguer, contra o ganho do P3. Opções: índice só (o plano); índice
`& ~apoio_caixa.found` (a letra da spec).

**P12. As 12 fatias do `caixa_faces` com `reduce="none"`.** Onde: cena.py:399-404; contact_sensor.py ("none: fast,
non-deterministic ordering"). Efeito: com mais de 12 contatos robô × caixa (pads, pulsos, tronco, antebraço), os que
sobram caem sem critério, e o F por face sai subestimado e ruidoso. Opções: `reduce="maxforce"` (1 linha na cena.py,
fora das lanes do H); medir no smoke quantas fatias enchem na pega do C3.

**P13. O `batente` não diz quais juntas vigia.** Onde: plano H3 e S §6.5/§10 (só "o ângulo do batente da tabela"); o
`NoBatente` tem a lista explícita, e o motivo está documentado (g1_limpo/terminacoes.py:168-172; vault
`g1-limpo-batente-e-restricao-mole`: o `ankle_roll` chega a 1,49 e o `ankle_pitch` a 1,23 do curso, empurrados pelo
contato com o chão; o limite do MJCF é mole). Efeito: com as 29 juntas e `frac_max` 1,0, os tornozelos encerram
episódios em todas as tarefas desde o degrau 1. O aceite da S §15 pede ≤ 5% de terminação nova. Opções: (a) lista sem
os tornozelos (as juntas "cujo batente é decisão de pose"); (b) `frac_max` por família (tornozelo 1,5); (c) as 29, com
medida na fase I. O campo novo do knobs depende da opção.

**P14. "100% do meio curso termina" contra o `>` estrito.** Onde: plano H3 (o teste) contra
g1_limpo/terminacoes.py:196 (`> frac_max`). Efeito: o teste do plano falha com a cópia fiel. Opções: `>=`; ou o teste
em 1,01.

**P15. A força na caixa é por passo, mas a máscara e a trava ficam fora da API** (ver P3). A trava
`dr()["forca_caixa"]` é a primeira subida de nível do PEGAR ou do CARREGAR (§6.8), e não o refino. Fica registrado
para o H4 não ligar a força pelo degrau global.

**P16. As faixas por degrau mudam no reset, e os `dr.*` do mjlab sorteiam no RNG global.** Onde:
`dr/actuator.py:76-87` (`dist.sample` sem gerador), como o `atrito_pad` de hoje. Efeito: a §6.4 diz que todo sorteio
usa o `v3_gerador`; com o RNG global, dois treinos com a mesma semente não são iguais. Opções: aceitar, como no
`atrito_pad`; ou escrever os campos à mão com o `v3_gerador` (+20 linhas).

**P17. O `solref` do chão e da caixa quase não chega ao contato que importa.** Onde: PD-46 e cena.py:212 (pés com
`priority` 1; terreno e caixa 0; laje 3). No MuJoCo, com `priority` diferente, o contato usa os parâmetros da geom de
maior prioridade, `solref` incluso. Efeito: o `solref` do terreno não muda o contato pé × chão (vale o do pé); o da
caixa não muda o caixa × laje (vale o da laje); e no caixa × pad (as duas com `priority` 0) entra só a média pelo
`solmix`. Opções: (a) sortear o `solref` das geoms dos pés e da laje no lugar do terreno; (b) sortear pés, terreno,
caixa e laje; (c) aceitar o efeito parcial. Conferir no smoke 36 com o `solref` do contato (`efc`) num par.

**P18. Selecionar só os braços no `dr.effort_limits` é frágil.** Onde: `dr/actuator.py:213-218` faz
`asset.actuators[i] for i in asset_cfg.actuator_ids`. O `asset.actuators` são os 9 GRUPOS (um por linha do
`Contrato.pd`, cena.py:185-195); o `actuator_names` resolve em 29 nomes (entity.py:461-462, 621-629). Efeito: com
`SceneEntityCfg(actuator_names=…)`, os ids (0..28) não batem com os grupos: estoura o índice ou pega o grupo errado.
Opções: passar `actuator_ids` = os índices dos grupos dos braços (as linhas 7 e 8 do `Contrato.pd`, achados pela regex no
`env_cfg`); ou escrever o `actuator_forcerange` à mão por `ctrl_ids`. O smoke 36 confere.

**P19. O `com_caixa` não cabe no `dr.body_com_offset`.** Onde: plano H4 contra `dr/body.py:330-354`. A faixa é uma
só por chamada, e a da §6.8 é fração da meia aresta, que muda por mundo (8 variantes). E a PD-38 suspeita do
`body_ipos` na heap. Efeito: com a faixa fixa, a caixa pequena ganha CoM fora dela. Opções: escrita própria (o padrão
do `carga`) com a faixa × `v3_meia`; ou um laço pelas 8 meias. Em qualquer caso, o termo entra no cfg só com
`DR.com_caixa` = True: com o termo no cfg e a chave desligada, o `body_ipos` seria expandido e o `set_const` rodaria
toda vez.

**P20. De onde sai o `tau_max` do RFI.** Onde: `recompensas._TAU_MAX` (recompensas.py:40) é privado, e importar o
`recompensas` no `eventos` fecha um ciclo (recompensas → curriculo → eventos). Opções: mover a conta para um
`knobs.tau_max_por_junta()` usado pelos dois (sem função repetida); ou ler o `actuator_forcerange` default do modelo
(`get_default_field`), mapeando atuador → junta.

**P21. Os números do estimador do refino não fecham com o código nem com o knobs.**
- (a) Orientação: o `sigma_ori_graus` do refino = 5,3 é "mediana 5,3°" (knobs.py:403), mas o código o usa como σ POR
  EIXO do rotvec (estimador.py:262-264). O ângulo de um rotvec gaussiano segue χ₃: mediana 1,538σ = **8,2°**, p90
  2,500σ = **13,3°**. A medida é mediana 5,3° e p90 15,3° (razão 2,89; a gaussiana dá 1,63): nenhuma gaussiana
  serve. Uma lognormal no ângulo (μ = ln 5,3; σ_ln = ln(15,3/5,3)/1,2816 = 0,827) põe 0,49% acima de 45° (medido: 0,2%).
  Faltam no knobs a forma e os números (mediana, p90, ou a tabela medida).
- (b) "Viés por episódio + 1° por leitura": o código faz um OU o outro, pelo `ori_persistente` (estimador.py:263). Os
  dois juntos pedem código, e o `ori_persistente` (True, True) fica morto.
- (c) Viés de posição de ±2 cm por episódio (S §13, H5): não há knob nem código. Só existe o da meia aresta (`_z_meia`).
- (d) "Até ±2 cm" (limitado) contra o `vies_meia` de hoje, que é N(0,1)·0,02 (32% das amostras passam de 2 cm). Decidir
  entre U[±] e N(0, σ) para posição, meia e alvo do BOTAR.
- (e) Rajadas de perda de 0,3–1,0 s (DR §3.5): só a duração tem número. A frequência não está no DR, na S nem no
  knobs, e não se sabe se os 2,2% da `perda` já incluem as rajadas.

**P22. O viés do alvo do BOTAR muda também a recompensa.** Onde: H5 contra comando.py:476-478. O `v3_alvo_w` alimenta a
observação (`_calcula_caixa`) e o Φ do B1 e do B2 (`alvo_da_caixa_w`), além do raio da perda. Com o viés no `v3_alvo_w`,
o robô aprende a pousar onde o controlador mandou (o real), como o PEGAR já faz com a caixa estimada (comando.py:330-347).
Com o viés só na observação, a recompensa pede um ponto que o ator não vê. A opção do PEGAR é a primeira, mas o plano
não diz. Os knobs `vies_alvo_botar_*` existem e não têm leitor.

**P23. Faltam medidas da S §14.** "Aceleração da caixa" não tem métrica, e o custo da sacudida vale 0 até o refino
(fator d/5): o ponto de operação só se vê no degrau ≥ 1, o canal-medida morto do docstring do metricas. As chaves
`atraso_max` e `mu_min_pad` da §6.7 não estão no log do `v3`. Opção: a métrica `aceleracao_da_caixa` e as duas chaves
no H6.

**P24. DR do refino no play.** Onde: env_cfg.py:144-149 (o play tira só o empurrão e o currículo). Efeito: com um
checkpoint no degrau global 5, o `pd`, a `forca_motor`, o `solref`, a `forca_caixa` e o `rfi` agem no play e na
paridade com o `pilota_v3`. O exportador lê o modelo C, sem DR, e não quebra. Opções: tirar os cinco do play, como o
empurrão; ou manter e documentar.

**Notas menores (sem decisão):**
- O H3 manda "cópia de `NoBatente`" (classe). A v3 usa núcleo + invólucro, e com os índices resolvidos no import a
  classe não precisa existir (−1 classe).
- O plano estima o `eventos.py` em 500 linhas no fim da fase H (§16). Ele tem 475 hoje e passa de ~545 com o H4.
- O H6 "degrau por tarefa" já está feito (curriculo.py:125).
- A §12.4 manda o H1, o H2, o H4 e o H5 em Sonnet, e a §20 (29/10) diz "a implementação segue com agentes Haiku".
- O `Refino.tarefas_fator` repete o `env_cfg.PEGAR_CARREGAR_BOTAR` (duas fontes para as tarefas da sacudida e do aperto).

---

## 5. Lanes (Haiku) por onda, com arquivos disjuntos

| onda | lane | arquivos (só estes) | conteúdo | linhas antes → depois |
|---|---|---|---|---|
| H0 | H0-a testes | `testes/test_balanceador_h.py`, `testes/test_curriculo_h.py` (P1-B), `testes/test_knobs.py` | H1 (seção 3) | 0 → ~140; test_knobs ±15 |
| H0 | H0-b testes | `testes/test_recompensas_h.py`, `testes/test_terminacoes_h.py`, `testes/test_estimador_h.py`, `testes/test_eventos_h.py`, `testes/test_cfg.py` | H2 a H5; pesos, modos e tarefas no cfg | 0 → ~260; test_cfg +15 |
| 1 | H1 | `knobs.py`, `balanceador.py`, `curriculo.py`, `recompensas.py` (só as chamadas do `por_degrau`), `terminacoes.py` (só a chamada da `chegada`); na opção P9-a, também `comando.py`, `metricas.py` (o rename para `v3_v_caixa_ant`) | `no_degrau`, `degrau_global`, chaves do `dr()`, P1, P2-a, P10, os campos novos de P13/P21 | knobs 485 → ~490 (P1-A −7, campos +12); balanceador 296 → ~318; curriculo 128 → 128 (A) ou ~138 (B); recompensas −4 (B) |
| 2 | H2 | `recompensas.py` | `n_sacudida`, `forcas_por_face`, `sacudida`, `aperto` (+ a classe se P9-b) | 783 → ~825 |
| 2 | H3 | `terminacoes.py` | `n_batente`, `batente` | 135 → ~158 |
| 2 | H4 | `eventos.py` | `pd`, `forca_motor`, `solref`, `com_caixa`, `forca_caixa`, `rfi` | 475 → ~545 |
| 2 | H5 | `estimador.py`, `observacoes.py`, `comando.py` | magnitude por g, vieses, rajada, deriva da odometria, viés do BOTAR, canal `forca_caixa` | estimador 316 → ~348; observacoes 375 → ~377; comando 767 → ~780 |
| 2 | H6 | `runner.py`, `metricas.py` (P23) | std por junta; `aceleracao_da_caixa` | runner 42 → ~55; metricas 497 → ~515 |
| 3 | H4b | `env_cfg.py`, `curriculo.py` (as 3 chaves do log) | as linhas da §6.5 (seção 2); P24 | env_cfg 185 → ~212; curriculo +3 |
| 3 | H7 | `smoke.py` | seções 34 a 38 | 1 272 → ~1 420 |
| 4 | H8 | (Opus, revisão) | — | — |

Por que o `env_cfg` vai à onda 3: ele referencia `recompensas.sacudida`, `terminacoes.batente` e os eventos novos,
que só existem depois da onda 2. Na onda 2, o H4 e o H5 não tocam o `env_cfg` nem o `observacoes` um do outro (P6).

Total do pacote, sem smoke e sem testes: ~+250 linhas (−4 a −11 se P1-A). Testes: ~+430. Smoke: ~+150. Nenhum termo
de recompensa sai do que a §6.5 já lista. Os termos "novos" (`sacudida`, `aperto`, `batente`) estão na spec desde 14/10.

---

## 6. O que já existe e se reaproveita

- `recompensas.por_degrau` = o `valor_do_degrau` do H1 (5 chamadas); o `test_knobs.test_dr_e_refino` já resolve o
  `Refino.regras` (a conta da opção P1-B).
- `potencial.custo_aperto` = o núcleo do `aperto`, com o teste de 0,909 já fechado no item 15.
- `recompensas._CAIXA_ALVO` = (P3, P4, B1, B2) = a máscara "caixa presa"; `recompensas._em` e `_fim`.
- `alvos.frame_da_caixa` (PD-29) e o sensor `caixa_faces` (pos, normal, force, tangent, global) = a régua por face.
- `v3_vz_caixa_ant` e o escritor no fim do `compute` da cadeia = o padrão do Δv da sacudida (P9-a).
- `Curriculo.dr()` com as travas, `Curriculo.degrau(t)` e o `estado()` que já guarda os degraus = a base do degrau
  global e da DR do refino.
- `curriculo.v3` + `eventos.atraso` = o atraso 0–3 sem código novo (só o número no `dr()`).
- `eventos.atrito_pad` = o molde do invólucro de um `dr.*` com a faixa do currículo; `eventos.tamanho_caixa` e
  `eventos.carga` = o molde da escrita direta com `@requires_model_fields` (`solref`, `com_caixa`).
- O mjlab tem `dr.pd_gains` e `dr.effort_limits` (escala sobre o default); `mdp.apply_body_impulse` cobre o ciclo
  do impulso (com os defeitos de P3); `entity.indexing.joint_v_adr` dá o índice do `qfrc_applied`.
- `EstimadorCaixa._z_meia` e `_z_ori` (vieses por episódio), `ori_persistente`, `perda`, `sigma_pos_rel` com `dist`;
  `observacoes._magnitude` (vira fração); `CadeiaCmd.deriva` (vira por degrau).
- `knobs`: `Terminacao.hip_yaw_max`, `batente_frac`; `Refino.degrau_batente`; `DR.pd`, `forca_motor`,
  `forca_motor_bracos`, `solref`, `solref_base`, `rfi`, `forca_caixa_*`, `com_caixa*`, `atraso_refino`,
  `degrau_atraso_refino`; os pares (início, refino) do `Estimador`; `Recompensa.sacudida`, `aperto`, `F_ref`.
- O `test_cfg` já lista `sacudida`, `aperto`, `batente`, `pd`, `forca_motor`, `solref`, `com_caixa`, `forca_caixa`
  e `rfi` na ordem da §6.5 (test_cfg.py:40-50).
- `observacoes.forca_caixa`: o canal existe com a forma certa, falta ler o wrench.
- rsl_rl `output_std` (29 por junta) para o H6.
