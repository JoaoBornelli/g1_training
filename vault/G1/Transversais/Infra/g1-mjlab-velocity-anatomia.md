---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
---
# g1-mjlab-velocity-anatomia (original)

Corpo original da memória `g1-mjlab-velocity-anatomia`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Levantamento completo de `mjlab/tasks/velocity` (mjlab **1.5.1**, rsl_rl_lib 5.4.0) feito 2026-08-06 por 5 agentes de leitura. Path: `g1_training/.venv/lib/python3.12/site-packages/mjlab`.

Fatos que não se deduzem do config à primeira vista:

- **`RewardManager.compute` faz `bruto × peso × dt`** (`managers/reward_manager.py:127`), `scale_rewards_by_dt=True` por default. Com `dt=0.02`, o peso `2.0` do rastreio vale **0.04** por passo. Nenhum peso do config é o peso efetivo. Confirma o que já estava em g1-multitask-spec-validacao-2026-08.
- **Currículo tem só 2 eixos**, e apenas um é gated por desempenho: `terrain_levels` (por-robô, promove por distância percorrida) e `command_vel` (**global**, degraus por contagem de passos nas iterações 5000 e 10000). `reward_curriculum` e `termination_curriculum` existem em `envs/mdp/curriculums.py` e **não são usados**. Não há fases sequenciais de habilidade.
- **Uma única condição de falha**: `bad_orientation` a 70°. `time_out` e `out_of_terrain_bounds` são truncados, com bootstrap `+γV` injetado na recompensa (`rsl_rl/algorithms/ppo.py:150-155`).
- **Assimetria actor/critic é dupla**: o critic ganha 4 termos privilegiados de pé (altura, air time, contato, força log-comprimida) **e** vê tudo sem ruído (`enable_corruption=False` força `noise=None` no deepcopy). Actor 286 dims no rough / 99 no flat.
- **Sem histórico temporal** na observação. O mecanismo existe (`history_length`) e não é ligado.
- **`action_scale` é derivada do hardware**: `0.25·effort_limit/stiffness` por grupo de atuador (`g1_constants.py:278-286`), não ajustada por tentativa e erro.
- **Quase todo termo de marcha é gateado por comando** (`command_threshold=0.05`): pé/escorregamento/pouso só contam se houver comando de movimento. É o que evita punir o robô parado.
- **DR usada é mínima** contra o menu disponível: só atrito do pé (0.3–1.2), viés de encoder (±0.015 rad), COM do torso (±2.5cm) — todos `startup`, fixos por env — e push por intervalo 1–3 s. Ganho PD, massa, atraso de ação e limite de esforço **não** são randomizados.
- **Bug real no mjlab 1.5.1**: `feet_swing_height` guarda `self.peak_heights` e a classe **não define `reset()`**, então o `RewardManager` nunca a zera. O pico vaza entre episódios se o reset pega o pé no ar (`tasks/velocity/mdp/rewards.py:271-323`).
- 29 juntas atuadas, **corpo inteiro com braços**, atuador `<position>` nativo (`torque = kp(alvo−q) − kd·q̇`), sem limite de velocidade imposto no sim.

## Armadilhas ao ESCREVER TESTE contra os managers (verificado 2026-08-18, g1_poc)

Três coisas fizeram o smoke do g1_poc reprovar por medição errada, com o env correto:

- **`ObservationManager.compute()` tem CACHE.** Ele devolve `self._obs_buffer` do passo anterior se `update_history=False`, que é o default (`managers/observation_manager.py:311`). Mexer num buffer e chamar `compute()` lê a obs ANTIGA. Em teste use **`compute_group(grupo)`**, que não passa pelo cache.
- **Os `SceneEntityCfg` do cfg NÃO são resolvidos.** O manager faz `deepcopy` do cfg e resolve a cópia: no cfg `site_names=(...)` continua com `site_ids=slice(None)` (todos os sites), e no manager já é a lista certa. Chamar a função com os params do cfg dá shape errado silencioso ou `RuntimeError` de broadcast. Use `reward_manager.get_term_cfg(nome).params` (público) ou `observation_manager._group_obs_term_cfgs[grupo]`.
- **O ruído entra DEPOIS da função.** Testar "esta obs é zero" pela obs concatenada falha por `Unoise`, que é intencional. O invariante é da função — chame-a direto.

Contagens da fundação, para teste de contrato: **13 recompensas** e **3 terminações** (`time_out`, `fell_over`, `out_of_terrain_bounds`). **Não existe `nonfinite`** nem `self_collisions` no `velocity_env_cfg.py` — quem quiser esses dois os cria.

**Nomes de geom de colisão do G1** (o URDF engana): há `pelvis`, `*_hip`, `*_thigh`, `*_shin`, `*_linkage_brace`, `*_foot1..7`, `torso`, `head`, `*_shoulder_yaw`, `*_elbow_yaw`, `*_wrist`, todos com sufixo `_collision`, mais os pads `*_palm_pad`/`*_hand_back_pad`. **Não existe geom de `waist` nem de `knee`.** Padrão que casa zero geom é `ValueError` no `resolve_matching_names`, não aviso.

## Anatomia do sorteio do twist (verificado 2026-08-27)

O comando de velocidade não é um sorteio uniforme simples. Cinco mecanismos agem juntos, todos em `tasks/velocity/mdp/velocity_command.py` e `velocity_env_cfg.py:180-192`:

- **Re-sorteio de 3 a 8 s POR ENV**, independente do reset (`command_manager.py:111-113`). Num episódio de 20 s dá 2 a 6 comandos. Se o episódio dura menos de 3 s, ele nunca dispara.
- **`rel_standing_envs = 0.1`** — 10% dos envs têm o comando inteiro zerado todo passo (`:136-138`).
- **`rel_forward_envs = 0.2`** — 20% recebem só frente: `vx = |vx|.clamp(min=0.3)`, `vy = wz = 0`. Nunca podem ser zero.
- **`rel_heading_envs = 0.3`** com `heading_command=True` — para 30% o `wz` não é sorteado; é malha fechada de rumo (`0,5 × erro`).
- `rel_world_envs` e `init_velocity_prob` ficam em 0,0. A config do G1 não os toca.

Faixas do molde: `lin_vel_x` e `lin_vel_y` em ±1,0, `ang_vel_z` em ±0,5. Uma variante do G1 abre para `lin_vel_x` (−1,5, 2,0) e `ang_vel_z` ±0,7 (`config/g1/env_cfgs.py:217`).

⚠ **Girar no lugar não é um caso do sorteio** (verificado 2026-09-02). `lin = 0` com `ang ≠ 0` exige dois contínuos independentes perto de zero ao mesmo tempo. Os 10% `standing` não giram; os 20% `forward` não giram; os 30% `heading` giram andando. Só um ramo próprio (`rel_turning_envs`) produz giro parado. O estimador `seg_proj/seg_pedido` é cego a giro: ele acumula só a norma linear.

Relacionado: g1-auditoria-mjlab-bloco2, g1-auditoria-mjlab-bloco3, g1-multitask-curriculum-design, g1-multitask-orcamento-igual-por-tarefa, g1-limpo-desenho.
