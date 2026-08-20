# Spec — a forma do episódio vira CONTROLADOR (MACRO 2b)

Decisão do usuário (20/08): **nenhuma configuração manual para o andar.** O bloco da
Fase 3 (`frac_locomocao = 0,85` à mão) morre; o sorteio da forma passa a se ajustar
sozinho durante o treino, e o "segure e ande" liga pelo nível do próprio env.

**Pré-requisito:** MACRO 2 (specs/g1-poc-macro2.md) aplicado. Mesmas regras: só
`g1_poc/`, português, `⚠` para armadilhas, sem commit, PARE e relate divergência.

---

## O problema, medido

`frac_locomocao = 0,30` é por EPISÓDIO e o PPO aprende por TRANSIÇÃO. A fatia real:

```
fatia = f·T_loco / ( f·T_loco + (1−f)·T_manip )
```

Com T_loco = 24 (medido) e T_manip = 961: fatia = **1,06%**. O laço se auto-sustenta:
sem dado não anda, sem andar o episódio morre em 24 passos.

## O controlador

Fixe a FATIA DE TRANSIÇÕES alvo e resolva o sorteio a partir das durações medidas:

```
f = alvo·T_manip / ( T_loco·(1−alvo) + alvo·T_manip )        alvo = 0,30
```

Propriedades (conferíveis à mão):
- T_loco = 24, T_manip = 961 → f = 0,945 (o robô não anda: despeja episódios curtos de
  andar; a manipulação CONTINUA com 70% das transições)
- T_loco = T_manip → f = alvo (marcha madura: o sorteio relaxa sozinho para 0,30)
- monótono e sem integrador → sem oscilação de controle; as EMAs dão a inércia (τ ≈ 100
  eventos ≈ 4 iterações)

---

## Tarefa 1 — `g1_poc/knobs.py`

Em `Episodio`, SUBSTITUA o comentário de `frac_locomocao` e acrescente os knobs:

```python
    # ⚠ Desde 20/08 isto é a FATIA DE TRANSIÇÕES alvo, e não o sorteio. O sorteio é
    # resolvido pelo controlador em `curriculo.sorteia_forma`, a partir das durações
    # MEDIDAS: f = alvo·T_manip / (T_loco·(1−alvo) + alvo·T_manip). Com o episódio
    # de andar morrendo em 24 passos, 30% de sorteio davam 1,06% dos dados — e o
    # bloco manual de "frac 0,85" que consertava isso era exatamente a configuração
    # manual que o usuário vetou. O controlador despeja episódios de andar enquanto
    # eles são curtos e relaxa sozinho para ~0,30 quando a marcha amadurece.
    frac_locomocao: float = 0.30
    # clamps do sorteio: nunca menos de 10% nem mais de 95% de locomoção
    frac_loco_min: float = 0.10
    frac_loco_max: float = 0.95
    forma_ema: float = 0.99
```

E TROQUE o comentário/semântica de `frac_twist_livre_manipula` (da MACRO 2):

```python
    # "segure e ande": fração dos envs de manipulação com o twist LIBERADO, ATIVA
    # SÓ a partir de `twist_livre_nivel_min`. Automático: o env começa a treinar
    # andar-segurando um nível ANTES de o `carregar` abrir (nível 4), fechando o
    # vão de distribuição (0,00% das transições tinham twist ≠ 0 com caixa válida)
    # sem bloco manual.
    frac_twist_livre_manipula: float = 0.30
    twist_livre_nivel_min: int = 3
```

---

## Tarefa 2 — `g1_poc/curriculo.py` — `sorteia_forma` vira o controlador

Assinatura nova:

```python
def sorteia_forma(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    frac_locomocao: float,
    frac_loco_min: float,
    frac_loco_max: float,
    ema: float,
) -> dict[str, torch.Tensor]:
```

Corpo (mantendo o docstring atual e ACRESCENTANDO o controlador; o ⚠ da ordem fica):

```python
    if not hasattr(env, "poc_manipula"):
        env.poc_manipula = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device)
        # as durações nascem NEUTRAS (episódio cheio): o sorteio começa no alvo e
        # se ajusta por medição em ~τ. Nascer pessimista (24) despejaria locomoção
        # antes de existir amostra.
        env.poc_dur_loco = torch.full((), float(env.max_episode_length),
                                      device=env.device)
        env.poc_dur_manip = torch.full((), float(env.max_episode_length),
                                       device=env.device)

    # --- mede as durações dos episódios que ACABARAM (a forma ainda é a antiga:
    # este termo lê ANTES de sobrescrever; episode_length_buf zera só no fim) ---
    if len(env_ids) > 0:
        antiga = env.poc_manipula[env_ids]
        loco = env_ids[~antiga]
        manip = env_ids[antiga]
        if len(loco) > 0:
            amostra = env.episode_length_buf[loco].float().mean()
            env.poc_dur_loco = ema * env.poc_dur_loco + (1.0 - ema) * amostra
        if len(manip) > 0:
            amostra = env.episode_length_buf[manip].float().mean()
            env.poc_dur_manip = ema * env.poc_dur_manip + (1.0 - ema) * amostra

    # --- o controlador: f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm) ---
    # Fixa a FATIA DE TRANSIÇÕES em `frac_locomocao` resolvendo o sorteio a partir
    # das durações medidas. Tl = 24 e Tm = 961 dão f = 0,945; Tl = Tm dá f = alvo.
    # Sem integrador: o mapa é estático e as EMAs dão a inércia — não oscila.
    alvo = frac_locomocao
    tl = float(env.poc_dur_loco)
    tm = float(env.poc_dur_manip)
    f = alvo * tm / max(tl * (1.0 - alvo) + alvo * tm, 1e-6)
    f = min(max(f, frac_loco_min), frac_loco_max)

    sorteio = torch.rand(len(env_ids), device=env.device)
    env.poc_manipula[env_ids] = sorteio >= f
    dev = env.device
    return {
        "frac_manipula_pop": env.poc_manipula.float().mean(),
        "frac_loco_sorteio": torch.tensor(f, device=dev),
        "dur_loco_ema": torch.tensor(tl, device=dev),
        "dur_manip_ema": torch.tensor(tm, device=dev),
    }
```

⚠ `twist_por_competencia` mantém a EMA PRÓPRIA (ela exclui os envs parados; esta inclui —
são filtros diferentes para perguntas diferentes). Acrescente uma linha no docstring de
cada uma dizendo isso, para ninguém "unificar" depois.

Docstring do módulo: a parte A agora tem DOIS adaptativos (a forma, por duração; o nível,
por sucesso).

---

## Tarefa 3 — `g1_poc/comando.py` — o twist livre liga pelo NÍVEL

No `_resample_command`, o sorteio do `_twist_livre` (da MACRO 2) ganha o gate de nível:

```python
        # "segure e ande" AUTOMÁTICO: só a partir de `twist_livre_nivel_min` (um
        # nível antes de o `carregar` abrir). Nada de bloco manual.
        if nivel is None:
            alto = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            alto = nivel[env_ids] >= self.cfg.twist_livre_nivel_min
        self._twist_livre[env_ids] = (
            (torch.rand(n, device=self.device) < self.cfg.frac_twist_livre)
            & manipula & alto)
```

`CaixaAlvoCommandCfg` ganha `twist_livre_nivel_min: int = 3`; o `env_cfg` o passa de
`ke.twist_livre_nivel_min`.

---

## Tarefa 4 — `g1_poc/env_cfg.py`

O termo `forma` passa os params novos:

```python
        "forma": CurriculumTermCfg(
            func=CU.sorteia_forma,
            params={"frac_locomocao": ke.frac_locomocao,
                    "frac_loco_min": ke.frac_loco_min,
                    "frac_loco_max": ke.frac_loco_max,
                    "ema": ke.forma_ema},
        ),
```

E `twist_livre_nivel_min=ke.twist_livre_nivel_min` no `CaixaAlvoCommandCfg`.

⚠ O play (`_ajusta_manipula`/`_ajusta_andar`) muta `k.episodio.frac_locomocao` para 0
ou 1 — o controlador tem de RESPEITAR isso: com alvo 0, f = 0 (o max(_, 1e-6) protege a
divisão); com alvo 1, f = 1. Confira a álgebra: alvo = 0 → f = 0 ✓; alvo = 1 →
f = tm/tm = 1 ✓. Os clamps min/max ESTRAGARIAM os dois extremos — por isso o clamp só
se aplica quando `0 < alvo < 1`:

```python
    if 0.0 < alvo < 1.0:
        f = min(max(f, frac_loco_min), frac_loco_max)
```

(Corrija o corpo da Tarefa 2 com este guard — ele é obrigatório, senão o `--pegar` do
play volta a sortear 10% de locomoção e o viewer abre sem mobília.)

---

## Tarefa 5 — `g1_poc/smoke.py`

Seção nova (depois da seção do gate do twist, reutilizando o env):

```python
    print("== 18. o controlador da forma (§11) ==")
    tc_f = env.curriculum_manager.get_term_cfg("forma")
    ke2 = k.episodio
    # a álgebra do controlador, sobre a fórmula
    def f_de(tl, tm, alvo):
        return alvo * tm / max(tl * (1.0 - alvo) + alvo * tm, 1e-6)
    checa(abs(f_de(24.0, 961.0, 0.30) - 0.945) < 0.005,
          f"não anda (Tl=24): sorteia {f_de(24.0, 961.0, 0.30):.3f} de locomoção")
    checa(abs(f_de(961.0, 961.0, 0.30) - 0.30) < 1e-9,
          "marcha madura (Tl=Tm): o sorteio relaxa para o alvo 0,30")
    checa(f_de(0.0, 961.0, 0.0) == 0.0 and abs(f_de(24.0, 961.0, 1.0) - 1.0) < 1e-9,
          "os extremos do play (alvo 0 e 1) saem exatos, sem clamp")
    # o termo mede a forma ANTIGA e sorteia a nova
    env.poc_dur_loco = torch.full((), 24.0, device=env.device)
    env.poc_dur_manip = torch.full((), 961.0, device=env.device)
    env.episode_length_buf[:] = 500
    saida = tc_f.func(env, todos, **tc_f.params)
    checa(float(saida["frac_loco_sorteio"]) > 0.90,
          f"com Tl na EMA em 24, o sorteio despeja locomoção "
          f"(medido {float(saida['frac_loco_sorteio']):.3f})")
```

(Adapte `todos`/`k` aos nomes já usados no arquivo; restaure o que a seção sujar.)

---

## Critério de pronto

`python -m g1_poc.smoke` com 0 falhas (contagem sobe ~5). Relate contagem, arquivos e
divergências. O comportamento em treino (f caindo de 0,94 para 0,30 conforme a marcha
amadurece) NÃO é testável no smoke — é leitura de log: `Curriculum/forma/frac_loco_sorteio`
e `dur_loco_ema` são as duas curvas que contam a história.
