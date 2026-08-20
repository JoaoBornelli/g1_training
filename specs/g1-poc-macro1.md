# Spec de implementação — MACRO 1: células, ficar de pé, andar (v2, pós-auditoria)

Executa as Fases 1, 2 e 3 de `docs/planos/2026-08-20-curriculo-progressivo-e-andar.md`,
com as correções das quatro auditorias de 20/08 (API do mjlab, viabilidade, travas,
incentivos). O adendo do plano lista os vereditos; este spec já os incorpora.

**Repositório:** `/home/joaobornelli/Documents/g1_training`, branch `exp/g1-poc`.
**Pacote:** `g1_poc/`. **NÃO editar `g1_training/`** (o pacote comum) **nem `.venv/`**.
**Idioma do código e dos comentários: português.** Siga o estilo do arquivo que você
está editando: docstring explicando o PORQUÊ, e `⚠` para armadilhas medidas.

**Verificação após cada tarefa:**
`cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_poc.smoke`
No fim tem de dar **0 falhas**. (Durante as tarefas intermediárias o smoke pode
falhar por contagem — termine a tarefa 8 antes de julgar.)

**Não commite.** Deixe a árvore com as mudanças; o revisor e o usuário cuidam do git.

---

## Contexto mínimo (medido, não suposto)

- Ordem no reset (`manager_based_rl_env._reset_idx`): **currículo (todos os termos,
  em ordem de inserção do dict) → eventos → … → comando.reset**. O
  `episode_length_buf[env_ids]` só zera na ÚLTIMA linha — dentro de um termo de
  currículo ele ainda vale a duração final do episódio que acabou.
- `RewardManager`/`CurriculumManager`/`TerminationManager`/`EventManager`/
  `ObservationManager` fazem `deepcopy` do cfg → pós-construção, mutar só via
  `<manager>.get_term_cfg(nome)`. **`CommandManager` NÃO deepcopia** —
  `command_term.cfg` é o mesmo objeto do cfg, e `_resample_command` relê
  `self.cfg.ranges` a cada resample.
- `torch.maximum`/`torch.where` devolvem tensor NOVO. Atribuir `self.x = ...` quebra
  qualquer alias `env.y = self.x`. Bug vivo consertado na tarefa 3.1.
- `Entity.find_sites` existe (`mjlab/entity/entity.py:561`), mesmo molde do
  `find_joints`.
- Sensores `reduce="netforce"`: `data.found` é `[B, 1]`, `data.force` é `[B, 1, 3]`
  no frame GLOBAL; `found > 0` é a leitura de contato.
- ⚠ MEDIDO 20/08: o `afasta_cena` está CORRETO — a caixa afastada fica APOIADA na
  laje a 5 m (z = 5,099 após 120 passos, mesmo com 5 kg de wrench). NÃO mexa nele.

---

## Tarefa 1 — `g1_poc/knobs.py`

### 1.1 Nova dataclass `Celulas`

Insira **depois** de `class Alvo` e antes de `class Tolerancia`:

```python
@dataclass
class Celulas:
    """§10.1 — a célula que cada nível seleciona. Sete níveis, de 0 a 6.

    Três regras da tabela, e elas explicam por que só o PISO varia:

    - o TETO do topo é 0,55 m em todo nível (`Cena.prateleira_topo_teto`). O robô
      continua treinando a altura que domina.
    - o PISO da carga é 1 kg em todo nível (`Cena.caixa_massa`). Mesmo motivo.
    - o nível ACRESCENTA cadeias. Ele não substitui cadeias.

    ⚠ O nível 6 da §10.1 pede rotação no eixo HORIZONTAL, que exige tombar a caixa.
    É o Risco 1 da §19, e o G1 não tem mão. A célula do 6 fica igual à do 5, e os
    critérios de aceite da §0 não pedem o 6.

    ⚠ `topo_min[4:] = 0,04` é a MESMA laje de `Cena.prateleira_topo_piso = 0,04` —
    dois knobs, um número físico. Quem mudar um tem de mudar o outro.
    """
    topo_min: tuple[float, ...] = (0.55, 0.45, 0.30, 0.15, 0.04, 0.04, 0.04)
    carga_max: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0)
    ang_max_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 45.0, 90.0, 180.0, 180.0)
    # ⚠ O jitter x da caixa também é da célula (auditoria de travas, 20/08): com o
    # topo a 0,04 m, poses de pega só existem até x relativo ≈ 0,45 — com o jitter
    # de 0,20 fixo, 60% dos episódios do nível 4 exigiriam um passo à frente, que o
    # twist zerado cobra (−0,44/s) e nenhum termo de marcha paga.
    jitter_x_max: tuple[float, ...] = (0.20, 0.20, 0.20, 0.15, 0.08, 0.08, 0.08)
    # Fração de cada cadeia, na ordem
    #   (`pegar`, `reorientar`->`pegar`, `pegar`->`carregar`, `pegar`->`botar`).
    # Somam 1,0 em cada nível. ⚠ Só a máquina de elo consome isto (MACRO 2); em
    # MACRO 1 o campo fica declarado e não lido — é a tabela da §10.1 inteira, num
    # lugar só.
    cadeias: tuple[tuple[float, float, float, float], ...] = (
        (1.00, 0.00, 0.00, 0.00),
        (1.00, 0.00, 0.00, 0.00),
        (1.00, 0.00, 0.00, 0.00),
        (0.50, 0.50, 0.00, 0.00),
        (0.40, 0.25, 0.35, 0.00),
        (0.30, 0.20, 0.25, 0.25),
        (0.30, 0.20, 0.25, 0.25),
    )
```

Registre em `class Knobs`: `celulas: Celulas = field(default_factory=Celulas)`
(logo depois de `alvo`).

### 1.2 `class Alvo` — um knob novo

Acrescente no fim da classe:

```python
    # folga entre o topo NOVO da prateleira e o fundo da caixa segurada, no
    # instante em que o `pegar` fecha na cadeia `pegar`->`botar` (§7.3).
    # ⚠ A §7.3 e a §10.1 se CONTRADIZEM: a §7.3 garante segurança dizendo "o topo
    # novo fica no máximo em 0,55 m", e a §10.1 manda sortear a colocação em
    # 0,30-0,80 m. Com a caixa segurada a 0,82 m o fundo dela está em 0,72 m, e uma
    # laje em 0,80 m nasceria DENTRO da caixa. O teto efetivo é
    # `fundo_da_caixa - botar_folga_laje`, resolvido por env. Consumido em MACRO 2.
    botar_folga_laje: float = 0.05
```

### 1.3 `class Recompensa` — os dois termos novos e uma nota no `reaching_std`

Troque a linha `reaching_std: float = 0.20` por:

```python
    # ⚠ `reaching_std` virou o PISO do σ, não o σ (20/08). O σ efetivo é
    # `max(reaching_std, distância inicial palma→face)`, por env, recalculado no
    # começo do elo — a MESMA correção que a §8.2 fez no `bringing`. Medido: com σ
    # fixo de 0,20 o gradiente de aproximação cai 1391× entre a prateleira a 0,55
    # (2,64/m) e a 0,04 (0,0019/m) — os níveis 3+ viravam sorte.
    reaching_std: float = 0.20
```

Acrescente depois de `unload_tol_queda`:

```python
    # §8.2.3 — a rampa da pelve, ligada em 20/08.
    # A condição 3 do fecho do `pegar` exige pelve >= 0,65 m e NADA pagava por ela.
    # ⚠ A justificativa correta (auditoria 20/08): quem precifica a pelve é só a
    # `pose`, a ~0,73/m (o default é o KNEES_BENT_KEYFRAME, pelve 0,76). O
    # `precise_pos` é INDIFERENTE à pelve abaixo do alvo e CONTRÁRIO acima
    # (−16,2/m no ponto de fecho) — não "paga por agachar", como uma versão
    # anterior deste pacote afirmou.
    #
    # A rampa tem DUAS partes, e cada uma fecha um buraco medido:
    #   longa (0,20→0,65): sem ela o termo é MORTO em 33% das pegas do nível 4
    #     (pelve na pega chega a 0,267 m) e a zona 0,20-0,45 não tem gradiente;
    #   fina (0,57→0,65): sem ela a inclinação é 5/m contra os −16,2/m do
    #     `precise_pos` no fecho — o robô perderia recompensa ao subir os últimos
    #     centímetros com os braços rígidos.
    # Com peso 2,0: 2,2/m na zona longa e 14,7/m na fina.
    postura_ereta: float = 2.0
    postura_ereta_rampa: float = 0.45        # a parte longa: 0,20 -> 0,65
    postura_ereta_rampa_fina: float = 0.08   # a parte fina : 0,57 -> 0,65
    # ⚠ O gate de DESCARGA (F_apoio < frac·m·g) é anti-hack medido: sem ele,
    # encostar as palmas e ficar de pé com a caixa APOIADA paga a rampa inteira —
    # +2,0/s por ficar exatamente no platô que o bloco 1 mediu.
    postura_ereta_frac_descarga: float = 0.2
    # §8.2.4 — a rampa da sustentação, ligada em 20/08.
    # O fecho exige 1,0 s ininterrupto e NENHUM termo diferencia 0,98 s de 0,00 s.
    # Medido: o push era o único fator que degradava o sucesso, exatamente porque
    # quebra o cronômetro. Esta é a rampa na coordenada TEMPO-NA-CONDIÇÃO.
    sustentacao: float = 0.5
```

### 1.4 `class Postura` — o regime deixa de depender da demanda

**Remova** os campos `peso_dist`, `peso_ang` e `limiar`.
**Acrescente** `running_threshold: float = 1.5` (o 1,5 continua servindo ao terceiro
regime de VELOCIDADE do mjlab, que vale nos episódios de locomoção).

Substitua a docstring da classe por:

```python
    """§9 — o quarto regime do `variable_posture`.

    Os três dicionários do G1 ficam intocados. Este é o quarto, e ele responde à
    FORMA do episódio: `caixa_valida = 1` -> `std_manipulando`.

    Regra: as juntas do plano sagital abrem, e as laterais ficam apertadas.

    ⚠ Ele já respondeu à DEMANDA da caixa (`peso_dist`, `peso_ang`, `limiar`), e
    aquilo era um PENHASCO, não um gradiente. Ver o docstring de
    `postura.postura_manipulacao`. Quem levanta o robô agora é o termo
    `postura_ereta` (§8.2.3), que é rampa.
    """
```

### 1.5 `class Cronograma` — o gate por competência

Acrescente no fim da classe:

```python
    # §10.3 — o gate por COMPETÊNCIA do twist, ligado em 20/08.
    # Dois dos três cronogramas por passo global já saíram de fase. O passo global
    # vira o PISO do degrau; o gatilho é o robô SUSTENTAR o teto atual, medido pela
    # duração do episódio de LOCOMOÇÃO (um robô que não anda cai em 24 passos; um
    # que anda chega ao time_out — e é a MESMA grandeza que governa a fatia de
    # transições).
    twist_duracao_min_frac: float = 0.60   # sobe com EMA >= 0,60 × episódio cheio
    twist_desce_frac: float = 0.8          # desce com EMA < 0,8 × alvo (histerese)
    twist_ema: float = 0.99                # τ ≈ 100 amostras ≈ 4 iterações (medido)
    # ⚠ Teto de UM degrau a cada N iterações. Sem ele, num warm-start com o passo
    # global além dos dois degraus e uma política que anda, o estágio saltaria
    # 0→2 em duas chamadas (0,08 iteração) com a EMA ainda medida nas faixas do
    # estágio 0 — a re-explosão da it 5099. 12 iterações ≈ 3τ da EMA.
    twist_iters_entre_degraus: int = 12
    # ⚠ `poc_estagio_twist` e a EMA NÃO vão para o checkpoint (o runner só salva
    # `common_step_counter`). Depois de um resume o gate recomeça pessimista
    # (estágio 0, EMA 0) e se recalibra em ~3τ ≈ 12 iterações. Declarado: é o
    # comportamento seguro, não um bug.
```

---

## Tarefa 2 — `g1_poc/eventos.py`

### 2.1 `reset_cena` lê o nível

Troque os parâmetros `topo_piso: float` e `jitter_x: tuple[float, float]` por
`topo_min_por_nivel: tuple[float, ...]` e `jitter_x_max_por_nivel: tuple[float, ...]`.

Substitua o bloco `# --- a altura do topo ---` por:

```python
    # --- a altura do topo, pela CÉLULA do nível (§10.1) ---
    # Só o PISO da faixa desce com o nível; o teto é 0,55 m em todos. No nível 0 a
    # faixa é degenerada em 0,55 — a cena de antes da tabela, número por número.
    # ⚠ `poc_nivel` já existe aqui mesmo no primeiro reset (o currículo roda
    # inteiro ANTES dos eventos); o `getattr` é defensivo, não necessário.
    nivel = getattr(env, "poc_nivel", None)
    if nivel is None:
        piso = torch.full((n,), topo_min_por_nivel[0], device=dev)
        jx_max = torch.full((n,), jitter_x_max_por_nivel[0], device=dev)
    else:
        piso = torch.tensor(topo_min_por_nivel, device=dev)[nivel[env_ids]]
        jx_max = torch.tensor(jitter_x_max_por_nivel, device=dev)[nivel[env_ids]]
    topo = piso + (topo_teto - piso) * torch.rand(n, device=dev)
    topo = topo + (2.0 * torch.rand(n, device=dev) - 1.0) * jitter_z
    topo = torch.maximum(topo, piso)
```

E o sorteio do `dx` da caixa passa a usar a célula (o piso do jitter é 0):

```python
    # o jitter x APERTA com o nível: no topo a 0,04 m o alcance acaba em ~0,45 m
    dx = jx_max * torch.rand(n, device=dev)
```

Atualize a docstring da função: a faixa de altura e o jitter x vêm da célula do
nível; `env.poc_topo` continua sendo gravado.

### 2.2 `carga_caixa` lê o nível

Troque `faixa_kg: tuple[float, float]` por `carga_max_por_nivel: tuple[float, ...]`.

Substitua o sorteio da carga por:

```python
    # o TETO vem da célula do nível; o PISO é sempre `massa_base` (§10.1)
    nivel = getattr(env, "poc_nivel", None)
    if nivel is None:
        teto = torch.full((n,), carga_max_por_nivel[0], device=dev)
    else:
        teto = torch.tensor(carga_max_por_nivel, device=dev)[nivel[env_ids]]
    kg = massa_base + (teto - massa_base).clamp(min=0.0) * torch.rand(n, device=dev)
```

### 2.3 Docstring do módulo

Acrescente ao `⚠` do topo: `reset_cena` e `carga_caixa` também leem `env.poc_nivel`,
escrito pelo mesmo currículo, que roda inteiro antes dos eventos — o `getattr` com
`None` é defesa, não necessidade (medido em 20/08).

**NÃO toque no `afasta_cena`.** Medido em 20/08: a caixa afastada fica apoiada na
laje a 5 m (z = 5,099 após 120 passos, mesmo com 5 kg).

---

## Tarefa 3 — `g1_poc/comando.py`

### 3.1 BUG: `env.poc_success` está congelado em zeros

**Medido em 20/08, duas vezes (independentes).** `__init__` faz
`env.poc_success = self.episode_success`, e `_update_command` faz
`self.episode_success = torch.maximum(...)` — que devolve tensor NOVO e religa o
atributo já na PRIMEIRA chamada. Consequência: `terminacoes.caixa_largada` lê
`env.poc_success` e vê sempre 0 — **essa terminação nunca disparou neste projeto**.
O `nivel_caixa` NÃO foi afetado (lê `cmd.episode_success` direto).

Conserto — escreva **no lugar**, nas duas linhas:

```python
        dt = self._env.step_dt
        # ⚠ `copy_`, e não atribuição. `torch.where`/`torch.maximum` devolvem tensor
        # NOVO, e o `__init__` publica `env.poc_success = self.episode_success`. Uma
        # atribuição religa o atributo e deixa o alias apontando para o tensor velho:
        # medido em 20/08, `env.poc_success` ficava em zeros para sempre e a
        # terminação `caixa_largada` nunca disparava.
        self._sustenta.copy_(torch.where(fecha, self._sustenta + dt,
                                         torch.zeros_like(self._sustenta)))
        self.episode_success.copy_(torch.maximum(
            self.episode_success,
            (self._sustenta >= self.cfg.sustenta_pegar_s).float(),
        ))
```

### 3.2 Remova o campo morto `frac_locomocao` do `CaixaAlvoCommandCfg`

`_resample_command` lê `env.poc_manipula` (escrito pelo currículo), nunca este campo.
Remova a linha `frac_locomocao: float = 0.30` da dataclass. (A remoção da passagem
no `env_cfg` está na tarefa 7.)

### 3.3 O σ inicial do `reaching`, por elo (§8.2, correção de 20/08)

O σ do `bringing` já é variável (`dist_inicial`). O do `reaching` não era, e é o
mesmo defeito uma coordenada antes: medido, o gradiente de aproximação cai 1391× do
nível 0 ao 4. A distância inicial palma→face é calculada AQUI, no `_resolver`,
contra a pose fresca — pelo mesmo motivo do `dist_inicial`.

1. `CaixaAlvoCommandCfg` ganha dois campos:

```python
    # σ inicial do `reaching` (§8.2): os sites das palmas e o piso do σ
    palm_sites: tuple[str, str] = ("left_palm", "right_palm")
    lateral_offset: float = 0.10
    reaching_std_piso: float = 0.20
```

(O `env_cfg` passa `C.PALM_SITES` e o `lateral` resolvido — tarefa 7.)

2. No `__init__`, depois de `self.robot = ...`:

```python
        # os sites das palmas, para o σ inicial do `reaching` (§8.2)
        self._palm_ids, _ = self.robot.find_sites(list(cfg.palm_sites))
        self.reach_inicial = torch.full(
            (self.num_envs,), cfg.reaching_std_piso, device=self.device)
```

e junto dos outros buffers publicados: `env.poc_reach_inicial = self.reach_inicial`.

3. No `_resolver`, junto do cálculo do `dist_inicial`:

```python
        # σ do `reaching` = a distância a vencer pelas PALMAS no começo do elo.
        # Mesma correção que a §8.2 fez no `bringing`: com σ fixo de 0,20 o
        # gradiente de aproximação cai 1391× entre a prateleira a 0,55 e a 0,04
        # (medido 20/08) — os níveis 3+ viravam sorte.
        from g1_poc.observacoes import alvos_das_palmas
        palmas = self.robot.data.site_pos_w[ids][:, self._palm_ids]
        alvos_p = alvos_das_palmas(self._env, "box", self.cfg.lateral_offset)[ids]
        d_p = torch.norm(palmas - alvos_p, dim=-1).mean(dim=-1)
        self.reach_inicial[ids] = torch.clamp(d_p, min=self.cfg.reaching_std_piso)
```

(Mova o import para o topo do arquivo se não criar ciclo — `observacoes` não importa
`comando`, então não cria.)

### 3.4 Só comentário: o giro do `pegar` continua 0

**Mantenha** `self._ang[env_ids] = 0.0`. Atualize o comentário acima:

```python
        # --- a face e o giro pedido ---
        # O `pegar` pede SEMPRE "erga sem torcer": `dir_alvo` recebe a normal ATUAL
        # da face. A rotação da célula do nível (§10.1) pertence ao elo
        # `reorientar` — pedi-la aqui tornaria as duas cadeias do nível 3 a mesma
        # tarefa, e o `reorientar` deixaria de ter função.
```

---

## Tarefa 4 — `g1_poc/postura.py`

### 4.1 O regime pela FORMA do episódio

Na assinatura de `__call__`, **remova** `peso_dist`, `peso_ang` e `limiar`.
Mantenha `walking_threshold` e `running_threshold`.

Substitua o bloco `# --- o quarto regime, pela demanda da caixa ---` por:

```python
        # --- o quarto regime, pela FORMA do episódio ---
        # ⚠ Ele já foi escolhido por LIMIAR DE DEMANDA
        # (`demanda = 10·‖caixa−alvo‖ + 6·Δθ`, troca em 1,5), e aquilo era um
        # PENHASCO, não um gradiente.
        #
        # O G1 usa `std_standing = {".*": 0.05}` — 0,05 rad em TODA junta, braços
        # incluídos (`mjlab/tasks/velocity/config/g1/env_cfgs.py:107`). Com a caixa
        # segurada a 0,82 m, quatro juntas de braço ficam a ≈0,7 rad do default:
        #
        #   std_manipulando (σ ombro/cotovelo 1,00) -> `pose` ≈ 0,93
        #   standing        (σ 0,05)                -> `pose` ≈ 0,00
        #
        # Trocar de regime com a caixa na mão DESTRUÍA o termo inteiro: o robô
        # pagava ≈0,93/s por TERMINAR a tarefa. E o penhasco caía dentro da
        # tolerância da régua — a troca exigia Δθ < 13,3° e o sucesso aceita 20°.
        # Medido na it 5217: demanda 1,60 contra limiar 1,5, Δθ preso em 14,2°.
        #
        # A §9.4 afirmava o contrário e foi retificada. Quem levanta o robô agora é
        # o termo `postura_ereta` (§8.2.3), que é RAMPA e não penhasco.
        bit = env.command_manager.get_term(caixa_command_name).command[:, 9]
        m_manip = (bit > 0.5).float().unsqueeze(1)
```

Mantenha `env.extras["log"]["Metrics/postura_frac_manipulando"] = m_manip.mean()`.
**Remova** o log `Metrics/postura_demanda_caixa`.

### 4.2 Docstring do módulo

Reescreva a parte "A solução é um quarto regime, e ele lê a DEMANDA DA CAIXA" para
descrever o gate pela forma, com os dois valores medidos de `pose` (0,93 × ≈0) e o
ponteiro para `postura_ereta`.

---

## Tarefa 5 — `g1_poc/recompensas.py`

### 5.1 Auxiliar `_preensao`

```python
def _preensao(env: ManagerBasedRlEnv, palm_sensors: tuple[str, ...]) -> torch.Tensor:
    """[B] bool — as DUAS palmas em contato com a caixa.

    Gate compartilhado por `unload` e `postura_ereta`. Uma palma sozinha não conta:
    sem preensão bimanual os dois termos pagariam por caminhos que não são erguer.
    """
    pega: torch.Tensor | None = None
    for nome in palm_sensors:
        found = env.scene[nome].data.found
        assert found is not None, f"sensor '{nome}' precisa do field 'found'."
        aqui = (found > 0).any(dim=-1)
        pega = aqui if pega is None else (pega & aqui)
    assert pega is not None, "palm_sensors vazio"
    return pega
```

Troque o laço equivalente dentro de `unload` por `pega = _preensao(env, palm_sensors)`.

### 5.2 `_reaching` com σ por elo

Substitua o corpo de `_reaching` para usar o buffer do comando quando ele existir:

```python
def _reaching(env, object_name, lateral_offset, std, asset_cfg):
    """O `reaching` BIMANUAL, com σ POR ELO.

    (docstring atual, mais:)
    ⚠ O σ é `max(std, distância inicial palma→face do elo)` — `env.poc_reach_inicial`,
    escrito pelo comando no começo do elo. Com σ fixo o gradiente de aproximação cai
    1391× entre a prateleira a 0,55 m e a 0,04 m (medido 20/08): os níveis 3+ do
    currículo viravam sorte. É a MESMA correção que a §8.2 fez no `bringing`.
    `std` vira o PISO (e o fallback quando o buffer não existe — smoke chama a
    função fora do laço do env).
    """
    robot: Entity = env.scene[asset_cfg.name]
    palmas = robot.data.site_pos_w[:, asset_cfg.site_ids]
    alvos = alvos_das_palmas(env, object_name, lateral_offset)
    d2 = torch.sum(torch.square(palmas - alvos), dim=-1)
    sigma = getattr(env, "poc_reach_inicial", None)
    if sigma is None:
        return torch.exp(-d2.mean(dim=-1) / std**2)
    return torch.exp(-d2.mean(dim=-1) / sigma.clamp(min=std) ** 2)
```

### 5.3 Termo novo `postura_ereta`

Depois de `unload`, antes de `joint_vel_hinge`:

```python
def postura_ereta(
    env: ManagerBasedRlEnv,
    command_name: str,
    palm_sensors: tuple[str, ...],
    support_sensor: str,
    massa_attr: str,
    pelve_min: float,
    rampa: float,
    rampa_fina: float,
    frac_descarga: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Rampa contínua na altura da pelve, gateada por preensão E descarga (§8.2.3).

    A condição 3 do fecho do `pegar` (§7.2) exige `pelve >= 0,65 m` e nada pagava
    por ela na coordenada certa: quem precifica a pelve é só a `pose`, a ~0,73/m
    (o default do G1 é o KNEES_BENT_KEYFRAME, pelve em 0,76 m). O `precise_pos` é
    INDIFERENTE à pelve abaixo do alvo, e CONTRÁRIO acima (−16,2/m no fecho) —
    por isso a rampa tem uma parte FINA, íngreme perto de `pelve_min`.

    A rampa em DUAS partes (medido 20/08):
        fracao = 0,5·clamp((z − (pelve_min − rampa)) / rampa)
               + 0,5·clamp((z − (pelve_min − rampa_fina)) / rampa_fina)
    longa 0,20→0,65 (o nível 4 pega com a pelve a 0,267 m — sem ela, zona morta em
    33% das pegas) e fina 0,57→0,65 (14,7/m com peso 2,0, contra os −16,2/m do
    `precise_pos` no fecho). Satura em `pelve_min`: a régua não pede mais, e pagar
    por mais convidaria a ponta dos pés.

    Os DOIS gates, e o que cada um fecha (mesmo idioma do `unload`):
    - preensão bimanual: antes de ter a caixa, agachar para a pega baixa sai de
      graça — sem este gate o termo brigaria com o `staged`.
    - descarga (`F_apoio < frac_descarga·m·g`): sem ele, encostar as palmas e ficar
      de pé com a caixa APOIADA paga a rampa inteira — +2,0/s exatamente no platô
      "encosta e para" que o bloco 1 mediu.
    """
    robot: Entity = env.scene[asset_cfg.name]
    z = robot.data.root_link_pos_w[:, 2]
    f_longa = ((z - (pelve_min - rampa)) / rampa).clamp(0.0, 1.0)
    f_fina = ((z - (pelve_min - rampa_fina)) / rampa_fina).clamp(0.0, 1.0)
    fracao = 0.5 * f_longa + 0.5 * f_fina

    f = env.scene[support_sensor].data.force
    assert f is not None, f"sensor '{support_sensor}' precisa do field 'force'."
    apoio_z = f[..., 2].abs().sum(dim=-1)
    peso = (getattr(env, massa_attr) * 9.81).clamp(min=1e-3)
    descarregada = apoio_z < frac_descarga * peso

    pega = _preensao(env, palm_sensors)
    return fracao * pega.float() * descarregada.float() * _valida(env, command_name)
```

### 5.4 Termo novo `sustentacao`

```python
def sustentacao(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rampa no TEMPO dentro da condição de fecho (§8.2.4).

    O fecho exige as 4 condições por 1,0 s ININTERRUPTO, e nenhum termo
    diferenciava 0,98 s de 0,00 s — o degrau que fazia o push (o único fator que
    degradava o sucesso, medido) decidir o currículo. Esta é a rampa na coordenada
    que faltava: o cronômetro do próprio comando.
    """
    cmd = env.command_manager.get_term(command_name)
    fracao = (cmd._sustenta / cmd.cfg.sustenta_pegar_s).clamp(0.0, 1.0)
    return fracao * _valida(env, command_name)
```

### 5.5 Docstring do módulo

De "Os 5 termos de tarefa" para **"Os 8 termos de tarefa"** (staged, precise_pos,
precise_ori, squeeze, unload, postura_ereta, sustentacao, joint_vel_hinge — o último
é qualidade de movimento). Todos menos o `joint_vel_hinge` multiplicam por
`caixa_valida`.

---

## Tarefa 6 — `g1_poc/curriculo.py`

### 6.1 Renomeie a métrica e atualize a docstring de `sorteia_forma`

A chave do retorno vira `frac_manipula_pop`. Acrescente à docstring:

```python
    ⚠ `frac_manipula_pop` é a fração POPULACIONAL, e NÃO o sorteio. Em regime ela é
    a fatia de TRANSIÇÕES — que é o que o PPO aprende — e essa fatia é governada
    pelo TEMPO DE VIDA do episódio, não pelo sorteio:

        0,30 × 24 passos / (0,30 × 24 + 0,70 × 961) = 1,06%

    O sorteio é `Episodio.frac_locomocao`; esta métrica é o resultado.

    ⚠ Este termo SOBRESCREVE `env.poc_manipula` com a forma do episódio NOVO.
    Quem precisa da forma do episódio que ACABOU (`nivel`, `twist_ranges`) tem de
    vir ANTES dele no dict de currículo. Medido 20/08: com `nivel` depois de
    `forma`, a promoção era gateada pela forma do episódio SEGUINTE — `p_up`
    caía de p para 0,7·p, e um episódio de LOCOMOÇÃO rebaixava o nível em 70%
    das vezes.
```

### 6.2 `nivel_caixa`: máscara defensiva, `nivel_forcado`, histograma

```python
def nivel_caixa(env, env_ids, command_name, nivel_forcado: int | None = None):
```

Logo depois de criar o buffer:

```python
    # ⚠ Atalho de MEDIÇÃO, não de treino. O `play`, a `sonda` e o `smoke` fixam o
    # nível para conferir a célula (§10.1); no treino fica em None. Forçar
    # `env.poc_nivel` de fora não funciona: este termo roda no reset e aplicaria o
    # delta ±1 por cima.
    if nivel_forcado is not None:
        env.poc_nivel[:] = int(nivel_forcado)
        return _metricas_nivel(env)

    # ⚠ A forma do episódio que ACABOU. Este termo roda ANTES de `sorteia_forma`
    # (ordem do dict, tarefa do env_cfg) — depois dele, a máscara já seria a do
    # episódio NOVO, e a promoção viraria moeda enviesada (medido 20/08:
    # p_up = 0,7·p; ponto fixo saía de 0,5 para 0,714). No primeiríssimo reset o
    # buffer ainda não existe (quem o cria é o `sorteia_forma`): sem forma que
    # tenha ACABADO, nada se promove.
    manipula = getattr(env, "poc_manipula", None)
    if manipula is None:
        return _metricas_nivel(env)
    manipula = manipula[env_ids]
```

(segue o corpo atual). E o retorno vira:

```python
def _metricas_nivel(env) -> dict[str, torch.Tensor]:
    """Média, extremos e as duas pontas do histograma.

    ⚠ A média sozinha mente duas vezes: `nivel_medio` "sai de zero" já com
    p = 0,006 (0,0042 — 17 envs de 4096 no nível 1), e `nivel_max` satura em 6 com
    UM env sortudo. As frações dizem onde a POPULAÇÃO está.
    """
    niveis = env.poc_nivel.float()
    return {
        "nivel_medio": niveis.mean(),
        "nivel_max": niveis.max(),
        "nivel_min": niveis.min(),
        "nivel_frac_0": (env.poc_nivel == 0).float().mean(),
        "nivel_frac_3mais": (env.poc_nivel >= 3).float().mean(),
    }
```

### 6.3 Termo novo `twist_por_competencia`

```python
def twist_por_competencia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list,
    duracao_min_frac: float,
    desce_frac: float,
    ema: float,
    iters_entre_degraus: int,
) -> dict[str, torch.Tensor]:
    """Avança as faixas do twist só quando o robô SUSTENTA o teto atual (§10.3).

    Substitui o `mdp.commands_vel`, que avança por passo global e só por isso —
    na it 5099 o robô recebia 2,0 m/s com `peak_height_mean = 2,7 mm`. O sinal de
    competência é a DURAÇÃO do episódio de locomoção: direto (quem não anda cai em
    24 passos) e é a MESMA grandeza que governa a fatia de transições.

    Regra: sobe quando (passo global >= degrau) E (EMA >= duracao_min_frac ×
    episódio cheio) E (passou `iters_entre_degraus` desde o último degrau);
    DESCE quando EMA < desce_frac × alvo. O degrau global é PISO, não gatilho.

    ⚠ ANTES de `forma` no dict: `sorteia_forma` sobrescreve `env.poc_manipula`, e
    aqui precisamos da forma do episódio que ACABOU.
    ⚠ `episode_length_buf[env_ids]` só zera no FIM do `_reset_idx` — aqui ainda
    vale a duração final. Medido.
    ⚠ Os envs PARADOS (`is_standing_env`) saem da EMA: ficar de pé até o time_out
    entregaria 8% do alvo sem andar. Os de giro no lugar CONTAM — girar é andar.
    ⚠ A EMA nasce PESSIMISTA (zero) e é tensor no device (uma sync por reset já
    basta para o degrau; 48 syncs/iteração não).
    ⚠ Nada disto vai para o checkpoint: depois de um resume o gate recomeça em
    (0, 0) e recalibra em ~3τ ≈ 12 iterações. Seguro por construção.
    """
    if not hasattr(env, "poc_estagio_twist"):
        env.poc_estagio_twist = 0
        env.poc_duracao_loco = torch.zeros((), device=env.device)
        env.poc_twist_ultimo_degrau = 0

    manipula = getattr(env, "poc_manipula", None)
    if manipula is not None and len(env_ids) > 0:
        loco = env_ids[~manipula[env_ids]]
        if len(loco) > 0:
            parado = env.command_manager.get_term(command_name).is_standing_env
            loco = loco[~parado[loco]]
        if len(loco) > 0:
            amostra = env.episode_length_buf[loco].float().mean()
            env.poc_duracao_loco = ema * env.poc_duracao_loco + (1.0 - ema) * amostra

    alvo = duracao_min_frac * env.max_episode_length
    dur = float(env.poc_duracao_loco)
    est = env.poc_estagio_twist
    passo = env.common_step_counter
    pode = passo - env.poc_twist_ultimo_degrau >= iters_entre_degraus * 24
    if (pode and est + 1 < len(velocity_stages)
            and passo >= velocity_stages[est + 1]["step"]
            and dur >= alvo):
        est += 1
        env.poc_twist_ultimo_degrau = passo
    elif pode and est > 0 and dur < desce_frac * alvo:
        est -= 1
        env.poc_twist_ultimo_degrau = passo
    env.poc_estagio_twist = est

    cfg = env.command_manager.get_term(command_name).cfg
    estagio = velocity_stages[est]
    for chave in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if estagio.get(chave) is not None:
            setattr(cfg.ranges, chave, estagio[chave])

    dev = env.device
    return {
        "estagio": torch.tensor(float(est), device=dev),
        "duracao_loco_ema": torch.tensor(dur, device=dev),
        "duracao_alvo": torch.tensor(alvo, device=dev),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=dev),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=dev),
    }
```

### 6.4 Docstring do módulo

Atualize o bloco "ESTADO DESTE ARQUIVO": tabela de células ligada (§10.1), gate por
competência no lugar do `commands_vel`, faltam as cadeias (MACRO 2). E registre a
regra de ordem: **`twist_ranges` e `nivel` leem a forma do episódio que ACABOU e vêm
ANTES de `forma`; os eventos leem a forma NOVA e rodam depois de todo o currículo.**

---

## Tarefa 7 — `g1_poc/env_cfg.py`

Na ordem do arquivo:

1. **Docstring do topo:** item 7 vira "os 8 termos de tarefa"; item 10 menciona o
   gate por competência e a ordem do currículo.
2. **`CaixaAlvoCommandCfg`:** remova `frac_locomocao=ke.frac_locomocao,` e acrescente
   `palm_sites=C.PALM_SITES, lateral_offset=lateral, reaching_std_piso=kr.reaching_std,`.
3. **`pose.params`:** remova `peso_dist`, `peso_ang`, `limiar`; troque
   `"running_threshold": kp.limiar` por `kp.running_threshold`.
4. **Termos novos**, depois do bloco do `unload`:

```python
    # a RAMPA da pelve (§8.2.3): a condição 3 do fecho, que nenhum termo pagava.
    cfg.rewards["postura_ereta"] = RewardTermCfg(
        func=R.postura_ereta, weight=kr.postura_ereta,
        params={"command_name": CMD_CAIXA, "palm_sensors": C.SENSOR_PALMA,
                "support_sensor": C.SENSOR_APOIO, "massa_attr": "poc_massa",
                "pelve_min": kt.pelve_min, "rampa": kr.postura_ereta_rampa,
                "rampa_fina": kr.postura_ereta_rampa_fina,
                "frac_descarga": kr.postura_ereta_frac_descarga,
                "asset_cfg": SceneEntityCfg("robot")},
    )
    # a RAMPA da sustentação (§8.2.4): 0,98 s e 0,00 s pagavam o mesmo.
    cfg.rewards["sustentacao"] = RewardTermCfg(
        func=R.sustentacao, weight=kr.sustentacao,
        params={"command_name": CMD_CAIXA},
    )
```

5. **`reset_cena` params:** `"topo_min_por_nivel": k.celulas.topo_min,` no lugar de
   `"topo_piso": ...`, e `"jitter_x_max_por_nivel": k.celulas.jitter_x_max,` no
   lugar de `"jitter_x": kc.caixa_jitter_x,`.
6. **`carga_caixa` params:** `"carga_max_por_nivel": k.celulas.carga_max` no lugar
   de `"faixa_kg": kd.carga_kg`.
7. **Dict de currículo — A ORDEM É A CORREÇÃO** (bug medido 20/08):

```python
    cr = k.cronograma
    cfg.curriculum = {
        # ⚠ ORDEM: `twist_ranges` e `nivel` leem a forma do episódio que ACABOU, e
        # `forma` a SOBRESCREVE com o sorteio do episódio novo. Com `nivel` depois
        # de `forma` (o bug), a promoção era gateada pela forma do episódio
        # SEGUINTE: p_up = 0,7·p, o ponto fixo saía de 0,5 para 0,714, e um bloco
        # com frac_locomocao = 0,85 limitaria nivel_medio a 0,214 mesmo com
        # manipulação perfeita. Os eventos leem a forma NOVA e rodam DEPOIS de todo
        # o currículo, portanto não são afetados.
        "twist_ranges": CurriculumTermCfg(
            func=CU.twist_por_competencia,
            params={"command_name": CMD_TWIST, "velocity_stages": cr.locomocao,
                    "duracao_min_frac": cr.twist_duracao_min_frac,
                    "desce_frac": cr.twist_desce_frac, "ema": cr.twist_ema,
                    "iters_entre_degraus": cr.twist_iters_entre_degraus},
        ),
        "nivel": CurriculumTermCfg(
            func=CU.nivel_caixa,
            params={"command_name": CMD_CAIXA, "nivel_forcado": None},
        ),
        "forma": CurriculumTermCfg(
            func=CU.sorteia_forma,
            params={"frac_locomocao": ke.frac_locomocao},
        ),
        "hinge": CurriculumTermCfg(...),        # inalterado
        "action_rate": CurriculumTermCfg(...),  # inalterado
    }
```

8. **Bloco `if play:`** — inalterado (ele já remove `twist_ranges`/`hinge`/
   `action_rate` e mantém `forma` e `nivel`).

---

## Tarefa 8 — `g1_poc/smoke.py`

### 8.1 Contagens e listas

- Seção 5: `n_rew == 22` (13 fundação + `self_collisions` + 8 de tarefa), comentário
  com a conta nova.
- Tupla `tarefa` ganha `"postura_ereta"` e `"sustentacao"`.
- Seção 7 (bit=0): acrescente `"postura_ereta"` e `"sustentacao"` ao laço.
- Seção 12: a checagem `ordem.index("forma") < ordem.index("nivel")` **INVERTE**:

```python
    ordem = list(cfg.curriculum)
    checa(ordem.index("twist_ranges") < ordem.index("forma")
          and ordem.index("nivel") < ordem.index("forma"),
          f"`twist_ranges` e `nivel` vêm ANTES de `forma` (ordem: {ordem}) — os dois "
          f"leem a forma do episódio que ACABOU, e `forma` a sobrescreve. Medido "
          f"20/08: com `nivel` depois, p_up = 0,7·p e locomoção rebaixava o nível")
```

### 8.2 Seção nova `11c. as rampas e os gates dos termos novos`

Depois da 11b (a caixa está no chão, longe das palmas e da prateleira):

```python
    print("== 11c. as rampas e os gates dos termos novos ==")
    tc_ereta = env.reward_manager.get_term_cfg("postura_ereta")
    v = tc_ereta.func(env, **tc_ereta.params)
    checa(bool((v.abs() < 1e-6).all()),
          f"sem preensão bimanual, `postura_ereta` é zero (medido max "
          f"{float(v.abs().max()):.3e}) — agachar para alcançar sai de graça")
    # a forma da rampa em duas partes, sobre a fórmula
    kt2, kr2 = k.tol, k.reward
    def rampa2(z):
        fl = min(max((z - (kt2.pelve_min - kr2.postura_ereta_rampa))
                     / kr2.postura_ereta_rampa, 0.0), 1.0)
        ff = min(max((z - (kt2.pelve_min - kr2.postura_ereta_rampa_fina))
                     / kr2.postura_ereta_rampa_fina, 0.0), 1.0)
        return 0.5 * fl + 0.5 * ff
    for z, esperado in ((0.20, 0.0), (0.425, 0.25), (0.65, 1.0), (0.75, 1.0)):
        f = rampa2(z)
        checa(abs(f - esperado) < 1e-9,
              f"rampa da pelve: z = {z:.3f} -> {f:.3f} (esperado {esperado:.2f})")
    checa(rampa2(0.61) > rampa2(0.57) + 0.25,
          "a parte FINA é íngreme: 4 cm perto do fecho valem mais que 25% da rampa")
    # a sustentação é rampa no cronômetro do comando
    tc_sus = env.reward_manager.get_term_cfg("sustentacao")
    cmd._sustenta[:] = 0.5 * cmd.cfg.sustenta_pegar_s
    v = tc_sus.func(env, **tc_sus.params)
    esperado_s = 0.5 * float(cmd.valida.max())
    checa(bool(((v - 0.5 * cmd.valida).abs() < 1e-6).all()),
          "meio cronômetro paga meia `sustentacao` (× valida)")
    cmd._sustenta[:] = 0.0
    # o σ do reaching é por elo, com piso
    checa(bool((env.poc_reach_inicial >= kr2.reaching_std - 1e-6).all()),
          f"`poc_reach_inicial` respeita o piso de {kr2.reaching_std} "
          f"(mín medido {float(env.poc_reach_inicial.min()):.3f})")
    checa(bool(torch.isfinite(env.poc_reach_inicial).all()),
          "`poc_reach_inicial` é finito")
```

### 8.3 Seção nova `14. a tabela de células (§10.1)`

Depois da seção 13, forçando o nível pelo param do manager:

```python
    print("== 14. a tabela de células (§10.1) ==")
    cel = k.celulas
    tc_nivel = env.curriculum_manager.get_term_cfg("nivel")
    todos = torch.arange(N_ENVS, device=env.device)
    jit = k.cena.prateleira_jitter_z
    teto = k.cena.prateleira_topo_teto

    def cena_no_nivel(n: int):
        tc_nivel.params["nivel_forcado"] = n
        env._reset_idx(todos)
        env.sim.forward()
        return env.poc_topo.clone(), env.poc_massa.clone()

    topo0, massa0 = cena_no_nivel(0)
    checa(bool(((topo0 >= teto - jit - 1e-6) & (topo0 <= teto + jit + 1e-6)).all()),
          f"nível 0 é NO-OP: topo em 0,55 ± jitter (medido {float(topo0.min()):.3f}"
          f" a {float(topo0.max()):.3f})")
    checa(bool((massa0 - k.cena.caixa_massa).abs().max() < 1e-6),
          f"nível 0 é NO-OP: carga fixa em {k.cena.caixa_massa:.1f} kg")

    for n in range(NIVEL_MAX + 1):
        topo, massa = cena_no_nivel(n)
        checa(float(topo.min()) >= cel.topo_min[n] - jit - 1e-6,
              f"nível {n}: topo >= {cel.topo_min[n]:.2f} − jitter "
              f"(medido {float(topo.min()):.3f})")
        checa(float(topo.max()) <= teto + jit + 1e-6,
              f"nível {n}: o TETO continua {teto:.2f} (medido {float(topo.max()):.3f})")
        checa(float(massa.max()) <= cel.carga_max[n] + 1e-6,
              f"nível {n}: carga <= {cel.carga_max[n]:.1f} kg "
              f"(medido {float(massa.max()):.2f})")
        checa(float(massa.min()) >= k.cena.caixa_massa - 1e-6,
              f"nível {n}: o PISO da carga continua {k.cena.caixa_massa:.1f} kg")

    t0, _ = cena_no_nivel(0)
    t4, c4 = cena_no_nivel(4)
    checa(float(t4.min()) < float(t0.min()) - 0.20,
          f"promover 0 -> 4 BAIXA a prateleira ({float(t0.min()):.3f} -> "
          f"{float(t4.min()):.3f})")
    checa(float(c4.max()) > 2.0,
          f"promover 0 -> 4 SOBE a carga (máx medido {float(c4.max()):.2f} kg)")
    tc_nivel.params["nivel_forcado"] = None
```

### 8.4 Seção nova `15. a promoção usa a forma do episódio que ACABOU`

A regressão do bug de 20/08 — roda depois da 14 (o `nivel_forcado` já voltou a None):

```python
    print("== 15. a promoção usa a forma que ACABOU (bug de 20/08) ==")
    tc_n = env.curriculum_manager.get_term_cfg("nivel")
    env.poc_nivel[:] = 2
    env.poc_manipula[:] = True                      # a forma do episódio que acabou
    cmd.episode_success.copy_(torch.ones(N_ENVS, device=env.device))
    tc_n.func(env, todos, **tc_n.params)            # o termo, ISOLADO do `forma`
    checa(bool((env.poc_nivel == 3).all()),
          f"manipulação + sucesso promove TODOS (medido "
          f"{int((env.poc_nivel == 3).sum())}/{N_ENVS})")
    env.poc_manipula[:] = False                     # locomoção que acabou
    cmd.episode_success.copy_(torch.zeros(N_ENVS, device=env.device))
    tc_n.func(env, todos, **tc_n.params)
    checa(bool((env.poc_nivel == 3).all()),
          "episódio de LOCOMOÇÃO não move o nível (nem para baixo)")
```

### 8.5 Seção nova `16. o gate por competência do twist (§10.3)`

Por ÚLTIMO, antes do `env.close()` (mexe em `common_step_counter`):

```python
    print("== 16. o gate por competência do twist (§10.3) ==")
    tc_tw = env.curriculum_manager.get_term_cfg("twist_ranges")
    degrau = k.cronograma.locomocao[1]["step"]
    env.common_step_counter = degrau + k.cronograma.twist_iters_entre_degraus * 24 + 1
    env.poc_estagio_twist = 0
    env.poc_duracao_loco = torch.zeros((), device=env.device)
    env.poc_twist_ultimo_degrau = 0
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 0,
          "passo global acima do degrau mas duração baixa: o estágio SEGURA")
    tw_cfg = env.command_manager.get_term("twist").cfg
    checa(tuple(tw_cfg.ranges.lin_vel_x) == tuple(k.cronograma.locomocao[0]["lin_vel_x"]),
          f"e a faixa fica no estágio 0 (medido {tw_cfg.ranges.lin_vel_x})")
    env.poc_duracao_loco = torch.tensor(float(env.max_episode_length), device=env.device)
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 1,
          "com a duração no alvo, o estágio SOBE")
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 1,
          "e NÃO sobe de novo na mesma janela (teto de 1 degrau por 12 iterações)")
    env.common_step_counter += k.cronograma.twist_iters_entre_degraus * 24 + 1
    env.poc_duracao_loco = torch.tensor(0.0, device=env.device)
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 0,
          "com a duração degradada, o estágio DESCE (histerese de 0,8×alvo)")
```

### 8.6 O rodapé "NÃO coberto"

Remova a linha da tabela de células. As linhas das cadeias e do movimento da
prateleira ficam (MACRO 2).

---

## Tarefa 9 — `g1_poc/play.py` e `g1_poc/sonda.py` — flag `--nivel`

### 9.1 `play.py`

`p.add_argument("--nivel", type=int, default=None, help="--pegar/--geometria: força a célula do nível (§10.1); default = promoção por sucesso")`.

Valide `0 <= nivel <= 6` (senão `SystemExit`) e rejeite junto com `--andar` (a
mobília é afastada; o nível não tem efeito). Em `_registra`:

```python
def _registra(task_id: str, ajusta, nivel: int | None = None) -> str:
    ...
    env_cfg = make_g1_poc_env_cfg(k, play=True)
    if nivel is not None:
        # o termo `nivel` FICA no play; forçar aqui congela a célula
        env_cfg.curriculum["nivel"].params["nivel_forcado"] = int(nivel)
    ...
```

O `task_id` ganha o sufixo `f"-N{nivel}"` quando forçado (`register_mjlab_task` não
sobrescreve). `imprime_geometria` ganha uma linha por nível com a célula
(`topo_min`, `carga_max`, `ang_max_deg`, `jitter_x_max`), lendo de `k.celulas`.

### 9.2 `sonda.py`

Mesma flag, repassada ao `_registra`. Imprima o nível no cabeçalho junto da massa.

---

## Tarefa 10 — `ESPECIFICACAO-g1_poc.md`

1. **§8 cabeçalho:** 20 → **22** termos (`13 + 1 + 8`); ajuste a frase do smoke.
2. **§8.2:** título "Tarefa — 8 termos"; tabela ganha `postura_ereta` (+2,0,
   `rampa2(pelve) × preensão × descarga`, §8.2.3) e `sustentacao` (+0,5,
   `clamp(t/1s)`, §8.2.4); os gates `×= caixa_valida` incluem os dois; registre o σ
   POR ELO do `reaching` ao lado do σ variável do `bringing`, com o número (1391×).
3. **§8.2.3 nova** — conteúdo obrigatório: os números medidos (pelve 0,6345 ×
   0,65; 34% de pé; −16,2/m do `precise_pos` no fecho contra 14,7/m da rampa fina);
   a justificativa CORRETA (a `pose` precifica a pelve a ~0,73/m; o `precise_pos` é
   indiferente abaixo do alvo e contrário acima — a atribuição "paga por agachar"
   de uma versão anterior está errada e é retificada aqui); os dois gates e o que
   cada um fecha (o de descarga mata o bônus do platô "encosta e para").
4. **§8.2.4 nova** — a rampa da sustentação e o porquê (o push era o único fator
   que degradava, e 0,98 s ≡ 0,00 s em recompensa).
5. **§8.3:** refazer com os números medidos: d₀ = 0,256 (não 0,18), `squeeze` em B =
   0,7616 (não 1,00), fundação 5,74/s no reset; com o `unload`, erguer paga
   **+54,7%** no episódio (não 31%); nota de que `postura_ereta` gateado por
   descarga NÃO paga no estado B.
6. **§9.2–§9.4 — retificação:** o regime é pela FORMA; a tabela do penhasco (0,93 ×
   ≈0); o limiar caía dentro da tolerância da régua (13,3° × 20°).
7. **§10.1:** tabela IMPLEMENTADA (altura, carga, jitter x por célula; a rotação
   pertence ao `reorientar`, MACRO 2); o bug da máscara (nivel lia a forma do
   episódio SEGUINTE; p_up = 0,7·p; consertado pela ORDEM do dict); o portão do
   passo 2 da §17 vira `nivel_medio ≥ 1,0` com o histograma
   (`nivel_frac_0`/`nivel_frac_3mais`), porque "sair de zero" já vale com p = 0,006.
8. **§10.2:** o gate por competência substitui o `commands_vel` — com DESCIDA
   (histerese 0,8×alvo), teto de 1 degrau/12 iterações, EMA sem os envs parados; o
   estado do gate não persiste no checkpoint (declarado: recalibra em ~12
   iterações).
9. **§12:** `⚠` — a `caixa_largada` nunca disparou até 20/08 (alias religado de
   `env.poc_success`); consertada, vigiar `Episode_Termination/caixa_largada` no
   primeiro bloco.
10. **§19, Risco 3 — INVERTIDO por medição:** o pico de `contato_ilegal` é no topo
    **0,30 m** (20–25% das poses de pega), não a 0,04 m (0–3%) — a 4 cm a laje é um
    degrau e o corpo passa por cima. A mitigação (prateleira em x = 0,60) vira
    contingência do NÍVEL 2. E o Risco 9 ganha a nota do alcance: sem o jitter por
    célula, 60% dos episódios do nível 4 exigiriam um passo com o twist zerado.
11. **§7.1, nota no `reorientar`:** girar a caixa EM MÃO é geometricamente
    impossível (diagonal a 45° = 0,283 m contra 0,20 m de vão das palmas); a
    solução compatível com o desenho é EMPURRAR a caixa apoiada (alvo = posição
    atual, 2 condições, não exige erguer) — registrado para a MACRO 2.
12. **§16:** atualizar a lista do smoke (células, rampas, ordem do currículo, gate
    por competência) e o que falta (cadeias).

---

## Critério de pronto

```
cd /home/joaobornelli/Documents/g1_training
.venv/bin/python -m g1_poc.smoke        # 0 falhas, contagem sobe de 52 para >= 90
.venv/bin/python -m g1_poc.cena         # continua compilando
```

Relate no fim: contagem do smoke antes/depois, e qualquer ponto em que a medição
discordou do que está escrito aqui — se discordar, **pare e relate**, não "conserte"
o spec sozinho.
