# Spec de implementação — MACRO 2: a máquina de elo e as cadeias (§7)

Executa a Fase 4 do plano, com as decisões fechadas pelas auditorias de 20/08 (adendo do
plano). As cadeias entram GATEADAS pelo nível: `reorientar` no 3, `carregar` no 4, `botar`
no 5 — e o nível só sobe com sucesso. Nenhum gate novo é necessário.

**Repositório:** `/home/joaobornelli/Documents/g1_training`, branch `exp/g1-poc`.
**Só edite `g1_poc/`.** NUNCA edite `g1_training/`, `.venv/`, `docs/`, `specs/` nem a
`ESPECIFICACAO-g1_poc.md` (o revisor cuida dela).
**Idioma: português**, no estilo dos arquivos (docstring com o porquê, `⚠` para armadilha).
**Não commite.** Se algo do spec não casar com o arquivo, PARE a tarefa e relate.

Verificação: `cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_poc.smoke`
tem de terminar com **0 falhas** depois da tarefa 8.

---

## Contexto medido (não deduza — está tudo auditado)

- A máquina de elo roda no `_update_command`, que executa DEPOIS do único `sim.forward()`
  do step: as poses estão FRESCAS para leitura, e uma escrita de mocap só vale no passo
  seguinte (bom o bastante para teleportar a laje). **NÃO chame `sim.forward()`** — custa
  um forward global por passo.
- `write_mocap_pose_to_sim(pose, env_ids=...)` aceita subconjunto; a prateleira é mocap.
- **Buffers publicados em `env.*` são atualizados IN-PLACE** (`copy_`, indexação), nunca
  reatribuídos — a lição do alias `poc_success` (bug consertado em 20/08).
- `de_pe` só entra no fecho do `pegar` (§7.2). `reorientar` e `botar` têm 2 e 3 condições.
- `Entity.find_sites` existe; `quat_apply`/`quat_apply_inverse` já são importados onde
  precisa.

---

## O desenho, numa página

Elos por id: `PEGAR = 0 · REORIENTAR = 1 · CARREGAR = 2 · BOTAR = 3`.

| cadeia | elos | prateleira quando o `pegar` fecha |
|---|---|---|
| 0 | `pegar` | não se move |
| 1 | `reorientar` → `pegar` | não se move |
| 2 | `pegar` → `carregar` | **+5 m** (só a MESA; a caixa está nas mãos) |
| 3 | `pegar` → `botar` | topo NOVO, sorteado na faixa da colocação |

| elo | alvo | twist | fecho | sustenta |
|---|---|---|---|---|
| `pegar` | mundo, sorteado | 0 | perto & alinhado & de_pe | 1,0 s |
| `reorientar` | posição ATUAL da caixa; `dir_alvo` girado ±U(0, ang_max[nivel]) | 0 | perto & alinhado | 0,5 s |
| `carregar` | `base + peito_b`, recalculado A CADA PASSO | sorteado | perto, com `elo_t ≥ 6 s` | 0,5 s |
| `botar` | topo novo + 0,10 · x 0,30–0,40 · y ±0,12 | 0 | perto & alinhado & apoiada (`F_apoio ≥ 0,8·m·g`) | 0,5 s |

O fecho de um elo com elo seguinte → `_avanca_elo`. O fecho do ÚLTIMO →
`episode_success = 1` travado, e o episódio CONTINUA (§7.5). Sem reset entre elos.

Máscaras de recompensa (auditadas): `unload` só no `pegar`; `squeeze` fora do `botar`;
termo novo `load` (+2,0) só no `botar` — o espelho do `unload`, senão soltar custa e não
paga. `postura_ereta` não precisa de máscara (o gate de descarga já zera com a caixa
apoiada).

Observação nova: `face_normal_b` (3 canais) — a normal ATUAL da face alvo, no frame da
base. Sem ela o `reorientar` é cego (o ator vê o desejado `dir_alvo`, não o atual).
**Ator 112 → 115, crítico 125 → 128.** Entra por ÚLTIMO nos dois grupos (a cirurgia de
checkpoint vira um append de 3 colunas).

---

## Tarefa 1 — `g1_poc/knobs.py`

1. Em `Recompensa`, depois de `sustentacao`:

```python
    # §8.2.5 — `load`, o espelho do `unload`, SÓ no elo `botar` (20/08).
    # O fecho do `botar` exige F_apoio >= 0,8·m·g, e os termos de segurar apontam
    # todos contra soltar: medido, satisfazer a 3ª condição custava −3,0/s e pagava
    # ZERO — o `botar` fecharia por sorte. `load = clamp(F_apoio/m·g)` é a mesma
    # grandeza contínua do `unload`, invertida, gateada por "perto do alvo".
    load: float = 2.0
    # o gate de posição do `load`: 2× o raio de sucesso. Sem ele, LARGAR a caixa em
    # qualquer lugar do tampo pagaria o máximo.
    load_raio_mult: float = 2.0
    # σ variável do `precise_ori` (mesmo idioma do bringing/reaching): piso 0,40 rad,
    # teto = Δθ inicial do elo. Com σ fixo, 90° dá 2,0e-7 — o `reorientar` dos
    # níveis 4+ era sorte.
    precise_ori_std: float = 0.40   # (a linha já existe — vira o PISO; só o comentário muda)
```

⚠ `precise_ori_std` JÁ existe — não duplique o campo; troque só o comentário dele.

2. Em `Episodio`:

```python
    # fração dos envs de MANIPULAÇÃO com o twist LIBERADO (bloco "segure e ande").
    # Medido: 0,00% das transições têm twist ≠ 0 E caixa_valida = 1, por construção.
    # O `carregar` estreia fora da distribuição das duas metades da recompensa; um
    # bloco com isto em ~0,3 antes do nível 4 fecha o vão. Default 0 = desligado.
    frac_twist_livre_manipula: float = 0.0
```

---

## Tarefa 2 — `g1_poc/observacoes.py`

Função nova (siga o estilo do arquivo; `FACE` importável de `comando` — se criar ciclo de
import, receba `lo=3, hi=6` como a `fatia_comando` faz e leia a fatia):

```python
def face_normal_b(env, command_name: str, object_name: str) -> torch.Tensor:
    """[B,3] — a normal ATUAL da face alvo, no frame da BASE.

    O ator vê o DESEJADO (`dir_alvo`) e não via o ATUAL: a orientação da caixa só
    era recuperável pela diferença dos dois vetores palma→face, dominada pela
    distância. O `reorientar` fecha por `Δθ < 20°` e o `precise_ori` paga por Δθ —
    sem este canal a coordenada é invisível (auditoria T18, 20/08).

    No deploy a percepção JÁ entrega esta grandeza: é a mesma orientação medida que
    preenche `face_alvo`/`dir_alvo` (§21.2, "os medidos").

    Zera com `caixa_valida = 0`, como os outros canais de caixa.
    """
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[object_name]
    robot = env.scene["robot"]
    face_b = cmd.command[:, 3:6]                                   # face, frame da caixa
    normal_w = quat_apply(obj.data.root_link_quat_w, face_b)
    normal_base = quat_apply_inverse(robot.data.root_link_quat_w, normal_w)
    return normal_base * cmd.command[:, 9:10]
```

(Importe `quat_apply`/`quat_apply_inverse` se o arquivo ainda não os tiver.)

---

## Tarefa 3 — `g1_poc/comando.py` (a máquina de elo)

### 3.1 Constantes de módulo, depois de `N_LATERAIS`

```python
# --- os elos e as cadeias (§7) ---
PEGAR, REORIENTAR, CARREGAR, BOTAR = 0, 1, 2, 3
# elos de cada cadeia, com -1 de padding. A ordem das cadeias é a de
# `knobs.Celulas.cadeias`: (pegar, reorientar->pegar, pegar->carregar, pegar->botar).
ELOS_DA_CADEIA = ((PEGAR, -1), (REORIENTAR, PEGAR), (PEGAR, CARREGAR), (PEGAR, BOTAR))
N_ELOS = (1, 2, 2, 2)
```

### 3.2 `CaixaAlvoCommandCfg` — campos novos

```python
    # --- a máquina de elo (§7) ---
    cadeias: tuple = ((1.0, 0.0, 0.0, 0.0),) * 7   # frações por nível; o env_cfg passa a tabela
    ang_max_deg: tuple[float, ...] = (0.0,) * 7    # rotação do `reorientar`, por nível
    sustenta_outros_s: float = 0.5
    carregar_s: float = 6.0
    fracao_apoio_botar: float = 0.80
    peito_b: tuple[float, float, float] = (0.25, 0.0, 0.15)
    botar_x: tuple[float, float] = (0.30, 0.40)
    botar_y: tuple[float, float] = (-0.12, 0.12)
    botar_topo_piso: float = 0.30
    botar_topo_teto: float = 0.80
    botar_folga_laje: float = 0.05
    caixa_meia_z: float = 0.10
    prateleira_meia_z: float = 0.02
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    afasta_z: float = 5.0
    support_sensor: str = "apoio_caixa"
    precise_ori_std_piso: float = 0.40
    # atalhos de MEDIÇÃO (play/sonda/smoke): força a cadeia; None = sorteio por nível
    cadeia_forcada: int | None = None
    # fração dos envs de manipulação com o twist LIBERADO ("segure e ande")
    frac_twist_livre: float = 0.0
```

### 3.3 `__init__` — buffers novos (publicados IN-PLACE)

Depois dos buffers existentes:

```python
        # --- a máquina de elo (§7) ---
        self._cadeia = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_t = torch.zeros(self.num_envs, device=self.device)
        self.pegou = torch.zeros(self.num_envs, device=self.device)
        self._twist_livre = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # σ variável do `precise_ori`: Δθ inicial do elo, com piso
        self.ori_inicial = torch.full(
            (self.num_envs,), cfg.precise_ori_std_piso, device=self.device)
        self._elos_tab = torch.tensor(ELOS_DA_CADEIA, dtype=torch.long, device=self.device)
        self._n_elos = torch.tensor(N_ELOS, dtype=torch.long, device=self.device)
        self._cadeias_tab = torch.tensor(cfg.cadeias, device=self.device)      # [7,4]
        self._ang_max = torch.deg2rad(torch.tensor(cfg.ang_max_deg, device=self.device))
        # o alvo de sustentação POR ELO (1,0 s no pegar; 0,5 s nos demais). Nasce
        # com o valor do pegar — antes do 1º reset nenhum termo o lê, mas zero aqui
        # explodiria a divisão do `sustentacao`.
        self._sust_alvo = torch.full(
            (self.num_envs,), cfg.sustenta_pegar_s, device=self.device)

        # ⚠ publicados UMA vez e atualizados IN-PLACE — a lição do alias poc_success
        env.poc_elo = self._elo_id
        env.poc_pegou = self.pegou
        env.poc_ori_inicial = self.ori_inicial
```

E nas métricas: `self.metrics["cadeia"]`, `self.metrics["elo"]`, `self.metrics["pegou"]`
(zeros como as demais).

### 3.4 `_resample_command` — sorteia a cadeia e arma o primeiro elo

Depois do bloco da forma e ANTES do bloco do alvo, acrescente:

```python
        # --- a cadeia do episódio, pela célula do nível (§10.1) ---
        nivel = getattr(self._env, "poc_nivel", None)
        if nivel is None:
            self._cadeia[env_ids] = 0
        elif self.cfg.cadeia_forcada is not None:
            # atalho de MEDIÇÃO (play/sonda/smoke); no treino fica None
            self._cadeia[env_ids] = int(self.cfg.cadeia_forcada)
        else:
            probs = self._cadeias_tab[nivel[env_ids]]
            self._cadeia[env_ids] = torch.multinomial(probs, 1).squeeze(-1)
        self._elo_idx[env_ids] = 0
        self._elo_id[env_ids] = self._elos_tab[self._cadeia[env_ids], 0]
        self._elo_t[env_ids] = 0.0
        self.pegou[env_ids] = 0.0
        self._sust_alvo[env_ids] = torch.where(
            self._elo_id[env_ids] == PEGAR,
            torch.full_like(self._sust_alvo[env_ids], self.cfg.sustenta_pegar_s),
            torch.full_like(self._sust_alvo[env_ids], self.cfg.sustenta_outros_s))
        # "segure e ande": fração dos envs de manipulação com o twist liberado
        self._twist_livre[env_ids] = (
            torch.rand(n, device=self.device) < self.cfg.frac_twist_livre) & manipula
```

O bloco do alvo/face/giro que já existe passa a valer para o PRIMEIRO elo:

- se o primeiro elo é `PEGAR` (cadeias 0, 2, 3): exatamente o código atual, `_ang = 0`.
- se é `REORIENTAR` (cadeia 1): o alvo é a posição ATUAL da caixa (pose fresca — fica no
  `_resolver`, marcado por `_pendente`), e `_ang[env_ids] = (2·rand − 1) · ang_max[nivel]`.

Implemente com máscara sobre `env_ids` (o sorteio do alvo de mundo roda para todos e o
`_resolver` SOBRESCREVE o alvo dos `REORIENTAR` com a pose fresca da caixa).

### 3.5 `_resolver` — o alvo do `reorientar` e o `ori_inicial`

No corpo existente (que já resolve `_dir_w`, `dist_inicial`, `reach_inicial`), acrescente
ANTES do cálculo de `d`:

```python
        # o alvo do `reorientar` é a posição ATUAL da caixa (§7.1) — pose fresca
        reori = ids[self._elo_id[ids] == REORIENTAR]
        if len(reori) > 0:
            self._command[reori, ALVO] = self.caixa.data.root_link_pos_w[reori]
```

E DEPOIS do `reach_inicial`:

```python
        # σ do `precise_ori` = o Δθ a vencer no começo do elo, com piso (§8.2)
        self.ori_inicial[ids] = torch.clamp(
            self.erro_ang()[ids], min=self.cfg.precise_ori_std_piso)
```

### 3.6 `_update_command` — o laço da máquina

Substitua o bloco do fecho (do comentário `# --- o fecho do elo` até o
`episode_success.copy_(...)`) por:

```python
        # --- o alvo do `carregar` é do CORPO, recalculado a cada passo (§7.1) ---
        carrega = (self._elo_id == CARREGAR) & self.manipula
        if bool(carrega.any()):
            ids_c = carrega.nonzero().flatten()
            peito = torch.tensor(self.cfg.peito_b, device=self.device).expand(len(ids_c), 3)
            alvo_c = (self.robot.data.root_link_pos_w[ids_c]
                      + quat_apply(self.robot.data.root_link_quat_w[ids_c], peito))
            self._command[ids_c, ALVO] = alvo_c

        # o twist é zero nos elos de manipulação, EXCETO no `carregar` e nos envs
        # "segure e ande" (frac_twist_livre)
        self._env.poc_twist_zero.copy_(
            self.manipula & ~carrega & ~self._twist_livre)

        # --- o fecho, condição a condição, POR ELO (§7.2) ---
        perto = self.erro_pos() < self.cfg.raio_sucesso
        alinhado = self.erro_ang() < self.cfg.angulo_sucesso_rad
        de_pe = self.de_pe()
        f_apoio = self._env.scene[self.cfg.support_sensor].data.force
        apoio_z = f_apoio[..., 2].abs().sum(dim=-1)
        peso = (getattr(self._env, "poc_massa") * 9.81).clamp(min=1e-3)
        apoiada = apoio_z >= self.cfg.fracao_apoio_botar * peso

        fecha = torch.zeros_like(perto)
        e = self._elo_id
        fecha |= (e == PEGAR) & perto & alinhado & de_pe
        fecha |= (e == REORIENTAR) & perto & alinhado
        fecha |= (e == CARREGAR) & perto
        fecha |= (e == BOTAR) & perto & alinhado & apoiada
        fecha &= self.manipula

        dt = self._env.step_dt
        self._elo_t += dt
        # ⚠ copy_, nunca atribuição: env.poc_success/poc_pegou são aliases in-place
        self._sustenta.copy_(torch.where(fecha, self._sustenta + dt,
                                         torch.zeros_like(self._sustenta)))

        sustentado = self._sustenta >= self._sust_alvo
        # o `carregar` fecha por TEMPO, sustentado: elo_t >= 6 s E perto por 0,5 s.
        # Um fecho INSTANTÂNEO com no_alvo ~57% seria uma moeda, e o nível viraria
        # passeio sem deriva (auditoria T16).
        sustentado &= (e != CARREGAR) | (self._elo_t >= self.cfg.carregar_s)

        ultimo = self._elo_idx + 1 >= self._n_elos[self._cadeia]
        fecha_elo = sustentado & self.manipula

        # o elo que fechou era `pegar`? arma a `caixa_largada` e o §7.3
        fechou_pegar = fecha_elo & (e == PEGAR)
        self.pegou.copy_(torch.maximum(self.pegou, fechou_pegar.float()))

        avanca = fecha_elo & ~ultimo
        if bool(avanca.any()):
            self._avanca_elo(avanca.nonzero().flatten())

        # sucesso TRAVADO no fecho do ÚLTIMO elo. O episódio continua (§7.5).
        self.episode_success.copy_(torch.maximum(
            self.episode_success, (fecha_elo & ultimo).float()))
```

⚠ O `getattr(self._env, "poc_massa")` só existe depois do primeiro `carga_caixa`; no
primeiríssimo `_update_command` do processo ele pode faltar — use
`getattr(self._env, "poc_massa", None)` e, se `None`, `apoiada = torch.zeros_like(perto)`.

### 3.7 `_avanca_elo(ids)` — método novo

```python
    def _avanca_elo(self, ids: torch.Tensor) -> None:
        """Escreve o elo seguinte no comando, SEM reset (§7.5) e SEM resample.

        ⚠ Não usa `_resample_command`: aquele zera `episode_success` e sorteia
        cadeia nova. E não usa `_pendente`: aqui as poses JÁ estão frescas — o
        `_update_command` roda depois do `sim.forward()` do step.

        §7.3 — a prateleira se move quando o `pegar` fecha: +5 m na cadeia
        `carregar` (o chão fica livre para andar); topo NOVO na cadeia `botar`
        (a faixa é a da COLOCAÇÃO, 0,30-0,80, com teto efetivo no fundo da caixa
        menos a folga — a §7.3 prometia 0,55 e a §10.1 manda 0,80; sem o teto
        efetivo a laje nasceria DENTRO da caixa). Só a MESA se move: a caixa está
        nas mãos. A escrita de mocap vale a partir do passo seguinte.
        """
        n = len(ids)
        dev = self.device
        self._elo_idx[ids] += 1
        novo = self._elos_tab[self._cadeia[ids], self._elo_idx[ids]]
        self._elo_id[ids] = novo
        self._elo_t[ids] = 0.0
        self._sustenta[ids] = 0.0
        self._sust_alvo[ids] = torch.where(
            novo == PEGAR,
            torch.full((n,), self.cfg.sustenta_pegar_s, device=dev),
            torch.full((n,), self.cfg.sustenta_outros_s, device=dev))
        origem = self._env.scene.env_origins[ids]

        # --- PEGAR (2º elo da cadeia `reorientar`): alvo de mundo, "erga sem torcer"
        m = (novo == PEGAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            r = self.cfg.pegar_range
            lo = torch.tensor([r[0][0], r[1][0], r[2][0]], device=dev)
            hi = torch.tensor([r[0][1], r[1][1], r[2][1]], device=dev)
            alvo = lo + (hi - lo) * torch.rand(len(i), 3, device=dev)
            alvo[:, 0] += origem[m][:, 0]
            alvo[:, 1] += origem[m][:, 1]
            self._command[i, ALVO] = alvo
            self._ang[i] = 0.0

        # --- CARREGAR: mesa +5 m; o alvo do corpo é escrito a cada passo
        m = (novo == CARREGAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            pose = torch.zeros(len(i), 7, device=dev)
            pose[:, 0] = origem[m][:, 0] + self.cfg.prateleira_xy[0]
            pose[:, 1] = origem[m][:, 1] + self.cfg.prateleira_xy[1]
            pose[:, 2] = self.cfg.afasta_z - self.cfg.prateleira_meia_z
            pose[:, 3] = 1.0
            self.prateleira.write_mocap_pose_to_sim(pose, env_ids=i)
            if hasattr(self._env, "poc_topo"):
                self._env.poc_topo[i] = self.cfg.afasta_z
            self._ang[i] = 0.0

        # --- BOTAR: topo novo + alvo lateral em cima dele
        m = (novo == BOTAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            fundo = self.caixa.data.root_link_pos_w[i, 2] - self.cfg.caixa_meia_z
            teto = torch.clamp(fundo - self.cfg.botar_folga_laje,
                               max=self.cfg.botar_topo_teto)
            piso = torch.full_like(teto, self.cfg.botar_topo_piso)
            teto = torch.maximum(teto, piso)   # nunca inverte a faixa
            topo = piso + (teto - piso) * torch.rand(len(i), device=dev)
            pose = torch.zeros(len(i), 7, device=dev)
            pose[:, 0] = origem[m][:, 0] + self.cfg.prateleira_xy[0]
            pose[:, 1] = origem[m][:, 1] + self.cfg.prateleira_xy[1]
            pose[:, 2] = topo - self.cfg.prateleira_meia_z
            pose[:, 3] = 1.0
            self.prateleira.write_mocap_pose_to_sim(pose, env_ids=i)
            if hasattr(self._env, "poc_topo"):
                self._env.poc_topo[i] = topo
            bx = self.cfg.botar_x
            by = self.cfg.botar_y
            alvo = torch.zeros(len(i), 3, device=dev)
            alvo[:, 0] = origem[m][:, 0] + bx[0] + (bx[1] - bx[0]) * torch.rand(len(i), device=dev)
            alvo[:, 1] = origem[m][:, 1] + by[0] + (by[1] - by[0]) * torch.rand(len(i), device=dev)
            alvo[:, 2] = topo + self.cfg.caixa_meia_z
            self._command[i, ALVO] = alvo
            self._ang[i] = 0.0

        # dir_alvo, σ do bringing, do reaching e do ori — contra a pose FRESCA
        face_b = torch.tensor(FACE_AXES, device=dev)[self._face_idx[ids]]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], face_b)
        dir_w = _rot_z(normal_w, self._ang[ids])
        self._dir_w[ids] = dir_w / dir_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._command[ids, FACE] = face_b
        d = torch.norm(self._command[ids, ALVO]
                       - self.caixa.data.root_link_pos_w[ids], dim=-1)
        self.dist_inicial[ids] = torch.clamp(d, min=self.cfg.bringing_std_piso)
        palmas = self.robot.data.site_pos_w[ids][:, self._palm_ids]
        from g1_poc.observacoes import alvos_das_palmas
        alvos_p = alvos_das_palmas(self._env, "box", self.cfg.lateral_offset)[ids]
        self.reach_inicial[ids] = torch.clamp(
            torch.norm(palmas - alvos_p, dim=-1).mean(dim=-1),
            min=self.cfg.reaching_std_piso)
        self.ori_inicial[ids] = torch.clamp(
            self.erro_ang()[ids], min=self.cfg.precise_ori_std_piso)
```

### 3.8 `_update_metrics`

Acrescente: `self.metrics["cadeia"] = self._cadeia.float() * v`,
`self.metrics["elo"] = self._elo_id.float() * v`, `self.metrics["pegou"] = self.pegou * v`.

### 3.9 Docstring do módulo

O bloco "ESTADO DESTE ARQUIVO — ESQUELETO" morre: a máquina de elo está aqui; descreva-a
em 5 linhas (cadeia sorteada pela célula, elo avança sem reset, prateleira se move no
§7.3, sucesso trava no último elo).

---

## Tarefa 4 — `g1_poc/recompensas.py`

### 4.1 Máscaras por elo em `unload` e `squeeze`

No fim do `unload`, antes do `return`:

```python
    # ⚠ SÓ no elo `pegar` (20/08). Nos outros a caixa já saiu da prateleira, e no
    # `botar` este termo é o OPOSTO do fecho (F_apoio >= 0,8·m·g): ligado lá, pagaria
    # 2,0/s para NÃO botar. A máscara vem antes de qualquer mexida em `poc_topo`.
    elo = getattr(env, "poc_elo", None)
    no_pegar = torch.ones_like(fracao) if elo is None else (elo == 0).float()
```

e multiplique `no_pegar` no retorno. No `squeeze`, idem com `fora_botar = (elo != 3)`:
apertar durante o `botar` paga contra soltar (−1,0/s medido).

### 4.2 `precise_ori` com σ por elo

No corpo, troque o uso de `std` por:

```python
    sigma = getattr(env, "poc_ori_inicial", None)
    if sigma is None:
        sigma = torch.full_like(theta, std)
    return reaching * torch.exp(-torch.square(theta) / sigma.clamp(min=std) ** 2) * _valida(...)
```

com um `⚠` curto: com σ fixo de 0,40 rad, 90° dá 2,0e-7 — o `reorientar` dos níveis 4+
era sorte; mesmo idioma do `bringing`/`reaching`.

### 4.3 Termo novo `load`

```python
def load(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    support_sensor: str,
    massa_attr: str,
    raio_sucesso: float,
    raio_mult: float,
) -> torch.Tensor:
    """`clamp(F_apoio/m·g)` — o espelho do `unload`, SÓ no elo `botar` (§8.2.5).

    Sem ele o `botar` não tem quem pague por soltar: `squeeze` e `unload` apontam
    contra, e com as máscaras deles o saldo vira exatamente ZERO — o fecho
    (`F_apoio >= 0,8·m·g`) seria descoberto por sorte. Medido: satisfazer a 3ª
    condição custava −3,0/s antes das máscaras.

    O gate de posição (`erro < 2·raio`) fecha o hack de largar a caixa em qualquer
    lugar do tampo. Sem gate de preensão, de propósito: soltar É o objetivo, e o
    termo continua pagando depois do fecho — é o estado colocado que mais paga.
    """
    elo = getattr(env, "poc_elo", None)
    if elo is None:
        return torch.zeros(env.num_envs, device=env.device)
    f = env.scene[support_sensor].data.force
    assert f is not None
    apoio_z = f[..., 2].abs().sum(dim=-1)
    peso = (getattr(env, massa_attr) * 9.81).clamp(min=1e-3)
    fracao = (apoio_z / peso).clamp(0.0, 1.0)
    err = torch.sqrt(_erro_pos_sq(env, command_name, object_name))
    perto = (err < raio_mult * raio_sucesso).float()
    return fracao * perto * (elo == 3).float() * _valida(env, command_name)
```

### 4.4 `sustentacao` acompanha o alvo POR ELO

O termo divide por `cmd.cfg.sustenta_pegar_s` fixo; com a máquina de elo o alvo é por env
(1,0 s no `pegar`, 0,5 s nos demais). Troque a linha por:

```python
    fracao = (cmd._sustenta / cmd._sust_alvo.clamp(min=1e-6)).clamp(0.0, 1.0)
```

e acrescente uma linha à docstring: o denominador é o alvo do ELO corrente, não uma
constante — senão os elos de 0,5 s pagariam só metade da rampa.

Docstring do módulo: 8 → **9** termos de tarefa; `load` na lista, com o gate por
`caixa_valida` como os outros.

---

## Tarefa 5 — `g1_poc/terminacoes.py`

`caixa_largada` ganha os gates da cadeia (auditoria T12/T14):

```python
    # armas por ramo (20/08):
    #   `caiu`    — desde a PREENSÃO (poc_pegou), sempre: a caixa no chão é falha
    #               em qualquer elo, e depois do sucesso também (largar o que se
    #               ergueu desfaz a tarefa e o episódio acaba SEM bootstrap).
    #   `escapou` — só nos elos de SEGURAR (pegar/carregar) e só até a cadeia
    #               fechar: no `botar` afastar as mãos é o objetivo, e depois do
    #               sucesso a caixa fica na prateleira longe das palmas por
    #               construção. ⚠ O gate antigo (`poc_success`) armava tudo de uma
    #               vez; o proposto no plano (`pegou & ~sucesso`) era identicamente
    #               FALSO na cadeia de um elo. Este é o terceiro desenho, e o smoke
    #               o exercita ramo a ramo.
    pegou = getattr(env, "poc_pegou", None)
    if pegou is None:
        return torch.zeros_like(caiu)
    sucesso = getattr(env, "poc_success", torch.zeros_like(caiu, dtype=torch.float))
    elo = getattr(env, "poc_elo", torch.zeros_like(caiu, dtype=torch.long))
    armada_caiu = pegou > 0.5
    armada_escapou = (pegou > 0.5) & (elo != 3) & (sucesso < 0.5)
    return (caiu & armada_caiu) | (escapou & armada_escapou)
```

(Adapte os nomes ao corpo atual da função; `caiu`/`escapou` já existem.)

---

## Tarefa 6 — `g1_poc/env_cfg.py`

1. `OBS_ATOR = 115`, `OBS_CRITICO = 128`, com o comentário do porquê (canal
   `face_normal_b`; a cirurgia de checkpoint é um append de 3 colunas).
2. No dict de obs do ator, **POR ÚLTIMO** (depois de `caixa_valida`):

```python
        "face_normal_b": ObservationTermCfg(
            func=OBS.face_normal_b,
            params={"command_name": CMD_CAIXA, "object_name": "box"},
        ),
```

   e o mesmo no crítico (o dict `caixa_ator` já é copiado — garanta que a ordem final
   nos DOIS grupos deixa `face_normal_b` como os 3 ÚLTIMOS canais; no crítico ele deve
   vir depois de `topo_prateleira`, então registre-o à parte, após os privilegiados).
3. `CaixaAlvoCommandCfg`: passe os campos novos —
   `cadeias=k.celulas.cadeias, ang_max_deg=k.celulas.ang_max_deg,
   sustenta_outros_s=kt.sustenta_outros_s, carregar_s=kt.carregar_s,
   fracao_apoio_botar=kt.fracao_apoio_botar, peito_b=ka.peito_b, botar_x=ka.botar_x,
   botar_y=ka.botar_y, botar_topo_piso=ka.botar_topo_piso,
   botar_topo_teto=ka.botar_topo_teto, botar_folga_laje=ka.botar_folga_laje,
   caixa_meia_z=kc.caixa_meia_aresta[2], prateleira_meia_z=kc.prateleira_meia_z,
   prateleira_xy=kc.prateleira_xy, afasta_z=kc.afasta_z,
   support_sensor=C.SENSOR_APOIO, precise_ori_std_piso=kr.precise_ori_std,
   frac_twist_livre=ke.frac_twist_livre_manipula`.
4. Termo novo, depois do `sustentacao`:

```python
    # o ESPELHO do unload, só no elo `botar` (§8.2.5)
    cfg.rewards["load"] = RewardTermCfg(
        func=R.load, weight=kr.load,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "support_sensor": C.SENSOR_APOIO, "massa_attr": "poc_massa",
                "raio_sucesso": kt.raio_sucesso, "raio_mult": kr.load_raio_mult},
    )
```

5. Docstring do topo: 8 → 9 termos de tarefa; contrato 115/128; item novo "11. a máquina
   de elo (§7)".

---

## Tarefa 7 — `g1_poc/expande_checkpoint.py` (novo)

CLI que expande um checkpoint treinado com ator 112 / crítico 125 para 115 / 128:

```
python -m g1_poc.expande_checkpoint --entrada model_5100.pt --saida model_5100_115.pt
```

Regras:
- Os 3 canais novos são os ÚLTIMOS de cada grupo → a cirurgia é APPEND de colunas.
- Camada de entrada do ator e do crítico: `weight` [h, 112] → [h, 115] com as 3 colunas
  novas em ZERO (a política começa ignorando o canal e aprende a usá-lo); `bias` intacto.
- Normalizador empírico dos DOIS grupos: `mean` ganha 3 zeros, a variância ganha 3 uns,
  `count` intacto. Inspecione as chaves reais do state_dict antes de assumir nomes.
- Qualquer outra chave passa intacta. Imprima um resumo: chaves mudadas, shapes
  antes/depois.
- ⚠ Descubra a estrutura REAL: monte um runner do g1_poc na CPU (como a `sonda.py` faz),
  chame `runner.save(...)` num arquivo do scratchpad, e liste as chaves/shapes. NÃO
  chute nomes de chave.
- Valide no fim do próprio script: recarregue a saída num runner novo (contrato 115) com
  `runner.load(..., strict=True)` — tem de carregar sem erro.

---

## Tarefa 8 — `g1_poc/smoke.py`

1. Contagens: 22 → **23** recompensas; `load` na tupla `tarefa`; ator 115 / crítico 128
   (as constantes já vêm do env_cfg — o teste existente pega sozinho).
2. Bit=0: acrescente `load` ao laço, e um checa de que `face_normal_b` (a FUNÇÃO, com os
   params do manager) é zero com bit 0.
3. Seção nova `17. a máquina de elo (§7)` — tudo por manipulação direta dos buffers do
   comando + `_update_command()`, como as seções existentes. Roda ANTES da seção do gate
   do twist (que mexe em `common_step_counter`):

```python
    print("== 17. a máquina de elo (§7) ==")
    from g1_poc.comando import PEGAR, REORIENTAR, CARREGAR, BOTAR
    cmd._command[:, 9] = 1.0
    cmd.manipula[:] = True
    env.poc_manipula[:] = True

    # 17a. cadeia `pegar`->`botar`: fecho do pegar move a prateleira e escreve o alvo novo
    cmd._cadeia[:] = 3
    cmd._elo_idx[:] = 0
    cmd._elo_id[:] = PEGAR
    cmd.pegou[:] = 0.0
    cmd.episode_success.copy_(torch.zeros(N_ENVS, device=env.device))
    cmd._sustenta[:] = cmd.cfg.sustenta_pegar_s + 1.0   # sustentado
    cmd._sust_alvo[:] = cmd.cfg.sustenta_pegar_s
    # força as 4 condições: caixa NO alvo, robô de pé
    caixa = env.scene["box"]
    pose_c = caixa.data.root_link_pose_w.clone()
    pose_c[:, 0:3] = cmd._command[:, 0:3]
    caixa.write_root_link_pose_to_sim(pose_c)
    env.sim.forward()
    topo_antes = env.poc_topo.clone()
    cmd._update_command()
    checa(bool((cmd._elo_id == BOTAR).all()),
          f"o fecho do `pegar` AVANÇA para o `botar` (elo medido {cmd._elo_id.tolist()})")
    checa(bool((cmd.pegou > 0.5).all()), "e arma `poc_pegou`")
    checa(bool((cmd.episode_success < 0.5).all()),
          "o sucesso NÃO trava no elo intermediário")
    checa(bool((cmd._sustenta.abs() < 1e-6).all()), "o cronômetro zera na troca")
    checa(bool(((env.poc_topo - topo_antes).abs() > 1e-6).all()),
          "a prateleira RECEBE topo novo (§7.3)")
    checa(bool((env.poc_topo >= cmd.cfg.botar_topo_piso - 1e-6).all()),
          f"o topo novo respeita o piso da colocação ({cmd.cfg.botar_topo_piso})")
    alvo_z = cmd._command[:, 2]
    checa(bool(((alvo_z - (env.poc_topo + cmd.cfg.caixa_meia_z)).abs() < 1e-5).all()),
          "o alvo do `botar` assenta a caixa no topo novo")

    # 17b. o `unload` é ZERO no elo `botar`, e o `load` só paga nele
    tc_unl = env.reward_manager.get_term_cfg("unload")
    v = tc_unl.func(env, **tc_unl.params)
    checa(bool((v.abs() < 1e-6).all()),
          f"`unload` mascarado fora do `pegar` (medido max {float(v.abs().max()):.2e})")
    tc_load = env.reward_manager.get_term_cfg("load")
    cmd._elo_id[:] = PEGAR
    v = tc_load.func(env, **tc_load.params)
    checa(bool((v.abs() < 1e-6).all()), "`load` é zero fora do `botar`")
    cmd._elo_id[:] = BOTAR

    # 17c. o fecho do ÚLTIMO elo trava o sucesso e o episódio continua
    cmd._sustenta[:] = cmd.cfg.sustenta_outros_s + 1.0
    # (as condições do botar exigem apoio >= 0,8·m·g — força pelo caminho do teste:
    #  o smoke não simula contato; valide a TRAVA pelo caminho de `fecha_elo`)
    ...   # ver 17d
    # 17d. `caixa_largada`: ramo a ramo
    tc_cl = env.termination_manager.get_term_cfg("caixa_largada")
    env.poc_pegou[:] = 0.0
    checa(bool(~tc_cl.func(env, **tc_cl.params).any()),
          "sem preensão, `caixa_largada` nunca dispara")
    env.poc_pegou[:] = 1.0
    cmd._elo_id[:] = CARREGAR
    pose_c = caixa.data.root_link_pose_w.clone()
    pose_c[:, 0] += 1.0            # a caixa escapa das duas palmas
    caixa.write_root_link_pose_to_sim(pose_c)
    env.sim.forward()
    checa(bool(tc_cl.func(env, **tc_cl.params).all()),
          "no `carregar`, escapar das palmas TERMINA")
    cmd._elo_id[:] = BOTAR
    checa(bool(~tc_cl.func(env, **tc_cl.params).any()),
          "no `botar`, afastar as mãos NÃO termina (soltar é o objetivo)")
```

   Complete o 17c como conseguir SEM contato real (ex.: valide via `_elos_tab`/`ultimo`
   com a cadeia 0: elo único fechado → `episode_success` vai a 1 e `_elo_id` não muda).
   Depois da seção, restaure a cena com `env._reset_idx(todos)` antes da seção do gate.
4. Rodapé: remova as linhas das cadeias/prateleira do "NÃO coberto"; acrescente
   "a física do `reorientar` (empurrar a caixa apoiada) — só a sonda/play medem".

---

## Tarefa 9 — `g1_poc/play.py` e `g1_poc/sonda.py`

Flag `--cadeia {0,1,2,3}` nos dois (mesmo molde do `--nivel`): muta
`env_cfg.commands["caixa_alvo"].cadeia_forcada` em `_registra` (o `CommandManager` NÃO
deepcopia — mutar o cfg antes do registro funciona; mas mute ANTES do
`register_mjlab_task`, no mesmo ponto do `nivel`). Sufixo `-C{n}` no task id. Valide
0..3. `--cadeia` exige `--pegar` (play) e vale sempre na sonda. No help, os nomes:
0 pegar · 1 reorientar→pegar · 2 pegar→carregar · 3 pegar→botar.

---

## Critério de pronto

```
.venv/bin/python -m g1_poc.smoke        # 0 falhas; contagem sobe de 102 para >= 120
.venv/bin/python -m g1_poc.cena         # compila
.venv/bin/python -m g1_poc.expande_checkpoint --auto-teste   # o próprio script valida
```

Relate: contagem do smoke, arquivos tocados, as chaves REAIS que a cirurgia mudou, e
qualquer divergência entre este spec e os arquivos — divergiu, PARE e relate.
