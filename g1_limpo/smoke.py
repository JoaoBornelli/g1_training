"""Os checks do g1_limpo.

    python -m g1_limpo.smoke

Imprime `N ok / M falhas` e sai com código 1 se houver falha. É o portão de cada
fase do plano: nenhuma fase começa com o smoke vermelho.

⚠ Nenhum check aqui importa `g1_training`, `g1_poc` ou `g1_multitask`. Quem compara
contra as referências é o `paridade.py`, que é descartável.

FASES COBERTAS: F0 (esqueleto, cena, física, remoções, contrato de não-import).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)

from g1_limpo import cena as C
from g1_limpo.env_cfg import colhe_sigmas_de_postura, make_env_cfg
from g1_limpo.knobs import Knobs

_ok = 0
_falhas: list[str] = []


def check(nome: str, cond: bool, detalhe: str = "") -> None:
    global _ok
    if cond:
        _ok += 1
    else:
        _falhas.append(f"{nome}{f'  ({detalhe})' if detalhe else ''}")


def secao(t: str) -> None:
    print(f"\n--- {t}")


# =============================================================================
k = Knobs()
c = k.cena
cfg = make_env_cfg(k, play=False)
play = make_env_cfg(k, play=True)
fab = unitree_g1_flat_env_cfg(play=False)

# ------------------------------------------------------------------ 1. entidades
secao("1. entidades")
check("as três entidades existem",
      sorted(cfg.scene.entities) == ["box", "robot", "table"],
      str(sorted(cfg.scene.entities)))

m_caixa = C.regroup(C.spec_caixa(c), c.grupo_mobilia).compile()
m_prat = C.regroup(C.spec_prateleira(c), c.grupo_mobilia).compile()

check("a caixa é corpo LIVRE (7 DoF)", m_caixa.nq == 7, f"nq={m_caixa.nq}")
check("a prateleira é FIXA -> mocap (0 DoF)", m_prat.nq == 0, f"nq={m_prat.nq}")
check("a mobília está fora do grupo 0",
      all(int(m.geom_group[i]) == c.grupo_mobilia
          for m in (m_caixa, m_prat) for i in range(m.ngeom)),
      "senão o foot_height_scan lê a prateleira COMO CHÃO")
# ⚠ NÃO checar "a prateleira não tem massa". O MuJoCo DERIVA massa da densidade
# quando o geom não passa `mass=`: 0,60 × 0,60 × 0,04 m³ × 1000 kg/m³ = 14,4 kg. A
# referência também não passa massa, portanto este valor BATE com ela — e ele é
# irrelevante, porque o corpo é FIXO (nq = 0) e a massa nunca entra na dinâmica.
# Quem confere que ela bate com a referência é o `paridade.py`.
check("a massa da prateleira é irrelevante porque ela não tem DoF",
      m_prat.nq == 0 and float(m_prat.body_mass.sum()) > 0.0,
      f"massa derivada = {float(m_prat.body_mass.sum()):.1f} kg, nq = {m_prat.nq}")
check("a caixa pesa o que o knob diz",
      abs(float(m_caixa.body_mass.sum()) - c.caixa_massa) < 1e-9)

# ------------------------------------------------------------------ 2. geometria
secao("2. geometria de repouso")
g = C.geometria_de_repouso(c)
check("o centro do mocap fica meia_z abaixo do topo",
      abs(g["centro_prateleira_z"] - (g["topo"] - c.prateleira_meia_z)) < 1e-12)
check("a caixa repousa EM CIMA da laje, sem afundar",
      abs(g["caixa_z"] - (g["topo"] + c.caixa_meia_aresta[2])) < 1e-12)
fundo_no_piso = c.prateleira_topo_piso - 2.0 * c.prateleira_meia_z
check("no PISO do currículo a laje APOIA no chão, sem atravessar",
      fundo_no_piso >= -1e-12, f"fundo = {fundo_no_piso:+.4f} m")
check("a caixa nasce DENTRO da pegada da prateleira",
      abs(c.caixa_xy[0] - c.prateleira_xy[0]) <= c.prateleira_meia_xy,
      f"caixa x={c.caixa_xy[0]} vs prateleira x={c.prateleira_xy[0]}±{c.prateleira_meia_xy}")
check("o jitter em y não joga a caixa fora da prateleira",
      max(abs(c.caixa_jitter_y[0]), abs(c.caixa_jitter_y[1])) <= c.prateleira_meia_xy,
      f"jitter={c.caixa_jitter_y}")
check("a laje é FINA, não paredão",
      2.0 * c.prateleira_meia_z <= 0.10,
      f"espessura = {2*c.prateleira_meia_z:.3f} m")

# --------------------------------------------------------------- 3. nível/células
secao("3. tabela de níveis")
n = k.nivel
check("as três colunas têm o mesmo comprimento",
      len(n.topo_min) == len(n.carga_max) == len(n.ang_max_deg) == n.n_niveis)
check("o piso do topo só DESCE (cada nível contém o anterior)",
      all(n.topo_min[i + 1] <= n.topo_min[i] for i in range(n.n_niveis - 1)),
      str(n.topo_min))
check("o teto da carga só SOBE",
      all(n.carga_max[i + 1] >= n.carga_max[i] for i in range(n.n_niveis - 1)))
check("o teto do ângulo só SOBE",
      all(n.ang_max_deg[i + 1] >= n.ang_max_deg[i] for i in range(n.n_niveis - 1)))
check("o topo mais baixo do currículo nunca ENTERRA a laje",
      min(n.topo_min) - 2.0 * c.prateleira_meia_z >= -1e-12,
      f"o nível 6 do g1_multitask enterrava a laje em −0,02 m; aqui o fundo "
      f"fica em {min(n.topo_min) - 2*c.prateleira_meia_z:+.3f} m")
check("o topo mais alto do currículo é o teto de repouso",
      abs(max(n.topo_min) - c.prateleira_topo_teto) < 1e-12)

# ------------------------------------------------------------------ 4. sensores
secao("4. sensores")
por_nome = {s.name: s for s in cfg.scene.sensors}
esperados = (*C.SENSOR_PALMA, *C.SENSOR_DORSO, C.SENSOR_APOIO,
             C.SENSOR_CORPO_PRATELEIRA, C.SENSOR_AUTO_COLISAO, C.SENSOR_PES)
check("os 6 sensores do pacote existem",
      all(nome in por_nome for nome in esperados),
      str([n for n in esperados if n not in por_nome]))
check("as PALMAS pedem `force` — sem isso o `squeeze` é impossível",
      all("force" in por_nome[n].fields for n in C.SENSOR_PALMA))
check("o APOIO pede `force` — é a ponte do `unload`",
      "force" in por_nome[C.SENSOR_APOIO].fields)
check("os DORSOS são booleanos (magnitude não importa)",
      all("force" not in por_nome[n].fields for n in C.SENSOR_DORSO))
check("os PÉS rastreiam tempo no ar",
      por_nome[C.SENSOR_PES].track_air_time is True)
check("os PÉS aceitam QUALQUER contato como chão",
      por_nome[C.SENSOR_PES].secondary is None,
      "senão pisar na prateleira fica invisível e o slip do pé cega")
check("o `foot_height_scan` do fabricante FICA",
      any(isinstance(s, RayCastSensorCfg) and s.name == "foot_height_scan"
          for s in cfg.scene.sensors))
# ⚠ Este check foi INVERTIDO de propósito em 25/08. Eu tentei remover o
# `feet_ground_contact` e o `self_collision` do fabricante, substituindo-os pelos
# nossos. Isso EXPLODE: o termo de obs `foot_air_time` dele referencia o primeiro POR
# NOME, e o reward `self_collisions` referencia o segundo. Os nossos são ADIÇÃO.
check("os sensores do fabricante FICAM (os nossos são adição, não substituição)",
      all(nome in por_nome
          for nome in ("feet_ground_contact", "self_collision")),
      "removê-los quebra a obs e a recompensa DELE, e o objetivo da F1 é reproduzir "
      "a locomoção do fabricante")

# ------------------------------------------------------------------- 5. física
secao("5. física de manipulação")
check("njmax", cfg.sim.njmax == c.njmax == 800)
check("nconmax", cfg.sim.nconmax == c.nconmax == 300)
check("impratio", cfg.sim.mujoco.impratio == 1.0)
check("cone é pyramidal",
      cfg.sim.mujoco.cone == "pyramidal",
      "elliptic com impratio=10 divergiu para NaN no reset parcial (15/07)")

# --------------------------------------------------------------------- 6. ação
secao("6. ação")
acao = cfg.actions["joint_pos"]
check("`G1_ACTION_SCALE` tem 16 PADRÕES REGEX, não 29 nomes de junta",
      len(G1_ACTION_SCALE) == 16, f"{len(G1_ACTION_SCALE)}")
check("todos os padrões sobrevivem à multiplicação",
      set(acao.scale) == set(G1_ACTION_SCALE))
check("a escala é o fabricante × escala_acao_mult",
      all(abs(acao.scale[p] - v * c.escala_acao_mult) < 1e-12
          for p, v in G1_ACTION_SCALE.items()))

# ------------------------------------------------------------------ 7. eventos
secao("7. eventos e remoções")
check("`base_com` SAIU", "base_com" not in cfg.events,
      "ele corrompe a heap em CPU e em GPU, e derruba a task do próprio fabricante")
# ⚠ UM evento escreve a POSE da mobília, e ele faz as duas entidades. Dois eventos na
# mesma entidade não se somam: o segundo apaga o primeiro, sem erro e sem log.
check("UM evento só escreve a pose da mobília",
      "posiciona_cena" in cfg.events
      and not any(e in cfg.events for e in ("reset_caixa", "reset_prateleira",
                                            "reset_box", "reset_table")),
      str([e for e in cfg.events if "reset" in e]))
check("o `push_robot` FICA no treino — resistir a empurrão é requisito",
      "push_robot" in cfg.events)
check("o reset da base tem o range do knob",
      cfg.events["reset_base"].params["pose_range"] == dict(c.reset_base_manipula))

secao("8. ramo de play")
check("`randomize_terrain` fora do play",
      "randomize_terrain" not in play.events,
      "roda depois dos eventos de cena e mexe na origem do env")
check("o currículo de comando fora do play",
      not any("command" in nome for nome in play.curriculum),
      "ele muta o cfg COMPARTILHADO e apaga velocidade pinada à mão")
check("o `push_robot` fora do play", "push_robot" not in play.events)

# ----------------------------------------------------------- 9. σ da postura
secao("9. os σ da postura foram COLHIDOS, não digitados")
s = colhe_sigmas_de_postura(cfg)
r = unitree_g1_rough_env_cfg(play=False).rewards["pose"].params
check("os três dicts batem com o cfg do fabricante",
      all(s[key] == r[key] for key in ("std_standing", "std_walking", "std_running")))
check("`std_standing` é o do fabricante", s["std_standing"] == {".*": 0.05})

# A prova de que não foram redigitados: a palavra `knee` não aparece em NENHUM fonte
# deste pacote. As tabelas de σ são o único lugar onde ela apareceria.
_raiz = pathlib.Path(__file__).parent
# ⚠ `smoke.py` e `paridade.py` ficam FORA do scan: os dois falam SOBRE os σ e sobre
# os imports proibidos, e se auto-acusariam.
_fontes = [p for p in _raiz.glob("*.py") if p.name not in ("paridade.py", "smoke.py")]
check("nenhum fonte do pacote contém `knee` (prova do colhimento)",
      not any("knee" in p.read_text(encoding="utf-8") for p in _fontes),
      str([p.name for p in _fontes if "knee" in p.read_text(encoding='utf-8')]))

# ------------------------------------------------- 10. contrato de NÃO-IMPORT
secao("10. contrato de não-import")
_proibidos = ("g1_training", "g1_poc", "g1_multitask")
_viola = []
for p in _fontes:
    for linha in p.read_text(encoding="utf-8").splitlines():
        nu = linha.strip()
        if nu.startswith(("import ", "from ")) and any(x in nu for x in _proibidos):
            _viola.append(f"{p.name}: {nu}")
check("nenhum import de código do projeto (fora de paridade.py)",
      not _viola, "; ".join(_viola))

# ------------------------------------------------------- 11. recompensa da F0
secao("11. recompensa (na F0 é a do fabricante, sem mudança)")
check("a tabela é a mesma do molde",
      set(cfg.rewards) == set(fab.rewards),
      str(set(cfg.rewards) ^ set(fab.rewards)))
check("`air_time` está em 0,0 — os DOIS módulos de referência o tinham desligado",
      cfg.rewards["air_time"].weight == 0.0)
check("`dof_pos_limits` é −1,0, o valor do fabricante",
      cfg.rewards["dof_pos_limits"].weight == -1.0)

# ================================================ 12. currículo e comando
secao("12. currículo, eventos e comando")
from g1_limpo import comando as CMD          # noqa: E402
from g1_limpo import curriculo as CU_        # noqa: E402

check("o layout do comando é por NOME, sem índice solto",
      (CMD.ALVO, CMD.FACE, CMD.ANG, CMD.VALIDA, CMD.ELO, CMD.DIM)
      == (slice(0, 3), slice(3, 6), 6, 7, 8, 9))
check("os 5 elos existem, e a numeração é a dos slots do one-hot",
      (CMD.ANDAR, CMD.REORIENTAR, CMD.PEGAR, CMD.CARREGAR, CMD.BOTAR)
      == (0, 1, 2, 3, 4) and len(CMD.ELOS) == 5)
check("`elo_por_nome` resolve os cinco nomes",
      [CMD.elo_por_nome(x) for x in CMD.ELOS] == [0, 1, 2, 3, 4])
check("o elo default do treino é o PEGAR",
      cfg.commands["alvo_caixa"].elo_forcado == CMD.PEGAR)
check("o raio de alcance de referência foi DERIVADO do envelope da Lift",
      abs(CMD.ALCANCE_R - 0.85) < 1e-9,
      "0,50 estava errado: era o box_xy do 19%, não um raio de alcance")
check("só as 4 faces LATERAIS servem para pegar", CMD.N_LATERAIS == 4,
      "ninguém agarra uma caixa pelo topo nem pelo fundo")
check("há 6 faces declaradas", len(CMD.FACE_AXES) == 6)
check("as 4 primeiras faces são as laterais (z = 0)",
      all(ax[2] == 0.0 for ax in CMD.FACE_AXES[:CMD.N_LATERAIS]))
check("o `pegar` e o `carregar` pedem EXATAMENTE o mesmo alvo",
      cfg.commands["alvo_caixa"].peito_b == k.alvo.peito_b
      and cfg.commands["alvo_caixa"].altura_carregar == k.alvo.altura_carregar,
      "o que difere os dois elos é o TWIST, e não a forma do alvo")
check("o alvo NÃO tem jitter (ele fica exatamente à frente do robô)",
      not hasattr(k.alvo, "pegar_jitter"),
      "um jitter de ±0,05 em y sobre x = 0,25 desloca o alvo 11° fora do eixo")
check("os elos que exigem o robô PARADO são reorientar, pegar e botar",
      tuple(cfg.commands["alvo_caixa"].elos_parados)
      == (CMD.REORIENTAR, CMD.PEGAR, CMD.BOTAR))
check("a altura de trabalho bate com a pelve nominal + peito_b.z",
      abs(k.alvo.altura_carregar - (0.798 + k.alvo.peito_b[2])) < 0.005,
      f"0,798 (pelve do keyframe, MEDIDA) + {k.alvo.peito_b[2]} = "
      f"{0.798 + k.alvo.peito_b[2]:.3f} vs knob {k.alvo.altura_carregar}")
check("o comando NÃO resampleia dentro do episódio",
      cfg.commands["alvo_caixa"].resampling_time_range[0] > 1e6,
      "com (20,20) o resample rodava UM passo antes do fim e zerava o sucesso")
check("o desenho de debug está LIGADO", cfg.commands["alvo_caixa"].debug_vis is True)
check("o currículo do nível existe e roda ANTES dos eventos",
      "nivel" in cfg.curriculum
      and list(cfg.curriculum).index("nivel") >= 0)
check("o evento de cena existe, e é UM só",
      sum(1 for e in cfg.events if e in ("posiciona_cena",)) == 1)
check("a carga entra por evento (nunca por dr.body_mass)",
      "carga_caixa" in cfg.events,
      "dr.body_mass e dr.pseudo_inertia corrompem a heap — medido")

insp = make_env_cfg(k, inspecao=True)
check("no modo inspeção o robô TRAVA", "trava_robo" in insp.events)
check("no modo inspeção as terminações saem", insp.terminations == {})
check("no TREINO o robô NÃO trava", "trava_robo" not in cfg.events)
# um cfg por ELO. Os cinco montam, e cada um tem a cena daquele elo.
for _nome, _i in zip(CMD.ELOS, range(5)):
    _c = make_env_cfg(k, inspecao=True, elo=_i)
    check(f"o cfg do elo `{_nome}` monta",
          _c.commands["alvo_caixa"].elo_forcado == _i)
    _segura = _i in (CMD.CARREGAR, CMD.BOTAR)
    check(f"`{_nome}`: a caixa {'É' if _segura else 'não é'} posta nas mãos",
          ("segura_caixa" in _c.events) == _segura)
    check(f"`{_nome}`: a caixa {'é' if _segura else 'não é'} PINADA a cada passo",
          ("pina_caixa" in _c.events) == _segura,
          "sem o pino ela cai em ~0,4 s e o clamp do `botar` mede uma caixa no chão")
    _anda = _i in (CMD.ANDAR, CMD.CARREGAR)
    check(f"`{_nome}`: o reset da base usa o yaw "
          f"{'do fabricante (±3,14)' if _anda else 'apertado (±0,2)'}",
          (_c.events["reset_base"].params["pose_range"]["yaw"][1] > 3.0) == _anda,
          "com a mobília a +5 m não há com que alinhar o rumo")

# ------------------------------------------- 13. o DESENHO roda e desenha
secao("13. o desenho do comando (headless, com visualizador de mentira)")


class _Grava:
    """Visualizador de mentira. Só registra o que foi desenhado."""

    def __init__(self) -> None:
        self.frames = self.arrows = self.spheres = self.boxes = 0
        self.labels: list[str] = []

    def get_env_indices(self, num_envs):
        return range(min(2, num_envs))

    @property
    def meansize(self) -> float:
        return 0.1

    def add_frame(self, *a, **kw):
        self.frames += 1
        self.labels.append(str(kw.get("label", "")))

    def add_arrow(self, *a, **kw):
        self.arrows += 1
        self.labels.append(str(kw.get("label", "")))

    def add_sphere(self, *a, **kw):
        self.spheres += 1
        self.labels.append(str(kw.get("label", "")))

    def add_box(self, *a, **kw):
        self.boxes += 1
        self.labels.append(str(kw.get("label", "")))

    def add_cylinder(self, *a, **kw): pass
    def add_ellipsoid(self, *a, **kw): pass
    def add_ghost_mesh(self, *a, **kw): pass
    def clear(self) -> None: pass


try:
    from mjlab.envs import ManagerBasedRlEnv

    _kk = Knobs()
    _kk.nivel.forcado = 3
    _cfg = make_env_cfg(_kk, inspecao=True)
    _cfg.scene.num_envs = 2
    _env = ManagerBasedRlEnv(cfg=_cfg, device="cpu")
    _env.reset()
    import torch as _t
    _env.step(_t.zeros(_env.num_envs, _env.action_manager.total_action_dim))

    _termo = _env.command_manager.get_term("alvo_caixa")
    _v = _Grava()
    _termo._debug_vis_impl(_v)

    check("desenha os EIXOS da caixa (1 por env)", _v.frames == 2, f"{_v.frames}")
    check("desenha a esfera do ALVO e a do ALCANCE", _v.spheres == 4, f"{_v.spheres}")
    check("desenha 3 setas por env (face, erguer, pelve->alvo)",
          _v.arrows == 6, f"{_v.arrows}")
    check("desenha o TOPO da laje", _v.boxes == 2, f"{_v.boxes}")
    check("os rótulos citam a face, o alvo, e o deslocamento até ele",
          any("face" in x for x in _v.labels)
          and any("alvo" in x for x in _v.labels)
          and any("caixa->alvo" in x for x in _v.labels),
          str(sorted({x[:22] for x in _v.labels if x})))
    check("o rótulo do eixo diz QUAL elo e QUAL nível",
          any("[pegar]" in x and "nivel" in x for x in _v.labels),
          str([x for x in _v.labels if x.startswith("[")]))

    # o nível forçado chega mesmo no sorteio
    _cmd = _env.command_manager.get_command("alvo_caixa")
    check("o nível forçado chega ao buffer do env",
          int(_env.limpo_nivel[0]) == 3, str(_env.limpo_nivel.tolist()))
    check("o ângulo respeita a célula do nível forçado",
          float(_t.rad2deg(_cmd[:, CMD.ANG]).abs().max()) <= _kk.nivel.ang_max_deg[3] + 1e-3)
    check("a normal da face é unitária",
          abs(float(_cmd[:, CMD.FACE].norm(dim=-1).min()) - 1.0) < 1e-4)
    check("o objetivo da caixa nasce ATIVO na F0/F1",
          float(_cmd[:, CMD.VALIDA].min()) == 1.0)
    check("o topo e a massa são publicados pelo evento",
          hasattr(_env, "limpo_topo") and hasattr(_env, "limpo_massa"))
    del _env
except Exception as _e:      # noqa: BLE001
    _falhas.append(f"o desenho/env não pôde ser exercitado: {type(_e).__name__}: {_e}")

# =============================================================================
print()
print("=" * 62)
if _falhas:
    print(f"{_ok} ok / {len(_falhas)} FALHAS")
    for f in _falhas:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"{_ok} ok / 0 falhas")
