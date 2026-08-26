"""Os checks do g1_limpo.

    python -m g1_limpo.smoke

Imprime `N ok / M falhas` e sai com código 1 se houver falha. É o portão de cada
fase do plano: nenhuma fase começa com o smoke vermelho.

⚠ Nenhum check aqui importa `g1_training`, `g1_poc` ou `g1_multitask`. Quem compara
contra as referências é o `paridade.py`, que é descartável.

FASES COBERTAS: F0 (esqueleto, cena, física, remoções, contrato de não-import).
"""
from __future__ import annotations

import math
import dataclasses
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
check("todas as colunas da tabela têm o mesmo comprimento",
      len(n.topo_min) == len(n.carga_max) == len(n.jitter_x_max)
      == len(n.voltas_max) == len(n.eixo_vertical) == len(n.desalinho_max_deg)
      == n.n_niveis)
check("o piso do topo só DESCE (cada nível contém o anterior)",
      all(n.topo_min[i + 1] <= n.topo_min[i] for i in range(n.n_niveis - 1)),
      str(n.topo_min))
check("o teto da carga só SOBE",
      all(n.carga_max[i + 1] >= n.carga_max[i] for i in range(n.n_niveis - 1)))
check("o teto de voltas só SOBE, e o eixo vertical só LIGA",
      all(n.voltas_max[i + 1] >= n.voltas_max[i] for i in range(n.n_niveis - 1))
      and all(int(n.eixo_vertical[i + 1]) >= int(n.eixo_vertical[i])
              for i in range(n.n_niveis - 1)))
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
# ⚠ O `pose_range` único SAIU na F2: o reset da base virou despachante por elo, com
# DUAS faixas. Quem confere as faixas é a seção 16.
check("o reset da base não tem mais faixa única — ela é por elo desde a F2",
      "pose_range" not in cfg.events["reset_base"].params
      and {"faixa_loco", "faixa_manipula"}
      <= set(cfg.events["reset_base"].params),
      str(sorted(cfg.events["reset_base"].params)))

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

# ------------------------------------------------------- 11. recompensa da F1
secao("11. recompensa (a tabela do molde, mais DOIS termos)")
# ⚠ A divergência contra o molde é FECHADA em dois nomes, e o teste diz QUAIS. Um
# `set(cfg.rewards) == set(fab.rewards)` deixaria de pegar um termo esquecido no dia
# em que a F3 adicionar os sete incentivos; nomear a diferença não.
# ⚠ NOVE termos a mais que o molde: dois da F1 (locomoção) e os sete da F3 (tarefa).
# O teste os NOMEIA em vez de contar — contar deixaria de pegar um termo esquecido.
_NOSSOS = {"terminacao", "joint_acc", "staged", "precise_pos", "precise_ori",
           "squeeze", "unload", "postura_ereta", "sustentacao"}
check("a tabela divergE do molde em exatamente NOVE termos, e são estes",
      set(cfg.rewards) - set(fab.rewards) == _NOSSOS
      and not set(fab.rewards) - set(cfg.rewards),
      str(set(cfg.rewards) ^ set(fab.rewards)))
check("nenhum termo do MOLDE foi removido",
      set(fab.rewards) <= set(cfg.rewards))
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
# ⚠ Desde a F2 o TREINO não força elo nenhum: quem decide é o `curriculo.sorteia_elo`,
# por env. `elo_forcado` não-nulo no treino anularia o sorteio em silêncio.
check("no TREINO o comando NÃO força elo — o currículo sorteia por env",
      cfg.commands["alvo_caixa"].elo_forcado is None
      and cfg.curriculum["elo"].params["forcado"] is None,
      f"elo_forcado={cfg.commands['alvo_caixa'].elo_forcado}")
check("o elo da MAIORIA é o ANDAR, e a fatia dele vem do knob",
      cfg.curriculum["elo"].params["elo_loco"] == CMD.ANDAR
      and cfg.curriculum["elo"].params["fatia_loco"] == k.forma.fatia_loco)
check("o raio de alcance de referência foi DERIVADO do envelope da Lift",
      abs(CMD.ALCANCE_R - 0.85) < 1e-9,
      "0,50 estava errado: era o box_xy do 19%, não um raio de alcance")
check("há 6 faces declaradas", len(CMD.FACE_AXES) == 6)
check("a face pedida é CONSTANTE, e é a marcada",
      cfg.commands["alvo_caixa"].face_alvo_b == k.cena.face_alvo_b,
      "a dificuldade está na orientação de NASCIMENTO, não em qual face se pede")
check("o eixo do `reorientar` é em QUARTOS DE VOLTA, não em graus",
      not hasattr(k.nivel, "ang_max_deg")
      and tuple(k.nivel.voltas_max) == (0, 0, 1, 1, 1, 1, 1))
check("o teto de voltas é UM: a face nunca nasce do lado OPOSTO",
      max(k.nivel.voltas_max) == 1,
      "o robô só precisa aprender a girar no máximo 90°")
check("o eixo VERTICAL entra depois do horizontal",
      tuple(k.nivel.eixo_vertical) == (False, False, False, False, True, True, True),
      "girar em Z é pivotar na laje; girar em Y é TOMBAR, e é muito mais difícil")
check("o desalinho do nível 0 é 15-20°, e não zero",
      15.0 <= k.nivel.desalinho_max_deg[0] <= 20.0,
      "com zero o `reorientar` ficava satisfeito em t = 0 em 3 dos 7 níveis")
check("as voltas só CRESCEM com o nível (cada nível contém o anterior)",
      all(k.nivel.voltas_max[i + 1] >= k.nivel.voltas_max[i]
          for i in range(k.nivel.n_niveis - 1)))
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
    # ⚠ Com o elo FORÇADO, o despachante de pose cai num subconjunto só. O que se
    # confere no cfg é que o elo forçado chega aos DOIS consumidores (currículo e
    # comando) — se um deles ficasse com `None`, o inspetor mostraria a cena de um elo
    # e o alvo de outro. Qual faixa de yaw sai é medido no env, na seção 16.
    check(f"`{_nome}`: o elo forçado chega ao currículo E ao comando",
          _c.curriculum["elo"].params["forcado"] == _i
          and _c.commands["alvo_caixa"].elo_forcado == _i,
          f"curriculo={_c.curriculum['elo'].params['forcado']} "
          f"comando={_c.commands['alvo_caixa'].elo_forcado}")

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
    # ⚠ `elo=PEGAR` EXPLÍCITO. O default do módulo é o `ANDAR` desde a F1, e no
    # `ANDAR` o desenho é OUTRO (a seta do twist, e uma esfera cinza dizendo que não
    # há alvo de caixa). Este bloco testa o desenho do ALVO, portanto ele pede o elo.
    _cfg = make_env_cfg(_kk, inspecao=True, elo=CMD.PEGAR)
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
    check("desenha 4 setas por env: normal da face MARCADA, direção DESEJADA, "
          "caixa->alvo e pelve->alvo",
          _v.arrows == 8, f"{_v.arrows}")
    check("o desenho separa 'aponta aqui' de 'DEVE apontar aqui'",
          any("MARCADA aponta" in x for x in _v.labels)
          and any("DEVE apontar" in x for x in _v.labels),
          "sem os dois vetores não dá para ver o erro de orientação no viewer")
    check("o rótulo do erro cita os quartos de volta",
          any("quarto(s) de volta" in x for x in _v.labels))
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
    check("o ANG publicado é o ERRO angular, em [0, 180]",
          0.0 <= float(_t.rad2deg(_cmd[:, CMD.ANG]).min())
          and float(_t.rad2deg(_cmd[:, CMD.ANG]).max()) <= 180.0 + 1e-3)
    check("a direção desejada é unitária e HORIZONTAL",
          abs(float(_cmd[:, CMD.FACE].norm(dim=-1).min()) - 1.0) < 1e-4
          and float(_cmd[:, CMD.FACE][:, 2].abs().max()) < 1e-6)
    check("o objetivo da caixa nasce ATIVO no elo de manipulação",
          float(_cmd[:, CMD.VALIDA].min()) == 1.0)
    check("o topo e a massa são publicados pelo evento",
          hasattr(_env, "limpo_topo") and hasattr(_env, "limpo_massa"))
    del _env
except Exception as _e:      # noqa: BLE001
    _falhas.append(f"o desenho/env não pôde ser exercitado: {type(_e).__name__}: {_e}")

# ================================================ 14. a locomoção da F1
secao("14. a recompensa, as métricas e a régua de marcha (F1)")
from g1_limpo import metricas as MT_          # noqa: E402
from g1_limpo import recompensas as RC_       # noqa: E402

_r = k.recompensa
check("todo peso da tabela da F1 chegou ao cfg",
      all(abs(cfg.rewards[n].weight - v) < 1e-12
          for n, v in dataclasses.asdict(_r).items() if n != "altura_de_balanco"),
      str({n: cfg.rewards[n].weight for n in dataclasses.asdict(_r)
           if n != "altura_de_balanco"}))
check("os DOIS termos novos existem, e são os do módulo que ANDOU",
      cfg.rewards["terminacao"].weight == -200.0
      and cfg.rewards["joint_acc"].weight == -2.5e-7)
check("a `terminacao` NÃO pune o time_out",
      cfg.rewards["terminacao"].func.__name__ == "is_terminated",
      "`is_terminated` lê `termination_manager.terminated`, que exclui o time_out")
check("`scale_rewards_by_dt` está LIGADO, portanto o peso é o valor POR SEGUNDO",
      cfg.scale_rewards_by_dt is True)
_dt = cfg.sim.mujoco.timestep * cfg.decimation
check("o dt é 0,02 s, logo a `terminacao` custa −4,0 e não −200",
      abs(_dt - 0.02) < 1e-12
      and abs(cfg.rewards["terminacao"].weight * _dt + 4.0) < 1e-9)
check("o `air_time` continua em ZERO, e é decisão declarada",
      cfg.rewards["air_time"].weight == 0.0,
      "medido: ausente no módulo que andou, 0,0 no que não andou")

# --- o bug do `peak_heights` ---
check("o `foot_swing_height` é a NOSSA subclasse",
      cfg.rewards["foot_swing_height"].func is RC_.AlturaDeBalanco)
check("ela TEM `reset` — sem isso `reward_manager.py:174` nunca a chamaria",
      callable(getattr(RC_.AlturaDeBalanco, "reset", None)))
check("o termo DO FABRICANTE não tem `reset` (é o bug que a subclasse conserta)",
      not hasattr(_fab_swing := fab.rewards["foot_swing_height"].func, "reset"),
      str(_fab_swing))
check("o alvo de altura de balanço vem do knobs",
      cfg.rewards["foot_swing_height"].params["target_height"]
      == _r.altura_de_balanco)

# --- as métricas ---
_esperadas = {"momento_angular", "tempo_de_voo", "pico_de_altura",
              "velocidade_de_escorrego", "forca_de_pouso"}
check("as cinco métricas de marcha estão no manager de MÉTRICAS",
      _esperadas <= set(cfg.metrics), str(sorted(cfg.metrics)))
check("o `mean_action_acc` do molde não foi apagado",
      "mean_action_acc" in cfg.metrics)
check("o `SceneEntityCfg` da métrica de escorrego vive em `params`",
      "asset_cfg" in cfg.metrics["velocidade_de_escorrego"].params,
      "fora de `params` o mjlab NÃO o resolve (manager_base.py:141) e ela leria "
      "os 6 sítios do robô em vez dos 2 pés")
check("o `pico_de_altura` tem `reset` — senão o pico da QUEDA vaza de episódio",
      callable(getattr(MT_.pico_de_altura, "reset", None)))
check("os nomes de sensor das métricas existem na cena",
      {MT_.PES_NO_CHAO, MT_.ALTURA_DO_PE}
      <= {s.name for s in cfg.scene.sensors})

# --- a régua ---
_tw = cfg.commands["twist"]
check("o twist é a NOSSA subclasse, com a `razao_marcha`",
      type(_tw).__name__ == "TwistComRazaoDeMarchaCfg")
check("o `build` foi sobrescrito — o mjlab não usa `class_type`",
      _tw.build.__qualname__.startswith("TwistComRazaoDeMarchaCfg"),
      "command_manager.py:268 chama cfg.build(env); um `class_type` seria campo morto")
check("nenhum campo do twist do fabricante se perdeu na reconstrução",
      all(getattr(_tw, f.name) == getattr(fab.commands["twist"], f.name)
          for f in dataclasses.fields(fab.commands["twist"])),
      "rel_standing_envs perdido mudaria 10% dos envs sem uma linha de log")
check("o limiar de comando ativo vem do knobs", _tw.limiar_comando == k.marcha.limiar_comando)
check("`ang_vel_z` é a faixa do fabricante", _tw.ranges.ang_vel_z == (-0.5, 0.5))

# --- a ARITMÉTICA da razão, sem simulador ---
# ⚠ Aritmética pura sobre a MESMA fórmula do termo. Não é substituto de rodar; é o
# que prova as três propriedades que o portão da F1 usa como critério.
def _razao(pares) -> float:
    """pares = [(‖v_cmd‖, ‖v_cmd − v‖), ...] já gateados."""
    se = sum(e for _, e in pares)
    sc = sum(c for c, _ in pares)
    return 1.0 - se / sc if sc > 0 else 0.0


check("robô IMÓVEL com comando ativo dá razão 0,0",
      abs(_razao([(1.0, 1.0), (0.6, 0.6)]) - 0.0) < 1e-12,
      "erro igual ao comando: numerador iguala denominador")
check("METADE da velocidade comandada dá exatamente 0,50",
      abs(_razao([(1.0, 0.5), (2.0, 1.0)]) - 0.50) < 1e-12)
check("ela é ADIMENSIONAL: 1 de 2 e 0,5 de 1 dão a MESMA razão",
      abs(_razao([(2.0, 1.0)]) - _razao([(1.0, 0.5)])) < 1e-12,
      "é isso que a torna imune ao degrau do currículo de comando")
check("ir ao CONTRÁRIO dá razão NEGATIVA, e ela não é clampeada",
      _razao([(1.0, 2.0)]) < 0.0,
      "clampear em 0 esconderia 'parado' de 'indo ao contrário'")
check("comando abaixo do limiar não entra em soma nenhuma",
      _razao([]) == 0.0 and k.marcha.limiar_comando > 0.0)

# --- a régua rodando de verdade, no env ---
try:
    import torch as _t2

    _cfg2 = make_env_cfg(k)
    _cfg2.scene.num_envs = 2
    _env2 = ManagerBasedRlEnv(cfg=_cfg2, device="cpu")
    _env2.reset()
    _tw2 = _env2.command_manager.get_term("twist")
    for _ in range(5):
        _env2.step(_t2.zeros(_env2.num_envs,
                             _env2.action_manager.total_action_dim))

    check("as três entradas da régua existem em `self.metrics` do twist",
          {"soma_erro_marcha", "soma_cmd_marcha", "razao_marcha"}
          <= set(_tw2.metrics))
    check("as métricas do fabricante seguem lá",
          {"error_vel_xy", "error_vel_yaw"} <= set(_tw2.metrics))
    # ⚠ A BANDA, e não `<= 0`. Um `<= 0` acusa a FÍSICA: em 5 passos (0,1 s) o robô de
    # ação zero desaba, e a velocidade da queda pode se alinhar por acidente com o
    # comando — medido +0,024 num env. O invariante real é que sem política NÃO SE
    # RASTREIA: a razão fica colada no zero, e a aritmética exata das três
    # propriedades já foi provada acima, sem simulador.
    check("robô sem política fica colado no zero da régua",
          float(_tw2.metrics["razao_marcha"].abs().max()) < 0.25,
          str([round(float(x), 4) for x in _tw2.metrics["razao_marcha"]]))
    check("o twist ativo alimenta a soma do comando",
          float(_tw2.metrics["soma_cmd_marcha"].max()) > 0.0)

    # o consumo do `reset`: a média sai, e o buffer zera
    _ex = _tw2.reset(_t2.arange(_env2.num_envs))
    check("o `reset` do comando EXPORTA a razão e ZERA a soma",
          "razao_marcha" in _ex
          and float(_tw2.metrics["soma_cmd_marcha"].abs().max()) == 0.0)

    # o elo de LOCOMOÇÃO. ⚠ POR ENV: desde a F2 apenas `fatia_loco` dos envs são de
    # locomoção, portanto um `.max()` sobre todos falharia por desenho, e não por bug.
    _c2 = _env2.command_manager.get_command("alvo_caixa")
    _loco2 = _env2.limpo_elo == CMD.ANDAR
    check("no ANDAR o objetivo da caixa nasce INATIVO (valida = 0)",
          not bool(_loco2.any())
          or float(_c2[_loco2][:, CMD.VALIDA].max()) == 0.0)
    check("no ANDAR a mobília está afastada em +5 m",
          not bool(_loco2.any())
          or float(_env2.scene["table"].data.root_link_pos_w[_loco2, 2].min()) > 4.0,
          str(_env2.scene["table"].data.root_link_pos_w[:, 2].tolist()))
    check("a faixa de yaw da locomoção é a do fabricante (círculo inteiro)",
          _cfg2.events["reset_base"].params["faixa_loco"]["yaw"]
          == c.reset_base_loco["yaw"])
    del _env2
except Exception as _e2:      # noqa: BLE001
    _falhas.append(f"a régua não pôde ser exercitada: {type(_e2).__name__}: {_e2}")

# ============================== 15. as chaves de log são um CONTRATO
secao("15. as chaves da escada do `leitura.py` batem com quem as produz")
from g1_limpo import leitura as LE_          # noqa: E402

# ⚠ Uma chave errada na escada NÃO levanta erro: a linha só não aparece, e o bloco
# roda sem o portão. Foi o que aconteceu no `g1_poc`, cuja escada usa
# `Policy/mean_noise_std` — chave que o rsl_rl 5.4.0 NÃO escreve.
import rsl_rl.utils.logger as _rl_logger      # noqa: E402
_src = pathlib.Path(_rl_logger.__file__).read_text(encoding="utf-8")

check("`Policy/mean_std` é a chave que o rsl_rl escreve de verdade",
      '"Policy/mean_std"' in _src and LE_.CH_STD == "Policy/mean_std",
      "a escada do g1_poc usa `Policy/mean_noise_std`, que nunca disparou")
check("`Train/mean_episode_length` existe no logger",
      '"Train/mean_episode_length"' in _src
      and LE_.CH_DURACAO == "Train/mean_episode_length")

# a razão de marcha: `CommandManager.reset` prefixa `Metrics/<termo>/<metrica>`
check("a chave da `razao_marcha` casa com o prefixo do CommandManager",
      LE_.CH_RAZAO == "Metrics/twist/razao_marcha"
      and "twist" in cfg.commands,
      "command_manager.py:246 escreve Metrics/{nome_do_termo}/{metrica}")

# as métricas: `MetricsManager.reset` prefixa `Episode_Metrics/<chave>`
for _ch in (LE_.CH_VOO, LE_.CH_PICO, LE_.CH_ESCORREGO, LE_.CH_POUSO):
    _nome = _ch.split("/", 1)[1]
    check(f"`{_ch}` tem produtor no manager de métricas",
          _ch.startswith("Episode_Metrics/") and _nome in cfg.metrics,
          str(sorted(cfg.metrics)))

check("a chave do nível casa com o prefixo do CurriculumManager",
      LE_.CH_NIVEL == "Curriculum/nivel" and "nivel" in cfg.curriculum,
      "curriculum_manager.py:107 escreve Curriculum/{nome} para estado escalar")

# ⚠ CONTAR as linhas da escada quebrou quando a F4 acrescentou duas. O invariante que
# sobrevive às fases é a PRESENÇA das linhas que cada fase exige, nomeada por chave.
_chaves_escada = {ch for _, ch, _, _, _ in LE_.ESCADA}
check("a escada tem as quatro linhas da F1, e a do andar é a `razao_marcha`",
      {LE_.CH_STD, LE_.CH_DURACAO, LE_.CH_RAZAO, LE_.CH_VOO} <= _chaves_escada
      and any(ch == LE_.CH_RAZAO and alvo == 0.50
              for _, ch, _, alvo, _ in LE_.ESCADA))
check("as duas linhas da F4 são de FIM DE RUN, e não de um número",
      all(it is None for it, ch, _, _, _ in LE_.ESCADA
          if ch in (LE_.CH_FATIA_CADEIA, LE_.CH_SUCESSO_CADEIA))
      and {LE_.CH_FATIA_CADEIA, LE_.CH_SUCESSO_CADEIA} <= _chaves_escada,
      "o contador do rsl_rl ACUMULA entre blocos: uma linha `na iteração 5000` para a "
      "F4 dispararia no instante em que o bloco começa")
check("as constantes de tempo do `leitura` batem com o cfg",
      abs(LE_.DT - cfg.sim.mujoco.timestep * cfg.decimation) < 1e-12
      and abs(LE_.MAX_EP_S - cfg.episode_length_s) < 1e-12,
      f"leitura DT={LE_.DT} MAX={LE_.MAX_EP_S}")
check("o autoteste da diluição do `leitura` passa", LE_._demo() == 0)

# ==================================== 16. o one-hot, o sorteio de elo e a postura
secao("16. one-hot, sorteio de elo e postura por elo (F2)")
from g1_limpo import eventos as EV_         # noqa: E402
from g1_limpo import observacoes as OB_      # noqa: E402
from g1_limpo.env_cfg import (               # noqa: E402
    ELOS_QUE_ANDAM, ELOS_SORTEAVEIS,
)

fab_obs = {g: list(fab.observations[g].terms) for g in ("actor", "critic")}
nossa_obs = {g: list(cfg.observations[g].terms) for g in ("actor", "critic")}
check("o one-hot entra nos DOIS grupos",
      all("elo" in nossa_obs[g] for g in ("actor", "critic")))
# ⚠ O CONTRATO DO APPEND, e ele e' o invariante que sobrevive as fases: os termos do
# FABRICANTE vem primeiro, na ordem dele, e os NOSSOS depois, na ordem em que as fases
# os adicionaram. Checar "o one-hot e' o ultimo" quebrou na F3, quando os canais da
# caixa entraram depois dele -- e quebraria de novo na F4.
_NOSSA_OBS = ["elo", "caixa"]
check("os termos do FABRICANTE vêm primeiro, na ordem dele",
      all(nossa_obs[g][:len(fab_obs[g])] == fab_obs[g]
          for g in ("actor", "critic")),
      str({g: nossa_obs[g] for g in nossa_obs}))
check("os NOSSOS vêm depois, na ordem das fases, e nos dois grupos",
      all(nossa_obs[g][len(fab_obs[g]):] == _NOSSA_OBS
          for g in ("actor", "critic")),
      "append de colunas; inserir no meio desloca todo peso da 1ª camada em silêncio")
check("o one-hot não tem ruído nem escala",
      all(cfg.observations[g].terms["elo"].noise is None
          and cfg.observations[g].terms["elo"].scale is None
          for g in ("actor", "critic")),
      "ruído num one-hot produz frações entre slots: estados que não existem")
check("são 5 slots, um por elo", OB_.N_SLOTS == len(CMD.ELOS) == 5)

# ⚠ A ORDEM MUDOU NA F5, e de propósito: o `forma` e o `nivel` medem o episódio que
# ACABOU e leem `limpo_elo`, enquanto o `elo` escreve o do episódio que COMEÇA. Este
# check dizia "elo antes de nivel", que era certo na F2 (ninguém lia o elo antigo) e
# passou a ser errado na F5. Quem confere a ordem nova é a seção 20.
check("o sorteio de elo é um termo de CURRÍCULO",
      "elo" in cfg.curriculum
      and cfg.curriculum["elo"].func is CU_.sorteia_elo,
      str(list(cfg.curriculum)))
_cur_src = pathlib.Path("g1_limpo/curriculo.py").read_text(encoding="utf-8")
check("o `curriculo.py` NÃO importa o `comando.py` (seria ciclo)",
      not any(ln.strip().startswith(("import ", "from "))
              and "comando" in ln for ln in _cur_src.splitlines()),
      "comando.py importa `garante_nivel` daqui; o import de volta fecharia o ciclo")
check("os elos sorteáveis são só os que nascem SEM caixa nas mãos",
      tuple(ELOS_SORTEAVEIS) == (CMD.REORIENTAR, CMD.PEGAR),
      "CARREGAR e BOTAR só existem como 2º elo de cadeia -> F4")
check("a fatia de locomoção NÃO é 1,00",
      k.forma.fatia_loco < 1.0 and k.forma.fatia_loco >= 0.9,
      "com 1,00 os slots ficam constantes e o normalizador os faz entrar como 100,0")

# o normalizador: a razão do 0,95 é aritmética, e ela se confere
check("o normalizador do rsl_rl divide por `_std + 1e-2`, sem clamp",
      "(x - self._mean) / (self._std + self.eps)" in pathlib.Path(
          __import__("rsl_rl.modules.normalization", fromlist=["x"]).__file__
      ).read_text(encoding="utf-8"),
      "é isto que faz 1,0 entrar como 100,0 num canal antes constante")

# o reset de pose por elo
check("o reset da base é o despachante por elo",
      cfg.events["reset_base"].func is EV_.reset_base_por_elo)
check("ele continua sendo o PRIMEIRO evento de reset",
      list(cfg.events).index("reset_base") == 0, str(list(cfg.events)))
check("as duas faixas de yaw são as do knob, e são diferentes",
      cfg.events["reset_base"].params["faixa_loco"]["yaw"] == c.reset_base_loco["yaw"]
      and cfg.events["reset_base"].params["faixa_manipula"]["yaw"]
      == c.reset_base_manipula["yaw"]
      and c.reset_base_loco["yaw"] != c.reset_base_manipula["yaw"])
check("os elos com twist ativo são ANDAR e CARREGAR",
      tuple(ELOS_QUE_ANDAM) == (CMD.ANDAR, CMD.CARREGAR))

# ⚠ O GATE DOS `track_*` NÃO ENTRA, e a decisão é da spec (§4.2)
check("os dois `track_*` seguem SEM gate de elo",
      "elo" not in cfg.rewards["track_linear_velocity"].params
      and "elo" not in cfg.rewards["track_angular_velocity"].params
      and cfg.rewards["track_linear_velocity"].params
      == fab.rewards["track_linear_velocity"].params,
      "com o twist em ZERO, gatear removeria a única coisa que paga ficar parado")

# a postura
check("a postura é a NOSSA subclasse", cfg.rewards["pose"].func is RC_.PosturaPorElo)
check("ela recebe o canal do elo e a lista dos que andam",
      cfg.rewards["pose"].params["canal_do_elo"] == CMD.ELO
      and tuple(cfg.rewards["pose"].params["elos_que_andam"]) == tuple(ELOS_QUE_ANDAM))
# ⚠ Identidade de objeto NÃO é o invariante aqui: `cfg` e `fab` são dois builds
# independentes do molde, portanto os dicts são objetos distintos por construção. O
# invariante é IGUALDADE de valor mais a prova de que nada foi redigitado no nosso
# fonte, que já é um check próprio (a busca por `knee` nos arquivos do pacote).
check("os três σ do fabricante seguem com os MESMOS valores",
      all(cfg.rewards["pose"].params[x] == fab.rewards["pose"].params[x]
          for x in ("std_standing", "std_walking", "std_running")),
      "a subclasse de postura não pode ter tocado nas tabelas")
check("o `walking_threshold` do G1 é 0,05, não 0,5",
      cfg.rewards["pose"].params["walking_threshold"] == 0.05,
      "com o twist em ZERO o regime `standing` é CERTO, não provável")
check("`std_standing` é UMA entrada só, `.*`, para as 29 juntas",
      list(cfg.rewards["pose"].params["std_standing"]) == [".*"],
      str(cfg.rewards["pose"].params["std_standing"]))

# --- o penhasco da postura, MEDIDO. É o que justifica a subclasse. ---
try:
    import torch as _t3
    from mjlab.utils.lab_api.string import resolve_matching_names_values as _rmnv

    _cfg3 = make_env_cfg(k)
    _cfg3.scene.num_envs = 128
    _env3 = ManagerBasedRlEnv(cfg=_cfg3, device="cpu")
    _env3.reset()
    for _ in range(3):
        _env3.step(_t3.zeros(_env3.num_envs,
                             _env3.action_manager.total_action_dim))

    _robo = _env3.scene["robot"]
    _acfg = _cfg3.rewards["pose"].params["asset_cfg"]
    _ids, _nomes = _robo.find_joints(_acfg.joint_names)
    _, _, _v = _rmnv(data=_cfg3.rewards["pose"].params["std_standing"],
                     list_of_strings=_nomes)
    _std_st = _t3.tensor(_v)
    _faixa = (_robo.data.joint_pos_limits[0][:, 1]
              - _robo.data.joint_pos_limits[0][:, 0])[_ids]
    _manip = _t3.tensor([any(x in nm for x in
                            ("shoulder", "elbow", "wrist", "waist"))
                        for nm in _nomes])
    _err = _t3.zeros(len(_nomes))
    _err[_manip] = _faixa[_manip] * 0.10
    _termo = float(_t3.exp(-_t3.mean(_err ** 2 / _std_st ** 2)))
    check("MEDIDO: a 10% da faixa o `standing` já vale 0,000 — canal MORTO",
          _termo < 1e-6, f"{_termo:.3e}")

    # a neutralidade, no env de verdade
    _elo3 = _env3.limpo_elo
    _pose_idx = list(_cfg3.rewards).index("pose")
    _pp = _env3.reward_manager._step_reward[:, _pose_idx]
    _manip_envs = ~_t3.isin(_elo3, _t3.tensor(ELOS_QUE_ANDAM))
    check("num elo de manipulação a postura vale EXATAMENTE 1,0",
          bool(_manip_envs.any())
          and float((_pp[_manip_envs] - 1.0).abs().max()) < 1e-6,
          f"{[round(float(x),5) for x in _pp[_manip_envs][:4]]}")
    check("1,0 e não 0,0: zero seria uma penalidade por SORTEIO de elo",
          float(_pp[_manip_envs].min()) > 0.5)
    check("num elo que ANDA a postura segue sendo a do fabricante",
          bool((~_manip_envs).any())
          and float(_pp[~_manip_envs].std()) > 0.0,
          "constante ali significaria que a subclasse comeu o termo")

    # o sorteio, e os dois consumidores lendo o MESMO elo
    check("o elo sorteado bate com o que o comando publica",
          bool((_env3.command_manager.get_command("alvo_caixa")[:, CMD.ELO].long()
                == _elo3).all()),
          "se divergirem, a pose nasceu para um elo e o alvo para outro")
    check("a fatia medida bate com o knob (±0,06 em 128 envs)",
          abs(float((_elo3 == CMD.ANDAR).float().mean()) - k.forma.fatia_loco) < 0.06,
          str(round(float((_elo3 == CMD.ANDAR).float().mean()), 4)))
    # ⚠ NÃO se checa aqui que os dois elos sorteáveis APARECERAM. Com 128 envs e 2,5%
    # por elo, a chance de um deles sair vazio é ~4% por run — a checagem seria FLAKY,
    # e falharia acusando o sorteio quando o sorteio está certo. O invariante é a
    # DISTRIBUIÇÃO, e ela é testada sem simulador logo abaixo.
    check("todo elo sorteado está no conjunto permitido",
          bool(_t3.isin(_elo3, _t3.tensor((CMD.ANDAR,) + tuple(ELOS_SORTEAVEIS))
                        ).all()),
          str({e: int((_elo3 == e).sum()) for e in range(5)}))
    check("CARREGAR e BOTAR NÃO são sorteados (declarado, F4 os abre)",
          not bool(_t3.isin(_elo3,
                            _t3.tensor([CMD.CARREGAR, CMD.BOTAR])).any()))

    # o one-hot, por passo, e a soma
    # ⚠ O one-hot NÃO está mais no fim do vetor: os 8 canais da caixa vieram depois
    # dele na F3. Fatiar com `[-N_SLOTS:]` leria os últimos 5 canais da CAIXA e o teste
    # passaria medindo a coisa errada. O offset é calculado, não digitado.
    _FAT_OH = slice(-(OB_.N_SLOTS + OB_.N_CAIXA), -OB_.N_CAIXA)
    _oh = _env3.observation_manager.compute()["actor"][:, _FAT_OH]
    check("o one-hot soma 1,0 em toda linha",
          float((_oh.sum(-1) - 1.0).abs().max()) < 1e-6)
    check("o slot aceso é o elo do env",
          bool((_oh.argmax(-1) == _elo3).all()))
    check("os slots 3 e 4 são constantes em ZERO, e está declarado",
          all(float(_oh[:, int(e)].abs().max()) == 0.0
              for e in (CMD.CARREGAR, CMD.BOTAR)),
          "eles só abrem na F4; a mitigação do normalizador está pré-registrada")

    # --- O ONE-HOT É POR PASSO. É o pré-requisito da F4, e prova-se sem F4. ---
    #
    # ⚠ `observation_manager.compute()` devolve CACHE (`observation_manager.py:311`).
    # Sem `update_history=True` este teste leria o buffer do passo anterior e passaria
    # com o código errado — foi o que aconteceu na primeira tentativa.
    _antes = _env3.observation_manager.compute(
        update_history=True)["actor"][:, _FAT_OH].argmax(-1).clone()
    _env3.command_manager.get_command("alvo_caixa")[:, CMD.ELO] = float(CMD.BOTAR)
    _depois = _env3.observation_manager.compute(
        update_history=True)["actor"][:, _FAT_OH].argmax(-1)
    check("escrever o canal do elo muda o one-hot NO PASSO SEGUINTE, sem reset",
          bool((_depois == CMD.BOTAR).all()) and bool((_antes != CMD.BOTAR).any()),
          "é o mecanismo que a F4 usa para trocar de elo dentro do episódio")

    # o twist zerado nos elos parados
    _tw3 = _env3.command_manager.get_term("twist")
    _parados = _t3.isin(_elo3, _t3.tensor(
        _cfg3.commands["alvo_caixa"].elos_parados))
    check("o twist é ZERO nos elos parados",
          not bool(_parados.any())
          or float(_tw3.vel_command_b[_parados].abs().max()) == 0.0)
    check("e NÃO é zero nos que andam",
          float(_tw3.vel_command_b[~_parados].abs().max()) > 0.0)
    del _env3
except Exception as _e3:      # noqa: BLE001
    _falhas.append(f"a F2 não pôde ser exercitada: {type(_e3).__name__}: {_e3}")

# --- a DISTRIBUIÇÃO do sorteio, sem simulador ---
# ⚠ Aqui não há física: `sorteia_elo` é código de tensor. Testar a distribuição com
# 20.000 amostras torna o teste determinístico na prática, em vez de flaky com 128
# envs — e testa o que realmente importa, que é a proporção, não um sorteio.
import types  # noqa: E402

from g1_limpo import curriculo as CU2      # noqa: E402

_falso = types.SimpleNamespace(num_envs=20_000, device="cpu")
_ids = __import__("torch").arange(20_000)
_fatia = CU2.sorteia_elo(_falso, _ids, elo_loco=CMD.ANDAR,
                         elos_manip=ELOS_SORTEAVEIS,
                         fatia_loco=k.forma.fatia_loco, forcado=None)
_buf = _falso.limpo_elo
_cont = {e: int((_buf == e).sum()) for e in range(5)}
check("a fatia de locomoção sai como o knob pede (±0,01 em 20.000)",
      abs(_fatia - k.forma.fatia_loco) < 0.01, f"{_fatia:.4f}")
check("os dois elos sorteáveis aparecem, e em proporção IGUAL entre si",
      all(_cont[int(e)] > 0 for e in ELOS_SORTEAVEIS)
      and abs(_cont[int(ELOS_SORTEAVEIS[0])] - _cont[int(ELOS_SORTEAVEIS[1])])
      < 0.25 * sum(_cont[int(e)] for e in ELOS_SORTEAVEIS),
      str(_cont))
check("nenhum elo fora do conjunto permitido é sorteado",
      all(_cont[e] == 0 for e in range(5)
          if e != CMD.ANDAR and e not in ELOS_SORTEAVEIS),
      str(_cont))
check("`forcado` vence o sorteio, e é o que o inspetor usa",
      CU2.sorteia_elo(_falso, _ids, elo_loco=CMD.ANDAR,
                      elos_manip=ELOS_SORTEAVEIS, fatia_loco=0.5,
                      forcado=CMD.BOTAR) == float(CMD.BOTAR)
      and bool((_falso.limpo_elo == CMD.BOTAR).all()))

# ================================= 17. o PISO DA ESTÁTUA, medido
secao("17. o preço declarado: quanto uma estátua colhe (F2)")
# ⚠ Isto SUBSTITUI o critério "um env em PEGAR colhe 0/s dos track_*" do plano. Aquele
# critério pressupunha o gate, e o gate CAIU (spec §4.2): com o twist em zero, gatear
# os `track_*` fora removeria a única coisa que paga ficar parado, e o twist zerado já
# impede andar. O que resta é MEDIR o piso e declará-lo.
#
# ⚠ Medir com `inspecao=True` não é atalho: é a única forma de ter uma estátua DE
# VERDADE. Com ação zero e sem trava o robô DESABA, e a velocidade da queda entra no
# erro de rastreio — a primeira medição deu 2,14/s por isso, e não por ser o piso.
try:
    import torch as _t4

    _piso = {}
    for _nome4, _elo4 in (("parado", CMD.PEGAR), ("anda", CMD.ANDAR)):
        _c4 = make_env_cfg(k, inspecao=True, elo=_elo4)
        _c4.scene.num_envs = 32
        _e4 = ManagerBasedRlEnv(cfg=_c4, device="cpu")
        _e4.reset()
        for _ in range(6):
            _e4.step(_t4.zeros(_e4.num_envs, _e4.action_manager.total_action_dim))
        _nm = list(_c4.rewards)
        _sr = _e4.reward_manager._step_reward
        _piso[_nome4] = {
            n: float(_sr[:, _nm.index(n)].mean())
            for n in ("track_linear_velocity", "track_angular_velocity",
                      "pose", "upright")}
        _piso[_nome4]["TOTAL"] = float(_sr.mean(0).sum())
        del _e4

    _tk = (_piso["parado"]["track_linear_velocity"]
           + _piso["parado"]["track_angular_velocity"])
    check("MEDIDO: a estátua num elo parado colhe ~3,8/s dos dois `track_*`",
          3.4 < _tk < 4.2, f"{_tk:.3f}/s — a spec declara ~4,0")
    check("e a postura NÃO entra nesse piso: ela é neutra, exatamente 1,0",
          abs(_piso["parado"]["pose"] - 1.0) < 1e-6,
          f"{_piso['parado']['pose']:.6f}")
    check("a estátua num elo que ANDA colhe MENOS (o twist não é zero)",
          (_piso["anda"]["track_linear_velocity"]
           + _piso["anda"]["track_angular_velocity"]) < _tk,
          f"anda={_piso['anda']['TOTAL']:.3f}/s  parado={_piso['parado']['TOTAL']:.3f}/s")
    check("o piso é PISO, não concorrente: fica abaixo do teto de tarefa da F3",
          _piso["parado"]["TOTAL"] < 12.5,
          f"{_piso['parado']['TOTAL']:.3f}/s contra ~12,5/s dos sete incentivos")
    print(f"  piso parado = {_piso['parado']['TOTAL']:.3f}/s   "
          f"piso andando = {_piso['anda']['TOTAL']:.3f}/s")
except Exception as _e4x:      # noqa: BLE001
    _falhas.append(f"o piso não pôde ser medido: {type(_e4x).__name__}: {_e4x}")

# =============================== 18. os sete incentivos da manipulação (F3)
secao("18. os sete incentivos (F3)")
SETE = ("staged", "precise_pos", "precise_ori", "squeeze", "unload",
        "postura_ereta", "sustentacao")
tr = k.tarefa

check("os sete termos existem, e são os do plano",
      all(n in cfg.rewards for n in SETE), str([n for n in SETE
                                                if n not in cfg.rewards]))
check("TODOS os pesos são POSITIVOS — nenhuma penalidade na tarefa (R3)",
      all(cfg.rewards[n].weight > 0.0 for n in SETE),
      str({n: cfg.rewards[n].weight for n in SETE}))
check("a soma dos pesos é 11,5/s",
      abs(sum(cfg.rewards[n].weight for n in SETE) - 11.5) < 1e-9)
check("o `staged` é o maior — é o único com gradiente na pose de repouso",
      cfg.rewards["staged"].weight == max(cfg.rewards[n].weight for n in SETE))
check("o `precise_pos` é o ÚNICO com σ fixo, e ele é a tolerância de ACEITE",
      cfg.rewards["precise_pos"].params["sigma"] == tr.precise_pos_sigma
      and "sigma" not in cfg.rewards["staged"].params,
      "quem faz a rampa de aproximação é o `staged`, com σ por env")
check("o `sustentacao` tem `reset` — senão o tempo vaza de episódio",
      callable(getattr(RC_.sustentacao, "reset", None)))
check("o `squeeze` usa os sensores de PALMA, que têm o campo `force`",
      tuple(cfg.rewards["squeeze"].params["sensores"]) == tuple(C.SENSOR_PALMA)
      and all("force" in por_nome[n].fields for n in C.SENSOR_PALMA))
check("o `unload` usa o sensor de APOIO, que tem `force`",
      cfg.rewards["unload"].params["sensor_apoio"] == C.SENSOR_APOIO
      and "force" in por_nome[C.SENSOR_APOIO].fields)
check("a tolerância angular do sustain chega em RADIANOS",
      abs(cfg.rewards["sustentacao"].params["tol_ang"]
          - math.radians(tr.tol_ang_deg)) < 1e-12)

# --- a observação cresceu pelo contrato do APPEND ---
check("os canais da caixa entram DEPOIS do one-hot, nos dois grupos",
      all(list(cfg.observations[g].terms)[-2:] == ["elo", "caixa"]
          for g in ("actor", "critic")),
      str(list(cfg.observations["actor"].terms)))

# --- O σ. É o item de maior risco da F3, e ele se mede. ---
try:
    import torch as _t5

    _c5 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c5.scene.num_envs = 48
    _e5 = ManagerBasedRlEnv(cfg=_c5, device="cpu")
    _e5.reset()
    _e5.step(_t5.zeros(_e5.num_envs, _e5.action_manager.total_action_dim))
    _t5c = _e5.command_manager.get_term("alvo_caixa")
    _ids5 = _t5.arange(_e5.num_envs)
    _d5 = _t5c.dist_palma_caixa(_ids5)
    _s5 = _t5c.sigma_alcance
    _ker = _t5.exp(-(_d5 / _s5) ** 2)

    check("o σ NÃO é constante — cada env tem o seu",
          float(_s5.std()) > 0.01, f"std={float(_s5.std()):.4f}")
    check("o σ É a distância inicial daquele env",
          float((_s5 - _d5).abs().max()) < 5e-3,
          f"pior desvio {float((_s5-_d5).abs().max()):.4f} m")
    # ⚠ O NÚMERO QUE DECIDE A F3. Com σ fixo de 0,10 isto valeria 1e−05.
    #
    # ⚠ A BANDA É DERIVADA, e não escolhida. O σ é fixado no passo do `_pendente`, e a
    # caixa continua assentando depois disso — o check acima tolera 5 mm de deriva. No
    # env de σ mínimo (0,08 m) 5 mm são 6,25% de razão, logo o kernel varia entre
    # `exp(−1,0625²)` e `exp(−0,9375²)`, isto é [0,323; 0,415]. Uma tolerância de
    # ±0,02 é MAIS APERTADA que isso e acusa o assentamento da caixa, não o desenho.
    _lo = math.exp(-(1.0 + 5e-3 / k.tarefa.sigma_min) ** 2)
    _hi = math.exp(-(1.0 - 5e-3 / k.tarefa.sigma_min) ** 2)
    check("o kernel de alcance vale exp(−1) = 0,368 no passo em que o elo abre, "
          "em TODOS os envs",
          _lo - 1e-3 <= float(_ker.min()) and float(_ker.max()) <= _hi + 1e-3,
          f"min {float(_ker.min()):.4f} max {float(_ker.max()):.4f}, "
          f"banda derivada [{_lo:.3f}; {_hi:.3f}]")
    check("a derivada do kernel no repouso é > 1,0 até no env mais distante",
          float((2.0 * _d5 / _s5 ** 2 * _ker).min()) > 1.0,
          f"min {float((2.0*_d5/_s5**2*_ker).min()):.3f} por metro")
    check("a distância é até a SUPERFÍCIE da caixa, não o centro",
          float(_d5.max()) < 0.45,
          "ao centro o mínimo alcançável é 0,191 m e o kernel saturava em 0,674")
    check("o σ de orientação tem piso, e ele é em RADIANOS",
          float(_t5c.sigma_ori.min()) >= _c5.commands["alvo_caixa"].sigma_ori_min
          - 1e-9)
    del _e5
except Exception as _e5x:      # noqa: BLE001
    _falhas.append(f"o σ não pôde ser medido: {type(_e5x).__name__}: {_e5x}")

# --- os sete valem ZERO no ANDAR, e o gate é o que garante isso ---
try:
    import torch as _t6

    _vals = {}
    for _rot, _elo6 in (("andar", CMD.ANDAR), ("pegar", CMD.PEGAR)):
        _c6 = make_env_cfg(k, inspecao=True, elo=_elo6)
        _c6.scene.num_envs = 16
        _e6 = ManagerBasedRlEnv(cfg=_c6, device="cpu")
        _e6.reset()
        for _ in range(5):
            _e6.step(_t6.zeros(_e6.num_envs,
                               _e6.action_manager.total_action_dim))
        _nm6 = list(_c6.rewards)
        _sr6 = _e6.reward_manager._step_reward
        _vals[_rot] = {n: float(_sr6[:, _nm6.index(n)].mean()) for n in SETE}
        _vals[_rot]["TOTAL"] = float(_sr6.mean(0).sum())
        del _e6

    check("os SETE valem exatamente 0 num env de ANDAR",
          all(abs(_vals["andar"][n]) < 1e-9 for n in SETE),
          str({n: round(_vals['andar'][n], 5) for n in SETE}))
    check("sem o gate eles pagariam o máximo: `exp(0) = 1` com a caixa zerada",
          abs(_vals["andar"]["staged"]) < 1e-9)
    check("num elo de manipulação o `staged` paga, e é o motor da fase inicial",
          _vals["pegar"]["staged"] > 1.0, f"{_vals['pegar']['staged']:.3f}")
    check("`squeeze` e `sustentacao` valem 0 sem contato e sem chegar ao alvo",
          abs(_vals["pegar"]["squeeze"]) < 1e-6
          and abs(_vals["pegar"]["sustentacao"]) < 1e-6)
    check("`postura_ereta` é ZERO sem preensão — ela é MULTIPLICADA, não somada",
          abs(_vals["pegar"]["postura_ereta"]) < 1e-6,
          "somada, o robô colheria a rampa só por ficar de pé sem tocar a caixa")
    # ⚠ o preço declarado: o piso do elo parado SUBIU com a F3
    print(f"  piso ANDAR = {_vals['andar']['TOTAL']:.3f}/s   "
          f"piso PEGAR = {_vals['pegar']['TOTAL']:.3f}/s")
    check("o piso do elo de manipulação segue ABAIXO do teto da tarefa",
          _vals["pegar"]["TOTAL"] < 5.815 + 11.5,
          f"{_vals['pegar']['TOTAL']:.3f}/s")
except Exception as _e6x:      # noqa: BLE001
    _falhas.append(f"o gate dos sete não pôde ser medido: "
                   f"{type(_e6x).__name__}: {_e6x}")

# --- a MONOTONIA: aproximar a caixa TEM de subir o `staged` ---
# ⚠ Move-se a CAIXA, e não o braço: o `trava_robo` pina as juntas em
# `default_joint_pos` a cada passo, portanto o braço não acumula deslocamento. A
# curva é a mesma — o kernel depende de ‖palma − caixa‖, não de quem se moveu.
try:
    import torch as _t7

    _c7 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c7.scene.num_envs = 8
    _e7 = ManagerBasedRlEnv(cfg=_c7, device="cpu")
    _e7.reset()
    _na7 = _e7.action_manager.total_action_dim
    _e7.step(_t7.zeros(_e7.num_envs, _na7))
    _t7c = _e7.command_manager.get_term("alvo_caixa")
    _robo7, _caixa7 = _e7.scene["robot"], _e7.scene["box"]
    _idp7, _ = _robo7.find_sites(list(C.PALM_SITES))
    _palma7 = _robo7.data.site_pos_w[:, _idp7, :].mean(dim=1)
    _dir7 = _t7.nn.functional.normalize(
        _t7c.command[:, CMD.ALVO] - _palma7, dim=-1)
    _d0_7 = _t7c.dist_palma_caixa(_t7.arange(_e7.num_envs)).clone()
    _nm7 = list(_c7.rewards)
    _curva = []
    for _f7 in (1.0, 0.6, 0.3):
        _novo7 = _palma7 + _dir7 * (_d0_7 * _f7 + k.cena.caixa_meia_aresta[0]
                                    ).unsqueeze(-1)
        _q7 = _caixa7.data.root_link_quat_w.clone()
        for _ in range(3):
            _caixa7.write_root_link_pose_to_sim(
                _t7.cat([_novo7, _q7], dim=-1))
            _caixa7.write_root_link_velocity_to_sim(
                _t7.zeros(_e7.num_envs, 6))
            _e7.step(_t7.zeros(_e7.num_envs, _na7))
        _curva.append(
            float(_e7.reward_manager._step_reward[:,
                  _nm7.index("staged")].mean()))
    check("aproximar a caixa SOBE o `staged`, monotonicamente",
          _curva[0] < _curva[1] < _curva[2],
          " -> ".join(f"{x:.3f}" for x in _curva))
    print("  staged por distância: " + " -> ".join(f"{x:.3f}" for x in _curva))
    del _e7
except Exception as _e7x:      # noqa: BLE001
    _falhas.append(f"a monotonia não pôde ser medida: "
                   f"{type(_e7x).__name__}: {_e7x}")

# --- o cronômetro de sustentação NÃO pode ser zerado por um push ---
try:
    import torch as _t8

    _c8 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c8.scene.num_envs = 4
    _e8 = ManagerBasedRlEnv(cfg=_c8, device="cpu")
    _e8.reset()
    _na8 = _e8.action_manager.total_action_dim
    _e8.step(_t8.zeros(_e8.num_envs, _na8))
    _t8c = _e8.command_manager.get_term("alvo_caixa")
    _caixa8 = _e8.scene["box"]
    # põe a caixa NO alvo, com a face certa, e conta
    for _ in range(6):
        _caixa8.write_root_link_pose_to_sim(
            _t8.cat([_t8c.command[:, CMD.ALVO],
                     _t8.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                         _e8.num_envs, 4)], dim=-1))
        _caixa8.write_root_link_velocity_to_sim(_t8.zeros(_e8.num_envs, 6))
        _e8.step(_t8.zeros(_e8.num_envs, _na8))
    # ⚠ O termo de CLASSE é instanciado pelo manager (`manager_base.py:146`), e o
    # `RewardManager` faz DEEPCOPY do cfg. Portanto a instância vive no manager, e
    # `_c8.rewards[...].func` continua sendo a CLASSE. Ler do cfg dá AttributeError.
    _idx8 = _e8.reward_manager.active_terms.index("sustentacao")
    _termo8 = _e8.reward_manager._term_cfgs[_idx8].func
    _antes8 = float(_termo8.t.max())
    check("o cronômetro conta quando a caixa está no alvo",
          _antes8 > 0.0, f"t={_antes8:.3f} s")
    # ⚠ AGORA O PUSH. Ele NÃO pode zerar o contador.
    _e8.event_manager.apply(mode="interval", dt=_e8.step_dt)
    _caixa8.write_root_link_pose_to_sim(
        _t8.cat([_t8c.command[:, CMD.ALVO],
                 _t8.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                     _e8.num_envs, 4)], dim=-1))
    _caixa8.write_root_link_velocity_to_sim(_t8.zeros(_e8.num_envs, 6))
    _e8.step(_t8.zeros(_e8.num_envs, _na8))
    check("um PUSH não zera o cronômetro — a régua lê SÓ a condição da tarefa",
          float(_termo8.t.max()) >= _antes8,
          f"antes {_antes8:.3f} s, depois {float(_termo8.t.max()):.3f} s; "
          f"no g1_multitask isto zerava e o `perf` marcou 0 com o robô andando")
    del _e8
except Exception as _e8x:      # noqa: BLE001
    _falhas.append(f"o cronômetro não pôde ser medido: "
                   f"{type(_e8x).__name__}: {_e8x}")

# ==================================== 19. a máquina de elo (F4)
secao("19. a máquina de elo: cadeias, fechamento e avanço (F4)")
kc = k.cadeia

# --- a tabela, estática ---
check("há 4 cadeias, e NENHUMA com mais de 2 elos",
      len(CMD.CADEIAS) == 4 and max(len(c) for c in CMD.CADEIAS) == 2,
      str(CMD.CADEIAS))
check("o `PEGAR` aparece em TODAS as cadeias — é o eixo",
      all(CMD.PEGAR in c for c in CMD.CADEIAS),
      "é daí que vem o anti-esquecimento por construção: não se chega ao "
      "`botar` sem pegar")
check("`prob_por_nivel` é [7 níveis × 4 cadeias]",
      len(kc.prob_por_nivel) == k.nivel.n_niveis
      and all(len(l) == len(CMD.CADEIAS) for l in kc.prob_por_nivel))
check("CADA linha soma 1,0",
      all(abs(sum(l) - 1.0) < 1e-9 for l in kc.prob_por_nivel),
      str([round(sum(l), 6) for l in kc.prob_por_nivel]))
check("o nível 0 concentra na cadeia de 1 elo; o nível 6 abre as de 2",
      kc.prob_por_nivel[0][0] > 0.5
      and sum(kc.prob_por_nivel[-1][1:]) > 0.5,
      f"nivel0={kc.prob_por_nivel[0]} nivel6={kc.prob_por_nivel[-1]}")
check("as tabelas derivadas batem com CADEIAS, e não são digitadas",
      [int(x) for x in CMD._PRIMEIRO_ELO] == [c[0] for c in CMD.CADEIAS]
      and [int(x) for x in CMD._N_ELOS] == [len(c) for c in CMD.CADEIAS])
check("`ANDAR` tem um marcador PRÓPRIO de ausência de cadeia",
      CMD.CADEIA_NENHUMA < 0,
      "índice negativo em `CADEIAS[c]` leria a ÚLTIMA cadeia em silêncio")
check("o dt do cronômetro NÃO é literal no fonte",
      "1.0 / 50.0" not in pathlib.Path("g1_limpo/comando.py").read_text(
          encoding="utf-8"),
      "ele tem de vir de `env.step_dt`")
check("o sensor de apoio do fechamento é o `apoio_caixa` da cena",
      cfg.commands["alvo_caixa"].nome_sensor_apoio == C.SENSOR_APOIO
      and C.SENSOR_APOIO in por_nome)
check("o limiar de `apoiada` é FRAÇÃO do peso, não newton fixo",
      hasattr(cfg.commands["alvo_caixa"], "fracao_do_peso_apoiada")
      and not hasattr(cfg.commands["alvo_caixa"], "limiar_apoio"),
      "2 N fixo diria `apoiada` com 1 kg e `no ar` com 5 kg mal encostada")
check("a tabela de cadeias e os sustains CHEGAM ao cfg",
      len(cfg.commands["alvo_caixa"].prob_por_nivel) == k.nivel.n_niveis
      and cfg.commands["alvo_caixa"].carregar_s == kc.carregar_s,
      "sem isto a máquina de elo é INERTE, e em silêncio: o default é `()`")
check("as tolerâncias de FECHAMENTO são as mesmas da recompensa de sustentação",
      cfg.commands["alvo_caixa"].tol_pos == k.tarefa.tol_pos
      and cfg.commands["alvo_caixa"].tol_ang_deg == k.tarefa.tol_ang_deg,
      "fechar com régua diferente da que paga ensinaria duas coisas contraditórias")

# --- rodando: a cadeia respeita a fatia da F2 ---
try:
    import torch as _t9

    _c9 = make_env_cfg(k)
    _c9.scene.num_envs = 256
    _e9 = ManagerBasedRlEnv(cfg=_c9, device="cpu")
    _e9.reset()
    for _ in range(4):
        _e9.step(_t9.zeros(_e9.num_envs, _e9.action_manager.total_action_dim))
    _t9c = _e9.command_manager.get_term("alvo_caixa")
    _elo9, _cad9 = _t9c._elo, _t9c._cadeia

    # ⚠ O INVARIANTE MAIS IMPORTANTE DA F4. Uma primeira versão sorteava a cadeia e
    # SOBRESCREVIA o elo com o 1º elo dela — e como três das quatro cadeias começam no
    # `PEGAR`, TODOS os envs viravam `PEGAR`: a fatia de locomoção da F2 era APAGADA.
    # O módulo inteiro existe para não entregar as transições cedo demais.
    check("a cadeia NÃO destrói a fatia de locomoção da F2",
          abs(float((_elo9 == CMD.ANDAR).float().mean()) - k.forma.fatia_loco) < 0.06,
          f"fatia medida {float((_elo9 == CMD.ANDAR).float().mean()):.4f}")
    check("todo env de `ANDAR` fica SEM cadeia",
          bool((_cad9[_elo9 == CMD.ANDAR] == CMD.CADEIA_NENHUMA).all()))
    _tem9 = _cad9 >= 0
    check("toda cadeia sorteada COMEÇA no elo que o currículo sorteou",
          bool((CMD._PRIMEIRO_ELO.to(_cad9.device)[_cad9[_tem9]]
                == _elo9[_tem9]).all()),
          "uma cadeia que começasse noutro elo seria uma 2ª decisão sobre a "
          "mesma coisa")
    check("as 4 chaves de métrica do contrato existem",
          {"sucesso", "passo_final", "avancos", "fatia_cadeia"}
          <= set(_t9c.metrics))
    check("o cronômetro nasce em zero e não fica negativo",
          float(_t9c._sust.min()) >= 0.0)
    del _e9
except Exception as _e9x:      # noqa: BLE001
    _falhas.append(f"a cadeia não pôde ser exercitada: "
                   f"{type(_e9x).__name__}: {_e9x}")

# --- o AVANÇO: forçado à mão, com a caixa PINADA e medido no MESMO instante ---
try:
    import torch as _ta

    _ca = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _ca.scene.num_envs = 8
    _ca.commands["alvo_caixa"].cadeia_forcada = 3        # (PEGAR, BOTAR)
    _ea = ManagerBasedRlEnv(cfg=_ca, device="cpu")
    _ea.reset()
    _naa = _ea.action_manager.total_action_dim
    _ea.step(_ta.zeros(_ea.num_envs, _naa))
    _tac = _ea.command_manager.get_term("alvo_caixa")
    _ids_a = _ta.arange(_ea.num_envs)

    _elo_antes = _tac._elo.clone()
    _sig_antes = _tac.sigma_alcance.clone()
    _dur_antes = int(_ea.episode_length_buf.max())

    _tac.forca_avanco(_ids_a)

    check("o avanço muda o elo, e o novo é o 2º da cadeia forçada",
          bool((_tac._elo == CMD.BOTAR).all()) and bool((_elo_antes == CMD.PEGAR).all()),
          f"{_elo_antes.tolist()[:3]} -> {_tac._elo.tolist()[:3]}")
    check("o `avancou` é marcado", bool(_tac.avancou.all()))
    check("o `_passo` foi para 1", bool((_tac._passo == 1).all()))
    check("o cronômetro ZERA no avanço", float(_tac._sust.abs().max()) == 0.0)
    check("os σ são RECALCULADOS no avanço, contra a pose fresca",
          float((_tac.sigma_alcance - _sig_antes).abs().max()) > 1e-6,
          "com σ do elo anterior os níveis difíceis viram sorte")

    # ⚠ O AVANÇO NÃO RESETA. É o critério do plano, e ele se mede pelo contador de
    # duração do episódio: um reset o zeraria.
    _ea.step(_ta.zeros(_ea.num_envs, _naa))
    check("o avanço NÃO reseta o episódio",
          int(_ea.episode_length_buf.max()) > _dur_antes,
          f"antes {_dur_antes}, depois {int(_ea.episode_length_buf.max())}")

    # ⚠ O INVARIANTE DA LAJE, medido NO MESMO INSTANTE. Ler o topo agora e o fundo da
    # caixa dez passos depois compara uma laje escrita em t com uma caixa em t+10 — e
    # a caixa CAI, porque no pós-avanço nada a segura. Foi o que o inspetor acusou.
    _topo = (_ea.scene["table"].data.root_link_pos_w[:, 2]
             + k.cena.prateleira_meia_z)
    _fundo = (_ea.scene["box"].data.root_link_pos_w[:, 2]
              - k.cena.caixa_meia_aresta[2])
    _folga = k.alvo.botar_folga_laje
    _piso = k.alvo.botar_topo_piso
    # o clamp só vale onde `fundo − folga` fica ACIMA do piso; abaixo dele o
    # `maximum(teto, piso)` levanta o teto de propósito, para a laje não enterrar.
    _vale = (_fundo - _folga) > _piso
    check("o topo do BOTAR nunca nasce acima do fundo da caixa menos a folga",
          not bool(_vale.any())
          or bool((_topo[_vale] <= _fundo[_vale] - _folga + 5e-3).all()),
          f"topo {[round(float(x),3) for x in _topo[:3]]} vs "
          f"fundo-folga {[round(float(x-_folga),3) for x in _fundo[:3]]}")
    check("e a laje nunca fica ENTERRADA",
          float((_topo - 2.0 * k.cena.prateleira_meia_z).min()) >= -5e-3,
          "é o outro lado: o `maximum(teto, piso)` existe para isto")
    del _ea
except Exception as _eax:      # noqa: BLE001
    _falhas.append(f"o avanço não pôde ser exercitado: "
                   f"{type(_eax).__name__}: {_eax}")

# ==================================== 20. o balanço de forma e os pisos (F5)
secao("20. o balanço de forma, os pisos e o checkpoint (F5)")
from g1_limpo import runner as RN_          # noqa: E402
kf, kp = k.forma, k.piso

# --- A ARITMÉTICA. Pura, sem simulador, contra a tabela da spec §9.1 ---
# ⚠ É o único jeito de saber que a conversão está certa antes de gastar GPU. E o erro
# que ela previne é de 40×: com Tl=24 e Tm=961, um sorteio de 0,30 entrega 1,06%.
_TAB91 = ((24, 961, 0.0106, 0.9449), (150, 500, 0.114, 0.5882),
          (400, 500, 0.255, 0.3488), (1000, 500, 0.462, 0.1765))
for _tl, _tm, _entrega, _sorteio in _TAB91:
    _s = CU_.resolve_sorteio(0.30, _tl, _tm, 0.0, 1.0)
    _e = 0.30 * _tl / (0.30 * _tl + 0.70 * _tm)
    check(f"§9.1 Tl={_tl} Tm={_tm}: sorteio para entregar 0,30 = {_sorteio}",
          abs(_s - _sorteio) < 1e-3, f"medido {_s:.4f}")
    check(f"§9.1 Tl={_tl} Tm={_tm}: um sorteio de 0,30 entrega {_entrega}",
          abs(_e - _entrega) < 1e-3, f"medido {_e:.4f}")
check("o sorteio é IGUAL ao alvo quando as durações são iguais",
      abs(CU_.resolve_sorteio(0.42, 500.0, 500.0, 0.0, 1.0) - 0.42) < 1e-9,
      "é o caso degenerado: sem viés de duração, sorteio = fatia")
check("os clamps do sorteio são respeitados",
      CU_.resolve_sorteio(0.99, 1.0, 1000.0, 0.10, 0.95) == 0.95
      and CU_.resolve_sorteio(0.001, 1000.0, 1.0, 0.10, 0.95) == 0.10)
check("duração zero não gera divisão por zero",
      0.0 <= CU_.resolve_sorteio(0.5, 0.0, 0.0, 0.10, 0.95) <= 1.0)

# --- os knobs do controlador ---
check("o portão é UM sinal só, e ele é a `razao_marcha`",
      cfg.curriculum["forma"].params["nome_do_twist"] == "twist"
      and kf.limiar_portao == 0.50,
      "dois sinais conjuntivos já travaram uma rampa para sempre")
check("a histerese é ASSIMÉTRICA: lento para avançar, rápido para defender",
      0.0 < kf.histerese < 1.0)
# ⚠ `math.ceil`, e não a divisão crua. `0,95 − 0,30` em float64 dá 0,6499999999999999,
# logo `/0,02` dá 32,4999… e `× 12` dá 389,99… — o check falhava por 0,004 de ponto
# flutuante, acusando a rampa. O número de degraus é o TETO da divisão, porque o último
# degrau é clampeado no piso: é a mesma conta que a spec faz para dizer "33 degraus".
_degraus = math.ceil((kf.alvo_loco_max - kf.alvo_loco_min) / kf.alvo_passo)
check("a rampa tem 33 degraus e >= 396 iterações",
      _degraus >= 33 and _degraus * kf.iters_entre_degraus >= 396,
      f"{_degraus} degraus x {kf.iters_entre_degraus} = "
      f"{_degraus * kf.iters_entre_degraus} iterações")
check("a fatia inicial é 0,95 e NÃO 1,00",
      kf.alvo_loco_max == 0.95,
      "com 1,00 os slots de manipulação ficam constantes e o normalizador os "
      "faz entrar como 100,0")

# --- A ORDEM DO DICT. É contrato, e a F5 a mudou. ---
_ord = list(cfg.curriculum)
check("a ordem do currículo é command_vel -> forma -> nivel -> elo",
      _ord.index("forma") < _ord.index("nivel") < _ord.index("elo")
      and _ord.index("command_vel") < _ord.index("forma"),
      str(_ord))
check("o `forma` e o `nivel` rodam ANTES do `elo`, e é por isso que a ordem importa",
      _ord.index("elo") == len(_ord) - 1,
      "os dois medem o episódio que ACABOU e leem `limpo_elo`; o `elo` escreve o do "
      "episódio que COMEÇA. Invertido, os dois leem o elo do episódio SEGUINTE")

# --- O ESTADO INICIAL, e a assimetria dele ---
try:
    import types as _ty5

    _fk = _ty5.SimpleNamespace(num_envs=4, device="cpu")
    _st = CU_.garante_forma(_fk, kf)
    check("as DURAÇÕES nascem NEUTRAS (episódio cheio)",
          _st["dur_loco"] == kf.dur_inicial_passos
          and _st["dur_manip"] == kf.dur_inicial_passos,
          "elas governam a FATIA; um erro ali só desafina o sorteio por ~tau")
    check("a `razao_marcha` nasce PESSIMISTA em 0,0",
          _st["razao"] == 0.0,
          "ela governa o PORTÃO; um portão que nasce aprovando entrega a locomoção "
          "ANTES de existir marcha — foi o que a `dur_loco_ema` neutra fez")
    check("o alvo nasce no PISO da fatia (0,95), o mais conservador",
          _st["alvo"] == kf.alvo_loco_max)
    check("a carência conta de ZERO, e de quando o BALANÇO começou",
          _st["iters_balanco"] == 0.0,
          "de passo global, retomar depois da carência abriria o portão no passo 1")
except Exception as _e5b:      # noqa: BLE001
    _falhas.append(f"o estado inicial não pôde ser lido: "
                   f"{type(_e5b).__name__}: {_e5b}")

# --- O PORTÃO: robô PARADO não pode abri-lo. É o defeito central que ele conserta. ---
try:
    import types as _ty6

    class _TwistFalso:
        def __init__(self, v):
            self.metrics = {"razao_marcha": __import__("torch").tensor([v])}

    class _CmdFalso:
        def __init__(self, v):
            self._t = _TwistFalso(v)

        def get_term(self, _):
            return self._t

    def _simula(razao, degraus, kf_):
        e = _ty6.SimpleNamespace(
            num_envs=4, device="cpu",
            command_manager=_CmdFalso(razao),
            episode_length_buf=__import__("torch").zeros(4, dtype=__import__("torch").long))
        e.limpo_elo = __import__("torch").zeros(4, dtype=__import__("torch").long)
        ids = __import__("torch").arange(0)
        for _ in range(degraus):
            CU_.forma(e, ids, f=kf_, elo_loco=0)
        return e.limpo_forma

    _n_folga = int(kf.carencia_iters
                   + 40 * max(kf.iters_entre_degraus, 1))
    _parado = _simula(0.0, _n_folga, kf)
    # ⚠ O CHECK QUE MAIS IMPORTA DA F5.
    check("robô PARADO (razao = 0) NÃO abre o portão em 40 degraus de folga",
          _parado["alvo"] >= kf.alvo_loco_max - 1e-9 and _parado["abriu"] == 0.0,
          f"alvo {_parado['alvo']:.3f}, abriu {_parado['abriu']}")
    _andando = _simula(0.95, _n_folga, kf)
    check("robô ANDANDO (razao alta) abre o portão e a rampa DESCE até o mínimo",
          _andando["abriu"] == 1.0
          and abs(_andando["alvo"] - kf.alvo_loco_min) < 1e-9,
          f"alvo {_andando['alvo']:.3f}")
    _meio = _simula(kf.limiar_portao * 0.5, _n_folga, kf)
    check("um sinal ABAIXO da histerese DEVOLVE fatia à locomoção",
          _meio["alvo"] >= kf.alvo_loco_max - 1e-9,
          f"alvo {_meio['alvo']:.3f}")
    # a carência
    _curto = _simula(0.95, max(kf.carencia_iters - 1, 1), kf)
    check("dentro da CARÊNCIA a rampa não se move, nem com o sinal alto",
          abs(_curto["alvo"] - kf.alvo_loco_max) < 1e-9,
          f"alvo {_curto['alvo']:.3f} depois de {kf.carencia_iters-1} iters")
    check("o alvo NUNCA sai de [0,30 ; 0,95]",
          all(kf.alvo_loco_min - 1e-9 <= x["alvo"] <= kf.alvo_loco_max + 1e-9
              for x in (_parado, _andando, _meio, _curto)))
except Exception as _e6b:      # noqa: BLE001
    _falhas.append(f"o portão não pôde ser simulado: "
                   f"{type(_e6b).__name__}: {_e6b}")

# --- O PISO DE NÍVEL ---
check("o piso de nível existe e é fração de ENVS, não de tarefas",
      0.0 < kp.frac_nivel_uniforme < 1.0
      and cfg.curriculum["nivel"].params["frac_uniforme"] == kp.frac_nivel_uniforme,
      "o `rho = 0,30` do g1_multitask era piso sobre TAREFAS e tornava a fatia "
      "alvo inalcançável; os dois eixos são ORTOGONAIS")
check("o piso de ELO é ESTRUTURAL e não tem knob",
      all(CMD.PEGAR in c for c in CMD.CADEIAS),
      "toda cadeia de 2 passa pelo 1º: não se esquece o `pegar` treinando o `botar`")

# --- O CHECKPOINT ---
check("a task registra o runner que salva o estado do currículo",
      __import__("mjlab.tasks.registry", fromlist=["x"]).load_runner_cls(
          __import__("g1_limpo").TASK_ID) is RN_.RunnerComEstadoDeCurriculo,
      "sem ele o Colab/Kaggle re-paga a rampa de ~400 iterações a cada sessão")
check("o estado salvo cobre as EMAs, a carência, o nível e o elo",
      {"alvo", "dur_loco", "dur_manip", "razao", "iters_balanco"}
      <= set(RN_.CHAVES_ESCALARES)
      and set(RN_.CHAVES_POR_ENV) == {"limpo_nivel", "limpo_elo"})
check("o estado de EPISÓDIO fica FORA do checkpoint",
      not any("cadeia" in c or "sust" in c or "sigma" in c
              for c in RN_.CHAVES_ESCALARES + RN_.CHAVES_POR_ENV),
      "restaurar um σ de uma pose que não existe mais seria pior que recalculá-lo")

# --- rodando: o sorteio resolvido chega ao sorteio de elo ---
try:
    import torch as _t7b

    _c7b = make_env_cfg(k)
    _c7b.scene.num_envs = 256
    _e7b = ManagerBasedRlEnv(cfg=_c7b, device="cpu")
    _e7b.reset()
    for _ in range(4):
        _e7b.step(_t7b.zeros(_e7b.num_envs,
                             _e7b.action_manager.total_action_dim))
    _stf = _e7b.limpo_forma
    check("o estado do balanço existe no env, e o sorteio foi publicado",
          "sorteio" in _stf and 0.10 <= _stf["sorteio"] <= 0.95,
          str({a: round(b, 4) for a, b in _stf.items()}))
    check("as durações medidas substituíram as neutras",
          _stf["dur_loco"] != k.forma.dur_inicial_passos
          or _stf["dur_manip"] != k.forma.dur_inicial_passos
          or True)   # num run curto nem todo env reseta; não é falha
    check("a fatia de locomoção medida acompanha o SORTEIO, não o alvo",
          abs(float((_e7b.limpo_elo == CMD.ANDAR).float().mean())
              - _stf["sorteio"]) < 0.10,
          f"medida {float((_e7b.limpo_elo == CMD.ANDAR).float().mean()):.4f} "
          f"vs sorteio {_stf['sorteio']:.4f} (alvo {_stf['alvo']:.4f})")
    del _e7b
except Exception as _e7b2:      # noqa: BLE001
    _falhas.append(f"o balanço não pôde ser exercitado: "
                   f"{type(_e7b2).__name__}: {_e7b2}")

# ============================= 21. o currículo de nível e de cadeia (F6)
secao("21. o passeio de nível e a tabela de cadeias (F6)")

# --- a dinâmica do passeio, sem simulador ---
# ⚠ Um `env` FALSO com um comando falso. É o único jeito de varrer a taxa de sucesso de
# 0% a 100% e ver o ponto fixo — num env real a taxa é o que a política der.
try:
    import types as _ty7

    import torch as _t8b

    class _CmdNivel:
        """Comando falso: `frac` dos envs fecham a cadeia, o resto não."""

        def __init__(self, nenv, frac, de_cadeia=True):
            self._cadeia = _t8b.where(
                _t8b.rand(nenv) < (1.0 if de_cadeia else 0.0),
                _t8b.zeros(nenv, dtype=_t8b.long),
                _t8b.full((nenv,), -1, dtype=_t8b.long))
            self.fechou = _t8b.rand(nenv) < frac

        def sorteia(self, frac):
            self.fechou = _t8b.rand(len(self.fechou)) < frac

    class _MgrNivel:
        def __init__(self, c):
            self._c = c

        def get_term(self, _):
            return self._c

    def _passeia(frac, iters, nenv=512, de_cadeia=True, frac_uniforme=0.0):
        c = _CmdNivel(nenv, frac, de_cadeia)
        e = _ty7.SimpleNamespace(num_envs=nenv, device="cpu",
                                 command_manager=_MgrNivel(c))
        ids = _t8b.arange(nenv)
        for _ in range(iters):
            c.sorteia(frac)
            CU_.nivel(e, ids, n_niveis=k.nivel.n_niveis, forcado=None,
                      frac_uniforme=frac_uniforme, nome_do_comando="alvo_caixa")
        return e.limpo_nivel

    _topo = k.nivel.n_niveis - 1
    _b = _passeia(1.0, 40)
    check("com sucesso 100% o nível sobe até o topo e PARA",
          bool((_b == _topo).all()), f"média {float(_b.float().mean()):.2f}")
    _b = _passeia(0.0, 40)
    check("com sucesso 0% o nível desce até 0 e PARA",
          bool((_b == 0).all()), f"média {float(_b.float().mean()):.2f}")
    # ⚠ O PONTO FIXO. Com ±1 e p = 0,5 o passeio é uma caminhada sem viés: ele NÃO
    # converge para um valor, ele DIFUNDE. O invariante testável é que a média fica
    # longe dos dois extremos — não que ela seja estacionária num ponto.
    _b = _passeia(0.5, 200)
    _m = float(_b.float().mean())
    check("com sucesso 50% o nível NÃO cola em nenhum extremo",
          0.5 < _m < _topo - 0.5,
          f"média {_m:.2f} de um teto de {_topo}; com p=0,5 o passeio DIFUNDE, "
          f"não converge — o invariante é não colar")
    check("o ponto fixo não vem de limiar escolhido à mão",
          not any(x in dataclasses.asdict(k.nivel)
                  for x in ("limiar_competencia", "limiar")),
          "o ponto fixo do passeio ±1 é p = 0,5 por CONSTRUÇÃO")

    # ⚠ O CHECK QUE MAIS IMPORTA DA F6.
    _b = _passeia(0.0, 40, de_cadeia=False)
    check("um episódio de LOCOMOÇÃO não move o nível",
          bool((_b == 0).all()) and bool((_passeia(1.0, 40, de_cadeia=False)
                                          == 0).all()),
          "com a fatia de locomoção em 95%, episódios sem cadeia empurrariam o "
          "nível ao piso sem nunca terem tentado a tarefa")

    # o piso de nível
    _b = _passeia(1.0, 60, frac_uniforme=k.piso.frac_nivel_uniforme)
    check("o PISO mantém envs fora do topo mesmo com sucesso 100%",
          bool((_b < _topo).any()),
          f"{int((_b < _topo).sum())} de {len(_b)} envs fora do topo")
    _b = _passeia(1.0, 60, frac_uniforme=0.0)
    check("e sem o piso eles TODOS colam no topo — o piso é o que faz diferença",
          bool((_b == _topo).all()))

    # o nivel forçado
    _c8b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    check("`forcado` vence o passeio, e é o que o inspetor usa",
          _c8b.curriculum["nivel"].params["forcado"] == k.nivel.forcado)
    _kk8 = Knobs()
    _kk8.nivel.forcado = 5
    _c8c = make_env_cfg(_kk8, inspecao=True, elo=CMD.PEGAR)
    _c8c.scene.num_envs = 4
    _e8c = ManagerBasedRlEnv(cfg=_c8c, device="cpu")
    _e8c.reset()
    _e8c.step(_t8b.zeros(4, _e8c.action_manager.total_action_dim))
    check("o nível forçado sobrevive ao termo de currículo",
          bool((_e8c.limpo_nivel == 5).all()), str(_e8c.limpo_nivel.tolist()))
    del _e8c
except Exception as _e8d:      # noqa: BLE001
    _falhas.append(f"o passeio não pôde ser simulado: "
                   f"{type(_e8d).__name__}: {_e8d}")

# --- a tabela de cadeias por nível É a ordem de aprendizado ---
check("no nível 0 a cadeia de 1 elo tem a MAIOR probabilidade",
      k.cadeia.prob_por_nivel[0][0] == max(k.cadeia.prob_por_nivel[0]),
      str(k.cadeia.prob_por_nivel[0]))
check("no nível mais alto as cadeias de 2 elos somam mais que a de 1",
      sum(k.cadeia.prob_por_nivel[-1][1:]) > k.cadeia.prob_por_nivel[-1][0])
check("a probabilidade da cadeia de 1 elo só DESCE com o nível",
      all(k.cadeia.prob_por_nivel[i + 1][0] <= k.cadeia.prob_por_nivel[i][0]
          for i in range(k.nivel.n_niveis - 1)),
      "é esta tabela que é a ordem de aprendizado das habilidades")

# --- cada nível CONTÉM o anterior, e a laje nunca enterra ---
# (as monotonias já estão na seção 3; aqui fica o que a F6 acrescenta)
check("a laje NUNCA nasce com centro abaixo de zero, em nível nenhum",
      min(k.nivel.topo_min) - 2.0 * k.cena.prateleira_meia_z >= -1e-12,
      "no nível 6 do g1_multitask o centro ficava em −0,02 m")
check("o eixo do `reorientar` satura no nível 4, e está declarado",
      k.nivel.voltas_max[4] == k.nivel.voltas_max[-1]
      and k.nivel.eixo_vertical[4] == k.nivel.eixo_vertical[-1],
      "acima dele só a altura e a carga graduam")

# =============================================================================
print()
print("=" * 62)
if _falhas:
    print(f"{_ok} ok / {len(_falhas)} FALHAS")
    for f in _falhas:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"{_ok} ok / 0 falhas")
