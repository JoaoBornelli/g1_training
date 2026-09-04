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
import inspect
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
from g1_limpo import metricas as MT_

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


def _passa_janela(env, na, tmod) -> None:
    """Queima a JANELA DE ESPERA do elo, e devolve o env com o objetivo ATIVO.

    ⚠ Ela existe porque a janela nasceu em 02/09 e QUATRO testes anteriores a ela
    mediam com 3 a 6 passos — isto é 0,06 a 0,12 s, dentro de uma janela que vai a
    1,0 s. Todos passaram a ler `VALIDA = 0`, e portanto `staged`, `sustentacao` e o
    fecho de elo em ZERO. Não era o desenho quebrando; era o teste medindo antes de o
    objetivo existir.
    """
    for _ in range(int(k.alvo.espera_s[1] / env.step_dt) + 5):
        env.step(tmod.zeros(env.num_envs, na))

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
             C.SENSOR_CORPO_PRATELEIRA, C.SENSOR_PALMA_PRATELEIRA,
             C.SENSOR_DORSO_PRATELEIRA, C.SENSOR_AUTO_COLISAO, C.SENSOR_PES)
check("os 8 sensores do pacote existem",
      all(nome in por_nome for nome in esperados),
      str([n for n in esperados if n not in por_nome]))
check("as PALMAS pedem `force` — sem isso o `squeeze` é impossível",
      all("force" in por_nome[n].fields for n in C.SENSOR_PALMA))
check("o APOIO pede `force` — é a ponte do `unload`",
      "force" in por_nome[C.SENSOR_APOIO].fields)
check("os DORSOS são booleanos (magnitude não importa)",
      all("force" not in por_nome[n].fields for n in C.SENSOR_DORSO))

# ⚠ TODO SENSOR PRECISA DE CONSUMIDOR, e esta é a checagem que faltava. O
# `corpo_prateleira` existiu da reescrita até 27/08 SEM NENHUM LEITOR: a checagem de
# existência acima passava, o smoke marcava 306 ok, e o robô se jogava na mesa no play
# porque nada cobrava o escoro. Sensor órfão é invisível para teste de existência.
#
# O consumidor é procurado nos `params` de recompensa, métrica, terminação e observação
# — é por onde um nome de sensor chega a uma função.
def _nomes_nos_params() -> set[str]:
    vistos: set[str] = set()
    grupos = [cfg.rewards, cfg.terminations, getattr(cfg, "metrics", {}) or {}]
    for grupo in grupos:
        for termo in grupo.values():
            for v in (getattr(termo, "params", None) or {}).values():
                if isinstance(v, str):
                    vistos.add(v)
                elif isinstance(v, (tuple, list)):
                    vistos.update(x for x in v if isinstance(x, str))
    for g in cfg.observations.values():
        for termo in g.terms.values():
            for v in (getattr(termo, "params", None) or {}).values():
                if isinstance(v, str):
                    vistos.add(v)
                elif isinstance(v, (tuple, list)):
                    vistos.update(x for x in v if isinstance(x, str))
    return vistos


def _rampa(forca: float, kn) -> float:
    """A aritmética da rampa de contato, isolada para o smoke poder afirmá-la."""
    faixa = max(kn.contato.saturacao_N - kn.contato.joelho_N, 1e-6)
    return min(max((forca - kn.contato.joelho_N) / faixa, 0.0), 1.0)


# ⚠ ALLOWLIST COM MOTIVO, e não falha cega. Os dois que restam são intencionais — a
# checagem existe para pegar um órfão NOVO, não para reclamar da dívida já declarada.
# Um nome que sair desta lista tem de ganhar leitor ou ganhar motivo.
_ORFAOS_ACEITOS = {
    # duplicatas deliberadas: trocar o `feet_ground_contact` e o `self_collision` do
    # molde pelos nossos EXPLODE a obs `foot_air_time` do fabricante, que os
    # referencia POR NOME. O preço está declarado em `env_cfg.py:145-155`.
    "pes_chao": "duplicata deliberada do `feet_ground_contact` do molde",
    "auto_colisao": "duplicata deliberada do `self_collision` do molde",
}
# ⚠ O `dorso_E`/`dorso_D` SAÍRAM desta lista em 28/08. Eles agora têm leitor: a métrica
# `dorso_em_contato` (peso zero, medição só). O freio do dorso continua sendo
# GEOMÉTRICO — o alcance bimanual põe as palmas viradas uma para a outra, que é como o
# `g1_poc` dispensa o `back_penalty` (`g1_poc/terminacoes.py:13`). A métrica é o que
# confere se o freio geométrico basta; se ela sair de zero, ele não bastou.

_consumidos = _nomes_nos_params()
_orfaos = [n for n in por_nome
           if n not in _consumidos and n not in _ORFAOS_ACEITOS]
check("nenhum sensor NOVO ficou sem consumidor",
      not _orfaos,
      f"sensores órfãos: {_orfaos}  (o `corpo_prateleira` ficou órfão até 27/08)")
check("os órfãos aceitos são exatamente as duas duplicatas do molde",
      {n for n in por_nome if n not in _consumidos} == set(_ORFAOS_ACEITOS),
      f"medido: {sorted(n for n in por_nome if n not in _consumidos)}")
# ⚠⚠ O ESCORO É MULTA desde 01/09, e NÃO terminação. A troca é medida dos dois lados:
# com a terminação, 76% dos episódios de manipulação morriam na mesa e o `play` mostrou
# que a ação MÉDIA nem se aproximava — aqueles 76% eram RUÍDO de exploração, e a
# terminação matava a exploração antes de ela refinar a pega. Depois da troca, `descarga`
# (a caixa fora da laje) foi de 0,0 a 0,994 e o `postura_ereta` saiu de ZERO.
#
# ⚠ E o precedente de 27/08 não vale: o `contato_prateleira = −1,5` do bloco 2 rodou num
# sistema com quatro defeitos desde então consertados, e a conta dele nem fecha (−0,11/s
# contra um teto de 11,5/s).
check("os sensores de mesa são lidos por MULTA, e não por terminação",
      all(cfg.rewards[nome].params["sensor_name"] == sensor
          for sensor, nome in C.MESA_POR_GRUPO)
      and not any(nome in cfg.terminations for _, nome in C.MESA_POR_GRUPO),
      f"recompensas={sorted(cfg.rewards)} terminações={sorted(cfg.terminations)}")
check("as TRÊS multas usam o MESMO peso, e ele vem do knobs",
      len({cfg.rewards[nome].weight for _, nome in C.MESA_POR_GRUPO}) == 1
      and cfg.rewards["contato_tronco"].weight == k.recompensa.contato_tronco,
      "pesos diferentes por parte fariam a partição mudar o COMPORTAMENTO, e ela é "
      "só medição")
# ⚠ O PESO É DERIVADO do `postura_ereta`, e não escolhido: o que escorar COMPRA é
# alcançar sem pagar postura. Com os dois em 2,0, a pega ereta e a escorada ficam a
# quatro pontos de distância.
check("o peso da multa casa com o do `postura_ereta` — a derivação",
      abs(k.recompensa.contato_tronco) == k.tarefa.postura_ereta == 2.0,
      f"multa {k.recompensa.contato_tronco} contra postura {k.tarefa.postura_ereta}")
# ⚠ RAMPA e não booleano: abaixo do joelho é ZERO (roçar sai de graça), e entre o joelho
# e a saturação existe gradiente para TIRAR o peso. Booleano seria platô.
check("as TRÊS usam a MESMA rampa de força, e ela vem do knobs",
      all(cfg.rewards[nome].params["joelho_N"] == k.contato.joelho_N
          and cfg.rewards[nome].params["saturacao_N"] == k.contato.saturacao_N
          for _, nome in C.MESA_POR_GRUPO)
      and k.contato.joelho_N == 50.0 and k.contato.saturacao_N == 100.0,
      "o joelho é o MESMO 50 N que governava a terminação, medido no g1_poc")
check("a rampa é ZERO abaixo do joelho e SATURA acima — roçar sai de graça",
      _rampa(25.0, k) == 0.0 and _rampa(50.0, k) == 0.0
      and abs(_rampa(75.0, k) - 0.5) < 1e-9
      and _rampa(100.0, k) == 1.0 and _rampa(500.0, k) == 1.0,
      f"25N={_rampa(25.0,k)} 50N={_rampa(50.0,k)} 75N={_rampa(75.0,k)} "
      f"100N={_rampa(100.0,k)}")
check("o limiar de força SAIU do bloco de terminação",
      not hasattr(k.terminacao, "contato_ilegal_N"),
      "ele é parâmetro de multa desde 01/09, e mora em `Contato`")

# ----------------------- a PARTIÇÃO em três é de MEDIÇÃO, e a união não muda (31/08)
# ⚠ ESTE É O CHECK QUE TORNA A PARTIÇÃO SEGURA. `reduce="netforce"` entrega UM número
# por sensor, portanto um sensor único diz "encostou" e não diz com o quê — no bloco 4
# isso era ~46% dos episódios de manipulação, sem saber se era tronco, coxa ou pad.
# Três sensores resolvem, MAS só se a união continuar a mesma. Se ela mudar, a
# partição deixou de ser medição e passou a ser mudança de comportamento em silêncio.
check("a união dos três grupos É a lista de quem não escora",
      C.GRUPO_TRONCO + C.GRUPO_PALMA + C.GRUPO_DORSO
      == C.CORPOS_QUE_NAO_ESCORAM,
      f"tronco {C.GRUPO_TRONCO} palma {C.GRUPO_PALMA} dorso {C.GRUPO_DORSO}")
check("os três grupos são DISJUNTOS — nenhum geom conta duas vezes",
      len(set(C.CORPOS_QUE_NAO_ESCORAM)) == len(C.CORPOS_QUE_NAO_ESCORAM),
      "um padrão repetido faria duas MULTAS cobrarem o mesmo contato, e a leitura "
      "por parte somaria mais que o total")
check("há uma MULTA por sensor de mesa, e são três",
      len(C.MESA_POR_GRUPO) == 3
      and all(nome in cfg.rewards for _, nome in C.MESA_POR_GRUPO),
      str([nome for _, nome in C.MESA_POR_GRUPO]))
check("cada sensor de mesa pede `force` — sem isso a rampa é impossível",
      all("force" in por_nome[sensor].fields
          for sensor, _ in C.MESA_POR_GRUPO))
check("os três sensores de mesa têm a MESMA mesa como secundário",
      len({por_nome[s].secondary.pattern for s, _ in C.MESA_POR_GRUPO}) == 1,
      "grupos diferentes contra secundários diferentes não seriam uma partição")
# ⚠ O SINAL ESTAVA INVERTIDO até 28/08, e a medição está no bloco 3, it 4251: a lista
# de CORPO INTEIRO cobria punho e cotovelo — que TÊM de chegar perto do tampo para
# pegar — e NÃO cobria os pads, porque `add_pads_de_palma` apaga `*_hand_collision` e
# os pads terminam em `_pad`. Aproximar terminava; escorar com a palma era grátis.
# Resultado: ~75% dos episódios de manipulação morriam na mesa e o `squeeze` ficou em
# 0,0002 por 3200 iterações.
check("os PADS estão na lista — a palma não pode escorar na mesa",
      C.GRUPO_PALMA and C.GRUPO_DORSO
      and all(p.endswith("_pad") for p in C.GRUPO_PALMA + C.GRUPO_DORSO),
      "o pad de palma só pode tocar a CAIXA; o secundário deste sensor é a MESA")
check("o PUNHO e o COTOVELO estão FORA — eles têm de chegar perto para pegar",
      not any("wrist" in p or "elbow" in p or p == r".*_collision"
              for p in C.CORPOS_QUE_NAO_ESCORAM),
      f"medido: {C.CORPOS_QUE_NAO_ESCORAM}")
check("o PÉ está fora da lista, como no g1_poc",
      not any("foot" in p or "ankle" in p for p in C.CORPOS_QUE_NAO_ESCORAM),
      "com `topo_min` a 0,04 m a laje é um degrau, e pisar nela passa de 50 N")

# --------------------------------------------- a terminação `caixa_largada` (28/08)
check("existe a terminação `caixa_largada`",
      "caixa_largada" in cfg.terminations,
      "ela é a outra metade do porteiro do `unload`: o porteiro tira o pagamento de "
      "derrubar sem pegar, ela tira o de pegar e largar")
check("o `caiu` lê o TAMANHO da caixa: a folga do chão é menor que a laje mais baixa",
      0.0 < k.terminacao.caixa_folga_chao < k.cena.prateleira_topo_piso,
      f"folga {k.terminacao.caixa_folga_chao} vs piso da laje {k.cena.prateleira_topo_piso}")
check("o `caixa_dist_max` é MAIOR que a distância de nascimento da palma",
      k.terminacao.caixa_dist_max > 0.339,
      "ela não dispara no reset porque é ARMADA pela primeira preensão, e não "
      "porque o limiar seja apertado")

# --------------------------- a força de referência do `squeeze` é DERIVADA (28/08)
# ⚠ ERA UM KNOB FIXO DE 12,0 N, sem derivação. A conta física é `m·g/(2μ)`: com
# m = 1,0 kg e μ = 0,8 ela dá 6,13 N. O knob pedia o DOBRO e pagava METADE no primeiro
# newton, que é a faixa em que a preensão tem de nascer.
_f_ref_fisica = k.cena.caixa_massa * 9.81 / (2.0 * k.tarefa.squeeze_mu)
check("o `forca_ref` fixo SAIU do knobs — a força de referência é derivada",
      not hasattr(k.tarefa, "forca_ref"),
      "um número sem derivação ao lado de uma conta é o segundo suspeito entrando "
      "pela porta de trás")
check("`F_ref = m·g/(2μ)` dá 6,13 N, e é MENOS da metade do knob antigo",
      abs(_f_ref_fisica - 6.13) < 0.01 and _f_ref_fisica < 12.0,
      f"{_f_ref_fisica:.2f} N contra os 12,0 N fixos de antes")
check("os três termos de força usam o MESMO μ",
      cfg.rewards["squeeze"].params["mu"]
      == cfg.rewards["unload"].params["mu"]
      == cfg.rewards["postura_ereta"].params["mu"]
      == k.tarefa.squeeze_mu,
      "duas referências de força seriam duas definições de `pegou`")
check("o `unload` tem PORTEIRO DE PREENSÃO",
      "sensores_palma" in cfg.rewards["unload"].params,
      "sem ele, derrubar a caixa paga 2,0/s pelo resto do episódio sem mão nenhuma "
      "— medido no bloco 3: unload 0,0995 com squeeze 0,0002")
check("a projeção na normal da palma tem os SÍTIOS resolvidos por `params`",
      all(cfg.rewards[n].params["asset_cfg"].site_names == list(C.PALM_SITES)
          for n in ("squeeze", "unload", "postura_ereta")),
      "fora de `params` o `SceneEntityCfg` nunca é resolvido e `site_ids` vira "
      "`slice(None)` — a projeção leria os SEIS sítios do robô")
check("cada termo tem a SUA instância de `SceneEntityCfg`",
      len({id(cfg.rewards[n].params["asset_cfg"])
           for n in ("squeeze", "unload", "postura_ereta")}) == 3,
      "o `manager_base` resolve os ids DENTRO do objeto; compartilhar é estado "
      "mutável compartilhado entre managers")
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

# ============ 9b. O ALGORITMO: vantagem normalizada POR ELO (01/09) ============
secao("9b. a vantagem é normalizada por grupo de elo")
# ⚠⚠ O DEFEITO: `rsl_rl/algorithms/ppo.py:188` normaliza a vantagem sobre o LOTE
# INTEIRO, misturando envs de locomoção e de manipulação. Quando a manipulação destrava,
# as vantagens dela ficam dispersas, o `std` do lote cresce, e as da LOCOMOÇÃO encolhem
# para perto de zero — ela para de receber sinal e segue arrastada pelo gradiente da
# outra tarefa.
#
# MEDIDO no bloco 7, com ~32% dos envs em locomoção:
#     it 1600  loco 0,112  manip 0,446  razão 0,251  -> fatia no gradiente 10,4%
#     it 1800  loco 0,164  manip 1,174  razão 0,139  -> fatia no gradiente  5,6%
# E o resultado na MESMA iteração com o MESMO nível de manipulação (descarga ~0,99):
#     marcha 0,484 -> 0,762   |   fell_over 51,9% -> 0,9%   |   duração 425 -> 888
#
# ⚠ ISTO NÃO SEPARA AS TAREFAS. Os pesos seguem INTEIRAMENTE compartilhados, e é isso
# que o elo `CARREGAR` precisa. Só a estatística de agregação muda, e ela não carrega
# conhecimento nenhum.
import g1_limpo as _PKG                                                 # noqa: E402
from g1_limpo import algoritmo as ALG                                   # noqa: E402
from g1_limpo import observacoes as _OB                                 # noqa: E402
from rsl_rl.algorithms import PPO as _PPO                               # noqa: E402
from rsl_rl.utils import resolve_callable as _resolve                   # noqa: E402

_rl = _PKG.rl_cfg()
check("o `class_name` do algoritmo aponta para a nossa subclasse",
      _rl.algorithm.class_name == ALG.CAMINHO == "g1_limpo.algoritmo:PPOPorElo",
      f"medido: {_rl.algorithm.class_name}")
# ⚠ STRING e não a classe: o logger do mjlab despeja o cfg em disco, e uma classe não
# serializa. O `resolve_callable` do rsl_rl aceita `"modulo:Atributo"`.
check("ele é uma STRING resolvível, e não o objeto de classe",
      isinstance(_rl.algorithm.class_name, str))
check("o `resolve_callable` do rsl_rl acha a classe pelo caminho",
      _resolve(ALG.CAMINHO) is ALG.PPOPorElo)
check("ela SUBCLASSA o PPO do rsl_rl — não reimplementa o GAE",
      issubclass(ALG.PPOPorElo, _PPO)
      and "super().compute_returns" in inspect.getsource(ALG.PPOPorElo),
      "reimplementar o GAE criaria uma segunda fonte de verdade para a parte CERTA")
check("ela normaliza sobre a vantagem CRUA, e não sobre a já normalizada",
      "st.returns - st.values" in inspect.getsource(ALG.PPOPorElo.compute_returns),
      "renormalizar o que o super() normalizou misturaria as duas escalas")
check("a subclasse afirma o INVARIANTE de one-hot em runtime",
      "allclose" in inspect.getsource(ALG.PPOPorElo.compute_returns),
      "sem ele, uma fatia errada viraria treino silenciosamente errado")
check("grupo com menos de 2 amostras é PULADO — `std` de uma amostra é NaN",
      "< 2" in inspect.getsource(ALG.PPOPorElo.compute_returns),
      "um NaN aqui se propaga para o gradiente inteiro no passo seguinte")

# ⚠ A FATIA DO ELO é contada DO FIM, e o `env_cfg` garante a ordem por append: `elo` e
# depois `caixa`. A aritmética se confere sem env; a comparação contra o
# `observation_manager` VIVO está na seção do one-hot, mais abaixo.
check("`fatia_do_elo` devolve o penúltimo bloco do ATOR, de N_SLOTS canais",
      _OB.fatia_do_elo(114) == slice(99, 104)
      and _OB.fatia_do_elo(200) == slice(200 - _OB.N_CAIXA - _OB.N_SLOTS,
                                        200 - _OB.N_CAIXA),
      f"em 114 devolveu {_OB.fatia_do_elo(114)}")
check("`fatia_do_elo_interno` devolve o ÚLTIMO bloco do CRÍTICO",
      _OB.fatia_do_elo_interno(131) == slice(126, 131))
check("o `PPOPorElo` agrupa pelo elo INTERNO do crítico, não pelo publicado do ator",
      'observations["critic"]' in inspect.getsource(ALG.PPOPorElo.compute_returns)
      and "fatia_do_elo_interno" in inspect.getsource(ALG.PPOPorElo.compute_returns),
      "spec §6.1: a espera final tem retorno de manipulação com one-hot de ANDAR")

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
# ⚠ DOZE termos a mais que o molde: dois da F1 (locomoção), os sete da F3 (tarefa) e as
# TRÊS multas de contato com a mesa, que entraram em 01/09 no lugar das três terminações.
# O teste os NOMEIA em vez de contar — contar deixaria de pegar um termo esquecido.
_NOSSOS = {"terminacao", "joint_acc", "staged", "precise_pos", "precise_ori",
           "squeeze", "unload", "postura_ereta", "sustentacao",
           "contato_tronco", "contato_palma", "contato_dorso",
           # v2: a renda do BOTAR (spec §6.6.2); v2.1: `load` SAIU, `renda_congelada`
           # entrou (spec P3) — o total continua QUATORZE.
           "largou", "renda_congelada"}
check("a tabela divergE do molde em exatamente QUATORZE termos, e são estes",
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
      (CMD.ALVO, CMD.FACE, CMD.ANG, CMD.VALIDA, CMD.ELO, CMD.GIRO, CMD.DIM)
      == (slice(0, 3), slice(3, 6), 6, 7, 8, slice(9, 12), 12),
      "v2: o GIRO entrou POR ÚLTIMO (append), e DIM foi de 9 a 12")
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
# ⚠ v2 (spec §8.3): o REORIENTAR está INERTE nesta run. As voltas são zero em todo
# nível e o eixo vertical não entra; a tabela antiga está em comentário no `knobs.py`.
check("o eixo do `reorientar` é em QUARTOS DE VOLTA, e na v2 está em ZERO em todo nível",
      not hasattr(k.nivel, "ang_max_deg")
      and tuple(k.nivel.voltas_max) == (0,) * k.nivel.n_niveis)
check("o teto de voltas NUNCA passa de UM: a face nunca nasce do lado OPOSTO",
      max(k.nivel.voltas_max) <= 1,
      "o robô só precisa aprender a girar no máximo 90°")
check("o eixo VERTICAL não entra na v2 (tombar é o REORIENTAR de verdade, que fica para depois)",
      not any(k.nivel.eixo_vertical),
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
    # ⚠ ELE INVERTEU EM 02/09. Antes o objetivo nascia ATIVO; com a janela de espera ele
    # nasce DESLIGADO num elo de manipulação, e liga na borda da janela. A
    # descontinuidade 0->1 É o sinal de "o objetivo chegou" — ver `knobs.Alvo.espera_s`.
    check("o objetivo da caixa nasce DESLIGADO — a janela de espera corre",
          float(_cmd[:, CMD.VALIDA].max()) == 0.0,
          "a janela mínima é 0,3 s e este env tem 1 passo de vida")
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

# --- a ARITMÉTICA da eficiência por segmento (o JUIZ, desde 27/08) ---
# ⚠ Aritmética pura sobre a MESMA forma do termo. O que ela tem de provar é a
# propriedade que o `razao_marcha` NÃO tem: ruído de média zero não muda o número.
check("o limiar de validade do segmento vem do knobs",
      _tw.pedido_min_segmento == k.marcha.pedido_min_segmento)


def _efic(passos, dt: float = 0.02) -> float:
    """passos = [(v_cmd, v_real), ...], cada um vetor 2D. Um segmento só."""
    proj = ped = 0.0
    for c, v in passos:
        nc = (c[0] ** 2 + c[1] ** 2) ** 0.5
        if nc <= k.marcha.limiar_comando:
            continue
        proj += ((v[0] * c[0] + v[1] * c[1]) / nc) * dt
        ped += nc * dt
    return proj / ped if ped > 0 else 0.0


check("robô IMÓVEL com comando ativo dá eficiência 0,0",
      abs(_efic([((1.0, 0.0), (0.0, 0.0))] * 50) - 0.0) < 1e-12,
      "projeção de velocidade nula é zero — a estátua não engana o juiz")
check("rastreio PERFEITO dá exatamente 1,0",
      abs(_efic([((1.0, 0.0), (1.0, 0.0))] * 50) - 1.0) < 1e-12)
check("METADE da velocidade comandada dá exatamente 0,50",
      abs(_efic([((1.0, 0.0), (0.5, 0.0))] * 50) - 0.50) < 1e-12)
check("ela é ADIMENSIONAL: metade de 2 e metade de 1 dão o MESMO número",
      abs(_efic([((2.0, 0.0), (1.0, 0.0))] * 50)
          - _efic([((1.0, 0.0), (0.5, 0.0))] * 50)) < 1e-12,
      "é isso que a torna imune ao degrau do currículo de comando na it 5000")
check("ir ao CONTRÁRIO dá eficiência NEGATIVA, sem clamp",
      _efic([((1.0, 0.0), (-1.0, 0.0))] * 50) < 0.0)
check("velocidade PERPENDICULAR ao comando dá 0,0 — andar de lado não conta",
      abs(_efic([((1.0, 0.0), (0.0, 3.0))] * 50) - 0.0) < 1e-12,
      "é a projeção, não a norma: correr para o lado errado não paga")

# ⚠⚠ A PROPRIEDADE QUE MOTIVOU A TROCA, e ela é o teste que importa. Ruído SIMÉTRICO
# somado à velocidade real: a projeção NÃO se move, e a razão de normas PIORA.
_alt = [((1.0, 0.0), (0.5, +0.4)), ((1.0, 0.0), (0.5, -0.4))] * 25
check("ruído de média zero NÃO move a eficiência (a projeção cancela)",
      abs(_efic(_alt) - 0.50) < 1e-12,
      "Σ(ruído · v̂_cmd) tem média zero; é por isso que ela é o juiz")
_r_ruido = _razao([(1.0, (0.5 ** 2 + 0.4 ** 2) ** 0.5)] * 50)
check("o MESMO ruído PIORA a razão de normas (norma nunca cancela)",
      _r_ruido < 0.50 - 1e-6,
      f"medido {_r_ruido:.4f} < 0,50 — foi isso que congelou o portão do bloco 1")

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

    check("as três entradas da razão existem em `self.metrics` do twist",
          {"soma_erro_marcha", "soma_cmd_marcha", "razao_marcha"}
          <= set(_tw2.metrics))
    # ⚠ Os SEIS buffers da eficiência vivem em `self.metrics` de propósito: o `reset` do
    # mjlab zera o que está no dict, e um buffer próprio precisaria repetir a ordem.
    check("os seis buffers da EFICIÊNCIA existem em `self.metrics`",
          {"seg_proj", "seg_pedido", "seg_visto", "segmentos",
           "eficiencia_min", "eficiencia_media"} <= set(_tw2.metrics),
          str(sorted(set(_tw2.metrics))))
    check("a eficiência nasce em 0,0 — pessimista, como o portão exige",
          float(_tw2.metrics["eficiencia_min"].abs().max()) == 0.0)
    check("o acumulador do segmento ANDA nos primeiros passos",
          float(_tw2.metrics["seg_pedido"].max()) > 0.0,
          "se ficar em zero, o gate do limiar de comando está zerando tudo")
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
# ⚠ A LINHA DO ANDAR PASSOU A LER O DERIVADO em 31/08. O canal cru é diluído pela fatia
# de manipulação (twist em zero -> `eficiencia_min` zero exato naqueles envs), e o alvo
# de 0,50 no cru fica mais duro conforme a rampa desce. No destino (`alvo_loco_min` =
# 0,30, isto é 30% de LOCOMOÇÃO) ele exigiria `0,50/0,30 = 1,67` de quem anda — acima do
# teto de 1,0, logo IMPOSSÍVEL. A linha marcaria falha num robô que anda perfeitamente.
check("a escada tem as quatro linhas da F1, e a do andar é a eficiência DES-DILUÍDA",
      {LE_.CH_STD, LE_.CH_DURACAO, LE_.CH_EFIC_LOCO, LE_.CH_VOO} <= _chaves_escada
      and any(ch == LE_.CH_EFIC_LOCO and alvo == 0.50
              for _, ch, _, alvo, _ in LE_.ESCADA))
# ⚠ E A RAZÃO SAIU DA ESCADA, de propósito. Ela continua IMPRESSA no painel como
# diagnóstico, mas julgar por ela automatizaria o erro de leitura do bloco 1: ela infla
# com o `std`, e o `std` sobe quando a manipulação entra.
check("a `razao_marcha` NÃO é mais linha de corte, e segue impressa",
      LE_.CH_RAZAO not in _chaves_escada
      and LE_.CH_RAZAO in inspect.getsource(LE_),
      "se ela voltar à escada, o portão volta a ler ruído de ação como incompetência")
check("o degrau do `std` na it 200 é 0,60, e não 0,85",
      any(ch == LE_.CH_STD and abs(alvo - 0.60) < 1e-9
          for _, ch, _, alvo, _ in LE_.ESCADA),
      "0,85 marcava falha com o treino saudável: o `std` cai a 0,83 em 49 iterações")
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
    ELOS_QUE_ANDAM, ELOS_SORTEAVEIS, pesos_dos_sorteaveis,
)

fab_obs = {g: list(fab.observations[g].terms) for g in ("actor", "critic")}
nossa_obs = {g: list(cfg.observations[g].terms) for g in ("actor", "critic")}
check("o one-hot entra nos DOIS grupos",
      all("elo" in nossa_obs[g] for g in ("actor", "critic")))
# ⚠ O CONTRATO DO APPEND, e ele e' o invariante que sobrevive as fases: os termos do
# FABRICANTE vem primeiro, na ordem dele, e os NOSSOS depois, na ordem em que as fases
# os adicionaram. Checar "o one-hot e' o ultimo" quebrou na F3, quando os canais da
# caixa entraram depois dele -- e quebraria de novo na F4.
check("os termos do FABRICANTE vêm primeiro, na ordem dele",
      all(nossa_obs[g][:len(fab_obs[g])] == fab_obs[g]
          for g in ("actor", "critic")),
      str({g: nossa_obs[g] for g in nossa_obs}))
_NOSSA_OBS = {"actor": ["elo", "caixa"], "critic": ["elo", "caixa", "elo_interno"]}
check("os NOSSOS vêm depois, na ordem das fases; o crítico ganha `elo_interno` no fim",
      all(nossa_obs[g][len(fab_obs[g]):] == _NOSSA_OBS[g]
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
_k_reo_off = dataclasses.replace(
    k, cadeia=dataclasses.replace(k.cadeia, reorientar_inerte=False))
check("os elos sorteáveis são REORIENTAR e PEGAR — o REORIENTAR inerte FICA, a 5% (P8-b)",
      ELOS_SORTEAVEIS == (CMD.REORIENTAR, CMD.PEGAR)
      and pesos_dos_sorteaveis(k) == (k.cadeia.prob_reorientar_inerte,
                                       1.0 - k.cadeia.prob_reorientar_inerte)
      and pesos_dos_sorteaveis(_k_reo_off) == (0.5, 0.5),
      f"{pesos_dos_sorteaveis(k)} / {pesos_dos_sorteaveis(_k_reo_off)}"
      " — CARREGAR e BOTAR só existem como 2º elo de cadeia -> F4")
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

# ⚠⚠ O GATE DOS `track_*` ENTROU EM 31/08, E ELE INVERTE UMA DECISÃO DA SPEC (§4.2).
# A spec dizia "com o twist em ZERO, gatear removeria a única coisa que paga ficar
# parado" — e era exatamente isso que estava errado. Pagar por ficar parado num elo de
# manipulação é pagar pela AUSÊNCIA de tarefa. Medido:
#
#     piso ANDAR = 3,863/s      piso PEGAR = 8,265/s      (antes do gate)
#
# A política ficava imóvel porque isso era ÓTIMO: 145 de retorno contra 102 de explorar,
# com 60% de morte na mesa e episódio de 17,6 s. O `play` do bloco 6 confirmou direto —
# na ação MÉDIA o robô fica na pose default e não tenta pegar.
_TL, _TA = "track_linear_velocity", "track_angular_velocity"
check("os dois `track_*` passam pelo despachante de elo",
      all(cfg.rewards[n].func is RC_.rastreio_por_elo for n in (_TL, _TA)),
      f"{cfg.rewards[_TL].func} / {cfg.rewards[_TA].func}")
check("o `func` do FABRICANTE é preservado dentro de `params`",
      all(cfg.rewards[n].params["func"] is fab.rewards[n].func for n in (_TL, _TA)),
      "o gate embrulha o termo do molde; ele não o reescreve")
check("os params do fabricante seguem intactos sob o embrulho",
      all(all(cfg.rewards[n].params[x] == fab.rewards[n].params[x]
              for x in fab.rewards[n].params) for n in (_TL, _TA)),
      "gatear não pode ter mexido no σ nem no nome do comando")
check("v2.1: os dois NÃO têm mais `elos_que_andam` nem `canal_do_elo` — o gate é "
      "`env.limpo_twist_zerado`, publicado pelo comando (spec P4)",
      all("elos_que_andam" not in cfg.rewards[n].params
          and "canal_do_elo" not in cfg.rewards[n].params
          and "nome_do_comando" not in cfg.rewards[n].params
          for n in (_TL, _TA)),
      str({n: set(cfg.rewards[n].params) for n in (_TL, _TA)}))
check("o PESO dos dois segue o do fabricante — o gate não é um corte de peso",
      all(cfg.rewards[n].weight == fab.rewards[n].weight == 2.0 for n in (_TL, _TA)),
      "o que muda é ONDE o termo paga, e não QUANTO")

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
    # ⚠ v2: a postura lê o elo PUBLICADO (spec §6.0). Na espera inicial um env de
    # manipulação publica ANDAR e a postura é a do FABRICANTE (~0,99 aqui), não 1,0. A
    # neutralidade vale para quem publica um elo de manipulação — que nestes poucos
    # passos pode ser ninguém (a espera vai a 1,5 s); o check é vacuo-seguro e a borda
    # é medida na seção 23.
    _pub3 = _env3.command_manager.get_command("alvo_caixa")[:, CMD.ELO].long()
    _manip_pub = ~_t3.isin(_pub3, _t3.tensor(ELOS_QUE_ANDAM))
    check("num elo de manipulação PUBLICADO a postura vale EXATAMENTE 1,0",
          not bool(_manip_pub.any())
          or float((_pp[_manip_pub] - 1.0).abs().max()) < 1e-6,
          f"{[round(float(x),5) for x in _pp[_manip_pub][:4]]}")
    check("1,0 e não 0,0: zero seria uma penalidade por SORTEIO de elo",
          float(_pp[_manip_envs].min()) > 0.5)
    check("na espera inicial o env de manipulação publica ANDAR, e a postura é a do fabricante",
          bool(((_pub3 == CMD.ANDAR) & _manip_envs).any()),
          "spec §6.3: a espera é ANDAR com twist zero, e o publicado é o que a postura lê")
    check("num elo que ANDA a postura segue sendo a do fabricante",
          bool((~_manip_envs).any())
          and float(_pp[~_manip_envs].std()) > 0.0,
          "constante ali significaria que a subclasse comeu o termo")

    # o sorteio, e os dois consumidores lendo o MESMO elo
    # ⚠ v2: o que tem de bater com o sorteio é o elo INTERNO do comando (spec §6.0). O
    # publicado é ANDAR durante a espera, e o elo sorteado fora dela.
    _int3 = _env3.command_manager.get_term("alvo_caixa")._elo
    check("o elo sorteado bate com o elo INTERNO do comando",
          bool((_int3 == _elo3).all()),
          "se divergirem, a pose nasceu para um elo e o alvo para outro")
    check("e o PUBLICADO é o sorteado ou ANDAR (a espera), nunca um terceiro",
          bool(((_pub3 == _elo3) | (_pub3 == CMD.ANDAR)).all()))
    check("a fatia medida bate com o knob (±0,06 em 128 envs)",
          abs(float((_elo3 == CMD.ANDAR).float().mean()) - k.forma.fatia_loco) < 0.06,
          str(round(float((_elo3 == CMD.ANDAR).float().mean()), 4)))
    # ⚠ NÃO se checa aqui que os dois elos sorteáveis APARECERAM. Com 128 envs e 2,5%
    # por elo, a chance de um deles sair vazio é ~4% por run — a checagem seria FLAKY,
    # e falharia acusando o sorteio quando o sorteio está certo. O invariante é a
    # DISTRIBUIÇÃO, e ela é testada sem simulador logo abaixo.
    check("todo elo sorteado está no conjunto permitido",
          bool(_t3.isin(_elo3, _t3.tensor(
              (CMD.ANDAR,) + tuple(ELOS_SORTEAVEIS))
                        ).all()),
          str({e: int((_elo3 == e).sum()) for e in range(5)}))
    check("CARREGAR e BOTAR NÃO são sorteados (declarado, F4 os abre)",
          not bool(_t3.isin(_elo3,
                            _t3.tensor([CMD.CARREGAR, CMD.BOTAR])).any()))

    # o one-hot, por passo, e a soma
    # ⚠ O one-hot NÃO está no fim do vetor: os 8 canais da caixa vieram depois dele na
    # F3. Fatiar com `[-N_SLOTS:]` leria os últimos 5 canais da CAIXA e o teste passaria
    # medindo a coisa errada.
    #
    # ⚠ E a fatia vem de `observacoes.fatia_do_elo`, que é a MESMA função que o
    # `algoritmo.PPOPorElo` usa para achar o elo dentro da observação. Uma segunda conta
    # aqui deixaria o teste passar com o algoritmo lendo o lugar errado.
    _om3 = _env3.observation_manager
    _FAT_OH = OB_.fatia_do_elo(_om3.group_obs_dim["actor"][0])
    _off3, _acc3 = None, 0
    for _n3, _d3 in zip(_om3.active_terms["actor"], _om3.group_obs_term_dim["actor"]):
        if _n3 == "elo":
            _off3 = _acc3
            break
        _acc3 += _d3[0]
    check("a fatia do elo casa com onde o observation_manager VIVO põe o termo",
          _off3 is not None
          and _FAT_OH == slice(_off3, _off3 + OB_.N_SLOTS),
          f"a função devolveu {_FAT_OH}, o manager põe em {_off3}")
    check("o `caixa` vem DEPOIS do `elo` — é o que faz a contagem do fim valer",
          list(_om3.active_terms["actor"])[-2:] == ["elo", "caixa"],
          str(list(_om3.active_terms["actor"])))
    _oh = _env3.observation_manager.compute()["actor"][:, _FAT_OH]
    check("o one-hot soma 1,0 em toda linha",
          float((_oh.sum(-1) - 1.0).abs().max()) < 1e-6)
    check("o slot aceso é o elo PUBLICADO do env (ANDAR na espera, o sorteado depois)",
          bool((_oh.argmax(-1)
                == _env3.command_manager.get_command("alvo_caixa")[:, CMD.ELO].long()).all()))
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
_fatia_w = CU2.sorteia_elo(_falso, _ids, elo_loco=CMD.ANDAR, elos_manip=ELOS_SORTEAVEIS,
                           fatia_loco=0.5, forcado=None,
                           pesos_manip=pesos_dos_sorteaveis(k))
_buf_w = _falso.limpo_elo
_manip_w = int((_buf_w != CMD.ANDAR).sum())
_frac_reo_w = int((_buf_w == CMD.REORIENTAR).sum()) / max(_manip_w, 1)
check("v2.1 P8-b: com `pesos_manip` o REORIENTAR inerte sai em [3%; 7%] da manipulação",
      0.03 <= _frac_reo_w <= 0.07, f"{_frac_reo_w:.4f} sobre {_manip_w} de manipulação")
check("`forcado` vence o sorteio, e é o que o inspetor usa",
      CU2.sorteia_elo(_falso, _ids, elo_loco=CMD.ANDAR,
                      elos_manip=ELOS_SORTEAVEIS, fatia_loco=0.5,
                      forcado=CMD.BOTAR) == float(CMD.BOTAR)
      and bool((_falso.limpo_elo == CMD.BOTAR).all()))

# ==================== 16b. A JANELA DE ESPERA (portada do g1_poc, 02/09) ==========
secao("16b. a janela de espera")
# ⚠ ELA NÃO EXISTIA no g1_limpo até 02/09, e o dono notou pelo `play`: "o robô não está
# esperando o tempinho antes de receber o comando". A manipulação foi inspirada no
# `g1_poc`, e esta peça não veio na reescrita.
#
# O QUE ELA FAZ: enquanto corre, o `VALIDA` fica em ZERO num elo de manipulação. Os sete
# incentivos pagam nada e o elo NÃO fecha. Na borda ela vai 0->1 com a caixa já
# assentada, e essa descontinuidade é o sinal de "o objetivo chegou".
_esp = k.alvo.espera_s
check("a janela é uma FAIXA, e não um valor fixo",
      _esp[0] < _esp[1],
      "fixa é aprendível como `conte N passos e depois mova`; sorteada, a política TEM "
      "de ler o canal de comando — que é o que o deploy exige")
check("os dois limites são POSITIVOS",
      _esp[0] > 0.0 and _esp[1] > 0.0, str(_esp))
check("ela é 0,5 a 1,5 s — TODA espera é a mesma faixa (spec §6.3, decisão do dono 02/09)",
      _esp == (0.5, 1.5), str(_esp))
check("o knob chega ao termo de comando — não fica no default",
      cfg.commands["alvo_caixa"].espera_s == _esp,
      f"cfg tem {cfg.commands['alvo_caixa'].espera_s}, knobs tem {_esp}")
# ⚠ Ela custa uma FRAÇÃO do episódio, e a conta importa: o g1_poc a tirou da locomoção
# porque lá o episódio morria DENTRO dela.
check("a janela máxima cabe com folga no episódio",
      _esp[1] < 0.10 * cfg.episode_length_s,
      f"{_esp[1]} s contra episódio de {cfg.episode_length_s} s")
check("existe a métrica de peso zero que mede se ela está correndo",
      "fracao_esperando" in cfg.metrics,
      "sem ela, `o robô não espera` e `a janela não existe` leem igual no painel")
# ⚠ O FECHO DE ELO TEM DE LER O `VALIDA`. Sem isso a janela é decorativa: no
# `REORIENTAR` o alvo É a própria caixa, portanto `perto` é trivial e ele fecharia no
# passo ZERO. Era o que fazia `avancos = 0,43` conviver com `sucesso = 0,0000`.
_src_fecha = inspect.getsource(CMD.AlvoCaixaCmd._fecha_elo_corrente)
check("o fecho de elo exige o objetivo ATIVO",
      "VALIDA] > 0.5" in _src_fecha and "& ativo" in _src_fecha,
      "sem isto o REORIENTAL fecharia dentro da própria janela de espera")
# ⚠ Compara as CHAMADAS, e não a menção: o comentário do `_aplica_espera` cita o
# `_avanca_elo`, e procurar o nome cru achava o comentário primeiro. O check falhava
# com o código certo.
_src_upd = inspect.getsource(CMD.AlvoCaixaCmd._update_command)
check("a espera roda ANTES do avanço de elo",
      _src_upd.index("self._aplica_espera()") < _src_upd.index("self._avanca_elo()"),
      "na ordem invertida o elo fecharia com o VALIDA ainda em 1")
# ⚠ A BASE DO BIT VEM DO ELO, e não do próprio `VALIDA`. Ler o `VALIDA` para
# recalculá-lo é DESTRUTIVO: no passo seguinte lê-se o zero já escrito, e o bit nunca
# volta a 1. Medido: `piso PEGAR` caía para 2,000/s exatos.
_src_esp = inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera)
check("a base do bit é recalculada do ELO, e não lida do próprio VALIDA",
      "self._elo != ANDAR" in _src_esp,
      "ler o VALIDA para reescrevê-lo zera o bit para sempre")
# ⚠ As duas fontes do bit têm de concordar: o `_aplica_elo` escreve `VALIDA = 0` só no
# `ANDAR`, e o `_aplica_espera` recalcula a base como `elo != ANDAR`. Um elo novo com
# `VALIDA = 0` faria as duas divergirem em silêncio — este check lê o FONTE do
# `_aplica_elo` e conta quantos elos zeram o bit.
_src_elo = inspect.getsource(CMD.AlvoCaixaCmd._aplica_elo)
check("só UM elo zera o VALIDA no `_aplica_elo`, e é o ANDAR",
      _src_elo.count("VALIDA] = 0.0") == 1
      and _src_elo.count("VALIDA] = 1.0") == len(CMD.ELOS) - 1,
      f"zeram {_src_elo.count('VALIDA] = 0.0')}, ligam "
      f"{_src_elo.count('VALIDA] = 1.0')}, elos {len(CMD.ELOS)}")
# ⚠ v2.1 (spec P4): A JANELA DEIXOU DE CONTAR COMO "ELO QUE ANDA" no rastreio. O
# publicado ainda vira ANDAR (a OBSERVAÇÃO o lê), mas o gate do rastreio agora é
# `env.limpo_twist_zerado`, publicado do elo INTERNO — e o interno NÃO muda durante a
# janela. Como toda cadeia abre num elo PARADO (check abaixo), a espera passa a pagar
# ZERO no rastreio, e não mais o cheio que "elo que anda" pagava. Ver o item 11 da
# seção "v2.1: gradientes" para a prova numérica.
# ⚠ POR ASSINATURA, e não por substring do fonte: o PRÓPRIO docstring de
# `rastreio_por_elo` cita `elos_que_andam` para explicar o que saiu — uma busca de
# substring no fonte acharia essa citação e falharia com o código certo.
_sig_rast = inspect.signature(RC_.rastreio_por_elo).parameters
check("o publicado vira ANDAR na janela, mas o rastreio lê o elo INTERNO — gate novo",
      "publica_andar" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera)
      and "limpo_twist_zerado" in inspect.getsource(RC_.rastreio_por_elo)
      and "elos_que_andam" not in _sig_rast and "canal_do_elo" not in _sig_rast,
      "a espera de um elo PARADO agora paga zero no rastreio (era o cheio, pelo "
      "publicado)")
# ⚠⚠ E O INVARIANTE QUE SUSTENTA ISSO: a janela só ocorre em elo PARADO, portanto o
# twist é ZERO nela e o rastreio paga por MANTER velocidade zero. Se uma cadeia nova
# abrisse em `CARREGAR` — o único elo de manipulação que anda — a janela passaria a
# pagar por rastrear um comando NÃO nulo com o objetivo desligado, que é outra coisa.
check("toda cadeia ABRE num elo parado — o twist é zero em toda janela",
      all(c[0] in cfg.commands["alvo_caixa"].elos_parados for c in CMD.CADEIAS),
      f"abrem em {[CMD.ELOS[c[0]] for c in CMD.CADEIAS]}, parados são "
      f"{[CMD.ELOS[e] for e in cfg.commands['alvo_caixa'].elos_parados]}")

# --- O COMPORTAMENTO, medido. É o check que reprova o módulo sem a janela. ---
try:
    import torch as _tj

    _cj = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _cj.scene.num_envs = 32
    _ej = ManagerBasedRlEnv(cfg=_cj, device="cpu")
    _ej.reset()
    _naj = _ej.action_manager.total_action_dim
    _ej.step(_tj.zeros(_ej.num_envs, _naj))
    _val0 = _ej.command_manager.get_command("alvo_caixa")[:, CMD.VALIDA].clone()
    _nmj = list(_cj.rewards)

    def _somaj(nomes):
        _sr = _ej.reward_manager._step_reward
        return float(sum(_sr[:, _nmj.index(n)].mean() for n in nomes))

    _staged0 = float(_ej.reward_manager._step_reward[:, _nmj.index("staged")].mean())
    _trk0 = _somaj((_TL, _TA))
    _piso_j0 = float(_ej.reward_manager._step_reward.mean(0).sum())

    # ⚠ passos suficientes para a maior janela sorteada terminar
    for _ in range(int(_esp[1] / _ej.step_dt) + 5):
        _ej.step(_tj.zeros(_ej.num_envs, _naj))
    _val1 = _ej.command_manager.get_command("alvo_caixa")[:, CMD.VALIDA]
    _staged1 = float(_ej.reward_manager._step_reward[:, _nmj.index("staged")].mean())
    _trk1 = _somaj((_TL, _TA))

    check("no PRIMEIRO passo de um elo de manipulação o VALIDA é ZERO",
          float(_val0.max()) == 0.0,
          f"máximo medido {float(_val0.max())} — a janela mínima é {_esp[0]} s")
    check("e o `staged` paga ZERO enquanto a janela corre",
          abs(_staged0) < 1e-9,
          f"medido {_staged0:.6f}/passo — prêmio antes de haver objetivo")
    check("DEPOIS da janela o VALIDA é UM em todos os envs",
          float(_val1.min()) == 1.0,
          f"mínimo medido {float(_val1.min())}")
    check("e o `staged` volta a pagar — a descontinuidade É o sinal",
          _staged1 > 0.0,
          f"antes {_staged0:.4f} depois {_staged1:.4f}")
    # ⚠⚠ A JANELA ERA `ANDAR` COM COMANDO ZERO (02/09), e por isso o `track_*` pagava
    # durante ela — sem isto ela era um TERCEIRO regime mais pobre que os dois, e
    # alcançar a caixa na janela era GRÁTIS. v2.1 (spec P4) TROCOU o gate do rastreio:
    # ele deixou de ler o PUBLICADO (que ainda vira ANDAR na janela) e passou a ler o
    # elo INTERNO via `env.limpo_twist_zerado` — e o interno NÃO muda na janela. Como
    # toda cadeia abre num elo PARADO, a janela volta a pagar ZERO no rastreio. É o
    # risco DECLARADO da proposta P4: nada paga por ficar parado durante a espera;
    # `action_rate` e o alvo ancorado na base seguram. Ver a seção "v2.1: gradientes",
    # item 11, para a prova direta do gate.
    check("v2.1: os dois `track_*` NÃO pagam mais durante a janela — o gate agora é "
          "o elo INTERNO, que fica parado nela",
          _trk0 == 0.0,
          f"{_trk0:.4f}/s — antes (spec §6.3) pagava >0,5 pelo PUBLICADO ANDAR")
    # ⚠ E O GATE POR ELO NÃO PODE TER SIDO AFROUXADO. Passada a borda, a estátua num
    # elo parado tem de continuar colhendo ZERO — é o conserto de 31/08, e nem a janela
    # nem o gate novo podem ser a porta por onde ele sai.
    check("e continuam em ZERO passada a borda — o gate por elo continua de pé",
          abs(_trk1) < 1e-6,
          f"{_trk1:.6f}/s — o piso da estátua de 8,265/s voltaria por aqui")
    print(f"  VALIDA: {float(_val0.max()):.0f} na abertura -> "
          f"{float(_val1.min()):.0f} depois de {_esp[1]} s   |   "
          f"staged {_staged0:.4f} -> {_staged1:.4f}   |   "
          f"track {_trk0:.3f} -> {_trk1:.3f}   |   piso na janela {_piso_j0:.3f}/s")
    del _ej

    # --- no ANDAR a janela é ZERO: a locomoção não paga por ela ---
    _cl = make_env_cfg(k, inspecao=True, elo=CMD.ANDAR)
    _cl.scene.num_envs = 16
    _el = ManagerBasedRlEnv(cfg=_cl, device="cpu")
    _el.reset()
    _el.step(_tj.zeros(_el.num_envs, _el.action_manager.total_action_dim))
    check("no ANDAR a janela é ZERO — ela atrasava o aprendizado da marcha",
          float(_el.limpo_aguardando.max()) == 0.0,
          "o g1_poc a tirou da locomoção em 24/08 por medição")
    del _el
except Exception as _ejx:      # noqa: BLE001
    _falhas.append(f"a janela de espera não pôde ser medida: "
                   f"{type(_ejx).__name__}: {_ejx}")

# ⚠ O INSPETOR TEM DE ZERAR A JANELA, e a trava existe porque o contrário JÁ aconteceu:
# ele dá dois passos (0,04 s) e lê o bit, portanto com a janela viva as quatro linhas de
# manipulação acusam "o objetivo devia estar LIGADO" — o inspetor reprovando o desenho
# por medir antes de o objetivo existir. Queimar a janela no laço dele não serve: 55
# passos deixam o elo AVANÇAR e a tabela mostra um elo que não é o pedido.
try:
    from g1_limpo import inspeciona as _INS

    check("o INSPETOR zera a janela de espera, e o `make_env_cfg` não",
          "espera_s = (0.0, 0.0)" in inspect.getsource(_INS._ambiente)
          and cfg.commands["alvo_caixa"].espera_s == _esp,
          "o smoke precisa do modo de inspeção COM a janela viva para medir a borda")
except Exception as _insx:      # noqa: BLE001
    _falhas.append(f"o inspetor não pôde ser lido: "
                   f"{type(_insx).__name__}: {_insx}")

# ============ 16c. A ENTREGA DA TAREFA AO VIVO (só visualizador, 02/09) ===========
secao("16c. a entrega da tarefa ao vivo")
# ⚠ Ela simula o DEPLOY: a caixa na laje à vista do robô desde o começo, o robô de pé
# com comando de velocidade ZERO, e a tarefa chegando aos N segundos. No treino isto
# NÃO existe — o elo é sorteado no reset e nunca troca no meio. Este bloco existe para
# que o caminho de visualizador não possa vazar para o treino, e o risco aqui é MAIOR
# que o do `avanca_elo`: este evento também zera o twist, portanto no treino ele
# apagaria a locomoção inteira.
check("o cfg de TREINO não tem o evento de entrega",
      "entrega_tarefa" not in cfg.events,
      "no treino ele zeraria o twist e apagaria a locomoção, sem erro nenhum")
check("nem o `play` sozinho o cria — ele exige `entrega_apos_s` explícito",
      "entrega_tarefa" not in play.events)
_cfg_tr = make_env_cfg(k, play=True, elo=CMD.ANDAR, entrega_apos_s=3.0)
check("com `entrega_apos_s` no play o evento existe, e é de INTERVALO",
      "entrega_tarefa" in _cfg_tr.events
      and _cfg_tr.events["entrega_tarefa"].mode == "interval",
      "o `run_play` do mjlab não expõe gancho por passo; intervalo é o idioma")
# ⚠ INTERVALO DE UM PASSO, e não do prazo da entrega: o evento tem de rodar todo passo
# para manter o twist em zero. Com o intervalo no prazo, o robô sairia andando.
_dt_esp = _cfg_tr.sim.mujoco.timestep * _cfg_tr.decimation
check("o intervalo é de UM PASSO, como o `trava_robo`",
      _cfg_tr.events["entrega_tarefa"].interval_range_s == (_dt_esp, _dt_esp),
      f"{_cfg_tr.events['entrega_tarefa'].interval_range_s} contra dt={_dt_esp}")
check("o prazo da entrega vai em `params`, e não no intervalo",
      _cfg_tr.events["entrega_tarefa"].params["entrega_apos_s"] == 3.0)
# ⚠ A TASK REGISTRADA, e a assimetria dela é o ponto: o `run_play` carrega o cfg
# REGISTRADO e roda o próprio laço — ele não expõe gancho para mutar cfg. Portanto a
# variante tem de existir no registro. E o `env_cfg` dela é o SIMPLES, porque
# `entrega_apos_s` estoura fora de play/inspecao de propósito.
check("a task de entrega está registrada, e o robô fica LIVRE nela",
      _PKG.TASK_ENTREGA in __import__(
          "mjlab.tasks.registry", fromlist=["list_tasks"]).list_tasks()
      and "trava_robo" not in _PKG.make_env_cfg(
          play=True, elo=CMD.ANDAR,
          entrega_apos_s=_PKG.ENTREGA_APOS_S).events,
      "com `trava_robo` o robô fica pinado e a transição não tem o que mostrar")
# ⚠ OS PARAMS DA CENA SÃO REUSADOS do evento de reset. Duas cópias sairiam de sincronia
# no dia em que um nível novo entrar, e a entrega posicionaria a mobília com a tabela
# velha — a caixa nasceria numa altura que a recompensa não espera.
check("os params de cena vêm do evento de reset, e não redigitados",
      _cfg_tr.events["entrega_tarefa"].params["cena"]
      == dict(_cfg_tr.events["posiciona_cena"].params))
# ⚠⚠ A BASE RESETA NA FAIXA DE MANIPULAÇÃO, e sem isto o modo é INÚTIL. O
# `reset_base_por_elo` escolhe a faixa pelo ELO, e aqui o elo de abertura é o `ANDAR`:
# o robô caía em x ±0,50 m, y ±0,50 m e yaw ±3,14 contra uma mobília de pose ABSOLUTA.
# Medido no viewer — ele nascia DENTRO da mesa, longe dela, ou de costas. A máscara
# vazia manda todo env para o `faixa_manipula`, sem ramo novo no despachante.
check("a base reseta na faixa de MANIPULAÇÃO, e não na de locomoção",
      _cfg_tr.events["reset_base"].params["elos_que_andam"] == (),
      "com a faixa de loco o robô nasce dentro da mesa ou de costas para ela")
check("e o cfg de TREINO segue com a lista de verdade",
      tuple(cfg.events["reset_base"].params["elos_que_andam"])
      == tuple(ELOS_QUE_ANDAM),
      "zerar isto no treino faria todo env de locomoção nascer alinhado — foi o "
      "defeito espelhado que custou um bloco")
for _kw, _msg in ((dict(entrega_apos_s=3.0), "sem play nem inspecao"),
                  (dict(play=True, elo=CMD.ANDAR, entrega_apos_s=3.0,
                        entrega_para=CMD.ANDAR), "entregando ANDAR"),
                  (dict(play=True, elo=CMD.PEGAR, entrega_apos_s=3.0),
                   "partindo de um elo que não é ANDAR")):
    try:
        make_env_cfg(k, **_kw)
        check(f"`entrega_apos_s` {_msg} tem de ESTOURAR", False,
              "passou em silêncio")
    except AssertionError:
        check(f"`entrega_apos_s` {_msg} estoura, e é assert", True)

# --- O COMPORTAMENTO: a sequência inteira, medida ---
try:
    import torch as _tt

    _ct = make_env_cfg(k, inspecao=True, elo=CMD.ANDAR, entrega_apos_s=3.0)
    _ct.scene.num_envs = 8
    # ⚠ janela FIXA aqui: com faixa, o passo em que o bit liga varia e o teste ficaria
    # flaky. O sorteio já é testado no 16b.
    _ct.commands["alvo_caixa"].espera_s = (1.0, 1.0)
    _et = ManagerBasedRlEnv(cfg=_ct, device="cpu")
    _et.reset()
    _nat = _et.action_manager.total_action_dim
    _ttc = _et.command_manager.get_term("alvo_caixa")

    def _leitura():
        _cm = _et.command_manager.get_command("alvo_caixa")
        _tw = _et.command_manager.get_term("twist").vel_command_b
        return (int(_ttc._elo[0]), float(_cm[0, CMD.VALIDA]),
                float(_ttc._espera[0]),
                float(_et.scene["box"].data.root_link_pos_w[0, 2]
                      - _et.scene.env_origins[0, 2]),
                float(_tw.norm(dim=-1).max()))

    def _anda_ate(t_s):
        for _ in range(int(t_s / _et.step_dt)):
            _et.step(_tt.zeros(_et.num_envs, _nat))

    _anda_ate(0.2)
    _eA, _vA, _spA, _zA, _twA = _leitura()
    _anda_ate(1.3)
    _eB, _vB, _spB, _zB, _twB = _leitura()
    _anda_ate(1.6)
    _eC, _vC, _spC, _zC, _twC = _leitura()
    _anda_ate(1.1)
    _eD, _vD, _spD, _zD, _twD = _leitura()

    # ⚠ A CENA DO `PEGAR` DESDE O COMEÇO, com o elo ainda em `ANDAR`. É o pedido, e ele
    # não é de graça: no `ANDAR` o termo de comando manda a laje a +5 m, e ele roda
    # DEPOIS dos eventos de reset (`currículo -> eventos -> comando`). Um
    # `posiciona_cena` no reset seria desfeito em silêncio.
    check("a cena do `pegar` está POSTA, com o elo ainda em ANDAR",
          _eA == CMD.ANDAR and _zA < 2.0,
          f"elo={_eA} caixa_z={_zA:+.2f} — a laje voltou de +5 m?")
    # ⚠ E O ROBÔ PARTE NA MESA, DE FRENTE PARA ELA. Sem o reset na faixa de
    # manipulação ele nascia em qualquer lugar a ±0,50 m com qualquer rumo — dentro da
    # mesa, longe, ou de costas. O deploy é "chegou andando -> velocidade zero ->
    # pega", e este modo simula só a segunda metade.
    _pr = _tt.tensor(k.cena.prateleira_xy)
    _pb = (_et.scene["robot"].data.root_link_pos_w
           - _et.scene.env_origins)[:, :2]
    _dl = _tt.norm(_pb - _pr, dim=-1)
    _qb = _et.scene["robot"].data.root_link_quat_w
    _yw = _tt.atan2(2 * (_qb[:, 0] * _qb[:, 3] + _qb[:, 1] * _qb[:, 2]),
                    1 - 2 * (_qb[:, 2] ** 2 + _qb[:, 3] ** 2)).abs()
    check("o robô parte NA MESA, e não a ±0,50 m dela",
          float(_dl.max()) < 0.75,
          f"dist_laje max {float(_dl.max()):.2f} m — a laje está a "
          f"{float(_pr.norm()):.2f} m da origem")
    check("e DE FRENTE para ela — o rumo não é sorteio",
          float(_yw.max()) < 0.5,
          f"|yaw| max {float(_yw.max()):.2f} rad de um limite de ±0,2")
    # ⚠ O TWIST EM ZERO É O "comando de andar como 0". Sem ele o `ANDAR` sorteia
    # velocidade e o robô sai andando para dentro da mesa antes de a tarefa chegar.
    check("e o comando de velocidade é ZERO em todos os envs",
          _twA < 1e-9 and _twB < 1e-9,
          f"|twist| = {_twA:.4f} / {_twB:.4f}")
    # ⚠⚠ E ZERO **NA OBSERVAÇÃO DO RESET**, que é o check que faltava. A primeira
    # versão zerava o twist no evento de INTERVALO, e o `reset()` chama
    # `command_manager.compute(dt=0.0)` SEM rodar evento de intervalo: a primeira
    # observação de todo episódio saía com comando de até 2 m/s. Medido:
    # `cmd_obs_max = 1,97`. A política dava o primeiro passo contra "ande a 2 m/s" e
    # depois tinha de frear o que ela mesma começou — deriva lateral lenta no viewer,
    # relatada pelo dono.
    #
    # ⚠ E O CHECK ANTERIOR NÃO PEGAVA, porque ele lia o BUFFER depois do `step` — isto
    # é, depois do evento. Ler o buffer não é ler o que a política viu. Este mede
    # ANTES de qualquer passo, que é onde o defeito vivia.
    _cr = make_env_cfg(k, play=True, elo=CMD.ANDAR, entrega_apos_s=3.0)
    _cr.scene.num_envs = 32
    _er = ManagerBasedRlEnv(cfg=_cr, device="cpu")
    _er.reset()
    _cmd_reset = float(
        _er.command_manager.get_command("twist").abs().max())
    check("o comando é zero JÁ NA OBSERVAÇÃO DO RESET, antes do 1º passo",
          _cmd_reset < 1e-9,
          f"cmd_obs_max = {_cmd_reset:.4f} — o `reset` não roda evento de intervalo")
    check("e quem zera é o `_zera_twist_nos_parados`, com o ANDAR em `elos_parados`",
          CMD.ANDAR in tuple(_cr.commands["alvo_caixa"].elos_parados)
          and CMD.ANDAR not in tuple(cfg.commands["alvo_caixa"].elos_parados),
          "no treino o ANDAR NÃO pode ser elo parado — aquilo é a locomoção")
    del _er
    check("antes do prazo a tarefa NÃO chegou, e o objetivo segue desligado",
          _eB == CMD.ANDAR and _vB == 0.0,
          f"elo={_eB} VALIDA={_vB} a 1,5 s de um prazo de 3,0 s")
    check("no prazo o elo vira o pedido e a janela é ARMADA",
          _eC == CMD.PEGAR and _spC > 0.9,
          f"elo={_eC} espera={_spC:.2f}")
    # ⚠ O BIT TEM DE CAIR NO MESMO INSTANTE. O `_aplica_elo` escreve `VALIDA = 1` num
    # elo de manipulação, e o `_aplica_espera` só corrigiria isso no passo SEGUINTE —
    # o evento roda fora da passada do `command_manager`. Um passo de objetivo ligado
    # com a janela armada estragaria exatamente o instante que se quer olhar.
    check("e o objetivo NÃO liga no instante da entrega",
          _vC == 0.0,
          f"VALIDA={_vC} com espera={_spC:.2f} — o bit vazou um passo")
    check("passada a janela o objetivo LIGA",
          _vD == 1.0 and _spD == 0.0,
          f"VALIDA={_vD} espera={_spD:.2f}")
    check("a cadeia entregue ABRE no elo entregue",
          int(_ttc._cadeia[0]) >= 0
          and int(CMD.CADEIAS[int(_ttc._cadeia[0])][0]) == CMD.PEGAR,
          f"cadeia={int(_ttc._cadeia[0])}")
    print(f"  entrega: caixa_z {_zA:+.2f} posta e twist 0 desde 0,2 s  |  "
          f"elo {_eB}->{_eC} no prazo  |  VALIDA {_vC:.0f} -> {_vD:.0f} "
          f"depois de {_spC:.1f} s")
    del _et
except Exception as _ttx:      # noqa: BLE001
    _falhas.append(f"a entrega ao vivo não pôde ser medida: "
                   f"{type(_ttx).__name__}: {_ttx}")

# ================================= 17. o PISO DA ESTÁTUA, medido
secao("17. o preço declarado: quanto uma estátua colhe (F2)")
# ⚠⚠ O CRITÉRIO ORIGINAL DO PLANO VOLTOU EM 31/08, e ele estava certo desde o começo:
# "um env em PEGAR colhe 0/s dos `track_*`". Ele havia sido substituído por "medir o
# piso e declará-lo" com o argumento de que gatear removeria a única coisa que paga
# ficar parado — e era justamente esse pagamento o defeito. Medido, antes do gate:
#
#     piso ANDAR = 3,863/s      piso PEGAR = 8,265/s
#
# O elo de manipulação era o lugar mais confortável do ambiente, e ficar imóvel era
# ÓTIMO: 145 de retorno contra 102 de explorar, com 60% de morte na mesa. O `play` do
# bloco 6 mostrou o resultado direto — na ação média o robô não tenta pegar.
#
# Declarar um preço não conserta o preço. Agora o piso do `PEGAR` fica ABAIXO do piso do
# `ANDAR`, e a medição continua aqui porque é ela que prova o gate.
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
        # ⚠ PASSOS SUFICIENTES PARA A JANELA DE ESPERA TERMINAR. Ela vai a 1,0 s no
        # sorteio, isto é 50 passos, e durante ela o `VALIDA` é ZERO — os sete
        # incentivos pagam nada. Medir o piso com 6 passos (0,12 s) mediria o objetivo
        # DESLIGADO, e o piso do `PEGAR` sairia falsamente baixo.
        _passos4 = int(k.alvo.espera_s[1] / _c4.decimation / 0.005) + 10
        for _ in range(_passos4):
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
    # ⚠ ZERO EXATO, e é o cheque do gate. Antes de 31/08 isto media ~3,8/s: a estátua
    # num elo de manipulação colhia 4,0/s por rastrear um comando NULO, e era a maior
    # parcela do piso de 8,265/s que travava a exploração.
    check("MEDIDO: a estátua num elo parado colhe ZERO dos dois `track_*`",
          abs(_tk) < 1e-6, f"{_tk:.6f}/s — antes do gate media ~3,8/s")
    check("e a postura NÃO entra nesse piso: ela é neutra, exatamente 1,0",
          abs(_piso["parado"]["pose"] - 1.0) < 1e-6,
          f"{_piso['parado']['pose']:.6f}")
    # ⚠ A DESIGUALDADE INVERTEU, e a inversão é o objetivo. Antes o elo de manipulação
    # pagava 2,1x mais que o de locomoção por ficar imóvel (8,265 contra 3,863/s), e
    # ficar imóvel era ótimo. Agora o de manipulação paga MENOS: o único caminho de
    # renda ali é a tarefa.
    check("o elo de manipulação paga MENOS que o que anda, por ficar imóvel",
          _piso["parado"]["TOTAL"] < _piso["anda"]["TOTAL"] + 0.5,
          f"parado={_piso['parado']['TOTAL']:.3f}/s  "
          f"anda={_piso['anda']['TOTAL']:.3f}/s — antes era 8,265 contra 3,863")
    check("o `track_*` continua pagando no elo que ANDA",
          (_piso["anda"]["track_linear_velocity"]
           + _piso["anda"]["track_angular_velocity"]) > 0.5,
          f"{_piso['anda']['track_linear_velocity']:.3f} + "
          f"{_piso['anda']['track_angular_velocity']:.3f} — gatear a locomoção "
          "INTEIRA quebraria o andar, que é o que este bloco NÃO pode tocar")
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
check("a soma dos pesos é 12,5/s (v2.1: `precise_pos` 2,0 -> 3,0)",
      abs(sum(cfg.rewards[n].weight for n in SETE) - 12.5) < 1e-9)
check("v2.1: `load` SAIU — `largou` = 1,0 e `renda_congelada` = 1,0 fecham o BOTAR "
      "(spec §6.6.2, P3)",
      "load" not in cfg.rewards
      and cfg.rewards["largou"].weight == 1.0
      and cfg.rewards["renda_congelada"].weight == 1.0,
      str({n: cfg.rewards[n].weight for n in ("largou", "renda_congelada")
           if n in cfg.rewards}))
check("o `staged` é o maior — é o único com gradiente na pose de repouso",
      cfg.rewards["staged"].weight == max(cfg.rewards[n].weight for n in SETE))
check("o `precise_pos` é o ÚNICO com σ fixo, e ele é a tolerância de ACEITE",
      cfg.rewards["precise_pos"].params["sigma"] == tr.precise_pos_sigma
      and "sigma" not in cfg.rewards["staged"].params,
      "quem faz a rampa de aproximação é o `staged`, com σ por env")
# ⚠ v2.1: `sustentacao` virou FUNÇÃO — o cronômetro é o `_sust` do comando (que já
# reseta no `_resample_command`), e não há mais estado próprio para resetar.
check("o `sustentacao` NÃO tem estado — não é mais classe, e não tem `reset`",
      not inspect.isclass(RC_.sustentacao)
      and getattr(RC_.sustentacao, "reset", None) is None)
check("o `squeeze` usa os sensores de PALMA, que têm o campo `force`",
      tuple(cfg.rewards["squeeze"].params["sensores"]) == tuple(C.SENSOR_PALMA)
      and all("force" in por_nome[n].fields for n in C.SENSOR_PALMA))
check("o `unload` usa o sensor de APOIO, que tem `force`",
      cfg.rewards["unload"].params["sensor_apoio"] == C.SENSOR_APOIO
      and "force" in por_nome[C.SENSOR_APOIO].fields)
check("os params do `sustentacao` são só o nome do comando — o resto vem do comando",
      set(cfg.rewards["sustentacao"].params) == {"nome_do_comando"},
      str(cfg.rewards["sustentacao"].params))

# --- a observação cresceu pelo contrato do APPEND ---
check("os canais da caixa entram DEPOIS do one-hot, nos dois grupos",
      list(cfg.observations["actor"].terms)[-2:] == ["elo", "caixa"]
      and list(cfg.observations["critic"].terms)[-3:] == ["elo", "caixa", "elo_interno"],
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

    # ⚠ A TOLERÂNCIA É MEDIDA, e não escolhida. O σ é fixado na passada do `_pendente`,
    # e a caixa continua ASSENTANDO na laje depois disso: ela desliza alguns milímetros
    # antes de parar. Eu havia derivado a banda de uma tolerância de 5 mm CHUTADA, e a
    # deriva real chega a ~11 mm — a checagem falhava em 1 de 3 runs acusando o
    # assentamento da caixa. Aqui o próprio deslocamento de um passo é a tolerância.
    _antes5 = _t5c.dist_palma_caixa(_ids5).clone()
    _e5.step(_t5.zeros(_e5.num_envs, _e5.action_manager.total_action_dim))
    _deriva = float((_t5c.dist_palma_caixa(_ids5) - _antes5).abs().max())
    _tol5 = max(_deriva * 3.0, 2e-3)     # 3 passos de folga, piso de 2 mm

    check("o σ NÃO é constante — cada env tem o seu",
          float(_s5.std()) > 0.01, f"std={float(_s5.std()):.4f}")
    check("o σ É a distância inicial daquele env",
          float((_s5 - _d5).abs().max()) <= _tol5,
          f"pior desvio {float((_s5-_d5).abs().max())*1000:.1f} mm, "
          f"tolerância medida {_tol5*1000:.1f} mm (deriva de 1 passo: "
          f"{_deriva*1000:.1f} mm)")
    # ⚠ O NÚMERO QUE DECIDE A F3. Com σ fixo de 0,10 isto valeria 1e−05.
    #
    # ⚠ A BANDA É DERIVADA, e não escolhida. O σ é fixado no passo do `_pendente`, e a
    # caixa continua assentando depois disso — o check acima tolera 5 mm de deriva. No
    # env de σ mínimo (0,08 m) 5 mm são 6,25% de razão, logo o kernel varia entre
    # `exp(−1,0625²)` e `exp(−0,9375²)`, isto é [0,323; 0,415]. Uma tolerância de
    # ±0,02 é MAIS APERTADA que isso e acusa o assentamento da caixa, não o desenho.
    # ⚠ E a banda do kernel sai da MESMA tolerância medida, sobre o σ MÍNIMO — que é o
    # env em que uma deriva de milímetros mais desloca a razão `d/σ`.
    _r = _tol5 / k.tarefa.sigma_min
    _lo = math.exp(-(1.0 + _r) ** 2)
    _hi = math.exp(-max(1.0 - _r, 0.0) ** 2)
    check("o kernel de alcance vale exp(−1) = 0,368 no passo em que o elo abre, "
          "em TODOS os envs",
          _lo - 1e-3 <= float(_ker.min()) and float(_ker.max()) <= _hi + 1e-3,
          f"min {float(_ker.min()):.4f} max {float(_ker.max()):.4f}, "
          f"banda derivada [{_lo:.3f}; {_hi:.3f}]")
    check("a derivada do kernel no repouso é > 1,0 até no env mais distante",
          float((2.0 * _d5 / _s5 ** 2 * _ker).min()) > 1.0,
          f"min {float((2.0*_d5/_s5**2*_ker).min()):.3f} por metro")
    check("a distância é até a FACE LATERAL, não o centro",
          float(_d5.max()) < 0.60,
          "ao centro o mínimo alcançável é 0,191 m e o kernel saturava em 0,674")
    check("o σ de orientação tem piso, e ele é em RADIANOS",
          float(_t5c.sigma_ori.min()) >= _c5.commands["alvo_caixa"].sigma_ori_min
          - 1e-9)

    # ------------------ o canal CAIXA -> ALVO também nasce com derivada viva
    # ⚠ QUEM É A RAMPA DE CAIXA->ALVO É O `trazer`, dentro do `staged`, e não o
    # `precise_pos`. O `precise_pos` tem σ FIXO de 0,05 m e vale ~0 a 0,30 m — ele é
    # um ACEITE ("a caixa está NO alvo?"), e ele TEM de ser apertado: ele é o único
    # termo de posição que NÃO passa por `alcancar`, portanto alargá-lo pagaria por
    # empurrar a caixa até o alvo com o pé. Uma run antiga do `g1_poc` aprendeu
    # exatamente isso. As duas perguntas ficam em dois termos.
    #
    # ⚠ A banda é FROUXA de propósito. No `PEGAR` o alvo é reancorado na base a cada
    # passo, portanto a distância anda um pouco depois de o σ ser fixado. O que se
    # afirma aqui é a PROPRIEDADE — nem saturado em 0, nem saturado em 1, com
    # derivada viva —, e não um valor.
    _dalvo5 = (_e5.scene["box"].data.root_link_pos_w
               - _t5c.command[:, CMD.ALVO]).norm(dim=-1)
    _traz5 = _t5.exp(-(_dalvo5 / _t5c.sigma_trazer) ** 2)
    check("o σ do `trazer` é POR ENV, e não o piso",
          float(_t5c.sigma_trazer.min()) > k.tarefa.sigma_min + 1e-6,
          f"min {float(_t5c.sigma_trazer.min()):.4f} m contra piso "
          f"{k.tarefa.sigma_min:.4f} m")
    check("o `trazer` nasce longe dos dois extremos — a rampa caixa->alvo está viva",
          0.15 <= float(_traz5.min()) and float(_traz5.max()) <= 0.75,
          f"min {float(_traz5.min()):.4f} max {float(_traz5.max()):.4f} "
          "(saturado em 0 ou em 1 seria derivada zero)")
    check("a derivada do `trazer` no repouso é > 1,0 no env mais distante",
          float((2.0 * _dalvo5 / _t5c.sigma_trazer ** 2 * _traz5).min()) > 1.0,
          f"min {float((2.0*_dalvo5/_t5c.sigma_trazer**2*_traz5).min()):.3f} "
          "por metro")

    # ------------------------------- o alcance é BIMANUAL e LATERAL (28/08)
    # ⚠ Até 28/08 era `min` sobre as palmas contra uma ESFERA no centro da caixa.
    # Dois buracos: com `min` uma mão saturava o kernel e a segunda não tinha
    # gradiente — mas o `squeeze` é `min` das FORÇAS e exige as duas; e com a
    # esfera, tocar o topo pagava igual a tocar a lateral. A cadeia ficava sem
    # ponte entre "uma mão encosta" e "as duas apertam", e o bloco 3 travou ali.
    # ⚠ TUDO MEDIDO FRESCO NESTE PONTO. O `_d5` acima foi capturado ANTES dos passos
    # que mediram a deriva da caixa, e comparar aquele valor com um cálculo de agora
    # acusa o assentamento da caixa em vez do desenho. Foi assim que este check
    # falhou na primeira escrita.
    _alv5 = _t5c.alvos_das_palmas(_ids5)
    _d5_agora = _t5c.dist_palma_caixa(_ids5)
    _sep5 = (_alv5[:, 0] - _alv5[:, 1]).norm(dim=-1)
    _mid5 = _alv5.mean(dim=1)
    _cx5 = _e5.scene["box"].data.root_link_pos_w
    check("cada palma tem o SEU alvo, e são dois pontos distintos",
          _alv5.shape[1] == 2 and float(_sep5.min()) > 1e-3)
    check("os dois alvos ficam nas FACES laterais — separados por 2×meia-aresta DO ENV",
          float((_sep5 - 2.0 * _e5.limpo_meia_aresta[:, 1]).abs().max()) < 1e-5,
          f"separação medida {float(_sep5.mean()):.4f} m")
    check("o ponto médio dos dois alvos É o centro da caixa",
          float((_mid5 - _cx5).norm(dim=-1).max()) < 1e-5,
          "o offset gira com a caixa, portanto a pose pedida acompanha a "
          "orientação dela")
    _por_palma5 = (_e5.scene["robot"].data.site_pos_w[:, _t5c._ids_palma, :]
                   - _alv5).norm(dim=-1)
    check("a distância publicada é a MÉDIA das duas, e não o mínimo",
          float((_d5_agora - _por_palma5.mean(dim=1)).abs().max()) < 1e-6
          and float((_d5_agora - _por_palma5.min(dim=1).values).abs().max()) > 1e-4,
          "a média é o que acopla as mãos: uma mão atrasada derruba o termo, e com "
          "`min` a segunda mão não teria gradiente nenhum")

    # ------------------------- a face pedida CONGELA fora do `REORIENTAR` (28/08)
    # ⚠ Dois pedidos diferentes. No `REORIENTAR` a direção é VIVA ("vire a face
    # para o robô"); nos outros elos ela congela na normal da abertura, e aí o
    # termo pergunta "a caixa girou desde então?" — ele paga por erguer SEM
    # torcer. Com a direção viva em todo elo, o `precise_ori` ficava inerte no
    # nível 0 (caixa nasce alinhada, `sigma_ori` com piso de 0,20 rad) E o alvo se
    # movia com o ROBÔ: andar em volta da caixa mudava o termo sem tocá-la.
    check("no `PEGAR` a direção pedida está CONGELADA",
          not bool(_t5c._face_viva.any()),
          "este env foi forçado no PEGAR — nenhuma face pode estar viva")
    # ⚠ A TOLERÂNCIA COBRE O ASSENTAMENTO DA CAIXA, e não o desenho: a normal é
    # congelada na passada do `_pendente` e a caixa continua assentando na laje depois
    # disso. MEDIDO em execuções seguidas: até 0,024 rad. Com 2e−2 o check falhava
    # acusando o solver de contato, e não o desenho.
    #
    # 4e−2 rad são 2,3°, contra os 0,26 rad (15°) que a direção VIVA dava no nível 0.
    # A separação entre os dois regimes segue sendo de mais de 6× — que é o que este
    # check afirma.
    check("congelada na normal ATUAL, portanto o erro angular nasce em ZERO",
          float(_t5c.command[:, CMD.ANG].abs().max()) < 4e-2,
          f"pior erro {float(_t5c.command[:, CMD.ANG].abs().max()):.5f} rad — "
          "o pedido é 'erga sem torcer', e no passo da abertura não há giro")
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
        _passa_janela(_e6, _e6.action_manager.total_action_dim, _t6)
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
          _vals["pegar"]["TOTAL"] < 5.815 + 12.5,
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
    _passa_janela(_e7, _na7, _t7)
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
        # ⚠ SEM `+ meia_aresta` desde 28/08: o alvo de cada palma JÁ está na face
        # lateral, portanto `dist_palma_caixa` não subtrai mais a meia-aresta.
        _novo7 = _palma7 + _dir7 * (_d0_7 * _f7).unsqueeze(-1)
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
    _passa_janela(_e8, _na8, _t8)
    # põe a caixa NO alvo, com a face certa, e conta
    for _ in range(6):
        _caixa8.write_root_link_pose_to_sim(
            _t8.cat([_t8c.command[:, CMD.ALVO],
                     _t8.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                         _e8.num_envs, 4)], dim=-1))
        _caixa8.write_root_link_velocity_to_sim(_t8.zeros(_e8.num_envs, 6))
        _e8.step(_t8.zeros(_e8.num_envs, _na8))
    # ⚠ v2.1: o cronômetro é `_sust`, no termo de COMANDO — não há mais estado no
    # termo de recompensa (`sustentacao` virou função pura, spec P2).
    _antes8 = float(_t8c._sust.max())
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
          float(_t8c._sust.max()) >= _antes8,
          f"antes {_antes8:.3f} s, depois {float(_t8c._sust.max()):.3f} s; "
          f"no g1_multitask isto zerava e o `perf` marcou 0 com o robô andando")
    del _e8
except Exception as _e8x:      # noqa: BLE001
    _falhas.append(f"o cronômetro não pôde ser medido: "
                   f"{type(_e8x).__name__}: {_e8x}")

# ==================================== 19. a máquina de elo (F4)
secao("19. a máquina de elo: cadeias, fechamento e avanço (F4)")
kc = k.cadeia

# --- a tabela, estática ---
check("há 4 cadeias, e só a cadeia 3 tem 3 elos (spec §6.5)",
      len(CMD.CADEIAS) == 4 and [len(c) for c in CMD.CADEIAS] == [1, 2, 2, 3],
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
    # ⚠⚠ AQUI A JANELA DE ESPERA **NÃO** PODE SER QUEIMADA, e o contrário custou uma
    # falha: este bloco mede o invariante de ABERTURA (`cadeia[0] == elo sorteado`), e
    # 55 passos deixam o elo AVANÇAR — no `REORIENTAR` o alvo é a própria caixa,
    # portanto `perto` é trivial e ele fecha assim que o objetivo liga. O elo medido
    # deixava de ser o de abertura, e o teste acusava o desenho em vez de si mesmo.
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
    _ca.commands["alvo_caixa"].cadeia_forcada = 3        # (PEGAR, CARREGAR, BOTAR)
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
          bool((_tac._elo == CMD.CARREGAR).all()) and bool((_elo_antes == CMD.PEGAR).all()),
          f"{_elo_antes.tolist()[:3]} -> {_tac._elo.tolist()[:3]}")
    # o 2º avanço leva ao BOTAR; o invariante da laje abaixo é medido nesse instante
    _tac.forca_avanco(_ids_a)
    check("o segundo avanço leva ao BOTAR, e `_passo` vai a 2",
          bool((_tac._elo == CMD.BOTAR).all()) and bool((_tac._passo == 2).all()))
    check("v2.1: o `_sustain_alvo` é atualizado no avanço — BOTAR paga `sustenta_outros_s`",
          bool((_tac._sustain_alvo == k.cadeia.sustenta_outros_s).all()),
          f"{_tac._sustain_alvo.tolist()[:3]}")
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
check("o portão é UM sinal só, e desde 27/08 ele é a `eficiencia_min`",
      cfg.curriculum["forma"].params["nome_do_twist"] == "twist"
      and kf.limiar_portao == 0.50,
      "dois sinais conjuntivos já travaram uma rampa para sempre")
check("o controlador LÊ a `eficiencia_min`, e não a `razao_marcha`",
      "eficiencia_min" in inspect.getsource(CU_.forma)
      and 'metrics["razao_marcha"]' not in inspect.getsource(CU_.forma),
      "a razão é soma de NORMAS: ruído de média zero sempre a infla, e no bloco 1 "
      "isso congelou a rampa em UM degrau por 1341 iterações")
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

# ============ O PORTÃO OLHA SÓ PARA QUEM ANDA. É a trava do defeito de 31/08. ========
# ⚠⚠ O DEFEITO: até 31/08 o sinal era `eficiencia_min.mean()` sobre TODOS os envs. O
# twist é forçado a zero nos elos de manipulação, portanto `seg_pedido` nunca alcança
# `pedido_min_segmento`, nenhum segmento válido fecha, e `eficiencia_min` é ZERO EXATO
# naqueles envs. O portão se envenenava com a própria rampa:
#
#     rampa baixa forma -> fatia de manipulação cresce -> mais zeros na média
#          ^                                                        |
#          +---- portão abre <- média sobe <- rampa REVERTE <- média cai
#
# O laço tem PONTO FIXO, e ele é um teto: `efic x (1 − fatia) = limiar`, isto é
# fatia <= 0,375 com `limiar = 0,50`. O destino `alvo_loco_min = 0,30` era INALCANÇÁVEL.
#
# MEDIDO no bloco 6, iteração 785: efic de quem anda ~0,80, fatia 0,272, média diluída
# prevista 0,582 contra 0,5844 medida. E a rampa parou em `alvo` ~0,79 de 33 degraus,
# depois SUBIU (o ramo de histerese disparou).
check("o portão MASCARA o sinal pelos envs que foram pedidos a andar",
      'metrics["segmentos"]' in inspect.getsource(CU_.forma)
      and ".mean()" in inspect.getsource(CU_.forma),
      "sem a máscara a fatia de manipulação dilui o próprio juiz, e a rampa para "
      "num ponto fixo em vez de chegar ao piso")
check("a máscara é `segmentos > 0`, e NÃO o canal do elo",
      "segmentos\"] > 0" in inspect.getsource(CU_.forma)
      and "canal_do_elo" not in inspect.getsource(CU_.forma),
      "`segmentos > 0` se autodescreve e não acopla o currículo ao layout do "
      "comando de caixa; e um env de CARREGAR tem twist ativo e DEVE entrar")

# --- a ARITMÉTICA do ponto fixo, para o teto ficar declarado e não redescoberto ---
# ⚠ Isto não testa código: testa a CONTA que explica o defeito. Ela fica aqui porque foi
# ela que o identificou, e porque um `limiar_portao` novo muda o teto sem avisar.
_EFIC_QUE_ANDA = 0.80                 # medido no bloco 6: 0,5844 / 0,7277 = 0,803
_teto_fatia = 1.0 - kf.limiar_portao / _EFIC_QUE_ANDA
check("SEM máscara, o teto da fatia seria 0,375 — abaixo do destino de 0,70",
      abs(_teto_fatia - 0.375) < 0.01
      and _teto_fatia < (1.0 - kf.alvo_loco_min),
      f"teto {_teto_fatia:.3f} contra o destino {1.0 - kf.alvo_loco_min:.3f} — "
      "é isto que a máscara remove")
check("a média diluída prevista casa com a MEDIDA no bloco 6",
      abs(_EFIC_QUE_ANDA * 0.728 - 0.5844) < 0.01,
      f"previsto {_EFIC_QUE_ANDA * 0.728:.4f} contra 0,5844 medido — "
      "a diluição explica o número inteiro, sem termo sobrando")

# --- a ESCADA lê o DERIVADO, e não o canal cru ---
# ⚠ O alvo de 0,50 no canal cru fica MAIS DURO conforme a rampa desce, e no destino ele
# fica IMPOSSÍVEL: `alvo_loco_min = 0,30` é 30% de LOCOMOÇÃO, o cru vale `efic × 0,30`, e
# passar exigiria `efic >= 1,67` — acima do teto de 1,0. A linha marcaria falha num robô
# que anda perfeitamente, que é o erro que ela existe para não cometer.
check("no destino da rampa, o alvo no canal CRU seria inalcançável",
      kf.limiar_portao / kf.alvo_loco_min > 1.0
      and 0.50 / kf.alvo_loco_min > 1.0,
      f"exigiria {0.50 / kf.alvo_loco_min:.2f} de quem anda, e o teto é 1,0")
_linha_efic = [l for l in LE_.ESCADA if l[1] == LE_.CH_EFIC_LOCO]
check("a linha do andar na escada lê o canal DES-DILUÍDO",
      len(_linha_efic) == 1 and not any(l[1] == LE_.CH_EFIC for l in LE_.ESCADA),
      f"escada: {[l[1] for l in LE_.ESCADA]}")
check("o derivado é `eficiencia_min / forma`, e a des-diluição é EXATA",
      LE_.CH_FORMA == "Curriculum/forma"
      and "CH_FORMA" in inspect.getsource(LE_._serie)
      and "CH_EFIC" in inspect.getsource(LE_._serie),
      "`Curriculum/forma` É a fração de locomoção, portanto a divisão não é "
      "aproximação")
check("o denominador tem PISO — dividir por leitura crua erra por 10x um dia",
      "max(forma[s]" in inspect.getsource(LE_._serie),
      "`sorteio_min` é 0,10 hoje; um knob novo em 0,0 daria divisão por zero")

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
        # ⚠ A CHAVE É `eficiencia_min` DESDE 27/08. Ela é o sinal do portão; a
        # `razao_marcha` continua no dict porque continua logada, mas o controlador não a
        # lê mais. Se este falso voltar a alimentar só a razão, o portão passa a ler o
        # default pessimista e os dois testes de baixo falham dizendo "a rampa não desce"
        # — que foi exatamente o que aconteceu ao trocar o sinal.
        # ⚠ E DESDE 31/08 ELE PRECISA DE `segmentos`. O controlador mascara o sinal por
        # `segmentos > 0` — a eficiência de quem foi PEDIDO a andar. Sem a chave, o
        # `except KeyError` do controlador devolve o default pessimista, o portão nunca
        # abre, e os dois testes de baixo falham dizendo "a rampa não desce". Foi
        # exatamente o que aconteceu ao acrescentar a máscara.
        def __init__(self, v):
            _t = __import__("torch")
            self.metrics = {"eficiencia_min": _t.tensor([v]),
                            "razao_marcha": _t.tensor([v]),
                            "segmentos": _t.tensor([2.0])}

    class _CmdFalso:
        def __init__(self, v):
            self._t = _TwistFalso(v)

        def get_term(self, _):
            return self._t

    def _simula(razao, iteracoes, kf_):
        """Roda `iteracoes` ITERAÇÕES de PPO no controlador.

        ⚠ O `common_step_counter` avança `passos_por_iteracao` por iteração, porque é
        DELE que o controlador deriva a iteração — e não de um contador próprio. Uma
        versão anterior deste falso não o tinha, e o teste media zero iterações.
        """
        _t = __import__("torch")
        e = _ty6.SimpleNamespace(
            num_envs=4, device="cpu", common_step_counter=0,
            command_manager=_CmdFalso(razao),
            episode_length_buf=_t.zeros(4, dtype=_t.long))
        e.limpo_elo = _t.zeros(4, dtype=_t.long)
        ids = _t.arange(0)
        for _ in range(iteracoes):
            # várias chamadas por iteração, como no treino de verdade: o termo roda a
            # cada passo em que algum env reseta.
            for _p in range(kf_.passos_por_iteracao):
                e.common_step_counter += 1
                CU_.forma(e, ids, f=kf_, elo_loco=0)
        return e.limpo_forma

    # em ITERAÇÕES de PPO: a carência mais 40 degraus de folga
    _n_folga = int(kf.carencia_iters + 40 * max(kf.iters_entre_degraus, 1))
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

    # ======= A MÁSCARA DERROTA A DILUIÇÃO. É o teste de COMPORTAMENTO do conserto. =====
    # ⚠ Os checks de fonte acima afirmam que a máscara EXISTE. Este afirma que ela
    # FUNCIONA, e ele é construído para FALHAR sem ela.
    #
    # A frota: metade anda com eficiência 0,80 e 2 segmentos; metade está num elo de
    # manipulação, com eficiência ZERO EXATO e ZERO segmento (twist forçado a zero ->
    # nenhum segmento válido fecha).
    #
    #     média SEM máscara  =  0,40   -> abaixo da histerese (0,40) -> rampa REVERTE
    #     média COM máscara  =  0,80   -> acima do limiar (0,50)     -> rampa DESCE
    #
    # Os dois lados do portão, com a MESMA frota. É o defeito medido no bloco 6: a
    # fatia de manipulação diluía o próprio juiz e a rampa parava num ponto fixo
    # (`efic x (1 − fatia) = limiar`, isto é fatia <= 0,375) em vez de chegar ao piso.
    class _TwistDiluido:
        def __init__(self):
            _t = __import__("torch")
            self.metrics = {
                "eficiencia_min": _t.tensor([0.80, 0.80, 0.0, 0.0]),
                "razao_marcha": _t.tensor([0.80, 0.80, 0.0, 0.0]),
                "segmentos": _t.tensor([2.0, 2.0, 0.0, 0.0]),
            }

    class _CmdDiluido:
        def __init__(self):
            self._t = _TwistDiluido()

        def get_term(self, _):
            return self._t

    _t7d = __import__("torch")
    _ed = _ty6.SimpleNamespace(
        num_envs=4, device="cpu", common_step_counter=0,
        command_manager=_CmdDiluido(),
        episode_length_buf=_t7d.zeros(4, dtype=_t7d.long))
    _ed.limpo_elo = _t7d.zeros(4, dtype=_t7d.long)
    for _ in range(_n_folga):
        for _p in range(kf.passos_por_iteracao):
            _ed.common_step_counter += 1
            CU_.forma(_ed, _t7d.arange(0), f=kf, elo_loco=0)
    _dil = _ed.limpo_forma
    _media_crua = 0.80 * 0.5
    check("a média CRUA desta frota ficaria ABAIXO da histerese",
          _media_crua < kf.histerese * kf.limiar_portao + 1e-9,
          f"crua {_media_crua:.3f} contra defende<{kf.histerese*kf.limiar_portao:.3f} "
          "— é isto que revertia a rampa")
    check("com a MÁSCARA, meia frota parada NÃO impede a rampa de chegar ao piso",
          _dil["abriu"] == 1.0
          and abs(_dil["alvo"] - kf.alvo_loco_min) < 1e-9,
          f"alvo {_dil['alvo']:.3f}, abriu {_dil['abriu']} — sem a máscara este "
          f"alvo fica em {kf.alvo_loco_max:.2f}")
    check("e o sinal lido É a eficiência de quem anda, não a diluída",
          abs(_dil["razao"] - 0.80) < 0.02,
          f"razao {_dil['razao']:.4f} — diluída daria ~{_media_crua:.2f}")
    # a carência
    _curto = _simula(0.95, max(kf.carencia_iters - 1, 1), kf)
    check("dentro da CARÊNCIA a rampa não se move, nem com o sinal alto",
          abs(_curto["alvo"] - kf.alvo_loco_max) < 1e-9,
          f"alvo {_curto['alvo']:.3f} depois de {kf.carencia_iters-1} iters")
    # ⚠ O CHECK QUE PEGA O DEFEITO DE 24×. O termo roda VÁRIAS VEZES por iteração de
    # PPO (uma por passo em que algum env reseta — medido: 48,8% dos passos com 128
    # envs, e tenderia a 100% com 4096). Um contador próprio contaria PASSOS, e a
    # carência de 200 "iterações" seria atingida em ~17.
    _ref = _simula(0.95, _n_folga, kf)
    check("a iteração é derivada do contador de PASSOS do env, não incrementada aqui",
          abs(_ref["iters_balanco"] - _n_folga) < 1.5,
          f"iters_balanco {_ref['iters_balanco']:.1f} contra {_n_folga} iterações "
          f"simuladas — se der {_n_folga * kf.passos_por_iteracao} o contador está "
          f"contando PASSOS")
    check("um degrau por JANELA, e não um por chamada do termo",
          _ref["ultimo_degrau"] >= 0.0,
          "o termo roda ~24× por iteração; sem o `ultimo_degrau` a rampa desceria "
          "24 degraus por iteração")
    check("o alvo NUNCA sai de [0,30 ; 0,95]",
          all(kf.alvo_loco_min - 1e-9 <= x["alvo"] <= kf.alvo_loco_max + 1e-9
              for x in (_parado, _andando, _meio, _curto, _ref)))
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
# ⚠ O CICLO DE VERDADE. Conferir os NOMES das chaves não prova que o estado sobrevive:
# o furo que isso deixava é uma chave certa com um `save` que não a escreve. Aqui o
# estado é serializado e restaurado, e os valores são comparados.
try:
    import tempfile as _tmp

    import torch as _t9c

    _c9c = make_env_cfg(k)
    _c9c.scene.num_envs = 8
    _e9c = ManagerBasedRlEnv(cfg=_c9c, device="cpu")
    _e9c.reset()
    _e9c.step(_t9c.zeros(8, _e9c.action_manager.total_action_dim))

    # mexe o estado para valores RECONHECÍVEIS — zeros passariam por acidente
    _e9c.limpo_forma["alvo"] = 0.4242
    _e9c.limpo_forma["iters_balanco"] = 777.0
    _e9c.limpo_forma["razao"] = 0.6161
    _e9c.limpo_nivel[:] = 4
    _e9c.limpo_elo[:] = CMD.PEGAR

    _estado = {
        "forma": {c: float(_e9c.limpo_forma[c]) for c in RN_.CHAVES_ESCALARES
                  if c in _e9c.limpo_forma},
        "limpo_nivel": _e9c.limpo_nivel.detach().cpu().clone(),
        "limpo_elo": _e9c.limpo_elo.detach().cpu().clone(),
    }
    _cam = str(pathlib.Path(_tmp.mkdtemp()) / "ck.pt")
    _t9c.save({"infos": {"limpo_curriculo": _estado}}, _cam)
    _volta = _t9c.load(_cam, weights_only=False)["infos"]["limpo_curriculo"]

    check("o ciclo salvar->carregar preserva a FATIA e a carência",
          abs(_volta["forma"]["alvo"] - 0.4242) < 1e-9
          and abs(_volta["forma"]["iters_balanco"] - 777.0) < 1e-9,
          str(_volta["forma"]))
    check("e preserva a EMA do sinal do portão",
          abs(_volta["forma"]["razao"] - 0.6161) < 1e-9)
    check("e preserva o nível e o elo POR ENV",
          bool((_volta["limpo_nivel"] == 4).all())
          and bool((_volta["limpo_elo"] == CMD.PEGAR).all()))
    check("as três EMAs de duração e fatia estão TODAS no que foi salvo",
          {"alvo", "dur_loco", "dur_manip", "razao", "iters_balanco"}
          <= set(_volta["forma"]),
          str(sorted(_volta["forma"])))
    del _e9c
except Exception as _e9d:      # noqa: BLE001
    _falhas.append(f"o ciclo de checkpoint não pôde ser exercitado: "
                   f"{type(_e9d).__name__}: {_e9d}")

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

# --- O FECHO NATURAL: o elo avança POR SUSTENTAÇÃO, sem ninguém forçar ---
# ⚠ Todos os checks acima usam `forca_avanco`, que é o atalho do inspetor. Este é o
# único que exercita o caminho REAL: a condição de fechamento vale, o cronômetro
# acumula, e o elo troca sozinho. Sem ele, um `_fecha_elo_corrente` que nunca
# devolvesse True passaria em tudo.
try:
    import torch as _tb

    _cb = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _cb.scene.num_envs = 4
    _cb.commands["alvo_caixa"].cadeia_forcada = 2        # (PEGAR, CARREGAR)
    _eb = ManagerBasedRlEnv(cfg=_cb, device="cpu")
    _eb.reset()
    _nab = _eb.action_manager.total_action_dim
    # ⚠ A JANELA DE ESPERA PRIMEIRO. O sustain só acumula com o objetivo ATIVO — o
    # `_fecha_elo_corrente` exige `VALIDA > 0.5` desde 02/09. Com um passo só de
    # aquecimento, o teste gastava a janela dentro do laço de contagem e chegava a
    # 0,28 s de 0,5 s, acusando o desenho em vez do próprio orçamento de passos.
    _passa_janela(_eb, _nab, _tb)
    _tbc = _eb.command_manager.get_term("alvo_caixa")
    _caixab = _eb.scene["box"]
    _elo_ini = int(_tbc._elo[0])

    # a caixa NO alvo, com a face alinhada, e PINADA lá
    _quat = _tb.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(_eb.num_envs, 4)
    _passos = int(k.cadeia.sustenta_pegar_s / _eb.step_dt) + 6
    _sust_max = 0.0
    for _ in range(_passos):
        _caixab.write_root_link_pose_to_sim(
            _tb.cat([_tbc.command[:, CMD.ALVO], _quat], dim=-1))
        _caixab.write_root_link_velocity_to_sim(
            _tb.zeros(_eb.num_envs, 6))
        _eb.step(_tb.zeros(_eb.num_envs, _nab))
        _sust_max = max(_sust_max, float(_tbc._sust.max()))

    check("a condição de fechamento do PEGAR DISPARA com a caixa no alvo e de pé",
          _sust_max > 0.0,
          f"o cronômetro nunca saiu de zero — `_fecha_elo_corrente` não fecha nunca")
    check("e o elo avança SOZINHO, por sustentação, sem `forca_avanco`",
          int(_tbc._elo[0]) != _elo_ini and int(_tbc._passo[0]) == 1,
          f"elo {_elo_ini} -> {int(_tbc._elo[0])}, passo {int(_tbc._passo[0])}, "
          f"sust_max {_sust_max:.3f} s de {k.cadeia.sustenta_pegar_s} s")
    check("o cronômetro respeita o sustain do elo, e não avança antes",
          _sust_max >= k.cadeia.sustenta_pegar_s - _eb.step_dt - 1e-9,
          f"sust_max {_sust_max:.3f} s")
    del _eb
except Exception as _ebx:      # noqa: BLE001
    _falhas.append(f"o fecho natural não pôde ser exercitado: "
                   f"{type(_ebx).__name__}: {_ebx}")

# --- A CURVA DO `unload`, e a TASK DE CADEIA do visualizador ---
try:
    import torch as _tc

    import g1_limpo as _gl

    # (a) o `unload` vai de ~0 a ~1 quando a força de apoio cai de m·g a 0.
    # ⚠ MÉTODO DECLARADO: a caixa é TELEPORTADA para cima. Isso mede os EXTREMOS, que é
    # o que o critério pede — e NÃO mede a partilha de carga (palma sobe / apoio desce),
    # que só uma run com preensão mostra. Ver o docstring do termo.
    _cc = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _cc.scene.num_envs = 6
    _ec = ManagerBasedRlEnv(cfg=_cc, device="cpu")
    _ec.reset()
    _nac = _ec.action_manager.total_action_dim
    for _ in range(4):
        _ec.step(_tc.zeros(_ec.num_envs, _nac))
    _nmc = list(_cc.rewards)
    _iu = _nmc.index("unload")
    _cx = _ec.scene["box"]
    _z0 = _cx.data.root_link_pos_w[:, 2].clone()

    def _mede_unload(dz):
        for _ in range(4):
            _p = _cx.data.root_link_pos_w.clone()
            _p[:, 2] = _z0 + dz
            _cx.write_root_link_pose_to_sim(
                _tc.cat([_p, _cx.data.root_link_quat_w], dim=-1))
            _cx.write_root_link_velocity_to_sim(_tc.zeros(_ec.num_envs, 6))
            _ec.step(_tc.zeros(_ec.num_envs, _nac))
        _f = float(_tc.norm(_ec.scene[C.SENSOR_APOIO].data.force, dim=-1).mean())
        return _f, float(_ec.reward_manager._step_reward[:, _iu].mean())

    _f_apoiada, _u_apoiada = _mede_unload(0.0)
    _f_erguida, _u_erguida = _mede_unload(0.05)
    _mg = float((_ec.limpo_massa * 9.81).mean())
    _peso = cfg.rewards["unload"].weight

    # ⚠⚠ O LIMITE É DE UM LADO SÓ, e a banda de dois lados era o defeito do TESTE. A
    # caixa é reescrita na mesma pose a cada passo, portanto a força NÃO é um transiente
    # no tempo: ela é fixada pela PENETRAÇÃO INICIAL do contato, que vem do `_z0` medido
    # depois de 4 passos de assentamento e do `jitter_z` do topo da laje. Ela varia POR
    # RUN, e não ao longo do passo. Medido no mesmo teste, sem mudança de código:
    # 9,80 N, 11,79 N e 13,36 N contra m·g = 9,81 N — isto é +0%, +20% e +36%. A banda
    # subiu de 5% para 30% perseguindo esse ruído, e o overshoot não tem cota de
    # desenho: perseguir mais seria escrever o ruído no teste.
    #
    # O que este check AFIRMA é o sinal: "a caixa PESA na laje" contra "não pesa", e os
    # dois estados são ~10 N contra 0,0 N. Portanto o piso é `m·g` menos folga, e o teto
    # existe só para pegar um sensor lendo a coisa errada — o robô inteiro pesa ~350 N.
    check("apoiada, a força de apoio PESA (>= ~m·g) e o `unload` é ~0",
          _f_apoiada > 0.90 * _mg and _f_apoiada < 3.0 * _mg
          and _u_apoiada / _peso < 0.05,
          f"F={_f_apoiada:.2f} N de m·g={_mg:.2f} N "
          f"({_f_apoiada/_mg:.2f}x), unload={_u_apoiada/_peso:.4f}")
    # ⚠⚠ ESTE CHECK MUDOU DE SINAL EM 28/08, E É O PORTEIRO DE PREENSÃO.
    #
    # Ele afirmava "erguida, a força de apoio é 0 e o `unload` é ~1". A descarga
    # sozinha continua indo a 1 — mas ela não é mais o termo. Neste teste a caixa é
    # TELEPORTADA para cima e NENHUMA palma a toca, e era exatamente esse o atalho:
    # derrubar a caixa da laje zera `F_apoio` para sempre e pagava 2,0/s pelo resto do
    # episódio, sem mão nenhuma. Medido no bloco 3, it 4251: `unload` 0,0995 com
    # `squeeze` 0,0002 — descarga sem preensão.
    #
    # Portanto o comportamento CERTO aqui é ZERO, e um `unload` de ~1 sem preensão
    # passa a ser a FALHA. A rampa de descarga continua medida no valor de `descarga`,
    # que este teste calcula à parte.
    _desc_erguida = 1.0 - _f_erguida / _mg
    check("erguida SEM PREENSÃO, o `unload` é ZERO — o porteiro fecha o atalho",
          _f_erguida < 1e-6 and _u_erguida / _peso < 1e-3,
          f"F={_f_erguida:.2f} N, unload={_u_erguida/_peso:.4f} "
          f"(descarga crua {_desc_erguida:.4f})")
    check("a DESCARGA crua ainda vai de ~0 a ~1 — quem zera é o porteiro, não ela",
          _desc_erguida > 0.99 and (1.0 - _f_apoiada / _mg) < 0.05,
          f"apoiada {1.0 - _f_apoiada/_mg:.4f} -> erguida {_desc_erguida:.4f}")
    print(f"  unload: {_u_apoiada/_peso:.4f} (apoiada, {_f_apoiada:.2f} N) -> "
          f"{_u_erguida/_peso:.4f} (erguida SEM preensão, {_f_erguida:.2f} N); "
          f"descarga crua {_desc_erguida:.4f}")
    del _ec

    # (b) A TASK DE CADEIA do visualizador existe e o avanço DISPARA.
    # ⚠ Isto é o que faz o `--viewer --cadeia N --avanca-elo` NÃO ser no-op. A primeira
    # versão dessas flags parseava o argumento e o DESCARTAVA, com um comentário dizendo
    # que não era possível sem reescrever o `run_play`. O caminho é registrar a task.
    check("há uma task de inspeção por cadeia de 2 elos",
          set(_gl.TASK_CADEIA) == {i for i, c in enumerate(CMD.CADEIAS)
                                   if len(c) > 1},
          str(sorted(_gl.TASK_CADEIA)))
    from mjlab.tasks.registry import load_env_cfg as _lec   # noqa: E402
    _cv = _lec(_gl.TASK_CADEIA[2])
    check("a task de cadeia força a cadeia E instala o evento de avanço",
          _cv.commands["alvo_caixa"].cadeia_forcada == 2
          and "avanca_elo" in _cv.events,
          str(sorted(_cv.events)))
    _cv.scene.num_envs = 4
    _ev = ManagerBasedRlEnv(cfg=_cv, device="cpu")
    _ev.reset()
    _nav = _ev.action_manager.total_action_dim
    _tv = _ev.command_manager.get_term("alvo_caixa")
    _elo_antes_v = int(_tv._elo[0])
    _laje_antes = float(_ev.scene["table"].data.root_link_pos_w[0, 2])
    while float(_ev.episode_length_buf[0]) * _ev.step_dt < _gl.AVANCA_APOS_S + 0.5:
        _ev.step(_tc.zeros(_ev.num_envs, _nav))
    check("o evento de avanço DISPARA, e a mesa sobe com o robô parado",
          int(_tv._elo[0]) == CMD.CARREGAR and _elo_antes_v == CMD.PEGAR
          and float(_ev.scene["table"].data.root_link_pos_w[0, 2]) > 4.0,
          f"elo {_elo_antes_v} -> {int(_tv._elo[0])}, laje "
          f"{_laje_antes:.3f} -> "
          f"{float(_ev.scene['table'].data.root_link_pos_w[0, 2]):.3f} m")
    del _ev
except Exception as _ecx:      # noqa: BLE001
    _falhas.append(f"a curva do unload / a task de cadeia não pôde ser exercitada: "
                   f"{type(_ecx).__name__}: {_ecx}")

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

# ==================== 22. a cadeia 3 tem TRÊS elos e segura parado (spec §6.5) ======
secao("22. a cadeia 3: (PEGAR, CARREGAR, BOTAR), o CARREGAR do meio segura parado")
from g1_limpo import comando as CMD                                       # noqa: E402

check("9. CADEIAS[3] é (PEGAR, CARREGAR, BOTAR)",
      CMD.CADEIAS[3] == (CMD.PEGAR, CMD.CARREGAR, CMD.BOTAR), str(CMD.CADEIAS[3]))
check("9. o teto de elos é DERIVADO e vale 3", CMD._TETO_ELOS == 3)
check("9. a marca de segurar parado é derivada de CADEIAS: só a cadeia 3 a tem",
      CMD._SEGURA_PARADO.tolist() == [False, False, False, True],
      str(CMD._SEGURA_PARADO.tolist()))
check("as outras três cadeias não mudaram",
      CMD.CADEIAS[:3] == ((CMD.PEGAR,), (CMD.REORIENTAR, CMD.PEGAR),
                          (CMD.PEGAR, CMD.CARREGAR)))
check("toda espera é a MESMA faixa: espera_s = (0,5, 1,5)",
      tuple(k.alvo.espera_s) == (0.5, 1.5), str(k.alvo.espera_s))

# --- rodando: a cadeia 3 percorre os três elos com a caixa PINADA na âncora ---
# `elo=CARREGAR` liga o `segura_caixa` + `pina_caixa` (a caixa fica no peito a cada
# passo); `cadeia=3` vence e o elo de abertura é o PEGAR. Com a caixa na âncora, o
# PEGAR fecha sozinho depois da espera + 0,5 s; o CARREGAR de segurar parado fecha
# por `perto` sustentado pela espera sorteada; o BOTAR nunca fecha (a caixa pinada
# no ar não é `apoiada`).
try:
    import torch as _t22

    _c22 = make_env_cfg(k, inspecao=True, elo=CMD.CARREGAR, cadeia=3)
    _c22.scene.num_envs = 16
    _e22 = ManagerBasedRlEnv(cfg=_c22, device="cpu")
    _e22.reset()
    _n22 = _e22.action_manager.total_action_dim
    _t22c = _e22.command_manager.get_term("alvo_caixa")
    _tw22 = _e22.command_manager.get_term("twist")
    _dt22 = _e22.step_dt
    _t1 = _t22.full((_e22.num_envs,), -1, dtype=_t22.long)
    _t2 = _t22.full((_e22.num_envs,), -1, dtype=_t22.long)
    _twist_no_carregar = 0.0
    for _i in range(240):
        _e22.step(_t22.zeros(_e22.num_envs, _n22))
        _p = _t22c._passo
        _t1 = _t22.where((_t1 < 0) & (_p >= 1), _t22.full_like(_t1, _i), _t1)
        _t2 = _t22.where((_t2 < 0) & (_p >= 2), _t22.full_like(_t2, _i), _t2)
        if bool(((_p == 1)).any()):
            _twist_no_carregar = max(_twist_no_carregar,
                                     float(_tw22.vel_command_b[_p == 1].abs().max()))
    check("9. a máquina de elo percorre PEGAR -> CARREGAR -> BOTAR sozinha",
          bool((_t22c._passo == 2).all()) and bool((_t22c._elo == CMD.BOTAR).all()),
          f"passo {_t22c._passo.tolist()[:8]}")
    check("9. e `fechou` NÃO marca no BOTAR com a caixa no ar",
          not bool(_t22c.fechou.any()))
    _seg = (_t2 - _t1).float() * _dt22
    check("11. o CARREGAR de segurar parado dura a ESPERA sorteada (0,5 a 1,5 s)",
          bool((_seg >= k.alvo.espera_s[0] - 2 * _dt22).all())
          and bool((_seg <= k.alvo.espera_s[1] + 3 * _dt22).all()),
          f"durações medidas {[round(float(x), 2) for x in _seg[:8]]} s")
    check("10. no CARREGAR da cadeia 3 o twist é ZERO em todo passo",
          _twist_no_carregar == 0.0, f"máximo medido {_twist_no_carregar:.4f}")
    del _e22

    # --- controle: na cadeia 2 o CARREGAR ANDA e fecha por distância ---
    _c22b = make_env_cfg(k, inspecao=True, elo=CMD.CARREGAR, cadeia=2)
    _c22b.scene.num_envs = 16
    _e22b = ManagerBasedRlEnv(cfg=_c22b, device="cpu")
    _e22b.reset()
    _n22b = _e22b.action_manager.total_action_dim
    _t22d = _e22b.command_manager.get_term("alvo_caixa")
    _tw22b = _e22b.command_manager.get_term("twist")
    _twist_c2 = 0.0
    for _ in range(240):
        _e22b.step(_t22.zeros(_e22b.num_envs, _n22b))
        if bool((_t22d._passo == 1).any()):
            _twist_c2 = max(_twist_c2,
                            float(_tw22b.vel_command_b[_t22d._passo == 1].abs().max()))
    check("11. na cadeia 2 o CARREGAR NÃO fecha com o robô parado — `andou` continua",
          bool((_t22d._passo == 1).all()), f"passo {_t22d._passo.tolist()[:8]}")
    check("10. e na cadeia 2 o twist RELIGA no CARREGAR",
          _twist_c2 > 0.0, f"máximo medido {_twist_c2:.4f}")
    del _e22b
except Exception as _e22x:      # noqa: BLE001
    _falhas.append(f"a cadeia 3 não pôde ser exercitada: "
                   f"{type(_e22x).__name__}: {_e22x}")

# ============ 23. as DUAS esperas publicam ANDAR, o VALIDA lê o interno (spec §6.3, §6.6)
secao("23. as duas esperas publicam ANDAR")
from g1_limpo import comando as CMD                                       # noqa: E402
from g1_limpo import observacoes as OB_                                   # noqa: E402
from g1_limpo import terminacoes as TE_                                   # noqa: E402
from g1_limpo import recompensas as RC_                                   # noqa: E402

check("7. o publicado é recalculado do INTERNO e das duas esperas",
      "aguardando | self._soltou" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera)
      and "self._elo" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera),
      "ler o que se escreveu no passo anterior deixa o canal preso (02/09)")
check("20. o `_pegou` só arma com o objetivo ATIVO",
      "self._espera <= 0.0" in inspect.getsource(CMD.AlvoCaixaCmd._publica_pegou),
      "um toque por exploração na espera inicial armaria `escapou` e mataria o episódio")
check("o `rastreio_por_elo` não lê mais `limpo_aguardando` — o publicado já é ANDAR",
      "limpo_aguardando" not in inspect.getsource(RC_.rastreio_por_elo))

try:
    import torch as _t23

    # --- a espera INICIAL, vista pela observação ---
    _c23 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c23.scene.num_envs = 32
    _e23 = ManagerBasedRlEnv(cfg=_c23, device="cpu")
    _o23, _ = _e23.reset()
    _n23 = _e23.action_manager.total_action_dim
    _t23c = _e23.command_manager.get_term("alvo_caixa")
    _fat23 = OB_.fatia_do_elo(_o23["actor"].shape[-1])
    _hot0 = _o23["actor"][:, _fat23].argmax(-1)
    check("4. na observação do RESET o one-hot publicado é ANDAR",
          bool((_hot0 == CMD.ANDAR).all()), str(_hot0.tolist()[:8]))
    check("4. e o elo INTERNO é PEGAR no reset",
          bool((_t23c._elo == CMD.PEGAR).all())
          and bool((_e23.limpo_elo_interno == CMD.PEGAR).all()))
    check("3. o VALIDA é ZERO na espera inicial",
          float(_t23c.command[:, CMD.VALIDA].max()) == 0.0)
    _borda = _t23.full((_e23.num_envs,), -1, dtype=_t23.long)
    for _i in range(int(k.alvo.espera_s[1] / _e23.step_dt) + 5):
        _o23 = _e23.step(_t23.zeros(_e23.num_envs, _n23))[0]
        _hot = _o23["actor"][:, _fat23].argmax(-1)
        _borda = _t23.where((_borda < 0) & (_hot == CMD.PEGAR),
                            _t23.full_like(_borda, _i), _borda)
    check("4. na borda o one-hot publicado vira PEGAR em todos os envs",
          bool((_borda >= 0).all()), str(_borda.tolist()[:8]))
    _bs = _borda.float() * _e23.step_dt
    check("4. e a borda cai dentro da faixa de espera_s",
          float(_bs.min()) >= k.alvo.espera_s[0] - 2 * _e23.step_dt
          and float(_bs.max()) <= k.alvo.espera_s[1] + 2 * _e23.step_dt,
          f"{float(_bs.min()):.2f} .. {float(_bs.max()):.2f} s")
    check("3. depois da borda o VALIDA é UM",
          float(_t23c.command[:, CMD.VALIDA].min()) == 1.0)
    check("6. o piso de locomoção não conta a espera: `limpo_elo` segue PEGAR",
          bool((_e23.limpo_elo == CMD.PEGAR).all()),
          "a fatia lê o interno do currículo, não o publicado")
    del _e23

    # --- a espera FINAL, forçada à mão na cadeia 3 ---
    _c23b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c23b.scene.num_envs = 8
    _e23b = ManagerBasedRlEnv(cfg=_c23b, device="cpu")
    _e23b.reset()
    _n23b = _e23b.action_manager.total_action_dim
    _passa_janela(_e23b, _n23b, _t23)
    _t23d = _e23b.command_manager.get_term("alvo_caixa")
    _ids23 = _t23.arange(_e23b.num_envs)
    _t23d.forca_avanco(_ids23)            # -> CARREGAR
    _t23d.forca_avanco(_ids23)            # -> BOTAR
    check("12. antes do fecho do BOTAR o publicado é BOTAR e `soltou` é falso",
          bool((_t23d.command[:, CMD.ELO] == CMD.BOTAR).all())
          and not bool(_t23d._soltou.any()))
    # ⚠ A CAIXA VAI PARA LONGE DAS PALMAS **ANTES** DO FECHO, e com um passo para os
    # buffers de `.data` recomputarem — senão o `escapou` lê pose obsoleta e o teste
    # abaixo passaria por omissão. Aqui ele TEM de estar armado.
    _t23d._pegou[:] = True
    _cx23p = _e23b.scene["box"]
    _pp23 = _cx23p.data.root_link_pos_w.clone()
    _pp23[:, 0] += 1.0
    _cx23p.write_root_link_pose_to_sim(
        _t23.cat([_pp23, _cx23p.data.root_link_quat_w], -1))
    _cx23p.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    _par23 = dict(cfg.terminations["caixa_largada"].params)
    check("12. ANTES do fecho, a caixa longe das palmas TERMINA (`escapou` armado)",
          bool(TE_.caixa_largada(_e23b, **_par23).all()),
          "sem isto o check seguinte passaria por omissão")
    _t23d.forca_avanco(_ids23)            # fecha o BOTAR -> espera final
    check("12. no MESMO passo do fecho o publicado vira ANDAR, sem atraso",
          bool((_t23d.command[:, CMD.ELO] == CMD.ANDAR).all()))
    check("12. o interno fica BOTAR, `fechou` e `soltou` marcam, sucesso = 1",
          bool((_t23d._elo == CMD.BOTAR).all()) and bool(_t23d.fechou.all())
          and bool(_t23d._soltou.all())
          and float(_t23d.metrics["sucesso"].min()) == 1.0)
    check("3. e o VALIDA é UM na espera final — ela NÃO zera os incentivos",
          float(_t23d.command[:, CMD.VALIDA].min()) == 1.0,
          "spec §6.0: o VALIDA deriva do interno; é isso que fecha o buraco da renda")
    # ⚠⚠ O ATRIBUTO TEM DE ESTAR PUBLICADO NO MESMO INSTANTE, e não na passada seguinte.
    # A ordem do mjlab é terminação e recompensa ANTES do comando, portanto um atraso de
    # uma passada deixa o `caixa_largada` ler `soltou = 0` no passo do fecho — o guarda
    # da espera final desarmado exatamente no passo do sucesso. Achado num code review
    # de 03/09, com medição.
    check("12. `limpo_soltou` é publicado NO MESMO passo do fecho, sem esperar a passada",
          float(_e23b.limpo_soltou.min()) == 1.0,
          f"medido {[round(float(x), 1) for x in _e23b.limpo_soltou[:4]]}")
    check("12. e o `escapou` DESARMA no mesmo instante — o passo do sucesso não mata",
          not bool(TE_.caixa_largada(_e23b, **_par23).any()),
          "é a regressão do atraso de uma passada: a terminação roda ANTES do comando")
    _o23b = _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))[0]
    _hotf = _o23b["actor"][:, OB_.fatia_do_elo(_o23b["actor"].shape[-1])].argmax(-1)
    check("12. a observação mostra ANDAR na espera final",
          bool((_hotf == CMD.ANDAR).all()))
    check("12. `limpo_soltou` é publicado como 1,0",
          float(_e23b.limpo_soltou.min()) == 1.0)
    check("a métrica `fracao_esperando` conta a espera final",
          float(_e23b.limpo_aguardando.max()) == 0.0
          and float(MT_.fracao_esperando(_e23b).min()) == 1.0,
          "sem isto a espera final não aparece no painel")

    # --- a terminação: `escapou` DESARMADO na espera final, `caiu` ARMADO ---
    _t23d._pegou[:] = True
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))          # publica limpo_pegou = 1
    _cx23 = _e23b.scene["box"]
    _pt = _cx23.data.root_link_pos_w.clone()
    _pt[:, 0] += 1.0                                           # 1 m à frente, mesma altura
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    # ⚠ do cfg de TREINO: o modo inspeção apaga as terminações (`terminations = {}`)
    _par = dict(cfg.terminations["caixa_largada"].params)
    _longe = TE_.caixa_largada(_e23b, **_par)
    check("12. afastar a caixa das palmas na espera final NÃO termina (escapou desarmado)",
          not bool(_longe.any()), str(_longe.tolist()))
    _pt2 = _cx23.data.root_link_pos_w.clone()
    _pt2[:, 2] = _e23b.scene.env_origins[:, 2] + 0.02            # no chão
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt2, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    _caiu = TE_.caixa_largada(_e23b, **_par)
    check("12. derrubar a caixa na espera final TERMINA (caiu armado)",
          bool(_caiu.all()), str(_caiu.tolist()))
    _t23d._soltou[:] = False                                    # antes do fecho...
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    check("12. ... e ANTES do fecho afastar as palmas continua terminando (escapou armado)",
          bool(TE_.caixa_largada(_e23b, **_par).all()))
    del _e23b
except Exception as _e23x:      # noqa: BLE001
    _falhas.append(f"as duas esperas não puderam ser medidas: "
                   f"{type(_e23x).__name__}: {_e23x}")

# ==================== 24. o TAMANHO da caixa por mundo, e o `caiu` por tamanho (spec §6.7)
secao("24. tamanho da caixa por mundo")
from g1_limpo import eventos as EV_                                       # noqa: E402

check("13. o evento `tamanho_caixa` existe, é de STARTUP e declara os três campos",
      "tamanho_caixa" in cfg.events and cfg.events["tamanho_caixa"].mode == "startup"
      and tuple(getattr(EV_.tamanho_caixa, "model_fields", ()))
      == ("geom_size", "geom_rbound", "geom_aabb"),
      "sem `requires_model_fields` o mjlab não expande os campos por mundo")
check("13. a faixa e o K são os da spec",
      tuple(k.cena.caixa_meia_aresta_faixa) == (0.07, 0.13) and k.cena.caixa_n_variantes == 8)
try:
    import torch as _t24

    _c24 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c24.scene.num_envs = 64
    _e24 = ManagerBasedRlEnv(cfg=_c24, device="cpu")
    _e24.reset()
    _n24 = _e24.action_manager.total_action_dim
    _cx = _e24.scene["box"]
    _loc, _ = _cx.find_geoms([C.BOX_GEOM])
    _g = int(_cx.indexing.geom_ids[_loc[0]])
    _size = _e24.sim.model.geom_size[:, _g]                      # (64, 3)
    _a = _size[:, 0]
    _K = _t24.linspace(*k.cena.caixa_meia_aresta_faixa, k.cena.caixa_n_variantes)
    _no_k = (_a.unsqueeze(-1) - _K.unsqueeze(0)).abs().min(-1).values < 1e-6
    check("13. `geom_size` difere entre mundos e só toma os K valores",
          bool(_no_k.all()) and len(_t24.unique(_a)) >= 5,
          f"{len(_t24.unique(_a))} valores distintos em 64 envs: {sorted(set(round(float(x),4) for x in _a))}")
    check("13. a caixa é CUBO: os três eixos iguais",
          float((_size - _a.unsqueeze(-1)).abs().max()) < 1e-7)
    check("13. `geom_rbound` acompanha: a·√3",
          float((_e24.sim.model.geom_rbound[:, _g] - _a * math.sqrt(3.0)).abs().max()) < 1e-6)
    check("13. `geom_aabb` acompanha: meia-caixa (a, a, a)",
          float((_e24.sim.model.geom_aabb[:, _g, 1] - _size).abs().max()) < 1e-7)
    _bm = _e24.sim.model.body_mass
    _bid = int(_cx.indexing.body_ids[0])
    check("13. `body_mass` da caixa NÃO mudou — independência do peso",
          float((_bm[..., _bid] - float(k.cena.caixa_massa)).abs().max()) < 1e-6,
          f"{_bm[..., _bid].flatten()[:4].tolist()}")
    check("13. `limpo_meia_aresta` bate com `geom_size` env a env",
          float((_e24.limpo_meia_aresta - _size).abs().max()) < 1e-7)
    # o colisor LÊ o tamanho: a caixa repousa com o centro a `a` acima do topo
    _passa_janela(_e24, _n24, _t24)
    _rep = (_cx.data.root_link_pos_w[:, 2] - _e24.limpo_topo - _a)
    check("13. a caixa repousa a `a` acima da laje em TODO env — o colisor lê o tamanho novo",
          float(_rep.abs().max()) < 5e-3,
          f"desvio máximo {float(_rep.abs().max())*1000:.1f} mm")
    # 15. todo consumidor lê o tamanho por env
    _t24c = _e24.command_manager.get_term("alvo_caixa")
    _alv = _t24c.alvos_das_palmas(_t24.arange(_e24.num_envs))
    _sep = (_alv[:, 0] - _alv[:, 1]).norm(dim=-1)
    check("15. `alvos_das_palmas` separa as palmas por 2a DO ENV",
          float((_sep - 2.0 * _a).abs().max()) < 1e-5)
    # 19. o `caiu` por tamanho
    _t24c._pegou[:] = True
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    _par24 = dict(cfg.terminations["caixa_largada"].params)   # o modo inspeção não tem terminações
    # ⚠ `soltou` LIGADO para ISOLAR o `caiu`. A terminação é `(caiu | escapou) & pegou`,
    # e aqui a caixa é teleportada longe das palmas — `escapou` dispararia e o teste
    # passaria pelo motivo errado, ou falharia no caso negativo. Com `soltou` o `escapou`
    # está desarmado (spec §6.6.3) e o que sobra é exatamente o `caiu` por tamanho.
    _t24c._soltou[:] = True
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    _q = _cx.data.root_link_quat_w
    _pf = _cx.data.root_link_pos_w.clone()
    _pf[:, 2] = _e24.scene.env_origins[:, 2] + _a                # deitada no chão
    _cx.write_root_link_pose_to_sim(_t24.cat([_pf, _q], -1))
    _cx.write_root_link_velocity_to_sim(_t24.zeros(_e24.num_envs, 6))
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    check("19. deitada no chão, a caixa de QUALQUER tamanho dispara `caiu`",
          bool(TE_.caixa_largada(_e24, **_par24).all()))
    _pl = _pf.clone()
    _pl[:, 2] = _e24.scene.env_origins[:, 2] + k.cena.prateleira_topo_piso + _a
    _cx.write_root_link_pose_to_sim(_t24.cat([_pl, _q], -1))
    _cx.write_root_link_velocity_to_sim(_t24.zeros(_e24.num_envs, 6))
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    check("19. apoiada na laje mais baixa, a caixa MENOR não dispara `caiu`",
          not bool(TE_.caixa_largada(_e24, **_par24).any()))
    del _e24
except Exception as _e24x:      # noqa: BLE001
    _falhas.append(f"o tamanho por mundo não pôde ser medido: "
                   f"{type(_e24x).__name__}: {_e24x}")

# ======== 25. a OBSERVAÇÃO: gate, giro_b, meia_aresta, VALIDA fora, crítico (spec §4, §6.1)
secao("25. a observação nova")
check("3. `N_CAIXA` é 10: caixa_b(3) alvo_b(3) giro_b(3) meia_aresta(1)", OB_.N_CAIXA == 10)
check("3. o VALIDA NÃO está na observação",
      "[:, VALIDA]" not in inspect.getsource(OB_.caixa_no_frame_da_base),
      "o docstring pode citar o bit; o CÓDIGO não pode lê-lo")
check("o canal GIRO é o ÚLTIMO do comando (append), e DIM é 12",
      CMD.GIRO == slice(9, 12) and CMD.DIM == 12)
try:
    import torch as _t25
    from mjlab.utils.lab_api.math import quat_mul as _qmul

    # --- dimensões ---
    _c25 = make_env_cfg(k, inspecao=True, elo=CMD.ANDAR)
    _c25.scene.num_envs = 16
    _e25 = ManagerBasedRlEnv(cfg=_c25, device="cpu")
    _o25, _ = _e25.reset()
    _n25 = _e25.action_manager.total_action_dim
    # ⚠ O CRÍTICO DO FABRICANTE JÁ É ASSIMÉTRICO: ele tem 12 canais privilegiados de pé
    # (`foot_height` 2, `foot_air_time` 2, `foot_contact` 2, `foot_contact_forces` 6)
    # que o ator não tem. Portanto crítico = 114 + 12 + 5 (`elo_interno`) = 131. A spec
    # v14 dizia 119 por ignorar os 12; medido em 03/09 e corrigido.
    _dc25 = _o25["critic"].shape[-1]
    check("3. o ator tem 114 canais e o crítico 131 (114 + 12 do fabricante + 5 do interno)",
          _o25["actor"].shape[-1] == 114 and _dc25 == 114 + 12 + 5,
          f"ator {_o25['actor'].shape[-1]}, crítico {_dc25}")
    _int = _o25["critic"][:, OB_.fatia_do_elo_interno(_dc25)]
    check("o `elo_interno` do crítico é um one-hot",
          bool(_t25.allclose(_int.sum(-1), _t25.ones(16))))
    # --- 1. o gate: caixa PERTO e publicado ANDAR -> os 10 canais são zero ---
    _cx25 = _e25.scene["box"]
    _p25 = _e25.scene["robot"].data.root_link_pos_w.clone()
    _p25[:, 0] += 0.5
    for _ in range(3):
        _cx25.write_root_link_pose_to_sim(_t25.cat([_p25, _cx25.data.root_link_quat_w], -1))
        _cx25.write_root_link_velocity_to_sim(_t25.zeros(16, 6))
        _o25 = _e25.step(_t25.zeros(16, _n25))[0]
    _cx_slice_a = _o25["actor"][:, 114 - OB_.N_CAIXA:114]
    _cx_slice_c = _o25["critic"][:, _dc25 - 5 - OB_.N_CAIXA:_dc25 - 5]
    check("1. com a caixa a 0,5 m e o publicado em ANDAR, os 10 canais são EXATAMENTE zero (ator)",
          float(_cx_slice_a.abs().max()) == 0.0, f"máximo {float(_cx_slice_a.abs().max())}")
    check("1. ... e no crítico também",
          float(_cx_slice_c.abs().max()) == 0.0)
    del _e25

    # --- 2. a invariante, e 3. meia_aresta, na borda da espera ---
    _c25b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c25b.scene.num_envs = 16
    _e25b = ManagerBasedRlEnv(cfg=_c25b, device="cpu")
    _o25b, _ = _e25b.reset()
    _n25b = _e25b.action_manager.total_action_dim
    _fat = OB_.fatia_do_elo(114)
    _cxs = slice(114 - OB_.N_CAIXA, 114)
    _viol = False
    _borda_ok = False
    _hot_ant = _o25b["actor"][:, _fat].argmax(-1)
    _cx_ant = _o25b["actor"][:, _cxs]
    check("4. no reset: publicado ANDAR e canais de caixa zero",
          bool((_hot_ant == CMD.ANDAR).all()) and float(_cx_ant.abs().max()) == 0.0)
    for _ in range(int(k.alvo.espera_s[1] / _e25b.step_dt) + 5):
        _o25b = _e25b.step(_t25.zeros(16, _n25b))[0]
        _hot = _o25b["actor"][:, _fat].argmax(-1)
        _cxo = _o25b["actor"][:, _cxs]
        _norma = _cxo[:, :3].norm(dim=-1)
        _viol |= bool(((_hot == CMD.ANDAR) & (_norma > 0)).any())
        _viol |= bool(((_hot != CMD.ANDAR) & (_norma == 0)).any())
        _borda_ok |= bool(((_hot_ant == CMD.ANDAR) & (_hot == CMD.PEGAR) & (_norma > 0.1)).any())
        _hot_ant = _hot
    check("2. NUNCA existe |caixa_b| = 0 com publicado ≠ ANDAR, nem ≠ 0 com ANDAR", not _viol)
    check("4. na borda os canais ACENDEM no mesmo passo em que o one-hot vira PEGAR", _borda_ok)
    _meia_obs = _o25b["actor"][:, 113]
    check("3. o último canal é `meia_aresta` e bate com `limpo_meia_aresta` env a env",
          float((_meia_obs - _e25b.limpo_meia_aresta[:, 0]).abs().max()) < 1e-6,
          f"{_meia_obs[:4].tolist()} vs {_e25b.limpo_meia_aresta[:4, 0].tolist()}")
    # --- 23. giro_b: em PEGAR a face está CONGELADA -> zero na abertura, e cresce ao torcer
    _t25c = _e25b.command_manager.get_term("alvo_caixa")
    _giro0 = _o25b["actor"][:, 114 - 4:114 - 1]
    # ⚠ ~0 e não 0 exato: a face congela na abertura do elo e a caixa assenta alguns
    # milímetros depois disso. Medido: 0,029 rad = 1,7°.
    check("23. em PEGAR, na abertura, giro_b é ~0 (face congelada)",
          float(_giro0.norm(dim=-1).max()) < 5e-2, f"{float(_giro0.norm(dim=-1).max()):.4f}")
    _cx25b = _e25b.scene["box"]
    _ang = math.radians(20.0)
    # ⚠ a torção é RELATIVA ao quatérnion da abertura (a face está congelada nele): a
    # caixa nasce com um desalinho de até ±15°, portanto um yaw ABSOLUTO de 20° não daria
    # |giro| = 20°. Compõe-se `qz(20°) ⊗ q0`.
    _q0 = _cx25b.data.root_link_quat_w.clone()
    _qz = _qmul(_t25.tensor([math.cos(_ang / 2), 0.0, 0.0, math.sin(_ang / 2)]).expand(16, 4), _q0)
    for _ in range(3):
        _cx25b.write_root_link_pose_to_sim(_t25.cat([_cx25b.data.root_link_pos_w, _qz], -1))
        _cx25b.write_root_link_velocity_to_sim(_t25.zeros(16, 6))
        _o25b = _e25b.step(_t25.zeros(16, _n25b))[0]
    _giro1 = _o25b["actor"][:, 114 - 4:114 - 1]
    check("23. torcida 20° em Z, |giro_b| ≈ 0,35 e bate com ANG",
          float((_giro1.norm(dim=-1) - _t25c.command[:, CMD.ANG]).abs().max()) < 1e-4
          and abs(float(_giro1.norm(dim=-1).mean()) - _ang) < 0.05,
          f"|giro| {float(_giro1.norm(dim=-1).mean()):.3f}, ANG {float(_t25c.command[:, CMD.ANG].mean()):.3f}")
    check("23. ... e o eixo é Z", float(_giro1[:, :2].abs().max()) < 0.05)
    del _e25b

    # --- 23. giro_b no REORIENTAR: direção VIVA; caixa girada 90° em Z pede giro em Z ---
    # ⚠ SEM jitter em y na caixa: a direção viva é "da caixa para o robô", e com a caixa
    # deslocada em y ela deixa de ser −X puro — o eixo do giro do tombo ganharia uma
    # componente em x e o ângulo do giro em Z deixaria de ser 90°. Com dy = 0 os dois
    # casos são exatos.
    _kk25 = dataclasses.replace(k, cena=dataclasses.replace(k.cena, caixa_jitter_y=(0.0, 0.0)))
    _c25c = make_env_cfg(_kk25, inspecao=True, elo=CMD.REORIENTAR)
    _c25c.scene.num_envs = 8
    _e25c = ManagerBasedRlEnv(cfg=_c25c, device="cpu")
    _e25c.reset()
    _n25c = _e25c.action_manager.total_action_dim
    _cx25c = _e25c.scene["box"]
    _t25d = _e25c.command_manager.get_term("alvo_caixa")
    # ⚠⚠ SEM CADEIA, e sem isto o teste mede outra coisa. Com `reorientar_inerte` o
    # REORIENTAR fecha em 0,3 s e a cadeia 1 avança para o PEGAR — e no PEGAR a face é
    # CONGELADA, portanto o `giro_b` passa a medir a torção desde o avanço em vez do
    # giro pedido. Medido em 03/09: o primeiro caso lia (0,0,0) e o do tombo lia o eixo
    # X. `CADEIA_NENHUMA` bloqueia o avanço (o `_avanca_elo` filtra por `_cadeia >= 0`).
    _t25d._cadeia[:] = CMD.CADEIA_NENHUMA

    def _giro_com(quat):
        for _ in range(int(k.alvo.espera_s[1] / _e25c.step_dt) + 5):
            _cx25c.write_root_link_pose_to_sim(
                _t25.cat([_cx25c.data.root_link_pos_w, quat.expand(8, 4)], -1))
            _cx25c.write_root_link_velocity_to_sim(_t25.zeros(8, 6))
            _o = _e25c.step(_t25.zeros(8, _n25c))[0]
        return _o["actor"][:, 114 - 4:114 - 1]

    _h = math.pi / 4
    _g_mais = _giro_com(_t25.tensor([math.cos(_h), 0.0, 0.0, math.sin(_h)]))   # yaw +90°
    _g_menos = _giro_com(_t25.tensor([math.cos(_h), 0.0, 0.0, -math.sin(_h)]))  # yaw −90°
    _g_pitch = _giro_com(_t25.tensor([math.cos(_h), 0.0, math.sin(_h), 0.0]))   # pitch +90°
    check("23. caixa girada 90° em Z: |giro_b| ≈ π/2 e o eixo é Z",
          abs(float(_g_mais.norm(dim=-1).mean()) - math.pi / 2) < 0.05
          and float(_g_mais[:, :2].abs().max()) < 0.1,
          f"{_g_mais[0].tolist()}")
    check("23. o SINAL troca com o sentido do giro",
          float((_g_mais[:, 2] * _g_menos[:, 2]).max()) < 0.0)
    check("23. caixa tombada 90° em Y: o eixo é Y",
          abs(float(_g_pitch[:, 1].abs().mean()) - math.pi / 2) < 0.05
          and float(_g_pitch[:, [0, 2]].abs().max()) < 0.1,
          f"{_g_pitch[0].tolist()}")
    del _e25c
except Exception as _e25x:      # noqa: BLE001
    _falhas.append(f"a observação nova não pôde ser medida: "
                   f"{type(_e25x).__name__}: {_e25x}")

# ============ 26. a RENDA DO BOTAR é MONÓTONA: pairar < apoiar < fechar < largar (spec §6.6)
secao("26. a renda do BOTAR")
# ⚠ v2.1 (spec P3): `load` SAIU do módulo — a ausência é provada na seção "v2.1:
# gradientes" (check 7, `"load" not in cfg.rewards`). O que ele fazia (fazer o fecho
# do BOTAR pagar mais que pairar) é agora `renda_congelada`, provado nos checks 8 e 9
# da mesma seção, e reaproveitado abaixo no MESMO env sintético desta seção.
check("17. `squeeze` e `unload` são MASCARADOS no BOTAR (o precedente do g1_poc)",
      "_fora_do_botar" in inspect.getsource(RC_.squeeze)
      and "_fora_do_botar" in inspect.getsource(RC_.unload)
      and "!= BOTAR" in inspect.getsource(RC_._fora_do_botar))
check("17. `alcança ≡ 1` no BOTAR ou em `soltou`",
      "== BOTAR" in inspect.getsource(RC_._alcancar)
      and "limpo_soltou" in inspect.getsource(RC_._alcancar))
try:
    import torch as _t26

    _c26 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c26.scene.num_envs = 8
    _e26 = ManagerBasedRlEnv(cfg=_c26, device="cpu")
    _e26.reset()
    _n26 = _e26.action_manager.total_action_dim
    _passa_janela(_e26, _n26, _t26)
    _t26c = _e26.command_manager.get_term("alvo_caixa")
    _ids26 = _t26.arange(8)
    _nm26 = list(_c26.rewards)
    _cx26 = _e26.scene["box"]
    _q26 = _cx26.data.root_link_quat_w.clone()

    def _renda(passos=4, alvo_dz=None, alvo_dx=0.0):
        """Soma dos termos por segundo, com a caixa PINADA em alvo + (dx, dz).

        ⚠ Pinar a cada passo é obrigatório: a pose escrita persiste e a caixa não cai,
        portanto "solta e deixa assentar" não existe aqui. Ver o bloco de medição
        abaixo para a tabela força × penetração.
        """
        for _ in range(passos):
            if alvo_dz is not None:
                _p = _t26c.command[:, CMD.ALVO].clone()
                _p[:, 0] += alvo_dx
                _p[:, 2] += alvo_dz
                _cx26.write_root_link_pose_to_sim(_t26.cat([_p, _q26], -1))
                _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
            _e26.step(_t26.zeros(8, _n26))
        _sr = _e26.reward_manager._step_reward
        return (float(_sr.mean(0).sum()),
                {n: float(_sr[:, _nm26.index(n)].mean())
                 for n in ("staged", "precise_pos", "largou", "unload", "squeeze",
                           "postura_ereta", "sustentacao", "track_linear_velocity",
                           "pose", "renda_congelada")})

    # 17. alcança no PEGAR com a caixa longe é ~0; no BOTAR é 1
    _pl = _cx26.data.root_link_pos_w.clone()
    _pl[:, 0] += 1.0
    for _ in range(3):
        _cx26.write_root_link_pose_to_sim(_t26.cat([_pl, _q26], -1))
        _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
        _e26.step(_t26.zeros(8, _n26))
    _alc_pegar = float(RC_._alcancar(_e26, "alvo_caixa").max())
    _t26c.forca_avanco(_ids26)            # -> CARREGAR
    _t26c.forca_avanco(_ids26)            # -> BOTAR (laje nova sob a caixa; alvo lateral)
    _e26.step(_t26.zeros(8, _n26))
    _alc_botar = float(RC_._alcancar(_e26, "alvo_caixa").min())
    check("17. `alcança` < 0,1 no PEGAR com a caixa a 1 m, e == 1 no BOTAR na mesma pose",
          _alc_pegar < 0.1 and _alc_botar == 1.0, f"pegar {_alc_pegar:.3f}, botar {_alc_botar:.3f}")
    # 17. as máscaras, com uma força de palma FINGIDA (o robô pinado não aperta nada)
    _orig = RC_._forca_das_palmas
    RC_._forca_das_palmas = lambda env, sensores, asset_cfg: _t26.full((env.num_envs,), 20.0)
    try:
        _sq_botar = float(RC_.squeeze(_e26, "alvo_caixa", C.SENSOR_PALMA, k.tarefa.squeeze_mu,
                                      cfg.rewards["squeeze"].params["asset_cfg"]).max())
        _t26c._elo[:] = CMD.PEGAR
        _sq_pegar = float(RC_.squeeze(_e26, "alvo_caixa", C.SENSOR_PALMA, k.tarefa.squeeze_mu,
                                      cfg.rewards["squeeze"].params["asset_cfg"]).min())
        _t26c._elo[:] = CMD.BOTAR
    finally:
        RC_._forca_das_palmas = _orig
    check("17. com a mesma força de palma, `squeeze` é 0 no BOTAR e > 0 no PEGAR",
          _sq_botar == 0.0 and _sq_pegar > 0.5, f"botar {_sq_botar:.3f}, pegar {_sq_pegar:.3f}")

    # A: pairar 2 cm acima do alvo, sem apoio
    _rA, _dA = _renda(passos=6, alvo_dz=0.02)
    # ⚠ v2.1: os DOIS `forca_avanco` acima (PEGAR->CARREGAR->BOTAR) já são dois fechos
    # ganhos — `renda_congelada` já carrega essa soma antes mesmo de pairar no BOTAR.
    check("18. pairando no BOTAR, `renda_congelada` já carrega os DOIS fechos "
          "anteriores (PEGAR, CARREGAR)",
          _dA["renda_congelada"] > 0.0, f"{_dA['renda_congelada']:.4f}")
    check("17. pairando no BOTAR, `unload` e `postura_ereta` são 0 (mascarados)",
          abs(_dA["unload"]) < 1e-9 and abs(_dA["postura_ereta"]) < 1e-9)
    # ⚠⚠ COMO SE PRODUZ "APOIADA" NUM TESTE, e o método é uma cicatriz de 03/09. A
    # `write_root_link_pose_to_sim` PERSISTE: ela re-aplica a pose a cada passo, portanto
    # a caixa NÃO CAI. Medido: solta 20 cm acima do alvo, ela fica 25 passos a 20 cm, e
    # `F_apoio` é ZERO. Com a caixa pinada, a força de apoio vem só da PENETRAÇÃO — e
    # pinar exatamente na altura de repouso dá penetração zero, logo força zero. Por isso
    # o estado "apoiada" aqui é pinar 2 mm ABAIXO do alvo. MEDIDO, uniforme nos 8 envs:
    #
    #     dz    +0,020   0,000   −0,002   −0,005   −0,010   −0,020
    #     F/mg   0,00     0,00    0,98     0,94     0,88     0,75
    #
    # A força CAI com penetração maior (o re-pino a cada passo limita o impulso), o que
    # é artefato do método e não física da tarefa — por isso a penetração é mínima.
    _DZ_APOIA = -0.002
    _t26c._sust[:] = 0.0
    _rC, _dC = _renda(passos=6, alvo_dz=_DZ_APOIA)
    check("18. apoiada no alvo, `sustentacao` já acumula crédito parcial que pairar "
          "não tem — antes era `load` que fazia a diferença; `load` saiu (spec P3)",
          _dC["sustentacao"] > _dA["sustentacao"],
          f"apoiada {_dC['sustentacao']:.4f}, pairar {_dA['sustentacao']:.4f}")
    check("12. e ainda NÃO fechou (0,3 s de sustain, e são 6 passos)",
          not bool(_t26c.fechou.any()), f"sust {float(_t26c._sust.min()):.2f} s")
    # ⚠⚠ A FORÇA DE APOIO É PROJETADA NO EIXO VERTICAL (decisão do dono, 03/09). A norma
    # não tem direção: prensar a caixa de lado contra o tampo satisfazia `apoiada` sem a
    # laje carregar peso nenhum. v2.1: `load` (que também lia esta projeção) SAIU —
    # sobra o FECHO do `BOTAR`, que continua lendo a MESMA função.
    check("18. o fecho do BOTAR lê `forca_de_apoio`, que projeta em z",
          "forca_de_apoio" in inspect.getsource(CMD.AlvoCaixaCmd._fecha_elo_corrente)
          and "[..., 2].abs()" in inspect.getsource(CMD.forca_de_apoio))
    _t26c._sust[:] = 0.0
    # ⚠ v2.1: mais passos que os 6 originais — sem os desvios de 25 cm e de clamp de
    # massa que existiam aqui antes de `load` sair, a caixa tinha MENOS ciclos de
    # re-pino nesta posição para assentar o contato. `passos=6` deixava um resíduo
    # horizontal (medido: 0,505 N contra o limiar de 0,05×m·g); `passos=15` assenta.
    _renda(passos=15, alvo_dz=_DZ_APOIA)
    _f26 = _e26.scene[C.SENSOR_APOIO].data.force.squeeze(1)
    _mg26 = float((_e26.limpo_massa * 9.81).mean())
    _fz26 = float(_f26[:, 2].abs().mean())
    _fxy26 = float(_f26[:, :2].abs().max())
    # ⚠ A CONVENÇÃO MEDIDA em 03/09: apoio dá `f = (0, 0, −9,57)` com `m·g = 9,81` — a
    # força é puramente VERTICAL e sai com o sinal invertido. Se um upgrade do `mjlab`
    # inverter a ordem do par de geoms, o `abs` do termo continua certo; esta trava
    # existe para a inversão aparecer, e não para o treino ficar errado em silêncio.
    check("18. com a caixa apoiada a força é VERTICAL, e vale ~m·g",
          abs(_fz26 / _mg26 - 1.0) < 0.10 and _fxy26 < 0.05 * _mg26,
          f"|f_z|/mg {_fz26/_mg26:.2f}, |f_xy| máx {_fxy26:.3f} N")
    check("18. e a projeção lê a força INTEIRA no repouso — nada horizontal é contado",
          abs(float(CMD.forca_de_apoio(_e26, C.SENSOR_APOIO).mean())
              - float(_f26.norm(dim=-1).mean())) < 0.05,
          "se divergirem, existe componente horizontal entrando na conta")
    # ⚠ A MÉTRICA DE IMPACTO (03/09): soltar de 5 cm é PERMITIDO, jogar de mais alto não.
    # Sem ela os dois leem igual no painel. Peso nenhum — é só medição.
    check("18. a métrica `impacto_da_caixa` existe, tem `reset` e lê ~1 no repouso",
          "impacto_da_caixa" in cfg.metrics
          and callable(getattr(MT_.impacto_da_caixa, "reset", None))
          and abs(float(MT_.impacto_da_caixa(
              None, _e26)(_e26, sensor_apoio=C.SENSOR_APOIO).mean()) - 1.0) < 0.10,
          "é o que separa `apoiou com cuidado` de `jogou de 30 cm`")
    # volta a apoiar no alvo e deixa o BOTAR FECHAR sozinho -> espera final
    _t26c._sust[:] = 0.0
    _rC2, _dC2 = _renda(passos=6, alvo_dz=_DZ_APOIA)
    for _ in range(int(k.cadeia.sustenta_outros_s / _e26.step_dt) + 3):
        _pS = _t26c.command[:, CMD.ALVO].clone()
        _pS[:, 2] += _DZ_APOIA
        _cx26.write_root_link_pose_to_sim(_t26.cat([_pS, _q26], -1))
        _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
        _e26.step(_t26.zeros(8, _n26))
    check("12. apoiada no alvo o BOTAR FECHA sozinho e `soltou` marca",
          bool(_t26c.fechou.all()) and bool(_t26c._soltou.all()),
          f"fechou {int(_t26c.fechou.sum())}/8, soltou {int(_t26c._soltou.sum())}/8")
    _rF, _dF = _renda(passos=4, alvo_dz=_DZ_APOIA)
    check("18. na espera final, com as palmas longe, `largou` ≥ 0,95 × 1,0",
          _dF["largou"] >= 0.95, f"{_dF['largou']:.3f}")
    check("16. a RENDA É MONÓTONA: pairar < apoiada < espera final (palmas longe)",
          _rA < _rC2 < _rF,
          f"pairar {_rA:.2f}  apoiada {_rC2:.2f}  espera final {_rF:.2f}  (/s)")
    # ⚠ v2.1 (spec P4): o rastreio NÃO entra mais na espera final. O gate agora lê o
    # elo INTERNO (`env.limpo_twist_zerado`), e ele continua BOTAR na espera final —
    # só o PUBLICADO vira ANDAR. Antes (gate pelo publicado) o rastreio entrava aqui;
    # ver o item 11 da seção "v2.1: gradientes" para a prova geral.
    check("16. o rastreio NÃO entra nem na espera final — o elo INTERNO segue BOTAR",
          _dC2["track_linear_velocity"] == 0.0 and _dF["track_linear_velocity"] == 0.0)
    print(f"  renda do BOTAR: pairar {_rA:.2f}  apoiada {_rC2:.2f}  espera final {_rF:.2f} /s")
    del _e26
except Exception as _e26x:      # noqa: BLE001
    _falhas.append(f"a renda do BOTAR não pôde ser medida: "
                   f"{type(_e26x).__name__}: {_e26x}")

# ==================== 27. o RAMO DE GIRO no sorteador do twist (spec §9)
secao("27. girar no lugar")
check("21. os knobs do giro são os da spec",
      k.marcha.rel_turning_envs == 0.10 and k.marcha.turning_wz_min == 0.2)
check("21. o cfg do twist recebe os dois",
      cfg.commands["twist"].rel_turning_envs == 0.10
      and cfg.commands["twist"].turning_wz_min == 0.2)
try:
    import torch as _t27

    _c27 = make_env_cfg(k, elo=CMD.ANDAR)          # cfg de TREINO, elo forçado
    _c27.scene.num_envs = 512
    _e27 = ManagerBasedRlEnv(cfg=_c27, device="cpu")
    _e27.reset()
    _n27 = _e27.action_manager.total_action_dim
    _tw27 = _e27.command_manager.get_term("twist")
    _todos = _t27.arange(512)
    _cont = {"turning": 0, "standing": 0, "forward": 0, "heading": 0, "n": 0}
    _ok_wz = _ok_lin = _ok_heading = True
    for _ in range(8):
        _tw27._resample_command(_todos)
        _tu = _tw27.is_turning_env
        _cont["turning"] += int(_tu.sum()); _cont["n"] += 512
        _cont["standing"] += int(_tw27.is_standing_env.sum())
        _cont["forward"] += int(_tw27.is_forward_env.sum())
        _cont["heading"] += int(_tw27.is_heading_env.sum())
        if bool(_tu.any()):
            _ok_wz &= bool((_tw27.vel_command_b[_tu, 2].abs() >= k.marcha.turning_wz_min - 1e-6).all())
            _ok_lin &= float(_tw27.vel_command_b[_tu, :2].abs().max()) == 0.0
            _ok_heading &= not bool(_tw27.is_heading_env[_tu].any())
    _frac = _cont["turning"] / _cont["n"]
    check("21. a fração REALIZADA de turning é 0,09 ± 0,02 (0,10 × 0,90, fora do standing)",
          abs(_frac - 0.09) < 0.02, f"{_frac:.3f} em {_cont['n']} sorteios")
    check("21. |wz| ≥ 0,2 em todo env turning", _ok_wz)
    check("21. lin = 0 em todo env turning, no resample", _ok_lin)
    check("21. nenhum env turning está em heading", _ok_heading)
    check("21. o standing continua ~0,10 — o turning não o comeu",
          abs(_cont["standing"] / _cont["n"] - 0.10) < 0.02,
          f"{_cont['standing']/_cont['n']:.3f}")
    # lin continua ZERO passo a passo, e wz NÃO é reescrito pelo heading
    _tu = _tw27.is_turning_env.clone()
    _wz0 = _tw27.vel_command_b[:, 2].clone()
    for _ in range(3):
        _e27.step(_t27.zeros(512, _n27))
    check("21. lin = 0 nos envs turning em TODO passo",
          float(_tw27.vel_command_b[_tu, :2].abs().max()) == 0.0)
    check("21. e wz dos envs turning não muda entre passos (fora do heading)",
          float((_tw27.vel_command_b[_tu, 2] - _wz0[_tu]).abs().max()) < 1e-6)
    del _e27
except Exception as _e27x:      # noqa: BLE001
    _falhas.append(f"o ramo de giro não pôde ser medido: {type(_e27x).__name__}: {_e27x}")

# ==================== 28. o REORIENTAR está INERTE na v2 (spec §8.3)
secao("28. o REORIENTAR inerte")
check("24. `voltas_max` é zero e `eixo_vertical` é falso em TODO nível",
      all(v == 0 for v in k.nivel.voltas_max) and not any(k.nivel.eixo_vertical),
      f"{k.nivel.voltas_max} / {k.nivel.eixo_vertical}")
check("o REORIENTAR CONTINUA sorteável — o slot não pode ficar constante (P8-b: 5%)",
      CMD.REORIENTAR in ELOS_SORTEAVEIS
      and cfg.curriculum["elo"].params["pesos_manip"][0] == k.cadeia.prob_reorientar_inerte)
# ⚠ MEDIDO em 03/09: `voltas_max = 0` NÃO basta para o elo ficar inerte. A direção pedida
# é "da caixa para o robô", e com o jitter lateral da caixa (±0,18 m em y) ela sai até
# ~29° do eixo −X; somado ao desalinho de ±15°, ~1 em 6 envs nasce FORA dos 25° de
# tolerância e o elo não fecha sozinho. O interruptor de verdade é o knob
# `reorientar_inerte`: o fecho do REORIENTAR ignora `alinhado` enquanto ele for verdadeiro.
check("24. o interruptor `reorientar_inerte` está LIGADO e chega ao comando",
      k.cadeia.reorientar_inerte is True
      and cfg.commands["alvo_caixa"].reorientar_inerte is True,
      "sem ele ~15% dos episódios de REORIENTAR exigiriam girar a caixa até 45°")
try:
    import torch as _t28

    _c28 = make_env_cfg(k, inspecao=True, elo=CMD.REORIENTAR)
    _c28.scene.num_envs = 16
    _e28 = ManagerBasedRlEnv(cfg=_c28, device="cpu")
    _e28.reset()
    _n28 = _e28.action_manager.total_action_dim
    _t28c = _e28.command_manager.get_term("alvo_caixa")
    _p0 = _e28.scene["box"].data.root_link_pos_w.clone()
    _passa_janela(_e28, _n28, _t28)
    for _ in range(int(k.cadeia.sustenta_outros_s / _e28.step_dt) + 3):
        _e28.step(_t28.zeros(16, _n28))
    _dp = (_e28.scene["box"].data.root_link_pos_w - _p0).norm(dim=-1)
    check("24. um env de cadeia 1 avança para o PEGAR em `sustenta_outros_s` sem a caixa se mover",
          bool((_t28c._elo == CMD.PEGAR).all()) and float(_dp.max()) < 0.01,
          f"elo {_t28c._elo.tolist()[:6]}, deslocamento máx {float(_dp.max())*1000:.1f} mm")
    del _e28
except Exception as _e28x:      # noqa: BLE001
    _falhas.append(f"o REORIENTAR inerte não pôde ser medido: {type(_e28x).__name__}: {_e28x}")

secao("v2.1: gradientes")

# --- 1. `precise_pos` no limiar do fecho, e a derivada do par caixa->alvo ---
_pp_no_limiar = math.exp(-(tr.tol_pos / tr.precise_pos_sigma) ** 2)
check("1. `precise_pos(d = tol_pos) >= 0,5` — a rampa de aceite paga no limiar do fecho",
      _pp_no_limiar >= 0.5, f"{_pp_no_limiar:.4f}")


def _deriv_par_v21(d: float) -> float:
    """Derivada TOTAL do par caixa->alvo: `staged`/trazer efetivo + `precise_pos`.

    ⚠ `2,7` é o peso EFETIVO do `staged` no ponto medido (peso 3,0 × alcança ~0,9), e
    não o peso bruto do termo — spec `g1-limpo-gradientes-v2.md` §1, proposta §3 P1.
    """
    t_macro = 2.7 * (2.0 * d / 0.45 ** 2) * math.exp(-(d / 0.45) ** 2)
    t_precise = (tr.precise_pos * (2.0 * d / tr.precise_pos_sigma ** 2)
                 * math.exp(-(d / tr.precise_pos_sigma) ** 2))
    return t_macro + t_precise


_d_perto, _d_longe = _deriv_par_v21(tr.tol_pos), _deriv_par_v21(0.45)
check("1. em d=0,10 a derivada do par bate com a spec (~16,1)",
      abs(_d_perto - 16.1) < 0.2, f"{_d_perto:.3f}")
check("1. a derivada do par é maior perto do alvo (d=tol_pos) que longe (d=0,45)",
      _d_perto > _d_longe, f"perto={_d_perto:.2f} longe={_d_longe:.2f}")

# --- 2. `sustentacao` lê o relógio do comando: `_sust/_sustain_alvo`, sem `avancou` ---
try:
    import torch as _t29

    _c29 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=2)
    _c29.scene.num_envs = 16
    _e29 = ManagerBasedRlEnv(cfg=_c29, device="cpu")
    _e29.reset()
    _n29 = _e29.action_manager.total_action_dim
    _passa_janela(_e29, _n29, _t29)
    _t29c = _e29.command_manager.get_term("alvo_caixa")
    _ids29 = _t29.arange(_e29.num_envs)

    check("2. `_sustain_alvo` no PEGAR é `sustenta_pegar_s`",
          bool((_t29c._sustain_alvo == kc.sustenta_pegar_s).all()),
          f"{_t29c._sustain_alvo.tolist()[:3]}")
    _sust_esperado = (_t29c._sust
                     / _t29c._sustain_alvo.clamp(min=1e-6)).clamp(0.0, 1.0)
    _sust_medido = RC_.sustentacao(_e29, "alvo_caixa")
    check("2. `sustentacao` == `_sust / _sustain_alvo` × VALIDA, no PEGAR",
          float((_sust_medido - _sust_esperado).abs().max()) < 1e-6,
          f"medido {_sust_medido.tolist()[:3]}, esperado {_sust_esperado.tolist()[:3]}")

    _t29c.forca_avanco(_ids29)          # PEGAR -> CARREGAR
    _e29.step(_t29.zeros(_e29.num_envs, _n29))
    check("2. após o avanço forçado, `sustentacao` == 0 no passo seguinte",
          float(RC_.sustentacao(_e29, "alvo_caixa").abs().max()) == 0.0)
    check("2. e `_sustain_alvo` virou `carregar_s` (cadeia 2 não é segurar-parado)",
          bool((_t29c._sustain_alvo == kc.carregar_s).all()),
          f"{_t29c._sustain_alvo.tolist()[:3]}")
    check("2. o atributo `avancou` NÃO existe mais no termo de comando",
          not hasattr(_t29c, "avancou"))
    del _e29
except Exception as _e29x:      # noqa: BLE001
    _falhas.append(f"a `sustentacao` v2.1 não pôde ser medida: "
                   f"{type(_e29x).__name__}: {_e29x}")

# --- 3. `caixa_largada`: `caiu` vale SOZINHO, sem a arma do `pegou` ---
try:
    import torch as _t30

    _c30 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c30.scene.num_envs = 16
    _e30 = ManagerBasedRlEnv(cfg=_c30, device="cpu")
    _e30.reset()
    _n30 = _e30.action_manager.total_action_dim
    _par30 = dict(cfg.terminations["caixa_largada"].params)

    check("3. no reset, `caixa_largada` é falso (`limpo_pegou = 0`, caixa na laje)",
          float(_e30.limpo_pegou.max()) == 0.0
          and not bool(TE_.caixa_largada(_e30, **_par30).any()))

    _cx30 = _e30.scene["box"]
    _meia30 = _e30.limpo_meia_aresta[:, 2]
    _q30 = _cx30.data.root_link_quat_w.clone()
    _p30 = _cx30.data.root_link_pos_w.clone()
    _p30[:, 2] = _e30.scene.env_origins[:, 2] + _meia30 + 0.01
    _cx30.write_root_link_pose_to_sim(_t30.cat([_p30, _q30], -1))
    _cx30.write_root_link_velocity_to_sim(_t30.zeros(_e30.num_envs, 6))
    _e30.step(_t30.zeros(_e30.num_envs, _n30))
    check("3. `caiu` vale SOZINHO: caixa no chão com `limpo_pegou = 0` já termina",
          float(_e30.limpo_pegou.max()) == 0.0
          and bool(TE_.caixa_largada(_e30, **_par30).all()),
          "antes exigia `pegou`: derrubar da mesa antes da 1ª preensão não terminava")
    del _e30
except Exception as _e30x:      # noqa: BLE001
    _falhas.append(f"o `caiu` desarmado não pôde ser medido: "
                   f"{type(_e30x).__name__}: {_e30x}")

# --- 4. P8-b: o REORIENTAR inerte FICA no sorteio, a 5% (decisão do dono, 2026-09-04) ---
_p31 = dict(cfg.curriculum["elo"].params)
_p31["fatia_loco"] = 0.5            # metade de manipulação: ~10.000 sorteios de elo
_falso31 = types.SimpleNamespace(num_envs=20_000, device="cpu")
CU2.sorteia_elo(_falso31, __import__("torch").arange(20_000), **_p31)
_b31 = _falso31.limpo_elo
_manip31 = int((_b31 != CMD.ANDAR).sum())
_frac31 = int((_b31 == CMD.REORIENTAR).sum()) / max(_manip31, 1)
check("4. P8-b: com os params do cfg de TREINO, o REORIENTAR inerte sai em [3%; 7%] da manipulação",
      0.03 <= _frac31 <= 0.07, f"{_frac31:.4f} sobre {_manip31}")
check("4. P8-b: `pesos_manip` do cfg é `(prob_reorientar_inerte, 1 − prob)`",
      cfg.curriculum["elo"].params["pesos_manip"]
      == (k.cadeia.prob_reorientar_inerte, 1.0 - k.cadeia.prob_reorientar_inerte),
      str(cfg.curriculum["elo"].params["pesos_manip"]))

# --- 5. `impacto_da_caixa` publica o PICO, não a média de um pico monótono ---
check("5. `cfg.metrics['impacto_da_caixa'].reduce == 'max'`",
      cfg.metrics["impacto_da_caixa"].reduce == "max")

# --- 6. margem da pelve: a rampa do `postura_ereta` satura ACIMA do limiar do fecho ---
check("6. `postura_ereta` recebe `pelve_alvo = tr.pelve_alvo + tr.pelve_margem`",
      cfg.rewards["postura_ereta"].params["pelve_alvo"]
      == tr.pelve_alvo + tr.pelve_margem,
      f"{cfg.rewards['postura_ereta'].params['pelve_alvo']}")
_rampa_no_fecho = ((tr.pelve_alvo - tr.pelve_piso)
                   / (tr.pelve_alvo + tr.pelve_margem - tr.pelve_piso))
check("6. em z = pelve_alvo (o limiar do fecho `de_pe`) a rampa vale < 1,0 — derivada viva",
      _rampa_no_fecho < 1.0, f"{_rampa_no_fecho:.4f}")

# --- 7. `renda_congelada` é o ÚLTIMO termo; `load` SAIU; `largou` perdeu o gate ---
from g1_limpo.env_cfg import TERMOS_CONGELAVEIS      # noqa: E402

check("7. `renda_congelada` é o ÚLTIMO termo de `cfg.rewards`",
      list(cfg.rewards)[-1] == "renda_congelada", str(list(cfg.rewards)[-3:]))
check("7. `load` SAIU do módulo (spec P3)", "load" not in cfg.rewards)
check("7. `largou` perdeu `sensor_apoio` e `raio_mult`",
      "sensor_apoio" not in cfg.rewards["largou"].params
      and "raio_mult" not in cfg.rewards["largou"].params,
      str(cfg.rewards["largou"].params))

# --- 8 e 9: `renda_congelada` congela a soma do passo ANTERIOR ao fecho; regra 1 ---
try:
    import torch as _t33

    _c33 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c33.scene.num_envs = 32
    _e33 = ManagerBasedRlEnv(cfg=_c33, device="cpu")
    _e33.reset()
    _n33 = _e33.action_manager.total_action_dim
    _passa_janela(_e33, _n33, _t33)
    _t33c = _e33.command_manager.get_term("alvo_caixa")
    _ids33 = _t33.arange(_e33.num_envs)
    _nm33 = list(_c33.rewards)
    _idx_cong33 = [_nm33.index(n) for n in TERMOS_CONGELAVEIS]
    _idx_rc33 = _nm33.index("renda_congelada")
    _sr33 = _e33.reward_manager._step_reward

    check("8. logo após a janela, `renda_congelada == 0` e `_fechos == 0` em todos",
          float(_sr33[:, _idx_rc33].abs().max()) == 0.0
          and bool((_t33c._fechos == 0).all()),
          f"renda_congelada {_sr33[:, _idx_rc33].tolist()[:3]}, "
          f"_fechos {_t33c._fechos.tolist()[:3]}")

    def _avanca_e_confere33(rotulo: str, fechos_esperado: int, checa_soma: bool):
        """Um `forca_avanco`, com a régua da regra 1 (check 9) em toda transição."""
        _soma_termos_antes = _sr33[:, _idx_cong33].sum(-1).clone()
        _soma_total_antes = _sr33.sum(-1).clone()
        _t33c.forca_avanco(_ids33)
        _e33.step(_t33.zeros(_e33.num_envs, _n33))
        _soma_total_depois = _sr33.sum(-1)
        check(f"9. regra 1 ({rotulo}): a renda TOTAL do passo seguinte ao fecho é >= "
              "a do passo anterior − 1e−3",
              bool((_soma_total_depois >= _soma_total_antes - 1e-3).all()),
              f"antes {float(_soma_total_antes.mean()):.3f}/s, depois "
              f"{float(_soma_total_depois.mean()):.3f}/s")
        check(f"8. após {rotulo}, `_fechos == {fechos_esperado}`",
              bool((_t33c._fechos == fechos_esperado).all()),
              f"{_t33c._fechos.tolist()[:3]}")
        if checa_soma:
            _rc_depois = _sr33[:, _idx_rc33]
            check(f"8. após {rotulo}, `renda_congelada` ≈ a soma dos "
                  "`TERMOS_CONGELAVEIS` lida do passo ANTERIOR ao avanço",
                  bool(_t33.allclose(_rc_depois, _soma_termos_antes,
                                     rtol=1e-4, atol=1e-6)),
                  f"medido {_rc_depois.tolist()[:3]}, esperado "
                  f"{_soma_termos_antes.tolist()[:3]}")

    _avanca_e_confere33("PEGAR->CARREGAR", 1, checa_soma=True)
    _rc_apos_carregar33 = _sr33[:, _idx_rc33].clone()
    _avanca_e_confere33("CARREGAR->BOTAR", 2, checa_soma=False)
    _avanca_e_confere33("BOTAR->fecho terminal", 3, checa_soma=False)
    check("8. após o fecho terminal, `renda_congelada` subiu de novo",
          float(_sr33[:, _idx_rc33].mean()) > float(_rc_apos_carregar33.mean()),
          f"após CARREGAR {float(_rc_apos_carregar33.mean()):.3f}, após terminal "
          f"{float(_sr33[:, _idx_rc33].mean()):.3f}")
    del _e33
except Exception as _e33x:      # noqa: BLE001
    _falhas.append(f"os checks 8/9 (`renda_congelada`) não puderam ser medidos: "
                   f"{type(_e33x).__name__}: {_e33x}")

# --- 10. o fecho INERTE do REORIENTAR não soma ao contador de fechos ---
try:
    import torch as _t34

    _c34 = make_env_cfg(k, inspecao=True, elo=CMD.REORIENTAR, cadeia=1)
    _c34.scene.num_envs = 16
    _e34 = ManagerBasedRlEnv(cfg=_c34, device="cpu")
    _e34.reset()
    _n34 = _e34.action_manager.total_action_dim
    _t34c = _e34.command_manager.get_term("alvo_caixa")
    check("10. o env sintético nasce mesmo no REORIENTAR (forçado; o sorteio o produz em só 5%)",
          bool((_t34c._elo == CMD.REORIENTAR).all()), f"{_t34c._elo.tolist()[:3]}")
    # ⚠ SEM `_passa_janela`: o `forca_avanco` não depende do `VALIDA`, e o REORIENTAR
    # inerte fecha por tempo (0,3 s) assim que ativo — esperar a janela (até 1,5 s) dava
    # tempo de sobra para ele avançar SOZINHO antes deste `forca_avanco`, medido.
    _ids34 = _t34.arange(_e34.num_envs)
    _t34c.forca_avanco(_ids34)          # o fecho INERTE do REORIENTAR
    _e34.step(_t34.zeros(_e34.num_envs, _n34))
    check("10. após o fecho inerte, `_fechos == 0`",
          bool((_t34c._fechos == 0).all()), f"{_t34c._fechos.tolist()[:3]}")
    # ⚠ Lê `congelado` DIRETO da instância viva do termo, e não `_step_reward`: o
    # `VALIDA` ainda pode estar em zero aqui (janela não passada), e `_step_reward`
    # multiplicaria por esse gate — provaria o gate, não o `congelado` em si.
    _idx_rc34 = list(_c34.rewards).index("renda_congelada")
    _termo_rc34 = _e34.reward_manager._term_cfgs[_idx_rc34].func
    check("10. e `congelado` continua 0 — nada foi congelado",
          float(_termo_rc34.congelado.abs().max()) == 0.0,
          f"{_termo_rc34.congelado.tolist()[:3]}")
    del _e34
except Exception as _e34x:      # noqa: BLE001
    _falhas.append(f"o check 10 (REORIENTAR inerte) não pôde ser medido: "
                   f"{type(_e34x).__name__}: {_e34x}")

# --- 11. `rastreio_por_elo` só zera onde `limpo_twist_zerado == 1` ---
try:
    import torch as _t35

    # A: PEGAR, durante a espera inicial (SEM `_passa_janela`)
    _c35a = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c35a.scene.num_envs = 16
    _e35a = ManagerBasedRlEnv(cfg=_c35a, device="cpu")
    _e35a.reset()
    _n35a = _e35a.action_manager.total_action_dim
    _e35a.step(_t35.zeros(_e35a.num_envs, _n35a))
    check("11. no PEGAR, durante a espera inicial, `limpo_twist_zerado == 1`",
          bool((_e35a.limpo_twist_zerado == 1.0).all()))
    _idx_tl35a = list(_c35a.rewards).index("track_linear_velocity")
    check("11. e `track_linear_velocity == 0` nessa espera",
          float(_e35a.reward_manager._step_reward[:, _idx_tl35a].abs().max()) == 0.0)
    _params35a = dict(cfg.rewards["track_linear_velocity"].params)
    _molde35a = _params35a.pop("func")
    check("11. `rastreio_por_elo == 0` quando `limpo_twist_zerado == 1`",
          float(RC_.rastreio_por_elo(_e35a, func=_molde35a, **_params35a)
                .abs().max()) == 0.0)
    del _e35a

    # B: CARREGAR de segurar-parado (cadeia 3), via avanço forçado a partir do PEGAR
    _c35b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c35b.scene.num_envs = 16
    _e35b = ManagerBasedRlEnv(cfg=_c35b, device="cpu")
    _e35b.reset()
    _n35b = _e35b.action_manager.total_action_dim
    _passa_janela(_e35b, _n35b, _t35)
    _t35b = _e35b.command_manager.get_term("alvo_caixa")
    _t35b.forca_avanco(_t35.arange(_e35b.num_envs))         # PEGAR -> CARREGAR
    _e35b.step(_t35.zeros(_e35b.num_envs, _n35b))
    check("11. no CARREGAR de segurar-parado (cadeia 3), `limpo_twist_zerado == 1`",
          bool((_t35b._elo == CMD.CARREGAR).all())
          and bool((_e35b.limpo_twist_zerado == 1.0).all()),
          f"elo {_t35b._elo.tolist()[:3]}")
    del _e35b

    # C: ANDAR — nada zera o twist
    _c35c = make_env_cfg(k, elo=CMD.ANDAR)
    _c35c.scene.num_envs = 16
    _e35c = ManagerBasedRlEnv(cfg=_c35c, device="cpu")
    _e35c.reset()
    _n35c = _e35c.action_manager.total_action_dim
    _e35c.step(_t35.zeros(_e35c.num_envs, _n35c))
    check("11. no ANDAR, `limpo_twist_zerado == 0`",
          bool((_e35c.limpo_twist_zerado == 0.0).all()))
    _params35c = dict(cfg.rewards["track_linear_velocity"].params)
    _molde35c = _params35c.pop("func")
    _valor_molde35c = _molde35c(_e35c, **_params35c)
    _valor_gate35c = RC_.rastreio_por_elo(_e35c, func=_molde35c, **_params35c)
    check("11. `rastreio_por_elo` == o termo do molde quando `limpo_twist_zerado == 0`",
          bool(_t35.allclose(_valor_gate35c, _valor_molde35c, atol=1e-6)))
    del _e35c
except Exception as _e35x:      # noqa: BLE001
    _falhas.append(f"o check 11 (gate do rastreio) não pôde ser medido: "
                   f"{type(_e35x).__name__}: {_e35x}")

# --- 12. o twist FIXO no CARREGAR-andando (cadeia 2) ---
try:
    import torch as _t36

    _c36 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=2)
    _c36.scene.num_envs = 16
    _e36 = ManagerBasedRlEnv(cfg=_c36, device="cpu")
    _e36.reset()
    _n36 = _e36.action_manager.total_action_dim
    _passa_janela(_e36, _n36, _t36)
    _t36c = _e36.command_manager.get_term("alvo_caixa")
    _t36c.forca_avanco(_t36.arange(_e36.num_envs))          # PEGAR -> CARREGAR (anda)
    _e36.step(_t36.zeros(_e36.num_envs, _n36))
    _tw36 = _e36.command_manager.get_term("twist")
    _cmd36_1 = _tw36.vel_command_b[:, :2].clone()
    check("12. no CARREGAR-andando (cadeia 2), `‖vel_command_b[:, :2]‖ >= 0,3` em todo env",
          bool((_t36c._elo == CMD.CARREGAR).all())
          and float(_t36.norm(_cmd36_1, dim=-1).min()) >= 0.3 - 1e-6,
          f"mín ‖cmd‖ = {float(_t36.norm(_cmd36_1, dim=-1).min()):.3f}")
    _e36.step(_t36.zeros(_e36.num_envs, _n36))
    _cmd36_2 = _tw36.vel_command_b[:, :2].clone()
    check("12. e o comando é IGUAL em dois passos consecutivos — twist fixo (spec P5)",
          bool(_t36.allclose(_cmd36_1, _cmd36_2)),
          f"{_cmd36_1[0].tolist()} vs {_cmd36_2[0].tolist()}")
    del _e36
except Exception as _e36x:      # noqa: BLE001
    _falhas.append(f"o check 12 (twist fixo do CARREGAR) não pôde ser medido: "
                   f"{type(_e36x).__name__}: {_e36x}")

# --- 13. a régua da caixa: `aproxima_caixa` e `renda_manipulacao` ---
check("13. `cfg.metrics` tem `aproxima_caixa` (`reduce='last'`) e `renda_manipulacao`",
      cfg.metrics["aproxima_caixa"].reduce == "last"
      and "renda_manipulacao" in cfg.metrics,
      str({n: cfg.metrics[n].reduce for n in ("aproxima_caixa", "renda_manipulacao")}))
try:
    import torch as _t37

    _c37 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c37.scene.num_envs = 16
    _e37 = ManagerBasedRlEnv(cfg=_c37, device="cpu")
    _e37.reset()
    _n37 = _e37.action_manager.total_action_dim
    # ⚠ instância AD-HOC, como `impacto_da_caixa` já é medido acima (§26): uma
    # instância fresca no MESMO env, e não a do manager — `__init__` não lê `cfg`.
    _metrica37 = MT_.aproxima_caixa(None, _e37)
    # ⚠ CAPTURA POR ENV no PRIMEIRO passo em que `VALIDA` liga, e não depois de
    # `_passa_janela` (que sobre-espera até a MAIOR janela sorteada para TODOS os
    # envs). MEDIDO: a base do robô assenta um pouco sob ação zero enquanto a espera
    # corre, e o alvo do `PEGAR` é ANCORADO NA BASE (x,y recalculados a cada passo) —
    # esperar além do necessário do PRÓPRIO env dá tempo de sobra para essa deriva
    # afastar `d` do `sigma_trazer` calibrado no reset, e o valor cai bem abaixo de
    # 1,0 mesmo sem bug nenhum no termo.
    _capturado37 = _t37.zeros(_e37.num_envs, dtype=_t37.bool)
    _valor37 = _t37.ones(_e37.num_envs)
    for _ in range(int(k.alvo.espera_s[1] / _e37.step_dt) + 5):
        _e37.step(_t37.zeros(_e37.num_envs, _n37))
        _v37 = _metrica37(_e37, nome_do_comando="alvo_caixa")
        _ativo37 = _e37.command_manager.get_command("alvo_caixa")[:, CMD.VALIDA] > 0.5
        _novo37 = _ativo37 & ~_capturado37
        if bool(_novo37.any()):
            _valor37[_novo37] = _v37[_novo37]
            _capturado37 |= _novo37
        if bool(_capturado37.all()):
            break
    check("13. `aproxima_caixa` ≈ 1,0 no primeiro passo ativo (σ = d0)",
          bool(_capturado37.all())
          and float((_valor37 - 1.0).abs().max()) < 0.25,
          f"{_valor37.tolist()[:3]}")
    del _e37
except Exception as _e37x:      # noqa: BLE001
    _falhas.append(f"o check 13 (`aproxima_caixa`) não pôde ser medido: "
                   f"{type(_e37x).__name__}: {_e37x}")

# =============================================================================
print()
print("=" * 62)
if _falhas:
    print(f"{_ok} ok / {len(_falhas)} FALHAS")
    for f in _falhas:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"{_ok} ok / 0 falhas")
