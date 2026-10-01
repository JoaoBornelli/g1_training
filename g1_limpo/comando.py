"""O comando do objetivo: ONDE a caixa deve ficar, e com que orientação.

É a ÚNICA fonte de verdade do alvo. O visualizador não reimplementa nada — ele
importa este módulo e desenha o que este termo publica. O `_debug_vis_impl` mora
aqui pela mesma razão: se o desenho vivesse no inspetor, ele seria uma segunda fonte
e mentiria no dia em que as duas divergissem.

OS CINCO ELOS, e o alvo de cada um é uma COISA DIFERENTE:

    ANDAR       uma VELOCIDADE (o twist), não um ponto. A mobília está a +5 m.
    REORIENTAR  uma ORIENTAÇÃO. A caixa fica onde está; o que muda é a face pedida.
    PEGAR       IDÊNTICO ao CARREGAR. O que difere é o twist: aqui ele é ZERO.
    CARREGAR    x,y no peito RELATIVOS ao robô; z ABSOLUTO na altura de trabalho.
    BOTAR       um ponto LATERAL num TOPO NOVO, com o teto travado no fundo da caixa.

⚠ Os ids dos elos são os MESMOS slots do one-hot da especificação. Uma numeração só.

LAYOUT DO COMANDO — canais novos entram sempre POR ÚLTIMO, para que uma migração de
checkpoint seja um APPEND de colunas e nunca uma inserção no meio:

    [0:3]  ALVO    posição alvo da caixa, em MUNDO
    [3:6]  FACE    normal da face pedida, em MUNDO (unitária)
    [6]    ANG     ângulo pedido, em radianos
    [7]    VALIDA  1,0 se o objetivo da caixa está ativo; 0,0 no `ANDAR`
    [8]    ELO     o elo corrente, como float
    [9:12] GIRO    eixo × ângulo do giro pedido, em MUNDO (spec §8.3)

⚠ O `PEGAR` e o `CARREGAR` pedem O MESMO PONTO — a âncora do peito. A diferença é o
REFERENCIAL, e ela é o desenho:

    `carregar`  RELATIVO ao robô, recalculado a cada passo: a caixa acompanha o peito
                enquanto ele anda.
    `pegar`     CONGELADO em mundo no resample: **agachar não move o alvo**, portanto
                erguer a caixa até o peito é a única forma de satisfazer.

Se o `pegar` fosse relativo, o robô satisfaria agachando até a caixa. E ele era
ABSOLUTO E FIXO em `z = (0,78; 0,85)` até 25/08 — herdado da skill Lift, que tinha a
laje travada em 0,55. Com a laje variando de 0,55 a 0,04 por nível, aquele número
fazia "erguer" valer 0,13 m no nível 0 e 0,71 m no nível 6.

ESCOPO DESTA FASE (F0/F1): o alvo de CADA elo, e o desenho. O elo é FORÇADO por knob.
A máquina de elo — a troca automática quando o elo fecha, e as cadeias de 2 —
entra na F4 e ESTENDE este termo: ela vai chamar o mesmo `_aplica_elo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (quat_apply, quat_apply_inverse, quat_apply_yaw,
                                      wrap_to_pi)

from g1_limpo.cena import JUNTAS_BRACO
from g1_limpo.curriculo import garante_elo, garante_nivel, resolve_p_c

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["AlvoCaixaCmd", "AlvoCaixaCmdCfg", "FACE_AXES", "forca_de_apoio", "avanco_prof_botar",
           "ALVO", "FACE", "ANG", "VALIDA", "ELO", "GIRO", "DIM",
           "ANDAR", "REORIENTAR", "PEGAR", "CARREGAR", "BOTAR", "ELOS", "elo_por_nome",
           "CADEIAS",
           "ESTADOS", "estado_de_recompensa",
           "ESTADO_ANDAR", "ESTADO_ESPERA_SEM", "ESTADO_ESPERA_COM",
           "ESTADO_REORIENTAR_SEM", "ESTADO_REORIENTAR_COM",
           "ESTADO_PEGAR_SEM", "ESTADO_PEGAR_COM",
           "ESTADO_CARREGAR", "ESTADO_BOTAR", "ESTADO_CAUDA",
           "TwistComRazaoDeMarcha", "TwistComRazaoDeMarchaCfg"]

# --- o layout, por nome. Nenhum índice solto no resto do pacote. ---
ALVO = slice(0, 3)
FACE = slice(3, 6)
ANG = 6
VALIDA = 7
ELO = 8
# ⚠ O VETOR DE GIRO (spec §8.3), em MUNDO: eixo × ângulo da rotação que leva a normal
# ATUAL da face pedida à direção pedida. `|GIRO| == ANG`. A observação o leva ao frame
# da base (`giro_b`). Entrou POR ÚLTIMO, como manda o contrato de append.
GIRO = slice(9, 12)
DIM = 12

# --- os elos. Mesma numeração dos slots do one-hot. ---
ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3, 4
ELOS = ("andar", "reorientar", "pegar", "carregar", "botar")

# --- o REGIME DA FACE, por elo. Ver `_atualiza_face`. ---
# ⚠⚠ DOIS REGIMES DESDE 28/09. O `FACE_CONGELADA` (PEGAR, CARREGAR) comparava a normal
# de UMA face lateral com a da abertura, e um vetor não vê o giro em torno de si mesmo:
# o tombo em torno do eixo PALMA-A-PALMA, o mais fácil com a pega bimanual, era
# invisível. MEDIDO no `model_2000` da `zero09`, PEGAR_COM: tombo real p50 93°, `ANG`
# 62°, dos quais 46° eram guinada. Requisito do dono: a face de baixo continua para
# baixo; a guinada é livre. Ver `docs/memoria/g1-limpo-ang-cego-ao-tombo.md`.
FACE_VIVA = 0       # REORIENTAR: a face marcada aponta para o robô, recalculado todo passo
FACE_DE_PE = 2      # os outros: o eixo que estava PARA CIMA na mesa, contra a vertical

# --- os DEZ estados de recompensa (spec `g1-limpo-tabela-por-estado.md` §1). ---
# A enumeração COMPLETA do que ocorre num env, publicada em `env.limpo_estado` por
# `_aplica_espera` e lida por `recompensas.PesoPorEstado`, que indexa com ela a tabela
# do `knobs.PesoPorEstado`. A ORDEM aqui é o contrato das colunas daquela tabela.
#
# ⚠ A divisão `_SEM/_COM` (por `pegou`) existe porque o rastreio e o `pose` dependem
# de já ter tocado a caixa — ela reproduz, estado a estado, o `engajado` que o
# `rastreio_por_elo` lia antes da tabela.
#
# ⚠ `knobs.py` NÃO importa isto: `cena.py` importa `knobs`, e este módulo importa
# `cena` — o import de volta fecharia o ciclo. O `knobs` rotula as colunas num
# comentário, e o `smoke` amarra os dois pelo comprimento das tuplas.
(ESTADO_ANDAR, ESTADO_ESPERA_SEM, ESTADO_ESPERA_COM,
 ESTADO_REORIENTAR_SEM, ESTADO_REORIENTAR_COM,
 ESTADO_PEGAR_SEM, ESTADO_PEGAR_COM,
 ESTADO_CARREGAR, ESTADO_BOTAR, ESTADO_CAUDA) = range(10)
ESTADOS = ("ANDAR", "ESPERA_SEM", "ESPERA_COM",
           "REORIENTAR_SEM", "REORIENTAR_COM",
           "PEGAR_SEM", "PEGAR_COM",
           "CARREGAR", "BOTAR", "CAUDA")


def estado_de_recompensa(elo: torch.Tensor, aguardando: torch.Tensor,
                         pegou: torch.Tensor, soltou: torch.Tensor) -> torch.Tensor:
    """O estado de recompensa por env (spec tabela-por-estado §1), como `long`.

    Precedência: `soltou` primeiro (-> CAUDA); depois `aguardando` (-> ESPERA_SEM ou
    ESPERA_COM, por `pegou`); depois por `elo`:

        ANDAR      -> ANDAR
        REORIENTAR -> REORIENTAR_SEM + pegou
        PEGAR      -> PEGAR_SEM + pegou
        CARREGAR   -> CARREGAR
        BOTAR      -> BOTAR

    ⚠ Função PURA, e de propósito: `_aplica_espera` a chama com os buffers frescos,
    e o `smoke` a chama com tensores sintéticos para provar a precedência e a faixa
    `range(10)` sem montar um env.
    """
    com = pegou.long()
    por_elo = torch.full_like(elo, ESTADO_ANDAR)
    por_elo = torch.where(elo == REORIENTAR, ESTADO_REORIENTAR_SEM + com, por_elo)
    por_elo = torch.where(elo == PEGAR, ESTADO_PEGAR_SEM + com, por_elo)
    por_elo = torch.where(elo == CARREGAR, torch.full_like(elo, ESTADO_CARREGAR), por_elo)
    por_elo = torch.where(elo == BOTAR, torch.full_like(elo, ESTADO_BOTAR), por_elo)
    estado = torch.where(aguardando, ESTADO_ESPERA_SEM + com, por_elo)
    return torch.where(soltou, torch.full_like(elo, ESTADO_CAUDA), estado)


# --- as cadeias de elo (spec dois-bits §2.1). O teto é DERIVADO (`_TETO_ELOS`),
# nunca redigitado. O `CARREGAR` tem DOIS papéis: na cadeia C ele é o ELO DO MEIO
# (30/09, enunciado §1: PEGAR → CARREGAR → BOTAR), parado, com fecho curto; nas cadeias
# B e R ele é o estado de CAUDA de quem fechou o PEGAR e não vai botar, escrito por
# `_aplica_espera` — não um passo da cadeia. B, R, C:
#     índice 0 (B)  (PEGAR,)                   — pegar, depois CAUDA carregar
#     índice 1 (R)  (REORIENTAR, PEGAR)        — reorientar, pegar, depois CAUDA carregar
#     índice 2 (C)  (PEGAR, CARREGAR, BOTAR)   — pegar, carregar parado, botar, depois
#                                                CAUDA botar (soltou)
CADEIAS = (
    (PEGAR,),
    (REORIENTAR, PEGAR),
    (PEGAR, CARREGAR, BOTAR),
)

# ⚠ `ANDAR` NÃO É CADEIA. Um env de locomoção recebe isto, e `n_elos_da_cadeia`
# devolve 1 para ele: não há 2º elo para avançar.
CADEIA_NENHUMA = -1

# o 1º elo de cada cadeia, e o comprimento de cada uma. Derivados de `CADEIAS`, nunca
# redigitados — uma tabela paralela escrita à mão sai de sincronia no dia em que uma
# cadeia mudar.
_PRIMEIRO_ELO = torch.tensor([c[0] for c in CADEIAS], dtype=torch.long)
_N_ELOS = torch.tensor([len(c) for c in CADEIAS], dtype=torch.long)
# CADEIAS achatada em (n_cadeias, teto_de_elos), com -1 no que não existe
_TETO_ELOS = max(len(c) for c in CADEIAS)
_ELO_EM = torch.full((len(CADEIAS), _TETO_ELOS), -1, dtype=torch.long)
for _i, _c in enumerate(CADEIAS):
    for _j, _e in enumerate(_c):
        _ELO_EM[_i, _j] = _e


def forca_de_apoio(env, nome_sensor: str) -> torch.Tensor:
    """[n] — quanto a LAJE carrega da caixa, em newtons. Só a componente VERTICAL.

    ⚠⚠ PROJEÇÃO NO EIXO VERTICAL, e não a norma. Decisão do dono em 2026-09-03, depois
    de um code review: a norma não tem direção, portanto prensar a caixa **de lado**
    contra o tampo satisfazia `apoiada` e saturava o `load` — o `BOTAR` fechava sem a
    laje carregar peso nenhum. A norma era herdada da `exp/g1-limpo`, onde só o fecho a
    lia; o `load` da v2 a herdou junto. O `g1_poc` já projetava (`recompensas.py::load`).

    ⚠ `abs` na componente z, e é escolha medida. MEDIDO em 03/09, caixa apoiada na laje:
    `f = (0,00; 0,00; −9,57)` com `m·g = 9,81`, isto é a força sai com o SINAL INVERTIDO
    do que a caixa sente, e ela é puramente vertical. Um `clamp(min=0)` sobre `−f_z`
    seria mais estrito (rejeitaria também prensar a caixa contra a face de BAIXO da
    laje), mas se um upgrade do `mjlab` inverter a ordem do par de geoms o sinal vira e o
    termo lê ZERO PARA SEMPRE em silêncio — o `BOTAR` deixaria de fechar e nada acusaria.
    Com `abs` o pior caso é o buraco pequeno de prensar por baixo; sem ele o pior caso é
    a tarefa morrer calada. O `smoke` fixa a convenção medida, para uma inversão futura
    aparecer como falha e não como treino errado.

    ⚠ O RESÍDUO, declarado: uma caixa prensada contra a ARESTA do tampo ainda gera reação
    vertical grande e ainda passa por aqui. MEDIDO: até 7×`m·g` de componente z. O que
    limita esse caso é o `perto` do fecho (`tol_pos = 0,10 m`, e a aresta fica ~0,10 a
    0,15 m abaixo do alvo) e o `precise_pos` pagando ~zero ali. A projeção mata o caso
    puramente horizontal, que era o que não tinha freio nenhum.

    ⚠ `sum` sobre os slots: com `reduce="netforce"` e `num_slots=1` é um número só, e a
    soma continua certa se alguém subir os slots.
    """
    f = env.scene[nome_sensor].data.force
    assert f is not None, f"sensor '{nome_sensor}' precisa do field 'force'."
    return f[..., 2].abs().sum(dim=-1)


def elo_por_nome(nome: str) -> int:
    try:
        return ELOS.index(nome.strip().lower())
    except ValueError:
        raise SystemExit(f"elo desconhecido: {nome!r}. Use um de {ELOS}.") from None


# As 6 faces da caixa, em coordenada LOCAL dela. Ficam aqui como DOCUMENTAÇÃO: a face
# pedida NÃO é sorteada entre elas.
#
# ⚠ O `reorientar` pede que UMA face — a marcada, `cfg.face_alvo_b` — fique normal ao
# robô. Qualquer uma das 6 chega à frente por composição de QUARTOS DE VOLTA (±90° em
# X, Y ou Z), portanto o robô precisa aprender 6 primitivas e nada mais. A dificuldade
# mora na ORIENTAÇÃO DE NASCIMENTO da caixa, e ela é sorteada em
# `eventos.orientacao_de_nascimento`.
FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)

# Raio de referência do alcance, a partir da pelve. É REFERÊNCIA, não limiar de
# recompensa.
#
# ⚠ DERIVAÇÃO, porque eu errei este número antes. Eu o pusera em 0,50 m, confundindo-o
# com o `box_xy = 0,50` medido no repositório — mas AQUELE número é a distância de
# SPAWN da caixa que rendeu só 19% de pega, e não um raio de alcance.
#
# O número defensável vem do envelope de spawn com que a skill Lift FECHOU a tarefa:
# a caixa nasce em x até `0,32 + 0,20 = 0,52`, y até ±0,18, e no nível mais baixo o
# centro dela fica em z ≈ 0,14. Com a pelve em z = 0,80:
#
#     sqrt(0,52² + 0,18² + (0,80 − 0,14)²) = 0,86 m
#
# Portanto o robô comprovadamente operou com alcances de até ~0,86 m. O raio abaixo é
# esse envelope, arredondado para baixo.
ALCANCE_R = 0.85

# ⚠ O TETO FÍSICO do topo do `BOTAR` (spec dois-bits §1.4). Era o knob
# `botar_topo_teto`; virou constante porque o topo não é mais sorteado numa faixa
# absoluta — ele deriva do topo CORRENTE (`limpo_topo`), e o guarda `teto = fundo −
# botar_folga_laje` continua sendo o que de fato limita.
_TOPO_TETO_FISICO = 0.80


def avanco_prof_botar(s_c: float, s_c_alvo: float) -> float:
    """Avanço da PROFUNDIDADE do sorteio da laje do BOTAR (R4b, spec
    `g1-limpo-botar-profundidade.md`): `clamp(s_C / s_c_alvo, 0, 1)`. 0 = laje só rasa;
    1 = todas as alturas. Sem termo `forma`, `s_C` = 1 e a profundidade é cheia."""
    return min(max(s_c / s_c_alvo, 0.0), 1.0)

# ⚠ O AVANÇO EM X DA LAJE NO BOTAR (spec dois-bits §1.4, revisão do coordenador,
# item 27). É NÚMERO MEDIDO, não sintonizável — não vira knob: `knobs.Cena.
# prateleira_xy` continua em 0,50 (o reset), e só a chamada do BOTAR usa este
# valor maior.
#
# ⚠⚠ A 1ª MEDIÇÃO (caminho do inspetor, robô TRAVADO, avanço FORÇADO) media
# pico de 282 N em `apoio_caixa` no nível 4 — mas era ARTEFATO do método: sem
# pega real, a caixa fica em cima da laje na pose do RESET, e o teleporte da
# laje (novo xy pela base + yaw) atravessa a caixa já apoiada ali. `auto_colisao`
# ficava em 0, o que confirma: não era o BOTAR real colidindo com nada.
#
# ⚠⚠ MEDIÇÃO REAL (2026-09-08, natural: policy `model_6999`, robô LIVRE,
# `cadeia_forcada=C`, 64 envs, 1100 passos, 5 passos após cada avanço PEGAR→BOTAR
# de verdade — exige `pegou ∧ perto`): nível 0, 30 avanços observados, força
# ZERO nos dois sensores. Nível 4, 30 avanços observados: `apoio_caixa` em
# 0 N, `auto_colisao` com pico de 121 N.
#
# ⚠⚠ REINTERPRETADO (revisão do coordenador): o pico de 121 N em `auto_colisao`,
# com `apoio_caixa` em ZERO, é o robô esbarrando NELE MESMO ao alcançar baixo —
# POSTURA de alcance, não contato com a laje. Mover a laje 5 cm não resolve
# isso, e afasta o alvo sem razão medida. REGRA CORRIGIDA: só `apoio_caixa`
# decide a distância da laje, e ele deu ZERO nos dois níveis — o valor volta
# a 0,50.
_AVANCO_LAJE_BOTAR = 0.50

_MAGENTA = (0.90, 0.20, 0.90, 1.00)
_CIANO = (0.20, 0.90, 0.90, 1.00)
_AMARELO = (0.95, 0.85, 0.20, 1.00)
_VERMELHO = (0.95, 0.25, 0.25, 1.00)
_VERDE = (0.25, 0.90, 0.35, 1.00)
_CINZA = (0.60, 0.60, 0.65, 0.35)
_BRANCO = (1.00, 1.00, 1.00, 0.10)


@dataclass(kw_only=True)
class AlvoCaixaCmdCfg(CommandTermCfg):
    # A ÂNCORA DO PEITO, no frame da BASE. Alvo dos DOIS elos que seguram a caixa; a
    # diferença é só o REFERENCIAL — `carregar` relativo ao robô, `pegar` congelado
    # em mundo.
    # ⚠ DERIVADO do alvo, e não medido do robô — ver `knobs.Alvo.peito_b`, fonte
    # única do valor real.
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.102)
    # ⚠ o z do alvo é ABSOLUTO nos dois elos que seguram: agachar não baixa o alvo.
    # `0,798 + peito_b.z (0,102) = 0,90` — ver `knobs.Alvo.altura_carregar`, que traz
    # o porquê de o piso ser 0,80 e não a anatomia. É só o default PRÉ-RESET: a
    # altura de verdade é sorteada em `altura_carregar_faixa`.
    altura_carregar: float = 0.90
    # ⚠ A faixa de sorteio da altura de trabalho, por episódio. Ver
    # `knobs.Alvo.altura_carregar_faixa`, fonte única do valor real.
    altura_carregar_faixa: tuple[float, float] = (0.85, 0.95)
    # os elos que exigem o robô PARADO. O twist deles é forçado a ZERO, e é isso —
    # e não a forma do alvo — que impede o robô de andar com a caixa.
    elos_parados: tuple[int, ...] = (1, 2, 4)      # REORIENTAR, PEGAR, BOTAR
    nome_do_twist: str = "twist"
    # ⚠ O SENSOR DE APOIO É `apoio_caixa`, o contato caixa<->laje declarado em
    # `cena.sensores()`. Uma versão anterior deste campo dizia `contact_caixa_laje`,
    # que NÃO EXISTE, e a leitura vinha dentro de um `try/except` cujo fallback era
    # `apoiada = True`: o `BOTAR` fechava com `perto & alinhado` apenas, em silêncio.
    nome_sensor_apoio: str = "apoio_caixa"
    # ⚠ O limiar NÃO é absoluto: ele é uma FRAÇÃO do peso da caixa. Um limiar fixo de
    # 2 N significaria "apoiada" com carga de 1 kg (9,8 N) e "no ar" com 5 kg mal
    # encostada. A caixa está apoiada quando a laje carrega metade do peso dela.
    fracao_do_peso_apoiada: float = 0.5
    # a tolerância que conta como "na condição de fechamento", em metros e radianos
    tol_pos: float = 0.10
    tol_ang_deg: float = 25.0
    # ⚠ `pelve_alvo` FICA (spec dois-bits §2.4), mas não é mais lido pelo fecho: o
    # `postura_ereta` (recompensas.py) é quem usa a altura da pelve, via
    # `knobs.Tarefa.pelve_alvo` — dois lugares com o MESMO conceito, e não um só, por
    # decisão do dono (a rampa da recompensa satura acima do limiar de fecho).
    pelve_alvo: float = 0.75
    # ⚠ `de_pe` do fecho (spec §2.4): a maior excursão de junta das PERNAS e da
    # CINTURA em relação ao default, em radianos. MEDIDO no PEGAR dos níveis 4–6
    # (laje a 0,04 m, exige agachar), com o robô DE PÉ e a caixa erguida: p90 0,69
    # rad — ver `knobs.Tarefa.de_pe_tol_rad`, fonte única do valor real.
    de_pe_tol_rad: float = 0.69
    # alvo do BOTAR — lateral, num topo novo PERTO do atual (spec dois-bits §1.4).
    # ⚠ saem `botar_x`, `botar_y`, `botar_topo_piso`, `botar_topo_teto`: o topo não é
    # mais sorteado numa faixa absoluta, ele deriva do topo CORRENTE (`limpo_topo`).
    botar_delta_topo: float = 0.10
    botar_delta_xy: float = 0.10
    botar_recuo_borda: float = 0.15
    botar_folga_laje: float = 0.05
    # Profundidade do BOTAR (R4b, spec g1-limpo-botar-profundidade.md); mesmos valores de
    # `knobs.Alvo`.
    botar_prof_s_c: float = 0.50
    botar_raso_topo_max: float = 0.64
    # geometria de que o termo precisa para mover a laje
    afasta_z: float = 5.0
    # posição da laje no RESET. O avanço do BOTAR usa `_AVANCO_LAJE_BOTAR`, uma
    # constante medida à parte — ver o comentário dela — e não este knob.
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    prateleira_meia_z: float = 0.02
    prateleira_meia_xy: float = 0.30
    # o topo da laje APOIADA no chão. É o piso físico do `BOTAR`.
    prateleira_topo_piso: float = 0.04
    caixa_meia_z: float = 0.10
    # ⚠ A meia-aresta entra no kernel de alcance: a distância medida é até a
    # SUPERFÍCIE da caixa, não até o centro. Ver `dist_palma_caixa`.
    caixa_meia_aresta: float = 0.10
    # a face MARCADA, no frame da caixa. Constante: é sempre ela que o `reorientar`
    # pede normal ao robô. A dificuldade está na ORIENTAÇÃO DE NASCIMENTO da caixa.
    face_alvo_b: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    # os sítios das palmas, para a distância que define o σ
    sitios_palma: tuple[str, ...] = ("left_palm", "right_palm")
    # a JANELA DE ESPERA, em segundos, sorteada por episódio. Ver `knobs.Alvo`.
    espera_s: tuple[float, float] = (0.3, 1.0)
    # os SENSORES de palma, para armar o `caixa_largada`. Só o campo `found` é lido:
    # a pergunta é "as duas palmas já tocaram a caixa neste episódio", e ela é
    # booleana por natureza. A FORÇA é assunto do `squeeze`, que é contínuo.
    sensores_palma: tuple[str, ...] = ("palma_E", "palma_D")

    # ⚠ O σ NÃO É UM NÚMERO: ele é a DISTÂNCIA INICIAL daquele env, vezes este fator.
    # Ver o bloco de σ no `__init__` e a §4.2b da spec. Medido: com σ fixo de 0,10 a
    # 0,339 m o kernel vale 1e−05 e a DERIVADA É ZERO — o robô não tem pista de onde
    # ir, e foi isto que travou o `g1_poc`.
    #
    # ⚠ PRÉ-REGISTRADO: se o alcance não aparecer na F3, este fator vai a 1,5. É o
    # PRIMEIRO e ÚNICO número a mover, e NUNCA o peso.
    sigma_fator: float = 1.0
    sigma_min: float = 0.08          # metros
    sigma_ori_min: float = 0.20      # radianos (~11°)

    # o elo. `None` = todos em `PEGAR` (o único que a F0/F1 treina).
    elo_forcado: int | None = None
    # ⚠ NUNCA igual à duração do episódio. Com `(20, 20)` o `time_left` do comando
    # cruza zero no passo 999 e o `time_out` da terminação só dispara no 1000: o
    # resample rodava UM PASSO antes do fim e zerava o sucesso do episódio. O nível
    # lia sucesso 0 em TODO episódio que chegava ao time_out. A meta é 1 resample por
    # episódio, e quem resampleia é o RESET.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = True

    # --- F4: máquina de elo ---
    cadeia_forcada: int | None = None    # índice em CADEIAS. Inspetor e play.
    # ⚠ O INTERRUPTOR DA MÁQUINA DE ELO (spec dois-bits §2.1). `prob_por_nivel = ()`
    # era o desliga; virou este bool explícito. Com `False`, `_resample_command` não
    # sorteia cadeia nenhuma — o mesmo comportamento de antes com a tabela vazia.
    cadeia_ativa: bool = True
    sustenta_pegar_s: float = 0.5
    # ⚠ 0,3 → 0,5 s na v3.4 (spec `g1-limpo-botar-fecha-e-para.md` §2.3). O ESPELHO
    # deste número é `knobs.Cadeia.sustenta_outros_s`, e `env_cfg` copia de lá — mude
    # os dois, ou eles derivam em silêncio.
    sustenta_outros_s: float = 0.5
    # ⚠ O BALANCEADOR B/C (spec dois-bits §2.5): decide, para quem começa no PEGAR,
    # entre a cadeia B (segurar e carregar) e a C (botar). `piso` trava `p_C` numa
    # faixa (nunca 0 nem 1, pela mesma regra do `fatia_loco`: um slot do one-hot
    # constante entraria como ×100 no normalizador do dia em que acendesse). `alpha` é
    # o ganho da EMA de `concluiu` por cadeia, aplicada uma vez por ITERAÇÃO de PPO.
    balanceador_piso: float = 0.20
    balanceador_alpha: float = 0.05
    # ⚠⚠ O CURRÍCULO DE CADEIA (spec g1-limpo-curriculo-de-cadeia §4 e §5, 30/09): os dez
    # campos abaixo ESPELHAM `knobs.Cadeia`, e `env_cfg` copia de lá — mude os dois, ou
    # eles derivam em silêncio. Ver o `knobs.Cadeia` para o porquê de cada número.
    # ⚠ Só o default de `fase_inicial` DIFERE: aqui é 4 (o comportamento de hoje), porque
    # o cfg cru é o que a inspeção e os testes montam. Quem treina recebe o knob (1) pelo
    # `env_cfg`, e `inspecao`/`play` o fixam em 4 lá.
    fase_inicial: int = 4
    fracao_cauda_fase1: float = 0.10
    fracao_anda_fase2: float = 0.10
    p_c_antes_do_botar: float = 0.05
    fase2_s_b: float = 0.50
    fase2_min_iters: int = 300
    fase3_s_cauda: float = 0.60
    fase3_min_iters: int = 200
    fase4_s_cauda: float = 0.60
    fase4_min_iters: int = 300
    # ⚠ O INTERRUPTOR DO REORIENTAR (spec §8.3, v2): com `True` o fecho do REORIENTAR
    # ignora `alinhado` e o elo fecha em `sustenta_outros_s` sem trabalho. MEDIDO em
    # 03/09: `voltas_max = 0` não bastava — a direção pedida é "da caixa para o robô", e
    # com o jitter lateral da caixa ela sai até ~29° do eixo; ~1 em 6 envs nascia fora
    # dos 25° e o elo não fechava. Desligar (False) é o primeiro passo quando a
    # reorientação virar foco.
    reorientar_inerte: bool = False

    def build(self, env: "ManagerBasedRlEnv") -> "AlvoCaixaCmd":
        return AlvoCaixaCmd(self, env)


class AlvoCaixaCmd(CommandTerm):
    cfg: AlvoCaixaCmdCfg

    def __init__(self, cfg: AlvoCaixaCmdCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self.caixa: Entity = env.scene["box"]
        self.prateleira: Entity = env.scene["table"]
        self.robot: Entity = env.scene["robot"]

        d, n = self.device, self.num_envs
        self._command = torch.zeros(n, DIM, device=d)
        self._face_b = torch.tensor(cfg.face_alvo_b, device=d)
        # --- O PEDIDO DE ATITUDE, e se ele é VIVO ou DE PÉ.
        #
        # ⚠ DOIS PEDIDOS. No `REORIENTAR` a direção é VIVA: "vire a face para o robô",
        # recalculada todo passo. Nos outros elos o pedido é "a face de baixo continua
        # para baixo": o eixo que estava PARA CIMA na mesa (`_cima_b`) contra a vertical.
        # A guinada fica livre, por decisão do dono (28/09).
        #
        # Até 28/08 a direção era viva em TODO elo, e o alvo se movia com o ROBÔ: andar
        # em volta da caixa mudava o termo sem tocar nela.
        self._regime_face = torch.full((n,), FACE_DE_PE, dtype=torch.long, device=d)
        # o eixo Z, no frame da CAIXA e no do MUNDO. Os dois são o mesmo vetor e mesmo
        # assim ficam separados: um é "que eixo da caixa eu meço", o outro é "para onde
        # ele tem de apontar", e juntá-los num só some com a distinção no dia em que o
        # segundo estágio (a pose de destino) trocar só o segundo.
        self._ez_b = torch.tensor([0.0, 0.0, 1.0], device=d)
        self._ez_w = torch.tensor([0.0, 0.0, 1.0], device=d)
        # o eixo de cima da CAIXA, no frame dela: a vertical capturada na mesa. No reset
        # é o Z; o `_captura_cima` o reescreve onde a caixa está apoiada.
        self._cima_b = self._ez_b.repeat(n, 1)
        self._elo = torch.full((n,), PEGAR, dtype=torch.long, device=d)
        # ⚠⚠ O `_pendente` existe por causa de uma armadilha MEDIDA em 25/08.
        #
        # No reset o command manager roda DEPOIS dos eventos que reposicionam a caixa
        # e a laje, mas os buffers de `data` das entidades **ainda não foram
        # recomputados**. Portanto tudo o que depende de POSE é lixo aqui.
        #
        # O que isso quebrou, medido: o teto do `BOTAR` é
        # `min(fundo_da_caixa − folga, teto_do_knob)`, e com a pose obsoleta o fundo
        # deu negativo — o teto colapsou no piso e o topo saiu 0,300 nos OITO envs.
        # Só o `maximum(teto, piso)` impediu uma laje enterrada, e o clamp real nunca
        # aconteceu. O alvo do `ANDAR` saiu 0,000 pelo mesmo motivo.
        #
        # Solução: marcar o env como PENDENTE no resample e concluir a parte
        # dependente de pose no primeiro `_update_command`, quando a pose está fresca.
        self._pendente = torch.zeros(n, dtype=torch.bool, device=d)

        # ⚠⚠ O `_sigma_pendente` existe por uma SEGUNDA armadilha, distinta da do
        # `_pendente` acima (spec `g1-limpo-espera-sigma-e-pose.md` §1, 2026-09-08). O
        # `_pendente` resolve "a pose está fresca?"; este resolve "a TAREFA já começou?".
        #
        # A janela de espera (`knobs.Alvo.espera_s`, 0,5 a 1,5 s) mantém `VALIDA = 0`
        # depois do reset, e o robô aproxima as mãos de graça nesse intervalo — o
        # `PosturaPorElo` (spec dois-bits §3.1) é quem paga por ele NÃO fazer isso
        # (o braço só sai da média com `pegou`, e na espera `pegou` ainda é falso),
        # mas nada o impede fisicamente. Calcular o σ no reset mede a distância
        # ERRADA: a de antes de a tarefa existir, não a de quando ela liga.
        #
        # MEDIDO no `play` do `bloco9` em 2026-09-08: σ fixado em 0,34 m no reset, mão a
        # 0,20 m no fim da espera — `alcancar = exp(−(0,20/0,34)²) = 0,71` em vez dos
        # `exp(−1) = 0,368` que o docstring do `_alcancar` promete.
        #
        # Fica verdadeiro do resample até o primeiro passo em que `VALIDA` acende
        # (`_aplica_espera`), e nunca mais — no `ANDAR` de locomoção `VALIDA` nunca
        # acende, e o buffer fica pendente o episódio todo, sem efeito: nenhum termo de
        # manipulação lê σ com `VALIDA = 0`.
        self._sigma_pendente = torch.zeros(n, dtype=torch.bool, device=d)

        # ⚠ Onde a base estava quando o elo corrente ABRIU. O CARREGAR não fecha mais
        # (spec dois-bits §2.4: `andou`, `carregar_dist_m` e o ramo CARREGAR do fecho
        # saíram), mas o BOTAR ainda lê a pose CORRENTE da base na abertura — não este
        # buffer, que ficaria um elo atrasado (ver `_aplica_elo`, ramo BOTAR).
        self._pos_no_elo = torch.zeros(n, 3, device=d)

        # ⚠ O TWIST FIXO do CARREGAR-andando (v2.1, spec P5) SAIU (spec dois-bits §1.1):
        # o CARREGAR agora é o estado de CAUDA, e recebe o twist do fabricante SEM
        # filtro — decisão do dono: a cauda de B é `normal`.

        # ---------------------------------------------------------- F4: máquina de elo
        # Os buffers que controlam o avanço entre elos.
        self._cadeia = torch.zeros(n, dtype=torch.long, device=d)
        self._passo = torch.zeros(n, dtype=torch.long, device=d)  # 0 .. _TETO_ELOS-1
        self._sust = torch.zeros(n, dtype=torch.float, device=d)  # cronômetro em s
        # ⚠ O ALVO do cronômetro acima, por env (v2.1, spec P2). Escrito em TODA
        # abertura de elo — reset (`_resample_command`) e avanço (`_avanca_elo_force`)
        # — com a MESMA regra que `_avanca_elo` lia inline: uma fonte só. `sustentacao`
        # (recompensas.py) SAIU no dois-bits; hoje só `_avanca_elo` lê este buffer,
        # para saber quando o `_sust` acumulado basta para armar o fecho.
        self._sustain_alvo = torch.zeros(n, dtype=torch.float, device=d)
        self.fechou = torch.zeros(n, dtype=torch.bool, device=d)
        # ⚠ O CONTADOR DE FECHOS GANHOS, por env (v2.1, spec P3). Sobe UM em todo fecho
        # de elo — avanço dentro da cadeia OU fecho terminal —, exceto o inerte do
        # REORIENTAR. `recompensas.renda_congelada` lê a SUBIDA deste contador para
        # saber em que passo congelar a soma dos termos do elo que acabou de fechar.
        self._fechos = torch.zeros(n, dtype=torch.long, device=d)

        # Métricas publicadas para o log (seção 4 do contrato F4)
        # ⚠ Todas são float para que `reset` possa tirar a média
        z = torch.zeros(n, dtype=torch.float, device=d)
        self.metrics["sucesso"] = z.clone()
        self.metrics["passo_final"] = z.clone()
        self.metrics["avancos"] = z.clone()
        self.metrics["fatia_cadeia"] = z.clone()
        # ⚠ K12 (lote 01/10, Parte 3; enunciado §7 item 6): métricas SEM PESO da cadeia C —
        # nenhuma recompensa as lê. Flags valem 1 até o fim do episódio; razões no log:
        # `c_chegou_botar/c_episodio`, `c_desvio_pegar/c_episodio`,
        # `c_desvio_carregar/c_chegou_carregar`.
        for _m in ("c_episodio", "c_chegou_carregar", "c_chegou_botar", "c_desvio_pegar",
                   "c_desvio_carregar", "cauda_twist_herdado", "c_botar_topo", "c_fechou_botar",
                   "c_fecho_topo"):
            self.metrics[_m] = z.clone()
        # fração dos passos de cauda andando com comando linear < 0,05 m/s (twist herdado)
        self._n_cauda_anda = torch.zeros(n, device=d)
        self._n_cauda_herdado = torch.zeros(n, device=d)

        # ---------------------------------------------------------- os σ POR ENV
        # ⚠ ELES NÃO SÃO KNOBS. Cada um é a DISTÂNCIA INICIAL daquele env, medida no
        # instante em que o elo abre. É a decisão de maior consequência da F3, e ela
        # vem de medição (spec §4.2b):
        #
        #   a palma nasce a 0,339 m da caixa (mín 0,211, máx 0,481). Com σ FIXO de
        #   0,10 o kernel `exp(−d²/σ²)` vale 1e−05 ali, E A DERIVADA É ZERO. O robô
        #   move a mão 1 cm para perto e nada muda; 1 cm para longe e nada muda. Não
        #   existe pista de onde ir. Foi isto que travou o `g1_poc`, e não uma
        #   preferência do robô por ficar parado.
        #
        # Com `σ = d₀`, todo env nasce em `exp(−1) = 0,368` com derivada
        # `2/d₀ × 0,368`: 3,49 no env mais perto e 1,53 no mais longe. Vivo nos dois
        # extremos, e sem número mágico.
        #
        # ⚠ ELES NÃO ENTRAM NA OBSERVAÇÃO, e isso é decisão. Publicar o σ diria à
        # política "este env é fácil/difícil", e ela condicionaria a ação à forma da
        # recompensa em vez de à tarefa. σ é moldagem, não estado do mundo.
        self.sigma_alcance = torch.full((n,), cfg.sigma_min, device=d)
        self.sigma_trazer = torch.full((n,), cfg.sigma_min, device=d)
        self.sigma_ori = torch.full((n,), cfg.sigma_ori_min, device=d)

        # os sítios das palmas, resolvidos UMA vez
        self._ids_palma, _ = self.robot.find_sites(list(cfg.sitios_palma))

        # ⚠ AS JUNTAS DE PÉ, o COMPLEMENTO de `JUNTAS_BRACO` (spec dois-bits §2.4). O
        # `de_pe` do fecho passa a ler a POSE das pernas e da cintura, e não a altura da
        # pelve — resolvidas UMA vez, aqui, e não a cada passo.
        _ids_braco, _ = self.robot.find_joints(list(JUNTAS_BRACO))
        _braco = set(_ids_braco)
        self._ids_de_pe = [i for i in range(len(self.robot.joint_names))
                           if i not in _braco]

        # ---------------------------------------------- a ARMA do `caixa_largada`
        # ⚠ "As duas palmas já tocaram a caixa NESTE episódio." A terminação de caixa
        # largada é armada por ela e nunca antes: no reset a caixa está na laje e as
        # palmas estão longe, que é exatamente a condição de `escapou`. Sem a arma,
        # todo episódio começaria terminando.
        #
        # ⚠ Ela mora AQUI, e não na terminação, porque aqui existe escopo de episódio:
        # o `_resample_command` roda no reset e zera o buffer. Um termo de terminação é
        # uma função sem `reset`, e o estado dele vazaria de um episódio para o outro.
        # ---------------------------------------------- a JANELA DE ESPERA (02/09)
        # ⚠ Segundos RESTANTES, por env. Enquanto > 0 num elo de manipulação, o bit
        # `VALIDA` fica em zero: os sete incentivos pagam nada e o elo não fecha. Ver
        # `knobs.Alvo.espera_s` para o porquê e para a origem no `g1_poc`.
        self._espera = torch.zeros(n, device=d)
        env.limpo_aguardando = torch.zeros(n, device=d)
        # ⚠ O ESTADO DE RECOMPENSA (spec tabela-por-estado §1): um inteiro em
        # `range(10)` por env, escrito IN-PLACE por `_aplica_espera` do MESMO
        # `aguardando` que escreve o `VALIDA`, e lido por `recompensas.PesoPorEstado`.
        # Nasce `ANDAR` (0); a leitura real começa no primeiro `_update_command`.
        env.limpo_estado = torch.zeros(n, dtype=torch.long, device=d)
        # ⚠ A MÁSCARA "esta tarefa zerou o twist deste env", por env (v2.1, spec P4).
        # Publicada por `_zera_twist_nos_parados`. Desde a tabela por estado NENHUMA
        # recompensa a lê (o gate do rastreio virou `limpo_estado`); quem lê é a
        # âncora do alvo do CARREGAR em `_update_command`, e o `smoke`.
        env.limpo_twist_zerado = torch.zeros(n, device=d)
        # ⚠ O RUMO A SEGURAR enquanto parado (17/09): o `heading_w` do passo em que o env
        # PAROU. `_zera_twist_nos_parados` escreve wz = k × (rumo_ref − rumo) nele.
        self._rumo_ref = torch.zeros(n, device=d)
        # ⚠ A ESPERA FINAL (spec §6.6): depois do fecho do BOTAR, o publicado é ANDAR
        # até o fim do episódio; o interno segue BOTAR. `soltou` desarma o `escapou` da
        # terminação e liga o `largou` da recompensa.
        self._soltou = torch.zeros(n, dtype=torch.bool, device=d)
        env.limpo_soltou = self._soltou.float()
        # ⚠ O ELO INTERNO, publicado para o crítico e para as recompensas de caixa. É
        # uma REFERÊNCIA a `_elo`, que só é escrito in-place (`self._elo[ids] = ...`),
        # portanto ela nunca fica obsoleta; `_update_command` a republica por segurança.
        env.limpo_elo_interno = self._elo

        self._pegou = torch.zeros(n, dtype=torch.bool, device=d)
        # ⚠ O CARREGAR PARADO (spec g1-limpo-curriculo-de-cadeia §4, 30/09): "este env está
        # num CARREGAR com comando de andar ZERO". Verdadeiro na cauda parada das fases 1 e
        # 2, e SEMPRE no elo CARREGAR da cadeia C (PEGAR → carregar parado → BOTAR). Escrita
        # UMA vez, na entrada da cauda ou no avanço (`_aplica_espera`), lida por
        # `_zera_twist_nos_parados`; zera no reset. Fora do CARREGAR ela não diz nada.
        self._carregar_parado = torch.zeros(n, dtype=torch.bool, device=d)
        env.limpo_ids_palma = self._ids_palma
        # ⚠ `_forcado` (spec §2.3, revisão independente item A8): `forca_avanco`
        # zera `_espera`, mas sem isto o gate `perto` de `_aplica_espera` ainda
        # bloqueia quem `pegou` e está longe do alvo — `--avanca-elo` do viewer
        # congelava com a caixa fora de posição. Lido como `perto | _forcado`, e
        # limpo assim que o avanço acontece.
        self._forcado = torch.zeros(n, dtype=torch.bool, device=d)
        # ⚠ A altura de trabalho é POR ENV e sorteada no reset (`_resample_command`).
        # Nasce no valor fixo do cfg para o caso de alguém ler o alvo antes do
        # primeiro reset — inspeção e paridade rodam assim.
        self._altura_alvo = torch.full((n,), float(cfg.altura_carregar), device=d)
        # ⚠ Publica ZEROS aqui, e não o resultado de `_publica_pegou`: no `__init__` os
        # buffers de sensor ainda não foram preenchidos. A leitura real começa no
        # primeiro `_update_command`.
        env.limpo_pegou = self._pegou.float()

    def _zera_aproxima_caixa(self, ids: torch.Tensor) -> None:
        """Zera o mínimo corrente de `metricas.aproxima_caixa` (spec dois-bits §2.3,
        revisão item 24). Chamado no AVANÇO de elo: o alvo mudou, e o mínimo do elo
        anterior não descreve o novo.
        """
        mm = getattr(self._env, "metrics_manager", None)
        # ⚠ `or {}`, e não só o `getattr` (revisão independente, item A9): o
        # `NullMetricsManager` do `play` tem `.cfg = None` — um atributo que EXISTE
        # com valor `None` não aciona o default do `getattr`, e `None.get(...)`
        # explode. Isto quebraria o `play` toda vez.
        cfg_term = (getattr(mm, "cfg", None) or {}).get("aproxima_caixa") if mm else None
        if cfg_term is not None and hasattr(cfg_term.func, "minimo"):
            cfg_term.func.minimo[ids] = 1.0

    def _aplica_espera(self) -> None:
        """Decrementa a espera; avança de elo ou entra na CAUDA; escreve o PUBLICADO
        e o `VALIDA` (spec `g1-limpo-dois-bits.md` §2.3).

        ⚠⚠ O AVANÇO DE ELO MORA AQUI, e não mais em `_avanca_elo_force` (spec §2.2):
        o fecho de um elo só ARMA a espera. É aqui, no fim dela, que `_passo`/`_elo`
        avançam para o próximo elo da cadeia, ou a cadeia entra na CAUDA (`CARREGAR`
        para B/R e para quem falha o `perto`; fica em `BOTAR` para C). Roda ANTES da
        publicação, para que o publicado e o `VALIDA` já reflitam o elo NOVO no mesmo
        passo:

            acabou   = fechou ∧ ¬aguardando
            tem_prox = (passo + 1 < n_elos) ∧ (elo ≠ CARREGAR)
            avanca   = acabou ∧ tem_prox ∧ (perto, SE já pegou)
            cauda    = acabou ∧ ¬avanca ∧ ¬já-em-cauda

        ⚠ `_perto` É RECONFERIDO no fim da espera, e só para quem JÁ PEGOU a caixa
        (revisão, item 10/32): com `push_robot` ativo o robô deriva durante a espera,
        e sem reconferir ele avançaria com a caixa longe do alvo. Quem falha `_perto`
        vai direto à cauda CARREGAR (29/09) — inclusive na cadeia C, que não tem
        CARREGAR como elo próprio — e não fica retentando no elo fechado — ver o
        motivo no comentário do `cauda` abaixo. Quem ainda não pegou (ex.:
        REORIENTAR fechando) não precisa dela — ali o alvo É a própria caixa, e
        `perto` é trivial.

        ⚠ `pendente = _sigma_pendente` guarda o avanço E a cauda: reaproveita um estado
        que já existe, em vez de um buffer novo. Funciona porque `_sigma_pendente` só
        volta a `True` num fecho novo (`_avanca_elo_force`). Por isso ele é lido AQUI,
        antes do bloco de σ rodar.

        ⚠⚠ NA CAUDA DO BOTAR A FLAG NÃO É MAIS LIMPA, e o bloco da cauda REEXECUTA
        todo passo (spec `g1-limpo-cauda-parada-de-pe.md` §2.1): entrar na cauda exige
        `_sigma_pendente = True`, e desde a v3.5 o `liga` abaixo lê o `VALIDA` já
        zerado pelo `soltou`, portanto nunca dispara ali. É INÓCUO — `vira_carregar` e
        `fica` são filtrados por `~_soltou` e ficam vazios, e sobra `_forcado = False`,
        idempotente. O que deixa de rodar na cauda C é `_recalcula_sigmas` (todo leitor
        de σ é `× VALIDA`, zerado) e `_pos_no_elo` (write-only no pacote).

        ⚠⚠ TUDO É RECALCULADO, e não lido de um canal que este método já escreveu num
        passo ANTERIOR. Uma versão antiga fazia `where(aguardando, 0,
        self._command[:, VALIDA])` — DESTRUTIVO: no passo seguinte lia o zero que ela
        mesma tinha escrito, e o bit nunca voltava a 1. Medido no smoke em 02/09. Ler o
        `ELO` publicado no `VALIDA` (v3.5) NÃO é esse defeito: a linha acima reescreve
        o canal inteiro a partir de `_elo` e `publica_andar`, todo passo.

            publicado = ANDAR   se soltou ∨ (aguardando ∧ ¬pegou), senão o interno
            VALIDA    = (PUBLICADO ≠ ANDAR) ∧ ¬aguardando

        ⚠⚠ `publicado` MUDOU (revisão do PM, item 2): antes era `ANDAR` em TODA
        espera. Agora, com a caixa JÁ na mão (`pegou`), a espera ENTRE elos publica o
        INTERNO — a caixa fica visível, e o crítico vê o estado real. Só a espera
        ANTES da primeira pega (`¬pegou`) publica `ANDAR` com os canais zerados.

        ⚠⚠ A espera FINAL (`soltou`) publica ANDAR e AGORA ZERA O VALIDA (v3.5, spec
        `g1-limpo-cauda-parada-de-pe.md` §2.1). A regra anterior — "não zera, os
        incentivos do estado 'caixa apoiada no alvo' continuam pagando" — valia
        enquanto a cauda mandava a caixa a +5 m e os termos paravam sozinhos. A v3.4
        tirou o teleporte, e eles voltaram a pagar AO VIVO por cima da renda já
        congelada no fecho: `precise_pos` (3,0) e `load` (2,0) contados duas vezes.
        Quem paga o estado "apoiada" na cauda é o `renda_congelada`, uma vez só.

        ⚠ Publica `env.limpo_aguardando` e `env.limpo_soltou` para as métricas e para a
        terminação. Sem elas, "o robô não espera" e "a janela não existe" leem igual.

        ⚠⚠ O σ NASCE AQUI, e não no reset (F1, spec `g1-limpo-espera-sigma-e-pose.md`
        §1). `VALIDA` acabou de ser escrito acima, DEPOIS da espera — portanto ele já é
        o do passo CORRENTE, e não o do passo anterior. Para quem tinha `_sigma_pendente`
        e viu `VALIDA` acender agora, o σ e o `_pos_no_elo` são calculados com a pose
        FRESCA de agora, que é a distância que a TAREFA de fato começa medindo — e não a
        do reset, quando o objetivo ainda nem existia. Ver o `⚠⚠` do `_sigma_pendente`
        no `__init__` para o defeito medido que isto conserta.
        """
        d = self.device
        todos = torch.arange(self.num_envs, device=d)

        self._espera.sub_(self._env.step_dt).clamp_(min=0.0)
        aguardando = self._espera > 0.0
        self._env.limpo_aguardando.copy_(aguardando.float())
        # ⚠ IN-PLACE, como o `limpo_aguardando` acima. Rebindar um tensor novo a cada
        # passo deixaria qualquer referência guardada lendo dado velho — e o fecho do
        # `BOTAR` escreve neste mesmo buffer por índice (ver `_avanca_elo_force`).
        self._env.limpo_soltou.copy_(self._soltou.float())
        self._env.limpo_elo_interno = self._elo

        # --- fim da espera: avança pro próximo elo, ou entra na cauda ---
        acabou = self.fechou & ~aguardando
        n_elos = self.n_elos_da_cadeia(todos)
        # ⚠ `& pendente`: a decisão do fim da espera está PENDENTE só entre o fecho (que
        # arma `_sigma_pendente`) e a primeira passada por aqui. Quem já decidiu — cauda ou
        # SEGURA — fica com `_sigma_pendente = False` e NUNCA reavalia: o env da cadeia C
        # desviado para a cauda (`fechou=True`, `_passo` parado no elo que falhou)
        # avançaria ao BOTAR assim que a caixa voltasse ao peito, e o `concluiu` contaria
        # sucesso de uma cadeia que de fato falhou. Até 30/09 o guarda era `elo !=
        # CARREGAR`; com o CARREGAR elo da cadeia C, ele barraria o avanço legítimo.
        pendente = self._sigma_pendente
        tem_prox = ((self._passo + 1) < n_elos) & pendente
        avanca = acabou & tem_prox
        if bool(self._pegou.any()):
            # ⚠ `perto | _forcado` (revisão independente, item A8): `_forcado`
            # contorna o gate para quem `forca_avanco` armou longe do alvo — senão
            # `--avanca-elo` do viewer congela pra sempre num env fora de posição.
            perto = self._perto(todos) | self._forcado
            avanca = torch.where(self._pegou, avanca & perto, avanca)

        ids_avanca = todos[avanca]
        if len(ids_avanca):
            self._forcado[ids_avanca] = False
            cad = self._cadeia[ids_avanca]
            prox = self._passo[ids_avanca] + 1
            self._passo[ids_avanca] = prox
            self._elo[ids_avanca] = _ELO_EM.to(d)[cad, prox]
            na_c = cad == 2
            self.metrics["c_chegou_carregar"][
                ids_avanca[na_c & (self._elo[ids_avanca] == CARREGAR)]] = 1.0
            self.metrics["c_chegou_botar"][
                ids_avanca[na_c & (self._elo[ids_avanca] == BOTAR)]] = 1.0
            self.fechou[ids_avanca] = False
            self._sust[ids_avanca] = 0.0
            self._sustain_alvo[ids_avanca] = self._sustain_alvo_de(ids_avanca)
            # ⚠ o BOTAR (§1.4): laje ±δ e alvo na borda. `so_pose=False` porque a
            # pose JÁ está fresca — isto roda no passo, não no reset.
            self._aplica_elo(ids_avanca, so_pose=False)
            # ⚠ O ELO CARREGAR da cadeia C é PARADO (30/09, enunciado §1: "pegar, carregar
            # parado, botar"): o robô segura a caixa no alvo de transporte, de pé, e só
            # depois abre o BOTAR. `_zera_twist_nos_parados` lê a marca.
            self._carregar_parado[ids_avanca[self._elo[ids_avanca] == CARREGAR]] = True
            self._recalcula_sigmas(ids_avanca)
            self._pos_no_elo[ids_avanca] = self.robot.data.root_link_pos_w[ids_avanca]
            self._sigma_pendente[ids_avanca] = False
            # ⚠ `avancos` mudou de lugar (revisão, item 28): antes era escrito no
            # fecho (`_avanca_elo_force`); agora é aqui, no avanço de verdade.
            self.metrics["avancos"][ids_avanca] += 1.0
            self._zera_aproxima_caixa(ids_avanca)

        # ⚠ Quem fecha o PEGAR e falha o `perto` no fim da espera ficava PRESO nele
        # (`fechou = True`), pagando os sete termos ao vivo MAIS o congelado — 32,3/s
        # MEDIDO no `model_1600`, contra 20,7 no BOTAR. Agora vai à cauda CARREGAR,
        # como a cadeia B: baixar a caixa na espera desvia da rota do BOTAR, que é
        # perda (29/09). O mesmo `pendente` de cima: a decisão é UMA por fecho.
        cauda = acabou & ~avanca & pendente
        ids_cauda = todos[cauda]
        if len(ids_cauda):
            self._forcado[ids_cauda] = False
            vira_carregar = ids_cauda[self._pegou[ids_cauda] & ~self._soltou[ids_cauda]]
            # ⚠⚠ O CURRÍCULO DE CADEIA, fase 1 (spec g1-limpo-curriculo-de-cadeia §4, 30/09).
            # Só `fracao_cauda_fase1` dos fechos B e R vai à cauda CARREGAR; o resto SEGURA:
            # o elo fica PEGAR, `fechou`, até o fim. O estado de recompensa segue PEGAR_COM
            # e os termos ao vivo pagam por cima da renda congelada (~32/s medido, contra
            # ~17 a 25/s de pairar). A escolha é por env, UMA vez, na entrada da cauda.
            # ⚠ A cadeia C (2) desviada vai SEMPRE à cauda, e a cauda dela é PARADA em toda
            # fase (abaixo): a C pratica o CARREGAR como ELO (o avanço acima), e não como
            # SEGURA; quem falhou o `perto` no fim da espera do PEGAR ou do CARREGAR vira
            # cauda parada, como sempre.
            fase = self._fase()
            if fase == 1 and len(vira_carregar):
                vai = ((torch.rand(len(vira_carregar), device=d) < self.cfg.fracao_cauda_fase1)
                       | (self._cadeia[vira_carregar] == 2))
                segura = vira_carregar[~vai]
                vira_carregar = vira_carregar[vai]
                # ⚠ a tarefa CONTINUA: sem reabrir σ, alvo e eixo de cima no bloco `liga`
                # abaixo, que só dispara com `_sigma_pendente`. E sem `_sigma_pendente` o
                # env sai de `cauda` e de `avanca` (`pendente`): a escolha não se repete.
                self._sigma_pendente[segura] = False
            if len(vira_carregar):
                # K12: de onde a C desviou (lê o `_elo` ANTES de virar CARREGAR)
                ids_c = vira_carregar[self._cadeia[vira_carregar] == 2]
                self.metrics["c_desvio_pegar"][ids_c[self._elo[ids_c] == PEGAR]] = 1.0
                self.metrics["c_desvio_carregar"][ids_c[self._elo[ids_c] == CARREGAR]] = 1.0
                self._elo[vira_carregar] = CARREGAR
                self._alvo_ancorado_na_base(vira_carregar)
                # ⚠ A CAUDA PARADA (spec §4): fase 1, sempre; fase 2, em 90% (os outros
                # 10% andam, `fracao_anda_fase2`); fase 3 em diante, nunca. Lida por
                # `_zera_twist_nos_parados`, que a conta como parado.
                if fase <= 1:
                    self._carregar_parado[vira_carregar] = True
                elif fase == 2:
                    self._carregar_parado[vira_carregar] = (
                        torch.rand(len(vira_carregar), device=d) >= self.cfg.fracao_anda_fase2)
                else:
                    self._carregar_parado[vira_carregar] = False
                # ⚠ A cauda da cadeia C DESVIADA é SEMPRE parada, em toda fase (spec K1b
                # `g1-limpo-rastreio-carregar-parado.md`): andando ela herdava twist zero por
                # ~2,9 s e o desvio rendia mais que avançar (rastreio ×3,5 contra ×1 do elo).
                self._carregar_parado[vira_carregar[self._cadeia[vira_carregar] == 2]] = True
            # senão, o `_elo` FICA `BOTAR` (revisão, item 3): o crítico vê o interno
            # BOTAR, o publicado ANDAR (via `soltou`), e prevê a renda congelada.
            # ⚠ O PÓS-BOTAR NÃO MEXE NA CENA (decisão do dono, 2026-09-10). O robô se
            # APOIA na caixa: a sonda de 10/09 mediu `forca_de_apoio` p50 = 1,24·m·g,
            # isto é MAIS que o peso da caixa — parte do corpo dele está ali. Tirar a
            # laje no instante do fecho o derruba, e o crédito da queda vai para o
            # fecho: seria desincentivo a botar.
            # ⚠ E a regra dos dois bits CONCORDA: a laje só saía porque a cauda ia ter
            # twist ≠ 0. A §2.2 zerou esse twist, logo `twist = 0 ⟹ laje presente`.
            # ⚠ A cauda de B, de R e do C desviado (CARREGAR, `soltou = False`) afasta a
            # laje: ali o robô sai andando com a caixa, e a laje na frente é obstáculo.
            fica = ids_cauda[~self._soltou[ids_cauda]]
            if len(fica):
                self._laje_para(fica, self.cfg.afasta_z,
                                sobe_caixa=~self._pegou[fica])

        # --- publicação ---
        publica_andar = self._soltou | (aguardando & ~self._pegou)
        self._command[:, ELO] = torch.where(
            publica_andar, torch.full_like(self._elo, ANDAR), self._elo).float()
        # ⚠ v3.5: o elo PUBLICADO, não o interno. Depois do fecho do BOTAR o interno
        # fica BOTAR (para o crítico) mas a tarefa ACABOU — e `VALIDA` significa "há
        # tarefa ativa". Com o interno, `precise_pos` e `load` pagavam ao vivo na
        # cauda POR CIMA da renda congelada (spec v3.5 §0.1).
        base = (self._command[:, ELO] != ANDAR).float()
        self._command[:, VALIDA] = base * (~aguardando).float()
        # ⚠⚠ O ESTADO DE RECOMPENSA, do MESMO `aguardando` que acabou de escrever o
        # `VALIDA` (spec tabela-por-estado §1) — a MESMA fase temporal que os termos
        # liam no `VALIDA`. `_pegou` e `_soltou` estão frescos (`_publica_pegou` roda
        # antes); `_elo` é o de depois do avanço acima, como o `VALIDA`. ⚠ NÃO mover
        # para o fim de `_update_command`: ali `_avanca_elo` já correu, a espera
        # apareceria um passo mais cedo que o `VALIDA`, e o `renda_congelada` — que
        # congela pela SUBIDA de `_fechos` com a soma do passo ANTERIOR — leria a
        # soma errada. IN-PLACE, como `limpo_aguardando` e `limpo_soltou`.
        self._env.limpo_estado.copy_(estado_de_recompensa(
            self._elo, aguardando, self._pegou, self._soltou))

        # ⚠ O σ da TAREFA, no instante em que ela liga. `_sigma_pendente` é verdadeiro
        # do resample até aqui; no `ANDAR` puro `VALIDA` nunca acende, e ele fica
        # pendente o episódio todo — inofensivo, porque nenhum termo de manipulação lê
        # σ com `VALIDA = 0`.
        liga = self._sigma_pendente & (self._command[:, VALIDA] > 0.5)
        ids = todos[liga]
        if len(ids):
            # ⚠ O EIXO DE CIMA É CAPTURADO DE NOVO AQUI, na abertura da TAREFA (23/09),
            # pelo mesmo motivo do alvo abaixo: na passada do `_pendente` a caixa ainda
            # assenta na mesa. ANTES do σ, que lê o `ANG` deste instante.
            self._captura_cima(ids)
            self._recalcula_sigmas(ids)
            # ⚠ REANCORA O ALVO do PEGAR e do CARREGAR aqui (spec dois-bits §1.2, 2º
            # momento). Com `push_robot` ativo, 0,5–1,5 s de espera movem o robô; o
            # alvo tem de nascer da pose de AGORA, não da do reset (ou da abertura do
            # elo anterior). O REORIENTAR e o ANDAR não entram: o alvo deles é a
            # própria caixa, e não a base.
            reancora = ids[torch.isin(
                self._elo[ids], torch.tensor((PEGAR, CARREGAR), device=d))]
            if len(reancora):
                self._alvo_ancorado_na_base(reancora)
            self._pos_no_elo[ids] = self.robot.data.root_link_pos_w[ids]
            self._sigma_pendente[ids] = False

    def _publica_pegou(self) -> None:
        """Atualiza a arma e a publica em `env.limpo_pegou`.

        ⚠ Monotônica DENTRO do episódio (`|=`), e zerada só no `_resample_command`.
        Soltar a caixa para reposicionar não desarma a terminação — se desarmasse,
        largar de vez deixaria de terminar.

        ⚠ Republica o tensor todo passo em vez de guardar uma referência: `.float()`
        cria um tensor NOVO, portanto uma publicação única no `__init__` congelaria o
        valor em zero para sempre.
        """
        tocou = None
        for nome in self.cfg.sensores_palma:
            achou = self._env.scene[nome].data.found
            assert achou is not None, f"sensor '{nome}' precisa do field 'found'."
            aqui = (achou > 0).any(dim=-1)
            tocou = aqui if tocou is None else (tocou & aqui)
        if tocou is not None:
            # ⚠ SÓ ARMA COM O OBJETIVO ATIVO (spec §6.3). Na espera inicial um toque
            # por exploração armaria `escapou` com as palmas longe, e o episódio
            # morreria por ter esperado. Lê `_espera` direto, e não o `VALIDA`, porque
            # este método roda ANTES de `_aplica_espera` na passada.
            ativo = (self._elo != ANDAR) & (self._espera <= 0.0)
            self._pegou |= tocou & ativo
        self._env.limpo_pegou = self._pegou.float()

    # -------------------------------------------------------------- o contrato
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def elo_de(self, ids: torch.Tensor) -> torch.Tensor:
        """O elo corrente daqueles envs.

        ⚠ Ele lê o BUFFER `_elo`, e não reconstrói o elo a partir de
        `CADEIAS[cadeia][passo]`. Duas razões, e as duas são defeitos que existiam aqui:

        1. `CADEIAS[cad]` com `cad = CADEIA_NENHUMA = −1` devolve a ÚLTIMA cadeia em
           Python, portanto um env de `ANDAR` reportava elo `BOTAR` em silêncio. E o
           `__init__.py` registra tasks de inspeção para os CINCO elos, três dos quais
           caem em `−1`.
        2. Reconstruir de duas fontes cria a chance de elas divergirem. O `_elo` é a
           fonte, e é o que o one-hot e o gate de recompensa leem.
        """
        return self._elo[ids]

    def n_elos_da_cadeia(self, ids: torch.Tensor) -> torch.Tensor:
        """Quantos elos tem a cadeia daqueles envs. **1** para quem não tem cadeia.

        ⚠ `CADEIAS[cad]` com `cad = CADEIA_NENHUMA = −1` devolve a ÚLTIMA cadeia em
        Python. Um env de `ANDAR` reportava "2 elos" e elo `BOTAR`, em silêncio. É o
        mesmo defeito que o `_avanca_elo_force` já guardava, e que ficou de fora destes
        dois acessores — que são justamente os que o inspetor usa na tabela ANTES/DEPOIS.
        """
        cad = self._cadeia[ids]
        n = torch.ones_like(cad)
        tem = cad >= 0
        if bool(tem.any()):
            n[tem] = _N_ELOS.to(cad.device)[cad[tem]]
        return n

    def concluiu(self, ids: torch.Tensor) -> torch.Tensor:
        """A cadeia CONCLUIU: `tem ∧ fechou ∧ (_passo == n_elos_da_cadeia − 1)`.

        ⚠⚠ A ÚNICA DEFINIÇÃO DE SUCESSO (spec `g1-limpo-dois-bits.md` §2.2). Um elo
        que fechou e AINDA NÃO avançou (o próximo elo continua na mesma cadeia) não é
        a cadeia inteira — só o fecho do ÚLTIMO elo conta. `metrics["sucesso"]` lê este
        predicado no instante do fecho; `curriculo.nivel` e o balanceador B/C leem
        `concluiu_ate_o_fim`, no reset, que soma a sobrevivência (30/09). Antes cada um
        recalculava a própria versão, e podiam divergir.

        ⚠ O GUARDA `tem` (revisão independente, item A4): SEM ele, um env de `ANDAR`
        (sem cadeia, `_passo` sempre 0) que `forca_avanco` fechasse por engano leria
        `n_elos_da_cadeia == 1` e `concluiu == True` — sucesso falso para quem nunca
        teve tarefa nenhuma.
        """
        tem = self._cadeia[ids] >= 0
        return (tem & self.fechou[ids]
               & (self._passo[ids] == (self.n_elos_da_cadeia(ids) - 1)))

    def _chegou_ao_fim(self, ids: torch.Tensor) -> torch.Tensor:
        """O episódio que ACABOU chegou ao fim pelo tempo, e não por terminação.

        ⚠ SÓ VALE NO RESET: o `mjlab` grava `env.reset_time_outs` antes do reset, e o
        comando reinicia antes das terminações (`manager_based_rl_env.py:438, 581, 587`),
        portanto aqui o `time_out` ainda é o do episódio que acabou. Antes do 1º passo o
        atributo não existe, e ninguém chegou ao fim.
        """
        to = getattr(self._env, "reset_time_outs", None)
        if to is None:
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        return to[ids].bool()

    def concluiu_ate_o_fim(self, ids: torch.Tensor) -> torch.Tensor:
        """`concluiu ∧ chegou ao fim`: o SUCESSO do episódio (enunciado §1, 30/09).

        ⚠ Um fecho seguido da queda da caixa ou do robô NÃO é sucesso: a terminação corta
        o episódio antes do `time_out`. É o sinal que move o `nivel` e as EMAs `s_B`/`s_C`
        (decisão do dono, 30/09; substitui a "limitação declarada" da spec dois-bits
        §2.5, que media só o fecho). `metrics["sucesso"]` continua no fecho, porque ali
        o fim do episódio ainda não aconteceu.
        """
        return self.concluiu(ids) & self._chegou_ao_fim(ids)

    def _perto(self, ids: torch.Tensor) -> torch.Tensor:
        """`‖caixa − alvo‖ <= tol_pos` — extraído de `_fecha_elo_corrente` (spec
        `g1-limpo-dois-bits.md` §2.3, revisão item 29). Também usado por
        `_aplica_espera` para reconferir o alvo no fim da espera, e por
        `recompensas.load`.
        """
        dist_alvo = torch.norm(
            self.caixa.data.root_link_pos_w[ids] - self._command[ids, ALVO], dim=-1)
        return dist_alvo <= self.cfg.tol_pos

    def forca_avanco(self, ids: torch.Tensor) -> None:
        """Força o avanço de elo, sem esperar sustain. Inspetor e `play`.

        ⚠⚠ NÃO chama mais o avanço direto (revisão, item 5): desde a spec
        `g1-limpo-dois-bits.md` §2.2, o fecho de um elo ARMA a espera e não avança — é
        `_aplica_espera` (§2.3) quem avança, no fim dela. Chamar `_avanca_elo_force`
        A CADA dt (como o `eventos.avanca_elo_no_viewer` fazia) rearmaria a espera com
        um sorteio NOVO todo passo, e a cadeia nunca avançaria — congelaria para
        sempre.

        Portanto:
          · quem AINDA não fechou (`fechou=False`) fecha AGORA (arma a espera).
          · TODOS (recém-fechados ou já fechados) têm a espera zerada, para o avanço
            de `_aplica_espera` processar no passo seguinte sem esperar o sorteio.

        Chamar de novo num env que já passou pela cauda (`_avanca_elo_force` nunca
        mais roda para ele, pois `_avanca_elo` só considera `~fechou`) só zera a
        espera de novo — no-op, e é o que torna a chamada repetida do viewer segura.
        """
        if len(ids) == 0:
            return
        ainda_aberto = ids[~self.fechou[ids]]
        if len(ainda_aberto):
            self._avanca_elo_force(ainda_aberto)
        self._espera[ids] = 0.0
        # ⚠ CONTORNA o gate `perto` de `_aplica_espera` (revisão independente, item
        # A8): sem isto, um env que `pegou` e está longe do alvo (o viewer moveu a
        # caixa, ou o `--avanca-elo` chegou antes de alcançar) arma a espera mas
        # NUNCA avança — `avanca = avanca & perto` fica falso para sempre, e
        # `--avanca-elo` congela. `forca_avanco` é o atalho do inspetor: forçar
        # SIGNIFICA avançar mesmo longe.
        self._forcado[ids] = True

    def recebe_tarefa(self, ids: torch.Tensor, elo_novo: int) -> None:
        """Entrega uma tarefa de manipulação AO VIVO a quem estava no `ANDAR`.

        ⚠⚠ SÓ PARA O VISUALIZADOR. Ela existe para simular o DEPLOY: o robô está de pé
        com comando de velocidade, um operador manda "pega a caixa", e o robô transita.
        No TREINO isso não acontece — o elo é sorteado no reset e nunca troca no meio
        (`resampling_time_range = 1e9`, e o `_avanca_elo` só caminha DENTRO de uma
        cadeia; nenhuma cadeia vai de `ANDAR` a `PEGAR`).

        ⚠ ELA NÃO POSICIONA A MOBÍLIA, e isso é do chamador. No `ANDAR` a laje foi
        mandada a +5 m com a caixa em cima (`_aplica_elo`, ramo `ANDAR`), e o ramo
        `PEGAR` NÃO a traz de volta — ele só ancora o alvo. Quem chama tem de rodar o
        `posiciona_cena` ANTES, senão a tarefa entregue é "pegue uma caixa a 5 m".
        O `eventos.troca_elo_no_viewer` faz as duas coisas na ordem certa.

        ⚠ A JANELA DE ESPERA É RE-ARMADA, e é o ponto do exercício: a tarefa chega, e o
        objetivo só liga 0,3 a 1,0 s depois. É a transição que se quer olhar.

        ⚠ O QUE TEM DE SER ZERADO, e cada um por um motivo medido:
          · `_pegou`  — a arma do `caixa_largada`. Sem zerar, uma pega anterior deixaria
            a terminação armada com a caixa longe, e o episódio morreria na entrega.
          · `_sust`   — o cronômetro do elo. Herdado, o elo novo nasceria quase fechado.
          · `fechou`  — senão o `_avanca_elo` ignora o env para sempre.
          · `_pos_no_elo` — a âncora de deslocamento do `CARREGAR`, que tem de ser a
            pose de AGORA e não a do reset.

        ⚠ A pose está FRESCA aqui (isto roda num evento de intervalo, dentro do passo),
        portanto o `_aplica_elo` e o `_recalcula_sigmas` podem ser chamados direto —
        sem a passada do `_pendente`, que existe só para o reset.
        """
        if len(ids) == 0:
            return
        d = self.device
        self._elo[ids] = int(elo_novo)
        # a cadeia compatível com o elo entregue. `cadeia_forcada` vence, como no reset.
        if self.cfg.cadeia_forcada is not None:
            self._cadeia[ids] = int(self.cfg.cadeia_forcada)
        else:
            compat = (_PRIMEIRO_ELO == int(elo_novo)).nonzero().flatten()
            self._cadeia[ids] = int(compat[0]) if len(compat) else CADEIA_NENHUMA
        self._aplica_elo(ids)
        self._recalcula_sigmas(ids)
        self._pos_no_elo[ids] = self.robot.data.root_link_pos_w[ids]
        self._pegou[ids] = False
        self._sust[ids] = 0.0
        # ⚠ SEM ISTO O ELO ENTREGUE FECHA NA HORA (achado pendente do lote C1, spec
        # P2): `_sustain_alvo` ficaria no zero do `__init__`, e `_sust >= 0` é
        # verdadeiro desde o primeiro passo. Mesma regra de toda abertura de elo.
        self._sustain_alvo[ids] = self._sustain_alvo_de(ids)
        self.fechou[ids] = False
        lo, hi = self.cfg.espera_s
        self._espera[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)
        # ⚠ E O BIT CAI NO MESMO INSTANTE. O `_aplica_elo` acima escreveu `VALIDA = 1`
        # (é o que ele faz em elo de manipulação), e o `_aplica_espera` só corrige isso
        # no passo SEGUINTE — este método roda num evento de intervalo, fora da passada
        # do `command_manager`. Sem esta linha o objetivo ficaria ligado por um passo
        # com a janela já armada, e o instante da entrega seria exatamente o que se
        # está tentando olhar.
        self._command[ids, VALIDA] = 0.0
        # ⚠ E O PUBLICADO JÁ NASCE ANDAR (spec §6.4): a espera acabou de ser armada, e o
        # `_aplica_espera` só a veria no passo seguinte.
        self._soltou[ids] = False
        self._command[ids, ELO] = float(ANDAR)

    def _fase(self) -> int:
        """A fase do currículo de cadeia (spec g1-limpo-curriculo-de-cadeia §4, 30/09).

        ⚠ Vem de `env.limpo_forma["fase_cadeia"]` — o estado que o runner salva e restaura,
        e que `_atualiza_balanceador` avança —, com `cfg.fase_inicial` de PISO. No treino
        o piso é 1 e não morde. Em `inspecao` e `play` ele é 4, e o `play` de um
        checkpoint salvo nas fases 1 a 3 roda a cadeia inteira, e não a fase salva.
        """
        st = getattr(self._env, "limpo_forma", None) or {}
        return max(int(st.get("fase_cadeia", 1)), int(self.cfg.fase_inicial))

    def _atualiza_balanceador(self, env_ids: torch.Tensor) -> None:
        """`s_B`, `s_C`: EMA de `concluiu` por cadeia, por ITERAÇÃO de PPO (spec
        `g1-limpo-dois-bits.md` §2.5).

        ⚠⚠ CHAMADO NO TOPO DE `_resample_command`, ANTES de `_cadeia`, `fechou` e
        `_passo` serem sobrescritos pelo reset: são os do EPISÓDIO QUE ACABOU. É a
        mesma ordem que `curriculo.nivel` já respeita para ler `concluiu`.

        ⚠ `iters_balanco` vem de `env.limpo_forma`, já escrito pelo termo de
        currículo `forma` NESTE MESMO reset — currículo roda antes do comando. O
        mesmo relógio de `knobs.Forma.passos_por_iteracao`, sem contador próprio.

        ⚠ O CURRÍCULO DE CADEIA mora na MESMA borda de iteração (spec
        g1-limpo-curriculo-de-cadeia §5, 30/09): `s_cauda`, a EMA da fração de episódios
        que ACABARAM no CARREGAR e acabaram por `time_out`, e a troca de fase
        (`_avalia_troca_de_fase`).
        """
        st = getattr(self._env, "limpo_forma", None)
        if st is None or "s_B" not in st or len(env_ids) == 0:
            return
        # ⚠ As chaves do currículo de cadeia nascem AQUI, e não em `curriculo.garante_forma`
        # (spec §5). `setdefault` não pisa no que o runner restaurou de um checkpoint, e a
        # fase nasce no `fase_inicial` do cfg.
        st.setdefault("fase_cadeia", float(self.cfg.fase_inicial))
        st.setdefault("iter_fase", 0.0)
        st.setdefault("s_cauda", 0.0)
        st.setdefault("n_ep_cauda", 0.0)
        st.setdefault("n_ok_cauda", 0.0)
        cad = self._cadeia[env_ids]
        # ⚠ `concluiu_ate_o_fim` (30/09): fechou o último elo E chegou ao `time_out`.
        concluiu = self.concluiu_ate_o_fim(env_ids)
        eh_b = cad == 0
        eh_c = cad == 2
        if bool(eh_b.any()):
            st["n_ep_B"] += float(eh_b.sum())
            st["n_concluiu_B"] += float(concluiu[eh_b].float().sum())
        if bool(eh_c.any()):
            st["n_ep_C"] += float(eh_c.sum())
            st["n_concluiu_C"] += float(concluiu[eh_c].float().sum())
        # ⚠ `s_cauda` (spec g1-limpo-curriculo-de-cadeia §5, 30/09): dos episódios que
        # ACABARAM no CARREGAR, a fração que acabou por `time_out` e não por terminação. O
        # `mjlab` grava `reset_time_outs` ANTES do reset, e o comando reinicia antes das
        # terminações (`manager_based_rl_env.py:438, 581, 587`), portanto o `time_out` do
        # episódio que acabou ainda está lá. Antes do 1º passo o atributo não existe, e
        # nenhum env está em cauda.
        # ⚠ `cad != 2`: a cauda CARREGAR é a de B e R. O CARREGAR da cadeia C é ELO (30/09),
        # e um episódio que acaba nele mede a cadeia C, e não a cauda.
        em_cauda = (self._elo[env_ids] == CARREGAR) & (cad != 2)
        if bool(em_cauda.any()):
            ok = em_cauda & self._chegou_ao_fim(env_ids)
            st["n_ep_cauda"] += float(em_cauda.sum())
            st["n_ok_cauda"] += float(ok.sum())

        # ⚠ A EMA SÓ APLICA NA BORDA DE ITERAÇÃO (spec §5 item 10): `janela` é o
        # inteiro da iteração corrente, e ela só avança quando o floor de
        # `iters_balanco` sobe — a MESMA janela-única-por-passagem que a rampa de
        # `curriculo.forma` usa para o degrau.
        janela = int(st.get("iters_balanco", 0.0))
        if janela > st["ultima_iter_bal"]:
            st["ultima_iter_bal"] = float(janela)
            alpha = self.cfg.balanceador_alpha
            if st["n_ep_B"] > 0.0:
                st["s_B"] = ((1.0 - alpha) * st["s_B"]
                            + alpha * (st["n_concluiu_B"] / st["n_ep_B"]))
            if st["n_ep_C"] > 0.0:
                st["s_C"] = ((1.0 - alpha) * st["s_C"]
                            + alpha * (st["n_concluiu_C"] / st["n_ep_C"]))
            # ⚠ `s_cauda` e a troca de fase, DEPOIS das EMAs de `s_B`/`s_C` (a troca 1 → 2
            # lê o `s_B` desta iteração). Zerar a contagem da cauda é parte da iteração.
            if st["n_ep_cauda"] > 0.0:
                st["s_cauda"] = ((1.0 - alpha) * st["s_cauda"]
                                + alpha * (st["n_ok_cauda"] / st["n_ep_cauda"]))
            st["n_ep_cauda"] = st["n_ok_cauda"] = 0.0
            self._avalia_troca_de_fase(st, janela)
            st["n_ep_B"] = st["n_concluiu_B"] = 0.0
            st["n_ep_C"] = st["n_concluiu_C"] = 0.0

    def _avalia_troca_de_fase(self, st: dict, janela: int) -> None:
        """A troca de fase do currículo de cadeia (spec g1-limpo-curriculo-de-cadeia §5).

            1 → 2   `s_B ≥ fase2_s_b`       e ≥ `fase2_min_iters` iterações na fase
            2 → 3   `s_cauda ≥ fase3_s_cauda` e ≥ `fase3_min_iters`
            3 → 4   `s_cauda ≥ fase4_s_cauda` e ≥ `fase4_min_iters`

        ⚠ PURA sobre o dict, de propósito: o `smoke` a chama com um dict sintético, sem
        env. Só AVANÇA, e UM degrau por chamada (o `elif` é o que garante) — as fases
        nunca voltam (decisão D5 do dono). A troca grava `iter_fase` (o `iters_balanco`
        do instante) e ZERA o `s_cauda`: a taxa da fase nova não herda a da anterior.
        """
        c = self.cfg
        fase = int(st["fase_cadeia"])
        n = janela - int(st["iter_fase"])
        nova = fase
        if fase == 1 and st["s_B"] >= c.fase2_s_b and n >= c.fase2_min_iters:
            nova = 2
        elif fase == 2 and st["s_cauda"] >= c.fase3_s_cauda and n >= c.fase3_min_iters:
            nova = 3
        elif fase == 3 and st["s_cauda"] >= c.fase4_s_cauda and n >= c.fase4_min_iters:
            nova = 4
        if nova != fase:
            st["fase_cadeia"] = float(nova)
            st["iter_fase"] = float(janela)
            st["s_cauda"] = 0.0

    def _resolve_p_c(self) -> float:
        """`p_C` do balanceador (spec `g1-limpo-dois-bits.md` §2.5), das médias
        `s_B`, `s_C` do `env.limpo_forma`. `0,0`/`1,0` de fallback se o termo de
        currículo `forma` não existir neste cfg (cfgs mínimos de teste).

        ⚠ ANTES DA FASE 4 ele é FIXO em `p_c_antes_do_botar` (spec
        g1-limpo-curriculo-de-cadeia §4, 30/09): prática rara de BOTAR, sem o balanceador.
        """
        if self._fase() < 4:
            return float(self.cfg.p_c_antes_do_botar)
        st = getattr(self._env, "limpo_forma", None)
        s_b = float(st["s_B"]) if st and "s_B" in st else 0.0
        s_c = float(st["s_C"]) if st and "s_C" in st else 1.0
        return resolve_p_c(s_b, s_c, self.cfg.balanceador_piso)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        d = self.device
        n = len(env_ids)
        # ⚠ Zerado no reset para que o primeiro passo parado do episódio novo congele o
        # `_rumo_ref` no rumo ATUAL — e não no do episódio que acabou parado no BOTAR.
        self._env.limpo_twist_zerado[env_ids] = 0.0
        self._n_cauda_anda[env_ids] = 0.0
        self._n_cauda_herdado[env_ids] = 0.0

        # ⚠ O BALANCEADOR B/C LÊ O EPISÓDIO QUE ACABOU (spec §2.5), antes de
        # `_cadeia`/`fechou`/`_passo` virarem os do episódio NOVO.
        self._atualiza_balanceador(env_ids)

        # ⚠ A ARMA DA TERMINAÇÃO `caixa_largada` ZERA AQUI, e ela precisa zerar em
        # algum lugar com escopo de EPISÓDIO. Sem isso um episódio que pegou a caixa
        # armaria todos os seguintes daquele env, e o reset — em que a caixa está na
        # laje e as palmas estão longe — dispararia `escapou` na hora.
        self._pegou[env_ids] = False
        self._soltou[env_ids] = False
        self._carregar_parado[env_ids] = False

        # O ELO. Desde a F2 ele é SORTEADO POR ENV, e o sorteio mora no currículo
        # (`curriculo.sorteia_elo`) porque a ordem de reset é currículo -> eventos ->
        # comando: o reset de pose da base já precisou do elo antes de chegarmos aqui.
        #
        # ⚠ Ler o buffer em vez de sortear aqui não é detalhe de organização. Se o
        # comando sorteasse, o evento de pose teria sorteado OUTRA COISA no mesmo
        # reset, e metade dos envs de manipulação nasceria de costas para a mobília —
        # sem erro e sem log.
        #
        # O `elo_forcado` continua vencendo: é o que o inspetor e o `play` usam.
        if self.cfg.elo_forcado is not None:
            self._elo[env_ids] = int(self.cfg.elo_forcado)
        else:
            self._elo[env_ids] = garante_elo(self._env, ANDAR)[env_ids]

        # ⚠ A JANELA DE ESPERA, sorteada, e DEPOIS do elo — ela depende dele: no `ANDAR`
        # a espera é ZERO. Sortear antes leria o elo do episódio ANTERIOR, que é a mesma
        # classe de defeito que a ordem do currículo já custou a este projeto.
        #
        # ⚠ E ela é sorteada AQUI, e não na passada do `_pendente`: ela não depende de
        # pose nenhuma, só do elo. Adiá-la para lá custaria um passo de janela.
        lo, hi = self.cfg.espera_s
        _sorteio_espera = lo + (hi - lo) * torch.rand(n, device=d)
        _anda = self._elo[env_ids] == ANDAR
        self._espera[env_ids] = torch.where(
            _anda, torch.zeros_like(_sorteio_espera), _sorteio_espera)

        # --- F4: A CADEIA, CONDICIONADA NO ELO QUE O CURRÍCULO JÁ SORTEOU ---
        #
        # ⚠⚠ ISTO É O PONTO MAIS DELICADO DA F4, e uma versão anterior o inverteu: ela
        # sorteava a cadeia e depois SOBRESCREVIA `self._elo` com o 1º elo dela. Como o
        # 1º elo de três das quatro cadeias é o `PEGAR`, TODOS os envs viravam `PEGAR` —
        # e a fatia de locomoção da F2 (95%) era APAGADA. O módulo inteiro existe para
        # não entregar as transições à manipulação cedo demais, e aquele `=` fazia
        # exatamente isso, sem uma linha de log.
        #
        # A ordem correta é a inversa: quem decide se o env é de LOCOMOÇÃO ou de
        # MANIPULAÇÃO é o currículo (`sorteia_elo`, a fatia). A cadeia só escolhe QUAL
        # transição praticar, DENTRO da manipulação — e ela tem de COMEÇAR no elo que o
        # currículo já sorteou. Uma cadeia que começa noutro elo seria uma segunda
        # decisão sobre a mesma coisa.
        #
        # `ANDAR` não é cadeia: ele recebe `CADEIA_NENHUMA`.
        elo_atual = self._elo[env_ids]
        if self.cfg.cadeia_forcada is not None:
            # inspetor/play: a cadeia manda, e o elo passa a ser o 1º dela
            self._cadeia[env_ids] = int(self.cfg.cadeia_forcada)
            self._elo[env_ids] = int(CADEIAS[int(self.cfg.cadeia_forcada)][0])
        elif self.cfg.cadeia_ativa:
            # ⚠ O BALANCEADOR B/C (spec §2.5) decide SÓ para quem começa no `PEGAR`:
            # cadeia 0 (B, segurar-e-carregar) ou 2 (C, botar). Quem começa no
            # `REORIENTAR` só tem a cadeia 1 (R) compatível — a fração do REORIENTAR já
            # vem de `pesos_dos_sorteaveis`, no sorteio de ELO; não há `p_R` aqui.
            p_c = self._resolve_p_c()
            escolhe_c = torch.rand(n, device=d) < p_c
            cadeia_de_pegar = torch.where(
                escolhe_c, torch.full((n,), 2, dtype=torch.long, device=d),
                torch.full((n,), 0, dtype=torch.long, device=d))
            eh_pegar = elo_atual == PEGAR
            eh_reorientar = elo_atual == REORIENTAR
            self._cadeia[env_ids] = torch.where(
                eh_pegar, cadeia_de_pegar,
                torch.where(eh_reorientar,
                            torch.full((n,), 1, dtype=torch.long, device=d),
                            torch.full((n,), CADEIA_NENHUMA, dtype=torch.long, device=d)))
        else:
            # `cadeia.ativa = False`: nenhuma cadeia — o mesmo desliga que
            # `prob_por_nivel = ()` fazia até a v3.
            self._cadeia[env_ids] = CADEIA_NENHUMA

        # Zerar os buffers de avanço
        self._passo[env_ids] = 0
        self._sust[env_ids] = 0.0
        self.fechou[env_ids] = False
        # ⚠ TODA abertura de elo escreve `_sustain_alvo` (v2.1, spec P2), e esta é a
        # do RESET. Tem de vir DEPOIS de `_cadeia` (acima) já existir — não por causa
        # do CARREGAR (que não fecha mais, spec dois-bits §2.4), mas porque
        # `_sustain_alvo_de` ainda lê o elo corrente por cadeia.
        self._sustain_alvo[env_ids] = self._sustain_alvo_de(env_ids)
        # ⚠ o CONTADOR DE FECHOS zera com escopo de EPISÓDIO (v2.1, spec P3) — o mesmo
        # motivo do `_sust` acima: sem isto, um episódio novo herdaria os fechos do
        # anterior e `renda_congelada` nasceria com crédito de um episódio que já acabou.
        self._fechos[env_ids] = 0

        # ⚠ NÃO se sorteia face nem ângulo aqui. A face pedida é CONSTANTE (a
        # marcada), e a dificuldade do `reorientar` vem da ORIENTAÇÃO DE NASCIMENTO da
        # caixa, sorteada em quartos de volta pelo evento `posiciona_cena`.
        # O eixo de cima volta ao Z: é o que vale para quem nasce com a caixa NA MÃO.
        self._cima_b[env_ids] = self._ez_b
        self._aplica_elo(env_ids)
        # a parte dependente de POSE fica pendente para o 1º passo (ver `_pendente`)
        self._pendente[env_ids] = True
        # o σ fica pendente até a TAREFA começar, e não até a pose ficar fresca — a
        # janela de espera ainda não correu aqui (ver `_sigma_pendente`)
        self._sigma_pendente[env_ids] = True
        # ⚠ SORTEIO POR EPISÓDIO, uniforme na faixa. O robô tem de generalizar entre
        # alturas de pega em vez de decorar uma. O alvo já é observável (`alvo_b`),
        # portanto isto é aprendível e não vira ruído.
        lo, hi = self.cfg.altura_carregar_faixa
        self._altura_alvo[env_ids] = lo + (hi - lo) * torch.rand(n, device=d)

    def _update_command(self) -> None:
        todos = torch.arange(self.num_envs, device=self.device)

        # ⚠ PRIMEIRO os pendentes: aqui a pose já está fresca. Só o ALVO e a pose são
        # refeitos aqui — o σ e o `_pos_no_elo` NÃO, desde a F1 (spec
        # `g1-limpo-espera-sigma-e-pose.md` §1): eles dependem de a TAREFA já ter
        # começado, e isso só se sabe quando `VALIDA` acende, dentro de
        # `_aplica_espera`, mais abaixo.
        pend = todos[self._pendente]
        if len(pend):
            self._aplica_elo(pend, so_pose=True)
            self._pendente[pend] = False

        # a caixa gira durante o episódio, portanto a normal da face acompanha. O
        # ÂNGULO pedido é fixo no episódio; a NORMAL não é.
        self._atualiza_face(todos)
        # o alvo do REORIENTAR e do ANDAR é a própria caixa (no ANDAR ele é inerte,
        # porque `VALIDA = 0` — mas um alvo em zero no log é armadilha de leitura).
        #
        # ⚠ O ALVO DO PEGAR e do CARREGAR NÃO é mais recalculado aqui, todo passo
        # (spec dois-bits §1.2). `_alvo_ancorado_na_base` roda só em TRÊS momentos:
        # na abertura do elo (`_aplica_elo`), no fim da espera (`_aplica_espera`,
        # bloco `liga`), e todo passo SÓ com o twist ≠ 0 — ver o fim deste método.
        # Com twist zero, o alvo fica CONGELADO no que foi escrito nos dois primeiros.
        segue = todos[(self._elo == REORIENTAR) | (self._elo == ANDAR)]
        if len(segue):
            self._command[segue, ALVO] = self.caixa.data.root_link_pos_w[segue]

        # a arma do `caixa_largada`, ANTES do avanço de elo: uma cadeia que avança não
        # desarma a terminação, porque a caixa continua sendo a mesma caixa.
        self._publica_pegou()

        # ⚠⚠ A JANELA DE ESPERA, e ela roda ANTES do `_avanca_elo` de propósito: durante
        # a espera o elo NÃO pode fechar. O `_fecha_elo_corrente` lê o `VALIDA`, e é
        # este bloco que o zera — na ordem invertida, um `REORIENTAR` fecharia DENTRO da
        # espera, porque ali o alvo É a própria caixa e `perto` é trivial.
        self._aplica_espera()

        # --- F4: AVANÇO DE ELO ---
        # Deve rodar APÓS a atualização do alvo e da face, porque usa pose fresca.
        self._avanca_elo()

        # ⚠ O TWIST ZERADO roda DEPOIS de `_aplica_espera` e `_avanca_elo` (spec
        # dois-bits §1.1): só assim ele lê `_elo` e `_espera` do passo CORRENTE, e não
        # do passo anterior.
        self._zera_twist_nos_parados()

        # K12: `cauda_twist_herdado` = fração dos passos de cauda ANDANDO (CARREGAR sem a
        # marca de parado e sem `soltou`) cujo comando linear é < 0,05 m/s. Sem peso.
        anda_c = (self._elo == CARREGAR) & ~self._carregar_parado & ~self._soltou
        vel = self._env.command_manager.get_term(self.cfg.nome_do_twist).vel_command_b
        self._n_cauda_anda[anda_c] += 1.0
        self._n_cauda_herdado[anda_c & (torch.norm(vel[:, :2], dim=1) < 0.05)] += 1.0
        self.metrics["cauda_twist_herdado"][:] = (
            self._n_cauda_herdado / self._n_cauda_anda.clamp(min=1.0))

        # ⚠ O ALVO DO CARREGAR, referenciado no robô, TODO PASSO — mas só com o twist
        # ATIVO (spec dois-bits §1.2, terceiro momento): `anda = (elo == CARREGAR) &
        # (twist_zerado < 0.5)`. É o que faz o alvo acompanhar o robô enquanto ele anda
        # com a caixa; com twist zero — a cauda PARADA das fases 1 e 2 do currículo de
        # cadeia, e o elo CARREGAR da cadeia C (30/09) — o alvo fica congelado na âncora
        # do fim da espera ou do avanço.
        anda = todos[(self._elo == CARREGAR) & (self._env.limpo_twist_zerado < 0.5)]
        if len(anda):
            self._alvo_ancorado_na_base(anda)

    def _update_metrics(self) -> None:
        pass

    def _zera_twist_nos_parados(self) -> None:
        """Força vx e vy a ZERO nos elos que exigem o robô parado (spec
        `g1-limpo-dois-bits.md` §1.1) e escreve em wz o laço de RUMO do fabricante.

        ⚠⚠ O RUMO (17/09). Zerar wz não segura o rumo: o `model_19300` girava a −0,38
        rad/s na pega com a caixa (89° em 7 s) e um kernel de velocidade não vê deriva
        lenta — a política é um MLP sem memória e não observa o rumo. O molde já tem
        o laço (`heading_command`, 30 % dos envs): wz = k × wrap(rumo_alvo − rumo), com
        `k = heading_control_stiffness` e o clip em `ranges.ang_vel_z`. Aqui o alvo é o
        rumo do passo em que o env PAROU (`_rumo_ref`), portanto a deriva vira comando
        que a política vê, o rastreio paga e o portão da renda cobra. No robô real o
        mesmo laço fecha com o yaw da IMU; a política nunca observa o rumo absoluto.

        ⚠ É ISTO que impede o robô de andar com a caixa no `pegar`, no `reorientar` e
        no `botar` — e não a forma do alvo. Decisão do dono em 25/08, e é o que o
        `g1_poc` faz (`comando.py:826`), cuja manipulação funcionou.

            parados = (elo ∈ elos_parados) ∨ (espera > 0) ∨ (elo = CARREGAR ∧ carregar_parado)

        ⚠ O `& ~soltou` SAIU (v3.4, spec `g1-limpo-botar-fecha-e-para.md` §2.2). Ele
        deixava o twist fluir na cauda pós-BOTAR. O dono reverteu em 10/09: depois do
        BOTAR o robô recebe o comando de ficar PARADO DE PÉ, e nada mais. Na cauda o
        `_elo` fica BOTAR, e BOTAR está em `elos_parados`, portanto o twist agora fica
        zero até o fim do episódio. O `espera > 0` cobre toda janela de espera, em
        qualquer elo.

        ⚠ Isto afeta SÓ a cauda pós-BOTAR. A cauda das cadeias B e R é `CARREGAR`, que
        NÃO está em `elos_parados` — ela continua andando, salvo a cauda PARADA das
        fases 1 e 2 do currículo de cadeia e o elo CARREGAR da cadeia C
        (`carregar_parado`, 30/09).

        ⚠ ORDEM NO PASSO: este método roda DEPOIS de `_aplica_espera` e de
        `_avanca_elo`, para ler `_elo` e `_espera` já do passo CORRENTE.
        Rodando antes (como fazia até a v2.1), `limpo_twist_zerado` ficava um passo
        atrasado.

        ⚠ ORDEM COM O `twist`: o `twist` é inserido no dict de comandos ANTES deste
        termo (ele vem do molde do fabricante), e o dict é ordenado por inserção.
        Portanto o `compute` dele já rodou quando este roda, e a escrita aqui não é
        sobrescrita no mesmo passo.

        ⚠ A escrita é DESTRUTIVA no buffer, e é de propósito: qualquer métrica que
        gateie por "comando ativo" passa a NÃO contar estes passos, que é o correto —
        eles são passos de comando zero de verdade, e não passos mascarados na
        leitura.

        ⚠⚠ `env.limpo_twist_zerado` É PUBLICADO ANTES do `return` cedo abaixo (v2.1,
        spec P4). Com o `return` antes da publicação, o buffer ficaria com o valor do
        passo anterior no passo em que ninguém está parado — e a âncora do alvo do
        CARREGAR (`_update_command`), que lê este buffer, leria o gate errado. (O
        rastreio já não o lê: o gate dele é `limpo_estado`, via tabela por estado.)

        ⚠ O CARREGAR-andando (spec dois-bits §1.1) NÃO tem twist filtrado: ele recebe
        o do fabricante SEM filtro — nem zerado, nem fixado. A v2.1 sorteava um twist
        próprio (P5); esse bloco saiu.
        """
        # ⚠ O CARREGAR PARADO conta como parado (spec g1-limpo-curriculo-de-cadeia §4,
        # 30/09): a cauda que entrou parada (fases 1 e 2) e o elo CARREGAR da cadeia C usam
        # o laço de rumo dos elos parados, e com isso o freio e o `pose` leem regime parado
        # — os dois leem `limpo_twist_zerado`. O CARREGAR sem a marca anda, como sempre.
        parados = (torch.isin(self._elo, torch.tensor(
            self.cfg.elos_parados, device=self.device)) | (self._espera > 0.0)
            | ((self._elo == CARREGAR) & self._carregar_parado))
        novos = parados & (self._env.limpo_twist_zerado < 0.5)
        self._env.limpo_twist_zerado.copy_(parados.float())

        if not bool(parados.any()):
            return
        tw = self._env.command_manager.get_term(self.cfg.nome_do_twist)
        rumo = tw.robot.data.heading_w
        self._rumo_ref[novos] = rumo[novos]
        lo, hi = tw.cfg.ranges.ang_vel_z
        wz = (tw.cfg.heading_control_stiffness * wrap_to_pi(self._rumo_ref - rumo)).clamp(lo, hi)
        tw.vel_command_b[parados] = 0.0
        tw.vel_command_b[parados, 2] = wz[parados]

    def _fecha_elo_corrente(self, ids: torch.Tensor) -> torch.Tensor:
        """Retorna BoolTensor indicando quais elos fecharam.

        Condição de fechamento POR ELO (spec `g1-limpo-dois-bits.md` §2.4):
            REORIENTAR: perto & alinhado
            PEGAR:      perto & alinhado & de pé
            CARREGAR:   perto & alinhado & de pé   (só como ELO da cadeia C, 30/09)
            BOTAR:      perto & alinhado & apoiada

        ⚠ O "de pé" SAIU DO BOTAR (v3.4, spec `g1-limpo-botar-fecha-e-para.md` §2.1).
        MEDIDO em 10/09 sobre 62 109 passos de BOTAR: os outros três portões passavam
        e o "de pé" reprovava em 99,975% deles, sempre pelo JOELHO. A causa é
        geométrica: a laje fica entre 0,30 e 0,55 m, apoiar ali exige agachar, e o
        "de pé" proíbe agachar. O PEGAR CONTINUA exigindo o "de pé" — o alvo dele é o
        peito, o robô ergue a caixa e se levanta antes de fechar, e ele fecha em 89%.

        ⚠ O CARREGAR FECHA SÓ COMO ELO DA CADEIA C (30/09, enunciado §1 e §4.4): a caixa
        no alvo de transporte, nivelada, o robô de pé — a mesma régua do PEGAR — por
        `sustenta_outros_s`. Como CAUDA de B e R ele nunca chega aqui: `fechou` fica
        True desde o fecho do PEGAR, e `_avanca_elo` só olha quem não fechou.
        """
        if len(ids) == 0:
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)

        c = self.cfg
        d = self.device

        # ⚠⚠ O OBJETIVO TEM DE ESTAR ATIVO. Sem isto o elo fecharia DURANTE a janela de
        # espera — e no `REORIENTAR` o alvo É a própria caixa, portanto `perto` é
        # trivialmente verdadeiro e ele fecharia no passo ZERO, sempre. Era o que fazia
        # `avancos = 0,43` conviver com `sucesso = 0,0000`: a cadeia avançava de graça no
        # primeiro elo e travava no `PEGAR`.
        ativo = self._command[ids, VALIDA] > 0.5

        # Condições "perto" e "alinhado"
        perto = self._perto(ids)

        erro_ang = self._command[ids, ANG]
        alinhado = erro_ang <= torch.deg2rad(torch.tensor(c.tol_ang_deg))

        # ⚠ Condição "de pé" (spec §2.4): a pose das PERNAS E DA CINTURA perto do
        # default, e não mais a altura da pelve. Um robô pode ficar de pé com a pelve
        # alta e as pernas tortas; o que o `PEGAR` exige é a postura de marcha, não só
        # a altura. `JUNTAS_BRACO` sai da conta: o braço trabalha para pegar a caixa, e
        # não é ele quem decide "de pé".
        q = self.robot.data.joint_pos[ids][:, self._ids_de_pe]
        q_default = self.robot.data.default_joint_pos[ids][:, self._ids_de_pe]
        dq = (q - q_default).abs().amax(dim=-1)
        de_pe = dq <= c.de_pe_tol_rad

        # Condição "apoiada" — a LAJE carrega o peso da caixa.
        #
        # ⚠ SEM `try/except`. Uma versão anterior lia um sensor inexistente
        # (`contact_caixa_laje`), pelo método errado (`robot.find_sites` — um SENSOR
        # não é um SÍTIO, e ele vive na CENA, não no robô), dentro de um `try` cujo
        # `except` deixava `apoiada = True`. Resultado: o `BOTAR` fechava com
        # `perto & alinhado` apenas, sem nunca conferir se a caixa estava apoiada, e
        # sem uma linha de erro. Se o sensor não existir, ISTO TEM DE EXPLODIR.
        forca = forca_de_apoio(self._env, c.nome_sensor_apoio)[ids]
        peso = self._env.limpo_massa[ids] * 9.81
        apoiada = forca >= c.fracao_do_peso_apoiada * peso

        # Condições por elo
        elo_corrente = self._elo[ids]
        fecha = torch.zeros(len(ids), dtype=torch.bool, device=d)

        for elo_tipo in (REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = elo_corrente == elo_tipo
            if not bool(m.any()):
                continue

            if elo_tipo == REORIENTAR:
                # ⚠ `reorientar_inerte` (v2): o REORIENTAR fecha só por `perto`, que é
                # trivial (o alvo É a caixa) — o elo vira um atraso de 0,3 s antes do
                # PEGAR. Ver o knob para a medição que exigiu isto.
                fecha[m] = perto[m] & (alinhado[m] | bool(c.reorientar_inerte))
            elif elo_tipo in (PEGAR, CARREGAR):
                fecha[m] = (perto[m] & alinhado[m] & de_pe[m])
            elif elo_tipo == BOTAR:
                fecha[m] = (perto[m] & alinhado[m] & apoiada[m])

        # ⚠ O `ativo` entra NO FIM, e sobre todos os elos de uma vez. Pôr o `& ativo`
        # dentro de cada ramo seria três lugares para esquecer um.
        return fecha & ativo

    def _sustain_alvo_de(self, ids: torch.Tensor) -> torch.Tensor:
        """O sustain exigido pelo elo CORRENTE daqueles `ids`, em segundos.

        ⚠ FONTE ÚNICA do alvo de sustain: PEGAR → `sustenta_pegar_s`; REORIENTAR,
        CARREGAR e BOTAR → `sustenta_outros_s`; ANDAR → 0 (não entra no laço). O
        CARREGAR só fecha como ELO da cadeia C (30/09); como cauda de B e R o buffer
        nunca é lido. Escrita em `self._sustain_alvo` em TODA abertura de elo (reset e
        avanço, spec P2), e `_avanca_elo` só lê o buffer.
        """
        d = self.device
        elo_corrente = self._elo[ids]
        alvo = torch.zeros(len(ids), device=d)
        for elo_tipo in (REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = elo_corrente == elo_tipo
            if not bool(m.any()):
                continue
            if elo_tipo == PEGAR:
                alvo[m] = self.cfg.sustenta_pegar_s
            else:  # REORIENTAR, CARREGAR, BOTAR
                alvo[m] = self.cfg.sustenta_outros_s
        return alvo

    def _avanca_elo(self) -> None:
        """Fecha o elo quando a condição vale por sustain (spec dois-bits §2.2).

        Roda a cada passo DENTRO de `_update_command`, com pose fresca.
        Não há reset nem resample — o one-hot acompanha o elo sem corte de episódio.

        Acumula `_sust` enquanto a condição vale, zera quando não vale.
        Quando `_sust >= sustain_do_elo`, ARMA a espera (`_avanca_elo_force`) — o
        AVANÇO para o próximo elo, ou a entrada na cauda, acontece em
        `_aplica_espera`, no fim da espera.
        """
        d = self.device
        todos = torch.arange(self.num_envs, device=d)
        # ⚠ O dt vem do ENV, e não de um literal. `1.0/50.0` estava escrito aqui à mão;
        # ele acerta hoje e passa a mentir no dia em que a decimação ou o timestep
        # mudar — e o cronômetro de sustentação erraria em silêncio.
        dt = self._env.step_dt

        # Verificar qual elo está aberto (não fechou ainda)
        nao_fechou = todos[~self.fechou]
        if len(nao_fechou) == 0:
            return

        # Condição de fechamento do elo corrente
        fecha = self._fecha_elo_corrente(nao_fechou)

        # Acumular sustain ou zerar
        self._sust[nao_fechou] = torch.where(
            fecha,
            self._sust[nao_fechou] + dt,
            torch.zeros_like(self._sust[nao_fechou])
        )

        # ⚠ v2.1: LIDO do buffer, escrito em toda abertura de elo (reset e avanço) —
        # uma fonte só. Antes recalculado aqui, todo passo; ver `_sustain_alvo_de`.
        sustain_alvo = self._sustain_alvo[nao_fechou]

        # ⚠ SÓ QUEM TEM CADEIA pode avançar. Sem este filtro, um env de `ANDAR` entrava
        # em `_avanca_elo_force` A CADA PASSO: o laço por elo acima não cobre o `ANDAR`,
        # portanto o `sustain_alvo` dele ficava no zero do `torch.zeros`, e
        # `0 >= 0` é True. Com 95% dos envs em locomoção isso era ~95% dos envs entrando
        # na função todo passo de física, e escrevendo `fatia_cadeia = 0` por cima.
        tem_cadeia = self._cadeia[nao_fechou] >= 0
        deve_fechar = (self._sust[nao_fechou] >= sustain_alvo) & tem_cadeia

        ids_fechar = nao_fechou[deve_fechar]
        if len(ids_fechar) > 0:
            self._avanca_elo_force(ids_fechar)

        # ⚠ AS MÉTRICAS SÃO ESCRITAS TODO PASSO, PARA TODOS OS ENVS. Antes elas só eram
        # escritas dentro do `_avanca_elo_force`, isto é, só no instante de um avanço —
        # logo um env de cadeia que NUNCA fechasse o 1º elo nunca escrevia o seu
        # `fatia_cadeia = 1`, e o `Metrics/alvo_caixa/fatia_cadeia` ficava perto de zero
        # no começo do treino. A escada da F4 leria isso e diagnosticaria "as cadeias
        # não estão sendo sorteadas", que é o oposto do que estaria acontecendo.
        cad_t = self._cadeia
        n_el = torch.ones_like(cad_t)
        tem_t = cad_t >= 0
        if bool(tem_t.any()):
            n_el[tem_t] = _N_ELOS.to(d)[cad_t[tem_t]]
        self.metrics["fatia_cadeia"][:] = (n_el > 1).float()
        self.metrics["c_episodio"][:] = (self._cadeia == 2).float()
        self.metrics["passo_final"][:] = self._passo.float()

    def _avanca_elo_force(self, ids: torch.Tensor) -> None:
        """O FECHO do elo corrente: ARMA a espera. NÃO avança (spec dois-bits §2.2).

        ⚠⚠ MUDANÇA DE PAPEL. Até a v2.1 esta função avançava o `_passo`/`_elo` na
        hora. Agora o fecho só arma a espera (`_espera`, `_sigma_pendente`, `_sust`) e
        marca `fechou = True`; é `_aplica_espera` (§2.3) quem avança para o próximo
        elo, ou entra na cauda, no fim da espera — e só então, com a checagem de
        `_perto` recontada contra a deriva do `push_robot` durante a espera.

        Usado pelo `_avanca_elo` (avanço natural, por sustain) e pelo `forca_avanco`
        (inspetor e play).

        ⚠ E ela lia `CADEIAS[cad]` com `cad = −1` para os envs de locomoção, que em
        Python devolve a ÚLTIMA cadeia. Um env de `ANDAR` era tratado como uma cadeia
        de verdade, em silêncio. O `tem` abaixo é o que impede isso.
        """
        if len(ids) == 0:
            return
        d = self.device
        cad = self._cadeia[ids]
        tem = cad >= 0                       # `ANDAR` não tem cadeia

        # ⚠⚠ O CONTADOR DE FECHOS (v2.1, spec P3), lido por `recompensas.renda_congelada`.
        # `origem` TEM DE SER LIDO AQUI, ANTES do `_elo` mudar (o fecho do BOTAR muda
        # `_soltou` logo abaixo) — senão um REORIENTAR que acabou de fechar já leria
        # como o elo NOVO, e o inerte contaria como fecho ganho. Sobe em TODO fecho,
        # exceto o do REORIENTAR inerte.
        origem = self._elo[ids]
        ganho = origem != REORIENTAR
        self._fechos[ids] += (ganho & tem).long()

        self.fechou[ids] = True
        lo, hi = self.cfg.espera_s
        self._espera[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)
        self._sigma_pendente[ids] = True
        self._sust[ids] = 0.0

        # ⚠ O FECHO DO PEGAR IMPLICA A PEGA (revisão independente, item A3): `perto`
        # no peito já é o critério de fecho, e o sensor de PALMA pode nunca disparar
        # — o tronco escora a maior parte do contato medido. Sem isto, a cauda de
        # B/R nunca vira CARREGAR (`vira_carregar` exige `pegou`): o `_elo` fica
        # PEGAR para sempre, o twist continua zerado, e o env fica INERTE até o
        # time_out sem nenhuma terminação acusar.
        ids_pegar = ids[origem == PEGAR]
        if len(ids_pegar):
            self._pegou[ids_pegar] = True
            self._env.limpo_pegou = self._pegou.float()

        # ⚠ O FECHO TERMINAL (spec §2.2): quem fecha o ÚLTIMO elo da cadeia — e só
        # ele — grava `sucesso` AQUI, no instante do fecho, e não quando a espera
        # final acaba (revisão, item 1: métrica escrita no `_resample` sai um
        # episódio atrasada). CHAMA `self.concluiu(ids)`, e não uma segunda conta
        # da mesma coisa (revisão independente, item A4): `fechou` já está True
        # duas linhas acima, e `concluiu` é a ÚNICA definição de sucesso do módulo.
        terminal = ids[self.concluiu(ids)]
        if len(terminal):
            self.metrics["sucesso"][terminal] = 1.0
            # ⚠ métricas SEM PESO do R4b: topo do fecho terminal da cadeia C (razão no log:
            # `c_fecho_topo / c_fechou_botar`), para comparar com `c_botar_topo / c_chegou_botar`.
            term_c = terminal[self._cadeia[terminal] == 2]
            self.metrics["c_fechou_botar"][term_c] = 1.0
            self.metrics["c_fecho_topo"][term_c] = self._env.limpo_topo[term_c]

        # ⚠ A ESPERA FINAL (spec §6.6, dois-bits §2.2): quem fecha o BOTAR já entra
        # `_soltou`, NO MESMO PASSO do fecho, e não só quando a cauda é escrita —
        # senão o `caixa_largada` leria `soltou = 0` por um passo inteiro (a ordem do
        # mjlab é `termination_manager.compute()` ANTES de `command_manager.compute()`)
        # e o robô que solta a caixa morreria por `escapou` em vez de colher a espera
        # final. MEDIDO em 03/09, num code review.
        solta = ids[self._elo[ids] == BOTAR]
        if len(solta):
            self._soltou[solta] = True
            self._env.limpo_soltou[solta] = 1.0

    # ------------------------------------------------------ o alvo, por elo
    def _aplica_elo(self, ids: torch.Tensor, *, so_pose: bool = False) -> None:
        """Escreve o alvo do elo corrente, e move a laje quando o elo pede.

        ⚠ `so_pose` NÃO É LIDO PELO CORPO, e isso é declarado em vez de prometido. O
        docstring anterior dizia "refaz APENAS o que depende de pose", o que era falso:
        a função refaz TUDO. Hoje isso é correto e desejado — o `topo` e o alvo do
        `BOTAR` sorteados no reset vêm de pose OBSOLETA e TÊM de ser descartados e
        re-sorteados na passada do `_pendente`.
        O parâmetro fica na assinatura como documentação do intento do chamador; quem
        acrescentar aqui um sorteio que deva sobreviver à passada do `_pendente` tem de
        passar a LÊ-LO.

        ⚠ A F4 chama exatamente esta função no avanço de elo — e lá ela pode rodar
        com `so_pose=False`, porque no avanço a pose JÁ está fresca (o `_avanca_elo`
        roda no `_update_command`, e não no reset).
        """
        if len(ids) == 0:
            return
        d = self.device
        c = self.cfg
        self._command[ids, ELO] = self._elo[ids].float()
        origem = self._env.scene.env_origins[ids]

        # ⚠ O REGIME DA FACE, e ele é por elo: VIVA no `REORIENTAR`, DE PÉ nos outros.
        # Escrito ANTES do laço, e o eixo de cima capturado aqui, porque esta função
        # roda na passada do `_pendente` com a pose fresca.
        self._regime_face[ids] = torch.where(
            self._elo[ids] == REORIENTAR, FACE_VIVA, FACE_DE_PE)
        self._captura_cima(ids)

        for elo in (ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = ids[self._elo[ids] == elo]
            if len(m) == 0:
                continue
            k = len(m)

            if elo == ANDAR:
                # não há alvo de caixa. O alvo é o TWIST, e ele é outro comando.
                #
                # ⚠ O alvo publicado fica na PRÓPRIA CAIXA, e não na pelve. Ler a
                # pose do robô aqui devolveria valor OBSOLETO (o command manager roda
                # no reset, e os buffers de `data` do robô só são recomputados no
                # forward seguinte). O desenho lê a pelve VIVA, no passo, onde ela
                # está correta.
                self._command[m, VALIDA] = 0.0
                self._laje_para(m, c.afasta_z, sobe_caixa=True)
                self._command[m, ALVO] = self.caixa.data.root_link_pos_w[m]

            elif elo == REORIENTAR:
                # a caixa NÃO se move: o pedido é de atitude
                self._command[m, VALIDA] = 1.0
                self._command[m, ALVO] = self.caixa.data.root_link_pos_w[m]

            elif elo == PEGAR:
                # ⚠ EXATAMENTE o mesmo alvo do `CARREGAR`. A diferença entre os dois
                # elos é o COMANDO DE VELOCIDADE: no `pegar` ele é zero (o robô ergue
                # parado), no `carregar` ele é ativo (o robô ergue e anda).
                #
                # Antes de 25/08 este alvo era congelado em mundo, para o robô não
                # "alcançar o alvo andando". O twist em zero resolve isso melhor, e
                # sem criar dois referenciais para a mesma âncora.
                self._command[m, VALIDA] = 1.0
                self._alvo_ancorado_na_base(m)

            elif elo == CARREGAR:
                # ⚠ a chamada `_laje_para(m, afasta_z, sobe_caixa=False)` SAIU deste
                # ramo (spec dois-bits §1.3): na cauda de B e R é `_aplica_espera` quem
                # manda a laje para longe, uma vez só. Como ELO da cadeia C (30/09) a
                # laje FICA onde o PEGAR a deixou: o BOTAR a reposiciona ao abrir.
                self._command[m, VALIDA] = 1.0
                self._alvo_ancorado_na_base(m)

            elif elo == BOTAR:
                self._command[m, VALIDA] = 1.0
                # ⚠ ABERTURA DO BOTAR (spec dois-bits §1.4). Usa a pose CORRENTE da
                # base, e não a origem do env: `_pos_no_elo` é o do PEGAR, e o robô se
                # moveu ao alcançar. O topo novo nasce PERTO do topo ATUAL
                # (`limpo_topo`), e não sorteado numa faixa absoluta.
                base_p = self.robot.data.root_link_pos_w[m]
                base_q = self.robot.data.root_link_quat_w[m]
                # ⚠ O GUARDA `teto = fundo − botar_folga_laje` FICA: o limite físico
                # (o fundo da caixa SEGURADA menos a folga) sempre vence — sem ele a
                # laje nasceria DENTRO da caixa. `botar_topo_teto` era um teto do teto;
                # virou a constante `_TOPO_TETO_FISICO`.
                fundo = self.caixa.data.root_link_pos_w[m, 2] - self._meia(m)[:, 2]
                teto = torch.clamp(fundo - c.botar_folga_laje, max=_TOPO_TETO_FISICO)
                # ⚠ CASO DECLARADO: se a caixa está segurada MAIS BAIXA que a laje mais
                # fina possível, nenhum topo satisfaz as duas coisas. O `clamp(min=...)`
                # sobe a laje até `prateleira_topo_piso` (nunca ENTERRADA), mesmo que
                # isso passe do `teto` — geometricamente impossível de satisfazer, e é
                # melhor declarar que violar em silêncio.
                # ⚠ PROFUNDIDADE DO BOTAR (R4b, spec `g1-limpo-botar-profundidade.md`; §2 do
                # enunciado, D17): a laje NÃO herda mais o topo do PEGAR (quase nunca ficava
                # abaixo de 0,15 m). Topo = `teto_raso − prof × U(0;1)`, com `prof` de
                # `botar_delta_topo` (s_C = 0) até o chão (s_C ≥ `botar_prof_s_c`). Sem `forma`,
                # s_C = 1: profundidade cheia. Se o s_C cai, a profundidade encolhe.
                st_r = getattr(self._env, "limpo_forma", None)
                s_c_r = float(st_r["s_C"]) if st_r and "s_C" in st_r else 1.0
                teto_raso = torch.clamp(teto, max=c.botar_raso_topo_max)
                prof = c.botar_delta_topo + avanco_prof_botar(s_c_r, c.botar_prof_s_c) * (
                    teto_raso - c.prateleira_topo_piso - c.botar_delta_topo)
                topo = (teto_raso - prof * torch.rand(k, device=d)).clamp(
                    min=c.prateleira_topo_piso)
                self.metrics["c_botar_topo"][m] = topo

                # ⚠ A LAJE nasce perto da BASE, não da origem do env (revisão, item 5):
                # `xy_laje = base_p + quat_apply_yaw(base_q, _AVANCO_LAJE_BOTAR + dxy)`.
                # ⚠ `_AVANCO_LAJE_BOTAR`, e NÃO `c.prateleira_xy` (revisão do
                # coordenador): o avanço do BOTAR é medido à parte do reset — subir o
                # knob afastaria a caixa do PEGAR em todo nível.
                dxy = c.botar_delta_xy * (2.0 * torch.rand(k, 2, device=d) - 1.0)
                # ⚠ A laje do BOTAR só se AFASTA em x, dx ∈ [0; δ] (lote 01/10, R1): ela nascia
                # dentro da perna em 31–72% das aberturas (toque 39% → 0,6%); y segue simétrico.
                # Spec `g1-limpo-lote-resume-01-10.md`, Parte 2.
                dxy[:, 0] = dxy[:, 0].abs()
                off_laje = torch.zeros(k, 3, device=d)
                off_laje[:, 0] = _AVANCO_LAJE_BOTAR + dxy[:, 0]
                off_laje[:, 1] = dxy[:, 1]
                xy_laje = base_p[:, :2] + quat_apply_yaw(base_q, off_laje)[:, :2]
                self._laje_para(m, topo, xy=xy_laje)

                # ⚠ O ALVO fica na BORDA PERTO do tampo, não no centro (revisão, item
                # 14). No centro o robô alcançaria por cima de 20 cm de tampo — defeito
                # datado em 16/07 —, e a 0,04 m de topo a distância passaria de
                # ALCANCE_R = 0,85. `botar_recuo_borda` recua o alvo do centro para a
                # borda perto do robô. ⚠ CORRIGIDO (revisão independente, item A10):
                # o `dxy[:, 1]` da laje NÃO basta — ele desloca laje e alvo JUNTOS, e o
                # deslocamento relativo entre os dois ficava sempre igual. O jitter
                # lateral de verdade entra no PRÓPRIO `off_recuo`, abaixo.
                off_recuo = torch.zeros(k, 3, device=d)
                off_recuo[:, 0] = c.botar_recuo_borda + dxy[:, 0] * 0.5
                # ⚠ JITTER LATERAL em y (spec §1.4, revisão independente item A10):
                # sem ele, o deslocamento alvo↔laje em y é SEMPRE o mesmo (o jitter
                # da própria laje cancela na subtração) — anti-decoreba do dono.
                off_recuo[:, 1] = dxy[:, 1] * 0.5
                a = torch.zeros(k, 3, device=d)
                a[:, :2] = xy_laje - quat_apply_yaw(base_q, off_recuo)[:, :2]
                a[:, 2] = topo + self._meia(m)[:, 2]
                self._command[m, ALVO] = a

        self._atualiza_face(ids)

    def _laje_para(self, ids: torch.Tensor, topo, *, sobe_caixa: bool | torch.Tensor = False,
                  xy: torch.Tensor | None = None) -> None:
        """Move a laje (mocap) para um topo. A pose é o CENTRO do corpo.

        `sobe_caixa=True` leva a CAIXA junto, apoiada no topo novo. É o que o `ANDAR`
        precisa: com a laje a +5 m e a caixa no chão, o robô tropeçaria nela.

        ⚠ `sobe_caixa` ACEITA TENSOR (spec dois-bits §2.3): a cauda chama isto com
        `~pegou`, uma máscara MISTA sobre `ids` — parte vira CARREGAR (a caixa fica
        nas mãos, `sobe_caixa=False` para esses) e parte nunca pegou (`True`). Um bool
        escalar continua funcionando para os outros dois chamadores (ANDAR, sempre
        `True`; BOTAR, sempre `False`).

        ⚠ QUEM SOLTOU NÃO CHEGA MAIS AQUI (v3.4, spec §2.4): a cauda pós-BOTAR não
        mexe na cena, portanto `ids` já vem filtrado por `~soltou`. É por isso que
        `limpo_topo` também deixa de ser reescrito para esses envs — e é o correto: a
        laje deles continua onde estava.

        `xy` (spec dois-bits §1.3): posição x,y em MUNDO da laje. `None` usa
        `org + prateleira_xy` (o comportamento de sempre). O `BOTAR` passa a sua
        própria x,y — o topo novo nasce perto da base CORRENTE, não da origem do env.
        """
        k = len(ids)
        d = self.device
        c = self.cfg
        org = self._env.scene.env_origins[ids]
        topo_t = (torch.full((k,), float(topo), device=d)
                  if not torch.is_tensor(topo) else topo)
        pose = torch.zeros(k, 7, device=d)
        if xy is None:
            pose[:, 0] = org[:, 0] + c.prateleira_xy[0]
            pose[:, 1] = org[:, 1] + c.prateleira_xy[1]
        else:
            pose[:, 0] = xy[:, 0]
            pose[:, 1] = xy[:, 1]
        pose[:, 2] = topo_t - c.prateleira_meia_z
        pose[:, 3] = 1.0
        self.prateleira.write_mocap_pose_to_sim(pose, env_ids=ids)
        sobe = (sobe_caixa if torch.is_tensor(sobe_caixa)
               else torch.full((k,), bool(sobe_caixa), dtype=torch.bool, device=d))
        if bool(sobe.any()):
            ids_sobe = ids[sobe]
            pc = pose[sobe].clone()
            pc[:, 2] = topo_t[sobe] + self._meia(ids_sobe)[:, 2]
            self.caixa.write_root_link_pose_to_sim(pc, env_ids=ids_sobe)
            self.caixa.write_root_link_velocity_to_sim(
                torch.zeros(len(ids_sobe), 6, device=d), env_ids=ids_sobe)
        if hasattr(self._env, "limpo_topo"):
            self._env.limpo_topo[ids] = topo_t

    def _alvo_ancorado_na_base(self, ids: torch.Tensor) -> None:
        """O alvo do CARREGAR. O referencial é dividido POR EIXO:

            x, y   RELATIVOS ao robô, reescritos a cada passo — a caixa está nas mãos
                   e tem de acompanhá-lo horizontalmente.
            z      ABSOLUTO, vem de `self._altura_alvo`, sorteado por episódio em
                   `altura_carregar_faixa`. Continuar absoluto é o que impede o robô
                   de satisfazer o alvo andando agachado.

        ⚠ O z NÃO pode ser relativo. Se fosse, o robô satisfaria o alvo ANDANDO
        AGACHADO: o alvo desceria junto com a pelve e a caixa nunca precisaria subir.
        Com o z absoluto, carregar exige manter a caixa na altura de trabalho — que é
        o comportamento pedido.
        """
        if len(ids) == 0:
            return
        d = self.device
        p = torch.tensor(self.cfg.peito_b, device=d).expand(len(ids), 3)
        base_p = self.robot.data.root_link_pos_w[ids]
        base_q = self.robot.data.root_link_quat_w[ids]
        a = base_p + quat_apply(base_q, p)
        a[:, 2] = self._altura_alvo[ids]
        self._command[ids, ALVO] = a

    def _meia(self, ids: torch.Tensor) -> torch.Tensor:
        """[k, 3] — a meia-aresta da caixa DE CADA ENV (spec §6.7).

        Lê `env.limpo_meia_aresta`, publicado pelo evento de startup `tamanho_caixa`. O
        knob `caixa_meia_aresta` é só o fallback de um env montado sem o evento.
        """
        meia = getattr(self._env, "limpo_meia_aresta", None)
        if meia is not None:
            return meia[ids]
        return torch.full((len(ids), 3), float(self.cfg.caixa_meia_aresta), device=self.device)

    def alvos_das_palmas(self, ids: torch.Tensor) -> torch.Tensor:
        """[k,2,3] — o ponto que CADA palma deve alcançar, em MUNDO.

        O alvo de cada palma é o centro da SUA face lateral. O offset gira com a
        caixa, portanto a pose pedida acompanha a orientação dela.

        ⚠ Ordem (esquerda, direita) = ordem de `cfg.sitios_palma`. A esquerda mira
        `+y` da caixa e a direita mira `−y`. É a convenção do `g1_poc`
        (`observacoes.alvos_das_palmas`), e ela casa com a geometria dos pads: o pad
        esquerdo fica em `y = −0,015` local e o direito em `y = +0,015`, portanto as
        duas palmas olham UMA PARA A OUTRA.
        """
        caixa = self.caixa.data.root_link_pos_w[ids]
        off = torch.zeros_like(caixa)
        off[:, 1] = self._meia(ids)[:, 1]
        off = quat_apply(self.caixa.data.root_link_quat_w[ids], off)
        return torch.stack((caixa + off, caixa - off), dim=1)

    def dist_palma_caixa(self, ids: torch.Tensor) -> torch.Tensor:
        """Distância MÉDIA das duas palmas às SUAS faces laterais, por env.

        ⚠ BIMANUAL E LATERAL, e as duas metades vêm de medição. Até 28/08 isto era
        `min` sobre as palmas contra a SUPERFÍCIE de uma esfera em volta do centro:

            d = ‖palma − centro‖.min(palmas) − meia_aresta

        Aquilo tem dois buracos, e o bloco 3 caiu nos dois. Com `min`, UMA mão satura
        o kernel e a segunda não tem gradiente nenhum — mas o `squeeze` exige as DUAS
        (ele é `min` das forças). A cadeia ficava sem ponte entre "uma mão encosta" e
        "as duas apertam", que é exatamente onde a run travou: `staged` parado no valor
        de nascimento e `squeeze` em 0,0002 depois de 3200 iterações.
        E com a esfera, tocar o TOPO, a FRENTE ou a BASE paga igual a tocar a lateral,
        portanto não existe gradiente para a pose de pega.

        O `g1_poc` já tinha consertado isto e escreveu o porquê
        (`g1_poc/observacoes.py:47`): "Com o centro, o `reaching` estagnava com UMA mão
        na face próxima, sem gradiente para o abraço."

        A MÉDIA é o que acopla as duas mãos: uma mão atrasada derruba o termo, portanto
        as duas se aproximam juntas. O máximo é a pose PRÉ-PEGA, com as palmas
        flanqueando a caixa — e é ela que torna o pad de DORSO geometricamente errado,
        que é como o `g1_poc` dispensa o `back_penalty` (`g1_poc/terminacoes.py:13`).

        ⚠ NÃO subtrai mais a meia-aresta: o alvo JÁ está na superfície. Subtrair de
        novo deixaria o kernel saturado antes do contato.
        """
        palmas = self.robot.data.site_pos_w[ids][:, self._ids_palma, :]
        alvos = self.alvos_das_palmas(ids)
        return torch.norm(palmas - alvos, dim=-1).mean(dim=1)

    def _recalcula_sigmas(self, ids: torch.Tensor) -> None:
        """σ = distância inicial × fator, com piso. Ver o bloco do `__init__`.

        ⚠ EXCEÇÃO: no regime `FACE_DE_PE` o `sigma_ori` é FIXO na TOLERÂNCIA DO FECHO, e
        não mais o erro inicial com piso. Ver o bloco ⚠⚠ no corpo — a caixa abre o PEGAR
        de pé, com erro inicial zero, e "σ = erro inicial" degenera nele.

        ⚠ O PISO não é estética: um env que nasce com a palma colada na caixa teria
        σ ≈ 0, e o kernel viraria um pico impossível de sustentar — a recompensa
        desabaria ao primeiro milímetro de tremor.

        ⚠ CHAMADO POR `_aplica_espera` NO INSTANTE EM QUE `VALIDA` ACENDE (F1), e não
        no reset. "Inicial" é a distância da TAREFA, não a do episódio: durante a
        janela de espera (`VALIDA = 0`) o robô aproxima as mãos de graça, e um σ
        travado no reset mediria a distância ERRADA. MEDIDO no `play` do `bloco9` em
        2026-09-08: σ fixado no reset em 0,34 m, mão a 0,20 m no fim da espera —
        `alcancar = exp(−(0,20/0,34)²) = 0,71` em vez do `exp(−1) = 0,368` que
        `recompensas._alcancar` promete.
        """
        if len(ids) == 0:
            return
        c = self.cfg
        d_palma = self.dist_palma_caixa(ids)
        d_alvo = torch.norm(
            self.caixa.data.root_link_pos_w[ids] - self._command[ids, ALVO], dim=-1)
        self.sigma_alcance[ids] = (d_palma * c.sigma_fator).clamp(min=c.sigma_min)
        self.sigma_trazer[ids] = (d_alvo * c.sigma_fator).clamp(min=c.sigma_min)
        # ⚠ O σ de ORIENTAÇÃO é o ÂNGULO inicial, em radianos — outra unidade, outro
        # piso. Com σ fixo de 0,40 rad um pedido de 90° dá `exp(−(1,57/0,40)²)` =
        # 2,0e−7, isto é zero: era a "sorte de nível 3+" medida no `g1_poc`.
        #
        # ⚠⚠ "σ = ERRO INICIAL" DEGENERA QUANDO O ERRO NASCE ZERO (22/09). A caixa abre o
        # PEGAR de pé na mesa: o σ cairia no piso de 0,20 rad, e a partir de uns 20° de
        # tombo a gaussiana é zero. MEDIDO na `zero02`: `caixa_na_pega` de 53° a 59°.
        #
        # No regime DE PÉ o σ é a TOLERÂNCIA DO FECHO (`tol_ang_deg`, a mesma régua do
        # `_fecha_elo_corrente`), FIXA — não mais `max(erro inicial × fator, tolerância)`
        # (29/09): com o `max`, tombar a caixa na espera (os sete termos valem zero ali)
        # alargava o σ de graça no CARREGAR e no BOTAR seguintes. A metade linear do
        # híbrido de `_alinha` (S3) já dá o gradiente de longe que o `max` existia para
        # dar.
        #
        # O `FACE_VIVA` NÃO entra: o erro inicial do `REORIENTAR` é real (até 90°), e
        # mantém `max(erro × fator, sigma_ori_min)`.
        self._atualiza_face(ids)
        sig = (self._command[ids, ANG] * c.sigma_fator).clamp(min=c.sigma_ori_min)
        de_pe = self._regime_face[ids] == FACE_DE_PE
        self.sigma_ori[ids] = torch.where(
            de_pe, torch.full_like(sig, float(np.deg2rad(c.tol_ang_deg))), sig)

    def _captura_cima(self, ids: torch.Tensor) -> None:
        """Guarda qual eixo da caixa aponta PARA CIMA agora, no frame dela.

        Chamado quando um elo abre (`_aplica_elo`) e quando a TAREFA liga
        (`_aplica_espera`). A partir daí o `precise_ori` e o `alinhado` medem o tombo
        desse eixo contra a vertical: a face de baixo continua para baixo.

        ⚠ SÓ COM A CAIXA NA MESA: `CARREGAR` e `BOTAR` ficam FORA. No avanço de cadeia
        eles herdam o eixo do `PEGAR`; capturar ali pegaria a caixa TOMBADA nas mãos e
        exigiria manter o tombo (o defeito de 21/09). Quem nasce neles fica com o Z que
        o reset escreve.
        """
        ids = ids[(self._elo[ids] != CARREGAR) & (self._elo[ids] != BOTAR)]
        if len(ids) == 0:
            return
        self._cima_b[ids] = quat_apply_inverse(
            self.caixa.data.root_link_quat_w[ids], self._ez_w.expand(len(ids), 3))

    def _atualiza_face(self, ids: torch.Tensor) -> None:
        """Publica a DIREÇÃO DESEJADA e o ERRO angular do eixo medido da caixa.

            FACE  a direção em que o eixo medido DEVE apontar.
            ANG   o erro angular ATUAL, em radianos, entre o eixo medido e essa
                  direção. Zero = alinhado.

        ⚠ DOIS REGIMES. Ver `FACE_VIVA`/`FACE_DE_PE` no topo do módulo:

            REORIENTAR   VIVA — a face marcada, para o robô, na horizontal, todo passo.
            os outros    DE PÉ — o eixo de cima capturado na mesa (`_cima_b`), contra a
                         vertical do mundo.

        ⚠⚠ O REGIME TROCA O VETOR MEDIDO, e não só a direção pedida. A face marcada é
        LATERAL (`face_alvo_b` = −x), e um vetor lateral é cego ao giro em torno de si
        mesmo — o eixo palma-a-palma (28/09). Quem responde "a face de baixo continua
        para baixo" é o eixo de cima, em QUALQUER eixo horizontal de tombo, de 0 a 180°.

        ⚠ A GUINADA fica LIVRE no `FACE_DE_PE`, e é decisão do dono (21/09 e 28/09).
        """
        if len(ids) == 0:
            return
        k = len(ids)
        r = self._regime_face[ids].unsqueeze(-1)
        de_pe = r == FACE_DE_PE

        # o eixo da CAIXA que este elo mede, e a direção em que ele tem de apontar
        eixo_b = torch.where(de_pe, self._cima_b[ids], self._face_b.expand(k, 3))
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], eixo_b)

        para_o_robo = (self.robot.data.root_link_pos_w[ids]
                       - self.caixa.data.root_link_pos_w[ids])
        para_o_robo = para_o_robo.clone()
        para_o_robo[:, 2] = 0.0        # a direção pedida é HORIZONTAL
        viva = para_o_robo / para_o_robo.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ⚠ `where` e não indexação por máscara: os ramos são densos e do mesmo
        # tamanho, e assim não há um segundo caminho de escrita para manter em dia.
        desejada = torch.where(de_pe, self._ez_w.expand(k, 3), viva)

        self._command[ids, FACE] = desejada
        cos = (normal_w * desejada).sum(-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos)
        self._command[ids, ANG] = ang
        # ⚠ O VETOR DE GIRO (spec §8.3): eixo `normal × desejada` normalizado, vezes o
        # ângulo. Diz para que LADO girar, o que o escalar `ANG` não diz — sem ele um MLP
        # sem memória não aprende a reorientar. Antiparalelo (ang ≈ π) não tem eixo
        # definido: usa-se Z, que é "meia volta pivotando na laje". Em ang ≈ 0 o
        # produto vetorial some e `where` põe Z também — vezes zero, dá zero.
        eixo = torch.cross(normal_w, desejada, dim=-1)
        norma = eixo.norm(dim=-1, keepdim=True)
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(eixo)
        eixo = torch.where(norma > 1e-6, eixo / norma.clamp(min=1e-6), ez)
        self._command[ids, GIRO] = eixo * ang.unsqueeze(-1)

    # --------------------------------------------------------------- o desenho
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Desenha o que ESTE termo publica. Nada é recalculado."""
        import mujoco

        for i in visualizer.get_env_indices(self.num_envs):
            elo = int(self._elo[i])
            nome = ELOS[elo]
            nivel = int(garante_nivel(self._env)[i])
            caixa_p = self.caixa.data.root_link_pos_w[i].cpu().numpy()
            caixa_q = self.caixa.data.root_link_quat_w[i].cpu().numpy()
            pelve = self.robot.data.root_link_pos_w[i].cpu().numpy()
            alvo = self._command[i, ALVO].cpu().numpy()
            face = self._command[i, FACE].cpu().numpy()
            valida = float(self._command[i, VALIDA])

            # OS EIXOS DA CAIXA, do quatérnion real dela. X vermelho, Y verde, Z azul.
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, caixa_q)
            visualizer.add_frame(
                position=caixa_p, rotation_matrix=mat.reshape(3, 3),
                scale=0.22, axis_radius=0.008,
                label=f"[{nome}] nivel {nivel}")

            if elo == ANDAR:
                # O ALVO DO `ANDAR` É UMA VELOCIDADE, e ela vem do outro comando.
                tw = self._env.command_manager.get_command("twist")[i]
                v_b = torch.stack((tw[0], tw[1], torch.zeros_like(tw[0])))
                v_w = quat_apply(self.robot.data.root_link_quat_w[i:i + 1],
                                 v_b.unsqueeze(0))[0].cpu().numpy()
                visualizer.add_arrow(
                    start=pelve, end=pelve + v_w, color=_VERDE, width=0.020,
                    label=f"[andar] v_cmd {float(tw[0]):+.2f},{float(tw[1]):+.2f} m/s"
                          f"  wz {float(tw[2]):+.2f} rad/s")
                # o alvo de caixa está DESLIGADO neste elo
                visualizer.add_sphere(center=pelve + np.array([0, 0, 1.0]),
                                      radius=0.03, color=_CINZA,
                                      label="sem alvo de caixa (valida=0)")
                continue

            # O EIXO MEDIDO da caixa (o que ele É) e a DIREÇÃO DESEJADA (onde ele DEVE
            # apontar). O erro é o ângulo entre os dois.
            # ⚠ O eixo SEGUE O REGIME (21/09): no regime de pé quem é medido é o eixo
            # de cima, e não a face marcada. Desenhar a face aqui faria o visualizador mentir —
            # a seta não casaria com o `ANG` impresso ao lado dela.
            de_pe_i = bool(self._regime_face[i] == FACE_DE_PE)
            fb = (self._cima_b[i] if de_pe_i else self._face_b).unsqueeze(0)
            nome_eixo = "eixo de CIMA da caixa" if de_pe_i else "face MARCADA"
            n_at = quat_apply(self.caixa.data.root_link_quat_w[i:i + 1], fb)[0]
            n_at = n_at.cpu().numpy()
            erro = np.degrees(float(self._command[i, ANG]))
            voltas = int(getattr(self._env, "limpo_voltas",
                                 torch.zeros(1, device=self.device))[i]) \
                if hasattr(self._env, "limpo_voltas") else 0
            visualizer.add_arrow(
                start=caixa_p, end=caixa_p + n_at * 0.30, color=_VERDE, width=0.014,
                label=f"{nome_eixo} aponta aqui")
            visualizer.add_arrow(
                start=caixa_p, end=caixa_p + face * 0.30, color=_MAGENTA, width=0.012,
                label=f"DEVE apontar aqui  ·  erro {erro:.0f}°  ·  "
                      f"{voltas} quarto(s) de volta")

            # O ALVO
            rot = "  (o alvo É a caixa: pede-se ATITUDE)" if elo == REORIENTAR else ""
            anc = "  (ancorado na BASE)" if elo == CARREGAR else ""
            visualizer.add_sphere(center=alvo, radius=0.05, color=_CIANO,
                                  label=f"[{nome}] alvo{rot}{anc}")

            # QUANTO MOVER a caixa até o alvo
            d = float(np.linalg.norm(alvo - caixa_p))
            visualizer.add_arrow(
                start=caixa_p, end=alvo, color=_AMARELO, width=0.010,
                label=f"caixa->alvo {d:.3f} m  (dz {float(alvo[2]-caixa_p[2]):+.3f})")

            # O TOPO DA LAJE
            prat = self.prateleira.data.root_link_pos_w[i].cpu().numpy()
            topo = prat[2] + self.cfg.prateleira_meia_z
            visualizer.add_box(
                center=np.array([prat[0], prat[1], topo]),
                size=np.array([self.cfg.prateleira_meia_xy,
                               self.cfg.prateleira_meia_xy, 0.002]),
                mat=np.eye(3), color=_CINZA, label=f"topo da laje {topo:.3f} m")

            # O ALCANCE, a partir da pelve
            visualizer.add_sphere(center=pelve, radius=ALCANCE_R, color=_BRANCO,
                                  label=f"alcance ~{ALCANCE_R:.2f} m")
            d_alvo = float(np.linalg.norm(alvo - pelve))
            visualizer.add_arrow(
                start=pelve, end=alvo, width=0.006,
                color=_CIANO if d_alvo <= ALCANCE_R else _VERMELHO,
                label=f"pelve->alvo {d_alvo:.3f} m"
                      + ("" if d_alvo <= ALCANCE_R else "  FORA DO ALCANCE"))
            _ = valida


# =============================================================================
# A RÉGUA DA LOCOMOÇÃO
# =============================================================================
class TwistComRazaoDeMarcha(UniformVelocityCommand):
    """O twist do fabricante, com DUAS réguas a mais.

    **JUIZ (desde 27/08) — `eficiencia_min`.** Por SEGMENTO de comando:

        e_s = ⟨v_real · v̂_cmd⟩_s / ⟨‖v_cmd‖⟩_s ,     e o portão lê min(e_s)

    ⚠ POR QUE ELA SUBSTITUIU A `razao_marcha` NO PORTÃO. A razão é soma de NORMAS, e
    norma nunca cancela: ruído de média zero SEMPRE a infla. MEDIDO no bloco 1 — o `std`
    subiu de 0,43 para 0,61 (a manipulação entrou, exploração voltou a valer) e a razão
    caiu de 0,514 para 0,426 com DURAÇÃO (984 -> 988) e QUEDA (0,000 -> 0,167) PARADAS e
    o `play` determinístico andando bem. O portão congelou na banda morta e a rampa deu
    UM degrau em 1341 iterações: ele leu ruído de ação como incompetência.
    A projeção cancela o ruído (`Σ(ruído · v̂_cmd)` tem média zero, encolhe com 1/√N), e o
    corte por segmento impede o robô de compensar um segmento ruim com outro bom.

    **DIAGNÓSTICO — `razao_marcha`.** Fica no log, fora do portão, para que o bloco 2
    tenha as duas curvas lado a lado e o limiar novo possa ser calibrado contra medição.

        razao_marcha = 1 − Σ‖v_cmd_xy − v_xy‖ / Σ‖v_cmd_xy‖

    ⚠ POR QUE ADIMENSIONAL. O currículo de comando do fabricante (`command_vel`)
    ALARGA a faixa de velocidade ao longo do treino. Uma régua em m/s daria um DEGRAU
    na iteração em que a faixa abre, e o portão leria progresso onde só houve mudança
    de escala. Aqui as duas somas crescem juntas, e o degrau se cancela. MEDIDO: o
    currículo de comando corta só 17% da colheita da estátua em 10k iterações,
    portanto ele mexe a escala de verdade e essa imunidade não é teórica.

    ⚠ POR QUE ELA NASCE EM 0,0, E ISSO É O DESENHO. Robô imóvel com comando ativo tem
    erro igual ao comando, portanto numerador = denominador e a razão é ZERO. É o
    oposto exato do portão que media DURAÇÃO DE EPISÓDIO: aquele dava nota máxima à
    estátua, porque a estátua não cai. Este dá zero.

    ⚠ AS SOMAS VIVEM EM `self.metrics`, e isso não é conveniência. `CommandTerm.reset`
    (`command_manager.py:99-107`) lê a métrica, tira a média dos envs e SÓ DEPOIS zera
    — e o `reset` do comando roda DEPOIS do currículo (currículo -> eventos ->
    comando). Portanto o consumidor lê o episódio que ACABOU, e não um buffer meio
    zerado. Um buffer próprio meu precisaria repetir essa ordem à mão.

    ⚠ SÓ O EIXO LINEAR XY. O erro de guinada tem régua própria do fabricante
    (`error_vel_yaw`), e misturar rad/s com m/s numa soma só faria um número sem
    unidade nem interpretação. Um robô que anda reto e não gira mostra
    `razao_marcha` alta e `error_vel_yaw` alto — dois números, dois defeitos.

    ⚠ A RAZÃO PODE FICAR NEGATIVA, e não é clampeada. Andar para o lado ERRADO dá erro
    de até 2× o comando, logo razão −1,0. Clampear em 0 esconderia a diferença entre
    "parado" e "indo ao contrário", que é justamente o que se quer ver num bloco em
    que a política derivou.
    """

    cfg: TwistComRazaoDeMarchaCfg

    def __init__(self, cfg: TwistComRazaoDeMarchaCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        z = torch.zeros(self.num_envs, device=self.device)
        self.metrics["soma_erro_marcha"] = z.clone()
        self.metrics["soma_cmd_marcha"] = z.clone()
        self.metrics["razao_marcha"] = z.clone()

        # ⚠ A EFICIÊNCIA POR SEGMENTO (27/08). Ela SUBSTITUI o `razao_marcha` como juiz;
        # ver o docstring da classe para o motivo. Os cinco buffers moram em
        # `self.metrics` pelo mesmo motivo que as somas: o `reset` do mjlab zera o que
        # está no dict, e um buffer próprio meu precisaria repetir a ordem à mão.
        self.metrics["seg_proj"] = z.clone()     # Σ (v_real · v̂_cmd) dt, no segmento
        self.metrics["seg_pedido"] = z.clone()   # Σ ‖v_cmd‖ dt, no segmento
        self.metrics["seg_visto"] = z.clone()    # cópia do `command_counter`
        self.metrics["segmentos"] = z.clone()    # quantos segmentos VÁLIDOS fecharam
        self.metrics["eficiencia_min"] = z.clone()
        self.metrics["eficiencia_media"] = z.clone()

        # ⚠ O RAMO DE GIRO (spec §9). Flag por env, re-sorteada com as do fabricante.
        self.is_turning_env = torch.zeros_like(self.is_standing_env)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # ⚠ AS FLAGS DO FABRICANTE SÃO SORTEIOS INDEPENDENTES, não uma partição (spec §9):
        # `standing` zera tudo todo passo; `heading` reescreve `wz` todo passo; `forward`
        # escreve só no resample. O `turning` entra com precedência EXPLÍCITA:
        # standing > turning > forward > heading. Ele cede ao standing (que zeraria o
        # wz todo passo), vence o forward (roda depois do super) e SAI do heading (senão
        # o heading reescreve o wz dele no passo seguinte).
        r = torch.empty(len(env_ids), device=self.device)
        turning = r.uniform_(0.0, 1.0) <= self.cfg.rel_turning_envs
        turning &= ~self.is_standing_env[env_ids]
        self.is_turning_env[env_ids] = turning
        ids = env_ids[turning]
        if len(ids) == 0:
            return
        self.is_heading_env[ids] = False
        self.is_forward_env[ids] = False
        lo, hi = self.cfg.ranges.ang_vel_z
        teto = max(abs(float(lo)), abs(float(hi)))
        # ⚠ A FAIXA É MUTADA PELO CURRÍCULO DO FABRICANTE (`command_vel` reescreve
        # `cfg.ranges.ang_vel_z` no cfg COMPARTILHADO, e o `make_env_cfg` só o remove no
        # ramo `play`). Hoje o estágio 0 abre em ±0,5 e só alarga, portanto o mínimo de
        # 0,2 cabe. Se um dia um estágio estreitar abaixo dele, `uniform_(from > to)`
        # levanta `RuntimeError` no primeiro re-sorteio — crash na iteração 0 de uma run
        # paga. Melhor falhar na montagem, com o número na mensagem.
        assert self.cfg.turning_wz_min <= teto, (
            f"`turning_wz_min` = {self.cfg.turning_wz_min} não cabe na faixa de "
            f"`ang_vel_z` (teto {teto}); o currículo de comando estreitou a faixa?")
        mag = torch.empty(len(ids), device=self.device).uniform_(
            float(self.cfg.turning_wz_min), teto)
        sinal = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = sinal * mag
        self.vel_command_w[ids] = self.vel_command_b[ids]

    def _update_command(self) -> None:
        super()._update_command()
        # ⚠ `lin = 0` TODO PASSO nos envs turning, como o fabricante faz com o standing.
        ids = self.is_turning_env.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.vel_command_b[ids, :2] = 0.0

    def _fecha_segmento(self, mudou: torch.Tensor) -> None:
        """Pontua o segmento que acabou, SÓ nos envs de `mudou`, e reinicia os deles.

        ⚠ A MÁSCARA É OBRIGATÓRIA. Num lote de 4096 envs cada um re-sorteia no seu
        próprio instante, então numa chamada típica só uma fração cruzou a fronteira.
        Sem a máscara, o zeramento apagaria o acumulador dos envs que estão NO MEIO do
        segmento deles — e a métrica mediria pedaços aleatórios de segmento.

        ⚠ VALIDADE POR `seg_pedido`, e não por duração. Um segmento só é pontuado se o
        que foi PEDIDO nele passa de `pedido_min_segmento`. Uma regra só descarta dois
        casos de uma vez: o comando quase nulo (`is_standing_env` e sorteio perto de
        zero), cujo denominador é ruído; e o fragmento curto do fim do episódio, onde
        150 passos ainda não cancelaram o ruído de ação.

        ⚠ E O PRIMEIRO SEGMENTO NÃO PODE ENTRAR NO `min` COMO SE HOUVESSE UM ANTERIOR:
        com `segmentos == 0` o mínimo É a eficiência dele, e não `min(e, 0.0)` — que
        travaria a métrica em zero para sempre.
        """
        ped = self.metrics["seg_pedido"]
        vale = mudou & (ped >= self.cfg.pedido_min_segmento)
        e = self.metrics["seg_proj"] / ped.clamp(min=1e-6)

        n = self.metrics["segmentos"]
        primeiro = vale & (n == 0.0)
        demais = vale & (n > 0.0)

        self.metrics["eficiencia_min"][:] = torch.where(
            primeiro, e,
            torch.where(demais,
                        torch.minimum(self.metrics["eficiencia_min"], e),
                        self.metrics["eficiencia_min"]))
        # média incremental, para não guardar a soma num sexto buffer
        self.metrics["eficiencia_media"][:] = torch.where(
            vale,
            (self.metrics["eficiencia_media"] * n + e) / (n + 1.0),
            self.metrics["eficiencia_media"])
        self.metrics["segmentos"] += vale.float()

        # ⚠ zera onde MUDOU, e não onde VALE: um segmento inválido também terminou, e
        # deixar o acumulador dele de pé o somaria ao segmento seguinte.
        z = torch.zeros_like(self.metrics["seg_proj"])
        self.metrics["seg_proj"][:] = torch.where(
            mudou, z, self.metrics["seg_proj"])
        self.metrics["seg_pedido"][:] = torch.where(
            mudou, z, self.metrics["seg_pedido"])

    def _update_metrics(self) -> None:
        super()._update_metrics()
        cmd = self.vel_command_b[:, :2]
        vel = self.robot.data.root_link_lin_vel_b[:, :2]
        norma_cmd = torch.norm(cmd, dim=-1)

        # ⚠ FRONTEIRA DE SEGMENTO, detectada pelo `command_counter`. A ORDEM DO MJLAB
        # torna isto correto: `CommandTerm.compute` chama `_update_metrics` ANTES do
        # `_resample` (`command_manager.py:110-115`), portanto nesta chamada o comando e
        # o contador ainda são os do segmento que está correndo. Quando o contador muda,
        # a mudança é vista na chamada SEGUINTE — e aí o acumulador a ser fechado é o do
        # segmento certo, sem nunca somar velocidade nova em comando velho.
        #
        # ⚠ O `seg_visto` zera no reset, junto com as outras métricas, e isso é seguro:
        # no passo seguinte o contador difere de 0, o fecho dispara com `seg_pedido = 0`,
        # e a regra de validade descarta o segmento vazio. Sem efeito no log.
        atual = self.command_counter.to(dtype=self.metrics["seg_visto"].dtype)
        mudou = atual != self.metrics["seg_visto"]
        if bool(mudou.any()):
            self._fecha_segmento(mudou)
            self.metrics["seg_visto"][:] = atual

        # ⚠ O GATE. Sem ele um comando de 0,001 m/s entraria nas duas somas com erro
        # quase nulo e inflaria a razão de graça — e o fabricante põe 10% dos envs em
        # `is_standing_env`, portanto esse caso NÃO é raro.
        ativo = (norma_cmd > self.cfg.limiar_comando).float()

        self.metrics["soma_erro_marcha"] += torch.norm(cmd - vel, dim=-1) * ativo
        self.metrics["soma_cmd_marcha"] += norma_cmd * ativo

        # ⚠ ASSINATURA IN-PLACE. `self.metrics[k] = ...` trocaria o objeto de tensor, e
        # o `reset` do mjlab zera o objeto que estiver no dict — funcionaria, mas
        # qualquer referência guardada apontaria para o buffer velho.
        soma_cmd = self.metrics["soma_cmd_marcha"]
        self.metrics["razao_marcha"][:] = torch.where(
            soma_cmd > 0.0,
            1.0 - self.metrics["soma_erro_marcha"] / soma_cmd.clamp(min=1e-6),
            torch.zeros_like(soma_cmd),
        )

        # ⚠ A ACUMULAÇÃO DO SEGMENTO. Projeção, e NÃO norma, e essa é a única diferença
        # que importa contra o `razao_marcha`:
        #
        #     Σ‖v_cmd − v_real‖   -> norma nunca cancela; ruído de média zero SEMPRE
        #                            aumenta a soma, monotonicamente.
        #     Σ(v_real · v̂_cmd)   -> o ruído entra como Σ(ruído · v̂_cmd), média zero,
        #                            e encolhe com 1/√N dentro do segmento.
        #
        # MEDIDO no bloco 1: o `std` subiu de 0,43 (it 1525) para 0,61 (it 4999) e o
        # `razao_marcha` caiu de 0,514 para 0,426 — enquanto DURAÇÃO (984 -> 988) e
        # QUEDA (0,000 -> 0,167) não se moveram, e o `play` determinístico andava bem.
        # A queda era da forma da métrica, não da política.
        #
        # ⚠ `dt` é `self._env.step_dt`, e não `1/50` escrito à mão.
        dt = self._env.step_dt
        dir_cmd = cmd / norma_cmd.clamp(min=1e-6).unsqueeze(-1)
        self.metrics["seg_proj"] += (vel * dir_cmd).sum(dim=-1) * ativo * dt
        self.metrics["seg_pedido"] += norma_cmd * ativo * dt


@dataclass(kw_only=True)
class TwistComRazaoDeMarchaCfg(UniformVelocityCommandCfg):
    """⚠ O `mjlab` constrói o termo por `cfg.build(env)`, e NÃO por um atributo
    `class_type` (`command_manager.py:268`). Um `class_type` aqui seria campo morto:
    o cfg passaria, o manager chamaria o `build` HERDADO, e o treino rodaria com o
    twist do fabricante — sem a métrica e sem nenhum erro."""

    limiar_comando: float = 0.05

    pedido_min_segmento: float = 0.5
    """Piso de VALIDADE do segmento: `Σ‖v_cmd‖dt` mínimo para ele ser pontuado.

    ⚠ NÃO É UM ALVO. A tarefa continua sendo rastrear velocidade; isto só decide se um
    segmento tem sinal suficiente para ser julgado. Uma regra descarta dois casos: o
    comando quase nulo, cujo denominador é ruído, e o fragmento curto no fim do
    episódio, onde o ruído de ação ainda não cancelou.

    **Derivado:** o comando médio vale ~0,765 m/s, e o re-sorteio do fabricante é de 3 a
    8 s. Um segmento inteiro rende então ~2,3 no mínimo. 0,5 aceita segmentos a partir
    de ~0,7 s de comando cheio e descarta os 10% de `is_standing_env` (que somam 0)."""

    rel_turning_envs: float = 0.0
    """Fração dos envs que só GIRAM: `lin = 0` todo passo, `|wz| ≥ turning_wz_min`
    (spec §9). Sorteada a cada re-sorteio de comando, como as flags do fabricante.
    Precedência: `standing > turning > forward > heading`."""
    turning_wz_min: float = 0.2

    def build(self, env: ManagerBasedRlEnv) -> TwistComRazaoDeMarcha:
        return TwistComRazaoDeMarcha(self, env)
