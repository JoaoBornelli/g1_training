"""Roda uma cadeia no MuJoCo clássico e grava as 29 juntas num CSV.

    python registra_juntas.py --cena ~/g1_pilota/ --checkpoint ~/Downloads/model_10500.pt
    python registra_juntas.py ... --roteiro carregar --saida carrega

Irmão do `pilota.py`: mesma cena, mesma observação de 114 canais, mesmo ator. A
diferença é que aqui NÃO tem teclado — um roteiro troca o elo sozinho a cada N
segundos, e cada passo vira uma linha do CSV.

⚠ Ele importa do `pilota.py` e por isso mora na RAIZ, não em `g1_limpo/`. Nada aqui
depende do `mjlab`: o MuJoCo clássico roda ~40× tempo real nesta CPU, contra 96× mais
LENTO por substep no Warp.

O que sai, e nada mais (o `giro_w` de correção do tombo entra na observação em todo
passo, como no treino; antes o registrador o deixava em zero e o ator ficava cego ao tombo):
  <saida>.csv           passo, t, fase, elo, one-hot, o ÂNGULO CRU das 29 juntas, e a
                        POSE DE MUNDO da raiz do robô e da caixa (7 números cada:
                        x, y, z, qw, qx, qy, qz)
                        + o ALVO do comando em mundo (alvo_x/y/z), o `giro_ang` (graus
                        de tombo da caixa, como o treino o mede) e a coluna `ckpt`
                        (nome do arquivo:sha256 de 10 dígitos do checkpoint)
                        + `a_<junta>` (a ação crua do ator) e `alvo_pd_<junta>` (o alvo do PD, rad)
                        + a FORÇA NA CAIXA: `fn_E`, `fn_D` (normal de cada palma), `fn_tot`
                        (soma das normais de todo contato robô→caixa), `fz_rob` (Fz que o
                        robô sustenta), `n_rob` (nº de contatos) e `rel_x/y/z` (caixa menos
                        o ponto médio dos pads)
  <saida>.limites.csv   junta, lo, hi, default — a régua para ler o CSV

`--mu-caixa μ` dá à caixa a PRIORIDADE do treino (caixa 2, laje 3) com este μ; 0 desliga.

⚠ A RAIZ E A CAIXA entraram em 15/09. Sem elas, uma sonda que queira a altura da pelve
no mundo, a inclinação do tronco, o CoM ou a palma no frame da caixa tem de SUPOR pelve
vertical e pés planos. A suposição erra justamente quando o robô inclina a base.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np

from pilota import (ELOS, Ator, alvo_do_elo, carrega_cena, gira, gira_inverso,
                    monta_observacao, restaura)

I_ANDAR, I_CARREGAR = ELOS.index("ANDAR"), ELOS.index("CARREGAR")
I_PEGAR, I_REORIENTAR = ELOS.index("PEGAR"), ELOS.index("REORIENTAR")
EZ = np.array([0.0, 0.0, 1.0])


def giro_de_pe(quat_caixa: np.ndarray, cima_b: np.ndarray) -> tuple[np.ndarray, float]:
    """`(giro_w, ang)`: o vetor de MUNDO que endireita a caixa, e o tombo em graus.

    A fórmula do `comando._atualiza_face` no regime de pé: `normal_w = R·cima_b` contra
    o `ez`; eixo = normal × ez; `giro_w = eixo · ang`. ⚠ Só o regime de pé: no
    REORIENTAR o treino usa o regime vivo, e um roteiro com REORIENTAR publica aqui o
    giro de pé.
    """
    normal_w = gira(quat_caixa, cima_b)
    ang = float(np.arccos(np.clip(normal_w @ EZ, -1.0, 1.0)))
    eixo = np.cross(normal_w, EZ)
    n = float(np.linalg.norm(eixo))
    eixo = eixo / n if n > 1e-6 else EZ
    return eixo * ang, float(np.degrees(ang))

# ⚠ A cena é exportada do reset do PEGAR: a caixa e a laje já estão à frente do robô.
# Por isso todo roteiro abre na espera e não numa aproximação.
#
# As duas cadeias que o treino fecha depois do PEGAR, uma em cada entrada:
#   botar     PEGAR -> BOTAR -> ANDAR. A caixa vai para a laje e o robô sai sem ela.
#   carregar  PEGAR -> CARREGAR. A caixa FICA NA MÃO e o robô anda com ela.
ROTEIROS = {
    "botar": ("espera:ANDAR:1.0,"
              "pegar:PEGAR:3.0,"
              "espera:BOTAR:1.0,"
              # ⚠ 2,0 s, e não 0,5. A sonda de forma lê os ÚLTIMOS 0,3 s da fase, que
              # é a aproximação do instante de fecho que o log por cronômetro permite.
              # Com 0,5 s a janela pegava 15 dos 25 passos — 60% da fase, e a pose
              # ainda estava assentando. Com 2,0 s ela pega 15 de 100, depois de a pose
              # parar de mudar.
              "botar:BOTAR:2.0,"
              "espera:ANDAR:1.0,"
              "andar:ANDAR:6.0:0.8"),
    "carregar": ("espera:ANDAR:1.0,"
                 "pegar:PEGAR:7.0,"
                 "espera:CARREGAR:1.0,"
                 "carregar:CARREGAR:10.0:0.5"),
}


class Fase:
    """Um trecho do roteiro: rótulo, elo publicado, duração e o vx pedido."""

    def __init__(self, rotulo: str, elo: int, segundos: float, vx: float):
        self.rotulo, self.elo, self.segundos, self.vx = rotulo, elo, segundos, vx

    @property
    def _marcha(self) -> bool:
        return abs(self.vx) > 1e-6

    @property
    def limpa_caixa(self) -> bool:
        """⚠ A caixa só some no `ANDAR`. No `CARREGAR` ela está NA MÃO.

        Confundir os dois apagava a caixa do meio da tarefa: `CARREGAR` É andar com
        ela. Só o `ANDAR` é andar sem nada.
        """
        return self._marcha and self.elo == I_ANDAR

    @property
    def limpa_laje(self) -> bool:
        """A laje sai no `CARREGAR` sempre, e no `ANDAR` quando ele marcha.

        ⚠ NO `CARREGAR` A MARCHA NÃO ENTRA NA CONTA. O treino manda a laje para longe
        na CAUDA de quem fechou o PEGAR, uma vez só e antes de qualquer passo
        (`comando.py:1546`). Parado ou andando, quem está em `CARREGAR` já não tem
        laje na frente. Amarrar isso à marcha deixava a laje no caminho durante a
        espera do `CARREGAR`, e o robô lia uma cena que o treino nunca mostra.
        """
        return self.elo == I_CARREGAR or self.limpa_caixa


def analisa_roteiro(texto: str) -> list[Fase]:
    """`"rotulo:ELO:segundos[:vx], ..."` -> lista de `Fase`. Um nome de `ROTEIROS` serve.

    ⚠ O elo é o NOME (`PEGAR`), e não o índice: um índice trocado é um erro silencioso
    que só aparece como "o robô não faz nada".
    """
    texto = ROTEIROS.get(texto.strip(), texto)
    fases: list[Fase] = []
    for pedaco in texto.split(","):
        pedaco = pedaco.strip()
        if not pedaco:
            continue
        campos = pedaco.split(":")
        if len(campos) not in (3, 4):
            raise SystemExit(f"fase malformada: {pedaco!r} — use rotulo:ELO:segundos[:vx] "
                             f"ou um nome pronto: {', '.join(ROTEIROS)}")
        rotulo, nome, seg = campos[0], campos[1].upper(), campos[2]
        if nome not in ELOS:
            raise SystemExit(f"elo {nome!r} não existe; use um de {', '.join(ELOS)}")
        vx = float(campos[3]) if len(campos) == 4 else 0.0
        fases.append(Fase(rotulo, ELOS.index(nome), float(seg), vx))
    if not fases:
        raise SystemExit("roteiro vazio")
    return fases


def enderecos_da_caixa(m: mujoco.MjModel, id_caixa: int) -> tuple[int, int]:
    """`(qpos_adr, dof_adr)` da junta livre da caixa."""
    jid = int(m.body_jntadr[id_caixa])
    if jid < 0:
        raise SystemExit("a caixa não tem junta — não dá para afastá-la")
    return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def limpa_a_cena(m: mujoco.MjModel, d: mujoco.MjData, c, afasta: float,
                 laje: bool, caixa: bool) -> None:
    """Manda para longe o que o elo manda, como o treino faz.

    ⚠ NO `ANDAR` VÃO AS DUAS: tirar só a laje deixaria a caixa caindo no caminho.
    ⚠ NO `CARREGAR` VAI SÓ A LAJE: a caixa está na mão, e andar com ela é a tarefa.

    No `ANDAR` os dez canais da caixa zeram no gate da observação, então a pose dela
    não muda o que a política vê — só a física.
    """
    if laje:
        id_mocap = int(m.body_mocapid[int(c.id_laje)])
        if id_mocap >= 0:
            d.mocap_pos[id_mocap, 0] += afasta
        else:
            print("⚠ a laje não é mocap; ela ficou onde estava")
    if caixa:
        adr_q, adr_v = enderecos_da_caixa(m, int(c.id_caixa))
        d.qpos[adr_q] += afasta
        d.qvel[adr_v:adr_v + 6] = 0.0
    mujoco.mj_forward(m, d)


def _topo_da_laje(m: mujoco.MjModel, d: mujoco.MjData, c) -> float:
    """A altura do TAMPO agora, lida da cena. `nan` se a laje não for mocap."""
    id_mocap = int(m.body_mocapid[int(c.id_laje)])
    if id_mocap < 0:
        return float("nan")
    return float(d.mocap_pos[id_mocap, 2]) + float(c.prateleira_meia_z)


def poe_a_laje(m: mujoco.MjModel, d: mujoco.MjData, c, topo: float) -> None:
    """Põe o TOPO da laje na altura pedida e leva a caixa junto.

    A cena exportada guarda o nível que o `posiciona_cena` sorteou no reset daquele
    treino. Para ler a pose de uma altura específica — o nível 3 são 0,15 m — a laje
    tem de ir para lá ANTES do primeiro passo.

    ⚠ A CAIXA SOBE O MESMO DELTA. Ela está apoiada no tampo; mover só a laje faria a
    caixa nascer flutuando ou dentro da pedra. O delta é o mesmo para as duas.

    ⚠ `topo`, e não o z do corpo: o mocap fica no CENTRO da laje, meia espessura
    abaixo do tampo. Quem pega a caixa encosta no tampo, então é ele que vale.
    """
    id_mocap = int(m.body_mocapid[int(c.id_laje)])
    if id_mocap < 0:
        print("⚠ a laje não é mocap; --topo não teve efeito")
        return
    z_novo = float(topo) - float(c.prateleira_meia_z)
    delta = z_novo - float(d.mocap_pos[id_mocap, 2])
    d.mocap_pos[id_mocap, 2] = z_novo
    adr_q, adr_v = enderecos_da_caixa(m, int(c.id_caixa))
    d.qpos[adr_q + 2] += delta
    d.qvel[adr_v:adr_v + 6] = 0.0
    mujoco.mj_forward(m, d)


def regua_das_juntas(m: mujoco.MjModel, c) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Nomes, faixa `(29, 2)` e default `(29,)`, na ordem do `nomes_juntas` da cena.

    ⚠ A JUNTA É ACHADA PELO ENDEREÇO DE `qpos`, e NÃO pelo nome. O `cena.npz` grava
    `left_hip_pitch_joint`, mas o modelo compilado chama `robot/left_hip_pitch_joint`
    — o `mj_name2id` devolvia −1 nas 29, a faixa saía toda NaN e o resumo vinha VAZIO,
    sem erro nenhum. O `ids_junta_qpos` veio do próprio exportador e aponta a junta
    certa, com prefixo ou sem.
    """
    nomes = [str(n) for n in c.nomes_juntas]
    ids_q = np.asarray(c.ids_junta_qpos, dtype=np.int64)
    por_adr = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
    faixa = np.zeros((len(nomes), 2))
    faltando = []
    for i in range(len(nomes)):
        jid = por_adr.get(int(ids_q[i]), -1)
        if jid < 0 or not bool(m.jnt_limited[jid]):
            # ⚠ Junta sem limite vira NaN, e NÃO 0: um zero calado viraria `frac`
            # inventado, e o resumo apontaria batente onde não há.
            faixa[i] = (np.nan, np.nan)
            faltando.append(nomes[i])
        else:
            faixa[i] = m.jnt_range[jid]
    if faltando:
        print(f"⚠ sem faixa no modelo ({len(faltando)}): {', '.join(faltando[:6])}"
              + (" ..." if len(faltando) > 6 else ""))
    return nomes, faixa, np.asarray(c.q_default, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb e cena.npz")
    ap.add_argument("--checkpoint", required=True, help="o model_*.pt")
    ap.add_argument("--saida", default="juntas", help="prefixo dos arquivos de saída")
    ap.add_argument("--roteiro", default="botar",
                    help=f"um nome pronto ({', '.join(ROTEIROS)}) ou "
                         "rotulo:ELO:segundos[:vx] separados por vírgula")
    ap.add_argument("--voltas", type=int, default=1, help="quantas vezes repetir o roteiro")
    ap.add_argument("--afasta", type=float, default=10.0,
                    help="metros em +x para onde a cena vai na marcha (ANDAR: laje e "
                         "caixa; CARREGAR: só a laje)")
    # ⚠ A SAÍDA PARA TESTAR O SOLVER. O `.mjb` traz `iterations=10, ls_iterations=20`,
    # que é o que o treino usa — mas o treino roda no MuJoCo WARP, outra implementação.
    # Um agarre com atrito é o caso mais sensível a esforço de solver: se subir para
    # 100/50 fizer a caixa parar de escorregar, a diferença é do solver e não da cena.
    ap.add_argument("--iteracoes", type=int, default=0,
                    help="sobrepõe opt.iterations do solver (0 = o do modelo)")
    ap.add_argument("--ls-iteracoes", type=int, default=0,
                    help="sobrepõe opt.ls_iterations (0 = o do modelo)")
    # ⚠ MULETAS DE DEBUG, NÃO CORREÇÕES. O MuJoCo clássico solta a caixa que o Warp
    # segura, e sem a caixa na mão não há pose de manipulação para ler. Os dois botões
    # abaixo seguram a caixa para que os ÂNGULOS DE JUNTA fiquem legíveis. Quem usa
    # aceita que a cena deixou de ser a do treino: não tire conclusão de força de
    # contato, de escorrego nem de largada com estes valores fora do padrão.
    ap.add_argument("--topo", type=float, default=0.0,
                    help="altura do TOPO da laje em metros (0 = a da cena exportada). "
                         "A caixa sobe junto. O nível 3 do currículo são 0,15")
    ap.add_argument("--atrito", type=float, default=1.0,
                    help="multiplica o atrito de escorrego da caixa e das palmas "
                         "(1,0 = o do modelo; tente 1,5)")
    ap.add_argument("--mu-caixa", type=float, default=0.0,
                    help="μ da caixa com PRIORIDADE: todo contato do robô com a caixa usa "
                         "este μ, e a mesa (prioridade maior) fica no dela. É a física do "
                         "treino desde 05/10 (`cena._spec_box`, caixa 2, laje 3). 0 = desliga")
    ap.add_argument("--impratio", type=float, default=0.0,
                    help="sobrepõe opt.impratio, o peso do atrito contra o normal no "
                         "solver (0 = o do modelo, que é 1,0; tente 10)")
    # ⚠ O QUE O TREINO NÃO CONSEGUE FAZER. No mjlab a carga entra como FORÇA externa
    # (`eventos.carga_caixa`), porque `dr.body_mass` corrompe a heap do Warp: a caixa de
    # 5 kg pesa 5 kg e gira como 1 kg. Aqui é MuJoCo clássico, e `body_mass` é um array
    # comum — a massa e a inércia mudam JUNTAS, e a dinâmica fica a de verdade.
    ap.add_argument("--massa", type=float, default=0.0,
                    help="massa da caixa em kg (0 = a do modelo, que é 1,0). A inércia "
                         "acompanha. O teto do currículo é `carga_max`; tente 5")
    ap.add_argument("--rumo-k", type=float, default=2.0,
                    help="laço de rumo: wz = k x (rumo inicial - rumo), como o driver do "
                         "robô real fecha com a IMU; 0 desliga (wz = 0 fixo, mostra a "
                         "deriva crua). Padrão 2,0 desde 02/10: com 0,5 o rumo assentava a "
                         "~6° da referência (erro de equilíbrio = viés/k); com 2,0 assentou "
                         "em -2,1° no teste `23250_k2`. O treino ainda usa o 0,5 do "
                         "`heading_control_stiffness` do mjlab; `--rumo-k 0.5` reproduz os "
                         "plays antigos")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="fator de tempo do viewer: 1,0 = tempo real, 0,25 = 4x lento")
    ap.add_argument("--sem-viewer", action="store_true", help="roda o mais rápido que der")
    args = ap.parse_args()

    fases = analisa_roteiro(args.roteiro)
    m, c = carrega_cena(args.cena)
    d = mujoco.MjData(m)
    ator = Ator(args.checkpoint)
    # ⚠ Duas cópias com a mesma iteração (`model_5000.pt`, `model_5000(1).pt`) geram CSVs
    # indistinguíveis; nome + hash dos bytes identifica o arquivo que de fato rodou.
    arq_ckpt = Path(args.checkpoint).expanduser()
    ckpt = f"{arq_ckpt.name}:{hashlib.sha256(arq_ckpt.read_bytes()).hexdigest()[:10]}"
    if ator.dim_entrada != int(c.dim_obs):
        raise SystemExit(f"o checkpoint espera {ator.dim_entrada} canais e a cena monta "
                         f"{int(c.dim_obs)}. Checkpoint de outra fase?")

    if args.iteracoes:
        m.opt.iterations = args.iteracoes
    if args.ls_iteracoes:
        m.opt.ls_iterations = args.ls_iteracoes
    if args.impratio:
        m.opt.impratio = args.impratio
    if args.atrito != 1.0:
        # ⚠ SÓ A COLUNA 0. `geom_friction` é (escorrego, torção, rolamento); mexer nas
        # outras duas trava a caixa por torque e esconde o giro do punho, que é
        # justamente o que se quer ler aqui.
        alvos = [i for i in range(m.ngeom)
                 if m.geom(i).name in ("box/box_geom", "robot/left_palm_pad",
                                       "robot/right_palm_pad")]
        if len(alvos) != 3:
            raise SystemExit(f"--atrito achou {len(alvos)} geoms de 3. Cena de outra versão?")
        for i in alvos:
            m.geom_friction[i, 0] *= args.atrito
    if args.mu_caixa:
        # ⚠ A MESMA prioridade do treino (`cena._spec_box`): a caixa (2) vence o robô (0) e os
        # pés (1), e o contato robô–caixa usa o μ, o condim e o solref DELA; a laje (3) vence a
        # caixa. Sem isso o MuJoCo usa o MAIOR μ do par, e a cápsula do pulso em 1,0 mascara
        # qualquer μ baixo da caixa.
        g_cx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
        g_ms = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
        if g_cx < 0 or g_ms < 0:
            raise SystemExit("--mu-caixa não achou a caixa ou a mesa. Cena de outra versão?")
        m.geom_priority[g_cx] = 2
        m.geom_priority[g_ms] = 3
        m.geom_friction[g_cx, 0] = args.mu_caixa
    if args.massa:
        # ⚠ MASSA E INÉRCIA JUNTAS. Escrever só `body_mass` reproduz o defeito do treino.
        # A caixa é um CUBO homogêneo, portanto `I = (2/3)·m·a²` com `a` a meia-aresta.
        # ⚠ O `mj_setConst` DEPOIS é obrigatório: ele refaz `body_invweight0` e
        # `body_subtreemass`, e sem ele o solver segue com os pesos da caixa antiga.
        # A placa `box/face_alvo` tem `density=0` e não entra na conta.
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box/box")
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
        if bid < 0 or gid < 0:
            raise SystemExit("--massa não achou `box/box` na cena. Cena de outra versão?")
        meia = float(m.geom_size[gid, 0])
        m.body_mass[bid] = args.massa
        m.body_inertia[bid] = (2.0 / 3.0) * args.massa * meia * meia
        mujoco.mj_setConst(m, d)

    nomes, faixa, q_def = regua_das_juntas(m, c)
    # ⚠ RESOLVIDO UMA VEZ, fora do laço: `enderecos_da_caixa` varre as juntas do
    # modelo, e chamá-lo a 50 Hz é desperdício puro.
    adr_caixa, _ = enderecos_da_caixa(m, int(c.id_caixa))
    # ⚠ A RÉGUA DO FREIO DE APERTO (05/10): as geoms que a força lê, resolvidas UMA vez.
    g_caixa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
    g_pe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_palm_pad")
    g_pd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/right_palm_pad")
    robo_geom = np.array([m.geom(i).name.startswith("robot/") for i in range(m.ngeom)])
    f6 = np.zeros(6)
    id_mocap_laje = int(m.body_mocapid[int(c.id_laje)])
    assert id_mocap_laje >= 0, "a laje não é mocap nesta cena; o CSV precisa da pose dela"
    ids_q = np.asarray(c.ids_junta_qpos, dtype=np.int64)
    ids_atuador = np.asarray(c.ids_atuador, dtype=np.int64)
    q_default_acao = np.asarray(c.q_default_acao, dtype=np.float64)
    escala_acao = np.asarray(c.escala_acao, dtype=np.float64)
    # Cada ação é rotulada pela junta que o ATUADOR dela move, sem supor que a ordem da
    # ação seja a de `nomes` (`exporta_cena.py:115-117`: as duas ordens podem divergir).
    nomes_acao = [m.joint(int(m.actuator_trnid[a, 0])).name.split("/")[-1]
                  for a in ids_atuador]
    decimation, physics_dt = int(c.decimation), float(c.physics_dt)
    dt = physics_dt * decimation

    restaura(m, d, c)
    if args.topo:
        poe_a_laje(m, d, c, args.topo)
    acao = np.zeros(ator.dim_saida)
    twist = np.zeros(3)
    rumo0 = None          # rumo do 1º passo; o laço de rumo segura este valor
    cima_b = EZ.copy()    # eixo da caixa que aponta para cima, no frame dela

    def rumo_da_raiz() -> float:
        qw, qx, qy, qz = d.qpos[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    linhas: list[dict] = []

    if args.tempo <= 0:
        raise SystemExit("--tempo tem de ser > 0")
    print(f"[registra] cena {args.cena}  checkpoint iter={ator.iteracao} ({ckpt})  "
          f"dt={dt*1000:.0f} ms ({1/dt:.0f} Hz)  tempo x{args.tempo:g}  rumo_k={args.rumo_k:g}  "
          f"solver {m.opt.iterations}/{m.opt.ls_iterations}  "
          f"topo {_topo_da_laje(m, d, c):.3f} m  "
          f"atrito x{args.atrito:g}  impratio {m.opt.impratio:g}  "
          f"caixa {args.massa or 1.0:g} kg  mu_caixa {args.mu_caixa or '-'}"
          + ("   ⚠ CENA FORA DO PADRÃO DO TREINO"
             if args.atrito != 1.0 or args.impratio else ""))
    print(f"[registra] roteiro: " + "  ".join(
        f"{f.rotulo}({ELOS[f.elo]},{f.segundos:g}s"
        + (f",vx={f.vx:g}" if f.vx else "") + ")" for f in fases)
        + f"   × {args.voltas}")

    viewer = None
    if not args.sem_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d, key_callback=lambda k: None)

    passo = 0
    relogio0 = time.perf_counter()
    try:
        for volta in range(args.voltas):
            for fase in fases:
                twist[:] = (fase.vx, 0.0, 0.0)
                # ⚠ Como o treino (`_captura_cima`): o PEGAR recaptura o "cima" da caixa
                # ao abrir; CARREGAR e BOTAR herdam o do PEGAR.
                if fase.elo in (I_PEGAR, I_REORIENTAR):
                    cima_b = gira_inverso(d.qpos[adr_caixa + 3:adr_caixa + 7], EZ)
                if fase.limpa_laje or fase.limpa_caixa:
                    limpa_a_cena(m, d, c, args.afasta,
                                 laje=fase.limpa_laje, caixa=fase.limpa_caixa)
                n = max(1, int(round(fase.segundos / dt)))
                for _ in range(n):
                    if viewer is not None and not viewer.is_running():
                        raise KeyboardInterrupt
                    t0 = time.perf_counter()

                    if args.rumo_k > 0:
                        rumo = rumo_da_raiz()
                        rumo0 = rumo if rumo0 is None else rumo0
                        erro = (rumo0 - rumo + np.pi) % (2 * np.pi) - np.pi
                        twist[2] = np.clip(args.rumo_k * erro, -1.6, 1.6)
                    alvo = alvo_do_elo(m, d, c, fase.elo)
                    giro, giro_ang = giro_de_pe(d.qpos[adr_caixa + 3:adr_caixa + 7], cima_b)
                    obs, _ = monta_observacao(m, d, c, twist, fase.elo, acao, alvo_w=alvo,
                                              giro_w=giro)
                    acao = ator(obs)
                    d.ctrl[ids_atuador] = q_default_acao + escala_acao * acao
                    for _ in range(decimation):
                        mujoco.mj_step(m, d)
                    # ⚠ obrigatório: o `mj_step` deixa xpos/xquat/sensordata atrasados
                    # um subpasso, e é depois dele que a observação e o log são lidos.
                    mujoco.mj_forward(m, d)

                    q = d.qpos[ids_q]
                    ln = {"passo": passo, "t": round(passo * dt, 4),
                          "fase": fase.rotulo, "elo": ELOS[fase.elo], "ckpt": ckpt}
                    # O alvo é o do estado ANTES do passo, o que o ator viu. No ANDAR é
                    # irrelevante: os canais da caixa zeram no gate.
                    for i, eixo in enumerate("xyz"):
                        ln[f"alvo_{eixo}"] = float(alvo[i])
                    ln["giro_ang"] = giro_ang
                    # ⚠ A FORÇA NA CAIXA, como o treino a mede (`aperto_excessivo`): soma das
                    # NORMAIS de todo contato robô → caixa (`fn_tot`, a régua do
                    # `forca_total_na_caixa`), as duas palmas em separado, e o Fz que o robô
                    # sustenta. `rel_*` = caixa − ponto médio dos pads: `rel_z` caindo é a
                    # caixa escorregando entre as mãos.
                    fn_e = fn_d = fn_tot = fz_rob = 0.0
                    n_rob = 0
                    for k_ct in range(d.ncon):
                        ct = d.contact[k_ct]
                        g1, g2 = int(ct.geom1), int(ct.geom2)
                        if g_caixa not in (g1, g2):
                            continue
                        outro = g2 if g1 == g_caixa else g1
                        if not robo_geom[outro]:
                            continue
                        mujoco.mj_contactForce(m, d, k_ct, f6)
                        fn_tot += abs(f6[0])
                        n_rob += 1
                        fw = ct.frame.reshape(3, 3).T @ f6[:3]
                        fz_rob += float((fw if g2 == g_caixa else -fw)[2])
                        if outro == g_pe:
                            fn_e += abs(f6[0])
                        elif outro == g_pd:
                            fn_d += abs(f6[0])
                    ln["fn_E"], ln["fn_D"], ln["fn_tot"] = fn_e, fn_d, fn_tot
                    ln["fz_rob"], ln["n_rob"] = fz_rob, n_rob
                    pm = 0.5 * (d.geom_xpos[g_pe] + d.geom_xpos[g_pd])
                    rel = d.qpos[adr_caixa:adr_caixa + 3] - pm
                    ln["rel_x"], ln["rel_y"], ln["rel_z"] = map(float, rel)
                    for k in range(len(ELOS)):
                        ln[f"oh_{ELOS[k].lower()}"] = int(k == fase.elo)
                    for i, nome in enumerate(nomes):
                        ln[f"q_{nome}"] = float(q[i])
                    # Com a escala de 0,075 rad/unidade e o torque de ±5 N·m do punho, uma
                    # junta longe do alvo do PD separa "a política comandou" de "o contato
                    # empurrou": `a_` enorme é comando; `a_` pequeno com `q` longe é contato.
                    for i, nome in enumerate(nomes_acao):
                        ln[f"a_{nome}"] = float(acao[i])
                        ln[f"alvo_pd_{nome}"] = float(q_default_acao[i]
                                                      + escala_acao[i] * acao[i])
                    # ⚠⚠ A RAIZ E A CAIXA, e sem elas metade das grandezas de forma é
                    # incalculável. As 29 juntas dão a pose ARTICULAR; a altura da
                    # pelve no mundo, a inclinação do tronco, o CoM e a palma no frame
                    # da caixa são todas de MUNDO. Sem estas 14 colunas a sonda tem de
                    # SUPOR pelve vertical e pés planos, e a suposição erra quando o
                    # robô inclina a base — que é justamente a pose sob suspeita.
                    for i, eixo in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                        ln[f"raiz_{eixo}"] = float(d.qpos[i])
                        ln[f"caixa_{eixo}"] = float(d.qpos[adr_caixa + i])
                        # ⚠ A LAJE TAMBÉM (17/09): sem a pose dela a sonda não mede
                        # perna, tronco ou pé encostado no tampo — o dono viu isso no
                        # viewer e o CSV não tinha como confirmar.
                        ln[f"laje_{eixo}"] = float(d.mocap_pos[id_mocap_laje, i] if i < 3
                                                   else d.mocap_quat[id_mocap_laje, i - 3])
                    linhas.append(ln)
                    passo += 1

                    if viewer is not None:
                        viewer.sync()
                        atraso = dt / args.tempo - (time.perf_counter() - t0)
                        if atraso > 0:
                            time.sleep(atraso)
                    # ⚠ O FATOR MEDIDO, e não o pedido. `sleep` só atrasa: se ele
                    # aparecer acima do `--tempo`, a CPU não está dando conta e a
                    # cena corre mais que o pedido — o número diz qual dos dois é.
                    if passo % 50 == 0:
                        parede = time.perf_counter() - relogio0
                        medido = (passo * dt) / parede if parede > 0 else 0.0
                        print(f"\r  passo {passo:<6d} fase {fase.rotulo:<8s} "
                              f"tempo medido x{medido:.2f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[registra] interrompido — gravando o que já rodou")
    finally:
        if viewer is not None:
            viewer.close()

    if not linhas:
        raise SystemExit("nada gravado")

    saida = Path(args.saida).expanduser()
    csv_grande = saida.with_suffix(".csv")
    with open(csv_grande, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    csv_regua = saida.with_suffix(".limites.csv")
    with open(csv_regua, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["junta", "lo", "hi", "default"])
        for i, nome in enumerate(nomes):
            w.writerow([nome, faixa[i, 0], faixa[i, 1], q_def[i]])

    print(f"\n[registra] {len(linhas)} linhas × {len(linhas[0])} colunas -> {csv_grande}")
    print(f"[registra] régua das juntas -> {csv_regua}")


if __name__ == "__main__":
    main()
