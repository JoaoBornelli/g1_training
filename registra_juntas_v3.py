"""Roda uma cadeia da v3 no MuJoCo clássico e grava as juntas num CSV.

    python registra_juntas_v3.py --cena ~/g1_pilota_v3/ --checkpoint ~/Downloads/model_9499.pt
    python registra_juntas_v3.py ... --roteiro carregar --saida carrega
    python registra_juntas_v3.py ... --roteiro escada --coluna escada --linha 4 --borda 1 --saida escada

Irmão do `pilota_v3.py` (D11 do plano): mesma cena, mesmo `Percepcao`, mesma `monta_observacao` e
mesmo `Ator`, todos IMPORTADOS dele. A diferença é que aqui NÃO tem teclado — um roteiro troca a
tarefa sozinho a cada N segundos, e cada passo vira uma linha do CSV.

⚠ Ele importa do `pilota_v3.py` e por isso mora na RAIZ. Nada aqui depende do `mjlab`.

O que sai, e nada mais:
  <saida>.csv           passo, t, fase, tarefa, one-hot, o ÂNGULO CRU das 29 juntas, a POSE DE
                        MUNDO da raiz, da caixa e da laje (7 números cada: x, y, z, qw, qx, qy, qz),
                        `ckpt`, `giro_ang`, `fracao_mapa_visto`, `a_<junta>`, `alvo_pd_<junta>`,
                        a FORÇA NA CAIXA (`fn_E`, `fn_D`, `fn_tot`, `fz_rob`, `n_rob`), `rel_x/y/z`,
                        `aviso_fim` (0/1, o detector de fim), `idade_caixa` (s) e `alvo_x/y/z` (m, mundo; NaN
                        fora do PEGAR e do BOTAR, e no PEGAR antes da primeira leitura válida da caixa),
                        `coluna`, `linha`, `borda` (o sub-terreno da cena; vazio e -1 sem ele)
  <saida>.limites.csv   junta, lo, hi, default — a régua para ler o CSV

⚠ O DRIVER É A BASE. Nada de física extra aqui (sem --atrito, --massa, --mu-caixa...): a cena é a que
o `pilota_v3` carrega, e o laço de rumo é o do `Piloto` dele. Se um registro precisar de física
diferente, ela entra no driver primeiro e depois aqui.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np

import pilota_v3 as pv
from pilota_v3 import (TAREFAS, Ator, Percepcao, Piloto, Visao, carrega_cena, gira_inverso,
                       monta_observacao, restaura)

EZ = np.array([0.0, 0.0, 1.0])
# ⚠ `gira_inverso(conj(q), v) = R(q)·v`. O `pilota_v3` só expõe a inversa, e o conjugado do quatérnion
# é a inversa da rotação: é isso que deixa a conta do tombo sem copiar o `gira` do `g1_limpo`.
CONJ = np.array([1.0, -1.0, -1.0, -1.0])

# ⚠ Tarefas que, ao abrir, recapturam o "cima" da caixa (como o treino, `_captura_cima`). As outras
# herdam o do PEGAR, e é com ele que o tombo é medido.
TAREFAS_QUE_CAPTURAM_CIMA = ("PEGAR", "REORIENTAR")


def angulo_de_tombo(quat_caixa: np.ndarray, cima_b: np.ndarray) -> float:
    """Tombo da caixa em graus: o ângulo entre a normal `R·cima_b` e o `ez` de mundo.

    Mesma conta do `giro_de_pe` do `registra_juntas.py`, sem o vetor de correção. É só leitura para o CSV:
    o giro não entra no ator.
    """
    normal_w = gira_inverso(quat_caixa * CONJ, cima_b)
    return float(np.degrees(np.arccos(np.clip(normal_w @ EZ, -1.0, 1.0))))


# ⚠ A cena é exportada do reset do PEGAR: a caixa e a laje já estão à frente do robô.
# Por isso todo roteiro abre na espera e não numa aproximação.
#
# As cadeias do treino, uma em cada entrada. Na v3 cada fase é a tecla do teclado: `v` = PEGAR, `n` = BOTAR,
# `z` = ANDAR, `b` = CARREGAR.
#   c         cadeia C (F7): PEGAR -> CARREGAR parado -> BOTAR -> ANDAR parado. A caixa vai para a laje de destino
#             (o alvo é o topo da laje + a) e o robô sai sem ela. A laje e a caixa ficam onde o BOTAR as deixou.
#   carregar  roteiro B (cadeia B, E9): ANDAR parado -> PEGAR -> CARREGAR. A caixa FICA NA MÃO e o robô anda com ela.
#             O PEGAR dura o tempo do roteiro; o aviso de fim vai só para a coluna `aviso_fim`.
_CARREGAR_TERRENO = "espera:ANDAR:1.0,pegar:PEGAR:7.0,espera:CARREGAR:1.0,carregar:CARREGAR:8.0:0.25"
ROTEIROS = {
    "c": ("espera:ANDAR:1.0,"
          "pegar:PEGAR:7.0,"
          "espera:CARREGAR:1.0,"
          # ⚠ 7,0 s, a mesma janela do PEGAR. No treino o BOTAR termina pelo detector (mais U[0; 1,5] s de atraso), com
          # B2 + B3 + B4 = 1,3 s de permanência e o trajeto até a laje. O aviso sai em `aviso_fim`, e o roteiro não corta nele.
          # M11 (F9): a duração se confere pela coluna `aviso_fim` na fase I (o treino troca em aviso + U[0; 1,5] s).
          "botar:BOTAR:7.0,"
          "parado:ANDAR:1.5"),
    "carregar": ("espera:ANDAR:1.0,"
                 "pegar:PEGAR:7.0,"
                 "espera:CARREGAR:1.0,"
                 "carregar:CARREGAR:10.0:0.8"),
    # G6: a mesma sequência do `carregar`, com o CARREGAR a 0,25 m/s por 8 s (a velocidade do terreno, viab. D10).
    # A coluna NÃO vem do nome: vem de `--coluna`. Sugestões: escada -> escada; rampa -> rampa; obstaculo -> obstaculos.
    "escada": _CARREGAR_TERRENO,
    "rampa": _CARREGAR_TERRENO,
    "obstaculo": _CARREGAR_TERRENO,
}
ROTEIROS_DE_TERRENO = ("escada", "rampa", "obstaculo")


class Fase:
    """Um trecho do roteiro: rótulo, tarefa publicada, duração e o vx pedido."""

    def __init__(self, rotulo: str, tarefa: str, segundos: float, vx: float):
        self.rotulo, self.tarefa, self.segundos, self.vx = rotulo, tarefa, segundos, vx

    @property
    def _marcha(self) -> bool:
        return abs(self.vx) > 1e-6

    @property
    def limpa_caixa(self) -> bool:
        """⚠ A caixa só some no `ANDAR`. No `CARREGAR` ela está NA MÃO.

        Confundir os dois apagava a caixa do meio da tarefa: `CARREGAR` É andar com ela.
        """
        return self._marcha and self.tarefa == "ANDAR"

    @property
    def limpa_laje(self) -> bool:
        """A laje sai só no `CARREGAR` que ANDA (com marcha), e no `ANDAR` quando ele marcha.

        M10 (F9): como o treino (`comando.py:501`, `laje_fora` na abertura do CARREGAR que anda). No CARREGAR parado
        (a espera dos roteiros B e C) a laje fica.
        Nota: o `pilota_v3` NÃO move a laje nem a caixa (só o reset as põe no lugar). A limpeza espelha o
        `estaciona_cena` da fase E (no ANDAR a laje e a caixa saem; §2.2).
        """
        return self._marcha and self.tarefa in ("ANDAR", "CARREGAR")


def analisa_roteiro(texto: str) -> list[Fase]:
    """`"rotulo:TAREFA:segundos[:vx], ..."` -> lista de `Fase`. Um nome de `ROTEIROS` serve.

    ⚠ A tarefa é o NOME (`PEGAR`), e não o índice: um índice trocado é um erro silencioso.
    """
    texto = ROTEIROS.get(texto.strip(), texto)
    fases: list[Fase] = []
    for pedaco in texto.split(","):
        pedaco = pedaco.strip()
        if not pedaco:
            continue
        campos = pedaco.split(":")
        if len(campos) not in (3, 4):
            raise SystemExit(f"fase malformada: {pedaco!r} — use rotulo:TAREFA:segundos[:vx] "
                             f"ou um nome pronto: {', '.join(ROTEIROS)}")
        rotulo, nome, seg = campos[0], campos[1].upper(), campos[2]
        if nome not in TAREFAS:
            raise SystemExit(f"tarefa {nome!r} não existe; use uma de {', '.join(TAREFAS)}")
        vx = float(campos[3]) if len(campos) == 4 else 0.0
        fases.append(Fase(rotulo, nome, float(seg), vx))
    if not fases:
        raise SystemExit("roteiro vazio")
    return fases


def enderecos_da_caixa(m: mujoco.MjModel, id_caixa: int) -> tuple[int, int]:
    """`(qpos_adr, dof_adr)` da junta livre da caixa."""
    jid = int(m.body_jntadr[id_caixa])
    if jid < 0:
        raise SystemExit("a caixa não tem junta — não dá para afastá-la")
    return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def _rz(yaw: float) -> np.ndarray:
    """Matriz de rotação de `yaw` em torno de z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produto de quatérnios wxyz (a·b)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def pose_no_sub_terreno(pos: np.ndarray, quat: np.ndarray, origem_reset: np.ndarray, origem_sub: np.ndarray,
                        borda: int, laje_do_centro: float, laje_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Leva a pose de um corpo da cena do reset ao sub-terreno, com a laje a `laje_do_centro` do centro, rumo à borda.

    A cena inteira gira `borda`·90° em torno da origem do reset e vai para `origem_sub`; depois o conjunto anda
    `laje_do_centro − laje_x` rumo à borda (o desloc da G1). Robô, caixa e laje recebem a MESMA transformada,
    então o robô continua onde estava em relação à caixa.
    """
    yaw = borda * np.pi / 2
    rz = _rz(yaw)
    desloc = rz @ np.array([laje_do_centro - laje_x, 0.0, 0.0])
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return np.asarray(origem_sub) + desloc + rz @ (pos - origem_reset), _qmul(q_yaw, quat)


def resolve_terreno(roteiro: str, coluna: str | None, linha: int | None, borda: int, colunas: list[str],
                    linha_max: dict[str, int]) -> int | None:
    """Índice da coluna do gerador para `--coluna`, ou None sem terreno. Recusa o que o CARREGAR não cobre.

    ⚠ Um roteiro de terreno sem `--coluna` sai com erro: o nome do roteiro não escolhe a coluna.
    ⚠ Só as colunas de `linha_max` têm plataforma de 2 m (PD-20); a linha tem de ficar até o teto da coluna.
    """
    if coluna is None:
        if linha is not None:
            raise SystemExit("--linha sem --coluna")
        if roteiro in ROTEIROS_DE_TERRENO:
            raise SystemExit(f"o roteiro {roteiro!r} é de terreno e pede --coluna e --linha (ver a docstring)")
        return None
    if linha is None:
        raise SystemExit("--coluna pede --linha")
    if coluna not in linha_max:
        raise SystemExit(f"a coluna {coluna!r} não tem plataforma de 2 m (PD-20): use uma de {', '.join(linha_max)}")
    if not 0 <= linha <= linha_max[coluna]:
        raise SystemExit(f"linha {linha} fora de 0..{linha_max[coluna]} na coluna {coluna!r} (teto de CARREGAR, PD-20)")
    if not 0 <= borda <= 3:
        raise SystemExit("--borda vai de 0 a 3 (90° em 90°)")
    return colunas.index(coluna)


def limpa_a_cena(m: mujoco.MjModel, d: mujoco.MjData, c: dict[str, np.ndarray], afasta: float,
                 laje: bool, caixa: bool) -> None:
    """Manda para longe o que a tarefa manda, como o treino faz.

    ⚠ NO `ANDAR` VÃO AS DUAS: tirar só a laje deixaria a caixa caindo no caminho.
    ⚠ NO `CARREGAR` VAI SÓ A LAJE: a caixa está na mão, e andar com ela é a tarefa.
    """
    if laje:
        id_mocap = int(m.body_mocapid[int(c["id_laje"])])
        if id_mocap >= 0:
            d.mocap_pos[id_mocap, 0] += afasta
        else:
            print("⚠ a laje não é mocap; ela ficou onde estava")
    if caixa:
        adr_q, adr_v = enderecos_da_caixa(m, int(c["id_caixa"]))
        d.qpos[adr_q] += afasta
        d.qvel[adr_v:adr_v + 6] = 0.0
    mujoco.mj_forward(m, d)


def regua_das_juntas(m: mujoco.MjModel, c: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Nomes, faixa `(29, 2)` e default `(29,)`, na ordem do `nomes_juntas` da cena.

    ⚠ A JUNTA É ACHADA PELO ENDEREÇO DE `qpos`, e NÃO pelo nome: o `cena.npz` grava
    `left_hip_pitch_joint`, e o modelo compilado chama `robot/left_hip_pitch_joint`. O `mj_name2id`
    devolveria −1 nas 29, e a faixa sairia toda NaN sem erro. O `ids_junta_qpos` aponta a junta certa.
    """
    nomes = [str(n) for n in c["nomes_juntas"]]
    ids_q = np.asarray(c["ids_junta_qpos"], dtype=np.int64)
    por_adr = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
    faixa = np.zeros((len(nomes), 2))
    faltando = []
    for i in range(len(nomes)):
        jid = por_adr.get(int(ids_q[i]), -1)
        if jid < 0 or not bool(m.jnt_limited[jid]):
            # ⚠ Junta sem limite vira NaN, e NÃO 0: um zero calado viraria `frac` inventado.
            faixa[i] = (np.nan, np.nan)
            faltando.append(nomes[i])
        else:
            faixa[i] = m.jnt_range[jid]
    if faltando:
        print(f"⚠ sem faixa no modelo ({len(faltando)}): {', '.join(faltando[:6])}"
              + (" ..." if len(faltando) > 6 else ""))
    return nomes, faixa, np.asarray(c["q_default"], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb, cena.npz e contrato.json")
    ap.add_argument("--checkpoint", required=True, help="o model_*.pt da v3")
    ap.add_argument("--saida", default="juntas", help="prefixo dos arquivos de saída")
    ap.add_argument("--roteiro", default="c",
                    help=f"um nome pronto ({', '.join(ROTEIROS)}) ou "
                         "rotulo:TAREFA:segundos[:vx] separados por vírgula")
    ap.add_argument("--voltas", type=int, default=1, help="quantas vezes repetir o roteiro")
    ap.add_argument("--afasta", type=float, default=10.0,
                    help="metros em +x para onde a cena vai na marcha (ANDAR: laje e "
                         "caixa; CARREGAR: só a laje)")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="fator de tempo do viewer: 1,0 = tempo real, 0,25 = 4x lento")
    ap.add_argument("--sem-viewer", action="store_true", help="roda o mais rápido que der")
    ap.add_argument("--coluna", default=None,
                    help="coluna do gerador de terreno (ex.: escada, rampa, obstaculos): leva a cena a esse sub-terreno")
    ap.add_argument("--linha", type=int, default=None, help="linha do gerador (até o teto da coluna, PD-20)")
    ap.add_argument("--borda", type=int, default=0,
                    help="borda da plataforma para onde a laje olha: 0..3, de 90° em 90° (só com --coluna)")
    args = ap.parse_args()
    if args.tempo <= 0:
        raise SystemExit("--tempo tem de ser > 0")

    fases = analisa_roteiro(args.roteiro)
    m, c, _contrato = carrega_cena(args.cena)
    if args.coluna is not None and "origens_terreno" not in c:
        raise SystemExit("a cena não tem o terreno (origens_terreno, colunas_terreno): reexporte com exporta_cena")
    indice_coluna = resolve_terreno(args.roteiro, args.coluna, args.linha, args.borda,
                                    [str(n) for n in c.get("colunas_terreno", ())],
                                    dict(pv._KN.terreno.linha_max_carregar))
    d = mujoco.MjData(m)
    ator = Ator(args.checkpoint)
    # ⚠ Duas cópias com a mesma iteração geram CSVs indistinguíveis; nome + hash dos bytes identifica.
    arq_ckpt = Path(args.checkpoint).expanduser()
    ckpt = f"{arq_ckpt.name}:{hashlib.sha256(arq_ckpt.read_bytes()).hexdigest()[:10]}"
    if ator.dim_entrada != int(c["dim_obs"]):
        raise SystemExit(f"o checkpoint espera {ator.dim_entrada} canais e a cena monta "
                         f"{int(c['dim_obs'])}. Checkpoint de outra fase?")

    nomes, faixa, q_def = regua_das_juntas(m, c)
    # ⚠ RESOLVIDO UMA VEZ, fora do laço: `enderecos_da_caixa` varre as juntas do modelo.
    adr_caixa, _ = enderecos_da_caixa(m, int(c["id_caixa"]))
    # ⚠ A RÉGUA DA FORÇA NA CAIXA: as geoms que a força lê, resolvidas UMA vez.
    g_caixa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
    g_pe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_palm_pad")
    g_pd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/right_palm_pad")
    robo_geom = np.array([m.geom(i).name.startswith("robot/") for i in range(m.ngeom)])
    f6 = np.zeros(6)
    id_mocap_laje = int(m.body_mocapid[int(c["id_laje"])])
    assert id_mocap_laje >= 0, "a laje não é mocap nesta cena; o CSV precisa da pose dela"
    ids_q = np.asarray(c["ids_junta_qpos"], dtype=np.int64)
    ids_atuador = np.asarray(c["ids_atuador"], dtype=np.int64)
    q_default_acao = np.asarray(c["q_default_acao"], dtype=np.float64)
    escala_acao = np.asarray(c["escala_acao"], dtype=np.float64)
    # Cada ação é rotulada pela junta que o ATUADOR dela move, sem supor que a ordem seja a de `nomes`.
    nomes_acao = [m.joint(int(m.actuator_trnid[a, 0])).name.split("/")[-1]
                  for a in ids_atuador]
    decimation, physics_dt = int(c["decimation"]), float(c["physics_dt"])
    dt = physics_dt * decimation
    id_base = int(c["id_base"])

    # ⚠ O MESMO laço de rumo e os MESMOS tetos do driver: o `Piloto` dele só segura o rumo na marcha
    # reta, e o `_TETOS_PADRAO` é o teto de quem dirige (o envelope do treino é o limite de cima).
    envelope = np.asarray(c["envelope_treino"], dtype=np.float64)
    tetos = np.minimum(pv._TETOS_PADRAO, envelope)
    piloto = Piloto(tuple(tetos), envelope)
    # ⚠ O SUB-TERRENO entra no `c` ANTES do `Percepcao` e do `restaura`: o `restaura` monta o estado do reset a
    # partir de `qpos_inicial` e `mocap_*_inicial`, e a percepção e a visão leem a pose já no sub-terreno.
    # Velocidades: o reset grava qvel zero; se um dia não for zero, a linear também gira (não feito aqui).
    # ⚠ DÚVIDA: o `--afasta` (laje e caixa, na marcha) anda em +x do MUNDO; com `--borda` ≠ 0 o afastamento sai de
    # lado do robô, e não na direção da borda. Não mudei: o orquestrador decide se o afasta passa a seguir o rumo.
    sub_terreno = indice_coluna is not None
    if sub_terreno:
        origem_reset = np.asarray(c["origem_reset"], dtype=np.float64)
        origem_sub = np.asarray(c["origens_terreno"], dtype=np.float64)[args.linha, indice_coluna]
        laje_x = float(pv._KN.cena.prateleira_xy[0])
        laje_do_centro = pv._KN.terreno.laje_do_centro

        def leva(pos, quat):
            return pose_no_sub_terreno(pos, quat, origem_reset, origem_sub, args.borda, laje_do_centro, laje_x)

        adr_caixa_reset, _ = enderecos_da_caixa(m, int(c["id_caixa"]))
        q = np.array(c["qpos_inicial"], dtype=np.float64)
        for ini in (0, adr_caixa_reset):  # a raiz (pelve) e a caixa: juntas livres, pose em qpos[adr:adr+7]
            q[ini:ini + 3], q[ini + 3:ini + 7] = leva(q[ini:ini + 3], q[ini + 3:ini + 7])
        mp = np.array(c["mocap_pos_inicial"], dtype=np.float64)
        mq = np.array(c["mocap_quat_inicial"], dtype=np.float64)
        id_mocap_reset = int(m.body_mocapid[int(c["id_laje"])])
        mp[id_mocap_reset], mq[id_mocap_reset] = leva(mp[id_mocap_reset], mq[id_mocap_reset])
        c.update(qpos_inicial=q, mocap_pos_inicial=mp, mocap_quat_inicial=mq)
    sub = dict(coluna=args.coluna or "", linha=args.linha if sub_terreno else -1,
               borda=args.borda if sub_terreno else -1)

    percepcao = Percepcao(m, c)
    # ⚠ Limiar do mapa: o mesmo de `metricas.fracao_mapa_visto` (a fração de células VISTAS).
    limiar_visto = pv._KN.percepcao.limiar_visto

    visao = Visao(m, c, percepcao)
    restaura(m, d, c, percepcao, visao)
    acao = np.zeros(ator.dim_saida)
    hist: np.ndarray | None = None
    cima_b = EZ.copy()            # eixo da caixa que aponta para cima, no frame dela
    linhas: list[dict] = []

    print(f"[registra_v3] cena {args.cena}  checkpoint iter={ator.iteracao} ({ckpt})  "
          f"dt={dt * 1000:.0f} ms ({1 / dt:.0f} Hz)  tempo x{args.tempo:g}  teto {tetos}")
    print("[registra_v3] roteiro: " + "  ".join(
        f"{f.rotulo}({f.tarefa},{f.segundos:g}s" + (f",vx={f.vx:g}" if f.vx else "") + ")"
        for f in fases) + f"   × {args.voltas}")
    print("[registra_v3] fase E: caixa pelo estimador da D435i (sem ruído); aviso de fim na coluna aviso_fim")
    if sub_terreno:
        print(f"[registra_v3] sub-terreno: coluna {args.coluna} linha {args.linha} borda {args.borda} "
              f"origem {origem_sub.round(3).tolist()}")

    viewer = None
    if not args.sem_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d, key_callback=lambda k: None)

    passo = 0
    relogio0 = time.perf_counter()
    try:
        for _volta in range(args.voltas):
            for fase in fases:
                piloto.tarefa = TAREFAS.index(fase.tarefa)
                piloto.twist[:] = (fase.vx, 0.0, 0.0)
                if fase.tarefa in ("PEGAR", "BOTAR"):
                    # Como a tecla: o alvo do PEGAR é a caixa VISTA (S5b); o do BOTAR é a laje de destino (F7).
                    visao.abre(piloto.tarefa, m, d)
                if fase.tarefa in TAREFAS_QUE_CAPTURAM_CIMA:
                    cima_b = gira_inverso(d.qpos[adr_caixa + 3:adr_caixa + 7], EZ)
                if fase.limpa_laje or fase.limpa_caixa:
                    limpa_a_cena(m, d, c, args.afasta,
                                 laje=fase.limpa_laje, caixa=fase.limpa_caixa)
                n = max(1, int(round(fase.segundos / dt)))
                for _ in range(n):
                    if viewer is not None and not viewer.is_running():
                        raise KeyboardInterrupt
                    t0 = time.perf_counter()

                    twist_pol = piloto.twist_da_politica(pv._yaw(d.xquat[id_base]))
                    # ⚠ A ORDEM DO DRIVER: `prediz`/`atualiza` do mapa acontecem dentro de `passo`, ANTES
                    # da observação. Trocar a ordem muda o mapa que o ator vê.
                    mapa = percepcao.passo(m, d)
                    # ⚠ A caixa depois do `Percepcao.passo` (a odometria é dele), e antes da observação.
                    caixa = visao.passo(m, d, piloto.tarefa)
                    obs, hist = monta_observacao(d, c, twist_pol, piloto.tarefa, acao, caixa, mapa, hist)
                    acao = ator(obs)
                    d.ctrl[ids_atuador] = q_default_acao + escala_acao * acao
                    for _ in range(decimation):
                        mujoco.mj_step(m, d)
                    # ⚠ obrigatório: o `mj_step` deixa xpos/xquat/sensordata atrasados um subpasso.
                    mujoco.mj_forward(m, d)

                    q = d.qpos[ids_q]
                    ln = {"passo": passo, "t": round(passo * dt, 4),
                          "fase": fase.rotulo, "tarefa": fase.tarefa, "ckpt": ckpt, **sub}
                    # O alvo gravado é o PONTO do PEGAR (a caixa vista) ou o de soltura do BOTAR (a laje, F7), o que o
                    # driver manda; I §4.1. O `alvos.caixa_alvo_w` pede o índice P0..B4, que o driver não tem.
                    ax, ay, az = (visao.alvo_w[0].tolist() if fase.tarefa in ("PEGAR", "BOTAR") and visao.aberto
                                  else (float("nan"),) * 3)
                    ln.update(alvo_x=ax, alvo_y=ay, alvo_z=az)
                    ln["giro_ang"] = angulo_de_tombo(d.qpos[adr_caixa + 3:adr_caixa + 7], cima_b)
                    # Status do detector, lido depois do `mj_forward` (a pose que acabou de sair do passo).
                    ln["aviso_fim"] = int(visao.detecta(d, piloto.tarefa))
                    # Idade da leitura que o ator viu neste passo (a `saida_b`, cortada em idade_max).
                    ln["idade_caixa"] = float(caixa[7])
                    # Mapa = 441 alturas e depois 441 confianças (I §2.4). Mesma conta de `metricas.py`.
                    ln["fracao_mapa_visto"] = float(np.mean(mapa[len(mapa) // 2:] > limiar_visto))
                    # ⚠ A FORÇA NA CAIXA, como o treino a mede (`aperto_excessivo`): soma das NORMAIS de todo
                    # contato robô → caixa, as duas palmas em separado, e o Fz que o robô sustenta.
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
                    for nome in TAREFAS:
                        ln[f"oh_{nome.lower()}"] = int(nome == fase.tarefa)
                    for i, nome in enumerate(nomes):
                        ln[f"q_{nome}"] = float(q[i])
                    # Com a escala e o torque do punho, uma junta longe do alvo do PD separa "a política
                    # comandou" de "o contato empurrou": `a_` enorme é comando.
                    for i, nome in enumerate(nomes_acao):
                        ln[f"a_{nome}"] = float(acao[i])
                        ln[f"alvo_pd_{nome}"] = float(q_default_acao[i] + escala_acao[i] * acao[i])
                    # ⚠⚠ A RAIZ, A CAIXA E A LAJE: sem elas a sonda tem de SUPOR pelve vertical e pés
                    # planos, e a suposição erra justamente quando o robô inclina a base.
                    for i, eixo in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                        ln[f"raiz_{eixo}"] = float(d.qpos[i])
                        ln[f"caixa_{eixo}"] = float(d.qpos[adr_caixa + i])
                        ln[f"laje_{eixo}"] = float(d.mocap_pos[id_mocap_laje, i] if i < 3
                                                   else d.mocap_quat[id_mocap_laje, i - 3])
                    linhas.append(ln)
                    passo += 1

                    if viewer is not None:
                        viewer.sync()
                        atraso = dt / args.tempo - (time.perf_counter() - t0)
                        if atraso > 0:
                            time.sleep(atraso)
                    # ⚠ O FATOR MEDIDO, e não o pedido: se ele passar do `--tempo`, a CPU não dá conta.
                    if passo % 50 == 0:
                        parede = time.perf_counter() - relogio0
                        medido = (passo * dt) / parede if parede > 0 else 0.0
                        print(f"\r  passo {passo:<6d} fase {fase.rotulo:<8s} "
                              f"tempo medido x{medido:.2f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[registra_v3] interrompido — gravando o que já rodou")
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

    print(f"\n[registra_v3] {len(linhas)} linhas × {len(linhas[0])} colunas -> {csv_grande}")
    print(f"[registra_v3] régua das juntas -> {csv_regua}")


if __name__ == "__main__":
    main()
