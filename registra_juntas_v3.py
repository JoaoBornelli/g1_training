"""Roda uma cadeia da v3 no MuJoCo clássico e grava as juntas num CSV.

    python registra_juntas_v3.py --cena ~/g1_pilota_v3/ --checkpoint ~/Downloads/model_9499.pt
    python registra_juntas_v3.py ... --roteiro carregar --saida carrega

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
                        fora do PEGAR)
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
# As duas cadeias que o treino fecha depois do PEGAR, uma em cada entrada. Na v3 cada fase é a tecla
# do teclado: `v` = PEGAR, `n` = BOTAR, `z` = ANDAR, `b` = CARREGAR.
#   botar     PEGAR -> BOTAR -> ANDAR. A caixa vai para a laje e o robô sai sem ela.
#   carregar  roteiro B (cadeia B, E9): ANDAR parado -> PEGAR -> CARREGAR. A caixa FICA NA MÃO e o robô anda com ela.
#             O PEGAR dura o tempo do roteiro; o aviso de fim vai só para a coluna `aviso_fim`.
ROTEIROS = {
    "botar": ("espera:ANDAR:1.0,"
              "pegar:PEGAR:3.0,"
              "espera:BOTAR:1.0,"
              # ⚠ 2,0 s, e não 0,5. A sonda de forma lê os ÚLTIMOS 0,3 s da fase (a aproximação
              # do instante de fecho). Com 0,5 s a janela pegava 60% da fase, ainda assentando.
              "botar:BOTAR:2.0,"
              "espera:ANDAR:1.0,"
              "andar:ANDAR:6.0:0.8"),
    "carregar": ("espera:ANDAR:1.0,"
                 "pegar:PEGAR:7.0,"
                 "espera:CARREGAR:1.0,"
                 "carregar:CARREGAR:10.0:0.8"),
}


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
        """A laje sai no `CARREGAR` sempre, e no `ANDAR` quando ele marcha.

        ⚠ NO `CARREGAR` A MARCHA NÃO ENTRA NA CONTA: quem está em `CARREGAR` já não tem laje na frente.
        Nota: o `pilota_v3` NÃO move a laje nem a caixa (só o reset as põe no lugar). A limpeza fica como está,
        espelha o `estaciona_cena` da fase E (no ANDAR a laje e a caixa saem; §2.2).
        """
        return self.tarefa == "CARREGAR" or self.limpa_caixa


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
    ap.add_argument("--roteiro", default="botar",
                    help=f"um nome pronto ({', '.join(ROTEIROS)}) ou "
                         "rotulo:TAREFA:segundos[:vx] separados por vírgula")
    ap.add_argument("--voltas", type=int, default=1, help="quantas vezes repetir o roteiro")
    ap.add_argument("--afasta", type=float, default=10.0,
                    help="metros em +x para onde a cena vai na marcha (ANDAR: laje e "
                         "caixa; CARREGAR: só a laje)")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="fator de tempo do viewer: 1,0 = tempo real, 0,25 = 4x lento")
    ap.add_argument("--sem-viewer", action="store_true", help="roda o mais rápido que der")
    args = ap.parse_args()
    if args.tempo <= 0:
        raise SystemExit("--tempo tem de ser > 0")

    fases = analisa_roteiro(args.roteiro)
    m, c, _contrato = carrega_cena(args.cena)
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
                if fase.tarefa == "PEGAR":
                    visao.abre_pegar(d)  # o alvo do PEGAR é o centro da caixa neste instante (como a tecla)
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
                          "fase": fase.rotulo, "tarefa": fase.tarefa, "ckpt": ckpt}
                    # ⚠ DÚVIDA (E9): o alvo gravado é o PONTO do PEGAR (centro da caixa), o que o driver manda.
                    # O `alvos.caixa_alvo_w` pede o índice P0..B4, que é estado do controlador e o driver não tem.
                    ax, ay, az = visao.alvo_w[0].tolist() if fase.tarefa == "PEGAR" else (float("nan"),) * 3
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
