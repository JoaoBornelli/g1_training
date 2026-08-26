"""Revisão da cena e dos alvos, ANTES de gastar GPU.

    python -m g1_limpo.inspeciona --viewer pegar        # o alvo do `pegar`
    python -m g1_limpo.inspeciona --viewer botar
    python -m g1_limpo.inspeciona --viewer carregar
    python -m g1_limpo.inspeciona --viewer andar
    python -m g1_limpo.inspeciona --viewer reorientar

    python -m g1_limpo.inspeciona --tabela              # os 5 elos, no nível 0
    python -m g1_limpo.inspeciona --tabela pegar        # o `pegar` nos 7 níveis

Existe porque a GPU é alugada, e um erro de geometria descoberto na iteração 800 de
um Kaggle custa a sessão.

⚠ UMA FONTE DE VERDADE. Este arquivo **não recalcula nada**. Ele INSTANCIA o mesmo
`make_env_cfg` que o treino usa, reseta, e REPORTA o que o ambiente produziu. O
desenho no viewer vem de dentro do `comando.py`. Se o alvo mudar lá, muda aqui.

⚠ O robô fica TRAVADO na pose de reset. Sem isso um robô sem política cai em meio
segundo e não dá para conferir alvo nenhum.

O ALVO DE CADA ELO É UMA COISA DIFERENTE, e é isso que se vê:

    andar       uma VELOCIDADE (o twist). Não há alvo de caixa: `valida = 0`.
    reorientar  uma ORIENTAÇÃO. A caixa fica onde está; o alvo É a caixa.
    pegar       um PONTO em altitude ABSOLUTA de mundo.
    carregar    um ponto no PEITO, no frame da BASE — anda com o robô.
    botar       um ponto LATERAL num TOPO NOVO, travado no fundo da caixa.
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

from g1_limpo import comando as CMD
from g1_limpo.knobs import Knobs

N_ENVS = 8


def _ambiente(nivel: int | None, elo: int, *, n_envs: int, device: str):
    """Constrói o MESMO cfg do treino, com nível e elo forçados. Nada de mock."""
    from mjlab.envs import ManagerBasedRlEnv

    from g1_limpo.env_cfg import make_env_cfg

    k = Knobs()
    k.nivel.forcado = nivel
    cfg = make_env_cfg(k, inspecao=True, elo=elo)
    cfg.scene.num_envs = n_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    for _ in range(2):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                             device=device))
    return env, k


def _lateral(env) -> torch.Tensor:
    """O deslocamento LATERAL do alvo no frame do robô. Zero = à frente."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    rb = env.scene["robot"]
    d = env.command_manager.get_command("alvo_caixa")[:, CMD.ALVO] \
        - rb.data.root_link_pos_w
    return quat_apply_inverse(rb.data.root_link_quat_w, d)[:, 1]


def _medidas(env) -> dict:
    o = env.scene.env_origins
    cx = env.scene["box"].data.root_link_pos_w
    pr = env.scene["table"].data.root_link_pos_w
    rb = env.scene["robot"].data.root_link_pos_w
    cmd = env.command_manager.get_command("alvo_caixa")
    tw = env.command_manager.get_command("twist")
    rel = torch.stack((o[:, 0], o[:, 1], torch.zeros_like(o[:, 2])), dim=-1)
    return {
        "nivel": env.limpo_nivel.clone(),
        "topo_laje": (pr[:, 2] + 0.0).clone(),
        "massa": env.limpo_massa.clone(),
        "caixa": (cx - rel).clone(),
        "alvo": (cmd[:, CMD.ALVO] - rel).clone(),
        "valida": cmd[:, CMD.VALIDA].clone(),
        "elo": cmd[:, CMD.ELO].clone(),
        "face_n": cmd[:, CMD.FACE].norm(dim=-1).clone(),
        "ang_deg": torch.rad2deg(cmd[:, CMD.ANG]).clone(),
        "robo_z": rb[:, 2].clone(),
        "caixa_alvo": (cmd[:, CMD.ALVO] - cx).norm(dim=-1).clone(),
        "pelve_alvo": (cmd[:, CMD.ALVO] - rb).norm(dim=-1).clone(),
        "dxy_pelve": (cmd[:, CMD.ALVO][:, :2] - rb[:, :2]).norm(dim=-1).clone(),
        # componente LATERAL do alvo no frame do robô: 0 = exatamente à frente
        "lateral": _lateral(env).clone(),
        "twist": tw.clone(),
        "voltas": (env.limpo_voltas.clone() if hasattr(env, "limpo_voltas")
                   else torch.zeros_like(rb[:, 0])),
        # o AZIMUTE da caixa: quanto a direção desejada (caixa->robô) foge do eixo
        # -x do mundo. Ele ENTRA no erro angular, e não é knob — é geometria.
        "azimute": torch.rad2deg(torch.acos(
            (-cmd[:, CMD.FACE][:, 0]).clamp(-1.0, 1.0))).clone(),
    }


def _sanidade(m: dict, k: Knobs, elo: int) -> list[str]:
    """Cada checagem pega um defeito REAL que este repositório já pagou."""
    c, n, a = k.cena, k.nivel, k.alvo
    f: list[str] = []
    niv = int(m["nivel"][0])
    meia_z, meia_cx = c.prateleira_meia_z, c.caixa_meia_aresta[2]
    topo = m["topo_laje"] + meia_z          # o topo real da laje

    # --- comuns a todos os elos ---
    if float((m["elo"] - float(elo)).abs().max()) > 1e-6:
        f.append("o elo publicado não é o elo forçado")
    # ⚠ LIMIAR DERIVADO, e não chutado. O evento escreve a pose e a física roda UM
    # passo de controle antes da leitura, portanto o robô cai `½·g·dt²` =
    # `0,5 · 9,81 · 0,02² = 2,0 mm` por construção. Medido: 2,5 mm de desvio e
    # desvio-padrão de 1e-5 a 3e-4 entre envs (o `botar` é o maior, porque o contato
    # da caixa varia). Um limiar de 1e-4 acusava a própria gravidade.
    if float(m["robo_z"].std()) > 1e-3:
        f.append(f"o robô não está travado: desvio-padrão da altura "
                 f"{float(m['robo_z'].std()):.2e} entre envs")
    if float((m["robo_z"] - 0.80).abs().max()) > 5e-3:
        f.append(f"o robô saiu da pose travada: "
                 f"{float((m['robo_z'] - 0.80).abs().max()):.4f} m de 0,80")
    # ⚠ `ang_deg` é o ERRO angular da face marcada, e não um pedido. O teto tem TRÊS
    # parcelas, e eu tinha esquecido a terceira:
    #
    #     voltas × 90°   o quarto de volta da orientação de nascimento
    #   + desalinho      o residual da célula
    #   + AZIMUTE        quanto a direção caixa->robô foge do eixo -x do mundo
    #
    # O azimute NÃO é knob, é geometria: a caixa nasce em y até ±0,18 com x ≈ 0,32,
    # portanto ele chega a `atan(0,18/0,32) = 29°`. Sem ele o check acusava o nível 0
    # com 25° contra um teto de 15°.
    #
    # O teto é POR ENV, com o azimute medido de cada um.
    teto = (m["voltas"] * 90.0 + n.desalinho_max_deg[niv] + m["azimute"])
    excesso = m["ang_deg"] - teto
    if float(excesso.max()) > 1.0:
        i = int(excesso.argmax())
        f.append(f"erro angular além do que a célula pode gerar, env {i}: "
                 f"{float(m['ang_deg'][i]):.1f}° > {float(teto[i]):.1f}° "
                 f"({float(m['voltas'][i]):.0f}×90 + {n.desalinho_max_deg[niv]:.0f} + "
                 f"azimute {float(m['azimute'][i]):.1f})")
    if float(m["massa"].max()) > n.carga_max[niv] + 1e-6:
        f.append(f"carga além da célula: {float(m['massa'].max()):.2f} kg")

    if elo == CMD.ANDAR:
        if float(m["valida"].max()) != 0.0:
            f.append("no `andar` o objetivo da caixa devia estar DESLIGADO (valida=0)")
        if float(topo.min()) < c.afasta_z - 1e-3:
            f.append(f"a mobília NÃO subiu: topo em {float(topo.min()):.2f} m, "
                     f"esperado {c.afasta_z}")
        if float(m["twist"].abs().max()) == 0.0:
            f.append("o twist está identicamente zero: não há velocidade a rastrear")
        return f

    if float(m["valida"].min()) != 1.0:
        f.append(f"no `{CMD.ELOS[elo]}` o objetivo da caixa devia estar LIGADO")
    if float((m["face_n"] - 1.0).abs().max()) > 1e-4:
        f.append("a normal da face não é unitária")

    if elo == CMD.REORIENTAR:
        if float(m["caixa_alvo"].max()) > 1e-3:
            f.append(f"no `reorientar` o alvo devia SER a caixa; distância "
                     f"{float(m['caixa_alvo'].max()):.4f} m")
        if float((topo - m["caixa"][:, 2] + meia_cx).abs().max()) > 5e-3:
            f.append("a caixa não está apoiada na laje")
        if float(m["ang_deg"].max()) < 1.0:
            f.append(f"no nível {niv} o erro angular máximo é "
                     f"{float(m['ang_deg'].max()):.1f}° — o `reorientar` é trivial "
                     f"aqui (informativo, não defeito)")

    if elo in (CMD.PEGAR, CMD.CARREGAR):
        # ⚠ Os dois pedem EXATAMENTE o mesmo alvo. O que difere é o twist.
        if float((m["alvo"][:, 2] - a.altura_carregar).abs().max()) > 1e-6:
            f.append(f"o z do alvo NÃO é absoluto em {a.altura_carregar}: "
                     f"{float(m['alvo'][:, 2].min()):.4f}–"
                     f"{float(m['alvo'][:, 2].max()):.4f}")
        # o xy segue o robô: distância HORIZONTAL = ‖peito_b.xy‖, e ela é IGUAL em
        # todos os envs porque não há jitter no alvo
        dxy_esp = (a.peito_b[0] ** 2 + a.peito_b[1] ** 2) ** 0.5
        if abs(float(m["dxy_pelve"].mean()) - dxy_esp) > 5e-3:
            f.append(f"o xy do alvo não acompanha o robô: "
                     f"{float(m['dxy_pelve'].mean()):.3f} m na horizontal, "
                     f"esperado {dxy_esp:.3f}")
        # ⚠ LIMIAR DERIVADO. A âncora é `quat_apply(base_q, peito_b)`, portanto a
        # projeção horizontal dela muda quando a base pende. Com o robô travado a
        # inclinação residual dá ~1e-4 de variação — isso é FÍSICA, não jitter de
        # alvo. Quem mede "está de lado?" de verdade é o check `lateral` abaixo.
        if float(m["dxy_pelve"].std()) > 1e-3:
            f.append(f"o alvo varia lateralmente ({float(m['dxy_pelve'].std()):.4f}): "
                     f"ele deve estar EXATAMENTE à frente do robô")
        # ⚠ o alvo tem de estar À FRENTE, no eixo do robô. Um alvo de lado é o defeito
        # que o dono viu no viewer em 25/08 (jitter em y de ±0,05 sobre x = 0,25
        # deslocava até 11° fora do eixo).
        if float(m["lateral"].abs().max()) > 5e-3:
            f.append(f"o alvo está DE LADO: {float(m['lateral'].abs().max()):.4f} m "
                     f"fora do eixo do robô")

    if elo == CMD.PEGAR:
        if float((m["alvo"][:, 2] - m["caixa"][:, 2]).min()) <= 0.0:
            f.append("existe env com o alvo ABAIXO da caixa: erguer seria de graça")
        if float(topo.min()) < -1e-6:
            f.append("laje ENTERRADA")
        # ⚠ o que impede o robô de ANDAR com a caixa é o twist em ZERO
        if float(m["twist"].abs().max()) > 1e-6:
            f.append(f"o twist do `pegar` NÃO está zerado: "
                     f"máximo {float(m['twist'].abs().max()):.4f}")

    if elo == CMD.CARREGAR:
        if float(topo.min()) < c.afasta_z - 1e-3:
            f.append("no `carregar` a mobília devia estar a +5 m")
        # no `carregar` o twist é ATIVO: é o que diferencia do `pegar`
        if float(m["twist"].abs().max()) == 0.0:
            f.append("o twist do `carregar` está zerado: ele devia estar ATIVO")

    if elo == CMD.BOTAR:
        # o TETO EFETIVO: a laje nunca pode nascer dentro da caixa
        # ⚠ ELEMENTO A ELEMENTO. Comparar `topo.max()` de um env contra
        # `(fundo−folga).min()` de OUTRO acusa violação onde não há: os dois vêm de
        # envs diferentes. O clamp é por env, e o check tem de ser também.
        # ⚠ TOLERÂNCIA DERIVADA. O clamp roda na passada de pose (passo 1) e a leitura
        # é no passo 2; entre os dois a caixa cai `½·g·dt² = 2,0 mm` antes de ser
        # re-pinada. Medido: 1 mm de violação residual. Um limiar de 0,1 mm acusava a
        # própria gravidade, e o que este check existe para pegar é violação GROSSA —
        # o defeito antigo punha a laje meio metro dentro da caixa.
        fundo = m["caixa"][:, 2] - meia_cx
        viola = topo - (fundo - a.botar_folga_laje)
        if float(viola.max()) > 5e-3:
            i = int(viola.argmax())
            f.append(f"a laje nasceu DENTRO da caixa no env {i}: topo "
                     f"{float(topo[i]):.3f} > fundo−folga "
                     f"{float(fundo[i] - a.botar_folga_laje):.3f}")
        if float(topo.min()) < a.botar_topo_piso - 1e-4:
            f.append(f"o topo do `botar` desceu abaixo do piso {a.botar_topo_piso}")
        if float((m["alvo"][:, 2] - (topo + meia_cx)).abs().max()) > 1e-3:
            f.append("o alvo do `botar` não está em cima do topo novo")

    if elo != CMD.CARREGAR and float(m["pelve_alvo"].max()) > CMD.ALCANCE_R + 1e-6:
        f.append(f"alvo FORA do alcance de referência: "
                 f"{float(m['pelve_alvo'].max()):.3f} > {CMD.ALCANCE_R}")
    return f


def tabela(args) -> int:
    fx = lambda t: f"{float(t.min()):.3f}–{float(t.max()):.3f}"        # noqa: E731

    if args.elo is not None:
        e = CMD.elo_por_nome(args.elo)
        casos = [(e, niv) for niv in range(Knobs().nivel.n_niveis)]
        titulo = f"ELO `{args.elo}` nos {Knobs().nivel.n_niveis} níveis"
    else:
        niv = args.nivel if args.nivel is not None else 0
        casos = [(i, niv) for i in range(len(CMD.ELOS))]
        titulo = f"OS {len(CMD.ELOS)} ELOS, no nível {niv}"

    print("=" * 118)
    print(f"{titulo} — MEDIDO no ambiente real. x,y relativos à origem do env; "
          f"z absoluto; metros")
    print("=" * 118)
    cab = (f"{'elo':>11} {'niv':>3} {'valida':>6} {'topo laje':>13} "
           f"{'caixa z':>13} {'alvo z':>13} {'caixa->alvo':>13} "
           f"{'pelve->alvo':>13} {'voltas':>7} {'azimute':>13} {'erro graus':>13}")
    print(cab)
    print("-" * len(cab))

    total = 0
    for elo, niv in casos:
        env, k = _ambiente(niv, elo, n_envs=args.envs_tabela, device=args.device)
        m = _medidas(env)
        print(f"{CMD.ELOS[elo]:>11} {niv:>3} "
              f"{float(m['valida'][0]):>6.0f} {fx(m['topo_laje'] + k.cena.prateleira_meia_z):>13} "
              f"{fx(m['caixa'][:, 2]):>13} {fx(m['alvo'][:, 2]):>13} "
              f"{fx(m['caixa_alvo']):>13} {fx(m['pelve_alvo']):>13} "
              f"{fx(m['voltas']):>7} {fx(m['azimute']):>13} {fx(m['ang_deg']):>13}")
        if elo == CMD.ANDAR:
            print(f"{'':>11}     twist vx,vy,wz = {fx(m['twist'][:, 0])} , "
                  f"{fx(m['twist'][:, 1])} , {fx(m['twist'][:, 2])}"
                  f"   (o alvo do `andar` é uma VELOCIDADE)")
        falhas = _sanidade(m, k, elo)
        for x in falhas:
            marca = "i" if "informativo" in x else "✗"
            print(f"      {marca} {x}")
            if marca == "✗":
                total += 1
        del env

    print()
    print(f"envs por caso: {args.envs_tabela}   |   robô: TRAVADO na pose de reset")
    print("=" * 118)
    if total:
        print(f"{total} FALHA(S) DE SANIDADE — não subir para a GPU assim")
        return 1
    print("0 falhas de sanidade")
    return 0


def viewer(args) -> int:
    import g1_limpo
    from mjlab.scripts.play import PlayConfig, run_play

    nome = (args.elo or "pegar").strip().lower()
    CMD.elo_por_nome(nome)                     # valida
    task = g1_limpo.TASK_INSPECAO[nome]

    print(f"task = {task}   |   robô TRAVADO   |   sem política (agent=zero)")
    print(f"nível = {Knobs().nivel.forcado if Knobs().nivel.forcado is not None else 0}"
          f"   (mude `Nivel.forcado` em knobs.py para outro)")
    print()
    print("o que está desenhado:")
    print("  eixos da caixa       X vermelho, Y verde, Z azul")
    print("  seta MAGENTA         a face pedida, em mundo")
    print("  esfera CIANA         o alvo deste elo")
    print("  seta AMARELA         caixa -> alvo (o quanto mover, e o dz)")
    print("  laje CINZA           o topo da prateleira")
    print("  esfera BRANCA        o alcance de referência da pelve (0,50 m)")
    if nome == "andar":
        print("  seta VERDE           o twist comandado — no `andar` o alvo é uma "
              "VELOCIDADE, e não")
        print("                       existe alvo de caixa (valida = 0)")
    if nome == "reorientar":
        print("  ⚠ no `reorientar` a esfera do alvo fica SOBRE a caixa: o que se pede "
              "é a ATITUDE")
    if nome == "carregar":
        print("  ⚠ no `carregar` o alvo é ancorado na BASE: ele anda com o robô")
    if nome == "botar":
        print("  ⚠ no `botar` a laje foi para um topo NOVO, travado no fundo da caixa "
              "menos a folga")
    run_play(task, PlayConfig(agent="zero", no_terminations=True,
                              num_envs=args.envs, device=args.device))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Revisa a cena e os alvos do g1_limpo antes de treinar.")
    p.add_argument("elo", nargs="?", default=None,
                   help=f"um de {CMD.ELOS}. Sem elo, a tabela mostra os cinco.")
    p.add_argument("--tabela", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--nivel", type=int, default=None)
    p.add_argument("--envs", type=int, default=1, help="envs no viewer")
    p.add_argument("--envs-tabela", type=int, default=N_ENVS)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    if not args.tabela and not args.viewer:
        args.tabela = True

    k = Knobs()
    if args.nivel is not None and not (0 <= args.nivel < k.nivel.n_niveis):
        print(f"nível fora da faixa 0..{k.nivel.n_niveis - 1}")
        return 2

    rc = 0
    if args.tabela:
        rc |= tabela(args)
    if args.viewer:
        rc |= viewer(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
