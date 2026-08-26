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


def _ambiente(nivel: int | None, elo: int, *, n_envs: int, device: str,
              cadeia_forcada: int | None = None):
    """Constrói o MESMO cfg do treino, com nível e elo forçados. Nada de mock.

    `cadeia_forcada`: índice de cadeia (0-3) para forçar. Requer que a máquina de elo (F4)
    esteja implementada. Ignorado se não existir.
    """
    from mjlab.envs import ManagerBasedRlEnv

    from g1_limpo.env_cfg import make_env_cfg

    k = Knobs()
    k.nivel.forcado = nivel
    cfg = make_env_cfg(k, inspecao=True, elo=elo)

    # ⚠ NOVO: passa cadeia_forcada se foi pedida
    if cadeia_forcada is not None and hasattr(cfg.commands["alvo_caixa"], 'cadeia_forcada'):
        cfg.commands["alvo_caixa"].cadeia_forcada = cadeia_forcada

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


def _nomes_de_cadeia() -> list[str]:
    """Os nomes das cadeias, DERIVADOS de `CMD.CADEIAS`.

    ⚠ Duas listas digitadas à mão viviam aqui. Elas saem de sincronia no dia em que uma
    cadeia mudar, e a prova do `--cadeia` passaria a imprimir o nome errado enquanto o
    índice estava certo — o pior tipo de mentira num diagnóstico.
    """
    return [f"({', '.join(CMD.ELOS[e].upper() for e in cad)})"
            for cad in CMD.CADEIAS]


def _cadeia_forcada_prova(env, cadeia_id: int | None) -> None:
    """Prova que --cadeia funciona de verdade: lê e imprime qual cadeia saiu.

    O plano avisa: 'um `--cadeia` no-op já passou desapercebido neste repo.'
    Portanto este relato é obrigatório.
    """
    if cadeia_id is None:
        return

    cmd_term = env.command_manager.get_term("alvo_caixa")

    # Tenta ler o buffer de cadeia que a máquina de elo publica
    if hasattr(cmd_term, "_cadeia"):
        # ⚠ Ler `env 0` às cegas foi um defeito: com a fatia de locomoção em 95%, o env
        # 0 é quase sempre de `ANDAR`, e ali `_cadeia` vale `CADEIA_NENHUMA`. E aí
        # `nomes[-1]` mostrava a ÚLTIMA cadeia — índice negativo lido como nome, que é
        # o pior tipo de mentira num diagnóstico. Lê-se um env que TENHA cadeia.
        _com_cadeia = (cmd_term._cadeia >= 0).nonzero().flatten()
        if len(_com_cadeia) == 0:
            print()
            print("=" * 118)
            print(f"PROVA DO --cadeia: forçado={cadeia_id}, e NENHUM env recebeu "
                  f"cadeia — o `--cadeia` NÃO chegou ao termo de comando")
            print("=" * 118)
            return
        cadeia_lida = int(cmd_term._cadeia[int(_com_cadeia[0])])
        cadeia_nomes = _nomes_de_cadeia()
        print()
        print("=" * 118)
        print(f"PROVA DO --cadeia: forçado={cadeia_id}, lido do env={cadeia_lida}")
        if cadeia_id < len(cadeia_nomes):
            print(f"  forçado: {cadeia_nomes[cadeia_id]}")
        if cadeia_lida < len(cadeia_nomes):
            print(f"  lido:    {cadeia_nomes[cadeia_lida]}")
        if cadeia_id == cadeia_lida:
            print("  ✓ MATCH — --cadeia funcionou de verdade")
        else:
            print(f"  ✗ MISMATCH — --cadeia não está funcionando")
        print("=" * 118)
        print()
    else:
        print()
        print("⚠ A máquina de elo ainda não foi implementada (falta _cadeia no cmd_term)")
        print()


def tabela(args) -> int:
    fx = lambda t: f"{float(t.min()):.3f}–{float(t.max()):.3f}"        # noqa: E731

    # ⚠ NOVO: Se --cadeia foi passado, a tabela cobre o PÓS-AVANÇO também.
    # Não é um erro para cadeias de 1 elo; é um no-op (não avança).
    fazer_pos_avanco = args.cadeia is not None
    cadeias_ids = [0, 1, 2, 3]  # As 4 cadeias

    # Parse --cadeia
    cadeia_forcada_id = None
    if args.cadeia is not None:
        if args.cadeia.isdigit():
            cadeia_forcada_id = int(args.cadeia)
            if not (0 <= cadeia_forcada_id < len(cadeias_ids)):
                print(f"cadeia {args.cadeia} fora da faixa 0..{len(cadeias_ids)-1}")
                return 2
        else:
            # ⚠ Os nomes são DERIVADOS de `CMD.CADEIAS`, e não digitados. Uma lista
            # paralela escrita à mão sai de sincronia no dia em que uma cadeia mudar,
            # e aí `--cadeia pegar_botar` selecionaria outra coisa em silêncio.
            nomes_cadeias = ["_".join(CMD.ELOS[e] for e in cad)
                             for cad in CMD.CADEIAS]
            if args.cadeia.lower().replace("-", "_") in nomes_cadeias:
                cadeia_forcada_id = nomes_cadeias.index(args.cadeia.lower().replace("-", "_"))
            else:
                print(f"cadeia {args.cadeia!r} desconhecida. Use 0-3 ou um de {nomes_cadeias}")
                return 2

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
    primeiro_env = True  # para só provar cadeia uma vez
    for elo, niv in casos:
        # ⚠ A TABELA POR ELO **NÃO** RECEBE `cadeia_forcada`, e isso é decisão. Forçar
        # a cadeia 2 (`PEGAR -> CARREGAR`) e forçar o elo `ANDAR` são pedidos
        # CONTRADITÓRIOS: `ANDAR` não pertence a cadeia nenhuma. Passar os dois fazia a
        # cadeia vencer, o elo publicado sair diferente do pedido, e cinco checagens
        # acusarem o código por um conflito de intenção meu.
        #
        # O `--cadeia` governa SÓ a seção de PÓS-AVANÇO, logo abaixo.
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

        # ⚠ A PROVA DO `--cadeia` NÃO PODE RODAR AQUI. A tabela por elo é montada
        # SEM `cadeia_forcada` de propósito (forçar elo e cadeia são pedidos
        # contraditórios), portanto `_cadeia` vale `CADEIA_NENHUMA` nestes envs e a
        # prova daria MISMATCH sempre — acusando o `--cadeia` de não funcionar
        # exatamente onde ele não foi pedido. Ela roda na seção de PÓS-AVANÇO.
        _ = primeiro_env

        del env

    # ⚠ NOVO: Pós-avanço. Se --cadeia foi passado, roda para cada cadeia de 2+ elos.
    if fazer_pos_avanco:
        print()
        print("=" * 118)
        print("PÓS-AVANÇO — 2º ELO de cada cadeia, mesmos níveis")
        print("=" * 118)

        for cadeia_id in cadeias_ids[1:]:  # Só as cadeias de 2+ elos (índices 1-3)
            niv = args.nivel if args.nivel is not None else 0

            try:
                # A cadeia foi forçada em _ambiente, acima
                env, k = _ambiente(niv, CMD.PEGAR, n_envs=args.envs_tabela,
                                   device=args.device, cadeia_forcada=cadeia_id)

                # Tenta avançar. Se não existir forca_avanco, ignora (F4 ainda não
                # foi implementada)
                cmd_term = env.command_manager.get_term("alvo_caixa")
                ids = torch.arange(env.num_envs, device=env.device)
                if hasattr(cmd_term, "forca_avanco"):
                    cmd_term.forca_avanco(ids)
                    # ⚠⚠ UM PASSO, E COM A CAIXA PINADA. Sem isto a leitura é do
                    # buffer VELHO: o `_laje_para` chama `write_mocap_pose_to_sim`, e
                    # os buffers de `.data` só são recomputados no forward seguinte.
                    # Era o que fazia esta checagem acusar "a laje não subiu" com o
                    # topo ANTIGO (0,566, a prateleira de antes do avanço) e "a laje
                    # nasceu dentro da caixa" — as duas eram a mesma leitura obsoleta,
                    # e é a mesma armadilha que o `_pendente` do comando existe para
                    # consertar, agora no lado da LEITURA.
                    #
                    # E a caixa é re-pinada nesse passo porque no pós-avanço nada a
                    # segura: ela cairia, e o `fundo` medido seria de um instante
                    # diferente do `topo`.
                    caixa = env.scene["box"]
                    pose = torch.cat([caixa.data.root_link_pos_w,
                                      caixa.data.root_link_quat_w], dim=-1).clone()
                    caixa.write_root_link_pose_to_sim(pose)
                    caixa.write_root_link_velocity_to_sim(
                        torch.zeros(env.num_envs, 6, device=env.device))
                    env.step(torch.zeros(
                        env.num_envs, env.action_manager.total_action_dim,
                        device=env.device))

                if cadeia_id == cadeias_ids[1]:
                    _cadeia_forcada_prova(env, cadeia_id)

                m = _medidas(env)

                # Identifica qual é o 2º elo desta cadeia (após avanço)
                elo_depois_i = int(m["elo"][0])
                elo_depois_nome = CMD.ELOS[elo_depois_i]
                cadeia_nomes = _nomes_de_cadeia()
                cadeia_nome = cadeia_nomes[cadeia_id] if cadeia_id < len(cadeia_nomes) else f"cadeia {cadeia_id}"
                print(f"\ncadeia {cadeia_id:>1} {cadeia_nome:>25} no nível {niv}:")

                # Checagem crítica: no CARREGAR a laje tem de estar em afasta_z
                # ⚠ TOLERÂNCIA DERIVADA: A laje é uma mocap, e é escrita de forma
                # síncrona no resample. Mas então o forward roda e aplica física por
                # 1 passo antes da leitura: a laje "cai" `½·g·dt² = 2 mm` antes de
                # ser re-pinada. Tolerância de 5 mm cobre esse desvio de gravidade.
                if elo_depois_i == CMD.CARREGAR:
                    topo = float(m["topo_laje"][0] + k.cena.prateleira_meia_z)
                    esperado = k.cena.afasta_z
                    diferenca = abs(topo - esperado)
                    if diferenca > 5e-3:  # 5 mm: derivado de gravidade
                        msg = (f"✗ no CARREGAR a laje NÃO está em {esperado:.3f}: "
                               f"está em {topo:.3f} (diff {diferenca:.4f})")
                        print(f"  {msg}")
                        total += 1
                    else:
                        print(f"  ✓ CARREGAR: laje em {topo:.3f} m (esperado {esperado:.3f})")

                # Checagem crítica: no BOTAR o topo não pode estar acima do fundo-folga
                # ⚠ TOLERÂNCIA DERIVADA: mesma justificativa que CARREGAR.
                if elo_depois_i == CMD.BOTAR:
                    topo = float(m["topo_laje"][0] + k.cena.prateleira_meia_z)
                    caixa_z = float(m["caixa"][0, 2])
                    # meia-aresta é scalar ou tuple?
                    meia_z = k.cena.caixa_meia_aresta[2] if isinstance(k.cena.caixa_meia_aresta, (tuple, list)) else k.cena.caixa_meia_aresta
                    fundo = caixa_z - meia_z
                    teto = fundo - k.alvo.botar_folga_laje
                    if topo > teto + 5e-3:  # 5 mm: derivado de gravidade
                        msg = (f"✗ no BOTAR laje nasceu DENTRO da caixa: topo "
                               f"{topo:.3f} > fundo−folga {teto:.3f}")
                        print(f"  {msg}")
                        total += 1
                    else:
                        print(f"  ✓ BOTAR: laje em {topo:.3f} m, fundo−folga "
                              f"{teto:.3f} m")

                del env

            except Exception as e:
                print(f"  ⚠ cadeia {cadeia_id}: não conseguiu testar (F4 ainda não pronta?)")
                print(f"     erro: {e}")

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

    # ⚠ NOVO: --cadeia
    if args.cadeia is not None:
        print(f"cadeia FORÇADA = {args.cadeia}   (será reportada depois que o env montar)")
    if args.avanca_elo:
        print(f"--avanca-elo: disparará o avanço manual com relatório ANTES/DEPOIS")

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

    # ⚠ NOVO: Passa cadeia_forcada ao cfg se --cadeia foi usado
    if args.cadeia is not None:
        try:
            # Tenta interpretar como índice ou nome de cadeia
            if args.cadeia.isdigit():
                cadeia_id = int(args.cadeia)
            else:
                # Tenta como nome: ex "PEGAR_CARREGAR"
                # A ser implementado com getattr na F4
                cadeia_id = int(args.cadeia)  # fallback: só suporta índice por enquanto
        except (ValueError, AttributeError):
            print(f"✗ cadeia {args.cadeia!r} não entendida (use 0-3 ou nome)")
            return 2

        # Modifica o cfg antes de run_play
        # O cfg está "dentro" do task. Precisa ser modificado ANTES de rodar.
        # Infelizmente run_play não expõe o cfg facilmente. Deixa o fallback:
        # se a máquina de elo existir, ela vai ler o cadeia_forcada;
        # se não, será ignorado.
        # Por hora, aviso que não é possível forçar no viewer sem reescrever run_play.
        print(f"⚠ --cadeia no viewer: a forçagem será feita apenas se a máquina de elo "
              f"(F4) já estiver implementada")

    # ⚠ NOVO: o --avanca-elo será rodado DEPOIS de run_play, não durante
    # (ou seria durante, numa callback do viewer — mas run_play é bloqueante)

    run_play(task, PlayConfig(agent="zero", no_terminations=True,
                              num_envs=args.envs, device=args.device))
    return 0


def _avanca_elo_manual(env, cmd_term, ids: torch.Tensor | None = None) -> None:
    """Avança o elo manualmente com relatório ANTES/DEPOIS.

    Prova que o avanço funciona sem reset e sem resample, e mostra as grandezas
    que mudam. Requer que `cmd_term` tenha os métodos da F4:
    - `elo_de(ids)` — elo corrente
    - `n_elos_da_cadeia(ids)` — quantos elos a cadeia tem
    - `forca_avanco(ids)` — dispara o avanço
    """
    if ids is None:
        ids = torch.arange(env.num_envs, device=env.device)

    # Verifica se os métodos existem
    if not hasattr(cmd_term, 'forca_avanco'):
        print("⚠ A máquina de elo ainda não foi implementada (falta F4).")
        print("  O --avanca-elo só funciona com a máquina de elo presente.")
        return

    # Estado ANTES do avanço
    elo_antes = cmd_term.elo_de(ids).clone()
    n_elos = cmd_term.n_elos_da_cadeia(ids)

    # Tenta ler os σ (pode não existir ainda)
    sigma_alcance_antes = getattr(cmd_term, 'sigma_alcance',
                                  torch.full_like(elo_antes, float('nan'), dtype=torch.float32))[ids].clone()
    sigma_trazer_antes = getattr(cmd_term, 'sigma_trazer',
                                 torch.full_like(elo_antes, float('nan'), dtype=torch.float32))[ids].clone()
    sigma_ori_antes = getattr(cmd_term, 'sigma_ori',
                              torch.full_like(elo_antes, float('nan'), dtype=torch.float32))[ids].clone()

    # Tenta ler o alvo e topo (pode não existir ainda, ou ser relativo ao env)
    cmd = cmd_term._command[ids].clone()
    alvo_antes = cmd[:, CMD.ALVO].clone() if CMD.ALVO != slice(0, 3) else cmd[:, 0:3].clone()
    topo_antes = None
    if hasattr(env, 'limpo_topo'):
        topo_antes = env.limpo_topo[ids].clone()

    # Dispara o avanço
    cmd_term.forca_avanco(ids)

    # Estado DEPOIS do avanço
    elo_depois = cmd_term.elo_de(ids).clone()
    sigma_alcance_depois = getattr(cmd_term, 'sigma_alcance',
                                   torch.full_like(elo_depois, float('nan'), dtype=torch.float32))[ids].clone()
    sigma_trazer_depois = getattr(cmd_term, 'sigma_trazer',
                                  torch.full_like(elo_depois, float('nan'), dtype=torch.float32))[ids].clone()
    sigma_ori_depois = getattr(cmd_term, 'sigma_ori',
                               torch.full_like(elo_depois, float('nan'), dtype=torch.float32))[ids].clone()

    cmd = cmd_term._command[ids].clone()
    alvo_depois = cmd[:, CMD.ALVO].clone() if CMD.ALVO != slice(0, 3) else cmd[:, 0:3].clone()
    topo_depois = None
    if hasattr(env, 'limpo_topo'):
        topo_depois = env.limpo_topo[ids].clone()

    # Reporta
    print()
    print("=" * 118)
    print("AVANÇO DE ELO — ANTES e DEPOIS (robô travado)")
    print("=" * 118)

    for i, env_id in enumerate(ids.cpu().numpy().astype(int)):
        elo_a = int(elo_antes[i])
        elo_d = int(elo_depois[i])
        nome_a = CMD.ELOS[elo_a]
        nome_d = CMD.ELOS[elo_d]
        n_e = int(n_elos[i])

        print(f"\nenv {env_id} (cadeia tem {n_e} elo{'s' if n_e != 1 else ''}):")
        print(f"  ELO:           {nome_a:>12} → {nome_d:>12}")

        # Alvo
        a_b = alvo_antes[i].cpu().numpy()
        a_d = alvo_depois[i].cpu().numpy()
        print(f"  ALVO [x, y, z]: [{a_b[0]:+.4f}, {a_b[1]:+.4f}, {a_b[2]:+.4f}] "
              f"→ [{a_d[0]:+.4f}, {a_d[1]:+.4f}, {a_d[2]:+.4f}]")

        # Topo (se existir)
        if topo_antes is not None and topo_depois is not None:
            t_b = float(topo_antes[i])
            t_d = float(topo_depois[i])
            print(f"  TOPO LAJE:     {t_b:+.4f} m → {t_d:+.4f} m")

        # Sigmas (se existirem e não forem NaN)
        sa_b = float(sigma_alcance_antes[i])
        st_b = float(sigma_trazer_antes[i])
        so_b = float(sigma_ori_antes[i])
        sa_d = float(sigma_alcance_depois[i])
        st_d = float(sigma_trazer_depois[i])
        so_d = float(sigma_ori_depois[i])

        if not (torch.isnan(sigma_alcance_antes[i]) or torch.isnan(sigma_alcance_depois[i])):
            print(f"  σ_alcance:     {sa_b:.4f} → {sa_d:.4f} m")
            print(f"  σ_trazer:      {st_b:.4f} → {st_d:.4f} m")
            print(f"  σ_ori:         {so_b:.4f} → {so_d:.4f} rad")

    print()
    print("=" * 118)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Revisa a cena e os alvos do g1_limpo antes de treinar.")
    p.add_argument("elo", nargs="?", default=None,
                   help=f"um de {CMD.ELOS}. Sem elo, a tabela mostra os cinco.")
    p.add_argument("--tabela", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--cadeia", type=str, default=None,
                   help="força uma cadeia: índice (0-3) ou nome (ex: PEGAR_CARREGAR)")
    p.add_argument("--avanca-elo", action="store_true",
                   help="dispara o avanço de elo manualmente com relatório ANTES/DEPOIS")
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
