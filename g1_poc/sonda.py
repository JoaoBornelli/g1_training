"""Sonda do elo `pegar`: mede FORÇA em Newtons, e não recompensa.

    python -m g1_poc.sonda --checkpoint CAMINHO/model_1682.pt

Ela existe para uma pergunta só. Quando o robô encosta as palmas na caixa e ela não
sobe, há DUAS causas possíveis, e elas pedem correções opostas:

    a) ele não aperta o bastante  -> a força normal fica longe de `F_ref`
    b) ele aperta e não tenta subir -> a força chega em `F_ref` e a caixa fica

Três números separam:

    F_n por palma  vs  F_ref = m·g/(2μ)   ele aperta o bastante?
    apoio_z        vs  m·g                ele chega a DESCARREGAR a caixa?
    subida da caixa                        ela sai da prateleira?

O `apoio_z` é a grandeza-ponte: ela cai de `m·g` a zero ANTES de a caixa se mover,
portanto ela mostra progresso onde a altura ainda é um degrau. Foi a lição do platô do
grasp no g1_multitask.

⚠ Roda 1 env na CPU. É leve — MLP no laço, sem BFM.
"""
from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import asdict

import torch

from mjlab.utils.lab_api.math import quat_apply

import g1_poc  # noqa: F401  (registra a task)
from g1_poc import cena as C
from g1_poc.play import TASK_MANIPULA, _ajusta_manipula, _registra


def _forcas_normais(env, asset_cfg) -> tuple[float, float]:
    """(F_n esquerda, F_n direita) em N, projetadas na normal de cada palma.

    Mesma matemática do `recompensas.squeeze`: a normal vem da ORIENTAÇÃO DO SITE, e
    não do campo `normal` do sensor — com `reduce="netforce"` aquele campo perde
    significado.
    """
    robot = env.scene[asset_cfg.name]
    quat = robot.data.site_quat_w[:, asset_cfg.site_ids]            # [B,2,4]
    locais = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=quat.device, dtype=quat.dtype)
    normais = quat_apply(quat, locais.expand(quat.shape[0], 2, 3))  # [B,2,3]

    saida = []
    for i, nome in enumerate(C.SENSOR_PALMA):
        f = env.scene[nome].data.force
        assert f is not None
        fn = torch.sum(f * normais[:, i].unsqueeze(1), dim=-1).abs().sum(dim=-1)
        saida.append(float(fn[0]))
    return saida[0], saida[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--passos", type=int, default=600)
    p.add_argument("--cada", type=int, default=50)
    p.add_argument("--com-jitter", action="store_true",
                   help="usa o jitter do treino; o default é o caso NOMINAL")
    p.add_argument("--envs", type=int, default=8,
                   help="as forças saem do env 0; o fecho é agregado sobre todos")
    p.add_argument("--com-push", action="store_true",
                   help="devolve o `push_robot` (1-3 s), que o play remove")
    p.add_argument("--com-ruido", action="store_true",
                   help="devolve o `Unoise` da observação, que o play desliga")
    p.add_argument("--nivel", type=int, default=None,
                   help="força a célula do nível (§10.1); default = promoção por sucesso")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    if args.nivel is not None:
        if not (0 <= args.nivel <= 6):
            raise SystemExit("--nivel: 0 <= valor <= 6")

    ckpt = pathlib.Path(args.checkpoint).expanduser()
    if not ckpt.is_file():
        raise SystemExit(f"não achei {ckpt}")

    sem_jitter = not args.com_jitter
    sufixo = "-Nominal" if sem_jitter else ""
    if args.nivel is not None:
        sufixo += f"-N{args.nivel}"
    task_id = _registra(TASK_MANIPULA + sufixo,
                        _ajusta_manipula(sem_jitter), nivel=args.nivel)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(task_id)

    # ⚠ O `play=True` remove o `push_robot` e desliga o `Unoise`. Isso torna a sonda
    # OTIMISTA em relação ao treino, e a diferença não é pequena: o fecho exige 50
    # passos seguidos e o push chega a cada 50-150. Estas flags devolvem cada um, uma
    # por vez, para atribuir a queda do `episode_success` a uma causa e não a um palpite.
    if args.com_push:
        from g1_poc.env_cfg import make_g1_poc_env_cfg
        from g1_poc.knobs import Knobs
        treino = make_g1_poc_env_cfg(Knobs())         # o cfg de TREINO tem o push
        env_cfg.events["push_robot"] = treino.events["push_robot"]
        print("[SONDA] push_robot DEVOLVIDO:",
              env_cfg.events["push_robot"].interval_range_s, "s")
    if args.com_ruido:
        env_cfg.observations["actor"].enable_corruption = True
        print("[SONDA] Unoise da observação DEVOLVIDO")

    base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    # os params RESOLVIDOS: no cfg o `SceneEntityCfg` não tem `site_ids`
    palmas = base.reward_manager.get_term_cfg("squeeze").params["asset_cfg"]
    mu = base.reward_manager.get_term_cfg("squeeze").params["mu"]
    cmd = base.command_manager.get_term("caixa_alvo")

    obs, _ = env.reset()
    massa = float(base.poc_massa[0])
    f_ref = massa * 9.81 / (2.0 * mu)
    peso = massa * 9.81
    caixa_z0 = float(base.scene["box"].data.root_link_pos_w[0, 2])

    nivel_str = f"   nível = {args.nivel}" if args.nivel is not None else ""
    print(f"\nmassa da caixa = {massa:.2f} kg   peso = {peso:.2f} N{nivel_str}")
    print(f"F_ref = m·g/(2μ) = {f_ref:.2f} N   (μ = {mu})")
    print(f"caixa nasce em z = {caixa_z0:.3f} m   alvo z = "
          f"{float(cmd.command[0, 2]):.3f} m\n")
    print(f"{'passo':>6} {'F_esq':>7} {'F_dir':>7} {'F_min/F_ref':>12} "
          f"{'apoio_z':>8} {'apoio/peso':>11} {'subida':>8} {'erro':>7}")
    print("-" * 74)

    pico_f = 0.0
    pico_subida = 0.0
    # ⚠ NÃO filtrar o apoio zero, e não usar o mínimo. As duas coisas já mentiram aqui:
    #   - o MÍNIMO global pega o passo de reset (0 N por falta de contato) e diz
    #     "descarregou" quando nunca descarregou;
    #   - filtrar `apoio > 0` descarta justamente os passos com a caixa NO AR, que são
    #     os bons, e a mediana sobra sobre os poucos passos de contato. Medido: a tabela
    #     mostrava 0% em todo passo e o resumo dizia 800% do peso.
    # A mediana sobre TODOS os passos descreve o estado sustentado, e a fração de passos
    # descarregados é o que diz se ele ergueu.
    apoios: list[float] = []
    pico_acao = 0.0

    # --- o fecho do elo, decomposto ---
    # O treino só dá MÉDIAS por iteração. Aqui dá para ver o instante em que o
    # `sustenta` zera e atribuir a QUAL condição caiu naquele passo — que é a pergunta
    # de verdade quando `no_alvo` é 57% e `episode_success` é 0,006.
    robot = base.scene["robot"]
    n = {"perto": 0, "alinhado": 0, "alto": 0, "reto": 0, "todas": 0}
    causa = {"posicao": 0, "angulo": 0, "pelve_baixa": 0, "inclinado": 0}
    pico_sustenta = 0.0
    fechos = 0
    sust_ant = torch.zeros(args.envs, device=args.device)
    pico_por_env = torch.zeros(args.envs, device=args.device)
    pelves: list[float] = []

    for i in range(1, args.passos + 1):
        with torch.inference_mode():
            acao = policy(obs)
        obs, _, dones, _ = env.step(acao)
        if int(dones[0]) == 1:      # reset: as referências mudam
            caixa_z0 = float(base.scene["box"].data.root_link_pos_w[0, 2])

        fe, fd = _forcas_normais(base, palmas)
        f_min = min(fe, fd)
        apoio = base.scene[C.SENSOR_APOIO].data.force
        apoio_z = float(apoio[..., 2].abs().sum()) if apoio is not None else float("nan")
        subida = float(base.scene["box"].data.root_link_pos_w[0, 2]) - caixa_z0
        erro = float(cmd.erro_pos()[0])

        pico_f = max(pico_f, f_min)
        pico_subida = max(pico_subida, subida)
        pico_acao = max(pico_acao, float(acao.abs().max()))
        apoios.append(apoio_z)

        # --- o fecho, condição por condição ---
        perto = cmd.erro_pos() < cmd.cfg.raio_sucesso
        alinhado = cmd.erro_ang() < cmd.cfg.angulo_sucesso_rad
        z_pelve = robot.data.root_link_pos_w[:, 2]
        g = robot.data.projected_gravity_b
        incl = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
        alto = z_pelve >= cmd.cfg.pelve_min
        reto = incl <= cmd.cfg.inclinacao_max_rad
        todas = perto & alinhado & alto & reto
        for k, t in (("perto", perto), ("alinhado", alinhado), ("alto", alto),
                     ("reto", reto), ("todas", todas)):
            n[k] += int(t.sum())
        pelves.append(float(z_pelve.mean()))
        pico_sustenta = max(pico_sustenta, float(cmd._sustenta.max()))

        # ⚠ Uma quebra é `sustenta > 0` virando 0 SEM reset. O `_resample_command`
        # zera o `_sustenta` no reset, e contar isso daria quebra fantasma.
        sust = cmd._sustenta.clone()
        quebrou = (sust_ant > 0) & (sust == 0) & (dones == 0)
        for idx in quebrou.nonzero().flatten().tolist():
            if not bool(perto[idx]):
                causa["posicao"] += 1
            elif not bool(alinhado[idx]):
                causa["angulo"] += 1
            elif not bool(alto[idx]):
                causa["pelve_baixa"] += 1
            elif not bool(reto[idx]):
                causa["inclinado"] += 1
        fechos += int(((sust_ant == 0) & (sust > 0)).sum())
        sust_ant = sust
        pico_por_env = torch.maximum(pico_por_env, sust)

        if i % args.cada == 0:
            print(f"{i:>6} {fe:>7.2f} {fd:>7.2f} {f_min/f_ref:>11.0%} "
                  f"{apoio_z:>8.2f} {apoio_z/peso:>10.0%} "
                  f"{subida*100:>7.1f}cm {erro:>7.3f}")

    apoios.sort()
    apoio_med = apoios[len(apoios) // 2] if apoios else float("nan")
    sustenta_por_env = [round(float(x), 2) for x in pico_por_env]
    tanh_sat = float(torch.tanh(torch.tensor(pico_f / f_ref)))

    print("-" * 74)
    print(f"\n== resumo de {args.passos} passos ==")
    print(f"  aperto MÁXIMO (o mín das duas palmas) : {pico_f:.2f} N = "
          f"{pico_f/f_ref:.0%} de F_ref")
    descarregado = sum(1 for a in apoios if a < 0.10 * peso) / max(len(apoios), 1)
    print(f"  apoio MEDIANO da prateleira           : {apoio_med:.2f} N = "
          f"{apoio_med/peso:.0%} do peso")
    print(f"  passos com a caixa DESCARREGADA       : {descarregado:.1%} "
          f"(apoio < 10% do peso)")
    print(f"  subida MÁXIMA da caixa                : {pico_subida*100:.1f} cm")
    print(f"  |ação| máximo                         : {pico_acao:.2f}")
    print(f"  `squeeze` no pico = tanh({pico_f/f_ref:.2f})       : {tanh_sat:.5f}   "
          f"(derivada {1-tanh_sat**2:.1e})")
    print("\n  leitura:")
    # ⚠ A subida vem ANTES da prensa. Se a caixa está no ar, o apoio alto de alguns
    # passos é o contato inicial, e não escora — foi assim que a versão anterior
    # acusou "ele prensa" com a caixa 16 cm acima do repouso.
    if pico_subida > 0.05 and descarregado > 0.5:
        print(f"   - ELE ERGUE a caixa ({pico_subida*100:.1f} cm) e a mantém "
              f"descarregada em {descarregado:.0%} dos passos.")
        print("     O elo de FORÇA está resolvido. O que sobra é o fecho, abaixo.")
    elif apoio_med > 1.05 * peso:
        print("   - ELE PRENSA A CAIXA CONTRA A PRATELEIRA. O apoio está ACIMA do peso,")
        print("     portanto a resultante vertical que ele aplica aponta para BAIXO. Não")
        print("     é falta de aperto nem escorregamento: é o hack de usar a caixa como")
        print("     escora. A projeção na normal do pad impede que isso seja PAGO como")
        print("     aperto, mas não impede que aconteça de graça.")
        print("     Falta o termo que paga por DESCARREGAR: `unload = 1 − F_apoio/m·g`.")
    elif pico_f < 0.5 * f_ref:
        print("   - APERTO INSUFICIENTE. A força normal não chega perto de F_ref,")
        print("     portanto o atrito nunca poderia vencer o peso. O gargalo é o")
        print("     incentivo do `squeeze` ou a escala de ação (veja |ação| máximo).")
    elif apoio_med > 0.5 * peso:
        print("   - APERTA E NÃO SOBE. A força chega e a prateleira ainda carrega a")
        print("     caixa: falta o comando de LEVANTAR, não o de apertar.")
    elif pico_subida < 0.02:
        print("   - ESCORREGA. O apoio cai, logo ele descarrega a caixa, mas ela não")
        print("     sobe: o atrito não sustenta. Suspeite do μ e da DR de atrito.")
    else:
        print(f"   - ELE ERGUE {pico_subida*100:.1f} cm. O elo funciona; o que falta é")
        print("     chegar ao alvo (erro < 5 cm).")
    if tanh_sat > 0.99:
        print(f"   - E o `squeeze` está SATURADO ({tanh_sat:.5f}): derivada "
              f"{1-tanh_sat**2:.1e}. Ele não guia mais nada — é constante paga.")

    # ------------------------------------------------------------------ o fecho
    total = args.passos * args.envs
    print(f"\n== o fecho do `pegar`, condição por condição ==")
    print(f"  (sucesso exige as QUATRO juntas por {cmd.cfg.sustenta_pegar_s:.1f} s "
          f"= {int(cmd.cfg.sustenta_pegar_s / base.step_dt)} passos seguidos)")
    for k, rotulo in (("perto", f"erro_pos < {cmd.cfg.raio_sucesso:.2f} m"),
                      ("alinhado", f"erro_ang < {math.degrees(cmd.cfg.angulo_sucesso_rad):.0f}°"),
                      ("alto", f"pelve >= {cmd.cfg.pelve_min:.2f} m"),
                      ("reto", f"inclin <= {math.degrees(cmd.cfg.inclinacao_max_rad):.0f}°"),
                      ("todas", "AS QUATRO JUNTAS")):
        print(f"  {rotulo:24s} {n[k]/total:6.1%}")
    print(f"\n  pelve média              {sum(pelves)/len(pelves):.3f} m   "
          f"(mínimo exigido {cmd.cfg.pelve_min:.2f})")
    # ⚠ O `episode_success` do PRÓPRIO comando é o número comparável com o treino, e
    # é ele que o currículo lê para promover o nível. Os agregados abaixo já mentiram
    # três vezes nesta sonda (mínimo, filtro do zero, e o máximo): um env sortudo entre
    # oito leva a "ele fecha" quando sete nunca fecharam. Fração, não extremo.
    print(f"  episode_success          {float(cmd.episode_success.mean()):.3f}   "
          f"<- comparável com o log do treino")
    print(f"  envs que JÁ fecharam     {float((cmd.episode_success > 0.5).float().mean()):.1%} "
          f"de {args.envs}")
    print(f"  sustentação por env      {sustenta_por_env}")
    print(f"  sustentação MÁXIMA       {pico_sustenta:.2f} s de "
          f"{cmd.cfg.sustenta_pegar_s:.1f} exigidos  (de UM env, não da média)")
    print(f"  fechos iniciados         {fechos}")
    if sum(causa.values()):
        print("  quando o cronômetro zerou, quem caiu:")
        for k, c in sorted(causa.items(), key=lambda kv: -kv[1]):
            if c:
                print(f"    {k:14s} {c:4d}  ({c/sum(causa.values()):.0%})")

    frac_fechou = float((cmd.episode_success > 0.5).float().mean())
    print("\n  leitura:")
    if n["todas"] == 0:
        pior = min(("perto", "alinhado", "alto", "reto"), key=lambda k: n[k])
        print(f"   - as quatro NUNCA coincidem. A mais rara é `{pior}` "
              f"({n[pior]/total:.1%}); é ela que bloqueia.")
    elif frac_fechou >= 0.5:
        print(f"   - {frac_fechou:.0%} dos envs FECHARAM o elo. Se o treino loga "
              f"`episode_success` muito abaixo disto, a diferença NÃO está na")
        print("     política — está no que o play remove (push, ruído) ou na forma.")
    elif frac_fechou > 0.0:
        print(f"   - só {frac_fechou:.0%} dos envs fecharam, e o máximo de "
              f"{pico_sustenta:.2f} s vem de UM env.")
        print("     Os outros nunca sustentaram 1 s. É frequência, não capacidade.")
    elif causa:
        dom = max(causa, key=causa.get)
        print(f"   - ele fecha e PERDE. O cronômetro chegou a {pico_sustenta:.2f} s de "
              f"{cmd.cfg.sustenta_pegar_s:.1f}, e quem")
        print(f"     mais derruba é `{dom}`. É problema de ESTABILIDADE, não de alcançar")
        print("     a condição — mexer no alvo ou na tolerância não resolveria.")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
