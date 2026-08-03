"""O residual VENCE o BFM mandando ficar de pé? Malha aberta, CPU, segundos.

    python g1_residual/autoridade.py
    python g1_residual/autoridade.py --alvo move-ego-low0.5-0-0
    python g1_residual/autoridade.py --passos 400

**A pergunta.** Com `z = move-ego-0-0` o BFM manda "fique em pé e quieto" a cada
passo, em malha fechada: se o robô sai da pose, ele corrige de volta. O `ΔA` soma até
±0,35 rad (20°) por junta de perna e cintura em cima disso. **Vinte graus bastam para
descer, ou o BFM ganha?**

**Por que sem PPO.** O PPO levaria horas para APRENDER a empurrar para baixo. A
pergunta aqui não é se ele aprende, é se DÁ. Então o residual vai para o limite
direto, sem política. Se não dá, nenhum treino resolve e o clamp tem que crescer. Se
dá, aí é questão de aprendizado e vai para a GPU.

**Por que a direção vem do próprio BFM, e não de um sinal por junta.** Agachar não é
"todas as juntas de perna para o mesmo lado" — o quadril flexiona num sentido, o
joelho no outro, o tornozelo compensa. Chutar sinal daria uma pose sem sentido. Em
vez disso eu rodo o BFM em `crouch-0.25` (ou no `--alvo` que você escolher), gravo a
pose de junta que ELE produz, e uso `q_baixo - q_alto` como direção. Assim o sinal
sai certo por construção, e o alvo é uma pose que o robô de fato assume.

**A etapa 2 pode já responder tudo.** Antes de simular empurrão nenhum, ela compara
`|q_baixo - q_alto|` por junta contra o clamp. Se a maior parte do agachamento pedir
mais que 0,35 rad, o residual **não consegue expressar** a pose — e isso é
aritmética, não física.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from g1_multitask import tasks as T  # noqa: E402
from g1_residual.poses import _monta, _troca_prior  # noqa: E402
from g1_training.common.robot import PALM_SITES  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

PADRAO_PERNA = r".*(hip|knee|ankle|waist).*"
"""O mesmo escopo do `postura_joints` (`knobs.py:261`) — provado válido no robô."""
PADRAO_BRACO = r".*(shoulder|elbow|wrist).*"

DE_PE = "move-ego-0-0"
ALVO_PADRAO = "crouch-0.25"

TAREFA = T.PARADO
"""Cena do `parado`: sem caixa, sem prateleira atrapalhando a leitura de altura."""


def _resolve(cfg: SceneEntityCfg, scene) -> SceneEntityCfg:
    cfg.resolve(scene)
    return cfg


def _grava_pose(env, termo, robot, nome: str, passos: int, palmas, dev):
    """Roda o BFM puro em `nome` e devolve a pose mais BAIXA que ele sustenta de pé.

    "Sustenta de pé" = tronco perto da vertical (gravidade projetada em z < -0,8).
    Sem esse filtro, um comportamento que colapsa devolveria a pose do robô caído —
    e o `base_z.py` já mediu que o `crouch-0` faz exatamente isso."""
    _troca_prior(termo, TAREFA, nome, 0)
    env.reset()
    acao = torch.zeros(1, env.action_manager.total_action_dim, device=dev)
    melhor = {"pelve": 9.9, "q": robot.data.joint_pos[0].clone(),
              "palma": 9.9, "de_pe_frac": 0.0}
    vertical = 0
    for _ in range(passos):
        env.step(acao)
        g = float(robot.data.projected_gravity_b[0, 2])
        pelve = float(robot.data.root_link_pos_w[0, 2])
        palma = float(robot.data.site_pos_w[0, palmas.site_ids, 2].min())
        ereto = g < -0.8
        vertical += int(ereto)
        if ereto and pelve < melhor["pelve"]:
            melhor.update(pelve=pelve, q=robot.data.joint_pos[0].clone(),
                          palma=palma)
    melhor["de_pe_frac"] = vertical / passos
    melhor["pelve_fim"] = float(robot.data.root_link_pos_w[0, 2])
    return melhor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alvo", default=ALVO_PADRAO,
                   help=f"comportamento que define a direção de descer "
                        f"(default {ALVO_PADRAO})")
    p.add_argument("--passos", type=int, default=300)
    args = p.parse_args()

    # `escala_delta` REAL do treino: o ponto aqui é empurrar de verdade.
    env, termo = _monta(TAREFA, envs=1, sem_terminacao=True, escala_delta=0.15)
    robot = env.scene["robot"]
    dev = env.device
    escala = float(termo.cfg.escala_delta)
    # `_limite` vem `[1, 29]` (é usado em lote no `acao.py`). Achatar para `[29]`,
    # senão todo indexing por junta lê a dimensão de lote.
    limite = termo._limite.clone().reshape(-1)           # [29] rad, por junta
    palmas = _resolve(SceneEntityCfg("robot", site_names=list(PALM_SITES)), env.scene)
    i_perna = _resolve(SceneEntityCfg("robot", joint_names=[PADRAO_PERNA]),
                       env.scene).joint_ids
    i_braco = _resolve(SceneEntityCfg("robot", joint_names=[PADRAO_BRACO]),
                       env.scene).joint_ids
    nomes_j = list(getattr(robot, "joint_names", [])) or [
        f"j{i}" for i in range(robot.data.joint_pos.shape[1])]

    print(f"escala_delta = {escala} | clamp por junta: perna/cintura "
          f"{float(limite[i_perna].max()):.3f} rad "
          f"({float(limite[i_perna].max()) * 57.3:.1f}°), braço "
          f"{float(limite[i_braco].max()):.3f} rad "
          f"({float(limite[i_braco].max()) * 57.3:.1f}°)")
    print(f"{len(i_perna)} juntas de perna/cintura, {len(i_braco)} de braço")

    # ================================================== 1. as duas poses do BFM
    print("\n" + "=" * 78)
    print("1. as duas poses, com o BFM PURO (ação zero)")
    print("=" * 78)
    poses = {}
    for nome in (DE_PE, args.alvo):
        poses[nome] = _grava_pose(env, termo, robot, nome, args.passos, palmas, dev)
        m = poses[nome]
        print(f"{nome:22} pelve_min {m['pelve']:.3f}  pelve_fim "
              f"{m['pelve_fim']:.3f}  palma_min {m['palma']:.3f}  "
              f"ereto {m['de_pe_frac']:.0%}")
    alto, baixo = poses[DE_PE], poses[args.alvo]
    if baixo["de_pe_frac"] < 0.5:
        print(f"\n⚠️ `{args.alvo}` não fica ereto na maior parte do tempo. A pose "
              f"gravada é a mais baixa AINDA ereta, mas ela pode ser transitória.")

    # ============================ 2. o delta cabe no clamp? (pura aritmética)
    print("\n" + "=" * 78)
    print("2. o delta de junta entre as duas poses, contra o clamp")
    print("=" * 78)
    d = baixo["q"] - alto["q"]                          # [29] rad
    print(f"{'junta':30} {'q_de_pe':>8} {'q_alvo':>8} {'delta':>8} "
          f"{'clamp':>7} {'cabe':>6}")
    print("-" * 74)
    excesso, n_fora = 0.0, 0
    pior = ""
    for i in list(i_perna):
        cabe = abs(float(d[i])) <= float(limite[i]) + 1e-9
        n_fora += int(not cabe)
        exc = abs(float(d[i])) - float(limite[i])
        if exc > excesso:
            excesso, pior = exc, nomes_j[i]
        print(f"{nomes_j[i]:30} {float(alto['q'][i]):8.3f} "
              f"{float(baixo['q'][i]):8.3f} {float(d[i]):8.3f} "
              f"{float(limite[i]):7.3f} {'sim' if cabe else 'NAO':>6}")
    # a fração do agachamento que o clamp permite expressar, junta a junta
    frac_expressavel = float(
        (torch.minimum(d[i_perna].abs(), limite[i_perna]).sum()
         / d[i_perna].abs().sum().clamp(min=1e-9)))
    print(f"\n{n_fora} de {len(i_perna)} juntas pedem MAIS que o clamp.")
    if n_fora:
        print(f"maior excesso: {excesso:.3f} rad ({excesso * 57.3:.1f}°) em {pior}")
    print(f"fração do agachamento EXPRESSÁVEL pelo clamp: {frac_expressavel:.1%}")
    print("  (isto é aritmética, não física: é o teto antes de qualquer empurrão)")

    # ================== 3. empurrando com o BFM mandando ficar de pé
    print("\n" + "=" * 78)
    print(f"3. residual empurrando para `{args.alvo}`, com z = `{DE_PE}`")
    print("=" * 78)
    print(f"{'grupo':16} {'fracao':>7} {'pelve_min':>10} {'pelve_fim':>10} "
          f"{'palma_min':>10} {'q_atingido':>11} {'ereto':>6}")
    print("-" * 76)
    _troca_prior(termo, TAREFA, DE_PE, 0)
    n_j = len(nomes_j)
    for rotulo, idx in (("perna+cintura", list(i_perna)),
                        ("braco", list(i_braco)),
                        ("ambos", list(i_perna) + list(i_braco))):
        for frac in (0.0, 0.25, 0.5, 1.0):
            # delta ALVO em rad, saturado no clamp, convertido para a ação CRUA que
            # o `acao.py` espera: delta = clamp(bruto*escala, -1, 1) * limite
            alvo_rad = torch.zeros(n_j, device=dev)
            alvo_rad[idx] = (d[idx] * frac).clamp(-limite[idx], limite[idx])
            bruto = (alvo_rad / limite.clamp(min=1e-9)).clamp(-1.0, 1.0) / escala
            acao = torch.zeros(1, env.action_manager.total_action_dim, device=dev)
            acao[0, :n_j] = bruto                      # os 20 de `c` ficam ZERO
            env.reset()
            pmin, pamin, vert = 9.9, 9.9, 0
            q_soma = torch.zeros(n_j, device=dev)
            for _ in range(args.passos):
                env.step(acao)
                pmin = min(pmin, float(robot.data.root_link_pos_w[0, 2]))
                pamin = min(pamin,
                            float(robot.data.site_pos_w[0, palmas.site_ids, 2].min()))
                vert += int(float(robot.data.projected_gravity_b[0, 2]) < -0.8)
                q_soma += robot.data.joint_pos[0]
            q_med = q_soma / args.passos
            # quanto do delta pedido sobreviveu à correção do BFM
            pedido = (d[idx] * frac).clamp(-limite[idx], limite[idx])
            obtido = q_med[idx] - alto["q"][idx]
            atingido = float((obtido * pedido).sum()
                             / (pedido * pedido).sum().clamp(min=1e-9))
            print(f"{rotulo:16} {frac:7.2f} {pmin:10.3f} "
                  f"{float(robot.data.root_link_pos_w[0, 2]):10.3f} "
                  f"{pamin:10.3f} {atingido:11.2f} {vert / args.passos:6.0%}")

    env.close()
    print("\n" + "=" * 78)
    print("COMO LER")
    print("=" * 78)
    print("`q_atingido` é a resposta da sua pergunta: quanto do delta pedido")
    print("sobrevive à correção do BFM. 1,0 = o residual venceu. 0,0 = o BFM ganhou.")
    print("`palma_min` diz se a mão chega perto do chão. O nível 0,00 do currículo")
    print("de altura põe a caixa no chão, então é esse número que decide o teto.")
    print("Se a etapa 2 já disser que a fração expressável é baixa, o clamp é o")
    print("limite e a etapa 3 só confirma.")


if __name__ == "__main__":
    main()
