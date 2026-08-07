"""Visualiza a política do MULTI-TAREFA no viewer. Roda no SEU PC, precisa de display.

    python g1_multitask/play.py ~/Downloads/model_1500.pt
    python g1_multitask/play.py ~/Downloads/model_1500.pt --tarefa pegar
    python g1_multitask/play.py ~/Downloads/model_1500.pt --velocidade 2.0
    python g1_multitask/play.py ~/Downloads/model_1500.pt --peso-cheio
    python g1_multitask/play.py --acao-zero --tarefa locomover   # linha de base, sem rede
    python g1_multitask/play.py ~/Downloads/model_1500.pt --video   # sem janela

Não precisa de GPU: 1 robô roda em CPU sem problema.

**Por que este arquivo existe (04/08/2026).** Não havia `play` do multi-tarefa. O
`play.py` da raiz tem `--task` restrito a `("Stand", "Stand-Step", "Lift")` e nem
registra esta task; o único que cobria as 7 tarefas era o `g1_residual/play.py` — e
aquele registra `build_env_residual`, ou seja **com o BFM no termo de ação**. Desde
`dim_c = 0` (commit `b931c9c`) a ação do residual é 29 e a obs 151, iguais às daqui,
então um checkpoint do multi-tarefa carrega lá **sem `size mismatch` e sem aviso** — e o
que aparece na tela é o BFM congelado segurando o robô de pé. Foi exatamente esse o
engano de 04/08. A etiqueta de espaço de ação no `MultitaskRunner` agora recusa o
cross-load, mas o conserto de raiz é este arquivo: quem quer ver o multi-tarefa não
precisa mais passar pelo play do residual.

⚠️ **`--acao-zero` é a linha de base, e ela importa mais do que parece.** O termo de
ação tem `use_default_offset = True`, então `alvo = ação × escala + default_joint_pos`, e
a pose padrão do G1 é **em pé** (base 0,784 m, joelho 0,3 rad) segurada por atuador de
posição. Com ação zero o robô fica de pé sem política nenhuma. E o `play` roda a MÉDIA
determinística do ator (`rsl_rl/models/mlp_model.py:93`), então o `std` de exploração
~1,0 nunca chega ao sim. Medido no `model_0.pt` da run `2026-08-04_17-20-38_multitask`:
|a| médio **0,0735**, que dá **1,4° por junta** — visualmente idêntico ao `--acao-zero`.
Ficar de pé na iteração 0 não é aprendizado; é o PD. O que separa é **cair e levantar**
(`--sem-quedas`) ou **resistir a push** (Ctrl+arrasto).

`--tarefa` força UMA das 5 no viewer. Sem ela o currículo sorteia, e num robô só você vê
a que caiu no sorteio. Ele mexe no `abertas` do orquestrador, que é a única forma que o
currículo respeita — escrever em `env.tarefa_sorteada` não tem efeito, porque o
`_amostrar` sobrescreve no reset.

No viewer nativo: Ctrl+arrasto empurra o robô, SPACE pausa, setinha dá 1 passo,
`-`/`=` muda a velocidade.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import g1_multitask  # noqa: E402  (o import registra a task)
from g1_multitask import tasks as T  # noqa: E402
from g1_multitask.configs import ACTIVE  # noqa: E402
from g1_multitask.curriculum import Orquestrador  # noqa: E402
from g1_multitask.env import build_multitask_env  # noqa: E402
from g1_multitask.runner import MultitaskRunner  # noqa: E402
from mjlab.scripts.play import PlayConfig, run_play  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

NOMES = {
    "locomover": T.LOCOMOVER,
    "pegar": T.PEGAR,
    "reorientar": T.REORIENTAR,
    "locomover_carregando": T.LOCOMOVER_CARREGANDO,
    "botar": T.BOTAR,
}


TASK_CUSTOM = g1_multitask.TASK_ID + "-PlayCustom"
"""Id SEPARADO. O `register_mjlab_task` NÃO sobrescreve: com o mesmo id ele levanta
`ValueError: Task ... is already registered`. É o mesmo padrão do `play.py` da raiz, que
registra `...-LiftPlay` em vez de mexer na entrada da Lift."""


def _indice_velocidade(ms: float) -> int:
    """m/s -> índice ABSOLUTO em `T.LEVELS["velocidade"]`.

    Em m/s e não em índice de propósito: índice é a convenção INTERNA e ela tem duas
    variantes que já causaram bug (a célula do orquestrador indexa a partir do início da
    tarefa, o `env.nivel` é absoluto — ver `curriculum._base`). m/s é inequívoco.

    ⚠️ O eixo mudou de `distancia_andar` para `velocidade` na reforma de 07/08. O
    `andar` deixou de ser "ir a um lugar": ele rastreia uma velocidade sorteada, e o
    nível é o TETO dessa velocidade."""
    niveis = T.LEVELS["velocidade"]
    for i, v in enumerate(niveis):
        if abs(v - ms) < 1e-6:
            return i
    raise SystemExit(
        f"velocidade {ms} não é um nível. Válidos: {list(niveis)}")


def _registra(tarefa: int | None, velocidade: float | None,
              peso_cheio: bool) -> str:
    """Registra uma task de play com tarefa fixada, teto de velocidade pinado e/ou a
    DR de carga aberta."""
    # `env_cfg`, não `cfg`: o `CurriculumManager` instancia o termo com KEYWORD
    # (`cfg=..., env=...`), então o parâmetro do `__init__` TEM que se chamar `cfg`.
    # Renomeá-lo para não colidir com a variável de fora dá
    # `TypeError: got an unexpected keyword argument 'cfg'`.
    env_cfg = build_multitask_env(ACTIVE, play=True)

    idx = None if velocidade is None else _indice_velocidade(velocidade)

    if tarefa is not None or idx is not None or peso_cheio:
        class _Fixo(Orquestrador):
            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                if tarefa is not None:
                    self.abertas = [tarefa]
                    # `_amostrar` sorteia de `abertas`, e o env nasce antes do primeiro
                    # `__call__`: sem isto o passo 0 rodaria com `parado`.
                    env.tarefa_sorteada[:] = tarefa
                if peso_cheio:
                    # A DR de carga não é eixo: ela é um booleano por tarefa dentro do
                    # orquestrador. O `play` não carrega o estado do currículo, então
                    # sem isto a caixa é sempre de 1 kg.
                    for t in T.COM_DR_PESO:
                        self.dr_peso[t] = True
                if idx is not None:
                    env.nivel["velocidade"][:] = idx
                    env.teto_velocidade[:] = T.LEVELS["velocidade"][idx]

            def _amostrar(self, env, env_ids):
                """Sorteia normal e depois SOBRESCREVE o teto de velocidade.

                Sobrescrever depois, e não mexer no `_dist`: o `_amostrar` escreve
                `env.nivel` para todos os eixos de todas as tarefas sorteadas, e
                reproduzir essa lógica só para trocar um eixo duplicaria a conversão de
                índice (`_base`) — que é justamente onde o bug de 30/07 morava.

                ⚠️ Pinar um nível que o CURRÍCULO não destravou é legítimo aqui: o
                `play` não carrega o estado do currículo (`load_cfg={"actor": True}`),
                então sem isto só o nível INICIAL de cada eixo aparece. Você está vendo
                comportamento fora da distribuição de treino, o que é o ponto."""
                super()._amostrar(env, env_ids)
                if idx is not None:
                    env.nivel["velocidade"][env_ids] = idx
                    # ⚠️ Os DOIS. O `env.nivel` é o que o log lê; o `teto_velocidade` é
                    # o que a subclasse de comando usa para reescalar o sorteio. Mexer
                    # só no primeiro não muda o comando.
                    env.teto_velocidade[env_ids] = T.LEVELS["velocidade"][idx]

        env_cfg.curriculum["orquestrador"].func = _Fixo

    register_mjlab_task(
        task_id=TASK_CUSTOM, env_cfg=env_cfg, play_env_cfg=env_cfg,
        rl_cfg=g1_multitask._rl_cfg(), runner_cls=MultitaskRunner)
    return TASK_CUSTOM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, nargs="?", default=None,
                   help="caminho pro model_<iter>.pt (dispensável com --acao-zero)")
    p.add_argument("--tarefa", choices=sorted(NOMES), default=None,
                   help="força uma das 5 no viewer (default: o currículo sorteia)")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--acao-zero", action="store_true",
                   help="ignora a política e manda AÇÃO ZERO. É a linha de base que diz "
                        "o que é do PD e o que é da rede: com `use_default_offset` a "
                        "ação zero já pede a pose padrão, que é EM PÉ. Se o checkpoint "
                        "não se distingue disto, ele não aprendeu nada ainda.")
    p.add_argument("--sem-quedas", action="store_true",
                   help="não encerra o episódio quando o robô cai. Ele fica no chão e "
                        "você vê se ele LEVANTA — que é um teste que a ação zero não "
                        "passa, ao contrário de ficar de pé.")
    p.add_argument("--velocidade", type=float, default=None, metavar="MS",
                   help=f"pina o TETO do comando de velocidade, em m/s. Válidos: "
                        f"{list(T.LEVELS['velocidade'])}. O `play` não carrega o "
                        f"estado do currículo, então sem isto só o nível INICIAL aparece "
                        f"(1.0 m/s). Com isto você vê um nível que o treino talvez ainda "
                        f"não tenha destravado.")
    p.add_argument("--peso-cheio", action="store_true",
                   help="abre a DR de carga nas 4 tarefas com caixa: a massa passa de "
                        "1 kg fixo para U(1, 5) kg. Ela não é eixo — é um booleano por "
                        "tarefa no orquestrador, e o `play` não o carrega do checkpoint.")
    p.add_argument("--video", action="store_true", help="grava mp4, sem janela")
    p.add_argument("--video-length", type=int, default=500)
    args = p.parse_args()

    if args.acao_zero:
        if args.checkpoint:
            print("[PLAY] --acao-zero: o checkpoint é IGNORADO")
        ckpt = None
    else:
        if not args.checkpoint:
            raise SystemExit("passe o checkpoint, ou use --acao-zero")
        ckpt = pathlib.Path(args.checkpoint).expanduser()
        assert ckpt.is_file(), f"não achei {ckpt}"

    task_id = g1_multitask.TASK_ID
    if args.tarefa or args.velocidade is not None or args.peso_cheio:
        task_id = _registra(NOMES.get(args.tarefa), args.velocidade,
                            args.peso_cheio)
        if args.tarefa:
            print(f"[PLAY] tarefa forçada: {args.tarefa}")
        if args.velocidade is not None:
            print(f"[PLAY] teto de velocidade pinado: {args.velocidade} m/s "
                  f"(nível {_indice_velocidade(args.velocidade)})")
        if args.peso_cheio:
            print("[PLAY] DR de carga ABERTA: massa da caixa em U(1, 5) kg")

    print("[PLAY] task MULTI-TAREFA — ação de junta direta, SEM BFM no laço")
    if args.acao_zero:
        print("[PLAY] AÇÃO ZERO — o que você vê é a pose padrão segurada pelo PD")
    if args.sem_quedas:
        print("[PLAY] terminações DESLIGADAS — o episódio não acaba quando ele cai")
    run_play(task_id, PlayConfig(
        agent="zero" if args.acao_zero else "trained",
        checkpoint_file=None if ckpt is None else str(ckpt),
        num_envs=args.envs,
        viewer="native", video=args.video, video_length=args.video_length,
        no_terminations=args.sem_quedas,
    ))


if __name__ == "__main__":
    main()
