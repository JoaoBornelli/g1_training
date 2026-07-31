"""O env do experimento: o CURRÍCULO INTEIRO, com o BFM dando equilíbrio.

Ele NÃO reimplementa nada. Chama o `build_multitask_env` e troca três coisas:

    1. a AÇÃO      residual sobre o BFM, 49 números em vez de 29
    2. o CURRÍCULO subclasse que já começa com o `pegar` aberto
    3. uma MÉTRICA log da taxa do alvo composto, para ver oscilação

Todo o resto do desenho continua igual, e é isso que faz o experimento comparável:
o comando de 17 números, a obs de 151, as rewards e os gates da §6b, as três
terminações, o sucesso físico em `env.success_buf`, a observabilidade por tarefa ×
termo, o congelamento do normalizador, o round-trip do currículo no checkpoint.

**Por que o currículo INTEIRO e não só o `pegar` (decisão do user, 30/07).** Porque
com o BFM o `parado` deveria ser quase de graça, e aí a cascata anda: `parado` a 0,90
libera o eixo de push (4 destravamentos), o push completo abre o `andar`, e o BFM já
sabe andar (`move-ego-0-0.3`). O orçamento da run vai para as partes difíceis em vez
de reaprender equilíbrio.

E tem um ganho de DIAGNÓSTICO que a versão só-`pegar` não dava: o `fumaca.py` mostra
que o BFM fica 150 de 150 passos de pé sozinho. Então, com o `parado` aberto, **se
ele não der sucesso a culpa é 100% do residual** — não existe outra explicação. No
`pegar` as duas causas ficavam misturadas.

⚠️ **Calibração honesta sobre "rápido":** a EMA usa `alpha = 0,03`, então ela precisa
de ~75 atualizações para subir de 0 a 0,90 mesmo com 100% de sucesso. Cada
destravamento leva algumas centenas de iterações, não uma.

`so_pegar=True` volta ao recorte anterior, se você quiser isolar.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_multitask import tasks as T  # noqa: E402
from g1_multitask.configs import ACTIVE  # noqa: E402
from g1_multitask.curriculum import Orquestrador  # noqa: E402
from g1_multitask.env import build_multitask_env  # noqa: E402
from g1_residual.acao import ResidualBFMActionCfg  # noqa: E402
from mjlab.managers.metrics_manager import MetricsTermCfg  # noqa: E402


class OrquestradorPegar(Orquestrador):
    """O mesmo orquestrador, começando pelo `pegar` em vez do `parado`.

    Só a lista de tarefas abertas muda. Os eixos do `pegar` (altura, peso,
    distância), o portão de 0,90, o congelamento, o sorteio PLR rank-based e o
    `state_dict` continuam os do desenho.

    O `FILHOS[PEGAR]` é `(PARADO_CAIXA,)`, então se o `pegar` chegar a 0,90 ele abre
    o `parado c/ caixa` sozinho. Numa run curta isso não acontece — e se acontecer é
    boa notícia, não problema."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.abertas = [T.PEGAR]
        # `_amostrar` sorteia de `abertas`, e o env nasce antes do primeiro
        # `__call__`: sem isto o passo 0 rodaria com a tarefa 0 (`parado`).
        env.tarefa_sorteada[:] = T.PEGAR
        if self.verboso:
            print(f"[CURRICULO] experimento residual: só `{T.NAMES[T.PEGAR]}` aberto, "
                  f"eixos {sorted(T.AXES[T.PEGAR])}")


def taxa_alvo(env, nome_termo: str = "joint_pos") -> torch.Tensor:
    """Métrica de LOG: quanto o alvo de junta composto varia por passo.

    Existe porque o `action_rate_l2` do fabricante lê a saída crua da política e
    portanto é cego para a parte do BFM. Se o residual e o reflexo do BFM oscilarem
    um contra o outro, aparece aqui e no `joint_acc`, em nenhum outro lugar."""
    return env.action_manager.get_term(nome_termo).taxa_alvo


def build_env_residual(knobs=ACTIVE, play: bool = False,
                       so_pegar: bool = False,
                       prior_unico: bool = False,
                       escala_delta: float = 0.15,
                       limite_rad: dict[str, float] | None = None,
                       rolagens_por_passo: int = 2):
    cfg = build_multitask_env(knobs, play)

    # --- 1. a ação --------------------------------------------------------
    # Copia o escopo e a escala do termo original: o `__init__` do pai usa os dois
    # para resolver `_target_names`, e a conta de conversão de unidade é explícita
    # dentro do `process_actions`.
    v = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = ResidualBFMActionCfg(
        entity_name=v.entity_name, transmission_type=v.transmission_type,
        actuator_names=v.actuator_names, scale=v.scale, offset=v.offset,
        preserve_order=v.preserve_order, use_default_offset=v.use_default_offset,
        clip=v.clip,
        limite_rad=limite_rad, prior_unico=prior_unico,
        escala_delta=escala_delta,
        rolagens_por_passo=rolagens_por_passo)

    # --- 2. o currículo ---------------------------------------------------
    # Por padrão NÃO troca nada: o orquestrador do desenho começa no `parado` e
    # abre as 7 tarefas em cascata, que é o ponto do experimento.
    if so_pegar:
        cfg.curriculum["orquestrador"].func = OrquestradorPegar

    # --- 3. a métrica de oscilação ---------------------------------------
    cfg.metrics["taxa_alvo"] = MetricsTermCfg(func=taxa_alvo)

    return cfg
