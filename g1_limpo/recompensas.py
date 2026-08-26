"""Os termos de recompensa PRÓPRIOS do g1_limpo.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

Na F1 este arquivo é pequeno de propósito: a locomoção do fabricante é a fundação, e
tudo que ela já entrega fica como está. Aqui vive só o que o molde NÃO tem, e cada
item traz o defeito medido que o justifica.

Os sete incentivos de manipulação entram na F3.
"""
from __future__ import annotations

import torch

from mjlab.tasks.velocity.mdp import feet_swing_height, variable_posture

__all__ = ["AlturaDeBalanco", "PosturaPorElo"]


class AlturaDeBalanco(feet_swing_height):
    """O `feet_swing_height` do fabricante, com o `reset` que falta.

    ⚠ ISTO É UM BUG DO MOLDE, e ele é silencioso. O termo do fabricante acumula
    `peak_heights` por pé e só zera no PRIMEIRO CONTATO. Mas
    `reward_manager.py:174` só registra um termo de classe em `_class_term_cfgs` —
    a lista dos que recebem `reset(env_ids)` — quando a classe TEM um método
    `reset`. O `feet_swing_height` não tem.

    Consequência: quando o episódio termina com um pé no ar (isto é, toda vez que o
    robô CAI), o pico daquele pé sobrevive ao reset e entra no episódio seguinte. O
    `Metrics/peak_height_mean` então INFLA com queda, e o painel mostra "o passo está
    subindo" exatamente quando o robô está caindo mais.

    Foi assim que um bloco leu `peak_height` em alta durante 5000 iterações com o robô
    imóvel: a altura vinha do vôo da queda, e não de passo nenhum.

    O conserto tem três linhas e nenhum número.
    """

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak_heights[env_ids] = 0.0


class PosturaPorElo(variable_posture):
    """O `variable_posture` do fabricante, NEUTRO nos elos de manipulação.

    ⚠ POR QUE NÃO É UM 4º REGIME DE σ, que era o desenho do plano. Porque medi, e
    nenhum σ resolve. O termo é `exp(−média(erro²/σ²))` sobre 29 juntas; com 17 delas
    fora do default ele é um produto de 17 gaussianas, e colapsa para qualquer σ.

    MEDIDO em 2026-08-26, com a excursão em fração da faixa real de cada junta (faixa
    média das 17 juntas de manipulação: 3,77 rad):

        fração   standing   walking   running   run×3   run×5
          0,10      0,000     0,014     0,184   0,829   0,935
          0,20      0,000     0,000     0,001   0,471   0,763
          0,30      0,000     0,000     0,000   0,184   0,544
          0,40      0,000     0,000     0,000   0,049   0,338

    Três coisas saem daí:

    1. `std_standing` é **uma entrada só**, `.*` = 0,05, para TODAS as juntas. E o
       `walking_threshold` do G1 é **0,05**, não 0,5 (medido no cfg). Com o twist
       forçado a zero num elo de manipulação, `total_speed = 0 < 0,05` SEMPRE — logo o
       regime `standing` é certo, não provável.
    2. O termo não vale 0,93/s a menos: ele vale **exatamente zero**, já a 10% da
       faixa. `exp(−muito)` é 0 em float32, **com gradiente zero**. Não é uma
       penalidade forte, é um canal morto.
    3. Nem `running×5` sobrevive a 40% da faixa. Um multiplicador só empurra o
       penhasco alguns centímetros para a direita.

    ⚠ E excluir os braços não basta. Medido: com os braços fora da média, um braço
    esticado custa 0,000 — ótimo — mas um AGACHAMENTO com as pernas em `running` dá
    0,128 a 10% da faixa e 0,000 a 20%. E o nível 4+ põe a laje a 0,04 m, o que EXIGE
    agachar. Excluir as pernas também não sobra nada: o termo inteiro é "mantenha a
    pose default", e um elo de manipulação exige sair dela.

    PORTANTO O TERMO NÃO TEM O QUE DIZER NUM ELO DE MANIPULAÇÃO, e a resposta certa é
    ficar calado. É R3 na forma mais limpa: o que segura o robô de pé passa a ser o
    `upright` (+1,0, do fabricante, e independente de elo) mais a própria condição de
    fechamento do elo — o `PEGAR` só fecha "de pé". **Incentivo para a ação certa, e
    não penalidade por transgressão.**

    ⚠ RETORNA 1,0, E NÃO 0,0, nos elos de manipulação. Zero faria o env de manipulação
    pagar 1,0/s só por estar naquele elo — uma penalidade por sorteio. Um faz o termo
    NEUTRO, e mantém a escala de retorno comparável entre elos, que é o que o
    controlador de fatia da F5 vai ler.

    ⚠ Os braços seguem contidos por outros cinco termos que não dependem de elo:
    `action_rate_l2`, `joint_acc`, `angular_momentum`, `body_ang_vel`,
    `dof_pos_limits` e `self_collisions`. Não é terra sem lei — é só a instrução
    "volte à pose default" que sai.
    """

    def __call__(self, env, *args, canal_do_elo: int, nome_do_comando: str,
                 elos_que_andam: tuple[int, ...], **kwargs) -> torch.Tensor:
        valor = super().__call__(env, *args, **kwargs)
        comando = env.command_manager.get_command(nome_do_comando)
        assert comando is not None
        elo = comando[:, canal_do_elo].long()
        anda = torch.isin(elo, torch.tensor(elos_que_andam, device=valor.device))
        return torch.where(anda, valor, torch.ones_like(valor))
