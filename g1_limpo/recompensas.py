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

__all__ = ["AlturaDeBalanco", "PosturaPorElo", "staged", "precise_pos",
           "precise_ori", "squeeze", "unload", "postura_ereta", "sustentacao"]


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


# =============================================================================
# OS SETE INCENTIVOS DA MANIPULAÇÃO (F3)
#
# ⚠ TODOS positivos e contínuos. Nenhuma penalidade aqui, e é R3: penalidade limita
# COMO fazer o que já existe, ela não ensina a fazer. E nenhum é booleano — o `pegar`
# do `g1_poc` travou 22 mil iterações num `squeeze` booleano, que é um platô.
#
# ⚠ TODOS multiplicam pelo canal `VALIDA` do comando. Sem esse gate, um env de `ANDAR`
# pagaria o MÁXIMO: com os canais de caixa zerados, `exp(0) = 1`.
#
# ⚠ E TODOS os σ vêm do TERMO DE COMANDO, por env. Eles não são knobs — cada um é a
# distância inicial daquele env. Ver `comando.AlvoCaixaCmd.__init__` e a §4.2b da spec.
# Com σ fixo de 0,10 a 0,339 m o kernel vale 1e−05 e a derivada é ZERO: o robô não tem
# pista de onde ir, e foi isto que travou o `g1_poc`.
# =============================================================================


def _t(env, nome: str):
    """O termo de comando, que é onde os σ e o alvo moram."""
    return env.command_manager.get_term(nome)


def _valida(env, nome: str) -> torch.Tensor:
    """O gate de manipulação: 1 nos elos com caixa, 0 no `ANDAR`."""
    from g1_limpo.comando import VALIDA
    return env.command_manager.get_command(nome)[:, VALIDA]


def _alvo(env, nome: str) -> torch.Tensor:
    from g1_limpo.comando import ALVO
    return env.command_manager.get_command(nome)[:, ALVO]


def _dist_caixa_alvo(env, nome: str) -> torch.Tensor:
    caixa = env.scene["box"].data.root_link_pos_w
    return torch.norm(caixa - _alvo(env, nome), dim=-1)


def _alcancar(env, nome: str) -> torch.Tensor:
    """`exp(−(d_palma/σ_alcance)²)`. O kernel de aproximação da mão.

    No passo em que o elo abre ele vale `exp(−1) = 0,368` por construção, porque
    `σ = d₀`. MEDIDO: 0,3679 a 0,3708 em 32 envs.
    """
    t = _t(env, nome)
    ids = torch.arange(env.num_envs, device=t.sigma_alcance.device)
    d = t.dist_palma_caixa(ids)
    return torch.exp(-(d / t.sigma_alcance.clamp(min=1e-6)) ** 2)


def _forca_das_palmas(env, sensores: tuple[str, ...]) -> torch.Tensor:
    """A força na palma MENOS apertada. `min`, e não soma nem média.

    ⚠ O `min` é o desenho: uma palma sozinha EMPURRA a caixa, ela não a segura. Com
    soma, apertar forte com uma mão pagaria tanto quanto pegar com as duas.
    """
    fs = [torch.norm(env.scene[s].data.force, dim=-1).squeeze(-1) for s in sensores]
    return torch.stack(fs, dim=-1).min(dim=-1).values


def staged(env, nome_do_comando: str) -> torch.Tensor:
    """`alcançar × (1 + trazer)`. O motor da fase inicial.

    ⚠ A forma é PRODUTO, e não soma, e isso importa: `trazer` só paga se a mão já
    estiver perto. Com soma, o robô ganharia por EMPURRAR a caixa até o alvo com o pé
    — e foi assim que uma run antiga aprendeu a chutar a caixa.

    ⚠ Teto de 2,0, e não 1,0. Com peso 3,0 ele contribui até 6,0/s. É o maior termo do
    conjunto de propósito: ele é o único que tem gradiente na pose de repouso.
    """
    t = _t(env, nome_do_comando)
    alcanca = _alcancar(env, nome_do_comando)
    d_alvo = _dist_caixa_alvo(env, nome_do_comando)
    traz = torch.exp(-(d_alvo / t.sigma_trazer.clamp(min=1e-6)) ** 2)
    return alcanca * (1.0 + traz) * _valida(env, nome_do_comando)


def precise_pos(env, nome_do_comando: str, sigma: float) -> torch.Tensor:
    """`exp(−‖caixa−alvo‖²/σ²)` com σ FIXO. É a tolerância de ACEITE.

    ⚠ Único termo com σ fixo, e de propósito: ele responde "a caixa está NO alvo?", que
    é um aceite, não uma rampa de aproximação. Quem faz a rampa é o `staged`, com σ
    por env. Dois termos, duas perguntas.
    """
    d = _dist_caixa_alvo(env, nome_do_comando)
    return torch.exp(-(d / sigma) ** 2) * _valida(env, nome_do_comando)


def precise_ori(env, nome_do_comando: str) -> torch.Tensor:
    """`alcançar × exp(−(Δθ/σ_ori)²)`. A face pedida apontando ao robô.

    ⚠ Gateado por `alcançar`: girar a caixa sem tocá-la não é a tarefa. E o σ é o
    ÂNGULO inicial daquele env — com σ fixo de 0,40 rad um pedido de 90° dava
    `exp(−(1,57/0,40)²) = 2,0e−7`, isto é zero. Era a "sorte de nível 3+" do `g1_poc`.
    """
    from g1_limpo.comando import ANG
    t = _t(env, nome_do_comando)
    erro = env.command_manager.get_command(nome_do_comando)[:, ANG]
    alinha = torch.exp(-(erro / t.sigma_ori.clamp(min=1e-6)) ** 2)
    return _alcancar(env, nome_do_comando) * alinha * _valida(env, nome_do_comando)


def squeeze(env, nome_do_comando: str, sensores: tuple[str, ...],
            forca_ref: float) -> torch.Tensor:
    """`tanh(min(F_E, F_D)/F_ref)`. Força nas DUAS palmas.

    ⚠ `tanh` e não limiar. Ele é contínuo desde a primeira décima de newton, portanto
    existe gradiente antes de a preensão "existir". Um booleano é platô, e o platô
    travou o `pegar` do `g1_poc` por 22 mil iterações — a ponte que faltava era
    exatamente esta continuidade.
    """
    f = _forca_das_palmas(env, sensores)
    return torch.tanh(f / forca_ref) * _valida(env, nome_do_comando)


def unload(env, nome_do_comando: str, sensor_apoio: str) -> torch.Tensor:
    """`1 − F_apoio/(m·g)`. A caixa deixou de pesar na laje.

    ⚠ É A PONTE do `pegar`, e ela é contínua: a força de apoio cai de `m·g` a 0
    conforme o robô assume a carga. MEDIDO no `g1_poc`: 9,7 N -> 0 N. Um "está na mão"
    booleano não tem esse meio-caminho, e sem o meio-caminho não há gradiente.

    ⚠ A massa vem de `env.limpo_massa`, em KG, publicada pelo evento `carga_caixa`.
    Publicar newtons obrigaria este consumidor a desfazer a conta, e é assim que se
    erra um fator 9,81 em silêncio.
    """
    f = torch.norm(env.scene[sensor_apoio].data.force, dim=-1).squeeze(-1)
    peso = env.limpo_massa * 9.81
    return (1.0 - f / peso.clamp(min=1e-6)).clamp(0.0, 1.0) \
        * _valida(env, nome_do_comando)


def postura_ereta(env, nome_do_comando: str, sensores_palma: tuple[str, ...],
                  sensor_apoio: str, forca_ref: float,
                  pelve_alvo: float, pelve_piso: float) -> torch.Tensor:
    """Rampa na pelve × preensão × descarga. Paga por erguer SEM agachar.

    ⚠ É o termo que impede o robô de satisfazer o alvo DESCENDO até a caixa. O alvo já
    tem z absoluto, o que remove o atalho de baixar o alvo; este termo remove o atalho
    de baixar o CORPO para encurtar o alcance.

    ⚠ A rampa é de DOIS LADOS (`clamp(0, 1)`): zero abaixo do piso, um acima do alvo, e
    linear no meio. Sem o clamp superior, esticar-se além do alvo pagaria cada vez
    mais, e o robô aprenderia a ficar na ponta dos pés.

    ⚠ E ela é MULTIPLICADA pela preensão e pela descarga, não somada. Somado, o robô
    colheria a rampa só por ficar de pé sem tocar a caixa — que é exatamente o que ele
    já faz de graça.
    """
    z = (env.scene["robot"].data.root_link_pos_w[:, 2]
         - env.scene.env_origins[:, 2])
    rampa = ((z - pelve_piso) / max(pelve_alvo - pelve_piso, 1e-6)).clamp(0.0, 1.0)
    preensao = torch.tanh(_forca_das_palmas(env, sensores_palma) / forca_ref)
    descarga = unload(env, nome_do_comando, sensor_apoio)
    # ⚠ o `unload` já traz o `VALIDA`; não multiplicar de novo (daria VALIDA²).
    return rampa * preensao * descarga


class sustentacao:
    """`t_na_condição / alvo`. Paga por FICAR lá, e não só por passar por lá.

    ⚠ O CRONÔMETRO LÊ SÓ A CONDIÇÃO DA TAREFA. No `g1_multitask` ele lia também o erro
    angular da base, e o `push_robot` (±0,78 rad/s a cada 1 a 3 s) estourava o teste e
    ZERAVA o contador: o `perf` do locomover marcou 0 nas iterações 13.700 e 17.297
    **com o robô já andando**. Uma régua que uma perturbação externa zera não mede
    competência. Push e régua ficam em compartimentos separados.

    ⚠ E ele TEM `reset` — sem isso `reward_manager.py:174` nunca o chamaria, e o tempo
    acumulado de um episódio entraria no seguinte.
    """

    def __init__(self, cfg, env):
        self.t = torch.zeros(env.num_envs, device=env.device)
        self.dt = env.step_dt

    def __call__(self, env, nome_do_comando: str, tol_pos: float,
                 tol_ang: float, sustenta_s: float) -> torch.Tensor:
        from g1_limpo.comando import ANG
        cmd = env.command_manager.get_command(nome_do_comando)
        perto = _dist_caixa_alvo(env, nome_do_comando) < tol_pos
        alinhado = cmd[:, ANG] < tol_ang
        na_condicao = perto & alinhado & (_valida(env, nome_do_comando) > 0.5)
        self.t = torch.where(na_condicao, self.t + self.dt,
                            torch.zeros_like(self.t))
        return (self.t / max(sustenta_s, 1e-6)).clamp(max=1.0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.t[env_ids] = 0.0
