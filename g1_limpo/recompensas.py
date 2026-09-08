"""Os termos de recompensa PRÓPRIOS do g1_limpo.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

Na F1 este arquivo é pequeno de propósito: a locomoção do fabricante é a fundação, e
tudo que ela já entrega fica como está. Aqui vive só o que o molde NÃO tem, e cada
item traz o defeito medido que o justifica.

Os sete incentivos de manipulação entram na F3.
"""
from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import feet_swing_height, variable_posture
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.string import resolve_matching_names_values

__all__ = ["AlturaDeBalanco", "PosturaPorElo", "rastreio_por_elo",
           "velocidade_por_regime", "contato_mesa",
           "staged", "precise_pos", "precise_ori", "squeeze", "unload",
           "postura_ereta", "load",
           "renda_congelada"]


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


def contato_mesa(env, sensor_name: str, joelho_N: float,
                 saturacao_N: float) -> torch.Tensor:
    """Rampa de 0 a 1 na força contra a mesa. O PESO é negativo, no `knobs`.

    ⚠⚠ ISTO SUBSTITUI UMA TERMINAÇÃO, e a troca é decisão do dono em 01/09 apoiada em
    medição. Com a terminação, 76% dos episódios de manipulação morriam na mesa e a
    aritmética favorecia ficar parado: 90 de retorno contra 66 de tentar. E o `play`
    fechou o caso — a ação MÉDIA nem se aproximava da mesa, portanto aqueles 76% eram
    RUÍDO de exploração encostando, não uma política que tenta e falha. A terminação
    matava a exploração antes de ela refinar a pega.

    MEDIDO depois da troca: `descarga` de 0,0 a 0,994 e `postura_ereta` saindo de zero
    pela primeira vez no módulo.

    ⚠ RAMPA, e não booleano. Abaixo do joelho a multa é ZERO — roçar o tampo ao alcançar
    sai de graça, exatamente como o limiar da terminação já fazia. Entre o joelho e a
    saturação existe GRADIENTE para tirar o peso da mesa; booleano seria platô, que é o
    defeito que travou o `squeeze` por 22 mil iterações.

    ⚠ O joelho é o MESMO 50 N que governava a terminação, medido no `g1_poc`
    (`knobs.py:328`). Não é número novo.

    ⚠ `amax` sobre os slots, e não `sum`: a pergunta é "algum ponto de contato passa do
    joelho", não "a soma de todos passa". Com `reduce="netforce"` e `num_slots=1` os dois
    coincidem hoje; o `amax` continua certo se alguém subir os slots.

    ⚠ Devolve POSITIVO em [0, 1]. Quem faz dela penalidade é o peso negativo, que é a
    convenção do molde — o `action_rate_l2` também devolve positivo.
    """
    f = env.scene[sensor_name].data.force
    assert f is not None, f"sensor '{sensor_name}' precisa do field 'force'."
    forca = torch.norm(f, dim=-1).amax(dim=-1)
    return ((forca - joelho_N) / max(saturacao_N - joelho_N, 1e-6)).clamp(0.0, 1.0)


def rastreio_por_elo(env, *, func, **kwargs) -> torch.Tensor:
    """O termo de rastreio do fabricante, com QUATRO estados: twist zerado, `VALIDA`
    e `limpo_pegou` (G1, spec `g1-limpo-lento-e-estavel.md` §2, correção 2026-09-08).

    ⚠⚠ ESTE TERMO EXISTE POR UMA MEDIÇÃO, e ela é a mais decisiva do módulo até hoje.
    O `smoke` mede o piso da estátua por elo — robô travado, sem fazer nada:

        piso ANDAR  = 3,863/s
        piso PEGAR  = 8,265/s      <- 2,1x mais

    No elo `PEGAR` o twist é FORÇADO A ZERO. Portanto ficar imóvel era a resposta
    PERFEITA para a metade de locomoção: os dois termos de rastreio pagavam cheio
    (2,0 + 2,0 = 4,0/s) por rastrear um comando nulo. O elo de manipulação era o lugar
    mais confortável do ambiente.

    A conta que a política fazia, com episódio de 17,6 s e 60% de morte na mesa
    (bloco 6, iteração 785 de 8192 envs = 1571 equivalentes):

        ficar parado   8,265 x 17,6                        = 145
        explorar       0,6 x (8,265 x 8,8) + 0,4 x 145     = 102

    Ficar parado ganhava por 43%. A política estava no ótimo — e o `play` confirmou de
    forma direta: **na ação MÉDIA o robô fica imóvel na pose default e não tenta pegar.**
    Não era gradiente morto nem preguiça. Era aritmética, e ela estava certa.

    ⚠ v2.1 (spec P4): zerar os dois SEMPRE que `env.limpo_twist_zerado` for 1 derrubou
    o piso para ~4,27/s. Mas isso também zerou o rastreio no `PEGAR` e no `BOTAR`
    ATIVOS — onde ficar parado É a tarefa — e a §0 da spec
    `g1-limpo-lento-e-estavel.md` mede o preço: manipulação parada corre 7,3× mais
    rápido por junta no p90 (3,13 rad/s) que locomoção parada (0,43), embora as duas
    recebam o MESMO comando zero.

    ⚠⚠ G1 ACRESCENTA UM TERCEIRO ESTADO — "elo parado ATIVO paga cheio" — e a revisão
    de 2026-09-08 acrescentou um QUARTO antes de ele ir ao treino: SEM o quarto, o
    terceiro estado só olha `VALIDA`, e um env que fica parado com a tarefa ativa e
    NUNCA toca a caixa colheria os 4,0/s por construção do fator (zerado=1, valida=1
    bastaria) — de volta ao piso da estátua que o P4 tinha acabado de remover. O
    quarto estado lê `env.limpo_pegou`, a arma monotônica que `comando._publica_pegou`
    liga na primeira vez em que as DUAS palmas tocam a caixa com a tarefa ativa, e que
    NÃO desarma ao soltar:

        locomoção, `zerado = 0`                    paga cheio — rastreio de verdade
        espera, `VALIDA = 0`                        paga ZERO  — não existe tarefa
        elo parado ATIVO, nunca tocou a caixa       paga ZERO  — ainda é estátua
        elo parado ATIVO, JÁ tocou a caixa          paga cheio — segurar É a tarefa

    ⚠⚠ `engajado = limpo_pegou`, SEM `× VALIDA` (spec `g1-limpo-dois-bits.md` §2.7,
    revisão do PM item 2). Até a v3, `engajado = VALIDA × limpo_pegou`: na espera
    ENTRE elos, com a caixa JÁ na mão, `VALIDA` é 0 e `engajado` caía a zero — o
    rastreio de v=0 pagava zero, e derivar durante a espera era grátis. Como
    `limpo_pegou` só liga depois do primeiro toque com a tarefa ATIVA
    (`comando._publica_pegou` lê `_espera`, não `VALIDA`), o gate contra a estátua
    ANTES da pega continua de pé sem o fator extra.

    `fator = 1 − zerado × (1 − engajado)` — BRANCHLESS, e os quatro estados
    conferem à mão: zerado=0 -> fator=1; zerado=1, pegou=0 -> engajado=0 -> fator=0;
    zerado=1, pegou=1 -> engajado=1 -> fator=1 (com ou sem VALIDA).

    ⚠ `limpo_pegou` SOBREVIVE ao soltar — é monotônico dentro do episódio
    (`comando._publica_pegou`, operador `|=`, só zera no resample). A espera final
    segue paga, porque o robô já tocou a caixa antes de largá-la.

    ⚠ O QUE ISTO COBRE: velocidade da BASE nos elos parados ativos JÁ ENGAJADOS.
    Conserta o arrasto lateral com a caixa na mão (observação do dono) — mas só
    depois do primeiro toque; antes dele a estátua parada continua de graça, de
    propósito: não existe "segurar" sem ter pegado.

    ⚠ RETORNA ZERO, e não 1,0 como o `PosturaPorElo`. A diferença é o propósito: lá o
    termo NÃO TEM O QUE DIZER num elo de manipulação, e 1,0 o deixa neutro. Aqui o
    termo tem algo a dizer nos dois primeiros estados de zerado — pagar pela ausência
    de tarefa está ERRADO — e nada a dizer nos dois últimos, onde ele volta a dizer a
    verdade.

    ⚠ E o offset constante por elo NÃO enviesa a política. O canal do elo está nas duas
    observações (`actor` e `critic`), portanto a função de valor condiciona nele e
    absorve o degrau; a vantagem é medida contra essa baseline.

    ⚠ O QUE NÃO SAI: `upright` (fundação, e é o que guarda o eixo vertical) e `pose`.
    Só os dois de rastreio. E `fell_over` continua guardando a queda.

    ⚠ O RISCO DECLARADO: o alvo do `PEGAR` é ancorado na BASE, então vagar move o alvo
    junto — não há ganho em vagar. O quarto estado só adia o pagamento até o primeiro
    toque; depois disso o risco é o mesmo que a v2.1 já assumia. Se `eficiencia_min`
    cair junto com `palmas_em_contato` subindo, é este risco se realizando.
    """
    valor = func(env, **kwargs)
    engajado = env.limpo_pegou
    fator = 1.0 - env.limpo_twist_zerado * (1.0 - engajado)
    return valor * fator


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


def _gate_espera(env) -> torch.Tensor:
    """1 nas DUAS janelas de espera (inicial e final): `aguardando + soltou`,
    saturado em 1. A MESMA expressão de `metricas.fracao_esperando` — ver
    `pose_de_braco` para o porquê de o gate certo não ser `1 − VALIDA`."""
    v = getattr(env, "limpo_aguardando", None)
    if v is None:
        return torch.zeros(env.num_envs, device=env.device)
    s = getattr(env, "limpo_soltou", None)
    if s is None:
        return v
    return torch.clamp(v + s, max=1.0)


def _elo_interno(env, nome: str) -> torch.Tensor:
    """O elo INTERNO do termo de comando (spec §6.0): o que paga lê o interno."""
    return env.command_manager.get_term(nome)._elo


def _fora_do_botar(env, nome: str) -> torch.Tensor:
    """1 fora do BOTAR, 0 nele. A máscara do g1_poc para `squeeze` e `unload`."""
    from g1_limpo.comando import BOTAR
    return (_elo_interno(env, nome) != BOTAR).float()


def _alvo(env, nome: str) -> torch.Tensor:
    from g1_limpo.comando import ALVO
    return env.command_manager.get_command(nome)[:, ALVO]


def _dist_caixa_alvo(env, nome: str) -> torch.Tensor:
    caixa = env.scene["box"].data.root_link_pos_w
    return torch.norm(caixa - _alvo(env, nome), dim=-1)


def _alcancar(env, nome: str) -> torch.Tensor:
    """`exp(−(d_palma/σ_alcance)²)`. O kernel de aproximação da mão.

    No passo em que `VALIDA` acende ele vale `exp(−1) = 0,368` por construção, porque
    `σ = d₀`, a distância medida NAQUELE passo (F1, `comando._recalcula_sigmas`, chamada
    de dentro de `_aplica_espera`). MEDIDO: 0,3679 a 0,3708 em 32 envs.

    ⚠ ANTES DA F1 o σ era travado no RESET, e a janela de espera deixa o robô
    aproximar as mãos de graça antes de `VALIDA` acender — a invariante acima era
    falsa. MEDIDO no `play` do `bloco9` em 2026-09-08: σ = 0,34 m no reset, mão a
    0,20 m no fim da espera, `alcancar = exp(−(0,20/0,34)²) = 0,71`.

    ⚠ `alcança ≡ 1` no BOTAR e na espera final (`soltou`) — spec §6.6.2 item 3, §8.3.
    No BOTAR as mãos já estão na caixa: σ cai no piso de 0,08 m e o kernel vale 1 por
    construção; ele não carrega informação ali, só paga 3/s por MANTER as mãos na caixa,
    que é o freio contra largar. Com `≡ 1`, `staged` vira `3 × (1 + trazer)` e
    `precise_ori` vira `alinha`: pagam pela caixa, indiferentes às mãos.
    """
    from g1_limpo.comando import BOTAR
    t = _t(env, nome)
    ids = torch.arange(env.num_envs, device=t.sigma_alcance.device)
    d = t.dist_palma_caixa(ids)
    kernel = torch.exp(-(d / t.sigma_alcance.clamp(min=1e-6)) ** 2)
    um = t._elo == BOTAR
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        um = um | (soltou > 0.5)
    return torch.where(um, torch.ones_like(kernel), kernel)


def _forca_das_palmas(env, sensores: tuple[str, ...],
                      asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A força NORMAL na palma MENOS apertada. `min`, e não soma nem média.

    ⚠ O `min` é o desenho: uma palma sozinha EMPURRA a caixa, ela não a segura. Com
    soma, apertar forte com uma mão pagaria tanto quanto pegar com as duas.

    ⚠ SÓ A COMPONENTE NORMAL AO PAD, e este é um ANTI-HACK que a reescrita tinha
    perdido. Até 28/08 isto era `‖F‖`, a magnitude inteira. Com a magnitude, apertar a
    caixa PARA BAIXO contra a prateleira paga como preensão — e aquela força é
    TANGENCIAL ao pad, não normal. A projeção a descarta sem precisar de um segundo
    termo. O `g1_poc` declara o mesmo em `recompensas.squeeze`.

    ⚠ A normal vem da ORIENTAÇÃO DO SÍTIO, e não do campo `normal` do sensor: os
    sensores de palma usam `reduce="netforce"`, que soma todos os contatos num wrench
    só — e aí "a normal do contato" perde significado. A força sai no frame GLOBAL,
    porque `netforce` implica global.

    ⚠ A palma esquerda olha para `−y` local e a direita para `+y` local. É o que a
    geometria dos pads produz: `_add_pad` põe o pad em `pos = (_PALM_X, dz, 0)`, com
    `dz = −0,015` na esquerda e `+0,015` na direita — o offset é em Y, apesar do nome
    do parâmetro. Ver `cena.add_pads_de_palma`.

    ⚠ `abs` em vez de sinal: errar a convenção zeraria o termo em silêncio, e o que
    queremos excluir é a tangencial, que a projeção já removeu.
    """
    quat = env.scene[asset_cfg.name].data.site_quat_w[:, asset_cfg.site_ids]  # [B,2,4]
    locais = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                          device=quat.device, dtype=quat.dtype)
    normais = quat_apply(quat, locais.expand(quat.shape[0], 2, 3))            # [B,2,3]

    fs = []
    for i, s in enumerate(sensores):
        f = env.scene[s].data.force                       # [B, slots, 3], global
        assert f is not None, f"sensor '{s}' precisa do field 'force'."
        fs.append(torch.sum(f * normais[:, i].unsqueeze(1), dim=-1).abs().sum(dim=-1))
    return torch.stack(fs, dim=-1).min(dim=-1).values


def _forca_ref(env, mu: float) -> torch.Tensor:
    """`m·g / (2μ)` — a força de aperto que o atrito precisa para segurar a caixa.

    ⚠ É DERIVADA, e não escolhida, e a diferença é medível. Até 28/08 isto era um knob
    fixo de 12,0 N. Com a caixa de 1,0 kg e μ = 0,8 o valor físico é **6,13 N**,
    portanto o knob pedia o DOBRO do necessário e pagava metade no primeiro newton —
    justo a faixa em que a preensão tem de nascer. O `g1_poc` usa esta mesma conta.

    ⚠ A massa é POR ENV: `env.limpo_massa`, em kg, publicada pelo evento `carga_caixa`.
    Uma caixa mais pesada exige mais aperto, e o termo acompanha sozinho.
    """
    return (env.limpo_massa * 9.81 / (2.0 * mu)).clamp(min=1e-3)


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
            mu: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """`tanh(min(F_n_E, F_n_D)/F_ref)`. Força NORMAL nas DUAS palmas.

    ⚠ `tanh` e não limiar. Ele é contínuo desde a primeira décima de newton, portanto
    existe gradiente antes de a preensão "existir". Um booleano é platô, e o platô
    travou o `pegar` do `g1_poc` por 22 mil iterações — a ponte que faltava era
    exatamente esta continuidade.

    ⚠ E a continuidade só serve se o passo ANTERIOR da cadeia levar as duas mãos às
    faces laterais. Ver `comando.dist_palma_caixa`: enquanto o alcance era `min` sobre
    uma esfera, uma mão saturava o `staged` e a segunda não tinha gradiente — este
    termo ficava em zero exato porque ele é `min` das duas forças.
    """
    f = _forca_das_palmas(env, sensores, asset_cfg)
    # ⚠ ZERO NO BOTAR (spec §6.6.2 item 2; g1_poc: "apertar durante o botar paga contra
    # soltar, −1,0/s medido"). Pagar por segurar é pagar contra a tarefa de largar.
    return (torch.tanh(f / _forca_ref(env, mu)) * _valida(env, nome_do_comando)
            * _fora_do_botar(env, nome_do_comando))


def unload(env, nome_do_comando: str, sensor_apoio: str,
           sensores_palma: tuple[str, ...], mu: float,
           asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """`1 − F_apoio/(m·g)`. A caixa deixou de pesar na laje.

    ⚠ É A PONTE do `pegar`: a força de apoio cai de `m·g` a 0 conforme o robô assume a
    carga. MEDIDO 2026-08-27, erguendo a caixa da laje: `F_apoio` 9,80 N -> 0,00 N e o
    termo 0,0005 -> 1,0. Os dois extremos são os que o plano pedia.

    ⚠ MAS A TRANSIÇÃO É ESTREITA, e isto CORRIGE o que este docstring afirmava antes.
    Erguendo a caixa em degraus, ela salta de 0 a 1 em **2 mm** — o contato é rígido, e
    a força de apoio não passeia por valores intermediários enquanto a caixa sobe. Como
    rampa de altura, o `unload` é quase um booleano.

    ⚠ E o meu método não decide o caso que importa. Eu TELEPORTEI a caixa, portanto o
    teste não modela PARTILHA DE CARGA: numa pega real a força da palma sobe enquanto a
    do apoio desce, e as duas somam `m·g` — ali a força de apoio PASSA pelos valores do
    meio, mesmo com a caixa quase imóvel. Isso só uma run com preensão mede.

    Consequência de desenho, e ela é tranquila: o gradiente de aproximação vem do
    `staged` e o de força vem do `squeeze` (`tanh`, contínuo desde o primeiro newton).
    O `unload` marca "a caixa saiu da laje". Como quase-booleano ele é um bônus, e não a
    rampa que o `pegar` precisa — e é bom que os três não dependam um do outro.

    ⚠ A massa vem de `env.limpo_massa`, em KG, publicada pelo evento `carga_caixa`.
    Publicar newtons obrigaria este consumidor a desfazer a conta, e é assim que se
    erra um fator 9,81 em silêncio.

    ⚠⚠ PORTEIRO DE PREENSÃO, acrescentado em 28/08 por MEDIÇÃO. Sem ele o termo lê só
    "a caixa não pesa na laje", e DERRUBAR a caixa satisfaz isso perfeitamente: uma vez
    no chão, `F_apoio` é zero para sempre e o termo paga 2,0/s pelo resto do episódio,
    sem mão nenhuma. No bloco 3, it 4251, o `unload` marcava 0,0995 com o `squeeze` em
    0,0002 — descarga sem preensão, que é a assinatura desse atalho.

    O porteiro é `tanh(F_palmas/F_ref)`, o MESMO fator do `squeeze`. Ele é contínuo
    desde o primeiro newton, portanto ele fecha o atalho sem virar um degrau.
    """
    # ⚠ AQUI A NORMA FICA, e é decisão declarada (03/09). O `load` e o fecho do `BOTAR`
    # passaram a projetar no eixo vertical porque a norma os fazia ACEITAR um estado
    # errado (prensar de lado contava como apoiar). Neste termo o erro da norma aponta
    # para o outro lado: uma força lateral INFLA `f` e portanto REDUZ a descarga — ela
    # subpaga o erguer, nunca superpaga. E este termo é metade da cadeia da pega, que
    # está medida funcionando (`descarga = 0,965` na `bloco7`); trocar a leitura dele
    # mexeria no que a §2 declara intocável, para consertar um erro conservador.
    f = torch.norm(env.scene[sensor_apoio].data.force, dim=-1).squeeze(-1)
    peso = env.limpo_massa * 9.81
    descarga = (1.0 - f / peso.clamp(min=1e-6)).clamp(0.0, 1.0)
    preensao = torch.tanh(
        _forca_das_palmas(env, sensores_palma, asset_cfg) / _forca_ref(env, mu))
    # ⚠ ZERO NO BOTAR (spec §6.6.2 item 2; g1_poc: "ligado no botar, pagaria 2,0/s para
    # NÃO botar"). `postura_ereta` é `rampa × unload` e zera junto, sem linha própria.
    return (descarga * preensao * _valida(env, nome_do_comando)
            * _fora_do_botar(env, nome_do_comando))


def postura_ereta(env, nome_do_comando: str, sensores_palma: tuple[str, ...],
                  sensor_apoio: str, mu: float,
                  pelve_alvo: float, pelve_piso: float,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rampa na pelve × preensão × descarga. Paga por erguer SEM agachar.

    ⚠ É o termo que impede o robô de satisfazer o alvo DESCENDO até a caixa. O alvo já
    tem z absoluto, o que remove o atalho de baixar o alvo; este termo remove o atalho
    de baixar o CORPO para encurtar o alcance.

    ⚠ A rampa é de DOIS LADOS (`clamp(0, 1)`): zero abaixo do piso, um acima do alvo, e
    linear no meio. Sem o clamp superior, esticar-se além do alvo pagaria cada vez
    mais, e o robô aprenderia a ficar na ponta dos pés.

    ⚠ E ela é MULTIPLICADA pela descarga, não somada. Somado, o robô colheria a rampa
    só por ficar de pé sem tocar a caixa — que é exatamente o que ele já faz de graça.

    ⚠ A PREENSÃO CONTINUA NO PRODUTO, e ela mudou de lugar em 28/08: o porteiro de
    preensão passou para dentro do `unload`. Multiplicar por ela aqui de novo daria
    `preensão²`, que aperta a rampa sem acrescentar informação.
    """
    z = (env.scene["robot"].data.root_link_pos_w[:, 2]
         - env.scene.env_origins[:, 2])
    rampa = ((z - pelve_piso) / max(pelve_alvo - pelve_piso, 1e-6)).clamp(0.0, 1.0)
    descarga = unload(env, nome_do_comando, sensor_apoio,
                      sensores_palma, mu, asset_cfg)
    # ⚠ o `unload` já traz o `VALIDA` E a preensão desde 28/08; não multiplicar
    # nenhum dos dois de novo (daria VALIDA² e preensão²).
    return rampa * descarga


def load(env, nome_do_comando: str, sensor_apoio: str) -> torch.Tensor:
    """`(1 − descarga) × perto × VALIDA`, só em BOTAR (spec `g1-limpo-dois-bits.md`
    §2.7, mudança v3→v3.1: VOLTA).

    ⚠⚠ POR QUE VOLTA. Sem ele, nada paga por `apoiada`: `unload` e `postura_ereta`
    pagam por NÃO apoiar (o gate `_fora_do_botar` os zera dentro do BOTAR), e
    `renda_congelada` só congela no FECHO terminal — antes dele, pairar a 1 cm do
    alvo valia o mesmo que apoiar de verdade. `load` é o incentivo que falta: ele
    paga pela caixa ASSENTADA (`descarga -> 0`) e PERTO do alvo, e é a máscara que o
    `g1_poc` já tinha.

    ⚠ `perto`, e não a distância crua: reusa `AlvoCaixaCmd._perto` (spec §2.3,
    revisão item 29) — o MESMO limiar `tol_pos` do fecho, uma fonte só.

    ⚠ `descarga` é a MESMA conta do `unload` (`1 − F_apoio/(m·g)`, clamp[0,1]):
    apoiada de verdade, `F_apoio -> m·g` e `descarga -> 0`; pairando, `F_apoio -> 0`
    e `descarga -> 1`.

    ⚠ `_fora_do_botar` é o gate que `unload`/`squeeze` já usam para ZERAR dentro do
    BOTAR; `load` usa o COMPLEMENTO — ele só existe DENTRO do BOTAR.
    """
    from g1_limpo.comando import BOTAR
    t = _t(env, nome_do_comando)
    ids = torch.arange(env.num_envs, device=env.device)
    f = torch.norm(env.scene[sensor_apoio].data.force, dim=-1).squeeze(-1)
    peso = env.limpo_massa * 9.81
    descarga = (1.0 - f / peso.clamp(min=1e-6)).clamp(0.0, 1.0)
    perto = t._perto(ids).float()
    dentro_do_botar = 1.0 - _fora_do_botar(env, nome_do_comando)
    return (1.0 - descarga) * perto * _valida(env, nome_do_comando) * dentro_do_botar


class velocidade_por_regime:
    """O `variable_posture` do fabricante, com ERRO DE VELOCIDADE em vez de POSIÇÃO
    (G2, spec `g1-limpo-lento-e-estavel.md` §3): três regimes de vmax por padrão de
    nome de junta, resolvidos por `resolve_matching_names_values` no `__init__`, como
    o molde faz.

    ⚠⚠ A MEDIÇÃO QUE JUSTIFICA O TERMO (spec §0, sonda `mede_vel_junta.py`,
    `model_4999` do `bloco9`, 32 envs, 600 passos, CPU): o robô se move MAIS RÁPIDO
    parado com a caixa do que correndo. p90 por junta, em rad/s:

        manipulação parada   3,13
        locomoção parada     0,43
        andando              2,00
        correndo             2,50

    A locomoção parada recebe o MESMO comando zero e produz 7,3× menos. A diferença é
    o `rastreio_por_elo`: ele paga por comando zero na locomoção e, sem este termo,
    nada cobrava o excesso de velocidade na manipulação parada.

    ⚠ O REGIME VEM DO COMANDO, não da velocidade medida: `total = ‖cmd[:2]‖ +
    |cmd[2]|`, lido de `command_name` — o `twist`, e não o `alvo_caixa`. Em todo elo
    de manipulação `comando._zera_twist_nos_parados` escreve zero no `twist`, portanto
    `total = 0 < walking_threshold` SEMPRE, e o regime é `standing` — O REGIME JÁ É O
    GATE DA TAREFA. Não acrescente gate por `limpo_twist_zerado` nem por `VALIDA`.

    ⚠ `walking_threshold = 0,05` e `running_threshold = 1,5`, os MESMOS do molde
    (`variable_posture`, `mjlab/tasks/velocity/mdp/rewards.py:437-438`) — não viram
    knob: um segundo lugar com o mesmo número é como o `std_standing` do `pose`
    deriva em silêncio num upgrade.

    ⚠⚠ RETORNA `1 − exp(−média(v²/vmax²))`, e NÃO a forma positiva do molde —
    correção medida na revisão de 2026-09-08. A forma positiva `exp(−média(v²/vmax²))`
    vale 1,0 com o robô PARADO, em TODO env, e com peso positivo isso pagaria 2,0/s de
    RENDA GRÁTIS que entra direto no piso da estátua (`rastreio_por_elo`: medido
    8,265/s no PEGAR contra 3,863/s no ANDAR, parado ganhando por 43%). A forma
    complementar tem a MESMA derivada e paga ZERO parado: ela cobra o excesso de
    velocidade, não premia a ausência dele — o mesmo idioma do `contato_mesa` deste
    arquivo (positivo em [0, 1]; o peso NEGATIVO, no `knobs`, é quem faz dela
    penalidade).

    ⚠ MÉDIA sobre as juntas, e não produto: um produto de 29 gaussianas colapsa para
    qualquer vmax — o mesmo defeito medido no `PosturaPorElo` para posição.
    """

    def __init__(self, cfg, env):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        _, joint_names = asset.find_joints(asset_cfg.joint_names)

        _, _, vel_max_standing = resolve_matching_names_values(
            data=cfg.params["vel_max_standing"], list_of_strings=joint_names)
        self.vel_max_standing = torch.tensor(
            vel_max_standing, device=env.device, dtype=torch.float32)

        _, _, vel_max_walking = resolve_matching_names_values(
            data=cfg.params["vel_max_walking"], list_of_strings=joint_names)
        self.vel_max_walking = torch.tensor(
            vel_max_walking, device=env.device, dtype=torch.float32)

        _, _, vel_max_running = resolve_matching_names_values(
            data=cfg.params["vel_max_running"], list_of_strings=joint_names)
        self.vel_max_running = torch.tensor(
            vel_max_running, device=env.device, dtype=torch.float32)

    def __call__(self, env, vel_max_standing, vel_max_walking, vel_max_running,
                 asset_cfg: SceneEntityCfg, command_name: str,
                 walking_threshold: float = 0.05,
                 running_threshold: float = 1.5) -> torch.Tensor:
        del vel_max_standing, vel_max_walking, vel_max_running  # resolvidos no __init__

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold)
                       & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        vmax = (self.vel_max_standing * standing_mask.unsqueeze(1)
               + self.vel_max_walking * walking_mask.unsqueeze(1)
               + self.vel_max_running * running_mask.unsqueeze(1))

        v = asset.data.joint_vel[:, asset_cfg.joint_ids]
        return 1.0 - torch.exp(-torch.mean((v / vmax) ** 2, dim=1))


class renda_congelada:
    """A SOMA dos `termos` dependentes de elo, CONGELADA em número no fecho (spec P3).

    ⚠ CONGELA A SOMA, e não os termos vivos contra o alvo velho. Um elo que fecha muda
    o alvo dos termos que ele mede — depois do `CARREGAR` fechar, a caixa tem de sair
    do peito, e termos ainda vivos contra o peito puniriam o elo novo por progredir. A
    soma em número, uma vez, não tem esse problema: ela só cresce.

    ⚠⚠ DEPENDE DE `_step_reward`, atributo PRIVADO do `RewardManager` do mjlab — que
    guarda `peso × valor` por termo, por passo, na ordem do dict — e de ser o ÚLTIMO
    termo de `cfg.rewards`: só assim a soma lida no passo `t` é a dos termos JÁ
    computados neste mesmo passo, e ainda não sobrescrita pelo elo novo (a troca de elo
    roda em `command_manager.compute`, DEPOIS de `reward_manager.compute` — mjlab
    `manager_based_rl_env.py`). O `smoke` fixa os dois; um upgrade que renomeie o
    buffer ou reordene o dict falha no smoke, e não no treino.

    ⚠⚠ OS ÍNDICES SÃO RESOLVIDOS NO PRIMEIRO `__call__`, e não no `__init__`. Em
    `__init__` o `RewardManager` deste PRÓPRIO termo ainda está se construindo:
    `env.reward_manager` só passa a existir DEPOIS que `RewardManager.__init__`
    retorna (`manager_based_rl_env.py:329`), e como este termo é UM DOS do próprio
    `reward_manager`, ler `env.reward_manager` no `__init__` explode com
    `AttributeError` — MEDIDO. Resolver no primeiro `__call__` ainda resolve uma vez só
    (memoizado), só que mais tarde.
    """

    def __init__(self, cfg, env):
        self._termos = cfg.params["termos"]
        self._idx: list[int] | None = None
        z = torch.zeros(env.num_envs, device=env.device)
        self.soma_anterior, self.congelado = z.clone(), z.clone()
        self.fechos_anterior = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, nome_do_comando: str, termos) -> torch.Tensor:
        if self._idx is None:
            self._idx = [env.reward_manager.active_terms.index(n) for n in self._termos]
        t = _t(env, nome_do_comando)
        # ⚠⚠ A ORDEM 2→4 É O QUE GARANTE que se congela a soma do passo ANTERIOR ao
        # fecho, e não a do passo do próprio fecho (que já lê o elo novo): (1) detecta
        # o fecho pela SUBIDA do contador; (2) soma o QUE JÁ ESTAVA em `soma_anterior`
        # — do passo passado — por cima do congelado; (3) atualiza o contador; (4) só
        # DEPOIS relê `_step_reward` para o PRÓXIMO passo.
        fechou_agora = t._fechos > self.fechos_anterior
        self.congelado = self.congelado + torch.where(
            fechou_agora, self.soma_anterior, torch.zeros_like(self.soma_anterior))
        self.fechos_anterior = t._fechos.clone()
        self.soma_anterior = env.reward_manager._step_reward[:, self._idx].sum(dim=-1)
        # ⚠ SEM `× _valida` (spec dois-bits §2.6): com esperas ENTRE elos (a cadeia
        # agora tem uma antes de cada um), `× _valida` zerava a renda ganha em toda
        # espera — o congelado existe justamente para pagar DURANTE elas.
        return self.congelado

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.soma_anterior[env_ids] = 0.0
        self.congelado[env_ids] = 0.0
        self.fechos_anterior[env_ids] = 0
