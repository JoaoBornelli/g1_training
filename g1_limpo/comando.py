"""O comando do objetivo: ONDE a caixa deve ficar, e com que orientação.

É a ÚNICA fonte de verdade do alvo. O visualizador não reimplementa nada — ele
importa este módulo e desenha o que este termo publica. O `_debug_vis_impl` mora
aqui pela mesma razão: se o desenho vivesse no inspetor, ele seria uma segunda fonte
e mentiria no dia em que as duas divergissem.

OS CINCO ELOS, e o alvo de cada um é uma COISA DIFERENTE:

    ANDAR       uma VELOCIDADE (o twist), não um ponto. A mobília está a +5 m.
    REORIENTAR  uma ORIENTAÇÃO. A caixa fica onde está; o que muda é a face pedida.
    PEGAR       IDÊNTICO ao CARREGAR. O que difere é o twist: aqui ele é ZERO.
    CARREGAR    x,y no peito RELATIVOS ao robô; z ABSOLUTO na altura de trabalho.
    BOTAR       um ponto LATERAL num TOPO NOVO, com o teto travado no fundo da caixa.

⚠ Os ids dos elos são os MESMOS slots do one-hot da especificação. Uma numeração só.

LAYOUT DO COMANDO — canais novos entram sempre POR ÚLTIMO, para que uma migração de
checkpoint seja um APPEND de colunas e nunca uma inserção no meio:

    [0:3]  ALVO    posição alvo da caixa, em MUNDO
    [3:6]  FACE    normal da face pedida, em MUNDO (unitária)
    [6]    ANG     ângulo pedido, em radianos
    [7]    VALIDA  1,0 se o objetivo da caixa está ativo; 0,0 no `ANDAR`
    [8]    ELO     o elo corrente, como float

⚠ O `PEGAR` e o `CARREGAR` pedem O MESMO PONTO — a âncora do peito. A diferença é o
REFERENCIAL, e ela é o desenho:

    `carregar`  RELATIVO ao robô, recalculado a cada passo: a caixa acompanha o peito
                enquanto ele anda.
    `pegar`     CONGELADO em mundo no resample: **agachar não move o alvo**, portanto
                erguer a caixa até o peito é a única forma de satisfazer.

Se o `pegar` fosse relativo, o robô satisfaria agachando até a caixa. E ele era
ABSOLUTO E FIXO em `z = (0,78; 0,85)` até 25/08 — herdado da skill Lift, que tinha a
laje travada em 0,55. Com a laje variando de 0,55 a 0,04 por nível, aquele número
fazia "erguer" valer 0,13 m no nível 0 e 0,71 m no nível 6.

ESCOPO DESTA FASE (F0/F1): o alvo de CADA elo, e o desenho. O elo é FORÇADO por knob.
A máquina de elo — a troca automática quando o elo fecha, e as cadeias de 2 —
entra na F4 e ESTENDE este termo: ela vai chamar o mesmo `_aplica_elo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply

from g1_limpo.curriculo import garante_elo, garante_nivel

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["AlvoCaixaCmd", "AlvoCaixaCmdCfg", "FACE_AXES",
           "ALVO", "FACE", "ANG", "VALIDA", "ELO", "DIM",
           "ANDAR", "REORIENTAR", "PEGAR", "CARREGAR", "BOTAR", "ELOS", "elo_por_nome",
           "CADEIAS",
           "TwistComRazaoDeMarcha", "TwistComRazaoDeMarchaCfg"]

# --- o layout, por nome. Nenhum índice solto no resto do pacote. ---
ALVO = slice(0, 3)
FACE = slice(3, 6)
ANG = 6
VALIDA = 7
ELO = 8
DIM = 9

# --- os elos. Mesma numeração dos slots do one-hot. ---
ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3, 4
ELOS = ("andar", "reorientar", "pegar", "carregar", "botar")

# --- as cadeias de elo (F4). O teto é DERIVADO (`_TETO_ELOS`), nunca redigitado. ---
# índice 0: cadeia de 1 elo (PEGAR, já treina desde F3)
# índice 1, 2: cadeias de 2 elos
# índice 3: (PEGAR, CARREGAR, BOTAR) — pegar, SEGURAR PARADO, botar (spec §6.5). O
#           controlador de campo nunca manda BOTAR a partir de PEGAR; ele passa por
#           CARREGAR com v = 0. A cadeia treina exatamente isso.
CADEIAS = (
    (PEGAR,),
    (REORIENTAR, PEGAR),
    (PEGAR, CARREGAR),
    (PEGAR, CARREGAR, BOTAR),
)

# ⚠ `ANDAR` NÃO É CADEIA. Um env de locomoção recebe isto, e `n_elos_da_cadeia`
# devolve 1 para ele: não há 2º elo para avançar.
CADEIA_NENHUMA = -1

# o 1º elo de cada cadeia, e o comprimento de cada uma. Derivados de `CADEIAS`, nunca
# redigitados — uma tabela paralela escrita à mão sai de sincronia no dia em que uma
# cadeia mudar.
_PRIMEIRO_ELO = torch.tensor([c[0] for c in CADEIAS], dtype=torch.long)
_N_ELOS = torch.tensor([len(c) for c in CADEIAS], dtype=torch.long)
# CADEIAS achatada em (n_cadeias, teto_de_elos), com -1 no que não existe
_TETO_ELOS = max(len(c) for c in CADEIAS)
_ELO_EM = torch.full((len(CADEIAS), _TETO_ELOS), -1, dtype=torch.long)
for _i, _c in enumerate(CADEIAS):
    for _j, _e in enumerate(_c):
        _ELO_EM[_i, _j] = _e

# ⚠ A CADEIA DE SEGURAR PARADO (spec §6.5): aquela em que o CARREGAR é seguido do BOTAR.
# Nela o CARREGAR tem twist ZERO e fecha por `perto` sustentado pela espera sorteada, em
# vez de `andou`. DERIVADA de `CADEIAS`, e o índice 3 não aparece no corpo do termo — uma
# tabela paralela escrita à mão sai de sincronia no dia em que uma cadeia mudar.
_SEGURA_PARADO = torch.tensor(
    [any(c[i] == CARREGAR and c[i + 1] == BOTAR for i in range(len(c) - 1))
     for c in CADEIAS], dtype=torch.bool)


def elo_por_nome(nome: str) -> int:
    try:
        return ELOS.index(nome.strip().lower())
    except ValueError:
        raise SystemExit(f"elo desconhecido: {nome!r}. Use um de {ELOS}.") from None


# As 6 faces da caixa, em coordenada LOCAL dela. Ficam aqui como DOCUMENTAÇÃO: a face
# pedida NÃO é sorteada entre elas.
#
# ⚠ O `reorientar` pede que UMA face — a marcada, `cfg.face_alvo_b` — fique normal ao
# robô. Qualquer uma das 6 chega à frente por composição de QUARTOS DE VOLTA (±90° em
# X, Y ou Z), portanto o robô precisa aprender 6 primitivas e nada mais. A dificuldade
# mora na ORIENTAÇÃO DE NASCIMENTO da caixa, e ela é sorteada em
# `eventos.orientacao_de_nascimento`.
FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)

# Raio de referência do alcance, a partir da pelve. É REFERÊNCIA, não limiar de
# recompensa.
#
# ⚠ DERIVAÇÃO, porque eu errei este número antes. Eu o pusera em 0,50 m, confundindo-o
# com o `box_xy = 0,50` medido no repositório — mas AQUELE número é a distância de
# SPAWN da caixa que rendeu só 19% de pega, e não um raio de alcance.
#
# O número defensável vem do envelope de spawn com que a skill Lift FECHOU a tarefa:
# a caixa nasce em x até `0,32 + 0,20 = 0,52`, y até ±0,18, e no nível mais baixo o
# centro dela fica em z ≈ 0,14. Com a pelve em z = 0,80:
#
#     sqrt(0,52² + 0,18² + (0,80 − 0,14)²) = 0,86 m
#
# Portanto o robô comprovadamente operou com alcances de até ~0,86 m. O raio abaixo é
# esse envelope, arredondado para baixo.
ALCANCE_R = 0.85

_MAGENTA = (0.90, 0.20, 0.90, 1.00)
_CIANO = (0.20, 0.90, 0.90, 1.00)
_AMARELO = (0.95, 0.85, 0.20, 1.00)
_VERMELHO = (0.95, 0.25, 0.25, 1.00)
_VERDE = (0.25, 0.90, 0.35, 1.00)
_CINZA = (0.60, 0.60, 0.65, 0.35)
_BRANCO = (1.00, 1.00, 1.00, 0.10)


@dataclass(kw_only=True)
class AlvoCaixaCmdCfg(CommandTermCfg):
    # A ÂNCORA DO PEITO, no frame da BASE. Alvo dos DOIS elos que seguram a caixa; a
    # diferença é só o REFERENCIAL — `carregar` relativo ao robô, `pegar` congelado
    # em mundo.
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.15)
    # ⚠ o z do alvo é ABSOLUTO nos dois elos que seguram: agachar não baixa o alvo
    altura_carregar: float = 0.95
    # os elos que exigem o robô PARADO. O twist deles é forçado a ZERO, e é isso —
    # e não a forma do alvo — que impede o robô de andar com a caixa.
    elos_parados: tuple[int, ...] = (1, 2, 4)      # REORIENTAR, PEGAR, BOTAR
    nome_do_twist: str = "twist"
    # ⚠ O SENSOR DE APOIO É `apoio_caixa`, o contato caixa<->laje declarado em
    # `cena.sensores()`. Uma versão anterior deste campo dizia `contact_caixa_laje`,
    # que NÃO EXISTE, e a leitura vinha dentro de um `try/except` cujo fallback era
    # `apoiada = True`: o `BOTAR` fechava com `perto & alinhado` apenas, em silêncio.
    nome_sensor_apoio: str = "apoio_caixa"
    # ⚠ O limiar NÃO é absoluto: ele é uma FRAÇÃO do peso da caixa. Um limiar fixo de
    # 2 N significaria "apoiada" com carga de 1 kg (9,8 N) e "no ar" com 5 kg mal
    # encostada. A caixa está apoiada quando a laje carrega metade do peso dela.
    fracao_do_peso_apoiada: float = 0.5
    # a tolerância que conta como "na condição de fechamento", em metros e radianos
    tol_pos: float = 0.10
    tol_ang_deg: float = 25.0
    # altura mínima da pelve para considerar "de pé" (não agachado)
    pelve_alvo: float = 0.75
    # alvo do BOTAR — lateral, num topo novo
    botar_x: tuple[float, float] = (0.30, 0.40)
    botar_y: tuple[float, float] = (-0.12, 0.12)
    botar_topo_piso: float = 0.30
    botar_topo_teto: float = 0.80
    botar_folga_laje: float = 0.05
    # geometria de que o termo precisa para mover a laje
    afasta_z: float = 5.0
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    prateleira_meia_z: float = 0.02
    prateleira_meia_xy: float = 0.30
    # o topo da laje APOIADA no chão. É o piso físico do `BOTAR`.
    prateleira_topo_piso: float = 0.04
    caixa_meia_z: float = 0.10
    # ⚠ A meia-aresta entra no kernel de alcance: a distância medida é até a
    # SUPERFÍCIE da caixa, não até o centro. Ver `dist_palma_caixa`.
    caixa_meia_aresta: float = 0.10
    # a face MARCADA, no frame da caixa. Constante: é sempre ela que o `reorientar`
    # pede normal ao robô. A dificuldade está na ORIENTAÇÃO DE NASCIMENTO da caixa.
    face_alvo_b: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    # os sítios das palmas, para a distância que define o σ
    sitios_palma: tuple[str, ...] = ("left_palm", "right_palm")
    # a JANELA DE ESPERA, em segundos, sorteada por episódio. Ver `knobs.Alvo`.
    espera_s: tuple[float, float] = (0.3, 1.0)
    # os SENSORES de palma, para armar o `caixa_largada`. Só o campo `found` é lido:
    # a pergunta é "as duas palmas já tocaram a caixa neste episódio", e ela é
    # booleana por natureza. A FORÇA é assunto do `squeeze`, que é contínuo.
    sensores_palma: tuple[str, ...] = ("palma_E", "palma_D")

    # ⚠ O σ NÃO É UM NÚMERO: ele é a DISTÂNCIA INICIAL daquele env, vezes este fator.
    # Ver o bloco de σ no `__init__` e a §4.2b da spec. Medido: com σ fixo de 0,10 a
    # 0,339 m o kernel vale 1e−05 e a DERIVADA É ZERO — o robô não tem pista de onde
    # ir, e foi isto que travou o `g1_poc`.
    #
    # ⚠ PRÉ-REGISTRADO: se o alcance não aparecer na F3, este fator vai a 1,5. É o
    # PRIMEIRO e ÚNICO número a mover, e NUNCA o peso.
    sigma_fator: float = 1.0
    sigma_min: float = 0.08          # metros
    sigma_ori_min: float = 0.20      # radianos (~11°)

    # o elo. `None` = todos em `PEGAR` (o único que a F0/F1 treina).
    elo_forcado: int | None = None
    # ⚠ NUNCA igual à duração do episódio. Com `(20, 20)` o `time_left` do comando
    # cruza zero no passo 999 e o `time_out` da terminação só dispara no 1000: o
    # resample rodava UM PASSO antes do fim e zerava o sucesso do episódio. O nível
    # lia sucesso 0 em TODO episódio que chegava ao time_out. A meta é 1 resample por
    # episódio, e quem resampleia é o RESET.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = True

    # --- F4: máquina de elo ---
    cadeia_forcada: int | None = None    # índice em CADEIAS. Inspetor e play.
    prob_por_nivel: tuple[tuple[float, ...], ...] = ()
    sustenta_pegar_s: float = 0.5
    sustenta_outros_s: float = 0.3
    carregar_s: float = 1.5
    carregar_dist_m: float = 0.50

    def build(self, env: "ManagerBasedRlEnv") -> "AlvoCaixaCmd":
        return AlvoCaixaCmd(self, env)


class AlvoCaixaCmd(CommandTerm):
    cfg: AlvoCaixaCmdCfg

    def __init__(self, cfg: AlvoCaixaCmdCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self.caixa: Entity = env.scene["box"]
        self.prateleira: Entity = env.scene["table"]
        self.robot: Entity = env.scene["robot"]

        d, n = self.device, self.num_envs
        self._command = torch.zeros(n, DIM, device=d)
        self._face_b = torch.tensor(cfg.face_alvo_b, device=d)
        # --- A DIREÇÃO PEDIDA PARA A FACE MARCADA, e se ela é VIVA ou CONGELADA.
        #
        # ⚠ DOIS PEDIDOS DIFERENTES, e confundi-los apagou o termo. No `REORIENTAR` a
        # direção pedida é VIVA: "vire a face para o robô", recalculada todo passo. Em
        # todos os outros elos ela é CONGELADA na normal ATUAL no instante em que o elo
        # abre, e aí o termo pergunta "a caixa girou desde então?" — isto é, ele paga
        # por ERGUER SEM TORCER.
        #
        # Até 28/08 a direção era viva em TODO elo, e o `precise_ori` (peso 1,0) ficava
        # inerte: no nível 0 a caixa nasce alinhada (`voltas_max = 0`, desalinho <= 15°)
        # e `sigma_ori` tem piso de 0,20 rad, portanto o termo nascia satisfeito com
        # derivada ~zero. Pior, o alvo se movia com o ROBÔ: andar em volta da caixa
        # mudava o termo sem tocar nela.
        #
        # O `g1_poc` faz exatamente esta separação (`g1_poc/comando.py:246`) e declara
        # o motivo: o `precise_ori` congelado é o que substitui o `box_shake` de −0,15.
        self._face_alvo_w = torch.zeros(n, 3, device=d)
        self._face_viva = torch.zeros(n, dtype=torch.bool, device=d)
        self._elo = torch.full((n,), PEGAR, dtype=torch.long, device=d)
        # ⚠⚠ O `_pendente` existe por causa de uma armadilha MEDIDA em 25/08.
        #
        # No reset o command manager roda DEPOIS dos eventos que reposicionam a caixa
        # e a laje, mas os buffers de `data` das entidades **ainda não foram
        # recomputados**. Portanto tudo o que depende de POSE é lixo aqui.
        #
        # O que isso quebrou, medido: o teto do `BOTAR` é
        # `min(fundo_da_caixa − folga, teto_do_knob)`, e com a pose obsoleta o fundo
        # deu negativo — o teto colapsou no piso e o topo saiu 0,300 nos OITO envs.
        # Só o `maximum(teto, piso)` impediu uma laje enterrada, e o clamp real nunca
        # aconteceu. O alvo do `ANDAR` saiu 0,000 pelo mesmo motivo.
        #
        # Solução: marcar o env como PENDENTE no resample e concluir a parte
        # dependente de pose no primeiro `_update_command`, quando a pose está fresca.
        self._pendente = torch.zeros(n, dtype=torch.bool, device=d)

        # ⚠ Onde a base estava quando o elo corrente ABRIU. É o que permite ao
        # `CARREGAR` exigir DESLOCAMENTO em vez de só tempo — sem isso ele fechava sem o
        # robô andar um centímetro. Ver `knobs.Cadeia.carregar_dist_m`.
        self._pos_no_elo = torch.zeros(n, 3, device=d)

        # ⚠ O SUSTAIN do CARREGAR de segurar parado (spec §6.5 item 3): a espera
        # sorteada do MESMO knob `espera_s`, por env. Só a cadeia marcada em
        # `_SEGURA_PARADO` o lê; as outras usam `carregar_s`.
        self._segurar = torch.zeros(n, device=d)

        # ---------------------------------------------------------- F4: máquina de elo
        # Os buffers que controlam o avanço entre elos.
        self._cadeia = torch.zeros(n, dtype=torch.long, device=d)
        self._passo = torch.zeros(n, dtype=torch.long, device=d)  # 0 .. _TETO_ELOS-1
        self._sust = torch.zeros(n, dtype=torch.float, device=d)  # cronômetro em s
        self.avancou = torch.zeros(n, dtype=torch.bool, device=d)
        self.fechou = torch.zeros(n, dtype=torch.bool, device=d)

        # Métricas publicadas para o log (seção 4 do contrato F4)
        # ⚠ Todas são float para que `reset` possa tirar a média
        z = torch.zeros(n, dtype=torch.float, device=d)
        self.metrics["sucesso"] = z.clone()
        self.metrics["passo_final"] = z.clone()
        self.metrics["avancos"] = z.clone()
        self.metrics["fatia_cadeia"] = z.clone()

        # ---------------------------------------------------------- os σ POR ENV
        # ⚠ ELES NÃO SÃO KNOBS. Cada um é a DISTÂNCIA INICIAL daquele env, medida no
        # instante em que o elo abre. É a decisão de maior consequência da F3, e ela
        # vem de medição (spec §4.2b):
        #
        #   a palma nasce a 0,339 m da caixa (mín 0,211, máx 0,481). Com σ FIXO de
        #   0,10 o kernel `exp(−d²/σ²)` vale 1e−05 ali, E A DERIVADA É ZERO. O robô
        #   move a mão 1 cm para perto e nada muda; 1 cm para longe e nada muda. Não
        #   existe pista de onde ir. Foi isto que travou o `g1_poc`, e não uma
        #   preferência do robô por ficar parado.
        #
        # Com `σ = d₀`, todo env nasce em `exp(−1) = 0,368` com derivada
        # `2/d₀ × 0,368`: 3,49 no env mais perto e 1,53 no mais longe. Vivo nos dois
        # extremos, e sem número mágico.
        #
        # ⚠ ELES NÃO ENTRAM NA OBSERVAÇÃO, e isso é decisão. Publicar o σ diria à
        # política "este env é fácil/difícil", e ela condicionaria a ação à forma da
        # recompensa em vez de à tarefa. σ é moldagem, não estado do mundo.
        self.sigma_alcance = torch.full((n,), cfg.sigma_min, device=d)
        self.sigma_trazer = torch.full((n,), cfg.sigma_min, device=d)
        self.sigma_ori = torch.full((n,), cfg.sigma_ori_min, device=d)

        # os sítios das palmas, resolvidos UMA vez
        self._ids_palma, _ = self.robot.find_sites(list(cfg.sitios_palma))

        # ---------------------------------------------- a ARMA do `caixa_largada`
        # ⚠ "As duas palmas já tocaram a caixa NESTE episódio." A terminação de caixa
        # largada é armada por ela e nunca antes: no reset a caixa está na laje e as
        # palmas estão longe, que é exatamente a condição de `escapou`. Sem a arma,
        # todo episódio começaria terminando.
        #
        # ⚠ Ela mora AQUI, e não na terminação, porque aqui existe escopo de episódio:
        # o `_resample_command` roda no reset e zera o buffer. Um termo de terminação é
        # uma função sem `reset`, e o estado dele vazaria de um episódio para o outro.
        # ---------------------------------------------- a JANELA DE ESPERA (02/09)
        # ⚠ Segundos RESTANTES, por env. Enquanto > 0 num elo de manipulação, o bit
        # `VALIDA` fica em zero: os sete incentivos pagam nada e o elo não fecha. Ver
        # `knobs.Alvo.espera_s` para o porquê e para a origem no `g1_poc`.
        self._espera = torch.zeros(n, device=d)
        env.limpo_aguardando = torch.zeros(n, device=d)
        # ⚠ A ESPERA FINAL (spec §6.6): depois do fecho do BOTAR, o publicado é ANDAR
        # até o fim do episódio; o interno segue BOTAR. `soltou` desarma o `escapou` da
        # terminação e liga o `largou` da recompensa.
        self._soltou = torch.zeros(n, dtype=torch.bool, device=d)
        env.limpo_soltou = self._soltou.float()
        # ⚠ O ELO INTERNO, publicado para o crítico e para as recompensas de caixa. É
        # uma REFERÊNCIA a `_elo`, que só é escrito in-place (`self._elo[ids] = ...`),
        # portanto ela nunca fica obsoleta; `_update_command` a republica por segurança.
        env.limpo_elo_interno = self._elo

        self._pegou = torch.zeros(n, dtype=torch.bool, device=d)
        env.limpo_ids_palma = self._ids_palma
        # ⚠ Publica ZEROS aqui, e não o resultado de `_publica_pegou`: no `__init__` os
        # buffers de sensor ainda não foram preenchidos. A leitura real começa no
        # primeiro `_update_command`.
        env.limpo_pegou = self._pegou.float()

    def _aplica_espera(self) -> None:
        """Decrementa a espera e escreve o PUBLICADO e o `VALIDA` (spec §6.0).

        ⚠⚠ TUDO É RECALCULADO DO INTERNO, e não lido do próprio canal. Uma versão
        anterior fazia `where(aguardando, 0, self._command[:, VALIDA])` — DESTRUTIVO: no
        passo seguinte lia o zero que ela mesma tinha escrito, e o bit nunca voltava a 1.
        Medido no smoke em 02/09.

            publicado = ANDAR   se aguardando ∨ soltou, senão o interno
            VALIDA    = (interno ≠ ANDAR) ∧ ¬aguardando

        ⚠ A espera FINAL (`soltou`) publica ANDAR mas NÃO zera o VALIDA: os incentivos
        do estado "caixa apoiada no alvo" continuam pagando depois do fecho do BOTAR. É
        o que fecha o buraco da renda (spec §6.6.1). A v12 dizia o contrário e estava
        errada.

        ⚠ Publica `env.limpo_aguardando` e `env.limpo_soltou` para as métricas e para a
        terminação. Sem elas, "o robô não espera" e "a janela não existe" leem igual.
        """
        self._espera.sub_(self._env.step_dt).clamp_(min=0.0)
        aguardando = self._espera > 0.0
        self._env.limpo_aguardando.copy_(aguardando.float())
        self._env.limpo_soltou = self._soltou.float()
        self._env.limpo_elo_interno = self._elo
        publica_andar = aguardando | self._soltou
        self._command[:, ELO] = torch.where(
            publica_andar, torch.full_like(self._elo, ANDAR), self._elo).float()
        base = (self._elo != ANDAR).float()
        self._command[:, VALIDA] = base * (~aguardando).float()

    def _publica_pegou(self) -> None:
        """Atualiza a arma e a publica em `env.limpo_pegou`.

        ⚠ Monotônica DENTRO do episódio (`|=`), e zerada só no `_resample_command`.
        Soltar a caixa para reposicionar não desarma a terminação — se desarmasse,
        largar de vez deixaria de terminar.

        ⚠ Republica o tensor todo passo em vez de guardar uma referência: `.float()`
        cria um tensor NOVO, portanto uma publicação única no `__init__` congelaria o
        valor em zero para sempre.
        """
        tocou = None
        for nome in self.cfg.sensores_palma:
            achou = self._env.scene[nome].data.found
            assert achou is not None, f"sensor '{nome}' precisa do field 'found'."
            aqui = (achou > 0).any(dim=-1)
            tocou = aqui if tocou is None else (tocou & aqui)
        if tocou is not None:
            # ⚠ SÓ ARMA COM O OBJETIVO ATIVO (spec §6.3). Na espera inicial um toque
            # por exploração armaria `escapou` com as palmas longe, e o episódio
            # morreria por ter esperado. Lê `_espera` direto, e não o `VALIDA`, porque
            # este método roda ANTES de `_aplica_espera` na passada.
            ativo = (self._elo != ANDAR) & (self._espera <= 0.0)
            self._pegou |= tocou & ativo
        self._env.limpo_pegou = self._pegou.float()

    # -------------------------------------------------------------- o contrato
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def elo_de(self, ids: torch.Tensor) -> torch.Tensor:
        """O elo corrente daqueles envs.

        ⚠ Ele lê o BUFFER `_elo`, e não reconstrói o elo a partir de
        `CADEIAS[cadeia][passo]`. Duas razões, e as duas são defeitos que existiam aqui:

        1. `CADEIAS[cad]` com `cad = CADEIA_NENHUMA = −1` devolve a ÚLTIMA cadeia em
           Python, portanto um env de `ANDAR` reportava elo `BOTAR` em silêncio. E o
           `__init__.py` registra tasks de inspeção para os CINCO elos, três dos quais
           caem em `−1`.
        2. Reconstruir de duas fontes cria a chance de elas divergirem. O `_elo` é a
           fonte, e é o que o one-hot e o gate de recompensa leem.
        """
        return self._elo[ids]

    def n_elos_da_cadeia(self, ids: torch.Tensor) -> torch.Tensor:
        """Quantos elos tem a cadeia daqueles envs. **1** para quem não tem cadeia.

        ⚠ `CADEIAS[cad]` com `cad = CADEIA_NENHUMA = −1` devolve a ÚLTIMA cadeia em
        Python. Um env de `ANDAR` reportava "2 elos" e elo `BOTAR`, em silêncio. É o
        mesmo defeito que o `_avanca_elo_force` já guardava, e que ficou de fora destes
        dois acessores — que são justamente os que o inspetor usa na tabela ANTES/DEPOIS.
        """
        cad = self._cadeia[ids]
        n = torch.ones_like(cad)
        tem = cad >= 0
        if bool(tem.any()):
            n[tem] = _N_ELOS.to(cad.device)[cad[tem]]
        return n

    def forca_avanco(self, ids: torch.Tensor) -> None:
        """Força o avanço imediato de elo, sem esperar sustain.

        Destinado ao inspetor (`inspeciona.py`) e play (`play.py`).
        """
        self._avanca_elo_force(ids)

    def recebe_tarefa(self, ids: torch.Tensor, elo_novo: int) -> None:
        """Entrega uma tarefa de manipulação AO VIVO a quem estava no `ANDAR`.

        ⚠⚠ SÓ PARA O VISUALIZADOR. Ela existe para simular o DEPLOY: o robô está de pé
        com comando de velocidade, um operador manda "pega a caixa", e o robô transita.
        No TREINO isso não acontece — o elo é sorteado no reset e nunca troca no meio
        (`resampling_time_range = 1e9`, e o `_avanca_elo` só caminha DENTRO de uma
        cadeia; nenhuma cadeia vai de `ANDAR` a `PEGAR`).

        ⚠ ELA NÃO POSICIONA A MOBÍLIA, e isso é do chamador. No `ANDAR` a laje foi
        mandada a +5 m com a caixa em cima (`_aplica_elo`, ramo `ANDAR`), e o ramo
        `PEGAR` NÃO a traz de volta — ele só ancora o alvo. Quem chama tem de rodar o
        `posiciona_cena` ANTES, senão a tarefa entregue é "pegue uma caixa a 5 m".
        O `eventos.troca_elo_no_viewer` faz as duas coisas na ordem certa.

        ⚠ A JANELA DE ESPERA É RE-ARMADA, e é o ponto do exercício: a tarefa chega, e o
        objetivo só liga 0,3 a 1,0 s depois. É a transição que se quer olhar.

        ⚠ O QUE TEM DE SER ZERADO, e cada um por um motivo medido:
          · `_pegou`  — a arma do `caixa_largada`. Sem zerar, uma pega anterior deixaria
            a terminação armada com a caixa longe, e o episódio morreria na entrega.
          · `_sust`   — o cronômetro do elo. Herdado, o elo novo nasceria quase fechado.
          · `fechou`  — senão o `_avanca_elo` ignora o env para sempre.
          · `_pos_no_elo` — a âncora de deslocamento do `CARREGAR`, que tem de ser a
            pose de AGORA e não a do reset.

        ⚠ A pose está FRESCA aqui (isto roda num evento de intervalo, dentro do passo),
        portanto o `_aplica_elo` e o `_recalcula_sigmas` podem ser chamados direto —
        sem a passada do `_pendente`, que existe só para o reset.
        """
        if len(ids) == 0:
            return
        d = self.device
        self._elo[ids] = int(elo_novo)
        # a cadeia compatível com o elo entregue. `cadeia_forcada` vence, como no reset.
        if self.cfg.cadeia_forcada is not None:
            self._cadeia[ids] = int(self.cfg.cadeia_forcada)
        else:
            compat = (_PRIMEIRO_ELO == int(elo_novo)).nonzero().flatten()
            self._cadeia[ids] = int(compat[0]) if len(compat) else CADEIA_NENHUMA
        self._aplica_elo(ids)
        self._recalcula_sigmas(ids)
        self._pos_no_elo[ids] = self.robot.data.root_link_pos_w[ids]
        self._pegou[ids] = False
        self._sust[ids] = 0.0
        self.fechou[ids] = False
        lo, hi = self.cfg.espera_s
        self._espera[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)
        self._segurar[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)
        # ⚠ E O BIT CAI NO MESMO INSTANTE. O `_aplica_elo` acima escreveu `VALIDA = 1`
        # (é o que ele faz em elo de manipulação), e o `_aplica_espera` só corrige isso
        # no passo SEGUINTE — este método roda num evento de intervalo, fora da passada
        # do `command_manager`. Sem esta linha o objetivo ficaria ligado por um passo
        # com a janela já armada, e o instante da entrega seria exatamente o que se
        # está tentando olhar.
        self._command[ids, VALIDA] = 0.0
        # ⚠ E O PUBLICADO JÁ NASCE ANDAR (spec §6.4): a espera acabou de ser armada, e o
        # `_aplica_espera` só a veria no passo seguinte.
        self._soltou[ids] = False
        self._command[ids, ELO] = float(ANDAR)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        d = self.device
        n = len(env_ids)

        # ⚠ A ARMA DA TERMINAÇÃO `caixa_largada` ZERA AQUI, e ela precisa zerar em
        # algum lugar com escopo de EPISÓDIO. Sem isso um episódio que pegou a caixa
        # armaria todos os seguintes daquele env, e o reset — em que a caixa está na
        # laje e as palmas estão longe — dispararia `escapou` na hora.
        self._pegou[env_ids] = False
        self._soltou[env_ids] = False

        # O ELO. Desde a F2 ele é SORTEADO POR ENV, e o sorteio mora no currículo
        # (`curriculo.sorteia_elo`) porque a ordem de reset é currículo -> eventos ->
        # comando: o reset de pose da base já precisou do elo antes de chegarmos aqui.
        #
        # ⚠ Ler o buffer em vez de sortear aqui não é detalhe de organização. Se o
        # comando sorteasse, o evento de pose teria sorteado OUTRA COISA no mesmo
        # reset, e metade dos envs de manipulação nasceria de costas para a mobília —
        # sem erro e sem log.
        #
        # O `elo_forcado` continua vencendo: é o que o inspetor e o `play` usam.
        if self.cfg.elo_forcado is not None:
            self._elo[env_ids] = int(self.cfg.elo_forcado)
        else:
            self._elo[env_ids] = garante_elo(self._env, ANDAR)[env_ids]

        # ⚠ A JANELA DE ESPERA, sorteada, e DEPOIS do elo — ela depende dele: no `ANDAR`
        # a espera é ZERO. Sortear antes leria o elo do episódio ANTERIOR, que é a mesma
        # classe de defeito que a ordem do currículo já custou a este projeto.
        #
        # ⚠ E ela é sorteada AQUI, e não na passada do `_pendente`: ela não depende de
        # pose nenhuma, só do elo. Adiá-la para lá custaria um passo de janela.
        lo, hi = self.cfg.espera_s
        _sorteio_espera = lo + (hi - lo) * torch.rand(n, device=d)
        _anda = self._elo[env_ids] == ANDAR
        self._espera[env_ids] = torch.where(
            _anda, torch.zeros_like(_sorteio_espera), _sorteio_espera)
        # o "segurar parado" da cadeia 3 usa a MESMA faixa; sorteio próprio, por env
        self._segurar[env_ids] = lo + (hi - lo) * torch.rand(n, device=d)

        # --- F4: A CADEIA, CONDICIONADA NO ELO QUE O CURRÍCULO JÁ SORTEOU ---
        #
        # ⚠⚠ ISTO É O PONTO MAIS DELICADO DA F4, e uma versão anterior o inverteu: ela
        # sorteava a cadeia e depois SOBRESCREVIA `self._elo` com o 1º elo dela. Como o
        # 1º elo de três das quatro cadeias é o `PEGAR`, TODOS os envs viravam `PEGAR` —
        # e a fatia de locomoção da F2 (95%) era APAGADA. O módulo inteiro existe para
        # não entregar as transições à manipulação cedo demais, e aquele `=` fazia
        # exatamente isso, sem uma linha de log.
        #
        # A ordem correta é a inversa: quem decide se o env é de LOCOMOÇÃO ou de
        # MANIPULAÇÃO é o currículo (`sorteia_elo`, a fatia). A cadeia só escolhe QUAL
        # transição praticar, DENTRO da manipulação — e ela tem de COMEÇAR no elo que o
        # currículo já sorteou. Uma cadeia que começa noutro elo seria uma segunda
        # decisão sobre a mesma coisa.
        #
        # `ANDAR` não é cadeia: ele recebe `CADEIA_NENHUMA`.
        elo_atual = self._elo[env_ids]
        if self.cfg.cadeia_forcada is not None:
            # inspetor/play: a cadeia manda, e o elo passa a ser o 1º dela
            self._cadeia[env_ids] = int(self.cfg.cadeia_forcada)
            self._elo[env_ids] = int(CADEIAS[int(self.cfg.cadeia_forcada)][0])
        elif len(self.cfg.prob_por_nivel) > 0:
            # ⚠ VETORIZADO. O laço Python sobre `env_ids` que estava aqui rodava 4096
            # iterações a cada reset em lote.
            tab = torch.tensor(self.cfg.prob_por_nivel, device=d, dtype=torch.float)
            nivel = garante_nivel(self._env)[env_ids].clamp(max=tab.shape[0] - 1)
            linha = tab[nivel]                                    # (n, 4)
            # só as cadeias cujo 1º elo é o elo sorteado
            compat = _PRIMEIRO_ELO.to(d).unsqueeze(0) == elo_atual.unsqueeze(1)
            pesos = linha * compat.float()
            soma = pesos.sum(dim=-1)
            tem = soma > 0.0
            # envs sem cadeia compatível (o `ANDAR`) ficam com CADEIA_NENHUMA
            seguro = torch.where(tem.unsqueeze(-1), pesos,
                                 torch.ones_like(pesos))
            escolha = torch.multinomial(seguro, num_samples=1).squeeze(1)
            self._cadeia[env_ids] = torch.where(
                tem, escolha, torch.full_like(escolha, CADEIA_NENHUMA))
        else:
            # F0-F3: não há cadeia. O elo é o que o currículo disse.
            self._cadeia[env_ids] = CADEIA_NENHUMA

        # Zerar os buffers de avanço
        self._passo[env_ids] = 0
        self._sust[env_ids] = 0.0
        self.avancou[env_ids] = False
        self.fechou[env_ids] = False

        # ⚠ NÃO se sorteia face nem ângulo aqui. A face pedida é CONSTANTE (a
        # marcada), e a dificuldade do `reorientar` vem da ORIENTAÇÃO DE NASCIMENTO da
        # caixa, sorteada em quartos de volta pelo evento `posiciona_cena`.
        self._aplica_elo(env_ids)
        # a parte dependente de POSE fica pendente para o 1º passo (ver `_pendente`)
        self._pendente[env_ids] = True

    def _update_command(self) -> None:
        todos = torch.arange(self.num_envs, device=self.device)

        # ⚠ PRIMEIRO os pendentes: aqui a pose já está fresca.
        pend = todos[self._pendente]
        if len(pend):
            self._aplica_elo(pend, so_pose=True)
            # ⚠ O σ é recalculado AQUI e em nenhum outro lugar do reset: é o único
            # ponto em que a pose da palma, da caixa e do alvo está fresca. Calculá-lo
            # no `_resample_command` daria σ de pose obsoleta — o mesmo defeito que o
            # `_pendente` existe para consertar.
            self._recalcula_sigmas(pend)
            # ⚠ A âncora de deslocamento do `CARREGAR`, na mesma passada em que a pose
            # está fresca. No `_resample_command` ela seria de pose obsoleta.
            self._pos_no_elo[pend] = self.robot.data.root_link_pos_w[pend]
            self._pendente[pend] = False

        # a caixa gira durante o episódio, portanto a normal da face acompanha. O
        # ÂNGULO pedido é fixo no episódio; a NORMAL não é.
        self._atualiza_face(todos)
        # o alvo do CARREGAR é ancorado na BASE, portanto ele anda com o robô.
        self._alvo_ancorado_na_base(todos[self._elo == CARREGAR])
        # o alvo do REORIENTAR e do ANDAR é a própria caixa (no ANDAR ele é inerte,
        # porque `VALIDA = 0` — mas um alvo em zero no log é armadilha de leitura).
        segue = todos[(self._elo == REORIENTAR) | (self._elo == ANDAR)]
        if len(segue):
            self._command[segue, ALVO] = self.caixa.data.root_link_pos_w[segue]
        # o alvo do PEGAR é o MESMO do CARREGAR: relativo ao robô em x,y
        self._alvo_ancorado_na_base(todos[self._elo == PEGAR])
        # e o que impede o robô de andar no `pegar` é o twist em ZERO
        self._zera_twist_nos_parados()

        # a arma do `caixa_largada`, ANTES do avanço de elo: uma cadeia que avança não
        # desarma a terminação, porque a caixa continua sendo a mesma caixa.
        self._publica_pegou()

        # ⚠⚠ A JANELA DE ESPERA, e ela roda ANTES do `_avanca_elo` de propósito: durante
        # a espera o elo NÃO pode fechar. O `_fecha_elo_corrente` lê o `VALIDA`, e é
        # este bloco que o zera — na ordem invertida, um `REORIENTAR` fecharia DENTRO da
        # espera, porque ali o alvo É a própria caixa e `perto` é trivial.
        self._aplica_espera()

        # --- F4: AVANÇO DE ELO ---
        # Deve rodar APÓS a atualização do alvo e da face, porque usa pose fresca.
        self._avanca_elo()

    def _update_metrics(self) -> None:
        pass

    def _segura_parado(self, ids: torch.Tensor) -> torch.Tensor:
        """Máscara: o env está no CARREGAR da cadeia de SEGURAR PARADO (spec §6.5).

        ⚠ `_cadeia == −1` no `ANDAR`: o `tem` impede indexar a tabela com −1, que em
        Python devolveria a ÚLTIMA cadeia — o mesmo defeito que `n_elos_da_cadeia`
        já guarda.
        """
        cad = self._cadeia[ids]
        tem = cad >= 0
        seg = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if bool(tem.any()):
            seg[tem] = _SEGURA_PARADO.to(self.device)[cad[tem]]
        return seg & (self._elo[ids] == CARREGAR)

    def _zera_twist_nos_parados(self) -> None:
        """Força o comando de velocidade a ZERO nos elos que exigem o robô parado.

        ⚠ É ISTO que impede o robô de andar com a caixa no `pegar`, no `reorientar` e
        no `botar` — e não a forma do alvo. Decisão do dono em 25/08, e é o que o
        `g1_poc` faz (`comando.py:826`), cuja manipulação funcionou.

        ⚠ ORDEM: o `twist` é inserido no dict de comandos ANTES deste termo (ele vem
        do molde do fabricante), e o dict é ordenado por inserção. Portanto o
        `compute` dele já rodou quando este roda, e a escrita aqui não é sobrescrita
        no mesmo passo.

        ⚠ A escrita é DESTRUTIVA no buffer, e é de propósito: qualquer métrica que
        gateie por "comando ativo" passa a NÃO contar estes passos, que é o correto —
        eles são passos de comando zero de verdade, e não passos mascarados na
        leitura.
        """
        parados = torch.isin(self._elo, torch.tensor(self.cfg.elos_parados,
                                                     device=self.device))
        # ⚠ REGRA POR CADEIA (spec §6.5 item 2): o CARREGAR de segurar parado também
        # tem twist zero. Na cadeia 2 o CARREGAR continua andando.
        parados = parados | self._segura_parado(
            torch.arange(self.num_envs, device=self.device))
        if not bool(parados.any()):
            return
        tw = self._env.command_manager.get_term(self.cfg.nome_do_twist)
        tw.vel_command_b[parados] = 0.0

    def _fecha_elo_corrente(self, ids: torch.Tensor) -> torch.Tensor:
        """Retorna BoolTensor indicando quais elos fecharam.

        Condição de fechamento POR ELO (tabela da F4):
            REORIENTAR: perto & alinhado
            PEGAR:      perto & alinhado & de pé
            CARREGAR:   perto
            BOTAR:      perto & alinhado & apoiada
        """
        if len(ids) == 0:
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)

        c = self.cfg
        d = self.device

        # ⚠⚠ O OBJETIVO TEM DE ESTAR ATIVO. Sem isto o elo fecharia DURANTE a janela de
        # espera — e no `REORIENTAR` o alvo É a própria caixa, portanto `perto` é
        # trivialmente verdadeiro e ele fecharia no passo ZERO, sempre. Era o que fazia
        # `avancos = 0,43` conviver com `sucesso = 0,0000`: a cadeia avançava de graça no
        # primeiro elo e travava no `PEGAR`.
        ativo = self._command[ids, VALIDA] > 0.5

        # Condições "perto" e "alinhado"
        dist_alvo = torch.norm(
            self.caixa.data.root_link_pos_w[ids] - self._command[ids, ALVO], dim=-1)
        perto = dist_alvo <= c.tol_pos

        erro_ang = self._command[ids, ANG]
        alinhado = erro_ang <= torch.deg2rad(torch.tensor(c.tol_ang_deg))

        # Condição "de pé" — altura da pelve acima de `pelve_alvo`
        de_pe = self.robot.data.root_link_pos_w[ids, 2] >= c.pelve_alvo

        # Condição "apoiada" — a LAJE carrega o peso da caixa.
        #
        # ⚠ SEM `try/except`. Uma versão anterior lia um sensor inexistente
        # (`contact_caixa_laje`), pelo método errado (`robot.find_sites` — um SENSOR
        # não é um SÍTIO, e ele vive na CENA, não no robô), dentro de um `try` cujo
        # `except` deixava `apoiada = True`. Resultado: o `BOTAR` fechava com
        # `perto & alinhado` apenas, sem nunca conferir se a caixa estava apoiada, e
        # sem uma linha de erro. Se o sensor não existir, ISTO TEM DE EXPLODIR.
        forca = torch.norm(
            self._env.scene[c.nome_sensor_apoio].data.force, dim=-1).squeeze(-1)[ids]
        peso = self._env.limpo_massa[ids] * 9.81
        apoiada = forca >= c.fracao_do_peso_apoiada * peso

        # Condições por elo
        elo_corrente = self._elo[ids]
        fecha = torch.zeros(len(ids), dtype=torch.bool, device=d)

        for elo_tipo in (REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = elo_corrente == elo_tipo
            if not bool(m.any()):
                continue

            if elo_tipo == REORIENTAR:
                fecha[m] = (perto[m] & alinhado[m])
            elif elo_tipo == PEGAR:
                fecha[m] = (perto[m] & alinhado[m] & de_pe[m])
            elif elo_tipo == CARREGAR:
                # ⚠ `perto & ANDOU`, e não `perto` sozinho. `perto` é subconjunto da
                # condição do `pegar` sobre o MESMO alvo, portanto o `carregar` fechava
                # no instante em que o `pegar` fechava, sem o robô sair do lugar.
                andou = torch.norm(
                    self.robot.data.root_link_pos_w[ids][m, :2]
                    - self._pos_no_elo[ids][m, :2], dim=-1) >= c.carregar_dist_m
                # ⚠ REGRA POR CADEIA (spec §6.5 item 3): em SEGURAR PARADO não há
                # `andou` — o twist é zero e a condição é `perto`, sustentado pela espera
                # sorteada (ver `_avanca_elo`). Sem `perto`, o BOTAR começaria com a
                # caixa em qualquer lugar.
                segura = self._segura_parado(ids)[m]
                fecha[m] = torch.where(segura, perto[m], perto[m] & andou)
            elif elo_tipo == BOTAR:
                fecha[m] = (perto[m] & alinhado[m] & apoiada[m])

        # ⚠ O `ativo` entra NO FIM, e sobre todos os elos de uma vez. Pôr o `& ativo`
        # dentro de cada ramo seria quatro lugares para esquecer um.
        return fecha & ativo

    def _avanca_elo(self) -> None:
        """Avança de elo quando a condição de fechamento é satisfeita por sustain.

        Roda a cada passo DENTRO de `_update_command`, com pose fresca.
        Não há reset nem resample — o one-hot acompanha o elo sem corte de episódio.

        Acumula `_sust` enquanto a condição vale, zera quando não vale.
        Quando `_sust >= sustain_do_elo`, avança ou marca fechou.
        """
        d = self.device
        todos = torch.arange(self.num_envs, device=d)
        # ⚠ O dt vem do ENV, e não de um literal. `1.0/50.0` estava escrito aqui à mão;
        # ele acerta hoje e passa a mentir no dia em que a decimação ou o timestep
        # mudar — e o cronômetro de sustentação erraria em silêncio.
        dt = self._env.step_dt

        # Verificar qual elo está aberto (não fechou ainda)
        nao_fechou = todos[~self.fechou]
        if len(nao_fechou) == 0:
            return

        # Condição de fechamento do elo corrente
        fecha = self._fecha_elo_corrente(nao_fechou)

        # Acumular sustain ou zerar
        self._sust[nao_fechou] = torch.where(
            fecha,
            self._sust[nao_fechou] + dt,
            torch.zeros_like(self._sust[nao_fechou])
        )

        # Determinar o sustain alvo de cada elo
        elo_corrente = self._elo[nao_fechou]
        sustain_alvo = torch.zeros(len(nao_fechou), device=d)
        for elo_tipo in (REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = elo_corrente == elo_tipo
            if bool(m.any()):
                if elo_tipo == PEGAR:
                    sustain_alvo[m] = self.cfg.sustenta_pegar_s
                elif elo_tipo == CARREGAR:
                    # ⚠ em SEGURAR PARADO o sustain É a espera sorteada (spec §6.5)
                    segura = self._segura_parado(nao_fechou)[m]
                    seg_s = self._segurar[nao_fechou][m]
                    sustain_alvo[m] = torch.where(
                        segura, seg_s, torch.full_like(seg_s, self.cfg.carregar_s))
                else:  # REORIENTAR, BOTAR
                    sustain_alvo[m] = self.cfg.sustenta_outros_s

        # ⚠ SÓ QUEM TEM CADEIA pode avançar. Sem este filtro, um env de `ANDAR` entrava
        # em `_avanca_elo_force` A CADA PASSO: o laço por elo acima não cobre o `ANDAR`,
        # portanto o `sustain_alvo` dele ficava no zero do `torch.zeros`, e
        # `0 >= 0` é True. Com 95% dos envs em locomoção isso era ~95% dos envs entrando
        # na função todo passo de física, e escrevendo `fatia_cadeia = 0` por cima.
        tem_cadeia = self._cadeia[nao_fechou] >= 0
        deve_avancar = (self._sust[nao_fechou] >= sustain_alvo) & tem_cadeia

        ids_avancar = nao_fechou[deve_avancar]
        if len(ids_avancar) > 0:
            self._avanca_elo_force(ids_avancar)

        # ⚠ AS MÉTRICAS SÃO ESCRITAS TODO PASSO, PARA TODOS OS ENVS. Antes elas só eram
        # escritas dentro do `_avanca_elo_force`, isto é, só no instante de um avanço —
        # logo um env de cadeia que NUNCA fechasse o 1º elo nunca escrevia o seu
        # `fatia_cadeia = 1`, e o `Metrics/alvo_caixa/fatia_cadeia` ficava perto de zero
        # no começo do treino. A escada da F4 leria isso e diagnosticaria "as cadeias
        # não estão sendo sorteadas", que é o oposto do que estaria acontecendo.
        cad_t = self._cadeia
        n_el = torch.ones_like(cad_t)
        tem_t = cad_t >= 0
        if bool(tem_t.any()):
            n_el[tem_t] = _N_ELOS.to(d)[cad_t[tem_t]]
        self.metrics["fatia_cadeia"][:] = (n_el > 1).float()
        self.metrics["passo_final"][:] = self._passo.float()

    def _avanca_elo_force(self, ids: torch.Tensor) -> None:
        """Avança aqueles envs UM elo, ou fecha a cadeia se já era o último.

        Usado pelo `_avanca_elo` (avanço natural, por sustain) e pelo `forca_avanco`
        (inspetor e play).

        ⚠ VETORIZADO, e não é otimização prematura. A versão anterior tinha um laço
        Python sobre os envs que chamava `_aplica_elo(ids[i:i+1])` UM POR UM — e o
        `_aplica_elo` já itera sobre os 5 elos por dentro. Com 4096 envs isso são
        ~20 mil iterações de Python num passo de física.

        ⚠ E ela lia `CADEIAS[cad]` com `cad = −1` para os envs de locomoção, que em
        Python devolve a ÚLTIMA cadeia. Um env de `ANDAR` era tratado como
        `(PEGAR, BOTAR)`, em silêncio. O `tem` abaixo é o que impede isso.
        """
        if len(ids) == 0:
            return
        d = self.device
        cad = self._cadeia[ids]
        passo = self._passo[ids]

        tem = cad >= 0                       # `ANDAR` não tem cadeia
        n_elos = torch.ones_like(cad)
        if bool(tem.any()):
            n_elos[tem] = _N_ELOS.to(d)[cad[tem]]
        prox = passo + 1
        pode = tem & (prox < n_elos)

        # --- os que AVANÇAM ---
        m = ids[pode]
        if len(m):
            np_ = prox[pode]
            self._passo[m] = np_
            self._elo[m] = _ELO_EM.to(d)[cad[pode], np_]
            self._sust[m] = 0.0
            # ⚠ UMA chamada em lote. E `so_pose=False` porque aqui a pose JÁ está
            # fresca: o avanço roda no `_update_command`, não no reset.
            self._aplica_elo(m, so_pose=False)
            self._recalcula_sigmas(m)
            # ⚠ o elo NOVO começa a contar deslocamento daqui
            self._pos_no_elo[m] = self.robot.data.root_link_pos_w[m]

        # --- os que FECHAM a cadeia ---
        # ⚠ `tem & ~pode`, e não `~pode`. Sem o `tem`, um env de `ANDAR` em que o
        # inspetor chamasse `forca_avanco` seria marcado como SUCESSO de manipulação.
        f = ids[tem & ~pode]
        if len(f):
            self.fechou[f] = True
            self.metrics["sucesso"][f] = 1.0
            # ⚠ A ESPERA FINAL (spec §6.6): quem fecha no BOTAR publica ANDAR daqui até
            # o fim do episódio, NO MESMO PASSO do fecho — sem esperar o `_aplica_espera`
            # do passo seguinte. O interno segue BOTAR.
            solta = f[self._elo[f] == BOTAR]
            if len(solta):
                self._soltou[solta] = True
                self._command[solta, ELO] = float(ANDAR)

        self.avancou[ids] = pode
        # ⚠ Só o CONTADOR de avanços mora aqui — ele é por evento. O `passo_final` e o
        # `fatia_cadeia` são ESTADO, e são escritos todo passo no `_avanca_elo`.
        self.metrics["avancos"][ids] += pode.float()

    # ------------------------------------------------------ o alvo, por elo
    def _aplica_elo(self, ids: torch.Tensor, *, so_pose: bool = False) -> None:
        """Escreve o alvo do elo corrente, e move a laje quando o elo pede.

        ⚠ `so_pose` NÃO É LIDO PELO CORPO, e isso é declarado em vez de prometido. O
        docstring anterior dizia "refaz APENAS o que depende de pose", o que era falso:
        a função refaz TUDO. Hoje isso é correto e desejado — o `topo` e o alvo do
        `BOTAR` sorteados no reset vêm de pose OBSOLETA e TÊM de ser descartados e
        re-sorteados na passada do `_pendente`.
        O parâmetro fica na assinatura como documentação do intento do chamador; quem
        acrescentar aqui um sorteio que deva sobreviver à passada do `_pendente` tem de
        passar a LÊ-LO.

        ⚠ A F4 chama exatamente esta função no avanço de elo — e lá ela pode rodar
        com `so_pose=False`, porque no avanço a pose JÁ está fresca (o `_avanca_elo`
        roda no `_update_command`, e não no reset).
        """
        if len(ids) == 0:
            return
        d = self.device
        c = self.cfg
        self._command[ids, ELO] = self._elo[ids].float()
        origem = self._env.scene.env_origins[ids]

        # ⚠ O REGIME DA FACE, e ele é por elo. Só o `REORIENTAR` pede uma direção
        # VIVA; os outros congelam a normal do instante da abertura e passam a medir
        # "a caixa girou desde então?". Escrito ANTES do laço porque o `_congela_face`
        # precisa da normal fresca, e esta função roda na passada do `_pendente`.
        self._face_viva[ids] = self._elo[ids] == REORIENTAR
        self._congela_face(ids[self._elo[ids] != REORIENTAR])

        for elo in (ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR):
            m = ids[self._elo[ids] == elo]
            if len(m) == 0:
                continue
            k = len(m)
            org = self._env.scene.env_origins[m]

            if elo == ANDAR:
                # não há alvo de caixa. O alvo é o TWIST, e ele é outro comando.
                #
                # ⚠ O alvo publicado fica na PRÓPRIA CAIXA, e não na pelve. Ler a
                # pose do robô aqui devolveria valor OBSOLETO (o command manager roda
                # no reset, e os buffers de `data` do robô só são recomputados no
                # forward seguinte). O desenho lê a pelve VIVA, no passo, onde ela
                # está correta.
                self._command[m, VALIDA] = 0.0
                self._laje_para(m, c.afasta_z, sobe_caixa=True)
                self._command[m, ALVO] = self.caixa.data.root_link_pos_w[m]

            elif elo == REORIENTAR:
                # a caixa NÃO se move: o pedido é de atitude
                self._command[m, VALIDA] = 1.0
                self._command[m, ALVO] = self.caixa.data.root_link_pos_w[m]

            elif elo == PEGAR:
                # ⚠ EXATAMENTE o mesmo alvo do `CARREGAR`. A diferença entre os dois
                # elos é o COMANDO DE VELOCIDADE: no `pegar` ele é zero (o robô ergue
                # parado), no `carregar` ele é ativo (o robô ergue e anda).
                #
                # Antes de 25/08 este alvo era congelado em mundo, para o robô não
                # "alcançar o alvo andando". O twist em zero resolve isso melhor, e
                # sem criar dois referenciais para a mesma âncora.
                self._command[m, VALIDA] = 1.0
                self._alvo_ancorado_na_base(m)

            elif elo == CARREGAR:
                # a caixa JÁ está nas mãos (no treino, o elo anterior a pegou), então
                # aqui só a LAJE sobe.
                self._command[m, VALIDA] = 1.0
                self._laje_para(m, c.afasta_z, sobe_caixa=False)
                self._alvo_ancorado_na_base(m)

            elif elo == BOTAR:
                self._command[m, VALIDA] = 1.0
                # ⚠ O TETO EFETIVO é o fundo da caixa SEGURADA menos a folga. O knob
                # `botar_topo_teto` é só um teto do teto: sem este clamp a laje
                # nasceria DENTRO da caixa.
                # ⚠⚠ O LIMITE FÍSICO VENCE O KNOB, e isto foi um DEFEITO MEDIDO em
                # 2026-08-26 ao estender o inspetor para os 7 níveis.
                #
                # A versão anterior fazia `piso = botar_topo_piso` (0,30) e depois
                # `teto = maximum(teto, piso)`. Com a caixa segurada BAIXA — o que
                # acontece nos níveis altos, onde a laje nasce a 0,04 m — o
                # `fundo − folga` cai abaixo de 0,30, e aquele `maximum` SOBREPUNHA o
                # limite físico com o knob: a laje nascia em 0,300 contra um
                # `fundo − folga` de 0,067. Dentro da caixa, exatamente o que a spec
                # avisa. O meu check da F4 não pegou porque rodava um nível só.
                #
                # Agora o PISO CEDE: ele é o valor desejado, mas nunca passa do teto
                # físico. O último recurso é a laje no chão (`prateleira_topo_piso`).
                fundo = self.caixa.data.root_link_pos_w[m, 2] - self._meia(m)[:, 2]
                teto = torch.clamp(fundo - c.botar_folga_laje, max=c.botar_topo_teto)
                piso = torch.clamp(
                    torch.full_like(teto, c.botar_topo_piso), max=teto)
                # ⚠ e nunca ENTERRADA: com o topo abaixo disto a laje atravessa o chão.
                piso = torch.maximum(
                    piso, torch.full_like(piso, c.prateleira_topo_piso))
                # ⚠ CASO DECLARADO: se a caixa está segurada MAIS BAIXA que a laje mais
                # fina possível, nenhum topo satisfaz as duas coisas. Aí a laje vai ao
                # chão, e o alvo fica acima do fundo da caixa — geometricamente
                # impossível de satisfazer, e é melhor declarar que violar em silêncio.
                teto = torch.maximum(teto, piso)
                topo = piso + (teto - piso) * torch.rand(k, device=d)
                self._laje_para(m, topo)
                # o alvo é LATERAL, em cima do topo novo. O frontal exigiria alcançar
                # por cima de 20 cm de tampo — defeito medido em 16/07.
                bx, by = c.botar_x, c.botar_y
                a = torch.zeros(k, 3, device=d)
                a[:, 0] = org[:, 0] + bx[0] + (bx[1] - bx[0]) * torch.rand(k, device=d)
                a[:, 1] = org[:, 1] + by[0] + (by[1] - by[0]) * torch.rand(k, device=d)
                a[:, 2] = topo + self._meia(m)[:, 2]
                self._command[m, ALVO] = a

        self._atualiza_face(ids)

    def _laje_para(self, ids: torch.Tensor, topo, *, sobe_caixa: bool = False) -> None:
        """Move a laje (mocap) para um topo. A pose é o CENTRO do corpo.

        `sobe_caixa=True` leva a CAIXA junto, apoiada no topo novo. É o que o `ANDAR`
        precisa: com a laje a +5 m e a caixa no chão, o robô tropeçaria nela.
        """
        k = len(ids)
        d = self.device
        c = self.cfg
        org = self._env.scene.env_origins[ids]
        topo_t = (torch.full((k,), float(topo), device=d)
                  if not torch.is_tensor(topo) else topo)
        pose = torch.zeros(k, 7, device=d)
        pose[:, 0] = org[:, 0] + c.prateleira_xy[0]
        pose[:, 1] = org[:, 1] + c.prateleira_xy[1]
        pose[:, 2] = topo_t - c.prateleira_meia_z
        pose[:, 3] = 1.0
        self.prateleira.write_mocap_pose_to_sim(pose, env_ids=ids)
        if sobe_caixa:
            pc = pose.clone()
            pc[:, 2] = topo_t + self._meia(ids)[:, 2]
            self.caixa.write_root_link_pose_to_sim(pc, env_ids=ids)
            self.caixa.write_root_link_velocity_to_sim(
                torch.zeros(k, 6, device=d), env_ids=ids)
        if hasattr(self._env, "limpo_topo"):
            self._env.limpo_topo[ids] = topo_t

    def _alvo_ancorado_na_base(self, ids: torch.Tensor) -> None:
        """O alvo do CARREGAR. O referencial é dividido POR EIXO:

            x, y   RELATIVOS ao robô, reescritos a cada passo — a caixa está nas mãos
                   e tem de acompanhá-lo horizontalmente.
            z      ABSOLUTO, a `altura_carregar`.

        ⚠ O z NÃO pode ser relativo. Se fosse, o robô satisfaria o alvo ANDANDO
        AGACHADO: o alvo desceria junto com a pelve e a caixa nunca precisaria subir.
        Com o z fixo, carregar exige manter a caixa na altura de trabalho — que é o
        comportamento pedido.
        """
        if len(ids) == 0:
            return
        d = self.device
        p = torch.tensor(self.cfg.peito_b, device=d).expand(len(ids), 3)
        base_p = self.robot.data.root_link_pos_w[ids]
        base_q = self.robot.data.root_link_quat_w[ids]
        a = base_p + quat_apply(base_q, p)
        a[:, 2] = self.cfg.altura_carregar
        self._command[ids, ALVO] = a

    def _meia(self, ids: torch.Tensor) -> torch.Tensor:
        """[k, 3] — a meia-aresta da caixa DE CADA ENV (spec §6.7).

        Lê `env.limpo_meia_aresta`, publicado pelo evento de startup `tamanho_caixa`. O
        knob `caixa_meia_aresta` é só o fallback de um env montado sem o evento.
        """
        meia = getattr(self._env, "limpo_meia_aresta", None)
        if meia is not None:
            return meia[ids]
        return torch.full((len(ids), 3), float(self.cfg.caixa_meia_aresta), device=self.device)

    def alvos_das_palmas(self, ids: torch.Tensor) -> torch.Tensor:
        """[k,2,3] — o ponto que CADA palma deve alcançar, em MUNDO.

        O alvo de cada palma é o centro da SUA face lateral. O offset gira com a
        caixa, portanto a pose pedida acompanha a orientação dela.

        ⚠ Ordem (esquerda, direita) = ordem de `cfg.sitios_palma`. A esquerda mira
        `+y` da caixa e a direita mira `−y`. É a convenção do `g1_poc`
        (`observacoes.alvos_das_palmas`), e ela casa com a geometria dos pads: o pad
        esquerdo fica em `y = −0,015` local e o direito em `y = +0,015`, portanto as
        duas palmas olham UMA PARA A OUTRA.
        """
        caixa = self.caixa.data.root_link_pos_w[ids]
        off = torch.zeros_like(caixa)
        off[:, 1] = self._meia(ids)[:, 1]
        off = quat_apply(self.caixa.data.root_link_quat_w[ids], off)
        return torch.stack((caixa + off, caixa - off), dim=1)

    def dist_palma_caixa(self, ids: torch.Tensor) -> torch.Tensor:
        """Distância MÉDIA das duas palmas às SUAS faces laterais, por env.

        ⚠ BIMANUAL E LATERAL, e as duas metades vêm de medição. Até 28/08 isto era
        `min` sobre as palmas contra a SUPERFÍCIE de uma esfera em volta do centro:

            d = ‖palma − centro‖.min(palmas) − meia_aresta

        Aquilo tem dois buracos, e o bloco 3 caiu nos dois. Com `min`, UMA mão satura
        o kernel e a segunda não tem gradiente nenhum — mas o `squeeze` exige as DUAS
        (ele é `min` das forças). A cadeia ficava sem ponte entre "uma mão encosta" e
        "as duas apertam", que é exatamente onde a run travou: `staged` parado no valor
        de nascimento e `squeeze` em 0,0002 depois de 3200 iterações.
        E com a esfera, tocar o TOPO, a FRENTE ou a BASE paga igual a tocar a lateral,
        portanto não existe gradiente para a pose de pega.

        O `g1_poc` já tinha consertado isto e escreveu o porquê
        (`g1_poc/observacoes.py:47`): "Com o centro, o `reaching` estagnava com UMA mão
        na face próxima, sem gradiente para o abraço."

        A MÉDIA é o que acopla as duas mãos: uma mão atrasada derruba o termo, portanto
        as duas se aproximam juntas. O máximo é a pose PRÉ-PEGA, com as palmas
        flanqueando a caixa — e é ela que torna o pad de DORSO geometricamente errado,
        que é como o `g1_poc` dispensa o `back_penalty` (`g1_poc/terminacoes.py:13`).

        ⚠ NÃO subtrai mais a meia-aresta: o alvo JÁ está na superfície. Subtrair de
        novo deixaria o kernel saturado antes do contato.
        """
        palmas = self.robot.data.site_pos_w[ids][:, self._ids_palma, :]
        alvos = self.alvos_das_palmas(ids)
        return torch.norm(palmas - alvos, dim=-1).mean(dim=1)

    def _recalcula_sigmas(self, ids: torch.Tensor) -> None:
        """σ = distância inicial × fator, com piso. Ver o bloco do `__init__`.

        ⚠ O PISO não é estética: um env que nasce com a palma colada na caixa teria
        σ ≈ 0, e o kernel viraria um pico impossível de sustentar — a recompensa
        desabaria ao primeiro milímetro de tremor.
        """
        if len(ids) == 0:
            return
        c = self.cfg
        d_palma = self.dist_palma_caixa(ids)
        d_alvo = torch.norm(
            self.caixa.data.root_link_pos_w[ids] - self._command[ids, ALVO], dim=-1)
        self.sigma_alcance[ids] = (d_palma * c.sigma_fator).clamp(min=c.sigma_min)
        self.sigma_trazer[ids] = (d_alvo * c.sigma_fator).clamp(min=c.sigma_min)
        # ⚠ O σ de ORIENTAÇÃO é o ÂNGULO inicial, em radianos — outra unidade, outro
        # piso. Com σ fixo de 0,40 rad um pedido de 90° dá `exp(−(1,57/0,40)²)` =
        # 2,0e−7, isto é zero: era a "sorte de nível 3+" medida no `g1_poc`.
        self._atualiza_face(ids)
        self.sigma_ori[ids] = (self._command[ids, ANG] * c.sigma_fator).clamp(
            min=c.sigma_ori_min)

    def _congela_face(self, ids: torch.Tensor) -> None:
        """Fixa a direção pedida na normal ATUAL da face marcada.

        Chamado no instante em que um elo que NÃO é o `REORIENTAR` abre. A partir daí
        o `precise_ori` mede o giro acumulado desde a abertura do elo — ele paga por
        erguer sem torcer, e não por apontar a face a lugar nenhum.
        """
        if len(ids) == 0:
            return
        fb = self._face_b.expand(len(ids), 3)
        self._face_alvo_w[ids] = quat_apply(
            self.caixa.data.root_link_quat_w[ids], fb)

    def _atualiza_face(self, ids: torch.Tensor) -> None:
        """Publica a DIREÇÃO DESEJADA e o ERRO angular da face marcada.

            FACE  a direção em que a face marcada DEVE apontar.
            ANG   o erro angular ATUAL, em radianos, entre a normal da face marcada e
                  essa direção. Zero = alinhada.

        ⚠ A DIREÇÃO PEDIDA TEM DOIS REGIMES, e é isso que dá função ao termo em todo
        elo. Ver o bloco de `_face_viva` no `__init__`:

            REORIENTAR   VIVA — da caixa para o robô, na horizontal, todo passo.
            os outros    CONGELADA na normal do instante em que o elo abriu.

        ⚠ O erro é o ângulo entre dois vetores 3D, e não uma rotação em torno de Z. É
        de propósito: se a caixa estiver TOMBADA, a normal da face marcada aponta para
        cima, e o erro tem de acusar isso — 90°, e não 0.
        """
        if len(ids) == 0:
            return
        k = len(ids)
        fb = self._face_b.expand(k, 3)
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], fb)

        para_o_robo = (self.robot.data.root_link_pos_w[ids]
                       - self.caixa.data.root_link_pos_w[ids])
        para_o_robo = para_o_robo.clone()
        para_o_robo[:, 2] = 0.0        # a direção pedida é HORIZONTAL
        viva = para_o_robo / para_o_robo.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ⚠ `where` e não indexação por máscara: os dois ramos são densos e do mesmo
        # tamanho, e assim não há um segundo caminho de escrita para manter em dia.
        desejada = torch.where(
            self._face_viva[ids].unsqueeze(-1), viva, self._face_alvo_w[ids])

        self._command[ids, FACE] = desejada
        cos = (normal_w * desejada).sum(-1).clamp(-1.0, 1.0)
        self._command[ids, ANG] = torch.acos(cos)

    # --------------------------------------------------------------- o desenho
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Desenha o que ESTE termo publica. Nada é recalculado."""
        import mujoco

        for i in visualizer.get_env_indices(self.num_envs):
            elo = int(self._elo[i])
            nome = ELOS[elo]
            nivel = int(garante_nivel(self._env)[i])
            caixa_p = self.caixa.data.root_link_pos_w[i].cpu().numpy()
            caixa_q = self.caixa.data.root_link_quat_w[i].cpu().numpy()
            pelve = self.robot.data.root_link_pos_w[i].cpu().numpy()
            alvo = self._command[i, ALVO].cpu().numpy()
            face = self._command[i, FACE].cpu().numpy()
            valida = float(self._command[i, VALIDA])

            # OS EIXOS DA CAIXA, do quatérnion real dela. X vermelho, Y verde, Z azul.
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, caixa_q)
            visualizer.add_frame(
                position=caixa_p, rotation_matrix=mat.reshape(3, 3),
                scale=0.22, axis_radius=0.008,
                label=f"[{nome}] nivel {nivel}")

            if elo == ANDAR:
                # O ALVO DO `ANDAR` É UMA VELOCIDADE, e ela vem do outro comando.
                tw = self._env.command_manager.get_command("twist")[i]
                v_b = torch.stack((tw[0], tw[1], torch.zeros_like(tw[0])))
                v_w = quat_apply(self.robot.data.root_link_quat_w[i:i + 1],
                                 v_b.unsqueeze(0))[0].cpu().numpy()
                visualizer.add_arrow(
                    start=pelve, end=pelve + v_w, color=_VERDE, width=0.020,
                    label=f"[andar] v_cmd {float(tw[0]):+.2f},{float(tw[1]):+.2f} m/s"
                          f"  wz {float(tw[2]):+.2f} rad/s")
                # o alvo de caixa está DESLIGADO neste elo
                visualizer.add_sphere(center=pelve + np.array([0, 0, 1.0]),
                                      radius=0.03, color=_CINZA,
                                      label="sem alvo de caixa (valida=0)")
                continue

            # A NORMAL ATUAL da face MARCADA (o que ela É) e a DIREÇÃO DESEJADA
            # (onde ela DEVE apontar). O erro é o ângulo entre as duas.
            fb = self._face_b.unsqueeze(0)
            n_at = quat_apply(self.caixa.data.root_link_quat_w[i:i + 1], fb)[0]
            n_at = n_at.cpu().numpy()
            erro = np.degrees(float(self._command[i, ANG]))
            voltas = int(getattr(self._env, "limpo_voltas",
                                 torch.zeros(1, device=self.device))[i]) \
                if hasattr(self._env, "limpo_voltas") else 0
            visualizer.add_arrow(
                start=caixa_p, end=caixa_p + n_at * 0.30, color=_VERDE, width=0.014,
                label=f"face MARCADA aponta aqui")
            visualizer.add_arrow(
                start=caixa_p, end=caixa_p + face * 0.30, color=_MAGENTA, width=0.012,
                label=f"DEVE apontar aqui  ·  erro {erro:.0f}°  ·  "
                      f"{voltas} quarto(s) de volta")

            # O ALVO
            rot = "  (o alvo É a caixa: pede-se ATITUDE)" if elo == REORIENTAR else ""
            anc = "  (ancorado na BASE)" if elo == CARREGAR else ""
            visualizer.add_sphere(center=alvo, radius=0.05, color=_CIANO,
                                  label=f"[{nome}] alvo{rot}{anc}")

            # QUANTO MOVER a caixa até o alvo
            d = float(np.linalg.norm(alvo - caixa_p))
            visualizer.add_arrow(
                start=caixa_p, end=alvo, color=_AMARELO, width=0.010,
                label=f"caixa->alvo {d:.3f} m  (dz {float(alvo[2]-caixa_p[2]):+.3f})")

            # O TOPO DA LAJE
            prat = self.prateleira.data.root_link_pos_w[i].cpu().numpy()
            topo = prat[2] + self.cfg.prateleira_meia_z
            visualizer.add_box(
                center=np.array([prat[0], prat[1], topo]),
                size=np.array([self.cfg.prateleira_meia_xy,
                               self.cfg.prateleira_meia_xy, 0.002]),
                mat=np.eye(3), color=_CINZA, label=f"topo da laje {topo:.3f} m")

            # O ALCANCE, a partir da pelve
            visualizer.add_sphere(center=pelve, radius=ALCANCE_R, color=_BRANCO,
                                  label=f"alcance ~{ALCANCE_R:.2f} m")
            d_alvo = float(np.linalg.norm(alvo - pelve))
            visualizer.add_arrow(
                start=pelve, end=alvo, width=0.006,
                color=_CIANO if d_alvo <= ALCANCE_R else _VERMELHO,
                label=f"pelve->alvo {d_alvo:.3f} m"
                      + ("" if d_alvo <= ALCANCE_R else "  FORA DO ALCANCE"))
            _ = valida


# =============================================================================
# A RÉGUA DA LOCOMOÇÃO
# =============================================================================
class TwistComRazaoDeMarcha(UniformVelocityCommand):
    """O twist do fabricante, com DUAS réguas a mais.

    **JUIZ (desde 27/08) — `eficiencia_min`.** Por SEGMENTO de comando:

        e_s = ⟨v_real · v̂_cmd⟩_s / ⟨‖v_cmd‖⟩_s ,     e o portão lê min(e_s)

    ⚠ POR QUE ELA SUBSTITUIU A `razao_marcha` NO PORTÃO. A razão é soma de NORMAS, e
    norma nunca cancela: ruído de média zero SEMPRE a infla. MEDIDO no bloco 1 — o `std`
    subiu de 0,43 para 0,61 (a manipulação entrou, exploração voltou a valer) e a razão
    caiu de 0,514 para 0,426 com DURAÇÃO (984 -> 988) e QUEDA (0,000 -> 0,167) PARADAS e
    o `play` determinístico andando bem. O portão congelou na banda morta e a rampa deu
    UM degrau em 1341 iterações: ele leu ruído de ação como incompetência.
    A projeção cancela o ruído (`Σ(ruído · v̂_cmd)` tem média zero, encolhe com 1/√N), e o
    corte por segmento impede o robô de compensar um segmento ruim com outro bom.

    **DIAGNÓSTICO — `razao_marcha`.** Fica no log, fora do portão, para que o bloco 2
    tenha as duas curvas lado a lado e o limiar novo possa ser calibrado contra medição.

        razao_marcha = 1 − Σ‖v_cmd_xy − v_xy‖ / Σ‖v_cmd_xy‖

    ⚠ POR QUE ADIMENSIONAL. O currículo de comando do fabricante (`command_vel`)
    ALARGA a faixa de velocidade ao longo do treino. Uma régua em m/s daria um DEGRAU
    na iteração em que a faixa abre, e o portão leria progresso onde só houve mudança
    de escala. Aqui as duas somas crescem juntas, e o degrau se cancela. MEDIDO: o
    currículo de comando corta só 17% da colheita da estátua em 10k iterações,
    portanto ele mexe a escala de verdade e essa imunidade não é teórica.

    ⚠ POR QUE ELA NASCE EM 0,0, E ISSO É O DESENHO. Robô imóvel com comando ativo tem
    erro igual ao comando, portanto numerador = denominador e a razão é ZERO. É o
    oposto exato do portão que media DURAÇÃO DE EPISÓDIO: aquele dava nota máxima à
    estátua, porque a estátua não cai. Este dá zero.

    ⚠ AS SOMAS VIVEM EM `self.metrics`, e isso não é conveniência. `CommandTerm.reset`
    (`command_manager.py:99-107`) lê a métrica, tira a média dos envs e SÓ DEPOIS zera
    — e o `reset` do comando roda DEPOIS do currículo (currículo -> eventos ->
    comando). Portanto o consumidor lê o episódio que ACABOU, e não um buffer meio
    zerado. Um buffer próprio meu precisaria repetir essa ordem à mão.

    ⚠ SÓ O EIXO LINEAR XY. O erro de guinada tem régua própria do fabricante
    (`error_vel_yaw`), e misturar rad/s com m/s numa soma só faria um número sem
    unidade nem interpretação. Um robô que anda reto e não gira mostra
    `razao_marcha` alta e `error_vel_yaw` alto — dois números, dois defeitos.

    ⚠ A RAZÃO PODE FICAR NEGATIVA, e não é clampeada. Andar para o lado ERRADO dá erro
    de até 2× o comando, logo razão −1,0. Clampear em 0 esconderia a diferença entre
    "parado" e "indo ao contrário", que é justamente o que se quer ver num bloco em
    que a política derivou.
    """

    cfg: TwistComRazaoDeMarchaCfg

    def __init__(self, cfg: TwistComRazaoDeMarchaCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        z = torch.zeros(self.num_envs, device=self.device)
        self.metrics["soma_erro_marcha"] = z.clone()
        self.metrics["soma_cmd_marcha"] = z.clone()
        self.metrics["razao_marcha"] = z.clone()

        # ⚠ A EFICIÊNCIA POR SEGMENTO (27/08). Ela SUBSTITUI o `razao_marcha` como juiz;
        # ver o docstring da classe para o motivo. Os cinco buffers moram em
        # `self.metrics` pelo mesmo motivo que as somas: o `reset` do mjlab zera o que
        # está no dict, e um buffer próprio meu precisaria repetir a ordem à mão.
        self.metrics["seg_proj"] = z.clone()     # Σ (v_real · v̂_cmd) dt, no segmento
        self.metrics["seg_pedido"] = z.clone()   # Σ ‖v_cmd‖ dt, no segmento
        self.metrics["seg_visto"] = z.clone()    # cópia do `command_counter`
        self.metrics["segmentos"] = z.clone()    # quantos segmentos VÁLIDOS fecharam
        self.metrics["eficiencia_min"] = z.clone()
        self.metrics["eficiencia_media"] = z.clone()

    def _fecha_segmento(self, mudou: torch.Tensor) -> None:
        """Pontua o segmento que acabou, SÓ nos envs de `mudou`, e reinicia os deles.

        ⚠ A MÁSCARA É OBRIGATÓRIA. Num lote de 4096 envs cada um re-sorteia no seu
        próprio instante, então numa chamada típica só uma fração cruzou a fronteira.
        Sem a máscara, o zeramento apagaria o acumulador dos envs que estão NO MEIO do
        segmento deles — e a métrica mediria pedaços aleatórios de segmento.

        ⚠ VALIDADE POR `seg_pedido`, e não por duração. Um segmento só é pontuado se o
        que foi PEDIDO nele passa de `pedido_min_segmento`. Uma regra só descarta dois
        casos de uma vez: o comando quase nulo (`is_standing_env` e sorteio perto de
        zero), cujo denominador é ruído; e o fragmento curto do fim do episódio, onde
        150 passos ainda não cancelaram o ruído de ação.

        ⚠ E O PRIMEIRO SEGMENTO NÃO PODE ENTRAR NO `min` COMO SE HOUVESSE UM ANTERIOR:
        com `segmentos == 0` o mínimo É a eficiência dele, e não `min(e, 0.0)` — que
        travaria a métrica em zero para sempre.
        """
        ped = self.metrics["seg_pedido"]
        vale = mudou & (ped >= self.cfg.pedido_min_segmento)
        e = self.metrics["seg_proj"] / ped.clamp(min=1e-6)

        n = self.metrics["segmentos"]
        primeiro = vale & (n == 0.0)
        demais = vale & (n > 0.0)

        self.metrics["eficiencia_min"][:] = torch.where(
            primeiro, e,
            torch.where(demais,
                        torch.minimum(self.metrics["eficiencia_min"], e),
                        self.metrics["eficiencia_min"]))
        # média incremental, para não guardar a soma num sexto buffer
        self.metrics["eficiencia_media"][:] = torch.where(
            vale,
            (self.metrics["eficiencia_media"] * n + e) / (n + 1.0),
            self.metrics["eficiencia_media"])
        self.metrics["segmentos"] += vale.float()

        # ⚠ zera onde MUDOU, e não onde VALE: um segmento inválido também terminou, e
        # deixar o acumulador dele de pé o somaria ao segmento seguinte.
        z = torch.zeros_like(self.metrics["seg_proj"])
        self.metrics["seg_proj"][:] = torch.where(
            mudou, z, self.metrics["seg_proj"])
        self.metrics["seg_pedido"][:] = torch.where(
            mudou, z, self.metrics["seg_pedido"])

    def _update_metrics(self) -> None:
        super()._update_metrics()
        cmd = self.vel_command_b[:, :2]
        vel = self.robot.data.root_link_lin_vel_b[:, :2]
        norma_cmd = torch.norm(cmd, dim=-1)

        # ⚠ FRONTEIRA DE SEGMENTO, detectada pelo `command_counter`. A ORDEM DO MJLAB
        # torna isto correto: `CommandTerm.compute` chama `_update_metrics` ANTES do
        # `_resample` (`command_manager.py:110-115`), portanto nesta chamada o comando e
        # o contador ainda são os do segmento que está correndo. Quando o contador muda,
        # a mudança é vista na chamada SEGUINTE — e aí o acumulador a ser fechado é o do
        # segmento certo, sem nunca somar velocidade nova em comando velho.
        #
        # ⚠ O `seg_visto` zera no reset, junto com as outras métricas, e isso é seguro:
        # no passo seguinte o contador difere de 0, o fecho dispara com `seg_pedido = 0`,
        # e a regra de validade descarta o segmento vazio. Sem efeito no log.
        atual = self.command_counter.to(dtype=self.metrics["seg_visto"].dtype)
        mudou = atual != self.metrics["seg_visto"]
        if bool(mudou.any()):
            self._fecha_segmento(mudou)
            self.metrics["seg_visto"][:] = atual

        # ⚠ O GATE. Sem ele um comando de 0,001 m/s entraria nas duas somas com erro
        # quase nulo e inflaria a razão de graça — e o fabricante põe 10% dos envs em
        # `is_standing_env`, portanto esse caso NÃO é raro.
        ativo = (norma_cmd > self.cfg.limiar_comando).float()

        self.metrics["soma_erro_marcha"] += torch.norm(cmd - vel, dim=-1) * ativo
        self.metrics["soma_cmd_marcha"] += norma_cmd * ativo

        # ⚠ ASSINATURA IN-PLACE. `self.metrics[k] = ...` trocaria o objeto de tensor, e
        # o `reset` do mjlab zera o objeto que estiver no dict — funcionaria, mas
        # qualquer referência guardada apontaria para o buffer velho.
        soma_cmd = self.metrics["soma_cmd_marcha"]
        self.metrics["razao_marcha"][:] = torch.where(
            soma_cmd > 0.0,
            1.0 - self.metrics["soma_erro_marcha"] / soma_cmd.clamp(min=1e-6),
            torch.zeros_like(soma_cmd),
        )

        # ⚠ A ACUMULAÇÃO DO SEGMENTO. Projeção, e NÃO norma, e essa é a única diferença
        # que importa contra o `razao_marcha`:
        #
        #     Σ‖v_cmd − v_real‖   -> norma nunca cancela; ruído de média zero SEMPRE
        #                            aumenta a soma, monotonicamente.
        #     Σ(v_real · v̂_cmd)   -> o ruído entra como Σ(ruído · v̂_cmd), média zero,
        #                            e encolhe com 1/√N dentro do segmento.
        #
        # MEDIDO no bloco 1: o `std` subiu de 0,43 (it 1525) para 0,61 (it 4999) e o
        # `razao_marcha` caiu de 0,514 para 0,426 — enquanto DURAÇÃO (984 -> 988) e
        # QUEDA (0,000 -> 0,167) não se moveram, e o `play` determinístico andava bem.
        # A queda era da forma da métrica, não da política.
        #
        # ⚠ `dt` é `self._env.step_dt`, e não `1/50` escrito à mão.
        dt = self._env.step_dt
        dir_cmd = cmd / norma_cmd.clamp(min=1e-6).unsqueeze(-1)
        self.metrics["seg_proj"] += (vel * dir_cmd).sum(dim=-1) * ativo * dt
        self.metrics["seg_pedido"] += norma_cmd * ativo * dt


@dataclass(kw_only=True)
class TwistComRazaoDeMarchaCfg(UniformVelocityCommandCfg):
    """⚠ O `mjlab` constrói o termo por `cfg.build(env)`, e NÃO por um atributo
    `class_type` (`command_manager.py:268`). Um `class_type` aqui seria campo morto:
    o cfg passaria, o manager chamaria o `build` HERDADO, e o treino rodaria com o
    twist do fabricante — sem a métrica e sem nenhum erro."""

    limiar_comando: float = 0.05

    pedido_min_segmento: float = 0.5
    """Piso de VALIDADE do segmento: `Σ‖v_cmd‖dt` mínimo para ele ser pontuado.

    ⚠ NÃO É UM ALVO. A tarefa continua sendo rastrear velocidade; isto só decide se um
    segmento tem sinal suficiente para ser julgado. Uma regra descarta dois casos: o
    comando quase nulo, cujo denominador é ruído, e o fragmento curto no fim do
    episódio, onde o ruído de ação ainda não cancelou.

    **Derivado:** o comando médio vale ~0,765 m/s, e o re-sorteio do fabricante é de 3 a
    8 s. Um segmento inteiro rende então ~2,3 no mínimo. 0,5 aceita segmentos a partir
    de ~0,7 s de comando cheio e descarta os 10% de `is_standing_env` (que somam 0)."""

    def build(self, env: ManagerBasedRlEnv) -> TwistComRazaoDeMarcha:
        return TwistComRazaoDeMarcha(self, env)
