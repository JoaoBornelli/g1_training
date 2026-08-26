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
    caixa_meia_z: float = 0.10
    # ⚠ A meia-aresta entra no kernel de alcance: a distância medida é até a
    # SUPERFÍCIE da caixa, não até o centro. Ver `dist_palma_caixa`.
    caixa_meia_aresta: float = 0.10
    # a face MARCADA, no frame da caixa. Constante: é sempre ela que o `reorientar`
    # pede normal ao robô. A dificuldade está na ORIENTAÇÃO DE NASCIMENTO da caixa.
    face_alvo_b: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    # os sítios das palmas, para a distância que define o σ
    sitios_palma: tuple[str, ...] = ("left_palm", "right_palm")

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

    # -------------------------------------------------------------- o contrato
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        d = self.device
        n = len(env_ids)

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

        # ⚠ NÃO se sorteia face nem ângulo aqui. A face pedida é CONSTANTE (a
        # marcada), e a dificuldade do `reorientar` vem da ORIENTAÇÃO DE NASCIMENTO da
        # caixa, sorteada em quartos de volta pelo evento `posiciona_cena`.
        _ = n
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

    def _update_metrics(self) -> None:
        pass

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
        if not bool(parados.any()):
            return
        tw = self._env.command_manager.get_term(self.cfg.nome_do_twist)
        tw.vel_command_b[parados] = 0.0

    # ------------------------------------------------------ o alvo, por elo
    def _aplica_elo(self, ids: torch.Tensor, *, so_pose: bool = False) -> None:
        """Escreve o alvo do elo corrente, e move a laje quando o elo pede.

        `so_pose=True` refaz APENAS o que depende de pose de entidade. É o que a
        passada de `_pendente` chama, no 1º passo depois do reset, quando os buffers
        de `data` já estão frescos.

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
                fundo = self.caixa.data.root_link_pos_w[m, 2] - c.caixa_meia_z
                teto = torch.clamp(fundo - c.botar_folga_laje, max=c.botar_topo_teto)
                piso = torch.full_like(teto, c.botar_topo_piso)
                teto = torch.maximum(teto, piso)      # a faixa nunca inverte
                topo = piso + (teto - piso) * torch.rand(k, device=d)
                self._laje_para(m, topo)
                # o alvo é LATERAL, em cima do topo novo. O frontal exigiria alcançar
                # por cima de 20 cm de tampo — defeito medido em 16/07.
                bx, by = c.botar_x, c.botar_y
                a = torch.zeros(k, 3, device=d)
                a[:, 0] = org[:, 0] + bx[0] + (bx[1] - bx[0]) * torch.rand(k, device=d)
                a[:, 1] = org[:, 1] + by[0] + (by[1] - by[0]) * torch.rand(k, device=d)
                a[:, 2] = topo + c.caixa_meia_z
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
            pc[:, 2] = topo_t + c.caixa_meia_z
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

    def dist_palma_caixa(self, ids: torch.Tensor) -> torch.Tensor:
        """Distância da palma MAIS PRÓXIMA à SUPERFÍCIE da caixa, por env.

        ⚠ `min` sobre as duas palmas, e não média: no início do treino uma mão chega
        antes da outra, e a média diluiria o sinal da que está chegando. O `squeeze`
        é que exige as DUAS.

        ⚠ SUPERFÍCIE, E NÃO CENTRO, e isso foi um DEFEITO MEDIDO em 2026-08-26. Ao
        centro, a distância mínima fisicamente alcançável é 0,191 m (a caixa colide com
        a mão antes) — logo `exp(−(d/σ)²)` saturava em **0,674** e o `staged` nunca
        passava de 3,35 de um teto de 6,0. O robô fazia a tarefa inteira e o termo
        dizia "faltam 33%".
        """
        palmas = self.robot.data.site_pos_w[ids][:, self._ids_palma, :]
        centro = self.caixa.data.root_link_pos_w[ids].unsqueeze(1)
        d = torch.norm(palmas - centro, dim=-1).min(dim=1).values
        return (d - self.cfg.caixa_meia_aresta).clamp(min=0.0)

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

    def _atualiza_face(self, ids: torch.Tensor) -> None:
        """Publica a DIREÇÃO DESEJADA e o ERRO angular da face marcada.

            FACE  a direção em que a face marcada DEVE apontar — da caixa para o robô,
                  na horizontal.
            ANG   o erro angular ATUAL, em radianos, entre a normal da face marcada e
                  essa direção. Zero = alinhada.

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
        desejada = para_o_robo / para_o_robo.norm(dim=-1, keepdim=True).clamp(min=1e-6)

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
    """O twist do fabricante, com UMA métrica a mais: a `razao_marcha`.

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

    def _update_metrics(self) -> None:
        super()._update_metrics()
        cmd = self.vel_command_b[:, :2]
        vel = self.robot.data.root_link_lin_vel_b[:, :2]
        norma_cmd = torch.norm(cmd, dim=-1)

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


@dataclass(kw_only=True)
class TwistComRazaoDeMarchaCfg(UniformVelocityCommandCfg):
    """⚠ O `mjlab` constrói o termo por `cfg.build(env)`, e NÃO por um atributo
    `class_type` (`command_manager.py:268`). Um `class_type` aqui seria campo morto:
    o cfg passaria, o manager chamaria o `build` HERDADO, e o treino rodaria com o
    twist do fabricante — sem a métrica e sem nenhum erro."""

    limiar_comando: float = 0.05

    def build(self, env: ManagerBasedRlEnv) -> TwistComRazaoDeMarcha:
        return TwistComRazaoDeMarcha(self, env)
