"""Os dois termos de comando: a META (17 números) e o TWIST derivado (3).

    lift_target                    17 números  — a meta que a política persegue
      [0:3]   alvo_pos    destino em MUNDO (a obs converte pra base)
      [3:6]   face_alvo   eixo do CORPO da caixa que é a face n
      [6:9]   dir_alvo    direção alvo, já em frame da BASE
      [9:17]  one-hot     8 slots, 7 em uso

    twist                           3 números  — DERIVADO, não vai na obs do ator
      [0:2]   vx, vy      perfil de velocidade em frame da base
      [2]     ωz          controle de heading

Duas escolhas de layout que economizam código:

- **`alvo_pos` fica em `[0:3]`** de propósito, porque assim
  `g1_training/common/observations.py::target_pos_b` é reusado sem uma linha de
  mudança.
- **o twist é um termo SEPARADO, chamado `"twist"`**, porque os 5 rewards de
  marcha do fabricante leem `command[:, :3]` do termo que recebem por nome. Nome
  igual ao do fabricante = fiação zero, nenhuma função reescrita.

🔧 **F5 — `alvo_pos` não é buffer, é regra avaliada a cada passo.** A convenção
óbvia era guardar a posição do robô no reset pra o `parado` ter `target_pos_b = 0`.
Mas gravado no reset, quando o robô se move aquele vetor passa a ser o vetor **de
volta ao spawn** — o vetor de deriva. Informar deriva implica uma tarefa ("volte")
pela qual a política não é recompensada, e o perfil de velocidade mandaria ela
caminhar de volta depois de cada empurrão. Por isso `_update_command` recalcula.

⚠️ **Divergência resolvida no doc.** A tabela "Preenchimento por tarefa" da §9 põe
`alvo_pos = 0` no `pegar` ("peito é constante na base"), mas a tabela do F5, que é
o conserto posterior, diz que `target_pos_b` do `pegar` contém **a caixa**. Vale o
F5: `alvo_pos` significa "o ponto que eu preciso alcançar", e no `pegar` a
distância chega a 2.0 m — com `alvo_pos = 0` o twist seria zero e o robô nunca
se aproximaria. O "0" da outra tabela é sobre não precisar transmitir o alvo do
PEITO, que é constante na base e vive no reward como `alvo_peito_b`.

Contrato com o currículo (preenchido na Tarefa 13; aqui há fallback):
  `env.task_dist`  [7] probabilidades por tarefa, ou ausente -> uniforme
  `env.nivel`      dict eixo -> [num_envs] long, ou ausente -> índice inicial
  `env.active_task` [num_envs] long — ESCRITO aqui, no `__init__`, porque os
                    rewards leem antes do 1º reset (§15).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

ALVO = slice(0, 3)
FACE = slice(3, 6)
DIR = slice(6, 9)
ONEHOT = slice(9, 17)
COMMAND_DIM = 17

FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),      # 0,1 — laterais (giro em torno de z)
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),      # 2,3 — laterais
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),      # 4,5 — topo e fundo (exigem erguer)
)
LATERAIS = (0, 1, 2, 3)
TOPO_FUNDO = (4, 5)


def _nivel(env, eixo: str, task_idx: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    """Índice de nível nesse eixo, shape igual a `task_idx`.

    Fallback (antes da Tarefa 13 existir): o índice INICIAL da tarefa, que já
    codifica a regra "quem anda começa a distância em 0.3"."""
    tabela = getattr(env, "nivel", None)
    if tabela is not None and eixo in tabela:
        return tabela[eixo][env_ids]
    inicio = torch.zeros_like(task_idx)
    for t, eixos in T.AXES.items():
        if eixo in eixos:
            inicio = torch.where(task_idx == t, eixos[eixo], inicio)
    return inicio


def _rot_z(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Roda `v` [B,3] em torno de z por `ang` [B]. Sem quaternion — é rotação plana."""
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack((v[:, 0] * c - v[:, 1] * s,
                        v[:, 0] * s + v[:, 1] * c,
                        v[:, 2]), dim=-1)


class LiftTargetCommand(CommandTerm):
    """A meta: qual tarefa, onde é o destino, e qual face vai pra qual direção."""

    cfg: "LiftTargetCommandCfg"

    def __init__(self, cfg: "LiftTargetCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self.box: Entity = env.scene[cfg.box_name]
        self.table: Entity = env.scene[cfg.table_name]

        self._command = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._destino_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._face_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._spawn_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._disparou = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        """O gatilho da tarefa já disparou? Lido pelo `metrics.Sucesso`.

        ⚠️ **Constante em True desde a S8**, que removeu o atraso de gatilho. A tarefa
        sorteada entra ativa no passo 0, portanto não existe mais janela de pré-gatilho
        em que uma tarefa poderia pontuar pelo critério do `parado`.

        O campo e o acessor `disparou` ficam por dois motivos: o `metrics.Sucesso` os
        consome, e o gate volta a ter conteúdo quando a troca de tarefa no meio do
        episódio entrar (hoje adiada de propósito). Remover agora seria arrancar e
        recolocar."""
        self._dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Direção alvo da face, em MUNDO. 🔧 Conserto de 31/07: era guardada no frame
        da BASE e comparada contra a normal recomputada na base a cada passo — então
        **girar o ROBÔ mudava o erro com a caixa imóvel**. Medido, 64 envs, ação zero: a
        caixa não sai do lugar (`desvio_xy = 0,0009 m`) e a fração dentro da tolerância
        de 10° sobe de 0/64 no spawn para 19/64 (30%) no passo 200 — o `reorientar` era
        aprovado pelo robô virar. Em mundo, só a rotação da CAIXA move o erro."""

        # INTENÇÃO sorteada, resolvida contra a pose só depois (ver `_resolver`).
        self._ang = torch.zeros(self.num_envs, device=self.device)
        self._topo = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dist = torch.zeros(self.num_envs, device=self.device)
        self._head = torch.zeros(self.num_envs, device=self.device)
        self._pendente = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # O ATRASO DE GATILHO saiu na S8. `_t` permanece como relógio do episódio,
        # porque o `env.trigger_t` é exposto e lido fora; ele deixou de gatilhar nada.
        self._t = torch.zeros(self.num_envs, device=self.device)
        self._sorteada = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ativa = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """A tarefa ATIVA agora. Desde a S8 ela é igual à sorteada desde o passo 0 —
        antes havia uma janela de até 2 s em `parado`. É esta que os gates de reward e
        as terminações leem. Escrita SEMPRE in-place, porque `env.active_task` guarda
        a referência ao tensor."""

        # Disciplina do §15: os rewards leem `active_task` antes do 1º reset, então
        # o buffer TEM que existir aqui. Mesma disciplina que o `PlrHeights` segue.
        env.active_task = self._ativa
        env.trigger_t = self._t

        self.metrics["erro_posicao"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["erro_angulo_deg"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    # ------------------------------------------------------------------ resample
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sorteia a INTENÇÃO. Nada que dependa de pose é resolvido aqui.

        ⚠️ Por que a separação existe (bug medido 30/07): no reset o command
        manager roda DEPOIS dos eventos que reposicionam robô/caixa
        (`manager_based_rl_env.py:581` contra `:560`), mas as grandezas
        DERIVADAS — `root_link_pos_w`, `root_link_quat_w` — ainda não foram
        recalculadas: o evento escreveu `qpos`, e a cinemática só roda no
        próximo passo. Ler pose aqui devolve a pose ANTERIOR ao reset.

        O sintoma era silencioso e destruía o eixo de giro: com `dir_alvo`
        derivado da orientação velha da caixa, o erro de ângulo no nível 15°
        saía espalhado de 0.35° a 40.65° em vez de 15° exatos — ou seja o
        currículo de giro não comandava rotação nenhuma em particular."""
        n = len(env_ids)
        if n == 0:
            return

        # A tarefa foi sorteada pelo CURRÍCULO (`:554`), não aqui (`:581`): o evento
        # `reset_segurando` (`:560`) precisa dela pra decidir quem nasce com a caixa
        # nas mãos, e isso acontece antes deste método rodar.
        tarefa = self._env.tarefa_sorteada[env_ids]
        self._sorteada[env_ids] = tarefa

        # A tarefa sorteada entra ATIVA já no passo 0 (S8). O atraso de gatilho saiu.
        #
        # Ele existia para dar um transiente de comando: o episódio começava em
        # `parado` (ou `parado c/ caixa` para quem nasce segurando) por até 2 s, e só
        # então a tarefa real entrava. O pré-gatilho distinto era necessário porque a
        # caixa escorrega 22 cm em 0.5 s com ação nula, e sem ele o episódio das três
        # tarefas de `SPAWN_SEGURANDO` já nascia perdido.
        #
        # ⚠️ **O desenho fica SEM treino de transiente de comando nesta rodada, e isso
        # é conhecido.** O substituto correto é a troca de tarefa no meio do episódio,
        # que está adiada de propósito — ver FORA DE ESCOPO da spec. Não implementar
        # aqui.
        #
        # Ganho de lado: o atraso consumia até 2 s dos 20 s. As células duras
        # recuperam esse tempo.
        self._ativa[env_ids] = tarefa

        # distância e rumo do destino: níveis, sem pose ainda
        self._dist[env_ids] = self._distancia_nivel(tarefa, env_ids)
        self._head[env_ids] = self._rumo_nivel(tarefa, env_ids)

        # ORIENTAÇÃO (só o `reorientar` usa). Nível 0-3 = giro em torno de z sobre
        # uma face LATERAL; nível 4 = topo/fundo, salto qualitativo que exige a mão.
        giro = _nivel(self._env, "giro", tarefa, env_ids)
        ang = torch.tensor([math.radians(a) for a in T.LEVELS["giro"]],
                           device=self.device)[giro]
        sinal = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self._ang[env_ids] = ang * sinal
        topo = giro >= len(T.LEVELS["giro"]) - 1
        self._topo[env_ids] = topo
        lat = torch.randint(0, len(LATERAIS), (n,), device=self.device)
        tf = torch.randint(0, len(TOPO_FUNDO), (n,), device=self.device) + TOPO_FUNDO[0]
        self._face_idx[env_ids] = torch.where(topo, tf, lat)

        self._pendente[env_ids] = True

    def _resolver(self) -> None:
        """Resolve contra a pose FRESCA o que o resample não podia resolver.

        Chamado de todo ponto de leitura (`_update_metrics`, `_update_command`,
        `erro_angulo_deg`), e é no-op quando não há nada pendente. Assim quem
        consome nunca vê um comando meio-resolvido, sem precisar saber da ordem
        interna do reset."""
        if not bool(self._pendente.any()):
            return
        ids = self._pendente.nonzero().flatten()
        tarefa = self._sorteada[ids]

        # DESTINO estático das tarefas que andam, ancorado no spawn REAL. É ponto do
        # MUNDO, não a posição do próprio robô -> não recai no F5.
        d, head = self._dist[ids], self._head[ids]
        self._destino_w[ids] = self.robot.data.root_link_pos_w[ids] + torch.stack(
            (d * torch.cos(head), d * torch.sin(head), torch.zeros_like(d)), dim=-1)

        # alvo x,y do `reorientar` = onde a caixa NASCEU (pose real pós-reset).
        self._spawn_xy[ids] = self.box.data.root_link_pos_w[ids, :2]

        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx[ids]]
        # ⚠️ TUDO em MUNDO daqui pra baixo. A normal da face vive no frame da CAIXA
        # (`face_b` é constante), então levá-la a mundo é uma rotação só. O alvo é a
        # normal do SPAWN girada em torno do z do mundo pelo ângulo do nível.
        normal_w = quat_apply(self.box.data.root_link_quat_w[ids], face_b)
        alvo_lateral = _rot_z(normal_w, self._ang[ids])
        # topo/fundo: a face tem que apontar PRA MIM. Resolvido no SPAWN e congelado —
        # é alvo relativo ao robô por natureza, e congelar mantém o critério medindo
        # rotação da caixa, igual ao `_spawn_xy`.
        para_mim = (self.robot.data.root_link_pos_w[ids, :2]
                    - self.box.data.root_link_pos_w[ids, :2])
        alvo_tf = torch.zeros_like(alvo_lateral)
        alvo_tf[:, :2] = para_mim / para_mim.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_alvo = torch.where(self._topo[ids].unsqueeze(-1), alvo_tf, alvo_lateral)
        dir_alvo = dir_alvo / dir_alvo.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._dir_w[ids] = dir_alvo

        # zera face/dir de quem não é `reorientar` (§9: tabela de preenchimento)
        so_reorienta = (tarefa == T.REORIENTAR).unsqueeze(-1)
        self._command[ids, FACE] = torch.where(
            so_reorienta, face_b, torch.zeros_like(face_b))
        self._dir_w[ids] = torch.where(
            so_reorienta, dir_alvo, torch.zeros_like(dir_alvo))
        # ⚠️ `DIR` da OBS não é escrito aqui: o `_update_command` o reescreve a CADA
        # passo, convertendo `_dir_w` para o frame da base atual. Antes ele era gravado
        # uma vez e ficava obsoleto — a política via um vetor que já não significava o
        # que significava no spawn.
        self._pendente[ids] = False

    def _distancia_nivel(self, tarefa: torch.Tensor, env_ids: torch.Tensor):
        idx = _nivel(self._env, "distancia_andar", tarefa, env_ids)
        niveis = torch.tensor(T.LEVELS["distancia_andar"], device=self.device)
        d = niveis[idx.clamp(max=len(T.LEVELS["distancia_andar"]) - 1)]
        # só as tarefas que ANDAM têm destino deslocado; as outras miram caixa/prateleira
        anda = torch.zeros_like(d, dtype=torch.bool)
        for t in T.ANDA:
            anda |= tarefa == t
        return torch.where(anda, d, torch.zeros_like(d))

    def _rumo_nivel(self, tarefa: torch.Tensor, env_ids: torch.Tensor):
        idx = _nivel(self._env, "rumo", tarefa, env_ids)
        graus = torch.tensor(T.LEVELS["rumo"], device=self.device)[
            idx.clamp(max=len(T.LEVELS["rumo"]) - 1)]
        meio = torch.deg2rad(graus) / 2.0
        return (torch.rand(len(tarefa), device=self.device) * 2.0 - 1.0) * meio

    # -------------------------------------------------------------------- update
    def _update_command(self) -> None:
        """Reavalia `alvo_pos` a cada passo pela regra da tarefa (F5)."""
        self._resolver()
        self._t += self._env.step_dt
        # S8: sem atraso de gatilho, a ativa é a sorteada desde o passo 0.
        # in-place: `env.active_task` guarda a REFERÊNCIA a este tensor
        self._ativa.copy_(self._sorteada)
        tarefa = self._ativa

        one = torch.zeros(self.num_envs, T.ONEHOT_DIM, device=self.device)
        one.scatter_(1, tarefa.unsqueeze(-1), 1.0)
        self._command[:, ONEHOT] = one

        robo = self.robot.data.root_link_pos_w
        caixa = self.box.data.root_link_pos_w
        # ponto de pouso na prateleira: centro dela + meia altura da caixa
        prateleira = self.table.data.root_link_pos_w.clone()
        prateleira[:, 2] += self.cfg.shelf_half_z + self.cfg.box_half_z

        alvo = robo.clone()                                  # parado / parado c/ caixa
        for t in T.ANDA:
            alvo = torch.where((tarefa == t).unsqueeze(-1), self._destino_w, alvo)
        for t in (T.PEGAR, T.REORIENTAR):
            alvo = torch.where((tarefa == t).unsqueeze(-1), caixa, alvo)
        alvo = torch.where((tarefa == T.BOTAR).unsqueeze(-1), prateleira, alvo)
        self._command[:, ALVO] = alvo

        # `dir_alvo` da OBS: o alvo vive em MUNDO (`_dir_w`) e a obs é egocêntrica, então
        # a conversão é por passo. Sem isso a política veria o vetor do spawn, que deixa
        # de apontar pro lugar certo assim que o robô gira.
        self._command[:, DIR] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self._dir_w)

    @property
    def disparou(self) -> torch.Tensor:
        """[B] bool — a tarefa sorteada já está ativa (o atraso de gatilho acabou)."""
        return self._disparou

    def _update_metrics(self) -> None:
        self._resolver()
        passos = self.cfg.resampling_time_range[1] / self._env.step_dt
        err = torch.norm(self._command[:, ALVO] - self.box.data.root_link_pos_w, dim=-1)
        self.metrics["erro_posicao"] += err / passos
        self.metrics["erro_angulo_deg"] += self.erro_angulo_deg() / passos

    # ------------------------------------------------------------------ leitores
    def erro_angulo_deg(self) -> torch.Tensor:
        """Ângulo entre a normal da face alvo e `dir_alvo`, **em MUNDO**, em graus.

        UM escalar, e a simetria do cubo se resolve sozinha: girar em torno da
        normal da face não muda o vetor, então o erro não se move — e como essa
        rotação é irrelevante pro objetivo, a métrica CONCORDA com o objetivo.
        Sem grupo de simetria, sem quaternion.

        🔧 **Conserto de 31/07: era comparado no frame da BASE, e girar o ROBÔ mudava o
        erro com a caixa parada.** Medido: caixa imóvel (`desvio_xy = 0,0009 m`) e 30% dos
        envs entrando na tolerância de 10° até o passo 200 — o `reorientar` marcava
        0,94-1,00 de competência sem tocar na caixa. Em mundo, só a rotação da CAIXA move
        o erro, e chegar aos 10° a partir dos 15° do nível 0 EXIGE girar ≥5° de verdade.
        Por isso não precisa de termo extra "girou": o gate já é a rotação."""
        self._resolver()
        face_b = self._command[:, FACE]
        normal_w = quat_apply(self.box.data.root_link_quat_w, face_b)
        cos = (normal_w * self._dir_w).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.rad2deg(torch.acos(cos))
        # sem comando de orientação (face zerada) o erro não existe -> 0
        return torch.where(face_b.abs().sum(dim=-1) > 0.0, ang, torch.zeros_like(ang))

    def spawn_xy_b(self) -> torch.Tensor:
        """O alvo x,y do `reorientar` em frame da base, [B,2]. Convertido a cada passo."""
        self._resolver()
        robo = self.robot.data.root_link_pos_w
        delta = torch.cat((self._spawn_xy - robo[:, :2],
                           torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        return quat_apply_inverse(self.robot.data.root_link_quat_w, delta)[:, :2]

    def desvio_xy(self) -> torch.Tensor:
        """Quanto a caixa saiu do x,y onde nasceu, em metros. [B]

        Distância é invariante de frame, então sai direto do mundo — não precisa
        converter pra base. O alvo é ponto ESTÁTICO do mundo (onde a caixa nasceu),
        e por isso não recai no F5: o problema de lá era guardar a posição do
        próprio ROBÔ, que se move, e transformar o alvo num vetor de deriva."""
        self._resolver()
        return (self.box.data.root_link_pos_w[:, :2] - self._spawn_xy).norm(dim=-1)

    def erro_rumo_deg(self) -> torch.Tensor:
        """Erro entre o yaw do robô e o RUMO do destino sorteado, em graus. [B]

        Serve ao `alinhado` do critério de sucesso do `andar` (S6). O rumo é
        `self._head`, o ângulo com que o destino foi posto em relação ao spawn — e não
        a direção instantânea robô->alvo.

        ⚠️ A diferença importa exatamente onde o critério é avaliado. A direção
        instantânea fica INDEFINIDA quando o robô chega em cima do alvo (o vetor vai a
        zero e o `atan2` devolve ruído), que é justamente o momento em que o
        `alinhado` precisa valer. O rumo sorteado é constante no episódio e não tem
        essa singularidade.

        Acessor público de propósito: o `metrics.py` lendo `_head` seria acoplamento a
        atributo privado de command term — o mesmo que o `dentro_do_morto` evita."""
        self._resolver()
        erro = self.robot.data.heading_w - self._head
        # embrulha em [−π, π]: sem isso, 350° de erro leria como 350 em vez de −10
        erro = torch.atan2(torch.sin(erro), torch.cos(erro))
        return erro.abs() * (180.0 / torch.pi)

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Esfera no destino + seta da direção alvo — este é o gizmo do item 6.

        A face alvo fica VISÍVEL no `play` sem mudar geometria nem recompilar: a
        seta sai do centro da caixa na direção que a face tem que apontar, e a
        esfera marca o destino da tarefa."""
        idx = visualizer.get_env_indices(self.num_envs)
        if not idx:
            return
        for b in idx:
            visualizer.add_sphere(
                center=self._command[b, ALVO].cpu().numpy(), radius=0.03,
                color=self.cfg.viz.cor_alvo, label=f"alvo_{b}")
            if float(self._command[b, FACE].abs().sum()) > 0.0:
                base = self.box.data.root_link_pos_w[b]
                dir_w = quat_apply(self.robot.data.root_link_quat_w[b: b + 1],
                                   self._command[b: b + 1, DIR])[0]
                visualizer.add_sphere(
                    center=(base + 0.25 * dir_w).cpu().numpy(), radius=0.02,
                    color=self.cfg.viz.cor_face, label=f"dir_alvo_{b}")


@dataclass(kw_only=True)
class LiftTargetCommandCfg(CommandTermCfg):
    box_name: str = "box"
    table_name: str = "table"
    box_half_z: float = 0.10
    shelf_half_z: float = 0.02

    @dataclass
    class VizCfg:
        cor_alvo: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)
        cor_face: tuple[float, float, float, float] = (0.2, 0.8, 1.0, 0.6)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: "ManagerBasedRlEnv") -> LiftTargetCommand:
        return LiftTargetCommand(self, env)


def _quintica(u: torch.Tensor) -> torch.Tensor:
    """Interpolação quíntica `s(u) = 10u³ − 15u⁴ + 6u⁵`, com `u` em [0, 1].

    `s(0) = 0`, `s(1) = 1`, e a primeira e a segunda derivadas são NULAS nas duas
    pontas. É isso que ela traz: o perfil de frenagem deixa de ter joelho na entrada e
    na saída da banda. Uma rampa linear tem derivada descontínua nos dois extremos, e
    o robô sente isso como degrau de aceleração."""
    u2 = u * u
    return u2 * u * (10.0 - 15.0 * u + 6.0 * u2)


class DesiredTwistCommand(CommandTerm):
    """`[vx, vy, ωz]` derivado do destino. NÃO vai na obs do ator.

    Existe pra destravar 5 recompensas de marcha do fabricante sem reescrever
    nenhuma: elas leem `command[:, :2]` e `command[:, 2]` do termo que recebem por
    nome, e o nome aqui é `"twist"` — o mesmo do cfg de velocity.

    --------------------------------------------------------------------------
    PERFIL DE DUAS FASES (S5)
    --------------------------------------------------------------------------

    O robô GIRA e depois ANDA. As duas fases não são uma máquina de estados; elas
    emergem de um fator contínuo:

        fase 1   `v` ≈ 0, `w` do erro de rumo      até o robô apontar para o alvo
        fase 2   `v` do perfil de distância        até `chegou` travar

    Não existe fase 3. O alvo de orientação **é** o rumo, portanto ao chegar andando
    para frente o robô já está alinhado.

    **A trava do `v` é contínua, e isso é essencial.** `v = v_perfil · clamp(cos(erro))`.
    ⚠️ Uma trava dura (`if erro < 10°`) devolve o degrau que a rampa existe para
    remover, e com frequência alta: ela abre em 10°, o robô anda, andar perturba o
    rumo, o erro sobe para 12°, `v` cai a zero de golpe, o robô para, o erro cai, e a
    trava abre outra vez.

    **Ordem dos estágios: fase -> trava -> rampa.** A rampa é o ÚLTIMO estágio. Se ela
    viesse antes da trava, não filtraria o degrau que a trava produz.

    **`vy` é sempre zero.** O perfil manda o robô girar para apontar, e depois andar
    para frente. Consequência declarada: o ator não treina marcha lateral nesta rodada.

    **Duas peças de suavização, para dois problemas diferentes:**

      - a quíntica age sobre a DISTÂNCIA, e tira o joelho do perfil de frenagem;
      - o limitador de taxa age sobre o TEMPO, e garante o teto de `a_max` / `alpha_max`
        nas transições de fase e no passo 0.

    ⚠️ O passo 0 é o caso que motiva o limitador. No reset o alvo está além de
    `d_morto + d_freio`, portanto o perfil pede `v_max` já no primeiro passo — um
    degrau. Ali o erro de rastreio é máximo, e o gradiente diz "acelere mais forte".

    O `d_morto` é 0.05 m nas tarefas que andam e 0.30 m na manipulação
    (`alvo_peito_b[0]` + 0.10), porque manipular exige parar mais longe do que chegar."""

    cfg: "DesiredTwistCommandCfg"

    def __init__(self, cfg: "DesiredTwistCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._dentro = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["v_comandada"] = torch.zeros(self.num_envs, device=self.device)

        # Estado do limitador de taxa. É por-env e por-episódio: zera no reset, senão
        # o primeiro passo do episódio novo herda a velocidade do episódio anterior.
        self._v_ant = torch.zeros(self.num_envs, device=self.device)
        self._w_ant = torch.zeros(self.num_envs, device=self.device)

        # A banda de frenagem NÃO é número livre: ela é `v²/2a`, a distância que o robô
        # precisa para parar de `v_max` com `a_max`. O assert amarra os três, para o
        # valor não voltar a ficar órfão como o 0.50 antigo ficou quando o `v_max` caiu
        # de 1.0 para 0.5.
        esperado = cfg.v_max ** 2 / (2.0 * cfg.a_max)
        assert abs(cfg.d_freio_extra - esperado) < 1e-6, (
            f"d_freio_extra={cfg.d_freio_extra} não bate com v_max²/2·a_max="
            f"{esperado:.4f}. Mudou `v_max` ou `a_max`? Recalcule a banda.")
        # Banda angular pela mesma conta: `w²/2α = 1.2²/6 = 0.24 rad = 13.8°`.
        self._banda_ang = cfg.w_max ** 2 / (2.0 * cfg.alpha_max)

        # S14: a inclinação de `v_max` por kg é DERIVADA das pontas do eixo de peso,
        # não digitada. Com (0.50 − 0.25) / (5 − 1) ela dá os 0.0625 m/s por kg da
        # spec, e continua correta se o eixo de peso mudar de extremo.
        self._peso_min = float(T.LEVELS["peso"][0])
        _span = float(T.LEVELS["peso"][-1]) - self._peso_min
        self._inclinacao_carga = ((cfg.v_max - cfg.v_max_carga_cheia) / _span
                                  if _span > 0 else 0.0)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Derivado: não sorteia nada. Só zera o estado do limitador, porque o episódio
        # novo começa parado.
        self._v_ant[env_ids] = 0.0
        self._w_ant[env_ids] = 0.0

    def _update_command(self) -> None:
        alvo_cmd = self._env.command_manager.get_term(self.cfg.meta_name)
        tarefa = getattr(self._env, "active_task")
        robo = self.robot.data.root_link_pos_w
        quat = self.robot.data.root_link_quat_w

        delta_w = alvo_cmd.command[:, ALVO] - robo
        delta_w[:, 2] = 0.0                                  # o twist é planar
        delta_b = quat_apply_inverse(quat, delta_w)
        d = delta_b[:, :2].norm(dim=-1)

        anda = torch.zeros_like(d, dtype=torch.bool)
        for t in T.ANDA:
            anda |= tarefa == t
        d_morto = torch.where(anda, self.cfg.d_morto_andar, self.cfg.d_morto_manipula)
        self._dentro.copy_(d <= d_morto)
        erro_rumo = torch.atan2(delta_b[:, 1], delta_b[:, 0])

        # --- estágio 0: velocidade máxima POR CARGA (S14) ---
        # Cai linearmente de `v_max` (1 kg) a `v_max_carga_cheia` (5 kg). A massa vem
        # do `payload_por_nivel`, que a sorteia em `U(1, peso(nível))`.
        #
        # ⚠️ Ligada à massa SORTEADA, não ao nível: pelo nível, um env que tirou 1.2 kg
        # andaria à velocidade de 5 kg.
        #
        # ⚠️ A banda de frenagem acompanha. Ela é `v²/2a`, portanto derivá-la do
        # `v_max` NOMINAL deixaria a carga pesada com banda longa demais — o robô
        # começaria a frear meio metro cedo.
        peso = getattr(self._env, "peso_amostrado", None)
        if peso is None:
            v_max = torch.full_like(d, self.cfg.v_max)
        else:
            v_max = (self.cfg.v_max - (peso - self._peso_min) * self._inclinacao_carga
                     ).clamp(min=self.cfg.v_max_carga_cheia)
        banda = (v_max ** 2 / (2.0 * self.cfg.a_max)).clamp(min=1e-4)

        # --- estágio 1: FASE. Perfil de distância, suavizado pela quíntica. ---
        rampa = ((d - d_morto) / banda).clamp(0.0, 1.0)
        v_alvo = v_max * _quintica(rampa)

        # --- estágio 2: TRAVA. Fator contínuo, nunca limiar. ---
        # cos(180°) = −1 -> clamp -> 0: parado de costas para o alvo, gira sem andar.
        # cos(0°) = 1: alinhado, anda à velocidade cheia do perfil.
        v_alvo = v_alvo * torch.cos(erro_rumo).clamp(0.0, 1.0)
        v_alvo = torch.where(self._dentro, torch.zeros_like(v_alvo), v_alvo)

        # perfil angular: mesma forma do linear, com a banda derivada de w²/2α
        mag = erro_rumo.abs()
        rampa_ang = ((mag - self.cfg.morto_angular_rad) / self._banda_ang).clamp(0.0, 1.0)
        w_alvo = torch.sign(erro_rumo) * self.cfg.w_max * _quintica(rampa_ang)
        w_alvo = torch.where(self._dentro, torch.zeros_like(w_alvo), w_alvo)

        # --- estágio 3: RAMPA. Limitador de taxa, o último estágio. ---
        dt = self._env.step_dt
        dv = (v_alvo - self._v_ant).clamp(-self.cfg.a_max * dt, self.cfg.a_max * dt)
        dw = (w_alvo - self._w_ant).clamp(
            -self.cfg.alpha_max * dt, self.cfg.alpha_max * dt)
        v = self._v_ant + dv
        w = self._w_ant + dw

        # Dentro do morto o zero é EXATO, e não "quase". O limitador não precisa segurar
        # nada aqui: a quíntica já levou o perfil a zero de forma contínua ao chegar em
        # `d_morto`, então o degrau só apareceria se o robô violasse o próprio perfil —
        # e nesse caso parar é o comportamento certo.
        v = torch.where(self._dentro, torch.zeros_like(v), v)
        w = torch.where(self._dentro, torch.zeros_like(w), w)
        self._v_ant.copy_(v)
        self._w_ant.copy_(w)

        # `vy` é sempre zero: o perfil gira para apontar e anda para frente.
        self._command[:, 0] = v
        self._command[:, 1] = 0.0
        self._command[:, 2] = w

    def _update_metrics(self) -> None:
        passos = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["v_comandada"] += self._command[:, :2].norm(dim=-1) / passos

    def dentro_do_morto(self) -> torch.Tensor:
        """[B] bool — o robô já está dentro do raio de chegada da tarefa.

        Exposto porque o item 11 precisa disso: dentro do `d_morto` a penalidade de
        velocidade em z tem que sair, senão ela briga com agachar pra pegar a caixa.
        Público de propósito — reward lendo atributo privado de command term é
        acoplamento que quebra em silêncio."""
        return self._dentro


@dataclass(kw_only=True)
class DesiredTwistCommandCfg(CommandTermCfg):
    meta_name: str = "lift_target"
    # Os defaults espelham `knobs.Command` (S5). Quem constrói o termo é o `env.py`, e
    # ele passa os valores de lá; estes só valem se alguém instanciar o cfg na mão.
    v_max: float = 0.5
    a_max: float = 1.0
    w_max: float = 1.2
    alpha_max: float = 3.0
    heading_gain: float = 0.5           # sem uso desde a S5; ver `knobs.Command`
    d_morto_andar: float = 0.05
    d_morto_manipula: float = 0.30
    d_freio_extra: float = 0.125        # = v_max²/2·a_max, afirmado por assert
    v_max_carga_cheia: float = 0.25     # S14 — v_max com 5 kg
    morto_angular_rad: float = 0.087

    def build(self, env: "ManagerBasedRlEnv") -> DesiredTwistCommand:
        return DesiredTwistCommand(self, env)
