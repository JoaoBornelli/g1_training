"""Monta a entrada do BFM-Zero a partir dos tensores do mjlab, em lote.

O `env.py` do BFM-Zero é numpy e roda UM robô: ele lê `mjd.qpos` e usa `np.roll`.
Aqui a mesma conta sai de `robot.data.*` para N ambientes, em torch, na GPU.

Erro aqui NÃO levanta exceção. Só sai comportamento ruim. Por isso o
`referencia.py` compara este caminho com o do fabricante, estado por estado.

As cinco coisas que o caminho original faz e são fáceis de perder:

1. **`dof_pos` é desvio da pose padrão DO BFM**, não da nossa. As duas diferem em
   até **34,4°** (joelho 0,669 contra 0,300; cotovelo 0,600 contra 0,000). Usar a
   nossa põe o ator num ponto que ele nunca viu.
2. **`ang_vel` é multiplicado por 0,25** (`env.py:378`).
3. **`last_action` é guardado multiplicado por 5,0** — o `action_rescale`, não o
   `action_scales` (`env.py:288`). O histórico de ação carrega essa escala também.
4. **O histórico tem o mais recente no índice 0** e é inicializado com ZEROS.
5. **O cronograma dos 4 slots NÃO é `[t, t-1, t-2, t-3]`.** Ele é:

       slot0 = x_t      slot1 = x_{t-1}      slot2 = x_{t-1}      slot3 = x_{t-2}

   A duplicata nos slots 1 e 2 é real e vale para os CINCO componentes. Ela vem do
   `step()` do BFM chamar `_create_observation()` duas vezes — antes e depois da
   física — e o `teste_sim.py` usar só o segundo. O estado "antes da física" no
   passo `t` é igual ao "depois" do passo `t-1`, porque nada acontece entre os dois,
   então a segunda rolagem duplica em vez de trazer informação nova.

   Isso foi **medido**, não deduzido: o `dump_referencia.py` grava os dois obs e o
   `referencia.py` identifica cada slot contra a sequência de estados.

   ⚠️ Minhas DUAS primeiras tentativas estavam erradas. Rolar duas vezes no mesmo
   instante dá `[x_t, x_t, x_{t-1}, x_{t-1}]` (a duplicata no lugar errado, e sem
   `x_{t-2}`); rolar uma vez dá `[x_t, x_{t-1}, x_{t-2}, x_{t-3}]` (sem duplicata).
   O primeiro apaga a informação de movimento e faz o robô se debater.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_residual.bfm import DIM_ESTADO, DIM_HISTORICO, PASSOS_HISTORICO  # noqa: E402

ESCALA_ANG_VEL = 0.25
"""`env.py:378` — `ang_vel = qvel[3:6] * 0.25`."""

RESCALE = 5.0
"""`action_rescale`. Entra DUAS vezes e por motivos diferentes: no alvo de junta
(`a * action_scales * 5.0`) e no `last_action` guardado (`a * 5.0`, sem o
`action_scales`)."""


class ObsBFM:
    """Buffers de histórico e montagem da entrada, para N ambientes.

    Uso, uma vez por passo de controle:

        estado, ultima_acao, historico = obs.monta()
        a_bfm = ator(estado, ultima_acao, historico, z)
        obs.guarda_acao(a_bfm)
    """

    PROFUNDIDADE = 3
    """Os buffers guardam 3 passos, não 4. O 4º slot é uma DUPLICATA do 2º."""

    ORDEM = (0, 1, 1, 2)
    """`[x_t, x_{t-1}, x_{t-1}, x_{t-2}]` — o cronograma medido do BFM."""

    def __init__(self, env, padrao_bfm: torch.Tensor):
        self._robot = env.scene["robot"]
        self._n = env.num_envs
        self._dev = env.device
        self._padrao = padrao_bfm.to(self._dev).view(1, -1)   # [1, 29]

        z = lambda *s: torch.zeros(*s, device=self._dev)      # noqa: E731
        self.ultima_acao = z(self._n, 29)                     # já x RESCALE
        P = self.PROFUNDIDADE
        self.h_acao = z(self._n, P, 29)
        self.h_ang_vel = z(self._n, P, 3)
        self.h_dof_pos = z(self._n, P, 29)
        self.h_dof_vel = z(self._n, P, 29)
        self.h_grav = z(self._n, P, 3)

    # ------------------------------------------------------------------ leitura
    def _agora(self):
        d = self._robot.data
        # `joint_pos` cru, não o `joint_pos_biased`: o BFM foi treinado sem viés de
        # encoder, então o viés fica só no caminho de escrita (`apply_actions` já
        # o subtrai). Trocar para o enviesado é a variante sim-to-real.
        dof_pos = d.joint_pos - self._padrao
        dof_vel = d.joint_vel
        grav = d.projected_gravity_b
        # `root_link_ang_vel_b`: o `qvel[3:6]` do freejoint do MuJoCo é a angular
        # do corpo no frame LOCAL, que é exatamente este campo.
        ang_vel = d.root_link_ang_vel_b * ESCALA_ANG_VEL
        return dof_pos, dof_vel, grav, ang_vel

    def monta(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devolve `(estado[N,64], ultima_acao[N,29], historico[N,372])`."""
        dof_pos, dof_vel, grav, ang_vel = self._agora()
        for buf, val in ((self.h_acao, self.ultima_acao),
                         (self.h_ang_vel, ang_vel),
                         (self.h_dof_pos, dof_pos),
                         (self.h_dof_vel, dof_vel),
                         (self.h_grav, grav)):
            buf.copy_(buf.roll(1, dims=1))
            buf[:, 0] = val

        estado = torch.cat([dof_pos, dof_vel, grav, ang_vel], dim=-1)
        # A ordem do concat é a do YAML do fabricante (`env.py:421`) e não é
        # alfabética: ação, ang_vel, dof_pos, dof_vel, gravidade. E cada buffer sai
        # reindexado por `ORDEM = (0, 1, 1, 2)`, que é o cronograma medido.
        o = list(self.ORDEM)
        historico = torch.cat([self.h_acao[:, o].reshape(self._n, -1),
                               self.h_ang_vel[:, o].reshape(self._n, -1),
                               self.h_dof_pos[:, o].reshape(self._n, -1),
                               self.h_dof_vel[:, o].reshape(self._n, -1),
                               self.h_grav[:, o].reshape(self._n, -1)], dim=-1)
        assert estado.shape[-1] == DIM_ESTADO, estado.shape
        assert historico.shape[-1] == DIM_HISTORICO, historico.shape
        return estado, self.ultima_acao, historico

    # ------------------------------------------------------------------ escrita
    def guarda_acao(self, acao_bfm: torch.Tensor) -> None:
        """Guarda a ação do BFM para o próximo passo, já com o `x 5.0`."""
        self.ultima_acao = acao_bfm * RESCALE

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zera histórico e última ação — o BFM começa episódio com zeros."""
        if env_ids is None:
            env_ids = slice(None)
        self.ultima_acao[env_ids] = 0.0
        for buf in (self.h_acao, self.h_ang_vel, self.h_dof_pos,
                    self.h_dof_vel, self.h_grav):
            buf[env_ids] = 0.0
