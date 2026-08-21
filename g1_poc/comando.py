"""Os dois comandos do g1_poc.

`CaixaAlvoCommand` — 10 números (§6.2):
    [0:3] a posição do alvo, em MUNDO
    [3:6] `face_alvo`  — qual das 6 faces, no frame da caixa
    [6:9] `dir_alvo`   — para onde a normal dessa face aponta, no frame da base
    [9:10] `caixa_valida` — 0 ou 1

`TwistPoc` — o `UniformVelocityCommand` do mjlab, com dois acréscimos:
    (a) o twist é forçado a zero nos elos de manipulação, EXCETO no `carregar`;
    (b) metade dos envs PARADOS recebe giro no lugar.

Andar com a caixa aparece na cadeia `pegar` -> `carregar`, e em nenhum outro lugar. O
`_avanca_elo` sobe a mobília 5 m no fecho do `pegar` (§7.3): o robô pega, a mesa sai,
e ele anda livre.

ORDEM IMPORTA no dict de comandos: `caixa_alvo` vem PRIMEIRO, porque ele resolve
`env.poc_twist_zero`, e o `twist` lê esse buffer no mesmo passo.

A máquina de elo (§7): cadeia sorteada pela célula do nível; elo avança sem reset
de episódio; prateleira se move no fecho do pegar (§7.3); sucesso trava no último elo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# fatias do comando
ALVO = slice(0, 3)
FACE = slice(3, 6)
DIR = slice(6, 9)
VALIDA = slice(9, 10)
COMANDO_DIM = 10

# As 6 faces, como eixos no frame da CAIXA. As 4 primeiras são LATERAIS; as duas
# últimas são topo e fundo, e exigem tombar a caixa (salto qualitativo, nível 6).
FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)
N_LATERAIS = 4

# --- os elos e as cadeias (§7) ---
PEGAR, REORIENTAR, CARREGAR, BOTAR = 0, 1, 2, 3
# elos de cada cadeia, com -1 de padding. A ordem das cadeias é a de
# `knobs.Celulas.cadeias`: (pegar, reorientar->pegar, pegar->carregar, pegar->botar).
ELOS_DA_CADEIA = ((PEGAR, -1), (REORIENTAR, PEGAR), (PEGAR, CARREGAR), (PEGAR, BOTAR))
N_ELOS = (1, 2, 2, 2)


def _rot_z(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Roda `v` [B,3] em torno de z pelo ângulo `ang` [B]."""
    c, s = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(v)
    out[:, 0] = c * v[:, 0] - s * v[:, 1]
    out[:, 1] = s * v[:, 0] + c * v[:, 1]
    out[:, 2] = v[:, 2]
    return out


class CaixaAlvoCommand(CommandTerm):
    """Publica o alvo da caixa, o bit de validade, e resolve a forma do episódio."""

    cfg: CaixaAlvoCommandCfg

    def __init__(self, cfg: CaixaAlvoCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self.caixa: Entity = env.scene["box"]
        self.prateleira: Entity = env.scene["table"]

        self._command = torch.zeros(self.num_envs, COMANDO_DIM, device=self.device)
        self._face_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ang = torch.zeros(self.num_envs, device=self.device)
        self._pendente = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # os sites das palmas, para o σ inicial do `reaching` (§8.2)
        self._palm_ids, _ = self.robot.find_sites(list(cfg.palm_sites))
        self.reach_inicial = torch.full(
            (self.num_envs,), cfg.reaching_std_piso, device=self.device)

        # a distância comandada no começo do elo. Ela é o σ do `bringing` (§8.2).
        self.dist_inicial = torch.full(
            (self.num_envs,), cfg.bringing_std_piso, device=self.device)

        # sucesso TRAVADO. O episódio NÃO termina no sucesso (§7.5).
        self.episode_success = torch.zeros(self.num_envs, device=self.device)
        self._sustenta = torch.zeros(self.num_envs, device=self.device)

        # a forma do episódio. `manipula` é o inverso de `locomocao`.
        self.manipula = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # §11.2 — a JANELA DE ESPERA, em segundos restantes. Nasce em zero: antes do
        # primeiro resample nenhum episódio existe, e um valor positivo aqui faria
        # o passo 0 nascer aguardando sem ter sorteado nada.
        self._espera = torch.zeros(self.num_envs, device=self.device)

        # --- a máquina de elo (§7) ---
        self._cadeia = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elo_t = torch.zeros(self.num_envs, device=self.device)
        self.pegou = torch.zeros(self.num_envs, device=self.device)
        # σ variável do `precise_ori`: Δθ inicial do elo, com piso
        self.ori_inicial = torch.full(
            (self.num_envs,), cfg.precise_ori_std_piso, device=self.device)
        self._elos_tab = torch.tensor(ELOS_DA_CADEIA, dtype=torch.long, device=self.device)
        self._n_elos = torch.tensor(N_ELOS, dtype=torch.long, device=self.device)
        self._cadeias_tab = torch.tensor(cfg.cadeias, device=self.device)      # [7,4]
        self._ang_max = torch.deg2rad(torch.tensor(cfg.ang_max_deg, device=self.device))
        # o alvo de sustentação POR ELO (1,0 s no pegar; 0,5 s nos demais). Nasce
        # com o valor do pegar — antes do 1º reset nenhum termo o lê, mas zero aqui
        # explodiria a divisão do `sustentacao`.
        self._sust_alvo = torch.full(
            (self.num_envs,), cfg.sustenta_pegar_s, device=self.device)

        # buffers que outros managers leem. Existem ANTES do 1º reset, porque as
        # recompensas leem `env.poc_valida` no primeiro passo.
        env.poc_valida = self._command[:, VALIDA].squeeze(-1)
        # ⚠ publicados UMA vez e atualizados IN-PLACE — a lição do alias poc_success
        env.poc_elo = self._elo_id
        env.poc_pegou = self.pegou
        env.poc_ori_inicial = self.ori_inicial
        env.poc_twist_zero = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        env.poc_dist_inicial = self.dist_inicial
        env.poc_reach_inicial = self.reach_inicial
        env.poc_success = self.episode_success
        # §11.2 — quem está na janela de espera. Publicado UMA vez, atualizado
        # in-place (a lição do alias `poc_success`).
        env.poc_aguardando = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["erro_posicao"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["erro_angulo_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["no_alvo"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frac_manipula"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["aguardando"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["espera_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cadeia"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["elo"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pegou"] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ leitura
    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def valida(self) -> torch.Tensor:
        """[B] — 1,0 quando existe tarefa de caixa."""
        return self._command[:, 9]

    def erro_pos(self) -> torch.Tensor:
        return torch.norm(
            self._command[:, ALVO] - self.caixa.data.root_link_pos_w, dim=-1)

    def erro_ang(self) -> torch.Tensor:
        """Ângulo entre a normal da face alvo, em MUNDO, e o `dir_alvo`, em MUNDO.

        UM escalar, e a simetria do cubo se resolve sozinha: girar em torno da
        normal da face não muda o vetor. A métrica CONCORDA com o objetivo.
        (Mesma formulação do `erro_angulo_deg` do g1_multitask.)
        """
        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w, face_b)
        cos = torch.sum(normal_w * self._dir_w, dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cos)

    def de_pe(self) -> torch.Tensor:
        """[B] bool — a pelve alta e o tronco pouco inclinado."""
        z = self.robot.data.root_link_pos_w[:, 2]
        g = self.robot.data.projected_gravity_b
        inclinacao = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
        return (z >= self.cfg.pelve_min) & (inclinacao <= self.cfg.inclinacao_max_rad)

    # ------------------------------------------------------------------ resample
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return

        # --- a forma do episódio ---
        # ⚠ Ela é sorteada pelo CURRÍCULO (`curriculo.sorteia_forma`), e não aqui.
        # No reset a ordem do mjlab é currículo → eventos → comando, e o
        # `afasta_cena` precisa da forma ANTES deste método rodar.
        forma = getattr(self._env, "poc_manipula", None)
        if forma is None:
            manipula = torch.ones(n, dtype=torch.bool, device=self.device)
        else:
            manipula = forma[env_ids]
        self.manipula[env_ids] = manipula

        # --- a cadeia do episódio, pela célula do nível (§10.1) ---
        # ⚠ o FORÇADO vence sempre, mesmo sem `poc_nivel` (play/sonda/smoke)
        nivel = getattr(self._env, "poc_nivel", None)
        if self.cfg.cadeia_forcada is not None:
            # atalho de MEDIÇÃO (play/sonda/smoke); no treino fica None
            self._cadeia[env_ids] = int(self.cfg.cadeia_forcada)
        elif nivel is None:
            self._cadeia[env_ids] = 0
        else:
            probs = self._cadeias_tab[nivel[env_ids]]
            self._cadeia[env_ids] = torch.multinomial(probs, 1).squeeze(-1)
        self._elo_idx[env_ids] = 0
        self._elo_id[env_ids] = self._elos_tab[self._cadeia[env_ids], 0]
        self._elo_t[env_ids] = 0.0
        self.pegou[env_ids] = 0.0
        self._sust_alvo[env_ids] = torch.where(
            self._elo_id[env_ids] == PEGAR,
            torch.full_like(self._sust_alvo[env_ids], self.cfg.sustenta_pegar_s),
            torch.full_like(self._sust_alvo[env_ids], self.cfg.sustenta_outros_s))

        # --- o alvo do elo `pegar`, em MUNDO ---
        # Altura ABSOLUTA. Agachar não move este alvo, portanto o robô tem de
        # ficar de pé. É a decisão central do ADR-0001, e ela está certa.
        r = self.cfg.pegar_range
        lo = torch.tensor([r[0][0], r[1][0], r[2][0]], device=self.device)
        hi = torch.tensor([r[0][1], r[1][1], r[2][1]], device=self.device)
        alvo = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        # o x,y do alvo é relativo à origem do env; o z é do MUNDO
        origem = self._env.scene.env_origins[env_ids]
        alvo[:, 0] += origem[:, 0]
        alvo[:, 1] += origem[:, 1]
        self._command[env_ids, ALVO] = alvo

        # --- a face e o giro pedido ---
        # O `pegar` pede SEMPRE "erga sem torcer": `dir_alvo` recebe a normal ATUAL
        # da face. A rotação da célula do nível (§10.1) pertence ao elo
        # `reorientar` — pedi-la aqui tornaria as duas cadeias do nível 3 a mesma
        # tarefa, e o `reorientar` deixaria de ter função.
        self._face_idx[env_ids] = torch.randint(
            0, N_LATERAIS, (n,), device=self.device)
        # o giro depende do primeiro elo: PEGAR (cadeias 0,2,3) → 0; REORIENTAR (cadeia 1) → sorteado
        pegar_mask = self._elo_id[env_ids] == PEGAR
        reori_mask = self._elo_id[env_ids] == REORIENTAR
        if nivel is not None:
            ang_max = self._ang_max[nivel[env_ids]]
            self._ang[env_ids] = torch.where(
                pegar_mask,
                torch.zeros_like(self._ang[env_ids]),
                (2.0 * torch.rand(n, device=self.device) - 1.0) * ang_max)
        else:
            self._ang[env_ids] = 0.0

        # --- o bit ---
        self._command[env_ids, 9] = manipula.float()

        # --- §11.2 — a janela de espera, SORTEADA ---
        # Fixa seria aprendível como "conte N passos e depois mova". Sorteada, a
        # política tem de LER o canal de comando — que é o que o deploy exige.
        lo, hi = self.cfg.espera_s
        self._espera[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.device)

        # zera o sucesso e o cronômetro de sustentação
        self.episode_success[env_ids] = 0.0
        self._sustenta[env_ids] = 0.0
        self._pendente[env_ids] = True

    def _resolver(self) -> None:
        """Resolve, contra a pose FRESCA, o que o resample não podia resolver.

        ⚠ No reset o command manager roda DEPOIS dos eventos que reposicionam a
        caixa, mas as grandezas derivadas ainda não foram recalculadas. Ler pose
        no `_resample_command` devolve a pose ANTERIOR ao reset. Bug medido em
        30/07 no g1_multitask: o erro de ângulo saía espalhado em vez de exato.
        """
        if not bool(self._pendente.any()):
            return
        ids = self._pendente.nonzero().flatten()

        # o alvo do `reorientar` é a posição ATUAL da caixa (§7.1) — pose fresca
        reori = ids[self._elo_id[ids] == REORIENTAR]
        if len(reori) > 0:
            self._command[reori, ALVO] = self.caixa.data.root_link_pos_w[reori]

        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx[ids]]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], face_b)
        dir_w = _rot_z(normal_w, self._ang[ids])
        self._dir_w[ids] = dir_w / dir_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._command[ids, FACE] = face_b

        # σ do `bringing` = a distância a vencer no começo do elo (§8.2).
        # O σ fixo de 0,30 m do mjlab já está saturado quando a caixa sobe 0,17 m.
        d = torch.norm(
            self._command[ids, ALVO] - self.caixa.data.root_link_pos_w[ids], dim=-1)
        self.dist_inicial[ids] = torch.clamp(d, min=self.cfg.bringing_std_piso)

        # σ do `reaching` = a distância a vencer pelas PALMAS no começo do elo.
        # Mesma correção que a §8.2 fez no `bringing`: com σ fixo de 0,20 o
        # gradiente de aproximação cai 1391× entre a prateleira a 0,55 e a 0,04
        # (medido 20/08) — os níveis 3+ do currículo viravam sorte.
        from g1_poc.observacoes import alvos_das_palmas
        palmas = self.robot.data.site_pos_w[ids][:, self._palm_ids]
        alvos_p = alvos_das_palmas(self._env, "box", self.cfg.lateral_offset)[ids]
        d_p = torch.norm(palmas - alvos_p, dim=-1).mean(dim=-1)
        self.reach_inicial[ids] = torch.clamp(d_p, min=self.cfg.reaching_std_piso)

        # σ do `precise_ori` = o Δθ a vencer no começo do elo, com piso (§8.2)
        self.ori_inicial[ids] = torch.clamp(
            self.erro_ang()[ids], min=self.cfg.precise_ori_std_piso)

        self._pendente[ids] = False

    # -------------------------------------------------------------------- update
    def _update_command(self) -> None:
        self._resolver()

        # --- §11.2 — a janela de espera ---
        #
        # O contrato durante a espera é UM: "fique parado, e não existe tarefa".
        # Ele é entregue pelo BIT em zero, que já é o canal que significa isso.
        # Com o bit em zero as três fatias de caixa são zeradas pelo bloco logo
        # abaixo, e os NOVE termos de tarefa se desligam sozinhos — todos eles
        # multiplicam por `caixa_valida` (ver o docstring de `recompensas`).
        # Portanto a janela custa uma linha e não um canal novo: o contrato de
        # observação fica em 118 e o `expande_checkpoint` não é necessário.
        #
        # ⚠ O invariante "bit 0 ⟺ caixa a 5 m" fica RELAXADO durante a espera: nos
        # envs de manipulação a mobília já está posicionada. Isto é seguro porque
        # nada depende do invariante nesse intervalo — o twist é zero (o robô não
        # anda para dentro da mesa) e o `reaching` está desligado (não há prêmio de
        # graça). Na borda da espera o bit vai 0→1 com a caixa já no lugar, e essa
        # descontinuidade É o sinal de "o objetivo chegou".
        dt = self._env.step_dt
        self._espera.sub_(dt).clamp_(min=0.0)
        aguardando = self._espera > 0.0
        self._env.poc_aguardando.copy_(aguardando)
        self._command[:, 9] = (self.manipula & ~aguardando).float()

        # `dir_alvo` da OBS: o alvo vive em MUNDO e a obs é egocêntrica, portanto
        # a conversão é por passo. Sem isto a política veria o vetor do spawn, que
        # deixa de apontar para o lugar certo assim que o robô gira.
        self._command[:, DIR] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self._dir_w)

        # com o bit em 0 as três fatias de caixa são zeradas (§5.1.1)
        zero = self.valida.unsqueeze(-1) < 0.5
        self._command[:, FACE] = torch.where(
            zero, torch.zeros_like(self._command[:, FACE]), self._command[:, FACE])
        self._command[:, DIR] = torch.where(
            zero, torch.zeros_like(self._command[:, DIR]), self._command[:, DIR])

        # --- o alvo do `carregar` é do CORPO, recalculado a cada passo (§7.1) ---
        carrega = (self._elo_id == CARREGAR) & self.manipula
        if bool(carrega.any()):
            ids_c = carrega.nonzero().flatten()
            peito = torch.tensor(self.cfg.peito_b, device=self.device).expand(len(ids_c), 3)
            alvo_c = (self.robot.data.root_link_pos_w[ids_c]
                      + quat_apply(self.robot.data.root_link_quat_w[ids_c], peito))
            self._command[ids_c, ALVO] = alvo_c

        # O twist é zero em TODO elo de manipulação, exceto no `carregar`.
        #
        # ⚠ Não existe mais exceção além dessa. O `frac_twist_livre` saiu em 21/08:
        # ele liberava o twist com a prateleira ainda na frente do robô, a 0,25 m, e
        # sem exigir a preensão. Andar com a caixa aparece na cadeia
        # `pegar` -> `carregar`, onde o `_avanca_elo` sobe a mobília 5 m no fecho do
        # `pegar` (§7.3) — o robô pega, a mesa sai, e ele anda livre.
        #
        # ⚠ O `aguardando` entra com OR, e vence o `carrega`: na janela de espera
        # "parado" quer dizer velocidade linear E angular zero, sem exceção. O
        # `TwistPoc._update_command` aplica isto DEPOIS do `super()`, portanto vence
        # também o `heading_command`.
        self._env.poc_twist_zero.copy_(
            (self.manipula & ~carrega) | aguardando)

        # --- o fecho, condição a condição, POR ELO (§7.2) ---
        perto = self.erro_pos() < self.cfg.raio_sucesso
        alinhado = self.erro_ang() < self.cfg.angulo_sucesso_rad
        de_pe = self.de_pe()
        f_apoio = self._env.scene[self.cfg.support_sensor].data.force
        apoio_z = f_apoio[..., 2].abs().sum(dim=-1)
        massa = getattr(self._env, "poc_massa", None)
        if massa is None:
            apoiada = torch.zeros_like(perto)
        else:
            peso = (massa * 9.81).clamp(min=1e-3)
            apoiada = apoio_z >= self.cfg.fracao_apoio_botar * peso

        fecha = torch.zeros_like(perto)
        e = self._elo_id
        fecha |= (e == PEGAR) & perto & alinhado & de_pe
        fecha |= (e == REORIENTAR) & perto & alinhado
        fecha |= (e == CARREGAR) & perto
        fecha |= (e == BOTAR) & perto & alinhado & apoiada
        # ⚠ o fecho é CONGELADO na janela de espera. Sem isto o `pegar` poderia
        # fechar antes de ser pedido, e o `sustenta` acumularia contra um objetivo
        # que a política nem viu.
        fecha &= self.manipula & ~aguardando

        # o cronômetro do elo também não corre na espera: o `carregar` fecha por
        # TEMPO (`elo_t >= carregar_s`), e contar a espera ali daria 1 s de graça.
        self._elo_t += dt * (~aguardando).float()
        # ⚠ copy_, nunca atribuição: env.poc_success/poc_pegou são aliases in-place
        self._sustenta.copy_(torch.where(fecha, self._sustenta + dt,
                                         torch.zeros_like(self._sustenta)))

        sustentado = self._sustenta >= self._sust_alvo
        # o `carregar` fecha por TEMPO, sustentado: elo_t >= 6 s E perto por 0,5 s.
        # Um fecho INSTANTÂNEO com no_alvo ~57% seria uma moeda, e o nível viraria
        # passeio sem deriva (auditoria T16).
        sustentado &= (e != CARREGAR) | (self._elo_t >= self.cfg.carregar_s)

        ultimo = self._elo_idx + 1 >= self._n_elos[self._cadeia]
        fecha_elo = sustentado & self.manipula

        # o elo que fechou era `pegar`? arma a `caixa_largada` e o §7.3
        fechou_pegar = fecha_elo & (e == PEGAR)
        self.pegou.copy_(torch.maximum(self.pegou, fechou_pegar.float()))

        avanca = fecha_elo & ~ultimo
        if bool(avanca.any()):
            self._avanca_elo(avanca.nonzero().flatten())

        # sucesso TRAVADO no fecho do ÚLTIMO elo. O episódio continua (§7.5).
        self.episode_success.copy_(torch.maximum(
            self.episode_success, (fecha_elo & ultimo).float()))

    def _avanca_elo(self, ids: torch.Tensor) -> None:
        """Escreve o elo seguinte no comando, SEM reset (§7.5) e SEM resample.

        ⚠ Não usa `_resample_command`: aquele zera `episode_success` e sorteia
        cadeia nova. E não usa `_pendente`: aqui as poses JÁ estão frescas — o
        `_update_command` roda depois do `sim.forward()` do step.

        §7.3 — a prateleira se move quando o `pegar` fecha: +5 m na cadeia
        `carregar` (o chão fica livre para andar); topo NOVO na cadeia `botar`
        (a faixa é a da COLOCAÇÃO, 0,30-0,80, com teto efetivo no fundo da caixa
        menos a folga — a §7.3 prometia 0,55 e a §10.1 manda 0,80; sem o teto
        efetivo a laje nasceria DENTRO da caixa). Só a MESA se move: a caixa está
        nas mãos. A escrita de mocap vale a partir do passo seguinte.
        """
        n = len(ids)
        dev = self.device
        self._elo_idx[ids] += 1
        novo = self._elos_tab[self._cadeia[ids], self._elo_idx[ids]]
        self._elo_id[ids] = novo
        self._elo_t[ids] = 0.0
        self._sustenta[ids] = 0.0
        self._sust_alvo[ids] = torch.where(
            novo == PEGAR,
            torch.full((n,), self.cfg.sustenta_pegar_s, device=dev),
            torch.full((n,), self.cfg.sustenta_outros_s, device=dev))
        origem = self._env.scene.env_origins[ids]

        # --- PEGAR (2º elo da cadeia `reorientar`): alvo de mundo, "erga sem torcer"
        m = (novo == PEGAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            r = self.cfg.pegar_range
            lo = torch.tensor([r[0][0], r[1][0], r[2][0]], device=dev)
            hi = torch.tensor([r[0][1], r[1][1], r[2][1]], device=dev)
            alvo = lo + (hi - lo) * torch.rand(len(i), 3, device=dev)
            alvo[:, 0] += origem[m][:, 0]
            alvo[:, 1] += origem[m][:, 1]
            self._command[i, ALVO] = alvo
            self._ang[i] = 0.0

        # --- CARREGAR: mesa +5 m; o alvo do corpo é escrito a cada passo
        m = (novo == CARREGAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            pose = torch.zeros(len(i), 7, device=dev)
            pose[:, 0] = origem[m][:, 0] + self.cfg.prateleira_xy[0]
            pose[:, 1] = origem[m][:, 1] + self.cfg.prateleira_xy[1]
            pose[:, 2] = self.cfg.afasta_z - self.cfg.prateleira_meia_z
            pose[:, 3] = 1.0
            self.prateleira.write_mocap_pose_to_sim(pose, env_ids=i)
            if hasattr(self._env, "poc_topo"):
                self._env.poc_topo[i] = self.cfg.afasta_z
            self._ang[i] = 0.0

        # --- BOTAR: topo novo + alvo lateral em cima dele
        m = (novo == BOTAR).nonzero().flatten()
        if len(m) > 0:
            i = ids[m]
            fundo = self.caixa.data.root_link_pos_w[i, 2] - self.cfg.caixa_meia_z
            teto = torch.clamp(fundo - self.cfg.botar_folga_laje,
                               max=self.cfg.botar_topo_teto)
            piso = torch.full_like(teto, self.cfg.botar_topo_piso)
            teto = torch.maximum(teto, piso)   # nunca inverte a faixa
            topo = piso + (teto - piso) * torch.rand(len(i), device=dev)
            pose = torch.zeros(len(i), 7, device=dev)
            pose[:, 0] = origem[m][:, 0] + self.cfg.prateleira_xy[0]
            pose[:, 1] = origem[m][:, 1] + self.cfg.prateleira_xy[1]
            pose[:, 2] = topo - self.cfg.prateleira_meia_z
            pose[:, 3] = 1.0
            self.prateleira.write_mocap_pose_to_sim(pose, env_ids=i)
            if hasattr(self._env, "poc_topo"):
                self._env.poc_topo[i] = topo
            bx = self.cfg.botar_x
            by = self.cfg.botar_y
            alvo = torch.zeros(len(i), 3, device=dev)
            alvo[:, 0] = origem[m][:, 0] + bx[0] + (bx[1] - bx[0]) * torch.rand(len(i), device=dev)
            alvo[:, 1] = origem[m][:, 1] + by[0] + (by[1] - by[0]) * torch.rand(len(i), device=dev)
            alvo[:, 2] = topo + self.cfg.caixa_meia_z
            self._command[i, ALVO] = alvo
            self._ang[i] = 0.0

        # dir_alvo, σ do bringing, do reaching e do ori — contra a pose FRESCA
        face_b = torch.tensor(FACE_AXES, device=dev)[self._face_idx[ids]]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], face_b)
        dir_w = _rot_z(normal_w, self._ang[ids])
        self._dir_w[ids] = dir_w / dir_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._command[ids, FACE] = face_b
        d = torch.norm(self._command[ids, ALVO]
                       - self.caixa.data.root_link_pos_w[ids], dim=-1)
        self.dist_inicial[ids] = torch.clamp(d, min=self.cfg.bringing_std_piso)
        palmas = self.robot.data.site_pos_w[ids][:, self._palm_ids]
        from g1_poc.observacoes import alvos_das_palmas
        alvos_p = alvos_das_palmas(self._env, "box", self.cfg.lateral_offset)[ids]
        self.reach_inicial[ids] = torch.clamp(
            torch.norm(palmas - alvos_p, dim=-1).mean(dim=-1),
            min=self.cfg.reaching_std_piso)
        self.ori_inicial[ids] = torch.clamp(
            self.erro_ang()[ids], min=self.cfg.precise_ori_std_piso)

    def _update_metrics(self) -> None:
        """⚠ A DECOMPOSIÇÃO DO FECHO é obrigatória, e não enfeite.

        No fim do bloco 2 o `no_alvo` chegou a 57% e o `erro_posicao` a 0,0488 — abaixo
        do raio de sucesso — com `episode_success` em 0,0060. Com as métricas antigas
        não havia como saber qual das outras três condições bloqueava: o ângulo era
        reportado só como MÉDIA (14,2°, que passa na média e não diz a fração), e o
        `de_pe` e a sustentação não eram medidos de forma alguma.

        É o mesmo instrumento que faltou no g1_multitask, onde `cond_fisica = 0,0000`
        com todos os fatores aparentemente satisfeitos travou o diagnóstico.
        """
        self._resolver()
        v = self.valida
        perto = self.erro_pos() < self.cfg.raio_sucesso
        alinhado = self.erro_ang() < self.cfg.angulo_sucesso_rad
        de_pe = self.de_pe()

        self.metrics["erro_posicao"] = self.erro_pos() * v
        self.metrics["erro_angulo_deg"] = torch.rad2deg(self.erro_ang()) * v
        self.metrics["no_alvo"] = perto.float() * v
        # ⚠ CLONE, nunca o buffer cru: os outros metrics são snapshots (`× v`), e
        # este era um ALIAS — qualquer coisa que zerasse o buffer antes de o reset
        # ler o metric apagava o sucesso do log (foi o rastro que denunciou o wipe
        # do resample na it 5306: `pegou` 0,97 com `episode_success` 0,00).
        self.metrics["episode_success"] = self.episode_success.clone()
        # ⚠ 21/08: o divisor da desdiluição é `v`, e NÃO `self.manipula`. Ele tem de
        # ser EXATAMENTE a máscara que multiplica os outros metrics, senão a divisão
        # não condiciona nada. Com a janela de espera (§11.2) os dois deixaram de
        # coincidir: durante a espera `manipula` é 1 e `valida` é 0, e um episódio
        # que morre dentro da janela entraria no divisor sem entrar no numerador —
        # puxando todo número desdiluído para baixo.
        self.metrics["frac_manipula"] = v
        self.metrics["aguardando"] = self._env.poc_aguardando.float()
        self.metrics["espera_s"] = self._espera.clone()

        # os três fatores que faltavam, cada um como FRAÇÃO
        self.metrics["fecha_angulo"] = alinhado.float() * v
        self.metrics["fecha_de_pe"] = de_pe.float() * v
        self.metrics["fecha_todas"] = (perto & alinhado & de_pe).float() * v
        # `de_pe` tem duas partes e o `upright` já cobre a inclinação; a altura da pelve
        # é a que ninguém observava, e agachar para pegar é o que a derruba.
        # ⚠ SEM `× valida`: esta é uma altura em metros, para comparar direto com
        # `pelve_min = 0,65`. Multiplicar por `valida` misturaria zeros dos envs sem
        # caixa e puxaria a média para baixo — 0,768 apareceria como 0,576.
        self.metrics["pelve_z"] = self.robot.data.root_link_pos_w[:, 2]
        # separa "nunca fecha" de "fecha e perde": se `fecha_todas` for alto e isto
        # ficar perto de zero, o problema é ESTABILIDADE, não a condição.
        self.metrics["sustenta_s"] = self._sustenta * v
        self.metrics["cadeia"] = self._cadeia.float() * v
        self.metrics["elo"] = self._elo_id.float() * v
        self.metrics["pegou"] = self.pegou * v

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        for i in visualizer.get_env_indices(self.num_envs):
            if float(self.valida[i]) < 0.5:
                continue
            visualizer.add_sphere(
                center=self._command[i, ALVO].cpu().numpy(),
                radius=self.cfg.raio_sucesso,
                color=(1.0, 0.5, 0.0, 0.3),
                label=f"alvo_{i}",
            )


@dataclass(kw_only=True)
class CaixaAlvoCommandCfg(CommandTermCfg):
    # faixas do alvo do `pegar`, em MUNDO (x, y, z)
    pegar_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    raio_sucesso: float = 0.05
    angulo_sucesso_rad: float = math.radians(20.0)
    sustenta_pegar_s: float = 1.0
    pelve_min: float = 0.65
    inclinacao_max_rad: float = math.radians(20.0)
    bringing_std_piso: float = 0.10

    # σ inicial do `reaching` (§8.2): os sites das palmas e o piso do σ
    palm_sites: tuple[str, str] = ("left_palm", "right_palm")
    lateral_offset: float = 0.10
    reaching_std_piso: float = 0.20

    # --- a máquina de elo (§7) ---
    cadeias: tuple = ((1.0, 0.0, 0.0, 0.0),) * 7   # frações por nível; o env_cfg passa a tabela
    ang_max_deg: tuple[float, ...] = (0.0,) * 7    # rotação do `reorientar`, por nível
    sustenta_outros_s: float = 0.5
    carregar_s: float = 6.0
    fracao_apoio_botar: float = 0.80
    peito_b: tuple[float, float, float] = (0.25, 0.0, 0.15)
    botar_x: tuple[float, float] = (0.30, 0.40)
    botar_y: tuple[float, float] = (-0.12, 0.12)
    botar_topo_piso: float = 0.30
    botar_topo_teto: float = 0.80
    botar_folga_laje: float = 0.05
    caixa_meia_z: float = 0.10
    prateleira_meia_z: float = 0.02
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    afasta_z: float = 5.0
    support_sensor: str = "apoio_caixa"
    precise_ori_std_piso: float = 0.40
    # atalhos de MEDIÇÃO (play/sonda/smoke): força a cadeia; None = sorteio por nível
    cadeia_forcada: int | None = None
    # §11.2 — a janela de espera, em segundos. Sorteada por episódio.
    espera_s: tuple[float, float] = (0.3, 1.0)

    def build(self, env: ManagerBasedRlEnv) -> CaixaAlvoCommand:
        return CaixaAlvoCommand(self, env)


class TwistPoc(UniformVelocityCommand):
    """O twist do mjlab, com dois acréscimos.

    (a) ZERO nos elos de manipulação. Aplicado DEPOIS do `super()`, portanto vence
        o sorteio de standing e de heading.
    (b) GIRO NO LUGAR: metade dos envs parados recebe vx = vy = 0 e |ωz| ≥ piso.
        Sem isto os envs parados nunca giram, e o robô não aprende a girar no
        lugar. (Acréscimo do g1_multitask, e ele fica.)
    """

    cfg: TwistPocCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        # (b) giro no lugar, dentro do subconjunto parado
        parado = self.is_standing_env[env_ids]
        sorteio = torch.rand(len(env_ids), device=self.device)
        giro = parado & (sorteio < self.cfg.frac_giro_no_standing)
        ids_giro = env_ids[giro]
        if len(ids_giro) > 0:
            self.vel_command_b[ids_giro, 0] = 0.0
            self.vel_command_b[ids_giro, 1] = 0.0
            w = self.vel_command_b[ids_giro, 2]
            sinal = torch.where(w >= 0, 1.0, -1.0)
            piso = self.cfg.piso_giro_rad_s
            self.vel_command_b[ids_giro, 2] = sinal * torch.clamp(w.abs(), min=piso)
            # Sai das TRÊS máscaras. Só sair da de standing não basta: o
            # `_update_command` do fabricante sobrescreve o ωz dos envs de heading,
            # e reprojeta o vx,vy dos envs de mundo. Sem isto o piso de giro é
            # apagado em silêncio.
            self.is_standing_env[ids_giro] = False
            self.is_heading_env[ids_giro] = False
            self.is_world_env[ids_giro] = False

    def _update_command(self) -> None:
        super()._update_command()
        # (a) zero nos elos de manipulação
        zero = getattr(self._env, "poc_twist_zero", None)
        if zero is not None:
            self.vel_command_b[zero] = 0.0


@dataclass(kw_only=True)
class TwistPocCfg(UniformVelocityCommandCfg):
    frac_giro_no_standing: float = 0.5
    piso_giro_rad_s: float = 0.15

    def build(self, env: ManagerBasedRlEnv) -> TwistPoc:
        return TwistPoc(self, env)
