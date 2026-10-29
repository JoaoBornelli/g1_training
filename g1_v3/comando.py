"""Termos de comando da v3: `cadeia` (tarefa, trechos, odometria) e `twist` (velocidade por célula, sucesso
por segmento). Spec: S §7 (cadeia), §11.2 (tetos), §11.3 (sucesso), §19 conflito 5; plano D3, §6.3, §6.4, §6.5.
PD-15 (âncora do canal zero), PD-44 (quem dá o passo da odometria), PD-50 (sucesso sem perda).

Os núcleos (`sorteia_velocidade`, `monta_tabela_trechos`, `avanca_trecho`, `tempo_da_ancora`,
`sucesso_do_segmento`) são puros: sobre tensores, sem o env. Os invólucros (`CadeiaCmd`, `TwistV3`) leem o env.
Contagem: g1_limpo/comando.py tinha 2 526 linhas, e a v3 copia só o que a D3 pede (a eficiência por segmento
e a razão de marcha ficam só como diagnóstico, que o §6.7 loga).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from g1_v3.curriculo import garante_v3
from g1_v3.knobs import TAREFAS, Cadeia, Comando, Estimador

# Índices de TAREFAS (knobs.py): o one-hot da §6.5 e a tabela de trechos usam esta ordem.
ANDAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3
# Índices de FATIAS (knobs.py): ANDAR, B, C.
FATIA_ANDAR, FATIA_B, FATIA_C = 0, 1, 2
# Folga de float no relógio: 25 passos de 0,02 s somam 0,4999… e não podem perder o aviso.
# Não é número de treino (mesmo papel do `_EPS` de alvos.py).
_EPS = 1e-6


# --- sorteio de velocidade (§6.3, D3 item 1) ----------------------------------------------------------

def _u(n, g):  # U[0,1) no device do gerador (None: default)
    return torch.rand(n, generator=g, device=None if g is None else g.device)


def _anda(n, lo, frente, tras, g):
    # 4 direções uniformes; módulo U[lo; teto], teto = `frente` para +x e `tras` para as outras três (S §11.2)
    eixo = torch.randint(0, 4, (n,), generator=g, device=None if g is None else g.device)
    teto = tras + (frente - tras) * (eixo == 0).float()
    mod = lo + (teto - lo) * _u(n, g)
    vx = mod * ((eixo == 0).float() - (eixo == 1).float())
    vy = mod * ((eixo == 2).float() - (eixo == 3).float())
    return torch.stack([vx, vy], dim=1)


def _giro(n, lo, hi, g):  # ±U[lo; hi] (sinal sorteado por env)
    sinal = 2.0 * (_u(n, g) >= 0.5).float() - 1.0
    return sinal * (lo + (hi - lo) * _u(n, g))


def sorteia_velocidade(tipo, nivel, tarefa, n, gerador, k: Comando) -> torch.Tensor:
    """(n,3) de (vx, vy, wz). `tarefa` não muda a faixa: o `tipo` já diz o que anda (o parâmetro fica na §6.3)."""
    frente = k.frente_tetos[nivel]
    tras = min(frente, k.tras_lado_teto_max)
    giro = k.giro_tetos[min(nivel, 3)]
    if tipo == "terreno":  # PD-12: o terreno usa o reto do nível, com teto de `terreno_v_max`
        frente, tras = min(frente, k.terreno_v_max), min(tras, k.terreno_v_max)
    v = torch.zeros(n, 3, device=None if gerador is None else gerador.device)
    if tipo == "parado":
        return v
    if tipo == "baixo":  # S §19, conflito 5: metade linear, metade giro no lugar
        lin = _anda(n, k.baixo[0], k.baixo[1], k.baixo[1], gerador)
        gir = _giro(n, k.baixo[0], k.baixo[1], gerador)
        metade = _u(n, gerador) < 0.5
        v[:, :2] = torch.where(metade[:, None], lin, torch.zeros_like(lin))
        v[:, 2] = torch.where(metade, torch.zeros_like(gir), gir)
        return v
    # reto/terreno: só linear; giro_andando/arco: linear + giro; giro_lugar: só giro (linear 0, §11.2)
    if tipo in ("reto", "terreno", "giro_andando", "arco"):
        v[:, :2] = _anda(n, k.piso_velocidade, frente, tras, gerador)
    if tipo in ("giro_andando", "arco", "giro_lugar"):
        v[:, 2] = _giro(n, k.piso_velocidade, giro, gerador)
    return v


# --- cadeia de trechos (§6.3, D3 item 2) --------------------------------------------------------------

@dataclass
class TabelaTrechos:  # a tabela da D3 achatada nas três fatias; tensores (T,)
    tarefa: torch.Tensor  # long, índice em TAREFAS publicado no trecho
    por_detector: torch.Tensor  # bool: o trecho acaba pelo aviso do detector mais o atraso
    proximo: torch.Tensor  # long, −1 = fim da cadeia
    inicio: torch.Tensor  # (3,) long, o primeiro trecho de ANDAR, B e C


def _monta(k: Cadeia, device):
    # Tabela da D3 (S §7). Cada trecho: (tarefa, por_detector, faixa de sorteio em s). Para trecho por
    # detector a faixa é o atraso; para trecho por tempo, a duração. Faixa (a, a) é valor fixo.
    espera, fim = k.espera_s, (k.andar_final_s, k.andar_final_s)
    fatias = (
        [("ANDAR", False, (k.episodio_s, k.episodio_s))],
        [("ANDAR", False, espera), ("PEGAR", True, k.atraso_pegar_s), ("CARREGAR", False, k.carregar_b_s)],
        [("ANDAR", False, espera), ("PEGAR", True, k.atraso_pegar_s), ("CARREGAR", False, k.carregar_c_s),
         ("BOTAR", True, k.atraso_botar_s), ("ANDAR", False, fim)],
    )
    tar, det, fx, prox, inicio = [], [], [], [], []
    for cadeia in fatias:
        inicio.append(len(tar))
        for j, (nome, d, faixa) in enumerate(cadeia):
            prox.append(len(tar) + 1 if j < len(cadeia) - 1 else -1)
            tar.append(TAREFAS.index(nome))
            det.append(d)
            fx.append(faixa)
    tab = TabelaTrechos(
        tarefa=torch.tensor(tar, dtype=torch.long, device=device),
        por_detector=torch.tensor(det, dtype=torch.bool, device=device),
        proximo=torch.tensor(prox, dtype=torch.long, device=device),
        inicio=torch.tensor(inicio, dtype=torch.long, device=device))
    return tab, torch.tensor(fx, dtype=torch.float32, device=device)


def monta_tabela_trechos(k_cad: Cadeia, device) -> TabelaTrechos:
    return _monta(k_cad, device)[0]


def avanca_trecho(trecho, relogio, duracao, aviso, atraso, dt, tabela: TabelaTrechos):
    """(novo_trecho, novo_relogio, avancou, fim). Por tempo: relógio corre sempre, avança com relógio ≥ duração.
    Por detector: relógio corre só com o aviso, avança com aviso e relógio ≥ atraso. Ao avançar, relógio = 0.
    A tarefa publicada é `tabela.tarefa[trecho]`: no atraso do BOTAR, o one-hot fica em BOTAR (§6.3).
    ⚠ O aviso é exigido também para avançar: com atraso 0, um trecho por detector sem aviso não pode avançar."""
    por = tabela.por_detector[trecho]
    alvo = torch.where(por, atraso, duracao)
    rel = torch.where(por, torch.where(aviso, relogio + dt, relogio), relogio + dt)
    avancou = rel >= alvo - _EPS
    avancou = torch.where(por, avancou & aviso, avancou)
    nxt = tabela.proximo[trecho]
    fim = avancou & (nxt < 0)
    novo_trecho = torch.where(avancou & (nxt >= 0), nxt, trecho)
    novo_relogio = torch.where(avancou, torch.zeros_like(rel), rel)
    return novo_trecho, novo_relogio, avancou, fim


# --- sucesso por segmento (§6.3, D3 item 4; PD-15, PD-50) ----------------------------------------------

def tempo_da_ancora(t_abertura, t_empurrao, ancora_s) -> torch.Tensor:
    # PD-15: a âncora fica `ancora_s` depois da abertura, e de novo depois de cada empurrão.
    # Sem empurrão, t_empurrao = −∞ e vale a abertura.
    return torch.maximum(t_abertura, t_empurrao) + ancora_s


def sucesso_do_segmento(cmd, int_proj, int_cmd, deriva_pos, deriva_rumo, t_fecho, t_ancora, perdeu,
                        carregar, k: Comando):
    """(registra, sucesso), (E,) bool. Canal ativo (|c| ≥ limiar): razão ≥ portão. Canal zero: deriva da âncora.
    ⚠ `carregar` é "tarefa com caixa" (PEGAR, CARREGAR, BOTAR): PD-50, o registro exige não perder a caixa.
    O nome vem da assinatura da §6.3."""
    ativo_lin = cmd[:, :2].norm(dim=1) >= k.limiar_canal
    ativo_giro = cmd[:, 2].abs() >= k.limiar_canal
    razao = int_proj / int_cmd.clamp(min=1e-6)
    ok_lin = torch.where(ativo_lin, razao[:, 0] >= k.razao_portao, deriva_pos < k.deriva_pos_m)
    ok_giro = torch.where(ativo_giro, razao[:, 1] >= k.razao_portao, deriva_rumo < k.deriva_rumo_rad)
    sucesso = ok_lin & ok_giro & ~(carregar & perdeu)
    registra = t_fecho >= t_ancora  # um segmento que fecha antes da âncora não entra (PD-15)
    return registra, sucesso


# --- forca_de_apoio (cópia de g1_limpo/comando.py:187-220) ---------------------------------------------

def forca_de_apoio(env, nome_sensor: str, eixo: int = 2) -> torch.Tensor:
    """[n] — quanto a LAJE carrega da caixa, em newtons. Só a componente VERTICAL (eixo 2, global).

    ⚠ Projeção no eixo vertical, e não a norma: prensar a caixa de lado satisfazia a norma e o BOTAR fechava
    sem a laje carregar peso. Decisão do dono (03/09, depois de code review); o `g1_poc` já projetava.
    ⚠ `abs` no z é medido: a força sai com o sinal invertido do que a caixa sente. Com `abs` o pior caso é o
    buraco pequeno de prensar por baixo; sem ele, a tarefa morre calada se o mjlab inverter a ordem dos geoms.
    ⚠ `sum` sobre os slots: com `num_slots=1` é um número só, e a soma continua certa se alguém subir os slots.
    """
    f = env.scene[nome_sensor].data.force
    assert f is not None, f"sensor '{nome_sensor}' precisa do field 'force'."
    return f[..., eixo].abs().sum(dim=-1)


# --- termo `cadeia` ------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class CadeiaCmdCfg(CommandTermCfg):
    k: Cadeia
    est: Estimador  # magnitude da odometria (fase D: a de início, PD-18)
    entity_name: str
    resampling_time_range: tuple = (float("inf"), float("inf"))  # o trecho é que decide quando trocar

    def build(self, env) -> CadeiaCmd:
        return CadeiaCmd(self, env)


class CadeiaCmd(CommandTerm):
    """Tarefa publicada e trechos da fatia (S §7). Escreve v3_trecho, v3_tarefa, v3_fim, v3_aviso, v3_freio,
    v3_degrau; dá o `passo` da `v3_odo` uma vez por passo (PD-44)."""

    cfg: CadeiaCmdCfg

    def __init__(self, cfg: CadeiaCmdCfg, env):
        super().__init__(cfg, env)
        garante_v3(env)
        self.robot = env.scene[cfg.entity_name]
        self.tab, self.faixas = _monta(cfg.k, self.device)
        self.relogio = torch.zeros(self.num_envs, device=self.device)
        self.alvo_s = torch.zeros(self.num_envs, device=self.device)  # duração (ou atraso) sorteada do trecho
        k = cfg.est
        # magnitude "início" (PD-18): índice 0 de cada par; o refino entra com o degrau (fase G)
        self.deriva = {"odo": k.deriva_odo[0], "rumo": k.deriva_rumo[0],
                       "pos_giro": k.deriva_pos_giro[0], "rumo_giro": k.deriva_rumo_giro[0]}

    @property
    def command(self) -> torch.Tensor:  # one-hot (E,6) de v3_tarefa (§6.5)
        return torch.nn.functional.one_hot(self._env.v3_tarefa, len(TAREFAS)).float()

    def _abre_trecho(self, ids, trecho) -> None:
        # Sorteia a duração (ou atraso) do trecho, publica a tarefa e o freio/degrau, e reamostra o comando
        # do twist (o CARREGAR de B fica fixo na abertura, §7; o parado sai com comando zero).
        env = self._env
        lo, hi = self.faixas[trecho, 0], self.faixas[trecho, 1]
        self.alvo_s[ids] = lo + (hi - lo) * _u(len(ids), env.v3_gerador).to(self.device)
        self.relogio[ids] = 0.0
        env.v3_trecho[ids] = trecho
        tar = self.tab.tarefa[trecho]
        env.v3_tarefa[ids] = tar
        cur = env.v3_cur
        freio = torch.zeros_like(tar)
        degrau = torch.zeros_like(tar)
        for i, nome in enumerate(TAREFAS[:4]):  # REORIENTAR e LEVANTAR ficam em 0 (reservados)
            m = tar == i
            freio[m] = cur.niveis_abertos(nome)
            degrau[m] = cur.degrau(nome)
        env.v3_freio[ids] = freio
        env.v3_degrau[ids] = degrau
        self._twist()._resample(ids)

    def _twist(self):
        return self._env.command_manager.get_term("twist")

    def _resample(self, env_ids) -> None:  # chamado no reset do episódio (CommandTerm.reset)
        env = self._env
        fat = env.v3_fatia[env_ids]
        self._abre_trecho(env_ids, self.tab.inicio.to(fat.device)[fat])
        env.v3_fim[env_ids] = False

    def compute(self, dt: float) -> None:
        env = self._env
        # ⚠ Fase D: o detector de fim (alvos.detecta_fim) ainda não entra; o aviso é sempre falso, e por isso
        # as fatias B e C nunca avançam (elas não abrem nesta fase).
        aviso = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        novo, self.relogio, avancou, fim = avanca_trecho(
            env.v3_trecho, self.relogio, self.alvo_s, aviso, self.alvo_s, dt, self.tab)
        env.v3_aviso[:] = aviso
        env.v3_fim |= fim
        ids = (avancou & ~fim).nonzero().flatten()  # no fim da cadeia o trecho não reabre (S §7)
        if ids.numel():
            self._abre_trecho(ids, novo[ids])
        env.v3_odo.passo(self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w, self.deriva)

    # abstratos do CommandTerm, sem uso aqui: o trecho avança em `compute` e o comando é do twist
    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids) -> None:
        pass

    def _update_command(self) -> None:
        pass


# --- termo `twist` -------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class TwistV3Cfg(UniformVelocityCommandCfg):
    k: Comando
    rel_heading_envs: float = 0.0  # o `parado` é célula, não sorteio do fabricante
    rel_standing_envs: float = 0.0

    def build(self, env) -> TwistV3:
        return TwistV3(self, env)


_METRICAS = ("razao_marcha", "eficiencia_min", "eficiencia_media", "segmentos")


class TwistV3(UniformVelocityCommand):
    """Cópia do TwistComRazaoDeMarcha (g1_limpo/comando.py:2276) com o comando por célula e o sucesso por segmento.
    ⚠ Não chama `super()._resample_command`: o sorteio é da célula (§6.3), e o `standing`/`turning` do fabricante
    ficam fora. As `error_vel_*` do fabricante seguem em `metrics` (§6.7); o juiz é o sucesso por segmento.
    ⚠ As somas da razão de marcha são acumuladores internos, fora de `metrics`: o log não as recebe."""

    cfg: TwistV3Cfg

    def __init__(self, cfg: TwistV3Cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.k = cfg.k
        for m in _METRICAS:
            self.metrics[m] = torch.zeros(n, device=dev)
        self.soma_erro_marcha = torch.zeros(n, device=dev)
        self.soma_cmd_marcha = torch.zeros(n, device=dev)
        # média da base (vx, vy, wz) nos últimos `janela_media_s` (S §19, conflito 5); zera só no reset
        self.v_media_b = torch.zeros(n, 3, device=dev)
        self._hist = torch.zeros(n, round(self.k.janela_media_s / env.step_dt), 3, device=dev)
        self._i = 0
        # relógio do episódio, próprio: o reset do mjlab zera episode_length_buf DEPOIS do command_manager.reset
        # (manager_based_rl_env.py:581-590; §6.4), por isso não se lê o episode_length_buf aqui
        self._agora = torch.zeros(n, device=dev)
        # segmento aberto: snapshot do comando, célula e "tarefa com caixa" no instante da abertura
        self.seg_cmd = torch.zeros(n, 3, device=dev)
        self.seg_cel = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.seg_carrega = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seg_proj = torch.zeros(n, 2, device=dev)  # [linear, giro]: Σ(v·ĉ)dt, Σ(ω·sinal)dt
        self.seg_int = torch.zeros(n, 2, device=dev)  # [linear, giro]: Σ|c|dt, Σ|wz|dt
        self.t_ab = torch.zeros(n, device=dev)  # instante da abertura do segmento (base da âncora)
        self.ancora_feita = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seg_visto = self.command_counter.clone()
        self._abre(torch.arange(n, device=dev))

    def _abre(self, ids) -> None:
        env = self._env
        self.t_ab[ids] = self._agora[ids]
        self.seg_cmd[ids] = self.vel_command_b[ids]
        self.seg_cel[ids] = env.v3_celula[ids]
        self.seg_carrega[ids] = env.v3_tarefa[ids] != ANDAR
        self.seg_visto[ids] = self.command_counter[ids]
        self.seg_proj[ids] = 0.0
        self.seg_int[ids] = 0.0
        self.ancora_feita[ids] = False

    def _fecha(self, ids, falha) -> None:
        # Fecha o segmento de `ids` com a pose e o relógio do instante; `falha` (terminação que não é
        # time_out) zera o sucesso e registra mesmo antes da âncora (D3 item 4; S §11.3 "sem queda").
        env, k = self._env, self.k
        ancora = env.v3_ancora[ids]
        pos = self.robot.data.root_link_pos_w[ids, :2]
        dpos = (pos - ancora[:, :2]).norm(dim=1)
        drum = wrap_to_pi(self.robot.data.heading_w[ids] - ancora[:, 2]).abs()
        reg, suc = sucesso_do_segmento(
            self.seg_cmd[ids], self.seg_proj[ids], self.seg_int[ids], dpos, drum,
            self._agora[ids], ancora[:, 3], env.v3_perdeu[ids], self.seg_carrega[ids], k)
        suc = suc & ~falha
        reg = reg | falha
        cel = self.seg_cel[ids]
        for c in cel[reg & (cel >= 0)].unique().tolist():  # uma chamada por célula (registra é escalar)
            m = reg & (cel == c)
            env.v3_cur.registra(int(c), int(m.sum()), int((m & suc).sum()))
        # eficiência por segmento (diagnóstico do §6.7): Σ(v·ĉ)/Σ|c| do linear, só com comando de pelo menos
        # `pedido_min_segmento` (ruído do denominador)
        ped = self.seg_int[ids, 0]
        vale = ped >= k.pedido_min_segmento
        e = self.seg_proj[ids, 0] / ped.clamp(min=1e-6)
        n = self.metrics["segmentos"][ids]
        mn, med = self.metrics["eficiencia_min"][ids], self.metrics["eficiencia_media"][ids]
        self.metrics["eficiencia_min"][ids] = torch.where(vale, torch.where(n == 0, e, torch.minimum(mn, e)), mn)
        self.metrics["eficiencia_media"][ids] = torch.where(vale, (med * n + e) / (n + 1.0), med)
        self.metrics["segmentos"][ids] = n + vale.float()
        self.seg_proj[ids] = 0.0
        self.seg_int[ids] = 0.0

    def _update_metrics(self) -> None:
        env, k = self._env, self.k
        dt = env.step_dt
        self._agora += dt
        # 1. fronteira de segmento: o contador mudou desde o último passo (o cadeia reamostrou)
        mudou = (self.command_counter != self.seg_visto).nonzero().flatten()
        if mudou.numel():
            self._fecha(mudou, falha=torch.zeros(len(mudou), dtype=torch.bool, device=self.device))
            self._abre(mudou)
        # 2. acumula o passo no segmento corrente (o comando é o do segmento)
        vel = self.robot.data.root_link_lin_vel_b[:, :2]
        wz = self.robot.data.root_link_ang_vel_b[:, 2]
        cmd = self.vel_command_b
        n_lin = cmd[:, :2].norm(dim=1)
        self.seg_proj[:, 0] += (vel * cmd[:, :2] / n_lin.clamp(min=1e-6)[:, None]).sum(dim=1) * dt
        self.seg_int[:, 0] += n_lin * dt
        sinal = torch.where(cmd[:, 2] >= 0, 1.0, -1.0)
        self.seg_proj[:, 1] += wz * sinal * dt
        self.seg_int[:, 1] += cmd[:, 2].abs() * dt
        # 3. âncora (PD-15): instante = max(abertura, último empurrão) + ancora_s; a pose é gravada nesse instante
        novo = tempo_da_ancora(self.t_ab, env.v3_t_empurrao, k.ancora_s)
        self.ancora_feita &= novo == env.v3_ancora[:, 3]  # instante novo (empurrão) regrava a pose
        env.v3_ancora[:, 3] = novo
        pronta = (~self.ancora_feita & (self._agora >= novo)).nonzero().flatten()
        if pronta.numel():
            env.v3_ancora[pronta, 0] = self.robot.data.root_link_pos_w[pronta, 0]
            env.v3_ancora[pronta, 1] = self.robot.data.root_link_pos_w[pronta, 1]
            env.v3_ancora[pronta, 2] = self.robot.data.heading_w[pronta]
            self.ancora_feita[pronta] = True
        # 4. diagnósticos (§6.7): error_vel_* do fabricante, razão de marcha fora do portão; média da base
        super()._update_metrics()
        ativo = (n_lin > k.limiar_canal).float()
        self.soma_erro_marcha += (cmd[:, :2] - vel).norm(dim=1) * ativo
        self.soma_cmd_marcha += n_lin * ativo
        soma = self.soma_cmd_marcha
        self.metrics["razao_marcha"][:] = torch.where(
            soma > 0.0, 1.0 - self.soma_erro_marcha / soma.clamp(min=1e-6), torch.zeros_like(soma))
        self._hist[:, self._i] = torch.cat([vel, wz[:, None]], dim=1)
        self._i = (self._i + 1) % self._hist.shape[1]
        self.v_media_b = self._hist.mean(dim=1)

    def reset(self, env_ids):
        # ⚠ A cadeia roda ANTES do twist (§6.5): o trecho novo já está publicado, por isso o segmento antigo usa o
        # snapshot da abertura (seg_cel, seg_carrega), e não os buffers vivos.
        time_out = self._env.termination_manager.time_outs[env_ids]
        self._fecha(env_ids, falha=~time_out)
        self._hist[env_ids] = 0.0
        self.v_media_b[env_ids] = 0.0
        self._agora[env_ids] = 0.0
        self.soma_erro_marcha[env_ids] = 0.0
        self.soma_cmd_marcha[env_ids] = 0.0
        extras = super().reset(env_ids)  # zera as métricas e o contador, e reamostra (chama `_resample`)
        self._abre(env_ids)
        return extras

    def _resample_command(self, env_ids) -> None:
        env = self._env
        g = env.v3_gerador
        tar, fat = env.v3_tarefa[env_ids], env.v3_fatia[env_ids]
        env.v3_celula[env_ids] = -1  # sem célula, sem registro: parado de trecho, PEGAR e BOTAR (fase D)
        self.vel_command_b[env_ids] = 0.0
        # anda: ANDAR na fatia ANDAR, e CARREGAR na fatia B (S §7); o resto fica parado
        anda = ((tar == ANDAR) & (fat == FATIA_ANDAR)) | ((tar == CARREGAR) & (fat == FATIA_B))
        for tarefa_id, nome in ((ANDAR, "ANDAR"), (CARREGAR, "CARREGAR")):
            ids = env_ids[anda & (tar == tarefa_id)]
            if ids.numel() == 0:
                continue
            # episódio de terreno: `terrain_types` ≠ 0 (coluna 0 = plano), escrito pelo `sorteia_fatia` antes deste
            # reset (§6.4). Nele a célula é a `terreno` da tarefa; fora dele, sorteia sem a `terreno`.
            # Com a célula `terreno` fechada, o `sorteia_fatia` não escolhe terreno (`eventos._fracao_terreno` = 0):
            # ela não recebe velocidade fechada.
            ter = env.scene["terrain"].terrain_types[ids].to(ids.device) != 0
            cel = torch.full((len(ids),), env.v3_cur.id_celula(nome, "terreno"), dtype=torch.long, device=ids.device)
            fora = ~ter
            if fora.any():
                cel[fora] = env.v3_cur.sorteia_celulas(nome, int(fora.sum()), g, sem_tipo="terreno").to(ids.device)
            env.v3_celula[ids] = cel
            nivel = env.v3_cur.nivel(nome)
            for c in cel.unique().tolist():
                m = cel == c
                tipo = env.v3_cur.celulas[c].tipo
                vel = sorteia_velocidade(tipo, nivel, nome, int(m.sum()), g, self.k)
                self.vel_command_b[ids[m]] = vel.to(self.device)
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        # CARREGAR de B: comando sorteado na abertura e fixo até o fim do trecho (S §7)
        fixo = env_ids[(tar == CARREGAR) & (fat == FATIA_B)]
        self.time_left[fixo] = float("inf")

    def _update_command(self) -> None:
        pass  # sem heading, sem mundo e sem standing do fabricante: nada a atualizar (ver docstring)
