"""Termos de comando da v3: `cadeia` (tarefa, trechos, odometria, alvos do PEGAR) e `twist` (velocidade por célula, sucesso
por segmento). Spec: S §7 (cadeia), §8.2 (alvos), §10 (caixa perdida), §11.2 (tetos), §11.3 (sucesso), §19 conflito 5;
plano D3, E3, §6.3, §6.4, §6.5. PD-15 (âncora do canal zero), PD-27 (caixa perdida vira ANDAR), PD-36, PD-44 (quem dá o
passo da odometria), PD-50 (sucesso sem perda).

Os núcleos (`ok_agora_pegar`, `caixa_perdida_agora`, `sucesso_do_pegar`, `sorteia_velocidade`, `monta_tabela_trechos`,
`avanca_trecho`, `tempo_da_ancora`, `sucesso_do_segmento`) são puros: sobre tensores, sem o env. Os invólucros
(`CadeiaCmd`, `TwistV3`) leem o env.
Contagem: g1_limpo/comando.py tinha 2 526 linhas, e a v3 copia só o que a D3 pede (a eficiência por segmento
e a razão de marcha ficam só como diagnóstico, que o §6.7 loga).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

from g1_v3 import alvos, estimador, eventos, recompensas
from g1_v3.alvos import B1, B2, B3, B4, P0, P1, P2, P3, P4
from g1_v3.curriculo import garante_v3
from g1_v3.knobs import ATIVO, TAREFAS, Alvos, Cadeia, Comando, Estimador
from g1_v3.observacoes import z_pe_min

# Índices de TAREFAS (knobs.py): o one-hot da §6.5 e a tabela de trechos usam esta ordem.
ANDAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3
# Índices de FATIAS (knobs.py): ANDAR, B, C.
FATIA_ANDAR, FATIA_B, FATIA_C = 0, 1, 2
# Folga de float no relógio: 25 passos de 0,02 s somam 0,4999… e não podem perder o aviso.
# Não é número de treino (mesmo papel do `_EPS` de alvos.py).
_EPS = 1e-6
A, D, T, CENA = ATIVO.alvos, ATIVO.detector, ATIVO.terminacao, ATIVO.cena
# Tempo de permanência por alvo, na ordem de ALVOS (S §8.2): só P4, B2, B3 e B4 têm tempo.
_TEMPOS = (0.0, 0.0, 0.0, 0.0, A.tempo_p4, 0.0, A.tempo_b2, A.tempo_b3, A.tempo_b4)


# --- núcleos do PEGAR e da caixa (§6.3, E3; S §8.2, §10, §11.3; PD-50) ---------------------------------------

def ok_agora_pegar(indice, d_palmas, contatos, apoio, incl_graus, incl_lim_graus, d_caixa_p4, d_pelve_p4,
                   invariantes, k) -> torch.Tensor:
    """(E,) bool: a condição de cada alvo de PEGAR, `& (invariantes == 1)` (S §8.2: o invariante impede o avanço).
    ⚠ B1..B4 e −1 dão False. Os B vêm de `ok_agora_botar` (F1), na cadeia, e não por esta condição."""
    p01 = (d_palmas <= k.tol_palma).all(-1)
    p2 = contatos.all(-1)
    p3 = ~apoio & (incl_graus <= incl_lim_graus)
    p4 = (d_caixa_p4 <= k.tol_caixa_p4) & (d_pelve_p4 <= k.tol_pelve_p4)
    ok = (((indice == P0) | (indice == P1)) & p01) | ((indice == P2) & p2) | ((indice == P3) & p3) \
        | ((indice == P4) & p4)
    return ok & (invariantes == 1.0)


def caixa_perdida_agora(fundo_sobre_chao, v_rel, fora_do_alvo, pegou, indice, sobre_laje, k) -> torch.Tensor:
    """(E,) bool = caiu | arremesso | fora_da_laje (S §10). Caiu: fundo abaixo da folga (estrito). Arremesso: pegou,
    v_rel > v_solta e fora do alvo de soltura. Fora da laje: P0 a P2 com o centro da caixa fora do topo da laje."""
    caiu = fundo_sobre_chao < k.caixa_folga_chao
    arremesso = pegou & (v_rel > k.v_solta) & fora_do_alvo
    fora_da_laje = (indice >= P0) & (indice <= P2) & ~sobre_laje
    return caiu | arremesso | fora_da_laje


def sucesso_do_pegar(p4_fechado, falhou, perdeu) -> torch.Tensor:
    """(E,) bool = P4 fechado, sem falha e sem caixa perdida (S §11.3, PD-50). O invólucro decide quando registra."""
    return p4_fechado & ~falhou & ~perdeu


def ok_agora_botar(indice, d_horiz_b1, d_caixa_b2, apoio_N, peso_N, afasta_palmas, apoiada, v_caixa,
                   dpelve, dtronco_graus, invariantes, k: Alvos) -> torch.Tensor:
    """(E,) bool: a condição de cada alvo de BOTAR, `& (invariantes == 1)` (S §8.2). Formas: v_caixa (E,3) m/s;
    afasta_palmas (E,2) m, uma por palma; apoio_N e peso_N (E,) N. B1..B4 são os índices 5..8; P0..P4 e −1 dão False.
    ⚠ O B4 da caixa perdida (E3) usa este núcleo com `invariantes` = 1: sem o portão desloc (E11)."""
    b1 = d_horiz_b1 <= k.tol_b1_horizontal
    b2 = (d_caixa_b2 <= k.raio_b2) & (apoio_N >= k.apoio_b2_mg * peso_N)
    b3 = (afasta_palmas >= k.b3_afasta).all(-1) & apoiada & (v_caixa.norm(dim=-1) < k.v_parada_b3)
    b4 = (dpelve.abs() <= k.tol_pelve_b4) & (dtronco_graus.abs() <= k.tol_tronco_b4_graus)
    ok = ((indice == B1) & b1) | ((indice == B2) & b2) | ((indice == B3) & b3) | ((indice == B4) & b4)
    return ok & (invariantes == 1.0)


def sucesso_do_botar(b4_fechado, falhou, perdeu) -> torch.Tensor:
    """(E,) bool = B4 fechado, sem falha e sem caixa perdida (S §11.3, PD-50)."""
    return b4_fechado & ~falhou & ~perdeu


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
    """Tarefa publicada, trechos da fatia (S §7) e alvos do PEGAR (§6.2 `Alvos`, E3). Escreve v3_trecho, v3_tarefa,
    v3_fim, v3_aviso, v3_freio, v3_degrau, v3_alvo_idx, v3_alvo_w, v3_caixa_pega_w, v3_sigma, v3_relogio_alvo,
    v3_avancou, v3_pegou, v3_p4_fechado, v3_b4_fechado, v3_perdeu, v3_perdeu_agora, v3_ponto_apoio (BOTAR, F1); dá o
    `passo` da `v3_odo` uma vez por passo (PD-44)."""

    cfg: CadeiaCmdCfg

    def __init__(self, cfg: CadeiaCmdCfg, env):
        super().__init__(cfg, env)
        garante_v3(env)
        self.robot = env.scene[cfg.entity_name]
        self.tab, self.faixas = _monta(cfg.k, self.device)
        self.relogio = torch.zeros(self.num_envs, device=self.device)
        self.alvo_s = torch.zeros(self.num_envs, device=self.device)  # duração (ou atraso) sorteada do trecho
        self.rel_fim = torch.zeros(self.num_envs, device=self.device)  # relógio do detector de fim (I §4.2)
        self.pega = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # PEGAR aberto, sem registro
        self.bota = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # BOTAR aberto, até o fim da cadeia
        self.sit_palmas = self.robot.find_sites(("left_palm", "right_palm"))[0]  # ordem = `palmas_alvo_caixa`
        self.tempos = torch.tensor(_TEMPOS, dtype=torch.float32, device=self.device)
        k = cfg.est
        # magnitude "início" (PD-18): índice 0 de cada par; o refino entra com o degrau (fase G)
        self.deriva = {"odo": k.deriva_odo[0], "rumo": k.deriva_rumo[0],
                       "pos_giro": k.deriva_pos_giro[0], "rumo_giro": k.deriva_rumo_giro[0]}

    @property
    def command(self) -> torch.Tensor:  # one-hot (E,6) de v3_tarefa (§6.5)
        return torch.nn.functional.one_hot(self._env.v3_tarefa, len(TAREFAS)).float()

    def _twist(self):
        return self._env.command_manager.get_term("twist")

    # --- medidas e alvos -----------------------------------------------------------------------------

    def _mede(self) -> dict:
        """Medidas do passo (E,...): base, caixa (verdade), laje, palmas, contatos e o tronco (§6.2 `Alvos`)."""
        env, d = self._env, self.robot.data
        caixa, laje = env.scene["box"].data, env.scene["table"].data
        h = env.scene["altura_pelve"].data.heights[:, 0]
        rumo, caixa_p, caixa_q = d.heading_w, caixa.root_link_pos_w, caixa.root_link_quat_w
        palmas_w = d.site_pos_w[:, self.sit_palmas]  # (E,2,3)
        R = alvos.frame_da_caixa(caixa_q, rumo)
        m = dict(
            h=h, base_p=d.root_link_pos_w, base_q=d.root_link_quat_w, caixa_p=caixa_p, palmas_w=palmas_w,
            palmas_c=torch.einsum("eji,ekj->eki", R, palmas_w - caixa_p[:, None]),
            incl_graus=torch.rad2deg(alvos.inclinacao_da_caixa(caixa_q)),
            contatos=torch.cat([env.scene["palma_E"].data.found, env.scene["palma_D"].data.found], -1) > 0,
            apoio=env.scene["apoio_caixa"].data.found.amax(-1) > 0,
            sobre=(quat_apply_inverse(laje.root_link_quat_w, caixa_p - laje.root_link_pos_w)[:, :2].abs()
                   <= CENA.prateleira_meia_xy).all(-1),
            h_caixa=env.scene["altura_caixa"].data.heights[:, 0],
            v_rel=(caixa.root_link_lin_vel_w - d.root_link_lin_vel_w).norm(dim=-1),
            tronco_graus=recompensas.incl_torso(env),  # a mesma medida da `cerca_tronco`
            v_caixa=caixa.root_link_lin_vel_w,  # absoluta: a caixa parada no B3 (PD-39)
            apoio_N=forca_de_apoio(env, "apoio_caixa"),
            peso_N=env.v3_massa * float(-env.sim.mj_model.opt.gravity[2]),  # m·g, com a gravidade do modelo
        )
        return m

    def _dist(self, m, indice) -> torch.Tensor:
        # d do alvo `indice` no estado do passo (E4, `recompensas.distancia_do_alvo`); usado no σ. O alvo da caixa é o
        # da recompensa, com o chão local (S10): uma conta só.
        env = self._env
        dpelve = m["h"] - env.v3_ref.pelve_de_pe
        alvo = recompensas.alvo_da_caixa_w(env, indice)
        return recompensas.distancia_do_alvo(indice, m["palmas_c"], m["caixa_p"], alvo, dpelve, m["tronco_graus"],
                                             env.v3_meia, A)

    def _sigma(self, m, indice) -> torch.Tensor:
        # Piso só nos alvos métricos (M12): o B4 é adimensional, e o piso de 1 cm não vale para ele.
        d = self._dist(m, indice)
        return torch.where(indice == B4, d, d.clamp(min=A.sigma_min))

    def _espera(self) -> torch.Tensor:
        # (E,) bool: PEGAR aberto sem alvo, à espera da primeira leitura válida da caixa (E11 S5b)
        env = self._env
        return self.pega & (env.v3_alvo_idx < 0) & (env.v3_tarefa == PEGAR)

    def _abre_p0(self, m) -> None:
        # Abre P0 do PEGAR na primeira leitura VÁLIDA da caixa (E11 S5b, decisão do dono): idade < idade_max, que só
        # acontece com a caixa já vista (sem leitura, a idade é idade_max). Até lá o trecho fica sem alvo (−1) e o robô
        # espera em pé: `_detector` não deixa o relógio do PEGAR correr. A pega e o alvo saem da caixa ESTIMADA (com os
        # erros do estimador, S §6.4); o frame volta ao mundo pela odometria (`do_frame_da_base`, S5a). σ = distância
        # inicial das palmas a P0.
        env = self._env
        if not self._espera().any():
            return
        odo = env.v3_odo
        est = env.v3_estimador.saida_b(odo, m["base_q"])  # (E,8): pos_b, rotvec, meia, idade
        ids = (self._espera() & env.v3_estimador.tem_leitura()).nonzero().flatten()
        if ids.numel() == 0:
            return
        pos = estimador.do_frame_da_base(est[:, :3], odo)[ids]
        env.v3_alvo_idx[ids] = P0
        env.v3_phi[ids] = -1.0  # salto fora do `avanca` (S8): a entrada no P0 paga 0
        env.v3_relogio_alvo[ids] = 0.0
        env.v3_caixa_pega_w[ids] = pos
        env.v3_alvo_w[ids] = pos
        env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, P0))[ids]

    def _perda(self, m) -> None:
        # Caixa perdida (E3, item 4; S §10). `v3_perdeu_agora` é um pulso no passo da perda. Com a chave, a caixa perdida
        # vira ANDAR parado até o fim (PD-27): o alvo passa a B4, com σ = a distância à pose de pé no instante da perda.
        env = self._env
        tar = env.v3_tarefa
        longe = (m["caixa_p"] - env.v3_alvo_w).norm(dim=-1) > A.raio_b2
        fora = torch.where(tar == BOTAR, longe, torch.ones_like(longe))  # fora do alvo de soltura, só no BOTAR
        perda = caixa_perdida_agora(m["h_caixa"] - env.v3_meia, m["v_rel"], fora, env.v3_pegou, env.v3_alvo_idx,
                                    m["sobre"], T) & (tar != ANDAR)
        env.v3_perdeu_agora[:] = perda
        if self.cfg.k.caixa_perdida_vira_andar and perda.any():
            ids = perda.nonzero().flatten()
            env.v3_perdeu[ids] = True
            env.v3_alvo_idx[ids] = B4
            env.v3_phi[ids] = -1.0  # salto fora do `avanca` (S8): sem isto, o passo da perda cobraria −w_p·Φ(alvo velho)
            env.v3_relogio_alvo[ids] = 0.0
            env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, B4))[ids]
            self._publica(ids, torch.full_like(ids, ANDAR))

    def _alvos_pegar(self, m, dt) -> None:
        # Alvos do PEGAR (E3, item 2; S §8.2) e do BOTAR (F1), num passo só: o `ok_agora` do alvo corrente, `alvos.avanca`
        # e os buffers do avanço. ⚠ Um só `avanca`: dois sobre o mesmo relógio se zerariam. O índice fica em P4 depois do
        # P4 (a troca de tarefa leva a B1). O B4 da caixa perdida fecha pela pose de pé, sem o portão desloc (E11).
        env = self._env
        idx, a, ref, tar = env.v3_alvo_idx, env.v3_meia, env.v3_ref, env.v3_tarefa
        zero = torch.zeros_like(m["h"])
        botar = (tar == BOTAR) | env.v3_perdeu  # B1..B4, e o B4 da caixa perdida (E3)
        desl = (m["caixa_p"][:, :2] - env.v3_caixa_pega_w[:, :2]).norm(dim=-1)
        if botar.any():
            # Primeiro apoio do BOTAR grava o ponto de apoio; zero = ainda sem apoio. O portão desloc de B3–B4 lê dele.
            primeiro = (tar == BOTAR) & m["apoio"] & (env.v3_ponto_apoio.abs().sum(-1) == 0)
            env.v3_ponto_apoio[primeiro] = m["caixa_p"][primeiro]
            desl_apoio = (m["caixa_p"][:, :2] - env.v3_ponto_apoio[:, :2]).norm(dim=-1)
            desl = torch.where(env.v3_perdeu, zero, torch.where(tar == BOTAR, desl_apoio, desl))
        # Invariante = desloc · incl · maos (§6.3). Tronco, recuo e CoM não entram nele: ficam em zero (rampa 1).
        g = recompensas.portoes_do_alvo(
            idx, desl, zero, m["incl_graus"], recompensas.incl_lim(env), m["contatos"], zero, zero, zero,
            torch.ones_like(idx, dtype=torch.bool), a, A)
        inv = g["desloc"] * g["incl"] * g["maos"]
        # PEGAR
        d_palmas = (m["palmas_c"] - alvos.palmas_alvo_caixa(idx, a, A)).norm(dim=-1)
        d_caixa_p4 = (m["caixa_p"] - recompensas.alvo_da_caixa_w(env, torch.full_like(idx, P4))).norm(dim=-1)
        d_pelve_p4 = (m["h"] - ref.pelve_com_caixa(a)).abs()
        pega = (tar == PEGAR) & ~env.v3_p4_fechado
        ok = ok_agora_pegar(idx, d_palmas, m["contatos"], m["apoio"], m["incl_graus"], recompensas.incl_lim(env),
                            d_caixa_p4, d_pelve_p4, inv, A) & pega
        if botar.any():  # BOTAR (B1..B4); o B4 da caixa perdida cai aqui com `inv` = 1 (o `desl` zerado)
            dif = m["caixa_p"] - recompensas.alvo_da_caixa_w(env, idx)
            afasta = (m["palmas_c"].abs() - a[:, None, None]).amax(-1)
            ok = ok | ok_agora_botar(idx, dif[:, :2].norm(dim=-1), dif.norm(dim=-1), m["apoio_N"], m["peso_N"], afasta,
                                     m["apoio"], m["v_caixa"], m["h"] - ref.pelve_de_pe, m["tronco_graus"], inv, A)
        novo, rel, av = alvos.avanca(idx, ok, env.v3_relogio_alvo, dt, self.tempos)
        fecha_p4 = av & (idx == P4)
        novo = torch.where(fecha_p4, idx, novo)
        env.v3_p4_fechado |= fecha_p4
        env.v3_pegou |= av & (idx == P2)  # a caixa passou ao P3: pega feita
        env.v3_b4_fechado |= av & (idx == B4)
        env.v3_avancou[:] = av
        env.v3_relogio_alvo[:] = rel
        env.v3_alvo_idx[:] = novo
        # σ do alvo velho, gravado antes da troca: o progresso paga Φ(s′, velho) com ele (E11 S3). No fecho do P4 não
        # troca, e o velho é o σ corrente.
        env.v3_sigma_velho[:] = env.v3_sigma
        troca = av & ~fecha_p4 & (novo >= 0)
        env.v3_sigma[:] = torch.where(troca, self._sigma(m, novo), env.v3_sigma)

    def _detector(self, m, dt) -> torch.Tensor:
        # Detector de fim (E3, item 3; I §4.2): aviso da pose de pé com caixa, só nos trechos que acabam por ele.
        env, ref, a = self._env, self._env.v3_ref, self._env.v3_meia
        palmas_b = quat_apply_inverse(m["base_q"][:, None].expand(-1, 2, -1), m["palmas_w"] - m["base_p"][:, None])
        # R6 (E11): a pelve sobre o pé mais baixo, como a I §4.2 e o treino (`z_pe_min`), e não sobre o terreno (`h`)
        pelve = self.robot.data.root_link_pos_w[:, 2] - z_pe_min(self.robot, ref)
        # BOTAR: a pose de pé sem caixa (F1); PEGAR: a pose com caixa (I §4.2)
        bot = env.v3_tarefa == BOTAR
        pelve_ref = torch.where(bot, torch.full_like(a, ref.pelve_de_pe), ref.pelve_com_caixa(a))
        palmas_ref = torch.where(bot[:, None, None], ref.palmas_de_pe_b, ref.palmas_com_caixa_b(a))
        aviso_det, rel = alvos.detecta_fim(pelve, torch.deg2rad(m["tronco_graus"]), palmas_b,
                                           self.robot.data.joint_vel, pelve_ref, palmas_ref, self.rel_fim, dt, D)
        # S5b: enquanto o PEGAR espera a caixa (sem alvo), nem o aviso nem o relógio do detector correm. Assim a pose
        # de pé, que pode já bater com a pega, não acaba o trecho antes do P0: o relógio do atraso (`avanca_trecho`) só
        # anda com o aviso, e fica parado na espera. O PEGAR acaba pelo aviso só depois que o alvo abriu.
        por = self.tab.por_detector[env.v3_trecho] & ~self._espera()
        self.rel_fim = torch.where(por, rel, torch.zeros_like(rel))
        aviso = aviso_det & por
        env.v3_aviso[:] = aviso
        return aviso

    # --- trechos -------------------------------------------------------------------------------------

    def _publica(self, ids, tar) -> None:
        # Publica a tarefa `tar` (um valor por env de `ids`) com o freio e o degrau do currículo, e reamostra o comando.
        env = self._env
        env.v3_tarefa[ids] = tar
        freio, degrau = torch.zeros_like(tar), torch.zeros_like(tar)
        for i, nome in enumerate(TAREFAS[:4]):  # REORIENTAR e LEVANTAR ficam em 0 (reservados)
            m = tar == i
            freio[m] = env.v3_cur.niveis_abertos(nome)
            degrau[m] = env.v3_cur.degrau(nome)
        env.v3_freio[ids] = freio
        env.v3_degrau[ids] = degrau
        self._twist()._resample(ids)

    def _registra(self, ids, falhou, botar=False) -> None:
        # Registra o PEGAR aberto, ou o BOTAR se `botar` (S §11.3, PD-50), e limpa o pendente. O PEGAR registra na abertura
        # do BOTAR (falhou = False) e no reset (falhou = terminação que não é time_out: a queda, por exemplo). O BOTAR
        # registra no fim da cadeia (falhou = False) e no reset (falhou = True: a cadeia não chegou ao fim).
        env = self._env
        pend = self.bota if botar else self.pega
        sel = pend[ids]
        ids, falhou = ids[sel], falhou[sel]
        if ids.numel() == 0:
            return
        if botar:
            suc = sucesso_do_botar(env.v3_b4_fechado[ids], falhou, env.v3_perdeu[ids])
            celula = env.v3_cur.id_celula("BOTAR", "botar")
        else:
            suc = sucesso_do_pegar(env.v3_p4_fechado[ids], falhou, env.v3_perdeu[ids])
            celula = env.v3_cur.id_celula("PEGAR", "pegar")
        env.v3_cur.registra(celula, int(ids.numel()), int(suc.sum()))
        pend[ids] = False

    def _abre_botar(self, ids, m) -> None:
        # Abertura do BOTAR (F1; PD-47): a laje de destino dá o ponto de soltura no topo, e o alvo sobe `a` (a caixa
        # fica com o centro a meia aresta acima). Alvo corrente B1, σ pelo `_sigma` como o P0 do PEGAR, e o salto
        # fora do `avanca` paga 0 (`v3_phi` = −1). Zero em `v3_ponto_apoio` = ainda sem apoio.
        env = self._env
        soltura = eventos.posiciona_destino(env, ids).to(env.v3_alvo_w.dtype)
        soltura[:, 2] += env.v3_meia[ids]
        env.v3_alvo_w[ids] = soltura
        env.v3_alvo_idx[ids] = B1
        env.v3_phi[ids] = -1.0
        env.v3_relogio_alvo[ids] = 0.0
        env.v3_ponto_apoio[ids] = 0.0
        env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, B1))[ids]
        self.bota[ids] = True

    def _abre_trecho(self, ids, trecho, m) -> None:
        # Sorteia a duração (ou atraso) do trecho, publica a tarefa e o freio/degrau, e reamostra o comando do twist
        # (o CARREGAR de B fica fixo na abertura, §7; o parado sai com comando zero). O PEGAR abre sem alvo (E11 S5b;
        # o P0 abre em `_abre_p0`), registra o PEGAR anterior quando abre o BOTAR (S §11.3) e tira a laje do caminho do
        # CARREGAR que anda (C39).
        env = self._env
        lo, hi = self.faixas[trecho, 0], self.faixas[trecho, 1]
        self.alvo_s[ids] = lo + (hi - lo) * _u(len(ids), env.v3_gerador).to(self.device)
        self.relogio[ids] = 0.0
        env.v3_trecho[ids] = trecho
        tar = self.tab.tarefa[trecho]
        self._publica(ids, tar)
        sem_alvo = (tar == ANDAR) | (tar == PEGAR)
        env.v3_alvo_idx[ids[sem_alvo]] = -1
        env.v3_phi[ids[sem_alvo]] = -1.0  # salto fora do `avanca` (S8)
        botar = tar == BOTAR
        self._registra(ids[botar], torch.zeros(int(botar.sum()), dtype=torch.bool, device=self.device))
        if botar.any():
            self._abre_botar(ids[botar], m)
        pega = tar == PEGAR
        self.pega[ids[pega]] = True
        anda = (tar == CARREGAR) & (env.v3_fatia[ids] == FATIA_B) & (self._twist().vel_command_b[ids].abs().sum(-1) > 0)
        if anda.any():
            eventos.laje_fora(env, ids[anda])

    def compute(self, dt: float) -> None:
        env = self._env
        m = self._mede()
        self._perda(m)
        self._abre_p0(m)
        self._alvos_pegar(m, dt)
        aviso = self._detector(m, dt)
        novo, self.relogio, avancou, fim = avanca_trecho(
            env.v3_trecho, self.relogio, self.alvo_s, aviso, self.alvo_s, dt, self.tab)
        # caixa perdida que vira ANDAR congela o trecho até o fim do episódio (PD-27, S §10)
        cong = env.v3_perdeu
        novo = torch.where(cong, env.v3_trecho, novo)
        avancou, fim = avancou & ~cong, fim & ~cong
        env.v3_fim |= fim
        fim_ids = fim.nonzero().flatten()  # fim da cadeia: registra o BOTAR pendente, se houver (PD-50)
        self._registra(fim_ids, torch.zeros_like(fim_ids, dtype=torch.bool), botar=True)
        ids = (avancou & ~fim).nonzero().flatten()  # no fim da cadeia o trecho não reabre (S §7)
        if ids.numel():
            self._abre_trecho(ids, novo[ids], m)
        env.v3_odo.passo(self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w, self.deriva)

    def _resample(self, env_ids) -> None:  # chamado no reset do episódio (CommandTerm.reset)
        env = self._env
        # o PEGAR aberto se registra no reset: queda ou outra terminação conta como falha (S §11.3); o BOTAR sem fim
        # de cadeia também (a cadeia não chegou ao ANDAR final)
        self._registra(env_ids, ~env.termination_manager.time_outs[env_ids])
        self._registra(env_ids, torch.ones_like(env_ids, dtype=torch.bool), botar=True)
        for nome in ("v3_pegou", "v3_perdeu", "v3_p4_fechado", "v3_b4_fechado", "v3_perdeu_agora", "v3_avancou",
                     "v3_aviso", "v3_fim"):
            getattr(env, nome)[env_ids] = False
        env.v3_relogio_alvo[env_ids] = 0.0
        env.v3_sigma[env_ids] = 1.0  # neutro: o alvo −1 não o usa, e σ = 0 daria NaN
        env.v3_phi[env_ids] = -1.0  # sem alvo no início do episódio: a entrada no alvo paga 0 (E4)
        env.v3_b_pagos[env_ids] = 0.0
        self.rel_fim[env_ids] = 0.0
        fat = env.v3_fatia[env_ids]
        self._abre_trecho(env_ids, self.tab.inicio.to(fat.device)[fat], self._mede())

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
        # 1. fronteira de segmento: o contador mudou desde o último passo (o cadeia reamostrou). A perda da caixa no
        # segmento com caixa (PEGAR, CARREGAR, BOTAR) fecha como falha, mesmo antes da âncora (E11 S11)
        mudou = (self.command_counter != self.seg_visto).nonzero().flatten()
        if mudou.numel():
            self._fecha(mudou, falha=env.v3_perdeu[mudou] & self.seg_carrega[mudou])
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
