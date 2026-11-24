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

import math
from dataclasses import dataclass, replace

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

from g1_v3 import alvos, balanceador, estimador, eventos, recompensas
from g1_v3.alvos import B1, B2, B3, P0, P1, P2, P3
from g1_v3.curriculo import garante_v3
from g1_v3.knobs import ALVOS, ATIVO, TAREFAS, Alvos, Cadeia, Comando, Estimador
from g1_v3.observacoes import z_pe_min

# Índices de TAREFAS (knobs.py): o one-hot da §6.5 e a tabela de trechos usam esta ordem.
ANDAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3
# Índices de FATIAS (knobs.py): ANDAR, B, C.
FATIA_ANDAR, FATIA_B, FATIA_C = 0, 1, 2
# Folga de float no relógio: 25 passos de 0,02 s somam 0,4999… e não podem perder o aviso.
# Não é número de treino (mesmo papel do `_EPS` de alvos.py).
_EPS = 1e-6
A, T, CENA = ATIVO.alvos, ATIVO.terminacao, ATIVO.cena


# --- núcleos do PEGAR e da caixa (§6.3, E3; S §8.2, §10, §11.3; PD-50) ---------------------------------------

def ok_agora_pegar(indice, d_palmas, contatos, apoio, subida, d_transporte, d_pelve, invariantes, k) -> torch.Tensor:
    """(E,) bool: a condição de cada alvo de PEGAR, `& (invariantes == 1)` (S §8.2: o invariante impede o avanço).
    ⚠ B1..B3 e −1 dão False. Os B vêm de `ok_agora_botar` (F1), na cadeia, e não por esta condição.
    18/11, T8: `subida` (E,) m = fundo da caixa sobre o topo da laje da pega. 23/11, T14: o P3 fecha na pose final do
    transporte (molde do g1_limpo): a caixa subiu (`subida_min_p3`), está no ponto (`d_transporte`), a pelve no lugar
    (`d_pelve`) e fora da laje (`~apoio`); o relógio do `avanca` exige tudo sem interrupção por `tempo_p3`. A inclinação
    e as duas palmas vêm do invariante.
    20/11, T13: `k` é o `Alvos` efetivo da zona (`env.v3_alvos`); o `tol_palma` dele é um escalar ou um valor por etapa (7,),
    porque P0 e P1 têm z próprios."""
    tol = torch.as_tensor(k.tol_palma, device=indice.device)
    tol = tol[indice.clamp(min=0)] if tol.ndim else tol
    p01 = (d_palmas <= tol[..., None]).all(-1)
    p2 = contatos.all(-1)
    p3 = (subida >= k.subida_min_p3) & (d_transporte <= k.tol_transporte_p3) & (d_pelve <= k.tol_pelve_p3) & ~apoio
    ok = (((indice == P0) | (indice == P1)) & p01) | ((indice == P2) & p2) | ((indice == P3) & p3)
    return ok & (invariantes == 1.0)


def caixa_perdida_agora(fundo_sobre_chao, v_rel, fora_do_alvo, pegou, indice, sobre_laje, k) -> torch.Tensor:
    """(E,) bool = caiu | arremesso | fora_da_laje (S §10). Caiu: fundo abaixo da folga (estrito). Arremesso: pegou,
    v_rel > v_solta e fora do alvo de soltura. Fora da laje: P0 a P2 com o centro da caixa fora do topo da laje."""
    caiu = fundo_sobre_chao < k.caixa_folga_chao
    arremesso = pegou & (v_rel > k.v_solta) & fora_do_alvo
    fora_da_laje = (indice >= P0) & (indice <= P2) & ~sobre_laje
    return caiu | arremesso | fora_da_laje


def sucesso_do_pegar(pega_fechada, falhou, perdeu) -> torch.Tensor:
    """(E,) bool = P3 fechado, sem falha e sem caixa perdida (S §11.3, PD-50). O invólucro decide quando registra."""
    return pega_fechada & ~falhou & ~perdeu


def ok_agora_botar(indice, d_caixa_b1, apoio_N, peso_N, afasta_palmas, apoiada, v_caixa,
                   dpelve, dtronco_graus, dpalmas, invariantes, k: Alvos) -> torch.Tensor:
    """(E,) bool: a condição de cada alvo de BOTAR, `& (invariantes == 1)` (S §8.2). Formas: v_caixa (E,3) m/s;
    afasta_palmas (E,2) m, uma por palma; apoio_N e peso_N (E,) N; dpalmas (E,) m, a maior das duas distâncias palma–pose
    de pé no frame da base (13/11, A9). B1..B3 são os índices 4..6; P0..P3 e −1 dão False.
    23/11, T14: o B1 leva ao destino e apoia num fecho só (a caixa no raio, com `apoio_b1_mg` do peso na laje e parada); a
    zona nunca afrouxa o `apoio_b1_mg`.
    ⚠ O B3 da caixa perdida (E3) usa este núcleo com `invariantes` = 1: sem o portão desloc (E11)."""
    # 13/11, A4: a caixa parada no fecho, para o ponto de apoio (`v3_ponto_apoio`) sair sem deslizamento
    parada = v_caixa.norm(dim=-1) < k.v_parada
    b1 = (d_caixa_b1 <= k.raio_b1) & (apoio_N >= k.apoio_b1_mg * peso_N) & parada
    b2 = (afasta_palmas >= k.b2_afasta).all(-1) & apoiada & parada
    # 13/11, A9: as mãos ao lado do tronco (§4.5) entram no fecho do B3
    b3 = (dpelve.abs() <= k.tol_pelve_b3) & (dtronco_graus.abs() <= k.tol_tronco_b3_graus) & (dpalmas <= k.tol_palmas_b3)
    ok = ((indice == B1) & b1) | ((indice == B2) & b2) | ((indice == B3) & b3)
    return ok & (invariantes == 1.0)


def sucesso_do_botar(bota_fechada, falhou, perdeu) -> torch.Tensor:
    """(E,) bool = B3 fechado, sem falha e sem caixa perdida (S §11.3, PD-50)."""
    return bota_fechada & ~falhou & ~perdeu


# --- sorteio de velocidade (§6.3, D3 item 1) ----------------------------------------------------------

def _u(n, g):  # U[0,1) no device do gerador (None: default)
    return torch.rand(n, generator=g, device=None if g is None else g.device)


def _anda(n, lo, lo_lado, frente, tras, lado, g):
    # 4 direções uniformes; módulo U[piso; teto], teto = `frente` para +x, `tras` para −x e `lado` para ±y (S §11.2; 24/11, T15)
    eixo = torch.randint(0, 4, (n,), generator=g, device=None if g is None else g.device)
    teto = torch.where(eixo == 0, frente, torch.where(eixo == 1, tras, lado))
    piso = torch.where(eixo < 2, lo, lo_lado)
    mod = piso + (teto - piso) * _u(n, g)
    vx = mod * ((eixo == 0).float() - (eixo == 1).float())
    vy = mod * ((eixo == 2).float() - (eixo == 3).float())
    return torch.stack([vx, vy], dim=1)


def _giro(n, lo, hi, g):  # ±U[lo; hi] (sinal sorteado por env)
    sinal = 2.0 * (_u(n, g) >= 0.5).float() - 1.0
    return sinal * (lo + (hi - lo) * _u(n, g))


def sorteia_velocidade(tipo, nivel, tarefa, n, gerador, k: Comando) -> torch.Tensor:
    """(n,3) de (vx, vy, wz). ANDAR (tipos `andar` e `terreno`, 16/11): (vx, vy) uniforme na elipse dos tetos do nível
    (semieixo x `frente` ou `tras`, semieixo y `lado`; 17/11, T7), wz independente em U(−giro; giro), com `andar_parado`
    dos sorteios zerados. Fora do ANDAR
    `tarefa` não muda a faixa: o `tipo` já diz o que anda (CARREGAR: `reto`, `giro_lugar`, `arco`, `terreno`)."""
    v = torch.zeros(n, 3, device=None if gerador is None else gerador.device)
    if tarefa == "ANDAR":
        # 24/11, T15: cada canal sobe do seu início até o seu teto final, `andar_passo` por nível
        lado = min(k.andar_lado_inicio + k.andar_passo * nivel, k.lado_teto_max)
        frente = min(k.andar_inicio + k.andar_passo * nivel, k.frente_tetos[-1])
        tras = min(k.andar_inicio + k.andar_passo * nivel, k.tras_teto_max)
        giro = min(k.andar_giro_inicio + k.andar_passo * nivel, k.giro_tetos[-1])
        if tipo == "terreno":  # PD-12: o terreno usa o teto de `terreno_v_max` para frente e para trás
            frente, tras = min(frente, k.terreno_v_max), min(tras, k.terreno_v_max)
        # 17/11, T7: elipse, e não o quadrado de vx e vy independentes: no quadrado a diagonal passava do teto em até 41 %,
        # e no nível 4 a nota linear caía a 0,58 com |v| > 0,9 (`segmentos.csv` do checkpoint da it 5605). Ponto uniforme
        # na meia unidade de disco com x ≥ 0 (raio √U); a meia elipse da frente sai com chance pela área, frente/(frente+tras)
        r, ang = _u(n, gerador).sqrt(), torch.pi * (_u(n, gerador) - 0.5)
        semi_x = -tras + (frente + tras) * (_u(n, gerador) * (frente + tras) < frente).float()
        v[:, 0] = semi_x * r * torch.cos(ang)
        v[:, 1] = lado * r * torch.sin(ang)
        v[:, 2] = giro * (2.0 * _u(n, gerador) - 1.0)
        v[:, 2][v[:, 2].abs() < k.giro_min] = 0.0  # 24/11, T15: giro pequeno é ruído para a razão ω̄/c_z: zero ou ≥ `giro_min`
        v[_u(n, gerador) < k.andar_parado] = 0.0
        return v
    frente = k.frente_tetos[nivel]
    tras, lado = min(frente, k.tras_teto_max), min(frente, k.lado_teto_max)
    giro = k.giro_tetos[min(nivel, 3)]
    if tipo == "terreno":  # PD-12: o terreno usa o reto do nível, com teto de `terreno_v_max`
        frente, tras = min(frente, k.terreno_v_max), min(tras, k.terreno_v_max)
    # reto/terreno: só linear; arco: linear + giro; giro_lugar: só giro (linear 0, §11.2)
    if tipo in ("reto", "terreno", "arco"):
        v[:, :2] = _anda(n, k.piso_velocidade, k.piso_lado, frente, tras, lado, gerador)
    if tipo in ("arco", "giro_lugar"):
        v[:, 2] = _giro(n, k.piso_velocidade, giro, gerador)
    return v


# --- cadeia de trechos (§6.3, D3 item 2) --------------------------------------------------------------

@dataclass
class TabelaTrechos:  # a tabela da D3 achatada nas três fatias; tensores (T,)
    tarefa: torch.Tensor  # long, índice em TAREFAS publicado no trecho
    por_detector: torch.Tensor  # bool: o trecho acaba pelo aviso (o fecho do P3 ou do B3, 13/11, M6) mais o atraso
    proximo: torch.Tensor  # long, −1 = fim da cadeia
    inicio: torch.Tensor  # (3,) long, o primeiro trecho de ANDAR, B e C


def _monta(k: Cadeia, device):
    # Tabela da D3 (S §7). Cada trecho: (tarefa, por_detector, faixa de sorteio em s). Para trecho por
    # aviso a faixa é o atraso; para trecho por tempo, a duração. Faixa (a, a) é valor fixo.
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
    Por aviso (o latch do fecho do P3 ou do B3): relógio corre só com o aviso, avança com aviso e relógio ≥ atraso. Ao
    avançar, relógio = 0.
    A tarefa publicada é `tabela.tarefa[trecho]`: no atraso do BOTAR, o one-hot fica em BOTAR (§6.3).
    ⚠ O aviso é exigido também para avançar: com atraso 0, um trecho por aviso sem aviso não pode avançar."""
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
                        carregar, incl_max_seg, terreno, k: Comando, deriva_pos_m=None):
    """(registra, sucesso), (E,) bool. Canal ativo (|c| ≥ limiar): razão ≥ portão. Canal zero: deriva da âncora.
    ⚠ `carregar` é "tarefa com caixa" (PEGAR, CARREGAR, BOTAR): PD-50, o registro exige não perder a caixa.
    Fase G (§6.3): a caixa que anda no `terreno` só passa com a inclinação máxima do segmento (`incl_max_seg`, graus)
    até `k.incl_max_terreno_graus`. O nome vem da assinatura da §6.3."""
    ativo_lin = cmd[:, :2].norm(dim=1) >= k.limiar_canal
    ativo_giro = cmd[:, 2].abs() >= k.limiar_canal
    razao = int_proj / int_cmd.clamp(min=1e-6)
    lim_pos = k.deriva_pos_m if deriva_pos_m is None else deriva_pos_m
    ok_lin = torch.where(ativo_lin, razao[:, 0] >= k.razao_portao, deriva_pos < lim_pos)
    ok_giro = torch.where(ativo_giro, razao[:, 1] >= k.razao_portao, deriva_rumo < k.deriva_rumo_rad)
    sucesso = ok_lin & ok_giro & ~(carregar & perdeu)
    sucesso &= ~(carregar & terreno & (incl_max_seg > k.incl_max_terreno_graus))
    registra = t_fecho >= t_ancora  # um segmento que fecha antes da âncora não entra (PD-15)
    return registra, sucesso


# --- odometria pelo degrau global (H5; PD-18, PD-41) -----------------------------------------------------

def deriva_do_degrau(k: Estimador, g: int, degraus: int) -> dict:
    # deriva da odometria no degrau global g (início em 0, refino em `degraus`); o par de cada chave é o de `Estimador`
    pares = {"odo": k.deriva_odo, "rumo": k.deriva_rumo, "pos_giro": k.deriva_pos_giro, "rumo_giro": k.deriva_rumo_giro}
    return {nome: balanceador.no_degrau(p[0], p[1], g, degraus) for nome, p in pares.items()}


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
    v3_fim, v3_aviso (latch do trecho, F9 P5; 13/11, M6: o fecho do P3 no PEGAR e o do B3 no BOTAR), v3_freio, v3_degrau,
    v3_alvo_idx, v3_alvo_w, v3_caixa_pega_w, v3_sigma, v3_relogio_alvo, v3_avancou, v3_pegou, v3_pega_fechada, v3_bota_fechada,
    v3_perdeu, v3_perdeu_agora, v3_ponto_apoio (fecho do B1, F9 S2), v3_v_chegada (primeiro toque do BOTAR, F9 P3), v3_vz_caixa_ant (F-2: a vz da caixa, todo env, todo
    passo); dá o `passo` da `v3_odo` uma vez por passo (PD-44)."""

    cfg: CadeiaCmdCfg

    def __init__(self, cfg: CadeiaCmdCfg, env):
        super().__init__(cfg, env)
        garante_v3(env)
        self.robot = env.scene[cfg.entity_name]
        self.tab, self.faixas = _monta(cfg.k, self.device)
        self.relogio = torch.zeros(self.num_envs, device=self.device)
        self.alvo_s = torch.zeros(self.num_envs, device=self.device)  # duração (ou atraso) sorteada do trecho
        self.pega = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # PEGAR aberto, sem registro
        self.bota = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # BOTAR aberto, até o fim da cadeia
        self.sit_palmas = self.robot.find_sites(("left_palm", "right_palm"))[0]  # ordem = `palmas_alvo_caixa`
        # 20/11, T13: o piso do σ por etapa; os tempos e o `Alvos` efetivo vêm da zona, refeitos a cada iteração em `_alvos`
        self.sigma_min = torch.tensor(A.sigma_min_etapa, device=self.device)
        self.it_zona = -1
        # visita da zona: a etapa que o env está visitando (−1 = nenhuma) e o tempo desde a abertura (−1 = visita encerrada)
        self.vis_e = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.t_vis = torch.full((self.num_envs,), -1.0, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # one-hot (E,6) de v3_tarefa (§6.5)
        return torch.nn.functional.one_hot(self._env.v3_tarefa, len(TAREFAS)).float()

    def _twist(self):
        return self._env.command_manager.get_term("twist")

    # --- medidas e alvos -----------------------------------------------------------------------------

    def _mede(self) -> dict:
        """Medidas do passo (E,...): base, caixa (verdade), laje, palmas, contatos, tronco (§6.2 `Alvos`) e a pelve sobre o
        pé mais baixo (`pelve`, R6 da E11: a medida do B3, também o da caixa perdida) e a distância das palmas à pose de
        pé (`dpalmas`, 13/11, A9)."""
        env, d = self._env, self.robot.data
        caixa, laje = env.scene["box"].data, env.scene["table"].data
        h = env.scene["altura_pelve"].data.heights[:, 0]
        rumo, caixa_p, caixa_q = d.heading_w, caixa.root_link_pos_w, caixa.root_link_quat_w
        palmas_w = d.site_pos_w[:, self.sit_palmas]  # (E,2,3)
        R = alvos.frame_da_caixa(caixa_q, rumo)
        m = dict(
            h=h, pelve=d.root_link_pos_w[:, 2] - z_pe_min(self.robot, env.v3_ref),
            base_p=d.root_link_pos_w, base_q=d.root_link_quat_w, caixa_p=caixa_p, palmas_w=palmas_w,
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
            # palmas no frame da base: a maior das duas distâncias à `palmas_de_pe_b`
            dpalmas=(quat_apply_inverse(d.root_link_quat_w[:, None].expand(-1, 2, -1), palmas_w - d.root_link_pos_w[:, None])
                     - env.v3_ref.palmas_de_pe_b.to(palmas_w.dtype)).norm(dim=-1).amax(-1),  # ref é float64
        )
        return m

    def _dist(self, m, indice) -> torch.Tensor:
        # d do alvo `indice` no estado do passo (E4, `recompensas.distancia_do_alvo`); usado no σ. O alvo da caixa é o
        # da recompensa, com o chão local (S10): uma conta só.
        # σ = distância inicial da mesma medida que o Φ do B3 usa (`distancia_do_alvo`, pela `h` do sensor altura_pelve).
        env = self._env
        dpelve = m["h"] - env.v3_ref.pelve_de_pe
        alvo = recompensas.alvo_da_caixa_w(env, indice)
        deficit = torch.relu(1.0 - m["apoio_N"] / (A.apoio_b1_mg * m["peso_N"]))  # 13/11, A3: o do `_phi_env`
        return recompensas.distancia_do_alvo(indice, m["palmas_c"], m["caixa_p"], alvo, dpelve, m["tronco_graus"],
                                             m["dpalmas"], deficit, env.v3_meia, A)

    def _sigma(self, m, indice) -> torch.Tensor:
        # 20/11, T13: piso por etapa (`sigma_min_etapa`); o B3 é adimensional e o piso dele é 0 (M12).
        d = self._dist(m, indice)
        return torch.maximum(d, self.sigma_min[indice.clamp(min=0)])

    def _espera(self) -> torch.Tensor:
        # (E,) bool: PEGAR sem alvo, à espera da primeira leitura válida da caixa (E11 S5b). Não usa `self.pega`: esse
        # marca o PEGAR só depois que o P0 abre (F9 P6), e a espera é justamente o trecho antes do P0
        env = self._env
        return (env.v3_alvo_idx < 0) & (env.v3_tarefa == PEGAR)

    def _abre_p0(self, m) -> None:
        # Abre P0 do PEGAR na primeira leitura VÁLIDA da caixa (E11 S5b, decisão do dono): idade < idade_max, que só
        # acontece com a caixa já vista (sem leitura, a idade é idade_max). Até lá o trecho fica sem alvo (−1) e o robô
        # espera em pé: sem alvo não há fecho do P3, e o relógio do PEGAR não corre. A pega e o alvo saem da caixa ESTIMADA (com os
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
        self.pega[ids] = True  # F9 P6: só com o P0 aberto o PEGAR fica pendente de registro (sem caixa vista, não conta)
        pos = estimador.do_frame_da_base(est[:, :3], odo)[ids]
        env.v3_alvo_idx[ids] = P0
        env.v3_phi[ids] = -1.0  # sentinela das métricas `phi_<alvo>` (o progresso grava o Φ do alvo corrente)
        env.v3_h[ids], env.v3_phi_melhor[ids] = 1.0, 0.0  # 13/11, M5: a renda da etapa recomeça
        env.v3_relogio_alvo[ids] = 0.0
        env.v3_caixa_pega_w[ids] = pos
        env.v3_alvo_w[ids] = pos
        env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, P0))[ids]

    def _perda(self, m) -> None:
        # Caixa perdida (E3, item 4; S §10). `v3_perdeu_agora` é um pulso no passo da perda, escrito aqui no `compute` da
        # cadeia, DEPOIS das terminações e das recompensas do mesmo passo: a terminação `caixa_perdida` e a recompensa
        # `perda_caixa` o leem só no passo SEGUINTE (0,02 s de atraso; o comando roda depois das terminações, como no
        # mjlab e no Isaac Lab). No ramo `caiu`, a medida é o raycast `altura_caixa`, que só recalcula no `sim.sense()`
        # (um passo atrás): a cobrança desse ramo sai dois passos depois da física.
        # Caso 6: no primeiro passo do episódio (`episode_length_buf` == 0) o `h_caixa` é o do fim do episódio anterior
        # (o `sense()` ainda não rodou sobre o estado novo). Por isso a perda não vale nesse passo: senão um PEGAR que
        # nasce com a caixa no chão, pelo dado velho, marcaria perda falsa e, com a chave, viraria ANDAR.
        # Com a chave, a caixa perdida vira ANDAR parado até o fim (PD-27): o alvo passa a B3, com σ = a distância à pose
        # de pé no instante da perda.
        env = self._env
        tar = env.v3_tarefa
        longe = (m["caixa_p"] - env.v3_alvo_w).norm(dim=-1) > A.raio_b1
        fora = torch.where(tar == BOTAR, longe, torch.ones_like(longe))  # fora do alvo de soltura, só no BOTAR
        perda = caixa_perdida_agora(m["h_caixa"] - env.v3_meia, m["v_rel"], fora, env.v3_pegou, env.v3_alvo_idx,
                                    m["sobre"], T) & (tar != ANDAR) & (env.episode_length_buf > 0)
        env.v3_perdeu_agora[:] = perda
        if self.cfg.k.caixa_perdida_vira_andar and perda.any():
            ids = perda.nonzero().flatten()
            env.v3_perdeu[ids] = True
            env.v3_alvo_idx[ids] = B3
            env.v3_phi[ids] = -1.0  # sentinela das métricas `phi_<alvo>`; o progresso grava o Φ do B3 no passo seguinte
            env.v3_h[ids], env.v3_phi_melhor[ids] = 1.0, 0.0  # 13/11, M5
            env.v3_relogio_alvo[ids] = 0.0
            env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, B3))[ids]
            self._publica(ids, torch.full_like(ids, ANDAR))

    def _alvos(self, m, dt) -> None:
        # Alvos do PEGAR (E3, item 2; S §8.2) e do BOTAR (F1), num passo só: o `ok_agora` do alvo corrente, `alvos.avanca`
        # e os buffers do avanço. ⚠ Um só `avanca`: dois sobre o mesmo relógio se zerariam. O índice fica em P3 depois do
        # P3 (a troca de tarefa leva a B1). O B3 da caixa perdida fecha pela pose de pé, sem o portão desloc (E11).
        env = self._env
        idx, a, ref, tar, cur = env.v3_alvo_idx, env.v3_meia, env.v3_ref, env.v3_tarefa, env.v3_cur
        if self.it_zona != cur.it:
            # 20/11, T13: a zona muda no fecho da iteração. O `Alvos` efetivo (final + z·(largo − final)) e os tempos de
            # permanência são refeitos uma vez por iteração; o fecho (`ok_agora_*`) e a qualidade do CARREGAR leem o efetivo,
            # e o Φ, o `d`, os portões e o ponto-alvo seguem no `A` final
            self.it_zona = cur.it
            palma, efetivo = [A.tol_palma] * len(ALVOS), {}
            for nome, campo, largo in A.zona:
                e = ALVOS.index(nome)
                v = balanceador.no_degrau(getattr(A, campo), largo, cur.z[e], 1)
                if campo == "tol_palma":
                    palma[e] = v  # P0 e P1 repartem o campo, cada uma com o z dela
                else:
                    efetivo[campo] = v
            env.v3_alvos = replace(A, tol_palma=torch.tensor(palma, device=self.device), **efetivo)
            self.tempos = torch.tensor([0.0, 0.0, efetivo["tempo_p2"], efetivo["tempo_p3"], efetivo["tempo_b1"],
                                        efetivo["tempo_b2"], efetivo["tempo_b3"]], device=self.device)
        # 20/11, T13: visitas da zona. Uma visita é a etapa aberta. Ela acaba com o fecho (conta fecho e visita) ou sem
        # fecho (conta só visita): `zona_t_visita` desde a abertura, a perda da caixa ou a troca da etapa por fora (a tarefa
        # mudou). O reset conta em `_resample`. Depois de encerrada a visita não reabre até a etapa mudar (nada conta duas vezes)
        etapa = torch.where((idx >= 0) & ~((idx == P3) & env.v3_pega_fechada) & ~env.v3_perdeu, idx, -1)
        mudou = etapa != self.vis_e
        acaba = torch.where((self.t_vis >= 0) & mudou, self.vis_e, -1)
        self.vis_e = etapa
        self.t_vis = torch.where(mudou, torch.where(etapa >= 0, 0.0, -1.0), self.t_vis)
        self.t_vis = torch.where(self.t_vis >= 0, self.t_vis + dt, self.t_vis)
        zero = torch.zeros_like(m["h"])
        botar = (tar == BOTAR) | env.v3_perdeu  # B1..B3, e o B3 da caixa perdida (E3)
        desl = (m["caixa_p"][:, :2] - env.v3_caixa_pega_w[:, :2]).norm(dim=-1)
        # F9 S2/P1: o ponto de apoio é a caixa no fecho do B1 (gravado depois do `avanca`); o portão desloc de B2–B3 lê dele
        desl_apoio = (m["caixa_p"][:, :2] - env.v3_ponto_apoio[:, :2]).norm(dim=-1)
        # 13/11, A5: cada deslocamento no argumento do próprio portão (antes, o do apoio ia no lugar do da pega e o
        # zero no lugar do do apoio: o fecho do B2 e do B3 ignorava o arrasto). Sem perda; a caixa perdida zera os dois.
        desl_pega = torch.where((tar != BOTAR) & ~env.v3_perdeu, desl, zero)
        desl_apoio = torch.where((tar == BOTAR) & ~env.v3_perdeu, desl_apoio, zero)
        # Invariante = desloc · incl · maos (§6.3). Tronco, recuo e CoM não entram nele: ficam em zero (rampa 1).
        g = recompensas.portoes_do_alvo(
            idx, desl_pega, desl_apoio, m["incl_graus"], recompensas.incl_lim(env), m["contatos"].float().sum(-1) / 2.0,
            zero, zero, zero, torch.ones_like(idx, dtype=torch.bool), a, A)
        inv = g["desloc"] * g["incl"] * g["maos"]
        # PEGAR
        d_palmas = (m["palmas_c"] - alvos.palmas_alvo_caixa(idx, a, A)).norm(dim=-1)
        d_transporte = (m["caixa_p"] - recompensas.alvo_da_caixa_w(env, torch.full_like(idx, P3))).norm(dim=-1)
        d_pelve = (m["h"] - ref.pelve_com_caixa(a)).abs()
        pega = (tar == PEGAR) & ~env.v3_pega_fechada
        ok = ok_agora_pegar(idx, d_palmas, m["contatos"], m["apoio"],
                            m["h_caixa"] - a - env.v3_topo,  # fundo da caixa sobre a laje (verdade, não a estimada)
                            d_transporte, d_pelve, inv, env.v3_alvos) & pega
        if botar.any():  # BOTAR (B1..B3); o B3 da caixa perdida cai aqui com `inv` = 1 (o `desl` zerado)
            dif = m["caixa_p"] - recompensas.alvo_da_caixa_w(env, idx)
            # F9 S3: a palma mede da face, a + palma_face (o sítio da palma com a almofada encostada fica a palma_face da face; T11a)
            afasta = (m["palmas_c"].abs() - (a + A.palma_face)[:, None, None]).amax(-1)
            ok = ok | ok_agora_botar(idx, dif.norm(dim=-1), m["apoio_N"], m["peso_N"], afasta,
                                     m["apoio"], m["v_caixa"], m["pelve"] - ref.pelve_de_pe, m["tronco_graus"], m["dpalmas"],
                                     inv, env.v3_alvos)
        novo, rel, av = alvos.avanca(idx, ok, env.v3_relogio_alvo, dt, self.tempos)
        fecho = av & (self.t_vis >= 0)
        venceu = ~av & (self.t_vis >= ATIVO.balanceador.zona_t_visita)
        encerra = fecho | venceu
        visitas = torch.bincount(torch.cat([acaba[acaba >= 0], self.vis_e[encerra]]), minlength=len(ALVOS))
        fechos = torch.bincount(self.vis_e[fecho], minlength=len(ALVOS))
        for e, (v, f) in enumerate(zip(*torch.stack([visitas, fechos]).tolist())):
            cur.vis_acum[e] += v
            cur.fech_acum[e] += f
        self.t_vis = torch.where(encerra, -1.0, self.t_vis)
        fecha_p3 = av & (idx == P3)
        novo = torch.where(fecha_p3, idx, novo)
        fecha_b1 = av & (idx == B1)
        env.v3_ponto_apoio[fecha_b1] = m["caixa_p"][fecha_b1]
        env.v3_pega_fechada |= fecha_p3
        env.v3_pegou |= av & (idx == P2)  # a caixa passou ao P3: pega feita
        env.v3_bota_fechada |= av & (idx == B3)
        env.v3_avancou[:] = av
        env.v3_relogio_alvo[:] = rel
        env.v3_alvo_idx[:] = novo
        troca = av & ~fecha_p3 & (novo >= 0)
        env.v3_sigma[:] = torch.where(troca, self._sigma(m, novo), env.v3_sigma)
        env.v3_h[troca], env.v3_phi_melhor[troca] = 1.0, 0.0  # 13/11, M5: a etapa nova começa com h cheio

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
        # registra no fim da cadeia (falhou = False) e no reset (falhou = terminação que não é time_out, como o PEGAR; F9 P7).
        env = self._env
        pend = self.bota if botar else self.pega
        sel = pend[ids]
        ids, falhou = ids[sel], falhou[sel]
        if ids.numel() == 0:
            return
        # 20/11, T13: o sucesso só conta com o critério final, o z da cadeia da tarefa todo em 0; antes disso entra como falha
        # (o nível, o subnível, o refino e as travas do DR da tarefa só avançam com o critério final)
        if botar:
            suc = sucesso_do_botar(env.v3_bota_fechada[ids], falhou, env.v3_perdeu[ids]) & (max(env.v3_cur.z[4:]) == 0.0)
            celula = env.v3_cur.id_celula("BOTAR", "botar")
        else:
            suc = sucesso_do_pegar(env.v3_pega_fechada[ids], falhou, env.v3_perdeu[ids]) & (max(env.v3_cur.z[:4]) == 0.0)
            celula = env.v3_cur.id_celula("PEGAR", "pegar")
        env.v3_cur.registra(celula, int(ids.numel()), int(suc.sum()))
        pend[ids] = False

    def _abre_botar(self, ids, m) -> None:
        # Abertura do BOTAR (F1; PD-47): a laje de destino dá o ponto de soltura no topo, e o alvo sobe `a` (a caixa
        # fica com o centro a meia aresta acima). Alvo corrente B1, σ pelo `_sigma` como o P0 do PEGAR, e o salto
        # (`v3_phi` = −1 é a sentinela das métricas). O ponto de apoio (F9 S2) zera aqui e é gravado no fecho do B1; o
        # primeiro toque (F9 P3) também volta a −1.
        env = self._env
        soltura = eventos.posiciona_destino(env, ids).to(env.v3_alvo_w.dtype)
        soltura[:, 2] += env.v3_meia[ids]
        if env.v3_gerador is not None:
            # P22: o robô real só conhece o alvo estimado, então o viés do alvo entra no `v3_alvo_w` (vale para a
            # observação, o Φ do B1 e o raio da perda). ‖viés‖ ≤ 7,7 cm < 0,10 m do enunciado
            g, d = env.v3_cur.degrau_global(), ATIVO.balanceador.degraus
            k = self.cfg.est
            vxy, vz = (balanceador.no_degrau(p[0], p[1], g, d) for p in (k.vies_alvo_botar_xy, k.vies_alvo_botar_z))
            u = _u((len(ids), 3), env.v3_gerador).to(soltura.device)
            soltura += (2 * u - 1) * torch.tensor([vxy, vxy, vz], dtype=soltura.dtype, device=soltura.device)
        env.v3_alvo_w[ids] = soltura
        env.v3_alvo_idx[ids] = B1
        env.v3_phi[ids] = -1.0
        env.v3_h[ids], env.v3_phi_melhor[ids] = 1.0, 0.0  # 13/11, M5
        env.v3_relogio_alvo[ids] = 0.0
        env.v3_ponto_apoio[ids] = 0.0
        env.v3_v_chegada[ids] = -1.0
        env.v3_sigma[ids] = self._sigma(m, torch.full_like(env.v3_alvo_idx, B1))[ids]
        self.bota[ids] = True

    def _abre_trecho(self, ids, trecho, m=None) -> None:
        # Sorteia a duração (ou atraso) do trecho, publica a tarefa e o freio/degrau, e reamostra o comando do twist
        # (o CARREGAR de B fica fixo na abertura, §7; o parado sai com comando zero). O PEGAR abre sem alvo (E11 S5b;
        # o P0 abre em `_abre_p0`, que o torna pendente de registro, F9 P6), registra o PEGAR anterior quando abre o
        # BOTAR (S §11.3) e tira a laje do caminho do CARREGAR que anda (C39). `m` (a medida do passo) só é preciso
        # para o BOTAR, que nunca abre no reset (F9 M6).
        env = self._env
        tar = self.tab.tarefa[trecho]
        if m is None and bool((tar == BOTAR).any()):
            raise ValueError("_abre_trecho: o BOTAR precisa da medida `m` (o reset não abre BOTAR, F9 M6)")
        lo, hi = self.faixas[trecho, 0], self.faixas[trecho, 1]
        self.alvo_s[ids] = lo + (hi - lo) * _u(len(ids), env.v3_gerador).to(self.device)
        self.relogio[ids] = 0.0
        env.v3_trecho[ids] = trecho
        env.v3_aviso[ids] = False  # F9 P5: o latch do aviso começa de zero em cada trecho
        env.v3_h[ids], env.v3_phi_melhor[ids] = 1.0, 0.0  # 13/11, M5: cada trecho começa com a renda da etapa cheia
        self._publica(ids, tar)
        sem_alvo = (tar == ANDAR) | (tar == PEGAR)
        env.v3_alvo_idx[ids[sem_alvo]] = -1
        env.v3_phi[ids[sem_alvo]] = -1.0  # sentinela das métricas `phi_<alvo>`: sem alvo
        botar = tar == BOTAR
        self._registra(ids[botar], torch.zeros(int(botar.sum()), dtype=torch.bool, device=self.device))
        if botar.any():
            self._abre_botar(ids[botar], m)
        anda = (tar == CARREGAR) & (env.v3_fatia[ids] == FATIA_B) & (self._twist().vel_command_b[ids].abs().sum(-1) > 0)
        if anda.any():
            eventos.laje_fora(env, ids[anda])

    def compute(self, dt: float) -> None:
        env = self._env
        m = self._mede()
        self._perda(m)
        # 13/11, M1: o `extras` do mjlab persiste entre passos (só o "log" é refeito); o `PPOPorTarefa` lê o pulso da perda
        env.extras["v3_perdeu_agora"] = env.v3_perdeu_agora
        self._abre_p0(m)
        self._alvos(m, dt)
        # 13/11, M6: o aviso do trecho é o fecho do alvo dele (P3 no PEGAR, B3 no BOTAR), só nos trechos por aviso. O
        # latch fica, e o `avanca_trecho` lê o latch, e não o fecho do instante (F9 P5). O atraso sorteado segue.
        fecho = torch.where(env.v3_tarefa == BOTAR, env.v3_bota_fechada, env.v3_pega_fechada)
        env.v3_aviso[:] = env.v3_aviso | (fecho & self.tab.por_detector[env.v3_trecho])
        novo, self.relogio, avancou, fim = avanca_trecho(
            env.v3_trecho, self.relogio, self.alvo_s, env.v3_aviso, self.alvo_s, dt, self.tab)
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
        # PD-44: o `passo` da odometria uma vez por passo, antes das escritas do fim. Sem ela a odometria do treino não
        # anda (`delta_b` zero, o mapa não é transportado, a caixa e o alvo do ator saem no frame do reset). O commit
        # fce6a42 a tirou sem registro (achado de 30/10, caso 8); a H5 a devolve, com a deriva do degrau global
        g, d = env.v3_cur.degrau_global(), ATIVO.balanceador.degraus
        env.v3_odo.passo(self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w,
                         deriva_do_degrau(self.cfg.est, g, d))
        self._grava_chegada()
        # vz da caixa para a `chegada` e a métrica do passo seguinte. A cadeia roda depois das terminações, das
        # recompensas, das métricas e do reset: quem lê no passo seguinte vê a vz do passo anterior (§6.4).
        env.v3_vz_caixa_ant[:] = env.scene["box"].data.root_link_lin_vel_w[:, 2]

    def _grava_chegada(self) -> None:
        # F9 P3: a velocidade do primeiro toque do BOTAR, gravada uma vez (−1 = sem toque). É a vz do passo anterior, a
        # da descida antes do pouso: a mesma conta de `metricas.velocidade_de_chegada` e da `chegada`.
        env = self._env
        primeiro = env.scene["apoio_caixa"].compute_first_contact(dt=env.step_dt)[:, 0]
        novo = (env.v3_tarefa == BOTAR) & primeiro & (env.v3_v_chegada < 0)
        env.v3_v_chegada[novo] = (-env.v3_vz_caixa_ant).clamp(min=0)[novo]

    def _resample(self, env_ids) -> None:  # chamado no reset do episódio (CommandTerm.reset)
        env = self._env
        # o PEGAR e o BOTAR abertos se registram no reset: queda ou outra terminação conta como falha (S §11.3); o
        # `time_out` não é terminação (S §10), então um BOTAR com B3 fechado que o tempo corta conta sucesso (F9 P7)
        falhou = ~env.termination_manager.time_outs[env_ids]
        self._registra(env_ids, falhou)
        self._registra(env_ids, falhou, botar=True)
        # 20/11, T13: a visita aberta no reset conta falha na zona
        aberta = self.vis_e[env_ids][self.t_vis[env_ids] >= 0]
        for e in aberta.tolist():
            env.v3_cur.vis_acum[e] += 1
        self.vis_e[env_ids], self.t_vis[env_ids] = -1, -1.0
        for nome in ("v3_pegou", "v3_perdeu", "v3_pega_fechada", "v3_bota_fechada", "v3_perdeu_agora", "v3_avancou",
                     "v3_aviso", "v3_fim"):
            getattr(env, nome)[env_ids] = False
        env.v3_relogio_alvo[env_ids] = 0.0
        env.v3_sigma[env_ids] = 1.0  # neutro: o alvo −1 não o usa, e σ = 0 daria NaN
        env.v3_phi[env_ids] = -1.0  # sem alvo no início do episódio: sentinela das métricas `phi_<alvo>`
        env.v3_h[env_ids], env.v3_phi_melhor[env_ids] = 1.0, 0.0  # 13/11, M5
        env.v3_v_chegada[env_ids] = -1.0
        fat = env.v3_fatia[env_ids]
        self._abre_trecho(env_ids, self.tab.inicio.to(fat.device)[fat])  # o reset não mede: o BOTAR nunca abre aqui (M6)

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


_METRICAS = ("razao_marcha", "acerto_relogio", "eficiencia_min", "eficiencia_media", "segmentos")


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
        self.soma_acerto = torch.zeros(n, device=dev)  # 13/11, M4: Σ acerto do relógio e nº de passos com comando
        self.passos_on = torch.zeros(n, device=dev)
        # relógio do episódio, próprio: o reset do mjlab zera episode_length_buf DEPOIS do command_manager.reset
        # (manager_based_rl_env.py:581-590; §6.4), por isso não se lê o episode_length_buf aqui
        self._agora = torch.zeros(n, device=dev)
        # 19/11, T12: fase da marcha própria, só anda com comando (`on`); o `_agora` segue contando a âncora e a abertura
        self._fase_t = torch.zeros(n, device=dev)
        # segmento aberto: snapshot do comando, célula e "tarefa com caixa" no instante da abertura
        self.seg_cmd = torch.zeros(n, 3, device=dev)
        self.seg_cel = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.seg_carrega = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seg_proj = torch.zeros(n, 2, device=dev)  # [linear, giro]: Σ(v·ĉ)dt, Σ(ω·sinal)dt
        self.seg_int = torch.zeros(n, 2, device=dev)  # [linear, giro]: Σ|c|dt, Σ|wz|dt
        # 24/11, T15: Σ v·dt desde a âncora, [vx, vy, ωz]. Com o comando fixo no segmento, a razão do canal sai da média: (v̄·ĉ)/‖c‖
        self.seg_nota = torch.zeros(n, 3, device=dev)
        self.v_filt = torch.zeros(n, 3, device=dev)  # 24/11, T15: velocidade da base filtrada [vx, vy, ωz], a que a recompensa lê; zera no reset
        self.seg_dur = torch.zeros(n, device=dev)  # 16/11, A1: Σ dt depois da âncora
        self.t_ab = torch.zeros(n, device=dev)  # instante da abertura do segmento (base da âncora)
        self.ancora_feita = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seg_visto = self.command_counter.clone()
        # pose (x, y, rumo no mundo) do fim do último passo: o `_reset_idx` do mjlab (manager_based_rl_env.py:553)
        # teletransporta o robô ANTES do `command_manager.reset`, então `robot.data` no `_fecha` do reset é o respawn
        self.pose_ult = torch.zeros(n, 3, device=dev)
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
        self.seg_nota[ids] = 0.0
        self.seg_dur[ids] = 0.0
        self.ancora_feita[ids] = False
        env.v3_incl_max_seg[ids] = 0.0  # G3: a inclinação máxima do segmento começa de zero

    def _mede_incl(self) -> None:
        # G3 (§6.3 "Fase G"): a inclinação máxima da caixa no segmento corrente, em graus. Mesma conta da cadeia (`_mede`).
        env = self._env
        graus = torch.rad2deg(alvos.inclinacao_da_caixa(env.scene["box"].data.root_link_quat_w))
        env.v3_incl_max_seg[:] = torch.maximum(env.v3_incl_max_seg, graus)

    def _fecha(self, ids, falha) -> None:
        # Fecha o segmento de `ids` com a pose e o relógio do instante; `falha` (terminação que não é
        # time_out) zera o sucesso e registra mesmo antes da âncora (D3 item 4; S §11.3 "sem queda").
        env, k = self._env, self.k
        ancora = env.v3_ancora[ids]
        pos = self.pose_ult[ids, :2]
        dpos = (pos - ancora[:, :2]).norm(dim=1)
        drum = wrap_to_pi(self.pose_ult[ids, 2] - ancora[:, 2]).abs()
        # G3: o terreno do segmento é a célula `terreno` do CARREGAR (só ele tem célula de terreno)
        terreno = self.seg_cel[ids] == env.v3_cur.id_celula("CARREGAR", "terreno")
        reg, suc = sucesso_do_segmento(
            self.seg_cmd[ids], self.seg_proj[ids], self.seg_int[ids], dpos, drum,
            self._agora[ids], ancora[:, 3], env.v3_perdeu[ids], self.seg_carrega[ids], env.v3_incl_max_seg[ids],
            terreno, k,
            deriva_pos_m=balanceador.no_degrau(k.deriva_pos_inicio, k.deriva_pos_m, env.v3_cur.restricao(), 1))
        # 24/11, T15: a nota do ANDAR é a média dos dois canais. Canal comandado: min(razão, 1) com a razão desde a âncora,
        # (v̄·ĉ)/‖c‖ no linear e ω̄/c_z no giro (a razão negativa vale 0, o excesso vale 1: o balanço da passada se anula na
        # média). Canal sem comando: a deriva, o núcleo exp(−v̄²/σ²) da média do segmento (o giro parado que gira é defeito
        # medido). Queda: 0. As tarefas com caixa seguem com o sucesso. Um segmento do ANDAR sem tempo depois da âncora não registra, a não ser que acabe em queda.
        # 17/11, T6: o ANDAR não usa a regra da âncora do `sucesso_do_segmento`: depois de cada empurrão só as
        # quedas registravam, e a EMA tomava zeros
        c = self.seg_cmd[ids]
        v_med = self.seg_nota[ids] / self.seg_dur[ids, None].clamp(min=1e-6)
        den = torch.stack([c[:, :2].norm(dim=1), c[:, 2].abs()], dim=1)
        ativo = den >= k.limiar_canal
        den = den.clamp(min=k.limiar_canal)
        proj = torch.stack([(v_med[:, :2] * c[:, :2]).sum(dim=1) / den[:, 0], v_med[:, 2] * c[:, 2].sign()], dim=1)
        razao = (proj / den).clamp(0.0, 1.0)
        deriva = torch.exp(-torch.stack([v_med[:, :2].pow(2).sum(dim=1), v_med[:, 2].pow(2)], dim=1) / ATIVO.recompensa.sigma_rastreio ** 2)
        nota = torch.where(ativo, razao, deriva).mean(dim=1)
        x = torch.where(self.seg_carrega[ids], suc.float(), nota) * (~falha).float()
        reg = torch.where(self.seg_carrega[ids], reg, self.seg_dur[ids] > 0) | falha
        cel = self.seg_cel[ids]
        for c in cel[reg & (cel >= 0)].unique().tolist():  # uma chamada por célula (registra é escalar)
            m = reg & (cel == c)
            env.v3_cur.registra(int(c), int(m.sum()), float(x[m].sum()))
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
        self.seg_nota[ids] = 0.0
        self.seg_dur[ids] = 0.0

    def _update_metrics(self) -> None:
        env, k = self._env, self.k
        dt = env.step_dt
        self._agora += dt
        self.pose_ult[:, :2] = self.robot.data.root_link_pos_w[:, :2]
        self.pose_ult[:, 2] = self.robot.data.heading_w
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
        self._mede_incl()
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
        # 24/11, T15: a velocidade do passo (`vel`, `wz` do bloco 2) soma desde a âncora (a razão da nota sai da média no `_fecha`)
        # e entra no filtro exponencial de τ `tau_fracao` que a recompensa lê. O filtro só vale para o canal comandado: a deriva
        # do canal sem comando e o `giro_xy` seguem instantâneos (16/11, T3: a média anulava a torção da pelve).
        v3 = torch.cat([vel, wz[:, None]], dim=1)
        apos = (self._agora >= env.v3_ancora[:, 3]).float()
        self.seg_nota += v3 * (apos * dt)[:, None]
        self.seg_dur += apos * dt
        self.v_filt += -math.expm1(-dt / k.tau_fracao) * (v3 - self.v_filt)
        # 5. relógio de marcha (16/11, R: a forma do `feet_gait` da Unitree): roda depois das recompensas, então o
        # `passo` do passo t lê a fase do passo t − 1. 19/11, T12: a fase é o tempo com comando (`on`: ‖comando‖ >
        # limiar) sobre o período fixo; com comando zero ela para onde estava e volta dali. Sem isso, o P3 (comando
        # zero) via a fase correr e levantava a caixa no pulso dela, numa janela menor que o `tempo_p3`.
        on = cmd.norm(dim=1) > k.limiar_marcha
        self._fase_t += dt * on
        env.v3_fase_marcha[:] = torch.remainder(self._fase_t / k.periodo_marcha, 1.0)
        env.v3_marcha_on[:] = on
        self.soma_acerto += env.v3_acerto_marcha * on
        self.passos_on += on
        self.metrics["acerto_relogio"][:] = self.soma_acerto / self.passos_on.clamp(min=1.0)

    def reset(self, env_ids):
        # ⚠ A cadeia roda ANTES do twist (§6.5): o trecho novo já está publicado, por isso o segmento antigo usa o
        # snapshot da abertura (seg_cel, seg_carrega), e não os buffers vivos.
        time_out = self._env.termination_manager.time_outs[env_ids]
        self._fecha(env_ids, falha=~time_out)
        self._agora[env_ids] = 0.0
        self._fase_t[env_ids] = 0.0
        self.v_filt[env_ids] = 0.0  # o filtro é estado do episódio: não vaza para o seguinte
        self.soma_erro_marcha[env_ids] = 0.0
        self.soma_cmd_marcha[env_ids] = 0.0
        self.soma_acerto[env_ids] = 0.0
        self.passos_on[env_ids] = 0.0
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
