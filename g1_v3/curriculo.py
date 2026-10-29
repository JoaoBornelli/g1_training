"""Estado do currículo no env e o termo de currículo `v3` (D4 do plano; §6.4, §6.6, §6.7; S §11).

O estado inteiro (células, níveis, fatias, refino) mora em `balanceador.Curriculo`, puro e serializável.
Este arquivo só o prende ao env: cria os buffers `v3_*` da §6.4 (`garante_v3`), dá a máscara por tarefa
(`mascara`, §6.6) e, a cada reset, fecha a iteração do balanceador e devolve o log (`v3`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g1_v3 import alvos, balanceador, estimador, eventos, knobs, percepcao_lidar

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_K = knobs.ATIVO
_CAMINHO_REF = str(knobs.caminho(knobs.Contrato().referencias))  # relativo à raiz do repo (A3)
_CAMINHO_MID360 = str(knobs.caminho("g1_v3/dados/mid360_passos.npz"))  # PD-5, gerado por gera_mid360.py

# Buffers da §6.4 que `garante_v3` cria, por forma e dtype. Forma sem a dimensão E. Os valores iniciais
# seguem a §6.4: alvo e célula −1 = nenhum; t_empurrao −∞ = sem empurrão; 0 no resto.
_LONGOS = ("v3_fatia", "v3_trecho", "v3_tarefa", "v3_freio", "v3_degrau")
_BOOLS = ("v3_pegou", "v3_perdeu", "v3_fim", "v3_aviso", "v3_avancou", "v3_p4_fechado", "v3_b4_fechado",
          "v3_perdeu_agora")
_ESCALARES = ("v3_meia", "v3_massa", "v3_topo", "v3_topo_destino", "v3_sigma", "v3_relogio_alvo",
              "v3_b_pagos", "v3_vz_caixa_ant", "v3_incl_max_seg", "v3_sigma_velho")
_VETORES = {"v3_caixa_pega_w": 3, "v3_alvo_w": 3, "v3_ponto_apoio": 3, "v3_vies_giro": 3, "v3_ancora": 4,
            "v3_mapa_obs": 882, "v3_caixa_obs": 11}  # 882 = mapa do ator (I §2.4); 11 = caixa_obs (§6.4)


def _cria(env: "ManagerBasedRlEnv", nome: str, forma: tuple, dtipo, valor=0) -> None:
    # ⚠ só cria se não existe: `garante_v3` roda em todo termo e não pode zerar o estado do episódio.
    if not hasattr(env, nome):
        setattr(env, nome, torch.full((env.num_envs, *forma), valor, dtype=dtipo, device=env.device))


def garante_v3(env: "ManagerBasedRlEnv") -> None:
    """Cria os buffers `v3_*` da §6.4 que ainda não existem. Idempotente (padrão de `g1_limpo/curriculo.py:61-97`).

    ⚠ `v3_odo` usa um gerador próprio, semeado pela seed do cfg: com `None` o ruído da odometria sairia
    desligado em silêncio.
    """
    n, dev = env.num_envs, env.device
    if not hasattr(env, "v3_cur"):
        env.v3_cur = balanceador.Curriculo(_K.balanceador, _K.niveis, _K.comando, _K.terreno, k_dr=_K.dr)
    if not hasattr(env, "v3_ref"):  # §6.4: uma instância de alvos.Referencias (float64)
        env.v3_ref = alvos.Referencias(_CAMINHO_REF, dev)
    if not hasattr(env, "v3_filtro"):
        env.v3_filtro = percepcao_lidar.FiltroDeElevacao(n, dev, _K.percepcao)
    if not hasattr(env, "v3_padrao"):
        env.v3_padrao = percepcao_lidar.PadraoMid360(_CAMINHO_MID360, dev)
    # guarda de iteração do termo `v3` (§6.4 não lista; sem ela não há como saber se a iteração virou).
    # Escalar, não buffer: é um inteiro do env inteiro (o mesmo papel do `ultima_iter_bal` do g1_limpo).
    if not hasattr(env, "v3_iter"):
        env.v3_iter = -1
    if not hasattr(env, "v3_percepcao_passo"):  # §6.4: int, o common_step_counter da última percepção
        env.v3_percepcao_passo = -1
    if not hasattr(env, "v3_gerador"):  # ruído da odometria: semente do cfg (0 se o env não tiver cfg)
        env.v3_gerador = torch.Generator(device=dev)
        env.v3_gerador.manual_seed(getattr(getattr(env, "cfg", None), "seed", None) or 0)
    if not hasattr(env, "v3_odo"):
        env.v3_odo = estimador.Odometria(n, dev, _K.estimador, env.v3_gerador)
    if not hasattr(env, "v3_estimador"):  # §6.4 (fase E): o estimador da caixa, com o mesmo gerador
        env.v3_estimador = estimador.EstimadorCaixa(n, dev, _K.estimador, env.v3_gerador)
    for nome in _LONGOS:
        _cria(env, nome, (), torch.long)
    _cria(env, "v3_alvo_idx", (), torch.long, -1)
    _cria(env, "v3_celula", (), torch.long, -1)
    for nome in _BOOLS:
        _cria(env, nome, (), torch.bool, False)
    for nome in _ESCALARES:
        _cria(env, nome, (), torch.float32)
    _cria(env, "v3_t_empurrao", (), torch.float32, float("-inf"))
    _cria(env, "v3_phi", (), torch.float32, -1.0)  # sentinela −1 = sem alvo; o passo de entrada no alvo paga 0 (E4)
    for nome, dim in _VETORES.items():
        _cria(env, nome, (dim,), torch.float32)
    if not hasattr(env, "v3_caixa_xy"):  # (E,2) float32, como os buffers de posição: xy da caixa de B e C (E11 S4)
        env.v3_caixa_xy = torch.zeros((n, 2), dtype=torch.float32, device=dev)
    if not hasattr(env, "v3_incl_imu"):  # (E,3,3) giro fixo do IMU: começa na identidade, não em zeros
        env.v3_incl_imu = torch.eye(3, device=dev).repeat(n, 1, 1)


def mascara(env: "ManagerBasedRlEnv", tarefas: tuple[str, ...]) -> torch.Tensor:
    """(E,) bool: True onde a `v3_tarefa` do env está em `tarefas` (§6.6).

    ⚠ Devolve bool e o termo aplica com `torch.where`. Multiplicar por ela dá NaN num env fora da tarefa
    (NaN × 0 = NaN), e o NaN chega à recompensa, ao ator e às métricas.
    """
    tabela = torch.tensor([t in tarefas for t in knobs.TAREFAS], dtype=torch.bool, device=env.device)
    return tabela[env.v3_tarefa]


def v3(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> dict:
    """Termo de currículo `v3` (um só, §6.5). Roda a cada reset; devolve o dict de log da §6.7.

    1) Fecha a iteração do balanceador uma vez por iteração de `passos_por_iteracao` passos, e não por reset
       (a iteração é `common_step_counter // passos_por_iteracao`). A 1ª chamada só fixa a iteração corrente.
    2) Devolve as chaves `frac_*`, `s_<tarefa>_<tipo>`, `nivel_<tarefa>` e `degrau_<tarefa>`. O mjlab as
       publica como `Curriculum/v3/<k>`.

    ⚠ `v3_freio` e `v3_degrau` NÃO são escritos aqui: são do `comando.cadeia` (§6.4).
    ⚠ `atraso_max` (DR, §6.8; PD-28) vai aos buffers só quando passa do teto inicial: o startup do env já
    põe o teto inicial (`env_cfg`), e um checkpoint retomado com o teto alto o põe de novo aqui.
    Os demais campos do `dr()` ficam lá para os consumidores.
    """
    garante_v3(env)
    cur = env.v3_cur
    it = env.common_step_counter // _K.balanceador.passos_por_iteracao
    if env.v3_iter >= 0 and it != env.v3_iter:
        cur.fecha_iteracao()
    env.v3_iter = it
    teto = cur.dr()["atraso_max"]
    if teto != _K.dr.atraso_inicio[1]:
        eventos.atraso(env, env_ids, teto, _K.contrato.decimacao)
    frac = cur.fracoes_fatias()
    log = {"frac_ANDAR": frac[0].item(), "frac_B": frac[1].item(), "frac_C": frac[2].item()}
    for c in cur.celulas:
        log[f"s_{c.tarefa}_{c.tipo}"] = c.s
    for t in balanceador.TAREFAS:
        log[f"nivel_{t}"] = cur.nivel(t)
        log[f"degrau_{t}"] = cur.degrau(t)
    return log
