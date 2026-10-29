"""Eventos da v3: reset por fatia, terreno por tarefa, DR de startup e empurrão (D7 do plano, §6.5, §6.8), e a cena
da caixa da fase E (`posiciona_cena`, `carga`, atritos, `vies_imu`; o núcleo `sorteia_topo_e_massa`, §6.3), e a
laje de destino do BOTAR (`posiciona_destino`, F2; PD-47).

⚠ Os termos de DR (`dr.geom_friction`, `dr.body_mass`, `dr.encoder_bias`) e `mdp.reset_joints_by_offset`
entram direto no cfg (D10), sem invólucro: o fabricante já aceita `env_ids`. Aqui fica só o que é nosso.

⚠ Ordem do reset (o mjlab roda CURRÍCULO → EVENTOS → COMANDO): o `v3_cur` e os buffers `v3_*` já existem
quando `sorteia_fatia` roda. Quem os cria é o `CadeiaCmd.__init__` (`comando.py:195`), que chama `garante_v3` na
construção do env (§6.4), antes do primeiro reset e dos eventos de startup. Por isso este módulo lê `env.v3_*`
e não importa o `curriculo`.

⚠ UM evento por entidade (padrão do g1_limpo/eventos.py): `base` é o único reset que escreve a pose da base.
"""
from __future__ import annotations

import math

import torch

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import push_by_setting_velocity, reset_root_state_uniform
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz, quat_mul

from g1_v3 import alvos, estimador
from g1_v3.knobs import ATIVO, Cena, Niveis

# índices de FATIAS (knobs.py): ANDAR, B, C
ANDAR, B, C = 0, 1, 2


def _fracao_terreno(cur, tarefa: str) -> float:
    # fração da célula `terreno` no sorteio da tarefa; 0 enquanto a célula está fechada
    ids, f = cur.fracoes_celulas(tarefa)
    i = cur.id_celula(tarefa, "terreno")
    return float(f[ids.index(i)]) if i in ids else 0.0


def escolhe_celula_terreno(fatia, terreno_ep, nivel_terreno, k_ter, gerador):
    """Núcleo puro (PD-20, S §11.2): (coluna (E,), linha (E,)) de cada episódio, sobre o gerador do terreno.

    ANDAR de terreno: coluna uniforme em 1..n−1 (a 0 é o plano), linha uniforme em 0..nível.
    B de terreno: coluna uniforme entre as que têm plataforma (`linha_max_carregar`), linha uniforme em
    0..min(nível, teto da coluna). ⚠ O `irregular` (coluna 6) não tem plataforma: a B nunca o sorteia (S §11.2).
    Sem terreno ou fatia C: (0, 0).
    """
    dev = fatia.device
    n_col = len(k_ter.colunas)
    lim = dict(k_ter.linha_max_carregar)
    cols_b = torch.tensor([k_ter.colunas.index(nome) for nome in lim], device=dev)
    teto_b = torch.tensor(list(lim.values()), device=dev)
    # quatro uniformes por episódio: coluna e linha do ANDAR, coluna e linha da B
    u = torch.rand((4, fatia.shape[0]), generator=gerador, dtype=torch.float64, device=gerador.device).to(dev)
    col_a = 1 + (u[0] * (n_col - 1)).long()
    lin_a = (u[1] * (nivel_terreno + 1)).long()
    j = (u[2] * len(lim)).long()
    col_b = cols_b[j]
    lin_b = (u[3] * (torch.minimum(nivel_terreno, teto_b[j]) + 1)).long()
    andar = fatia == ANDAR
    ter = terreno_ep & (andar | (fatia == B))
    coluna = torch.where(andar, col_a, col_b)
    linha = torch.where(andar, lin_a, lin_b)
    return torch.where(ter, coluna, 0), torch.where(ter, linha, 0)


def _uniforme(gerador, forma, dev):
    # U[0; 1) no device do gerador, depois no device do env (o mesmo padrão de `escolhe_celula_terreno`)
    return torch.rand(forma, generator=gerador, dtype=torch.float64, device=gerador.device).to(dev)


def sorteia_topo_e_massa(fatia, nivel_pegar: int, nivel_massa_c: int, origem_z, k_niv: Niveis, k_cena: Cena,
                         gerador):
    """Núcleo puro (E1, §6.3 "Fase E"; S §7, §11.2): topo e massa da caixa de cada episódio.

    Devolve (topo, topo_w, massa): topo sobre o chão LOCAL, topo_w = origem_z + topo (a altura absoluta, §6.4)
    e a massa em kg. (h_k, m_k) = `Niveis.pegar[nivel_pegar]`; h_0 é o topo do nível 0, o teto das fatias B e C.
    - B: topo U[h_k; h_0], massa U[m_min; m_k];
    - C: topo U[max(h_k; topo_min_c); h_0], massa U[m_min; m_j], com j = `nivel_massa_c` (−1 → m_min);
    - ANDAR: topo 0 (a cena estaciona, §6.5) e massa m_min.

    ⚠ O piso da C (`topo_min_c`) é o que mantém a fatia C fora do topo baixo: com h_k < 0,30 o sorteio cola no piso.
    ⚠ `nivel_massa_c` existe porque a C mede a massa pelo maior PEGAR que já passou no portão (S §11.2), e não
    pelo nível em curso.
    """
    dev = fatia.device
    t = lambda x: torch.tensor(float(x), dtype=torch.float64, device=dev)
    h_k, m_k = k_niv.pegar[nivel_pegar]
    h_0 = k_niv.pegar[0][0]
    m_min = k_cena.caixa_massa_base
    m_j = k_niv.pegar[nivel_massa_c][1] if nivel_massa_c >= 0 else m_min
    andar, c = fatia == ANDAR, fatia == C
    u = _uniforme(gerador, (2, fatia.shape[0]), dev)
    lo = torch.where(c, t(max(h_k, k_niv.topo_min_c)), t(h_k))
    topo = torch.where(andar, t(0.0), lo + (t(h_0) - lo) * u[0])
    teto_m = torch.where(fatia == B, t(m_k), torch.where(c, t(m_j), t(m_min)))
    massa = t(m_min) + (teto_m - t(m_min)) * u[1]
    return topo, origem_z.to(dev, torch.float64) + topo, massa


def sorteia_fatia(env, env_ids, k_ter):
    """Reset, o primeiro: sorteia a fatia e o terreno do episódio (§6.4 `v3_fatia`; D7).

    ⚠ Todo sorteio usa o `v3_gerador` (§6.4), no device dele. Não escreve `v3_celula`: o `twist` é o único escritor.
    ⚠ `v3_t_empurrao` volta a −∞ aqui: sem isto, o instante do empurrão do episódio anterior ancoraria o
    canal zero do novo (PD-15). O mesmo reset é o começo do episódio.
    """
    dev = env.device
    cur = env.v3_cur
    g = env.v3_gerador
    fatia = cur.sorteia_fatias(len(env_ids), g).to(dev)
    env.v3_fatia[env_ids] = fatia
    env.v3_t_empurrao[env_ids] = float("-inf")
    # episódio de terreno: a fração da célula `terreno` da tarefa da fatia (ANDAR; CARREGAR na B)
    p = torch.where(fatia == ANDAR, torch.tensor(_fracao_terreno(cur, "ANDAR"), dtype=torch.float64, device=dev),
                    torch.tensor(_fracao_terreno(cur, "CARREGAR"), dtype=torch.float64, device=dev))
    terreno_ep = torch.rand(len(env_ids), generator=g, dtype=torch.float64, device=g.device).to(dev) < p
    nivel = torch.where(fatia == ANDAR, torch.tensor(cur.nivel_terreno("ANDAR"), device=dev),
                        torch.tensor(cur.nivel_terreno("CARREGAR"), device=dev))
    coluna, linha = escolhe_celula_terreno(fatia, terreno_ep, nivel, k_ter, g)
    # ⚠ escrita no terreno como o `update_env_origins` do mjlab (terrain_entity.py:216-233): nível e tipo,
    # e a origem do sub-terreno escolhido (`scene.env_origins` é o mesmo tensor do terreno)
    terreno = env.scene["terrain"]
    linha, coluna = linha.to(dev), coluna.to(dev)
    terreno.terrain_levels[env_ids] = linha
    terreno.terrain_types[env_ids] = coluna
    env.scene.env_origins[env_ids] = terreno.terrain_origins[linha, coluna]


def _base_da_pega(env, ids, faixa: dict):
    """Base das fatias B e C na partida do PEGAR (E11 S4, 29/10; E §4.3: o controlador já leva o robô à pega).

    A base fica no eixo x da caixa, a `ref.recuo(topo, a)` ± `Alvos.faixa_recuo` (sorteio no `v3_gerador`), e o
    rumo é a faixa `yaw` (de frente para a caixa ± o jitter). z pela faixa `z`. O x e o y da faixa não entram mais.
    ⚠ Reimplementa só a escrita de `reset_root_state_uniform` (mjlab events.py:156-175), porque a distância depende
    do topo e da caixa de cada episódio: as mesmas contas (origem do env, pose default, quat_mul com o yaw).
    """
    if len(ids) == 0:
        return
    dev, g, robo = env.device, env.v3_gerador, env.scene["robot"]
    n = len(ids)
    u = _uniforme(g, (3, n), dev)
    recuo = env.v3_ref.recuo(env.v3_topo[ids], env.v3_meia[ids]).to(dev, torch.float64)
    d = recuo + (2.0 * u[0] - 1.0) * ATIVO.alvos.faixa_recuo
    lo_yaw, hi_yaw = faixa["yaw"]
    lo_z, hi_z = faixa["z"]
    yaw = lo_yaw + (hi_yaw - lo_yaw) * u[1]
    dz = lo_z + (hi_z - lo_z) * u[2]
    origem = env.scene.env_origins[ids].to(torch.float64)
    xy = origem[:, :2] + env.v3_caixa_xy[ids] - torch.stack([d, torch.zeros_like(d)], -1)
    root = robo.data.default_root_state[ids].clone()
    z = origem[:, 2] + root[:, 2].to(torch.float64) + dz
    quat = quat_mul(root[:, 3:7], quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw).to(root.dtype))
    pose = torch.cat([torch.cat([xy, z[:, None]], -1).to(root.dtype), quat], -1)
    robo.write_root_link_pose_to_sim(pose, env_ids=ids)
    robo.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=ids)


def reset_base_por_fatia(env, env_ids, faixa_andar: dict, faixa_manipula: dict):
    """Reset da base: ANDAR com a faixa de locomoção; B e C na partida do PEGAR (`_base_da_pega`, E11 S4).

    É o `reset_base_por_elo` do g1_limpo com a divisão trocada: a chave é a fatia, não o elo.
    ⚠ Roda DEPOIS de `cena` (a caixa de B e C já tem xy e topo). O ANDAR chama `reset_root_state_uniform`.
    ⚠ O z do ANDAR é `pelve_de_pe + 0,01` sem somar nada aqui: o `init_state` da cena já é esse
    (cena.py:219), e o z da faixa é (0, 0).
    """
    if len(env_ids) == 0:
        return
    anda = env.v3_fatia[env_ids] == ANDAR
    ids = env_ids[anda]
    if len(ids):
        reset_root_state_uniform(env, ids, pose_range=dict(faixa_andar), velocity_range={})
    _base_da_pega(env, env_ids[~anda], dict(faixa_manipula))


def reinicia_filtros(env, env_ids):
    """Reset, depois da base: estado de estimação do episódio (D7, `filtros`).

    Zera o filtro de elevação, a odometria e o `v3_estimador` (o sorteio do viés de meia aresta e de orientação
    sai de novo por episódio; sem isso a primeira leitura sairia com o estado do anterior).
    ⚠ Roda DEPOIS de `base`: a pose que a odometria recebe é a que a base acabou de ganhar. Pelo mesmo
    motivo, o `v3_odo` começa com a pose verdadeira (PD-18).
    ⚠ Lê o `qpos`, não o `xpos`: no reset o `xpos` ainda é o do fim do episódio anterior (o `forward` vem depois).
    """
    robo = env.scene["robot"]
    env.v3_filtro.reinicia(env_ids)
    q = env.sim.data.qpos[env_ids][:, robo.indexing.free_joint_q_adr]
    env.v3_odo.reinicia(env_ids, q[:, :3], q[:, 3:7])
    env.v3_estimador.reinicia(env_ids)


def atraso(env, env_ids, passos_max: int, decimacao: int):
    """Startup: põe o teto corrente do atraso nos buffers, com a capacidade do build (D7; PD-28).

    `passos_max` = `DR.atraso_inicio[1]` (passos da política) para a observação; o atuador conta em passos de
    física, então recebe `passos_max * decimacao`. A capacidade (12 e 3, D1 e D2) fica; só o teto sobe.
    ⚠ `_delay_buffer` e `_group_obs_term_delay_buffer` são privados do mjlab 1.5.3. O PD-25 evita a API privada
    no sorteio, mas o teto por nível só se muda pelo objeto do buffer.
    """
    robo = env.scene["robot"]
    buffers_do_atuador = {id(a._delay_buffer): a._delay_buffer for a in robo.actuators if a._delay_buffer is not None}
    for buf in buffers_do_atuador.values():  # grupos fundidos compartilham o buffer: um por id
        buf.max_lag = passos_max * decimacao
    for grupo in env.observation_manager._group_obs_term_delay_buffer.values():
        for buf in grupo.values():
            buf.max_lag = passos_max


def empurrao(env, env_ids, velocity_range: dict):
    """Intervalo: o empurrão do fabricante, mais o instante gravado em `v3_t_empurrao` (D7; PD-15).

    ⚠ O instante é em segundos desde o começo do episódio (`episode_length_buf * step_dt`). O `comando.twist`
    tem de ler a mesma unidade em `tempo_da_ancora`.
    """
    push_by_setting_velocity(env, env_ids, velocity_range=velocity_range)
    env.v3_t_empurrao[env_ids] = env.episode_length_buf[env_ids].to(torch.float64) * env.step_dt


# ------------------------------------------------------------------ fase E: a cena da caixa (E1; §6.5)
# ⚠ Toda escrita de z soma `env_origins[:, 2]`, e `v3_topo` fica sobre o chão local (§6.4; viab. D3).


def _laje(env, ids, topo_w, xy=None, yaw=None):
    # a laje é MOCAP: a pose é o CENTRO do corpo, `prateleira_meia_z` abaixo do topo pedido (g1_limpo.posiciona_cena)
    # `xy` e `yaw` (mundo) são só do destino do BOTAR (F2); sem eles, a laje fica na posição e no rumo de sempre
    k = ATIVO.cena
    n, dev = len(ids), env.device
    origem = env.scene.env_origins[ids]
    pose = torch.zeros(n, 7, device=dev)
    if xy is None:
        pose[:, 0] = origem[:, 0] + k.prateleira_xy[0]
        pose[:, 1] = origem[:, 1] + k.prateleira_xy[1]
    else:
        pose[:, :2] = xy
    pose[:, 2] = topo_w - k.prateleira_meia_z
    if yaw is None:
        pose[:, 3] = 1.0  # quat (w, x, y, z): identidade
    else:
        pose[:, 3] = torch.cos(yaw / 2)
        pose[:, 6] = torch.sin(yaw / 2)  # rotação só em z
    env.scene["table"].write_mocap_pose_to_sim(pose, env_ids=ids)


def _caixa(env, ids, topo_w, dx, dy):
    # a caixa repousa no topo: o centro fica a meia aresta de `v3_meia` (o tamanho sorteado no startup) acima
    k = ATIVO.cena
    n, dev = len(ids), env.device
    origem = env.scene.env_origins[ids]
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = origem[:, 0] + k.caixa_xy[0] + dx
    pose[:, 1] = origem[:, 1] + k.caixa_xy[1] + dy
    pose[:, 2] = topo_w + env.v3_meia[ids]
    pose[:, 3] = 1.0
    caixa = env.scene["box"]
    caixa.write_root_link_pose_to_sim(pose, env_ids=ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)


def estaciona_cena(env, ids):
    """Fatia ANDAR (§6.5): laje e caixa `estaciona_z` acima da origem, fora do alcance do lidar e do terreno.

    É o `afasta_cena` do g1_limpo (sem chamador lá), com o z somando a origem. A caixa fica apoiada na laje.
    """
    if len(ids) == 0:
        return
    topo_w = env.scene.env_origins[ids, 2] + ATIVO.cena.estaciona_z
    _laje(env, ids, topo_w)
    _caixa(env, ids, topo_w, 0.0, 0.0)


def laje_fora(env, env_ids):
    """Chamada pelo `comando` na abertura do CARREGAR que anda (C39): tira a laje do caminho, só a laje.

    A conta é a do `estaciona_cena`. A caixa NÃO se move aqui.
    """
    if len(env_ids) == 0:
        return
    _laje(env, env_ids, env.scene.env_origins[env_ids, 2] + ATIVO.cena.estaciona_z)


def posiciona_cena(env, env_ids):
    """Reset `cena`: sorteia topo e massa pelo núcleo e põe a laje e a caixa. A fatia ANDAR vai a `estaciona_cena`.

    Publica `v3_topo` (sobre o chão local) e `v3_massa`; o `carga` lê a massa daqui, por isso roda depois.
    E11 S4: publica também o xy da caixa de B e C (`v3_caixa_xy`), que a `base` lê, por isso roda antes dela.
    ⚠ O nível vem do `v3_cur` pelo nome do contrato: `nivel("PEGAR")` e `nivel_passou("PEGAR")` (E6).
    ⚠ O jitter em x é de UM LADO (0..jx_max), como no g1_limpo, e aperta com o nível (`jitter_x_max[nível]`).
    """
    cur, g, dev = env.v3_cur, env.v3_gerador, env.device
    fatia = env.v3_fatia[env_ids]
    topo, topo_w, massa = sorteia_topo_e_massa(fatia, cur.nivel("PEGAR"), cur.nivel_passou("PEGAR"),
                                               env.scene.env_origins[env_ids, 2], ATIVO.niveis, ATIVO.cena, g)
    env.v3_topo[env_ids] = topo.to(env.v3_topo.dtype)
    env.v3_massa[env_ids] = massa.to(env.v3_massa.dtype)
    andar = fatia == ANDAR
    estaciona_cena(env, env_ids[andar])
    ids, tw = env_ids[~andar], topo_w[~andar]
    if len(ids) == 0:
        return
    k = ATIVO.cena
    u = _uniforme(g, (2, len(ids)), dev)
    dx = k.jitter_x_max[cur.nivel("PEGAR")] * u[0]
    dy = k.caixa_jitter_y[0] + (k.caixa_jitter_y[1] - k.caixa_jitter_y[0]) * u[1]
    env.v3_caixa_xy[ids] = torch.stack([k.caixa_xy[0] + dx, k.caixa_xy[1] + dy], -1)  # a `base` põe o robô em relação a ela
    _laje(env, ids, tw)
    _caixa(env, ids, tw, dx, dy)


# ------------------------------------------------------------------ fase F: a laje de destino do BOTAR (F2; PD-47)


def sorteia_destino(nivel_botar: int, n: int, k_niv: Niveis, k_cena: Cena, gerador):
    """Núcleo puro (F2, §6.3 "Fase F"; PD-47, [C38]): a laje de destino do BOTAR de cada episódio.

    Devolve (theta, y, longe, topo): θ U[±radians(destino_giro_max_graus)] em rad, y U[±destino_jitter_y] (m),
    longe U[0; jitter_x_max[nível]] (m, só para longe, [C38]) e topo U[botar[nível]; botar[0]] sobre o chão local.
    ⚠ Quatro uniformes por episódio, sobre o gerador (o sorteio sai no device dele).
    """
    u = _uniforme(gerador, (4, n), gerador.device)
    giro = math.radians(k_cena.destino_giro_max_graus)
    theta = (2.0 * u[0] - 1.0) * giro
    y = (2.0 * u[1] - 1.0) * k_cena.destino_jitter_y
    longe = u[2] * k_cena.jitter_x_max[nivel_botar]
    topo = k_niv.botar[nivel_botar] + (k_niv.botar[0] - k_niv.botar[nivel_botar]) * u[3]
    return theta, y, longe, topo


def posiciona_destino(env, ids):
    """Abertura do BOTAR (chamado pela cadeia, F1; não é evento do cfg): a laje de destino, PD-47.

    Nível do BOTAR pelo `v3_cur`. `alvos.laje_destino` dá o alvo e o centro no frame da base, com o recuo
    `ref.recuo(topo, a)` + longe. Leva ao mundo pelo rumo da base; a laje (mocap) gira para rumo + θ, com o topo
    em `env_origins` z + topo. Grava `v3_topo_destino` e devolve o ponto de soltura no mundo (E,3), sobre o topo
    da laje (a F1 soma (0, 0, a)).
    """
    cur, g, dev, k = env.v3_cur, env.v3_gerador, env.device, ATIVO.cena
    f64 = torch.float64
    theta, y, longe, topo = sorteia_destino(cur.nivel("BOTAR"), len(ids), ATIVO.niveis, k, g)
    theta, y, longe, topo = (x.to(dev, f64) for x in (theta, y, longe, topo))
    recuo = env.v3_ref.recuo(topo.to(env.v3_topo.dtype), env.v3_meia[ids]).to(dev, f64) + longe
    alvo_b, centro_b = alvos.laje_destino(recuo, theta, y, k.prateleira_meia_xy, k.destino_borda)
    robo = env.scene["robot"].data
    base, rumo = robo.root_link_pos_w[ids, :2].to(f64), estimador.yaw(robo.root_link_quat_w[ids]).to(f64)
    topo_w = env.scene.env_origins[ids, 2].to(f64) + topo
    _laje(env, ids, topo_w, xy=alvos.base_para_mundo(base, rumo, centro_b), yaw=rumo + theta)
    env.v3_topo_destino[ids] = topo.to(env.v3_topo_destino.dtype)
    alvo_w = alvos.base_para_mundo(base, rumo, alvo_b)
    return torch.cat([alvo_w, topo_w[:, None]], -1)


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def tamanho_caixa(env, env_ids, *, faixa, n_variantes, asset_cfg):
    """Startup `tamanho_caixa`: DR de tamanho da caixa, K meias arestas discretas por mundo (S §12).

    Cópia do `g1_limpo.tamanho_caixa`, com três diferenças: a meia aresta vai para `v3_meia` (E,), porque a caixa é
    cubo; o sorteio sai do `v3_gerador` (§6.4), e não do gerador global; e o `body_mass` fica para o `carga`.
    """
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev, dtype=torch.int)
    else:
        env_ids = env_ids.to(dev, dtype=torch.int)
    n = len(env_ids)
    valores = torch.linspace(float(faixa[0]), float(faixa[1]), int(n_variantes), device=dev)
    idx = (_uniforme(env.v3_gerador, (n,), dev) * int(n_variantes)).long().clamp(max=int(n_variantes) - 1)
    a = valores[idx]

    caixa = env.scene[asset_cfg.name]
    gid = caixa.indexing.geom_ids[asset_cfg.geom_ids]
    env_grid, geom_grid = torch.meshgrid(env_ids, gid, indexing="ij")
    tam = a.unsqueeze(-1).unsqueeze(-1).expand(n, len(gid), 3)
    env.sim.model.geom_size[env_grid, geom_grid] = tam
    env.sim.model.geom_rbound[env_grid, geom_grid] = (a * math.sqrt(3.0)).unsqueeze(-1)
    env.sim.model.geom_aabb[env_grid, geom_grid, 1] = tam
    env.v3_meia[env_ids.long()] = a


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def carga(env, env_ids):
    """Reset `carga`: escreve a massa de `v3_massa` e a inércia dela como MASSA E INÉRCIA de verdade (g1_limpo).

    ⚠ A massa NÃO é sorteada aqui: quem a sorteia é o `sorteia_topo_e_massa`, no `cena`. Sortear de novo daria
    uma massa publicada diferente da simulada.
    ⚠ A inércia é a do cubo homogêneo de meia aresta `v3_meia`: `I = (2/3)·m·a²`.
    """
    dev = env.device
    env_ids = env_ids.to(dev, dtype=torch.int)
    n = len(env_ids)
    caixa = env.scene["box"]
    kg = env.v3_massa[env_ids.long()]
    a = env.v3_meia[env_ids.long()]
    bid = caixa.indexing.body_ids
    env_grid, body_grid = torch.meshgrid(env_ids, bid, indexing="ij")
    env.sim.model.body_mass[env_grid, body_grid] = kg.unsqueeze(-1)
    inercia = (2.0 / 3.0) * kg * a * a
    env.sim.model.body_inertia[env_grid, body_grid] = inercia.unsqueeze(-1).unsqueeze(-1).expand(n, len(bid), 3)


def atrito_pad(env, env_ids, *, asset_cfg):
    """Reset `atrito_pad`: μ dos pads de palma, coluna 0, por `dr.geom_friction` do mjlab (PD-46; §6.8).

    ⚠ `asset_cfg` chega RESOLVIDO pelos `params` do cfg (`env_cfg.py`); sem isso o `dr.geom_friction` mexe em todas
    as geoms do robô (S1). É o padrão de `tamanho_caixa`.
    Ligado por `v3_cur.dr()["atrito_pad"]` (portão do PEGAR no nível 0). A borda de baixo desce com o nível
    (`mu_min_pad`). Antes do portão, μ = `Cena.pad_atrito[0]` (R5).
    """
    d = env.v3_cur.dr()
    mu = ATIVO.cena.pad_atrito[0]
    faixa = (d["mu_min_pad"], ATIVO.dr.atrito_pad[1]) if d.get("atrito_pad") else (mu, mu)
    dr.geom_friction(env, env_ids, ranges=faixa, asset_cfg=asset_cfg)


def atrito_laje(env, env_ids, *, asset_cfg):
    """Reset `atrito_laje`: μ da laje, coluna 0, por `dr.geom_friction` do mjlab (PD-46; §6.8).

    ⚠ `asset_cfg` chega resolvido pelos `params` do cfg, pelo mesmo motivo de `atrito_pad` (S1).
    Ligado por `v3_cur.dr()["atrito_laje"]` (portão do PEGAR no nível 0). Antes do portão, μ =
    `Cena.prateleira_atrito[0]` (R5).
    ⚠ A laje tem `priority` 3 (`Cena.prioridade_laje`): o contato com a caixa usa o μ DELA.
    """
    d = env.v3_cur.dr()
    mu = ATIVO.cena.prateleira_atrito[0]
    faixa = ATIVO.dr.atrito_laje if d.get("atrito_laje") else (mu, mu)
    dr.geom_friction(env, env_ids, ranges=faixa, asset_cfg=asset_cfg)


def vies_imu(env, env_ids):
    """Reset `vies_imu`: viés fixo do giro da base e inclinação fixa do IMU, por episódio (§6.8, ANDAR no portão).

    Ligado por `v3_cur.dr()["vies_imu"]`. Desligado, o giro é zero e a inclinação é a identidade (o neutro dos
    termos `giro_base` e `gravidade`, E8).
    """
    n, dev, g = len(env_ids), env.device, env.v3_gerador
    if env.v3_cur.dr().get("vies_imu", False):
        u = _uniforme(g, (2, n, 3), dev)
        giro = (2.0 * u[0] - 1.0) * ATIVO.dr.vies_imu_giro
        ang = ((2.0 * u[1] - 1.0) * math.radians(ATIVO.dr.vies_imu_incl_graus)).to(torch.float32)
        ang[:, 2] = 0.0  # ⚠ sem giro em z: a inclinação do IMU é em x e y (DR §3.4); o z gira o mapa (PD-42; M5)
        incl = matrix_from_quat(quat_from_euler_xyz(ang[:, 0], ang[:, 1], ang[:, 2]))
    else:
        giro = torch.zeros(n, 3, device=dev)
        incl = torch.eye(3, device=dev).repeat(n, 1, 1)
    env.v3_vies_giro[env_ids] = giro.to(env.v3_vies_giro.dtype)
    env.v3_incl_imu[env_ids] = incl.to(env.v3_incl_imu.dtype)
