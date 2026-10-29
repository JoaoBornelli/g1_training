"""Eventos da v3: reset por fatia, terreno por tarefa, DR de startup e empurrão (D7 do plano, §6.5, §6.8).

⚠ Os termos de DR (`dr.geom_friction`, `dr.body_mass`, `dr.encoder_bias`) e `mdp.reset_joints_by_offset`
entram direto no cfg (D10), sem invólucro: o fabricante já aceita `env_ids`. Aqui fica só o que é nosso.

⚠ Ordem do reset (o mjlab roda CURRÍCULO → EVENTOS → COMANDO): o `v3_cur` e os buffers `v3_*` já existem
quando `sorteia_fatia` roda. Quem os cria antes do primeiro reset é o `CadeiaCmd.__init__` (`comando.py:195`), que
chama `garante_v3` na construção do env (§6.4); o termo `v3` do currículo roda depois dos eventos e não cria nada
aqui. Por isso este módulo lê `env.v3_*` e não importa o `curriculo`.

⚠ UM evento por entidade (padrão do g1_limpo/eventos.py): `base` é o único reset que escreve a pose da base.
"""
from __future__ import annotations

import torch

from mjlab.envs.mdp.events import push_by_setting_velocity, reset_root_state_uniform

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


def reset_base_por_fatia(env, env_ids, faixa_andar: dict, faixa_manipula: dict):
    """Reset da base: ANDAR com a faixa de locomoção; B e C com a faixa de manipulação (D7, `base`).

    É o `reset_base_por_elo` do g1_limpo com a divisão trocada: a chave é a fatia, não o elo.
    ⚠ Não reimplementa a amostragem: chama `reset_root_state_uniform` uma vez por subconjunto, para que o
    `env_origins` e a orientação default entrem como no fabricante.
    ⚠ O z do ANDAR é `pelve_de_pe + 0,01` sem somar nada aqui: o `init_state` da cena já é esse
    (cena.py:219), e o z da faixa é (0, 0).
    """
    if len(env_ids) == 0:
        return
    anda = env.v3_fatia[env_ids] == ANDAR
    for mascara, faixa in ((anda, faixa_andar), (~anda, faixa_manipula)):
        ids = env_ids[mascara]
        if len(ids):
            reset_root_state_uniform(env, ids, pose_range=dict(faixa), velocity_range={})


def reinicia_filtros(env, env_ids):
    """Reset, depois da base: estado de estimação do episódio (D7, `filtros`).

    ⚠ Roda DEPOIS de `base`: a pose que a odometria recebe é a que a base acabou de ganhar. Pelo mesmo
    motivo, o `v3_odo` começa com a pose verdadeira (PD-18).
    ⚠ Lê o `qpos`, não o `xpos`: no reset o `xpos` ainda é o do fim do episódio anterior (o `forward` vem depois).
    """
    robo = env.scene["robot"]
    env.v3_filtro.reinicia(env_ids)
    q = env.sim.data.qpos[env_ids][:, robo.indexing.free_joint_q_adr]
    env.v3_odo.reinicia(env_ids, q[:, :3], q[:, 3:7])


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
