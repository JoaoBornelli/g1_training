"""As métricas de marcha, como `MetricsTermCfg` — fora dos termos de recompensa.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

POR QUE ELAS SAEM DA RECOMPENSA. O fabricante escreve cinco `Metrics/*` de DENTRO
de termos de recompensa (`mjlab/tasks/velocity/mdp/rewards.py:205,229,317,352,373`).
E `reward_manager.py:122` PULA termo com peso 0:

    if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue

Portanto **desligar uma penalidade apaga a medição dela, em silêncio.** Isso não é
hipótese: o `air_time` do molde já vem com peso 0,0, e por isso o
`Metrics/air_time_mean` do fabricante NÃO EXISTE no painel de quem roda a receita
dele. A métrica que responderia "o robô levanta o pé?" está desligada junto com o
incentivo, e é exatamente a pergunta da F1.

Medir num manager separado desacopla as duas coisas: o peso decide o INCENTIVO, o
`metrics` decide a MEDIÇÃO, e mexer num não apaga o outro.

DIVERGÊNCIA DECLARADA contra o fabricante, e ela é uma melhoria. As médias dele são
escalares de LOTE (`torch.sum(...) / num_in_air` sobre o batch inteiro), portanto
não têm eixo de env e não podem entrar num manager por env. Aqui cada métrica é
POR ENV, e o `MetricsManager` faz a média de envs no fim do episódio
(`metrics_manager.py:125`). O número final é comparável; a rota é melhor.

⚠ E o `MetricsManager` divide por `step_count`, não por `max_episode_length_s`.
Métrica em [0,1] fica em [0,1] no log — **não** existe a diluição por duração que o
`Episode_Reward/*` tem, e que o `leitura.py` desfaz.
"""
from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

__all__ = [
    "termos",
    "momento_angular",
    "tempo_de_voo",
    "pico_de_altura",
    "velocidade_de_escorrego",
    "forca_de_pouso",
    "pads_em_contato",
    "fracao_esperando",
    "impacto_da_caixa",
    "aproxima_caixa",
    "renda_manipulacao",
    "PES_NO_CHAO",
    "ALTURA_DO_PE",
    "MOMENTO_ANGULAR",
    "SITIOS_DOS_PES",
]

# os nomes de sensor do MOLDE. Transcritos dos `params` dos termos do fabricante, e o
# `smoke` confere que eles existem na cena — um nome errado aqui só apareceria como
# métrica ausente no painel, sem erro.
PES_NO_CHAO = "feet_ground_contact"
ALTURA_DO_PE = "foot_height_scan"
MOMENTO_ANGULAR = "robot/root_angmom"
SITIOS_DOS_PES = ("left_foot", "right_foot")


def termos(sensores_palma: tuple[str, ...] = ("palma_E", "palma_D"),
           sensores_dorso: tuple[str, ...] = ("dorso_E", "dorso_D"),
           sensor_apoio: str = "apoio_caixa",
           nome_do_comando: str = "alvo_caixa",
           termos_congelaveis: tuple[str, ...] = (),
           ) -> dict[str, MetricsTermCfg]:
    """Os termos, montados aqui e em nenhum outro lugar.

    ⚠ O `SceneEntityCfg` TEM DE VIVER EM `params`, e este é o bug que custou a
    primeira execução: `manager_base.py:141-145` só resolve `SceneEntityCfg` que
    esteja em `term_cfg.params`. Como argumento default de função ele NUNCA é
    resolvido — `site_ids` fica `slice(None)`, e a métrica lê os SEIS sítios do robô
    em vez dos dois pés. O erro só apareceu por sorte de forma (6 contra 2); com um
    robô de dois sítios ele passaria e mediria a coisa errada em silêncio.
    """
    pes = {"sensor_name": PES_NO_CHAO}
    return {
        "momento_angular": MetricsTermCfg(
            func=momento_angular, params={"sensor_name": MOMENTO_ANGULAR}),
        "tempo_de_voo": MetricsTermCfg(func=tempo_de_voo, params=dict(pes)),
        "pico_de_altura": MetricsTermCfg(
            func=pico_de_altura,
            params={**pes, "height_sensor_name": ALTURA_DO_PE}),
        "velocidade_de_escorrego": MetricsTermCfg(
            func=velocidade_de_escorrego,
            params={**pes, "asset_cfg": SceneEntityCfg(
                "robot", site_names=SITIOS_DOS_PES)}),
        "forca_de_pouso": MetricsTermCfg(func=forca_de_pouso, params=dict(pes)),
        # ⚠ AS DUAS MEDIÇÕES QUE FALTAVAM PARA LER A PEGA. Nenhuma tem peso: elas
        # existem para que um estado e outro deixem de ler igual no painel.
        #
        #   `palmas_em_contato`  0 / 0,5 / 1,0 — nenhuma, uma, ou as DUAS palmas na
        #       caixa. Sem ela, "uma mão encostada" e "nenhuma mão" davam o mesmo
        #       `squeeze` de zero exato, porque o `squeeze` é `min` das duas forças. Eu
        #       chamei isso de abandono da tarefa uma vez, e estava errado: era uma mão
        #       na caixa e a outra fora.
        #
        #   `dorso_em_contato`  o pad de DORSO tocando a caixa. Ele não deve tocar
        #       nunca. O freio é geométrico (o alcance bimanual põe as palmas viradas
        #       uma para a outra), e esta métrica é o que confere se o freio funciona.
        #       Se ela sair de zero, o freio geométrico não bastou.
        "palmas_em_contato": MetricsTermCfg(
            func=pads_em_contato, params={"sensores": sensores_palma}),
        "dorso_em_contato": MetricsTermCfg(
            func=pads_em_contato, params={"sensores": sensores_dorso}),
        # ⚠ A FRAÇÃO DE PASSOS COM A JANELA DE ESPERA CORRENDO. Sem ela, "o robô não
        # espera" e "a janela não existe no módulo" leem IGUAL no painel — e essa
        # confusão custou uma conversa inteira. O `MetricsManager` divide por
        # `step_count`, portanto isto sai direto como fração.
        "fracao_esperando": MetricsTermCfg(func=fracao_esperando),
        # ⚠ O PICO DE IMPACTO DA CAIXA NA LAJE (03/09). Decisão do dono: soltar de 5 cm
        # é permitido, jogar de mais alto não — e sem esta métrica os dois leem igual.
        "impacto_da_caixa": MetricsTermCfg(
            func=impacto_da_caixa, params={"sensor_apoio": sensor_apoio},
            reduce="max"),
        # ⚠ A RÉGUA DA CAIXA (v2.1, spec P10): sem ela não existia medição de quanto a
        # caixa se aproximou do alvo — só os termos de recompensa, que SATURAM
        # (`exp(-d²/σ²)`) e não distinguem "não saiu do lugar" de "oscila perto do alvo".
        "aproxima_caixa": MetricsTermCfg(
            func=aproxima_caixa, params={"nome_do_comando": nome_do_comando},
            reduce="last"),
        # ⚠ A RÉGUA DA REGRA 1 (v2.1, spec P10, proposta §0): "em todo instante existe
        # incentivo até o alvo do elo; do fecho de um elo até o seguinte a renda não
        # cai." Soma TUDO que depende de elo, ao vivo mais congelado — é o número que
        # o smoke e o TensorBoard leem para provar que nenhuma transição tem degrau.
        "renda_manipulacao": MetricsTermCfg(
            func=renda_manipulacao,
            params={"termos": termos_congelaveis + ("renda_congelada",)}),
    }


def _media_por_env(valor: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    """Média de `valor` sobre os pés onde `mascara`, por env. Zero se nenhum pé.

    ⚠ `clamp(min=1)` no denominador, e não `+1e−6`: com nenhum pé mascarado o
    numerador é exatamente 0, portanto o resultado é 0 e não um NaN nem um número
    enorme.
    """
    m = mascara.float()
    return (valor * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


def fracao_esperando(env) -> torch.Tensor:
    """1 enquanto o env publica ANDAR dentro de um episódio de manipulação: a espera inicial ou a final. Por env.

    ⚠ Ela lê `env.limpo_aguardando`, publicado pelo termo de comando a cada passo. Não
    é import de módulo do projeto — é o mesmo contrato por atributo de env que o
    `limpo_massa` e o `limpo_topo` já usam.

    ⚠ No `ANDAR` a janela é ZERO, portanto esta métrica também mede a fatia: com 30% de
    locomoção e janela média de 0,65 s em episódio de ~800 passos, espere algo da ordem
    de 0,7 × 0,04 = 0,03.
    """
    v = getattr(env, "limpo_aguardando", None)
    if v is None:
        return torch.zeros(env.num_envs, device=env.device)
    # ⚠ AS DUAS ESPERAS (spec §6.4): a inicial (`aguardando`) e a final (`soltou`), que
    # são os passos em que um episódio de manipulação publica ANDAR.
    s = getattr(env, "limpo_soltou", None)
    if s is None:
        return v
    return torch.clamp(v + s, max=1.0)


class impacto_da_caixa:
    """PICO de `|F_apoio_z| / m·g` no episódio, por env. Sem peso: é só medição.

    ⚠ ELA EXISTE POR UMA DECISÃO DO DONO (03/09). No `BOTAR`, soltar a caixa de ~5 cm
    fecha o elo e nenhum termo cobra a queda — e isso está **permitido**: "5 cm é
    permitido, se começar a jogar de mais alto vira problema". Sem esta métrica, "apoiou
    com cuidado" e "jogou de 30 cm" leem IGUAL no painel, e o dia em que a política
    escolher jogar passaria sem sinal.
    A leitura: caixa apoiada em repouso dá ~1,0. Uma queda de 5 cm dá um pico de poucas
    unidades. Um valor que sobe ao longo da run é a política aprendendo a jogar.

    ⚠ `reduce="max"` no `MetricsTermCfg` (v2.1). O `self.pico` já é o MÁXIMO corrente a
    cada passo, mas o `MetricsManager` reduzia isso com `reduce="mean"` (o default):
    a média de um pico MONÓTONO ao longo do episódio é um PISO, não o pico — o painel
    publicava um número sistematicamente menor que o impacto real.

    ⚠ E ela TEM `reset`: sem ele o pico de um episódio entra no seguinte
    (`metrics_manager.py:132` só chama `reset` em termo de classe que o tenha).
    """

    def __init__(self, cfg, env):
        self.pico = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env, sensor_apoio: str) -> torch.Tensor:
        from g1_limpo.comando import forca_de_apoio
        razao = forca_de_apoio(env, sensor_apoio) / (env.limpo_massa * 9.81).clamp(min=1e-6)
        self.pico = torch.maximum(self.pico, razao)
        return self.pico

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.pico[env_ids] = 0.0


class aproxima_caixa:
    """MÍNIMO CORRENTE de `d_alvo / sigma_trazer`, por env — a régua da caixa (spec P10).

    ⚠ LEITURA: 1,0 é "a caixa não saiu de onde o elo abriu" (`sigma_trazer` é a
    distância inicial daquele env, spec §4.2b — `comando.AlvoCaixaCmd.__init__`); 0 é
    "a caixa está NO alvo". Sem esta régua, "a caixa não se move" e "a caixa oscila
    perto do alvo" liam igual no painel: os termos de recompensa são `exp(-d²/σ²)`, que
    SATURA — eles não distinguem progresso fino perto do alvo.

    ⚠ SÓ ATUALIZA ONDE `VALIDA > 0,5`: no `ANDAR` não existe alvo de caixa, e computar
    `d` ali mediria contra um alvo que não é da tarefa.

    ⚠ `reduce="last"` no `MetricsTermCfg`: é MÍNIMO CORRENTE, portanto o último valor
    do episódio JÁ é o mínimo acumulado — uma média diluiria o progresso, o mesmo
    defeito que `impacto_da_caixa` e `pico_de_altura` já evitam com `reduce` não-padrão.

    ⚠ E ela TEM `reset`: sem ele o mínimo de um episódio entraria no seguinte
    (`metrics_manager.py:132` só chama `reset` em termo de classe que o tenha).
    """

    def __init__(self, cfg, env):
        self.minimo = torch.ones(env.num_envs, device=env.device)

    def __call__(self, env, nome_do_comando: str) -> torch.Tensor:
        from g1_limpo.comando import ALVO, VALIDA
        t = env.command_manager.get_term(nome_do_comando)
        comando = env.command_manager.get_command(nome_do_comando)
        caixa = env.scene["box"].data.root_link_pos_w
        d = torch.norm(caixa - comando[:, ALVO], dim=-1) / t.sigma_trazer.clamp(min=1e-6)
        valida = comando[:, VALIDA] > 0.5
        self.minimo = torch.where(valida, torch.minimum(self.minimo, d), self.minimo)
        return self.minimo

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.minimo[env_ids] = 1.0


class renda_manipulacao:
    """Soma de `_step_reward` sobre os `termos` dependentes de elo — a régua da REGRA 1
    (spec P10, proposta §0): "em todo instante existe incentivo até o alvo do elo; do
    fecho de um elo até o seguinte a renda não cai."

    ⚠ A MESMA lista que `recompensas.renda_congelada` congela, MAIS o próprio
    `renda_congelada`: é o total instantâneo de manipulação, ao vivo mais congelado —
    a soma que os checks 8 e 9 do smoke conferem no comando (não aqui, esta classe só
    publica o número no painel).

    ⚠ Os índices são resolvidos uma vez em `__init__`, e não a cada `__call__` — e AQUI
    isso é seguro (diferente de `renda_congelada`): o `MetricsManager` só carrega
    DEPOIS que `env.reward_manager` já existe e está completo
    (`manager_based_rl_env.py:329-339`).
    """

    def __init__(self, cfg, env):
        self.idx = [env.reward_manager.active_terms.index(n)
                    for n in cfg.params["termos"]]

    def __call__(self, env, termos) -> torch.Tensor:
        return env.reward_manager._step_reward[:, self.idx].sum(dim=-1)


def pads_em_contato(env, sensores: tuple[str, ...]) -> torch.Tensor:
    """Fração dos pads da lista que tocam a caixa, por env. 0, 0,5 ou 1,0.

    ⚠ FRAÇÃO, e não `min` nem `any`. O `squeeze` usa `min` das forças, portanto uma
    palma sozinha e nenhuma palma dão o MESMO zero exato — e essa ambiguidade já me
    fez ler abandono da tarefa onde havia uma mão na caixa. A fração separa os três
    estados que importam: nenhuma, uma, as duas.

    ⚠ Lê `found`, e não `force`: a pergunta é de CONTATO, não de intensidade. A
    intensidade é assunto do `squeeze`, que é contínuo desde o primeiro newton.
    """
    partes = []
    for nome in sensores:
        achou = env.scene[nome].data.found
        assert achou is not None, f"sensor '{nome}' precisa do field 'found'."
        partes.append((achou > 0).any(dim=-1).float())
    return torch.stack(partes, dim=-1).mean(dim=-1)


def momento_angular(env, sensor_name: str = MOMENTO_ANGULAR) -> torch.Tensor:
    """Módulo do momento angular do corpo todo. Proxy de balanço de braço."""
    return torch.norm(env.scene[sensor_name].data, dim=-1)


def tempo_de_voo(env, sensor_name: str = PES_NO_CHAO) -> torch.Tensor:
    """Tempo de voo médio dos pés que estão NO AR, por env, em segundos.

    ⚠ É a métrica central da F1, e a que o peso 0 do `air_time` apagava. Robô imóvel
    dá 0,0 — nenhum pé no ar, numerador zero.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    t = sensor.data.current_air_time
    assert t is not None, f"{sensor_name} precisa de track_air_time=True"
    return _media_por_env(t, t > 0.0)


class pico_de_altura:
    """Altura de pico do pé no balanço, medida NO POUSO.

    ⚠ Buffer PRÓPRIO, independente do da recompensa. Não é duplicação por descuido:
    se esta classe lesse o buffer do termo de recompensa, desligar o peso do
    `foot_swing_height` congelaria a métrica — que é o defeito que este arquivo
    existe para consertar.

    ⚠ E ela TEM `reset` — sem ele o pico do pé que estava no ar quando o robô caiu
    sobreviveria ao episódio, e a métrica subiria com a queda. É o mesmo bug que o
    `AlturaDeBalanco` conserta na recompensa, e o `MetricsManager` aplica a mesma
    regra: só chama `reset` em termo de classe que o tenha
    (`metrics_manager.py:132`).
    """

    def __init__(self, cfg, env):
        n_pes = env.scene[cfg.params.get("height_sensor_name", ALTURA_DO_PE)].num_frames
        self.picos = torch.zeros((env.num_envs, n_pes), device=env.device)
        self.dt = env.step_dt

    def __call__(self, env, sensor_name: str = PES_NO_CHAO,
                 height_sensor_name: str = ALTURA_DO_PE) -> torch.Tensor:
        contato: ContactSensor = env.scene[sensor_name]
        alturas = env.scene[height_sensor_name].data.heights
        no_ar = contato.data.found == 0
        self.picos = torch.where(no_ar, torch.maximum(self.picos, alturas), self.picos)
        pousou = contato.compute_first_contact(dt=self.dt)
        valor = _media_por_env(self.picos, pousou)
        self.picos = torch.where(pousou, torch.zeros_like(self.picos), self.picos)
        return valor

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.picos[env_ids] = 0.0


def velocidade_de_escorrego(env, sensor_name: str,
                            asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Velocidade horizontal do pé APOIADO. Um pé apoiado deveria estar parado."""
    robo: Entity = env.scene[asset_cfg.name]
    contato: ContactSensor = env.scene[sensor_name]
    assert contato.data.found is not None
    v = torch.norm(robo.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    return _media_por_env(v, contato.data.found > 0)


def forca_de_pouso(env, sensor_name: str = PES_NO_CHAO) -> torch.Tensor:
    """Módulo da força no instante do pouso, em newtons. Proxy de impacto."""
    contato: ContactSensor = env.scene[sensor_name]
    assert contato.data.force is not None
    f = torch.norm(contato.data.force, dim=-1)
    return _media_por_env(f, contato.compute_first_contact(dt=env.step_dt))
