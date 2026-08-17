"""Runner do multi-tarefa. Duas responsabilidades, e um requisito de não-quebrar.

1. **Congelar o normalizador nos canais de comando** (item 2 / F4). É a única das
   duas que muda o TREINO.
2. **Levar o estado do currículo no checkpoint** (item 0). É a que o workflow de
   blocos de 2k-3k depende: sem ela o currículo volta ao nível 0 a cada retomada,
   e ele retoma 10-15 vezes.

⚠️ **E não quebrar o `play`.** O `play.py` do mjlab resolve o `runner_cls` no MESMO
registro que o treino (`play.py:203`) e instancia **sem `log_dir`** (`:204`). Ou
seja esta classe roda no notebook toda vez que se vai olhar o robô se mexer — que é
a única coisa que roda local. Duas consequências, tratadas abaixo:

- nada aqui pode assumir `log_dir` nem currículo existindo;
- a reinjeção do currículo não pode atropelar o **pin manual** (no `play` você fixa
  tarefa e nível na mão pra inspecionar um caso). Saída de graça: o `play` chama
  `load(..., load_cfg={"actor": True})`, então basta reinjetar só quando o
  `load_cfg` pedir. Sem flag nova, sem `if play:`.
"""
from __future__ import annotations

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner

from . import tasks as T

CANAIS_CONGELADOS = ("target_pos_b", "face_alvo", "dir_alvo", "task_onehot",
                     "twist_cmd")
"""Os canais que recebem `mean=0, var=1` fixos — 20 números desde a S10.

Por que ESTES: são os canais cuja DISTRIBUIÇÃO muda no meio da run. Durante a
Fase 0 só o `parado` roda, então `target_pos_b`, `face_alvo` e `dir_alvo` ficam
constantes em 0 e 7 dos 8 slots do one-hot também. O `EmpiricalNormalization`
computa `_std = 0` pra canal constante **no primeiro update**, e o `forward` é
`(x − mean)/(_std + 1e-2)` -> quando o canal acende, 1.0 entra como **100.0**.
Isso infla a norma do gradiente, e o `clip_grad_norm_(1.0)` do PPO então encolhe
o passo da rede TODA. O pior caso não é o one-hot, é o `target_pos_b`: vira ~2 m
quando o `andar` abre (~200×) e é um canal que a política tem que ler
QUANTITATIVAMENTE, não só detectar como flag.

Por que `box_pos_b` e `box_rot_b` NÃO entram: a caixa está sempre na cena e sempre
tem posição e orientação. Esses canais variam desde a primeira iteração, então o
normalizador tem estatística de verdade pra aprender.

⚠️ **`twist_cmd` entrou na S10, e ele é obrigatório aqui.** Na Fase 0 só o `parado`
roda e o comando fica constante em zero — exatamente o caso que produz `_std = 0`.

⚠️ **O congelamento do `target_pos_b` deixou de ser precaução e virou necessidade.**
Depois da S10 o ator o vê preenchido em apenas duas das sete tarefas (`botar` e
`reorientar`), portanto ele é zero na maior parte da amostra.
"""


STD_AO_ABRIR_TAREFA = 0.8
"""Piso do `std` da política no instante em que o currículo abre uma tarefa. (17/08)

O `std` começa em 1,0 (`init_std` do cfg do fabricante) e decai: o bloco 3 media 0,45.
Uma tarefa nova entra na distribuição justamente quando a exploração já está estreita.

0,8 e não 1,0: a política já sabe andar e alcançar, e voltar ao ruído inicial
desmancharia essas habilidades. Combina com `learning_rate = 5e-4` no começo do bloco,
pelo mesmo motivo do warm-start em degrau novo."""


def _indices_congelados(env, modelo) -> torch.Tensor:
    """Índices dos canais de comando NO VETOR QUE ESTE MODELO CONCATENA.

    Derivado do `observation_manager`, nunca hardcodado: apagar ou reordenar um
    termo desloca todos os índices seguintes, e o modo term-major do histórico
    reordena o vetor inteiro. Esta é a falha que a micro-sessão da §15 pega mas que
    **não dá erro nenhum** — a fatia errada congela propriocepção em vez de comando,
    e o treino só fica pior sem dizer por quê.

    O ator e o crítico concatenam grupos diferentes (o crítico tem 12 canais
    privilegiados a mais), então os índices SÃO diferentes entre os dois. Por isso a
    função recebe o modelo e lê o `obs_groups` dele."""
    am = env.observation_manager
    idx: list[int] = []
    base = 0
    for grupo in modelo.obs_groups:
        for nome, dim in zip(am.active_terms[grupo], am.group_obs_term_dim[grupo]):
            n = int(dim[0])
            if nome in CANAIS_CONGELADOS:
                idx.extend(range(base, base + n))
            base += n
    return torch.tensor(idx, dtype=torch.long, device=modelo.obs_normalizer._mean.device)


def _congelar(modelo, idx: torch.Tensor) -> None:
    """Envolve o `update` do normalizador pra manter `mean=0, var=1` em `idx`.

    Por que envolver o `update` e não usar o `until` do rsl_rl: o `until` é GLOBAL
    — para de aprender em TODOS os canais. Aqui só os canais de comando param, e a
    propriocepção continua normalizando normalmente.

    Por que no normalizador e não no `update_normalization` do modelo: o
    `torch.compile` substitui `alg.actor` por um wrapper, mas o objeto normalizador
    é o mesmo. Envolver aqui funciona compilado ou não."""
    norm = modelo.obs_normalizer
    if getattr(norm, "_congelado", False):
        return
    original = norm.update

    def update(x):
        original(x)
        norm._mean[:, idx] = 0.0
        norm._var[:, idx] = 1.0
        norm._std[:, idx] = 1.0

    norm.update = update
    norm._congelado = True
    # aplica já, pra o estado inicial também estar congelado
    norm._mean[:, idx] = 0.0
    norm._var[:, idx] = 1.0
    norm._std[:, idx] = 1.0


class MultitaskRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None,
                 device: str = "cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        self.idx_congelados: dict[str, torch.Tensor] = {}
        for nome in ("actor", "critic"):
            modelo = getattr(self.alg, f"_raw_{nome}", None) or getattr(self.alg, nome)
            if not getattr(modelo, "obs_normalization", False):
                continue
            idx = _indices_congelados(self.env.unwrapped, modelo)
            self.idx_congelados[nome] = idx
            _congelar(modelo, idx)
        self._ligar_diagnostico_vantagem()
        self._ligar_reaquecimento_std()

    def _ligar_diagnostico_vantagem(self) -> None:
        """Faz a série de vantagem por tarefa chegar ao log. (S15)

        ⚠️ **A normalização é a ÚLTIMA linha do `compute_returns`** (`ppo.py:186-188`),
        e não a primeira do `update` como parecia. Não há ponto de gancho entre as
        duas: quando este código roda, `st.advantages` já está normalizado.

        Por isso o `_std_vantagem_por_tarefa` reconstrói a vantagem de
        `returns − values` em vez de lê-la. O gancho aqui só decide QUANDO medir —
        depois do `compute_returns`, com os buffers da iteração corrente cheios.

        O valor entra no `loss_dict` do `update`, que é o canal que o
        `on_policy_runner` já passa ao logger (`:115`). Sem gancho novo no laço.

        Só LÊ. A S15 proíbe implementar a normalização por tarefa nesta rodada."""
        self._diag_vantagem: dict[str, float] = {}
        _compute_returns = self.alg.compute_returns
        _update = self.alg.update

        def compute_returns(*a, **kw):
            saida = _compute_returns(*a, **kw)
            self._diag_vantagem = self._std_vantagem_por_tarefa()
            return saida

        def update(*a, **kw):
            perdas = _update(*a, **kw)
            if isinstance(perdas, dict):
                perdas.update(self._diag_vantagem)
            return perdas

        self.alg.compute_returns = compute_returns
        self.alg.update = update

    # -------------------------------------------- exploração ao abrir tarefa
    def _ligar_reaquecimento_std(self) -> None:
        """Sobe o `std` da política quando o currículo ABRE uma tarefa. (17/08)

        **Por que ele existe.** O `std` cai ao longo do treino: começa em 1,0 e o bloco
        3 media 0,45. Quando uma tarefa nova abre, a política encontra uma distribuição
        que nunca viu com a exploração já estreita — e a tarefa nova é justamente a que
        precisa explorar.

        ⚠️ **`clamp_(min=...)`, nunca `fill_()`.** Ele só SOBE. Se o `std` já estiver
        acima do alvo, mexer nele cortaria exploração em vez de dar.

        ⚠️ Ele NÃO troca a parametrização do `std`. Trocar `std_type` de `"scalar"` para
        `"log"` renomeia o parâmetro (`std_param` -> `log_std_param`,
        `rsl_rl/modules/distribution.py:165-168`) e o `load` com `strict=True` quebra. E
        não daria o que se quer: o `GaussianDistribution` tem UM vetor global, então o
        `std` não pode variar por tarefa de jeito nenhum. Exploração por tarefa exigiria
        o `HeteroscedasticGaussianDistribution`, que dobra a última camada do ator —
        Categoria C, checkpoint não carrega.

        **Gancho pelo `alg.update`**, o mesmo padrão do diagnóstico de vantagem: não há
        laço a tocar. Roda uma vez por iteração de PPO.

        O contador nasce do estado do currículo, não de zero — assim um resume no meio
        de um bloco não dispara o reaquecimento de novo."""
        self._n_abertas: int | None = None
        _update = self.alg.update

        def update(*a, **kw):
            self._reaquece_std_se_abriu()
            return _update(*a, **kw)

        self.alg.update = update

    def _reaquece_std_se_abriu(self) -> None:
        orq = self._termos_curriculo().get("orquestrador")
        if orq is None:
            return
        n = len(getattr(orq, "abertas", ()))
        if self._n_abertas is None:
            self._n_abertas = n           # 1ª iteração: só registra a linha de base
            return
        if n <= self._n_abertas:
            return
        self._n_abertas = n
        modelo = getattr(self.alg, "_raw_actor", None) or getattr(self.alg, "actor")
        param = getattr(getattr(modelo, "distribution", None), "std_param", None)
        if param is None:
            print("[RUNNER] tarefa nova, mas o ator não tem `std_param` — sem "
                  "reaquecimento (distribuição diferente de GaussianDistribution?)")
            return
        antes = float(param.data.mean())
        param.data.clamp_(min=STD_AO_ABRIR_TAREFA)
        print(f"[RUNNER] tarefa nova ({n} abertas): std {antes:.3f} -> "
              f"{float(param.data.mean()):.3f}")

    # ------------------------------------------------------- S15: diagnóstico
    def _std_vantagem_por_tarefa(self) -> dict[str, float]:
        """Desvio padrão da vantagem POR TAREFA, antes de normalizar. Só log. (S15)

        Responde se a normalização de vantagem por tarefa é necessária. O rsl_rl
        normaliza UMA VEZ sobre o rollout inteiro (`ppo.py:186-188`,
        `normalize_advantage_per_mini_batch=False` por default), e as sete tarefas
        entram no mesmo tensor. Se uma tiver dispersão muito maior, ela domina o
        gradiente das outras proporcionalmente.

        ⚠️ **Reconstrói a vantagem de `returns − values`, e NÃO lê `st.advantages`.**
        A normalização é a ÚLTIMA linha do `compute_returns`, não a primeira do
        `update` — portanto `st.advantages` já está normalizado em qualquer ponto em
        que este código consiga rodar, e medi-lo dá 1.0 em toda tarefa. Medido em
        05/08: `diag/std_vantagem/parado = 1.0000`, um número que não diz nada.

        `returns` e `values` são os buffers que a linha 185 usa para montar a vantagem
        crua, e a normalização não os altera.

        ⚠️ **Se os sete forem parecidos, o assunto morre.** É esse o ponto de medir:
        a S15 proíbe implementar a normalização por tarefa nesta rodada. Só o
        diagnóstico.

        Devolve `{}` quando o storage ainda não tem retorno calculado — é o caso na
        primeira iteração e em `play`."""
        st = getattr(self.alg, "storage", None)
        ret = getattr(st, "returns", None)
        val = getattr(st, "values", None)
        if ret is None or val is None:
            return {}
        adv = ret - val
        tarefa = getattr(self.env.unwrapped, "tarefa_sorteada", None)
        if tarefa is None:
            return {}
        # advantages: [passos, envs, 1] -> o rótulo de tarefa é por ENV, e vale para a
        # coluna inteira daquele env no rollout.
        a = adv.squeeze(-1)                                   # [passos, envs]
        out: dict[str, float] = {}
        for t in range(T.NUM_TASKS):
            m = tarefa == t
            if not bool(m.any()):
                continue
            out[f"diag/std_vantagem/{T.NAMES[t]}"] = float(a[:, m].std())
        return out

    # ------------------------------------------------------------------ currículo
    def _termos_curriculo(self) -> dict:
        """Termos de currículo que sabem se serializar. Vazio é resposta válida —
        no `play` o env pode nem ter currículo."""
        mgr = getattr(self.env.unwrapped, "curriculum_manager", None)
        if mgr is None or not hasattr(mgr, "get_term_cfg"):
            return {}
        out = {}
        for nome in getattr(mgr, "active_terms", []):
            termo = mgr.get_term_cfg(nome).func
            if hasattr(termo, "state_dict") and hasattr(termo, "load_state_dict"):
                out[nome] = termo
        return out

    # --------------------------------------------------- espaço de ação (etiqueta)
    def _assinatura(self) -> dict:
        """Identidade do ESPAÇO DE AÇÃO em que este checkpoint foi treinado.

        Existe por causa de 04/08/2026. Desde `dim_c = 0` (`g1_residual`, commit
        `b931c9c`) a ação do residual é **29** e a obs **151** — exatamente as do
        multi-tarefa. Os dois checkpoints passaram a ser intercambiáveis para o
        `load_state_dict`, então trocar um pelo outro **não dá `size mismatch`
        nenhum**. Antes disso (49 canais) o cross-load falhava alto sozinho.

        O que muda entre eles é o TERMO DE AÇÃO: no residual o alvo de junta sai do
        ator do BFM congelado e a rede só soma um delta clampeado. Rodar um checkpoint
        do multi-tarefa no `play` do residual mostra o BFM de pé, e isso se lê como
        política treinada — foi o engano de 04/08, que custou uma sessão.

        A CLASSE do termo é o discriminador, e ela é derivável do env nos dois lados,
        `save` e `load`. Por isso a etiqueta não precisa carregar o `task_id` — que o
        runner não tem — nem depender de o `play` ter registrado o id certo.
        """
        mgr = getattr(self.env.unwrapped, "action_manager", None)
        if mgr is None:                       # env sem ação não existe, mas não custa
            return {}
        return {"termos": {n: type(mgr.get_term(n)).__name__
                           for n in mgr.active_terms},
                "dim": int(mgr.total_action_dim)}

    def _confere_assinatura(self, infos: dict | None) -> None:
        """Recusa checkpoint de outro espaço de ação. Sem etiqueta, só avisa.

        Falha ALTO e no carregamento, não depois: o sintoma de errar é o robô se
        comportando bem por um motivo que não é a política, e isso não tem sinal
        próprio no log.

        Checkpoint sem etiqueta é anterior a 04/08 e continua carregando de propósito
        — travar aqui quebraria o resume das runs em voo, que é justamente o workflow
        de 10-15 retomadas."""
        atual = self._assinatura()
        if not atual:
            return
        gravada = (infos or {}).get("assinatura")
        if gravada is None:
            print("[AVISO] checkpoint sem etiqueta de espaço de ação (anterior a "
                  f"04/08). O env atual usa {atual['termos']} — confira você mesmo "
                  f"que o checkpoint é desta task.")
            return
        if gravada != atual:
            raise SystemExit(
                "[RECUSADO] este checkpoint foi treinado em OUTRO espaço de ação.\n"
                f"  gravado no checkpoint: {gravada}\n"
                f"  env atual:             {atual}\n"
                "Não dá `size mismatch` porque multi-tarefa e residual têm ação 29 e "
                "obs 151 desde `dim_c = 0`. Se o env atual é `ResidualBFMAction` e o "
                "checkpoint é do multi-tarefa, o que apareceria na tela é o BFM "
                "segurando o robô, não a sua política.")

    def save(self, path: str, infos: dict | None = None) -> None:
        extra: dict = {}
        estado = {n: t.state_dict() for n, t in self._termos_curriculo().items()}
        if estado:
            extra["curriculum"] = estado
        assinatura = self._assinatura()
        if assinatura:
            extra["assinatura"] = assinatura
        if extra:
            infos = {**(infos or {}), **extra}
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        # Antes de qualquer coisa, e nos DOIS caminhos (treino e `play`): o `play`
        # passa `load_cfg={"actor": True}` e cai fora do bloco de currículo abaixo.
        self._confere_assinatura(infos)

        # `load_cfg is None` = "carregue tudo", que é o caminho do TREINO. O `play`
        # passa `{"actor": True}` e cai fora daqui sozinho, preservando o pin manual.
        quer = load_cfg is None or bool(load_cfg.get("curriculum", False))
        if not quer or not infos or "curriculum" not in infos:
            return infos

        termos = self._termos_curriculo()
        for nome, estado in infos["curriculum"].items():
            if nome in termos:
                termos[nome].load_state_dict(estado)
            else:
                print(f"[AVISO] checkpoint traz currículo '{nome}' que não existe "
                      f"no env atual — ignorado.")
        return infos
