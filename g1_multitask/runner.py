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

CANAIS_CONGELADOS = ("target_pos_b", "face_alvo", "dir_alvo", "task_onehot")
"""Os canais que recebem `mean=0, var=1` fixos — 17 números.

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
"""


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

    def save(self, path: str, infos: dict | None = None) -> None:
        estado = {n: t.state_dict() for n, t in self._termos_curriculo().items()}
        if estado:
            infos = {**(infos or {}), "curriculum": estado}
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)

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
