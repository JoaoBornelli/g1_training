"""O ator do BFM-Zero, congelado e em lote. Carrega o `.pt` de 122 MB do passo 1.

Reconstrói SÓ duas peças do modelo do fabricante:

    _obs_normalizer   ajusta a escala da entrada  (BatchNorm, uma por chave)
    _actor            2048 de largura, 6 camadas, entrada 465 + z 256

O caminho de reconstrução espelha o `load_model` do fabricante (`base_model.py:31`):
`FBcprAuxModelConfig(**config.json)` -> `json_to_space(init_kwargs.json)` ->
`arch.actor.build(...)` e `cfg.obs_normalizer.build(...)`. As outras seis redes do
checkpoint (forward, backward, critic, discriminador) são de TREINO e ficam de fora,
o que economiza 3 GB.

⚠️ Duas coisas que o erro NÃO avisa:

1. **O normalizador é obrigatório.** O BFM ajusta a escala da entrada com
   estatísticas próprias. Sem ele o ator recebe números crus e devolve lixo — sem
   exceção nenhuma.
2. **A chave `privileged_state` tem que existir na entrada, mesmo zerada.** O
   normalizador foi construído sobre as 4 chaves do espaço de obs e roda antes do
   filtro do ator. O ator não lê essa chave, e o próprio código de inferência do
   fabricante passa zeros ali (`env.py:431`).
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_residual.bfm_code.fb_cpr_aux.model import FBcprAuxModelConfig  # noqa: E402
from g1_residual.bfm_code.misc.gym_spaces import json_to_space  # noqa: E402

PESO = pathlib.Path(__file__).resolve().parent / "peso" / "bfm_ator.pt"

DIM_ESTADO = 64
"""`dof_pos - default (29) + dof_vel (29) + gravidade projetada (3) + ang_vel (3)`."""
DIM_HISTORICO = 372
"""4 passos de: ação (29) + ang_vel (3) + dof_pos (29) + dof_vel (29) + gravidade (3)."""
DIM_PRIVILEGIADO = 463
"""O ator NÃO lê. Existe só porque o normalizador foi construído sobre ele."""
PASSOS_HISTORICO = 4


class AtorBFM(torch.nn.Module):
    """Ator do BFM-Zero congelado, com lote de N ambientes.

    Uso:
        ator = AtorBFM(device="cuda")
        a = ator(estado, ultima_acao, historico, z)     # [N, 29]

    A ação sai em UNIDADE DO BFM. Converta antes de somar com a nossa: o alvo de
    junta do BFM é `a * ACTION_SCALES * 5.0`, e o nosso é `a * ACTION_SCALES * 0.8`,
    ou seja o mesmo padrão por junta com um fator global de 6.25 = 5.0 / 0.8.
    """

    def __init__(self, caminho: pathlib.Path = PESO, device: str = "cpu"):
        super().__init__()
        assert caminho.is_file(), (
            f"não achei {caminho} — rode `python g1_residual/extrai_ator.py` primeiro")
        # `weights_only=True`: o `.pt` sai do nosso `extrai_ator.py` e tem só
        # tensores, dicionários de JSON e primitivos. Nada de objeto Python.
        d = torch.load(caminho, map_location="cpu", weights_only=True)

        cfg = FBcprAuxModelConfig(**{**d["config"], "device": device})
        espaco = json_to_space(d["init_kwargs"]["obs_space"])
        dim_acao = int(d["init_kwargs"]["action_dim"])

        self._normalizador = cfg.obs_normalizer.build(espaco)
        self._ator = cfg.archi.actor.build(espaco, cfg.archi.z_dim, dim_acao)
        self.std = float(cfg.actor_std)
        self.z_dim = int(cfg.archi.z_dim)
        self.dim_acao = dim_acao

        # `strict=True` de propósito: chave faltando aqui é bug silencioso depois.
        self._carrega("_obs_normalizer.", self._normalizador, d["pesos"])
        self._carrega("_actor.", self._ator, d["pesos"])

        self.z_tabela: dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in d["z"].items()}
        """41 comportamentos x 10 sementes, cada um [10, 256] com norma 16."""

        self.eval()
        self.requires_grad_(False)
        self.to(device)
        self._device = device

    @staticmethod
    def _carrega(prefixo: str, modulo: torch.nn.Module,
                 pesos: dict[str, torch.Tensor]) -> None:
        sd = {k[len(prefixo):]: v for k, v in pesos.items() if k.startswith(prefixo)}
        assert sd, f"nenhum peso com prefixo {prefixo}"
        modulo.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def forward(self, estado: torch.Tensor, ultima_acao: torch.Tensor,
                historico: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """[N,64] · [N,29] · [N,372] · [N,256]  ->  [N,29] na unidade do BFM."""
        n = estado.shape[0]
        obs = {
            "state": estado,
            "last_action": ultima_acao,
            "history_actor": historico,
            # zeros, igual ao caminho de inferência do fabricante. O ator não lê,
            # mas o normalizador foi construído sobre esta chave.
            "privileged_state": estado.new_zeros((n, DIM_PRIVILEGIADO)),
        }
        dist = self._ator(self._normalizador(obs), z, self.std)
        return dist.mean.float()

    def z_de(self, nome: str, semente: int | None = None) -> torch.Tensor:
        """`z` de um comportamento. `semente=None` devolve a média reprojetada.

        ⚠️ A média NÃO é sempre a melhor escolha. As 10 sementes de `move-ego-0-0`
        ("fica parado") estão a ~60° umas das outras — cos médio 0.500 — porque
        "não se mexa" tem muitas soluções e a inferência de reward fica
        indeterminada. Já `move-arms-0-0.7-m-m` tem cos 0.986. Para comportamento
        difuso, comparar a média com a semente 0 é obrigatório.
        """
        v = self.z_tabela[nome]
        v = v.mean(0) if semente is None else v[semente]
        return 16.0 * torch.nn.functional.normalize(v, dim=-1)
