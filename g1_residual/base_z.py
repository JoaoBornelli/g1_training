"""A base de comportamento: 20 direções em que a RL procura o "como fazer".

O BFM-Zero é comandado por um latente `z` de 256 números, com norma fixa em
sqrt(256) = 16. Mas o `z` VÁLIDO não é a esfera inteira: ele vem do
`backward_map(estados)`, que é uma superfície fina lá dentro. Um ponto sorteado na
esfera não é comportamento nenhum.

Os 41 comportamentos prontos do `reward_locomotion.pkl` amostram essa superfície.
Medido: eles ocupam **~14 dimensões efetivas**, e uma base de 20 direções
reconstrói todos os 41 com cosseno **>= 0,93 no pior caso**.

    k=12  pior caso cos 0,76
    k=16  pior caso cos 0,81
    k=20  pior caso cos 0,93   <- escolhido
    k=32  pior caso cos 0,98

Então a RL emite **20 números**, não 256 nem 41. Ela alcança o mesmo espaço útil
com 6 vezes menos dimensões de busca, e começa dentro da superfície válida em vez
de ter que achá-la.

**O prior é partida, não cerca.** Calibrado com a escala 0,3 e o desvio típico da
política:

    |c| ~ 0,25  ->  12° do prior
    |c| ~ 1,0   ->  40° (até 70°)
    |c| ~ 2,0   ->  59° (até 102°)

Entre dois comportamentos DIFERENTES o ângulo médio é 83°. Ou seja a rede sai do
prior e chega em qualquer outro comportamento. Se agachar não for o melhor jeito de
pegar a caixa, ela sai de `crouch-0` e o `mais_proximo()` diz para onde foi.
"""
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_multitask import tasks as T  # noqa: E402

DIM_C = 20
"""Dimensões da base. 20 dá pior caso 0,93 na reconstrução dos 41."""

NORMA_Z = 16.0
"""sqrt(256). O `norm_z=true` do config exige isto."""

ESCALA_C = 0.3
"""Multiplica o coeficiente que a política emite.

Com o desvio padrão inicial de ~0,9 por dimensão, isto põe a base a ~12-14° do
prior no começo — partida mansa. A MÉDIA da política não tem teto, então depois ela
vai aonde quiser: 40° com |c|~1, 59° com |c|~2."""

PRIOR = {
    T.PARADO:       "move-ego-0-0",
    T.ANDAR:        "move-ego-0-0.3",
    T.PEGAR:        "crouch-0",
    T.BOTAR:        "raisearms-m-m",
    T.REORIENTAR:   "raisearms-m-m",
    T.PARADO_CAIXA: "raisearms-m-m",
    T.ANDAR_CAIXA:  "move-arms-0-0.7-m-m",
}
"""Onde cada tarefa COMEÇA. Não onde ela termina.

`move-arms-0-0.7-m-m` é literalmente "anda para frente com os braços erguidos", que
é a postura do `andar c/ caixa`. E `crouch-0` é um palpite meu sobre pegar caixa —
a rede tem autoridade para sair dele, e é essa a intenção.

Prior errado custa iterações, não custa resultado."""

PRIOR_UNICO = "move-ego-0-0"
"""Alternativa sem suposição nenhuma: as 7 tarefas partem de ficar de pé.

Custa mais iterações, porque a rede tem que descobrir agachar sozinha. Ligue com
`BaseZ(..., prior_unico=True)` quando quiser tirar meu palpite do caminho."""


class BaseZ:
    """A base de 20 direções, os priors, e a leitura humana do que a rede escolheu."""

    def __init__(self, z_tabela: dict[str, torch.Tensor], device: str = "cpu",
                 dim: int = DIM_C, prior_unico: bool = False):
        self.nomes = sorted(z_tabela)
        # média das 10 sementes, reprojetada na esfera
        M = self._projeta(torch.stack([z_tabela[n].mean(0) for n in self.nomes]))
        self.M = M.to(device)                       # [41, 256]

        mu = M.mean(0, keepdim=True)
        _, S, Vh = torch.linalg.svd((M - mu).double(), full_matrices=False)
        # escala cada direção pelo valor singular dela, para que um coeficiente de
        # ordem 1 desloque o `z` na mesma ordem em que os comportamentos diferem
        self.B = (Vh[:dim] * (S[:dim] / len(M) ** 0.5).unsqueeze(1)).float().to(device)
        self.dim = dim
        self.energia = float((S[:dim] ** 2).sum() / (S ** 2).sum())

        alvo = {t: PRIOR_UNICO for t in PRIOR} if prior_unico else PRIOR
        self.prior = torch.stack([
            self.M[self.nomes.index(alvo[t])] for t in range(T.NUM_TASKS)
        ]).to(device)                                # [7, 256]
        self.prior_unico = prior_unico

    @staticmethod
    def _projeta(v: torch.Tensor) -> torch.Tensor:
        return NORMA_Z * F.normalize(v, dim=-1)

    def z(self, tarefa: torch.Tensor, c: torch.Tensor,
          escala: float = ESCALA_C) -> torch.Tensor:
        """`[N] long` + `[N, dim]` -> `[N, 256]` com norma 16.

        O deslocamento é somado ANTES da projeção, então a norma sai certa sempre e
        a política nunca precisa aprender a manter a norma."""
        return self._projeta(self.prior[tarefa] + escala * (c @ self.B))

    # ---------------------------------------------------------------- leitura
    def mais_proximo(self, z: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Qual dos 41 comportamentos está mais perto, e o cosseno.

        É a tabela de contribuição, mas para comportamento: dá para ler no log
        "em `pegar` a política andou 47° e agora está perto de
        `move-ego-low0.5-0-0`" em vez de olhar 256 números opacos."""
        c = F.normalize(z, dim=-1) @ F.normalize(self.M, dim=-1).T   # [N, 41]
        val, idx = c.max(dim=-1)
        return [self.nomes[i] for i in idx.tolist()], val

    def graus_do_prior(self, tarefa: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Quantos graus a política andou desde o ponto de partida."""
        c = F.cosine_similarity(z, self.prior[tarefa], dim=-1).clamp(-1.0, 1.0)
        return torch.rad2deg(c.arccos())
