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

PRIOR: dict[int, tuple[str, int | None]] = {
    #                nome                    semente (None = média das 10)
    T.PARADO:       ("move-ego-0-0",         0),
    T.ANDAR:        ("move-ego-0-0.3",       0),
    T.PEGAR:        ("move-ego-0-0",         0),
    T.BOTAR:        ("raisearms-m-m",     None),
    T.REORIENTAR:   ("raisearms-m-m",     None),
    T.PARADO_CAIXA: ("raisearms-m-m",     None),
    T.ANDAR_CAIXA:  ("move-arms-0-0.7-m-m", 0),
}
"""Onde cada tarefa COMEÇA. Não onde ela termina.

⚠️ **O prior do `pegar` era `crouch-0` e ele DESABA.** Medido no `fumaca.py`, 16 envs,
150 passos, residual em zero:

    crouch-0                  0% de pé   10 passos   pelve mín 0,124 m
    crouch-0.25               0% de pé    9 passos   pelve mín 0,159 m
    move-ego-low0.5-0-0       0% de pé    8 passos   pelve mín 0,293 m
    move-ego-low0.6-0-0.7   6,2% de pé   47 passos   pelve mín 0,478 m
    move-ego-0-0            100% de pé  150 passos   pelve mín 0,735 m
    raisearms-m-m           100% de pé  150 passos   pelve mín 0,762 m
    sitonground            18,8% de pé   83 passos   pelve mín 0,101 m

E a CARGA não derruba o BFM: no `parado c/ caixa` com `raisearms-m-m` ele fica
150 de 150 passos de pé, pelve mín 0,763 m — igual ao caso sem carga (0,766). A caixa
escorrega para 0,647 m, porque com ação nula as palmas só tocam; mas o robô não cai.
Isso mata o maior risco que eu tinha levantado (o BFM treinou em LaFAN, sem peso nas
mãos). Ressalva: a caixa sai rápido das mãos, então ele não carrega peso por muito
tempo — o teste limpo exige o residual já segurando.

**Nenhum comportamento de agachar do BFM sobrevive sozinho no nosso env.** O
`crouch-N` é ALTURA ALVO, não intensidade — `crouch-0` quer dizer "vai ao chão", e
sem residual para estabilizar a descida ele não segura. O `sitonground` de fato
senta, o que é comportamento correto, não falha.

Consequência boa: agachar passa a ser o que a REDE descobre. E `crouch-0` continua
alcançável na base (é um dos 41), então ela pode ir para lá quando aprender a
estabilizar a descida. Era exatamente o pedido — não fixar minha suposição.

Consequência a vigiar: o residual tem trabalho real na perna, então o clamp de
0,35 rad pode ser pouco. Se o `pegar` não sair do lugar, alargue a perna antes de
mexer em qualquer outra coisa.

⚠️ **A SEMENTE importa, e a média é pior onde o `z` é difuso.** Medido em 300 passos
(6 s), residual em zero, olhando a DERIVA horizontal:

    move-ego-0-0[média]    deriva 0,993 m   <- quase 1 m em 6 s
    move-ego-0-0[0]        deriva 0,194 m   <- 5x menos
    raisearms-m-m[média]   deriva 0,324 m

É essa deriva que se vê como "o robô fica dançando". Ela casa com o achado do cosseno:
as 10 sementes de `move-ego-0-0` estão a ~60° umas das outras (cos médio 0,500), porque
"não se mexa" tem muitas soluções e a inferência de reward fica indeterminada. A média
de vetores tão espalhados cai num ponto que não é nenhum deles.

Onde as sementes CONCORDAM a média é boa: `move-arms-0-0.7-m-m` tem cos 0,986, e
`raisearms-m-m` deriva só 0,324 m. Daí a coluna de semente ser por tarefa.

`move-ego-0-0.3` e `move-arms-0-0.7-m-m` levam semente 0 por analogia, não por medição
— a varredura de deriva só cobriu o que o `parado` e o `pegar` usam."""

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

        alvo = ({t: (PRIOR_UNICO, 0) for t in PRIOR} if prior_unico else PRIOR)
        self.prior = torch.stack([
            self._de(z_tabela, *alvo[t]) for t in range(T.NUM_TASKS)
        ]).to(device)                                # [7, 256]
        self.prior_unico = prior_unico

    @staticmethod
    def _projeta(v: torch.Tensor) -> torch.Tensor:
        return NORMA_Z * F.normalize(v, dim=-1)

    @classmethod
    def _de(cls, tabela: dict[str, torch.Tensor], nome: str,
            semente: int | None) -> torch.Tensor:
        """`z` de um comportamento, por semente ou pela média das 10."""
        v = tabela[nome]
        return cls._projeta((v.mean(0) if semente is None else v[semente]).unsqueeze(0))[0]

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
