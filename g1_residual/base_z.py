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

ESCALA_C = 0.0
"""Multiplica o coeficiente que a política emite. É a TAXA da busca de comportamento:
define quantos graus o `z` anda por unidade de `|c|`.

⚠️ **É ZERO desde 03/08/2026: a busca de comportamento está DESLIGADA.** `z = prior`
sempre, e o prior é `move-ego-0-0` nas 7 tarefas. Os 20 canais de `c` continuam no
espaço de ação mas não têm efeito nenhum — e o `action_rate_l2` deixou de cobrá-los
(ver `rewards.action_rate_l2_juntas`), então eles são ruído sem custo.

Por que desligar em vez de apagar: apagar muda a largura do espaço de ação, o que
mexe no `_raw_actions`, no normalizador e no checkpoint. Zerar a escala é uma linha e
é reversível. Apagar de verdade é a limpeza depois de o experimento decidir.

**O motivo é medição, não economia.** Com a busca ligada em 1.0 a run desmontou
(episódio 765 -> 17,9 passos, `fell_over` 0,58 -> 114). Duas explicações, e as duas
levam ao mesmo lugar: (a) o ruído de exploração nos 20 canais sacode o `z` a cada
20 ms, e o BFM tem histórico interno, então o comportamento nunca assenta; (b) a
busca FUNCIONOU e achou locomoção — o robô começou a andar tentando pegar. A
hipótese (b) é do user e é mais simples que a minha (a).

Histórico do valor, para não repetir a conta: ele era **0,3**, e nesse valor a busca
não existia. A tabela abaixo é a medição que mostrou isso.

⚠️ **Era 0.3, e nesse valor a busca não existe.** O docstring antigo afirmava
"|c| ~ 1,0 -> 40°, |c| ~ 2,0 -> 59°, o prior é partida não cerca". Isso estava errado
por ~12×. Medido em 31/07/2026 por mínimos quadrados — não por amostragem aleatória,
que subestima muito em 20 dimensões:

    |c| necessário para trocar de comportamento, com escala 0.3
    andar  -> parado        75,5°      |c| = 27,7   (12,3 por dimensão)
    parado -> andar         75,5°      |c| = 46,3   (28,2 por dimensão)
    pegar  -> agachar       89,8°      |c| = 37,5   (30,3 por dimensão)

E a política de fato emite `|c| ≈ 10` (medido no checkpoint `model_1950`, que move o
`z` só **24,6°**). Faltavam 3 a 4×.

**A causa não é a multa do `action_rate_l2`** — ele pune a DIFERENÇA entre passos, então
um `c` grande e constante custaria zero. A causa é gradiente que desaparece: com escala
0.3, mexer `c` de 1 muda o `z` em ~0,8°, o efeito no reward fica abaixo do ruído, e os 20
canais nunca aprendem a crescer. Assinatura no log: `action_rate_l2` marcava
−3,07/−3,10/−3,10/−3,14 em quatro tarefas de física completamente diferente (variação de
2,4%), ou seja era dominado por ruído idêntico em toda tarefa.

**Sintomas que isso explicava:** `parado` funcionava (o prior dele JÁ é o comportamento
certo); `andar` andava mas não parava no alvo (travado em `move-ego-0-0.3` = frente a
0,3 m/s, e o BFM **não vê o nosso twist**, só o `z`); `pegar` nunca agachava; `grasp`,
`lift`, `box_at_peito`, `box_at_prateleira`, `orienta_face` e `hold_still` todos em zero.

**O compromisso de escolher 1.0:**

    escala   excursão no início (|c|~4)   |c| para 75°
    0.3               13°                    28      <- não alcançável
    0.5               21°                    17
    1.0               38°                     8      <- 2× o init, PPO aprende
    2.1               75°                     4      <- o prior deixa de valer

Em 38° o prior ainda importa: a média entre dois comportamentos DISTINTOS é 83°. E o
gradiente é **constante** em todo o espaço, sem região plana onde a busca morre — foi por
isso que preferi subir a escala em vez de trocar para simplex sobre os 41 (que chega
exato, mas satura perto do prior: `dw/dc = 0,0023` com k=6).

⚠️ **Vigiar:** a 1.0 a excursão aleatória inicial vira ~38°, contra 13° antes. O TESTE 2
do `fumaca.py` mostra o BFM de pé 100% com o residual aleatório INTEIRO, então não deve
derrubar — mas é o único número desta mudança que não foi medido nesta escala."""

PRIOR: dict[int, tuple[str, int | None]] = {
    #                nome                    semente (None = média das 10)
    T.PARADO:       ("move-ego-0-0", 0),
    T.ANDAR:        ("move-ego-0-0", 0),
    T.PEGAR:        ("move-ego-0-0", 0),
    T.BOTAR:        ("move-ego-0-0", 0),
    T.REORIENTAR:   ("move-ego-0-0", 0),
    T.PARADO_CAIXA: ("move-ego-0-0", 0),
    T.ANDAR_CAIXA:  ("move-ego-0-0", 0),
}
"""Onde cada tarefa COMEÇA. Não onde ela termina.

⚠️ **As 7 apontam para o MESMO `z` desde 03/08/2026, e com `ESCALA_C = 0` esse `z`
nunca muda.** O BFM deixou de escolher comportamento: ele é só controlador de
equilíbrio em pé, e TODO o resto — andar, agachar, alcançar, segurar — sai do
residual de junta.

Antes a tabela era `move-ego-0-0.3` no `andar`, `raisearms-m-m` em três tarefas e
`move-arms-0-0.7-m-m` no `andar c/ caixa`. Saíram depois de o user navegar os 41 no
`poses.py` (visor com ação zero, o BFM puro) e concluir: *"somente a pose
`move-ego-0-0` que é parado quieto, é válida. todas as outras são movimentações que
vão perturbar ou dificultar a movimentação do robo"*. Coerente com a origem — os 41
vêm de `reward_locomotion.pkl`, é biblioteca de LOCOMOÇÃO, e comandar movimento briga
com manipulação.

Medido em 03/08 no `autoridade.py`: `move-ego-0-0` semente 0 fica **ereto 100%** dos
300 passos, pelve 0,770 a 0,788. É a única das candidatas que se sustenta sozinha —
`crouch-0.25` fica ereto **9%** e termina com a pelve em 0,128.

**Consequência a vigiar:** o `andar` perdeu o prior de marcha. O residual tem que
produzir locomoção em cima de um controlador que resiste a sair do lugar. É de
propósito — é o experimento que o user pediu ("ver se o robo aprende a se mover") — e
é o motivo de o clamp de pitch ter subido de 0,35 para 1,05 rad no `acao.py`.

⚠️ **A coluna de SEMENTE não é capricho.** A tabela do BFM guarda **10 vetores `z` por
nome**, não um: cada um é uma execução da inferência de reward com semente diferente, e
as 10 deveriam descrever o mesmo comportamento. A média das 10 só faz sentido se elas
concordarem. Medido em 31/07/2026:

    comportamento          ângulo médio entre as 10   cos(média, sementes)
    move-ego-0-0.3                 4,5°                     0,999    média OK
    move-ego-0-0.7                 3,0°                     0,999    média OK
    move-ego-90-0.3                6,3°                     0,997    média OK
    move-arms-0-0.7-m-m            9,6°                     0,994    média OK
    move-ego-0-0                  60,0°                     0,742    média INVÁLIDA
    raisearms-m-m                 74,2°                     0,587    média INVÁLIDA

O número que fecha o argumento: **entre dois comportamentos DIFERENTES o ângulo médio é
83°.** Duas sementes de `raisearms-m-m` a 74° uma da outra estão quase tão longe quanto
dois comportamentos distintos — a média delas cai num ponto que não é nenhuma das duas.

**Por que essas duas discordam:** a recompensa é indeterminada. `move-ego-0-0` quer dizer
"não se mexa", e existem infinitas posturas paradas; `raisearms-m-m` também tem muitas
soluções. Já `move-ego-0-0.3` ("ande para frente a 0,3 m/s") restringe, e as 10 concordam.

**Consequência medida**, prior do `pegar` (`move-ego-0-0`) com a MÉDIA e residual em zero:
a distância caixa→peito cresce de 0,539 m para **2,631 m** em 1000 passos — o robô anda
2,6 m para longe da caixa tentando "não se mexer". Com a semente 0 a deriva cai de
0,993 m para 0,194 m em 300 passos, 5× menos. Por isso as duas linhas inválidas levam
semente 0 e as outras continuam na média (que ainda filtra ruído de inferência).

`raisearms-m-m` é prior de TRÊS tarefas (`botar`, `reorientar`, `parado c/ caixa`), então
essa era a média inválida mais cara.

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

`move-arms-0-0.7-m-m` (`andar c/ caixa`) ainda NÃO foi medido — a varredura só cobriu
o que o experimento do `pegar` usa."""

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
        """`z` de um comportamento, por SEMENTE ou pela média das 10.

        A projeção vem depois da escolha, então a norma sai 16 nos dois casos. Ver a
        tabela de concordância de sementes no docstring de `PRIOR` — a média só vale
        onde as 10 concordam."""
        v = tabela[nome]
        return cls._projeta(
            (v.mean(0) if semente is None else v[semente]).unsqueeze(0))[0]

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
