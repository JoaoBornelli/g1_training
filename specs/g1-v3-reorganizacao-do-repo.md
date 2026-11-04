# Reorganização: um repositório, quatro blocos

Data: 2026-10-19. Status: rascunho aguardando aprovação.

## Intenção

Hoje o treino, o lidar e a visão vivem em repositórios separados. Cada um carrega a sua cópia do robô, da cena e dos imports. As cópias divergem. O objetivo é um repositório único (`g1_training`, branch `g1_v3`) com um núcleo do G1 e quatro blocos que importam desse núcleo. Todo desenvolvimento novo acontece aqui, com scripts de teste ao lado do código.

Atende a `specs/g1-limpo-comportamento.md` de forma indireta: o robô que treina é o mesmo robô que o lidar, a visão e a simulação usam.

## Estrutura alvo

| Pasta | Conteúdo | Origem |
|---|---|---|
| `nucleo/` | Robô (`g1.xml`), cena (caixa, prateleira, depósito), camada de sensores (câmera, lidar, IMU) | `g1_training/common`, `g1_limpo/cena.py`, `lidar_sim/sensores` |
| `treino/` | Recompensa, currículo, knobs, checkpoints de referência, Kaggle/Colab, specs, planos, relatórios, vault | `g1_limpo`, `g1_multitask`, `g1_poc`, `g1_residual`, `specs`, `docs`, `vault` |
| `lidar/` | Percepção, SLAM, ESKF, odometria, `estudo_lidar` | `lidar_sim/percepcao`, `lidar_sim/estudo_lidar` |
| `visao/` | Treino do detector e do orientador, dataset, câmera D435i, estudos de CNN e memória das faces | `box-vision` (com histórico) |
| `simulacao/` | Registro de juntas, `autonomo`, `ao_vivo`, `pilota`, `play`, `volta`, cenários | `lidar_sim/autonomo`, `lidar_sim/*.py`, `box-vision/ao_vivo*.py`, `robo_ao_vivo.py` |
| `testes/` | Um subdiretório por bloco, `pytest` | `lidar_sim/tests`, scripts `confere_*.py` |
| `arquivo/` | Não entra no Git. Lixo preservado (CSV soltos, logs, `.pt` sem nome, `dados.zip`) | raiz de `g1_training` e de `g1-box-vision` |

## Decisões do usuário

- Um repositório só, quatro pastas de bloco dentro do `g1_v3`.
- O `g1_v3` é o centro. Onde o `lidar_sim` e o `g1_v3` divergem, o `g1_v3` vence.
- O histórico de commits do `box-vision` fica preservado dentro do `g1_v3`.
- Os arquivos de teste soltos saem da raiz.

## Fatos medidos em 2026-10-19

- `lidar_sim` é um worktree do mesmo repositório Git. Ele aponta para o `master` (snapshot de 30/07).
- `g1_training/` (núcleo): 0 diferenças entre treino e lidar. `g1_limpo`: 40 diferenças. `g1_multitask`: 23 diferenças.
- Código só do `lidar_sim`: `autonomo/`, `percepcao/`, `sensores/`, `estudo_lidar/`, `pilota.py`, `play.py`. Ele depende do `g1_limpo` antigo e precisa de port para o v3.
- Visão: 10 arquivos `.py`, sem import do treino.
- Os XML de `runs/**/cena.xml` são saída de execução. A fonte da cena é `g1_limpo/cena.py`.
- `lidar_sim/runs/` pesa 3,2 GB. O `.venv` pesa 6 GB em cada repo.

## Não objetivos

- Não renomear pacotes Python na fase 1. O import `g1_training.common` e `g1_limpo` continua válido por `pyproject`. A renomeação fica para uma fase posterior.
- Não mexer em `unitree-g1-sim`, `BFM-Zero`, `camera_Test`. O usuário ainda não decidiu o destino deles.
- Não apagar nada. O lixo vai para `arquivo/`.
- Não mudar recompensa, currículo nem knob. Só localização de arquivo e port de import.

## Anti-objetivos (falha mesmo que "funcione")

- O treino v3 perde reprodutibilidade (um import quebra e ninguém percebe).
- O histórico do `box-vision` se perde ou o `.git` incha com blobs pesados (`.pt`, `dados.zip`).
- Um arquivo útil some no meio da limpeza.

## Restrições

- O branch `g1_v3` está 6 commits à frente do remote e tem 53 arquivos sujos. A reorganização começa de uma árvore limpa e commitada.
- Cada passo é um commit atômico e reversível. O repositório roda `pytest` antes e depois de cada passo.
- Convenção do projeto: spec em `specs/`, plano em `docs/planos/`, relatório em `docs/relatorios/`.

## Decisões minhas (o usuário ainda não aprovou)

- Importar o `box-vision` com `git subtree add --prefix=visao` (ou `git filter-repo` para mudar o prefixo antes). O construtor mede o `.git` do `box-vision` antes e remove blobs grandes se necessário.
- Um `.venv` único na raiz do `g1_v3`. Os outros três viram descartáveis só depois de os testes passarem no novo.
- `runs/` (3,2 GB) sai do Git e vai para `arquivo/runs/`, com ponteiro no README do bloco `simulacao/`.
- Um `README.md` por bloco, com o que entra, o que sai e como rodar o teste.
- O `lidar_sim` (worktree) é removido só depois da verificação final.

## Deixado ao construtor

- Ordem exata dos commits.
- Se `nucleo/` vira pacote Python ou só pasta de dados e módulos (decidir pelo custo de import).
- Nomes finais dos arquivos movidos, desde que cada movimento preserve o histórico (`git mv`).

## Pendências abertas

- Destino de `unitree-g1-sim`, `BFM-Zero`, `camera_Test` e `Documents/Project/box-vision`.
- O que fazer com os 10 branches `exp/*` depois da reorganização.
