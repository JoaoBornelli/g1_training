# g1_training

Treino RL (mjlab / MuJoCo Warp) do Unitree G1 (29 DOF, base flutuante) pra
manipulação — pegar e levantar uma caixa (bimanual), como habilidade central
de uma futura biblioteca de skills.

## Estrutura

Skills = habilidades de alto nível, cada uma com vários treinos/fine-tunes:

```
g1_training/
├── base_env.py          # a INVARIANTE: contrato de obs/ação (shape fixo -> warm-start
│                         # entre skills), física, comando, fundação de equilíbrio.
├── common/               # blocos reusados por todas as skills (robô, caixa/mesa,
│                         # observações, comando, eventos, terminações, rewards de fundação)
├── skills/
│   ├── stand/            # ficar de pé parado
│   │   ├── env.py        # monta o env a partir da base + knobs
│   │   ├── knobs.py       # dataclasses dos números tunáveis
│   │   └── configs/       # 1 arquivo por treino salvo (baseline, step_recovery, ...)
│   ├── lift/              # pegar a caixa e levar ao alvo
│   │   ├── env.py, rewards.py, knobs.py, configs/
│   └── place/              # futuro
├── rl_cfg.py
├── smoke.py               # roda local (CPU), sem GPU: valida registro + 1 step de cada skill
└── play.py                # visualiza um checkpoint treinado no viewer
```

## Fine-tuning (knobs)

Cada treino é uma instância de `LiftKnobs`/`StandKnobs` salva em
`skills/<skill>/configs/<AAAA_MM_DD_nome>.py` (só os campos que mudam vs. os
defaults em `knobs.py`). `configs/__init__.py` aponta `ACTIVE` pro treino que
está rodando agora.

- **Voltar a um treino antigo:** trocar 1 linha em `configs/__init__.py`.
- **Puxar um valor de outro treino:** importar o config antigo direto.
- **Comparar dois treinos:** `git diff` entre os arquivos de config.

## Warm-start entre skills

Todas as skills compartilham o MESMO `experiment_name` (`g1_lifting_box`, em
`rl_cfg.py`) e o MESMO shape de obs/ação (garantido por `base_env.py`) —
`--agent.resume --agent.load-run <run-da-skill-anterior>` transfere o
checkpoint. Checkpoints ficam no Google Drive, não neste repo.

## Rodando

```
python smoke.py          # local, CPU, sem GPU — valida antes de sincronizar pro Colab
python play.py CAMINHO/model_999.pt --task Lift   # viewer, sem GPU
```

Treino roda no Google Colab (GPU). Sincronizar este repo lá e, no notebook:

```python
import g1_training  # registra as tasks (Stand / Stand-Step / Lift)
```
