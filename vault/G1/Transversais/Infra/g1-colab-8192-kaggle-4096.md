---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "O Colab roda 8192 envs do g1_limpo numa GPU de 16 GB; a Kaggle, tambem com 16 GB, so roda 4096. O discriminador e o hospedeiro, nao a VRAM"
cluster: transversais
---
Fato do dono, 2026-09-11, medido nas duas plataformas.

| hospedeiro | GPU | `NUM_ENVS` que roda |
|---|---|---|
| Colab | 16 GB | **8192** |
| Kaggle | 16 GB (T4) | **4096** |

A VRAM nominal nao explica a diferenca, e nao vale inventar um limiar de memoria. O notebook escolhe por `COLAB`, nao por `nvidia-smi`.

Isso tambem fecha uma duvida antiga: a `bloco8` rodou no Colab com 8192 envs a 8,54 s/iter (23 022 passos/s), e a GPU era de 16 GB — nao um A100.

**How to apply:** `NUM_ENVS = 8192 if COLAB else 4096`. Se um dia a Kaggle passar a segurar 8192, muda-se a tabela, nao a regra.

Relacionado: g1-leitura-log-e-kaggle, g1-treino-gargalo-banda-nao-flops
