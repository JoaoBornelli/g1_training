# O `limite_de_junta` ganha limiar POR FAMÍLIA, e o `hip_yaw` começa a pagar a 40°

Data: 2026-09-17. Medição: `docs/memoria/g1-limpo-limite-de-junta-curso-largo.md`.

## O defeito

O `limite_de_junta` mede a fração do CURSO MECÂNICO e tem um limiar único, 0,85. No
`hip_yaw` o curso é ±158°, então a rampa começa a ±134°. A marcha usa ±6°, o carregar
±10°, a referência da IK ±14° e o transiente da pega 36°. O `model_20500` pousa a caixa
com o quadril esquerdo a −144° em média (0,91 do curso) e o direito a −63°. A pose piora
a cada bloco: −0,94 → −1,97 → −2,52 rad em 17999 → 19300 → 20500.

## A decisão do dono

Limiar do `hip_yaw` em ±40°, que é 0,25 do curso. Fica 4° acima do maior uso legítimo
medido (o transiente da pega) e não toca a marcha, o carregar nem a IK.

## A forma, a menor que resolve

A tabela por família passa de `(k, teto)` para `(k, teto, limiar)`. O `limiar` global sai.
O `hip_yaw` vira a quarta família, ao lado de tornozelo, punho/cintura e resto.

| família | resume (bloco 22) | zero |
|---|---|---|
| `hip_yaw` | `(4.0, 0.75, 0.25)`: rampa 40° → 158°, custo 19 no teto | `(20.0, 0.15, 0.25)`: rampa 40° → 64° |
| tornozelo | `(4.5, 0.70, 0.85)` | `(20.0, 0.15, 0.85)` |
| punho, cintura | `(12.0, 0.25, 0.85)` | `(20.0, 0.15, 0.85)` |
| resto | `(20.0, 0.15, 0.85)` | `(20.0, 0.15, 0.85)` |

No resume o teto de 0,75 é a regra do próprio knob: o teto fica ALÉM de onde a junta vive
(0,96 no pico do 20500), senão a derivada é zero e o termo vira imposto. O `k` de 4,0
mantém o custo do teto em ~19, como nas outras famílias.

## Os toques

| arquivo | mudança |
|---|---|
| `knobs.py` `LimiteDeJunta` | campo `hip_yaw`; triplas; `por_padrao()` mapeia `hip_yaw` para a família nova; `limiar` sai; docstring |
| `recompensas.py` `LimiteDeJunta` | `self.limiar` tensor por junta a partir de `p[2]`; `__call__` perde o param `limiar` |
| `env_cfg.py:719` | a linha `"limiar": ...` sai |
| `smoke.py` 641–650 | `_PIOR` ganha `hip_yaw: 0.96`; o laço lê a tripla |
| `g1_limpo_kaggle.ipynb`, `g1_limpo_zero.ipynb` | `_lj` com as quatro famílias em tripla; o assert do `params["limiar"]` sai; o print diz "40° no hip_yaw" |

Contagem prevista: cerca de +12 linhas no `knobs.py` (campo, docstring), 0 no
`recompensas.py`, −1 no `env_cfg.py`, +3 no `smoke.py`, +2 em cada notebook.

## O que NÃO entra

- `hip_roll` (curso assimétrico, repouso a 0,70 do centro) e a referência da IK além da
  rampa no `hip_pitch` e no `waist_pitch`: achados registrados, decisão separada.
- Peso do `forma_postural` e pesos internos das seis grandezas: o limiar de 40° é o freio
  duro; o incentivo fica como está até a próxima medição.
- O RUN do resume continua `bloco22`: o conjunto de termos não muda, só a tabela.

## A verificação

1. Smoke verde, com o check novo da rampa do `hip_yaw`.
2. `registra_juntas` no próximo checkpoint, botar em 0,25: `hip_yaw` no BOTAR abaixo de
   0,40 rad nos dois lados, e a marcha igual à de hoje (`eficiencia_min`, quedas).
3. Se a marcha piorar, o limiar sobe para 0,33 (52°) antes de qualquer outra coisa.
