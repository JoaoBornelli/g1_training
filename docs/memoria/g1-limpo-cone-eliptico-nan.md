# Por que o cone elíptico diverge para NaN

Investigado em 05/10/2026, só leitura de código e de issues públicas. Nenhum teste rodado.

## Os dois incidentes

| Data | Config | Sintoma |
|---|---|---|
| 15/07 | `elliptic`, `impratio=10`, Newton, 10 iterações, 20 de busca | NaN "no reset parcial" |
| 17/09 | idem, Colab, 8192 envs, treino do zero | `check_nan` da rsl_rl no grupo `actor` na iteração ~755; log limpo antes |

Os dois usaram o par `elliptic` + `impratio=10`. O elíptico nunca foi testado com impratio baixo.

## Três defeitos conhecidos, que se somam

### 1. Hessiano indefinido perto do ápice do cone (mujoco_warp #1657)

No cone elíptico o solver de Newton monta um Hessiano por contato. O código de 3.10 e 3.11
clampa `t` (norma do resíduo tangencial) e `t³` separadamente em `MJ_MINVAL`. Para
`MJ_MINVAL < t < cbrt(MJ_MINVAL)`, só o termo de posto 1 é suprimido, até 10⁹×, e o Hessiano
perde a semidefinição positiva. O Cholesky falha e produz NaN. O caso que reproduz: contato
quase parado, caixa de 37,5 g sobre a mesa, `t ≈ 6e−6`.

O robô de pé e a caixa pousada são exatamente contatos quase parados. E o `impratio` alto
endurece a tangencial, o que deixa `t` ainda menor.

Fix: PR #1660, merged em 23/09/2026, depois do 3.11.0. **A versão instalada (3.10.0.3)
tem o defeito**: `solver.py:2495-2496, 2672-2673, 2846-2847` ainda fazem
`ttt = wp.max(t*t*t, MJ_MINVAL)`.

### 2. Busca de linha esgotada com impratio 10 e 20 iterações (mujoco #3628)

Com `elliptic`, `impratio=10`, `ls_iterations=20`, Newton e `implicitfast`, o solver às vezes
devolve um `qacc` não convergido, com gradiente ~1e2 contra tolerância 1e−8, num ponto de
sela de onde não sai: reiniciar dali faz 0 iterações. O motivo: o custo ao longo da direção
de busca muda de curvatura onde o contato cruza a borda do cone, e 20 avaliações não bastam.
No Menagerie `unitree_g1` foram 16 casos divergentes em 20 000 soluções. Aberto em
27/09/2026; o PR #3651 ainda estava aberto em 05/10.

O mjlab usa `iterations=10, ls_iterations=20` em todas as tarefas
(`velocity_env_cfg.py:450`); o padrão do MuJoCo é 100/50.

### 3. O warm-start do Warp propaga o NaN e o reset não o limpa

O MJWarp sempre inicia o solver com `qacc_warmstart`; o MuJoCo C compara com `qacc_smooth`
antes (documentação do MJWarp). Um NaN em `qacc_warmstart` de um env entra no solve
seguinte. O reset do mjlab escreve `qpos` e `qvel` e não toca em `qacc_warmstart` (grep em
`mjlab/`: o campo só aparece no `nan_guard`). Por isso o NaN "no reset parcial": o env
reseta e volta a divergir no primeiro passo.

Agravantes: float32 (tolerância clampada a 1e−6; a documentação avisa que iterações e
atrito pequeno são sensíveis) e o `impratio` que divide o `invweight` tangencial por 10
(`constraint.py:4267-4275`), piorando o condicionamento.

## O que isso diz para a v3

- O piramidal com impratio 2 não passa pelo Hessiano do cone (defeito 1) nem pela busca de
  linha do cone (defeito 2). Subir o impratio no piramidal só muda o `invweight`
  (`constraint.py:4289-4292`), e isso não foi testado acima de 2.
- O elíptico volta a ser candidato só com mujoco_warp ≥ versão que inclua o PR #1660
  (depois de 23/09) e com `ls_iterations` maior que 20, ou com o PR #3651 merged. Mesmo
  assim, com impratio ≤ 5 e uma sonda de 100 iterações primeiro.
- Independente do cone: o `nan_guard` do mjlab (`sim.py:494`) existe e grava um dump por
  env no primeiro NaN. Ligar na v3 dá o diagnóstico que faltou em 17/09.

## Fontes

- https://github.com/google-deepmind/mujoco_warp/issues/1657
- https://github.com/google-deepmind/mujoco_warp/pull/1660
- https://github.com/google-deepmind/mujoco/issues/3628
- https://github.com/google-deepmind/mujoco/pull/3651
- https://mujoco.readthedocs.io/en/latest/mjwarp/ (warm-start, float32, tolerância)
- https://mujoco.readthedocs.io/en/latest/changelog.html (3.10.0: convergência em float32, issue #2313)
