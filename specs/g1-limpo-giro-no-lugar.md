# G1 limpo — girar no lugar, com e sem caixa

Estado: spec, aguarda "implementa". Data: 30/09. Enunciado: §2 (ANDAR) e §4 (CARREGAR).

## 0. Resumo

O robô gira pouco no lugar porque o treino sorteia pouco esse caso. Esta spec sobe a
fração do ramo `turning` de 0,10 para 0,30. Ela também cria uma métrica sem peso para a
deriva em xy durante o giro, separada entre com caixa e sem caixa. Nenhuma recompensa
muda. O conserto da deriva, se a métrica mostrar deriva, vem numa spec separada.

## 1. O problema, medido no código

| Fato | Onde |
|---|---|
| O ramo `turning` manda lin = 0 e \|wz\| em [0,2; teto] | `comando.py:2288-2312` |
| `rel_turning_envs` = 0,10, e o `standing` tem precedência: a fração real é 9% dos sorteios do twist | `knobs.py:544` |
| O twist só vale no ANDAR e na cauda CARREGAR que anda; o ANDAR tem ~30% das transições | `knobs.Forma`, alvo 0,30 |
| O giro no lugar fica com ≈ 0,09 × 0,30 ≈ 2,7% das transições | conta |
| O teto de wz é ±0,5 até a it 5000. Depois, o estágio 1 do `command_vel` põe ±1,6 | `env_cfg.py:525-531`, `velocity_env_cfg.py:407` |
| O `common_step_counter` vai no checkpoint, portanto o estágio 1 abre na it 5000 também num resume | `runner.py:6` |
| A deriva em xy no giro só paga pelo `track_linear_velocity`, σ = 0,5 m/s: 0,1 m/s custa 4% do termo | `velocity_env_cfg.py:282` |
| Nenhuma métrica separa os envs `turning` | `metricas.py` |

## 2. Objetivo

- O robô gira no lugar até 1,6 rad/s, sem caixa e com caixa.
- Durante o giro, a base não anda em xy.
- O log mostra as duas coisas, com e sem caixa.

Fora do escopo: mudar recompensa, mudar o teto de wz, mudar o currículo de comando.

## 3. A mudança

1. `knobs.Marcha.rel_turning_envs`: 0,10 → 0,30. Fração real 0,27 dos sorteios do twist,
   ≈ 8% das transições. Na cauda CARREGAR que anda (fase 3 em diante), ~27% das caudas
   giram no lugar com a caixa.
2. Duas métricas sem peso, na mesma classe `metricas.media_por_estado`, com uma grandeza
   nova `vel_xy_giro`: a norma da velocidade linear da base em xy, no frame da base, só
   nos passos em que o env é `turning` no twist.
   - `giro_deriva_sem_caixa`: estado ANDAR.
   - `giro_deriva_com_caixa`: estado CARREGAR, fora do CARREGAR parado.
   A máscara `turning` entra como um parâmetro `so_giro=True` da classe, lido de
   `twist.is_turning_env`. O resto da classe (soma, contagem, preenchimento dos envs
   sem passo, `reset`) fica igual.
3. `smoke.py`: um check de que as duas métricas existem e de que `so_giro` zera fora
   dos envs `turning`. O dono roda o smoke.
4. Resume do último checkpoint. `RUN` segue `zero17`.

Linhas estimadas: knobs +0 (troca de valor); metricas +15; smoke +8; spec nova.

## 4. Por que não quebra o treino

- Observação, ação, rede e recompensa ficam iguais. O checkpoint carrega sem mudança.
- O sorteio do twist só muda a fração dos ramos. Os envs `heading` caem de ~72% para
  ~56% dos sorteios; o `forward` e o `standing` ficam como estão.
- O controlador da forma mede a duração dos episódios e reajusta o sorteio sozinho.
- As métricas não têm peso. Métrica não entra no gradiente.

## 5. Riscos

- Marcha reta com laço de rumo recebe menos amostra. A leitura é `razao_marcha` e
  `error_vel_xy` no log.
- O giro abre em ±1,6 rad/s na it 5000, junto com `lin_vel_x` até 2,0 m/s. A cauda que
  anda também recebe esse envelope. A leitura é `caixa_largada` e `s_cauda` depois da it
  5000.

## 6. Verificação

| Quando | O que ler | Esperado |
|---|---|---|
| No resume | `Curriculum/command_vel/ang_vel_z_max` | 0,5 antes da it 5000, 1,6 depois |
| 300 its depois do resume | `giro_deriva_sem_caixa` | cai; a meta é < 0,05 m/s |
| Fase 3 em diante | `giro_deriva_com_caixa` | valor presente e em queda |
| Sempre | `razao_marcha`, `error_vel_xy` | sem piora maior que 10% |

Se a deriva ficar acima de 0,10 m/s por 500 its, abre-se a spec do conserto: o kernel do
linear só nos envs `turning`, com a derivada medida no ponto de operação antes do peso.

## 7. Decisões

- D1 (30/09, dono): itens 1 e 3 da proposta; o item 3 já acontece na it 5000.
- D2 (30/09, dono): a métrica de deriva entra junto.
- D3 aberta: implementar.
