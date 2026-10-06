---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
---
# g1-auditoria-mjlab-bloco3 (original)

Corpo original da memória `g1-auditoria-mjlab-bloco3`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Verificado 2026-08-06, branch `exp/curriculo-teste`. Os itens "de projeto" da auditoria
contra o mjlab. **Nenhum sobreviveu à medição.** O que apareceu foi outra coisa.

## A2 — `condim=3` nas palmas: REJEITADO, medido

Alegação: sem atrito torsional, o `reorientar` treina contra física errada.

`condim=3` de fato exclui o termo torsional (o coeficiente 0,02 fica inerte). **Mas não
importa nesta geometria.** Palma e caixa são os dois `mjGEOM_BOX` (palma 7 × 1,6 × 9 cm,
caixa 20 cm), e contato caixa-caixa gera **4 pontos**. O atrito tangencial nesses pontos
deslocados já produz o torque restaurador pela alavanca de ~4 cm.

Medido, caixa prensada contra a palma, torque de 0,50 N·m em torno da normal do aperto:

| N normal | condim 3 | condim 4 | razão |
|---:|---:|---:|---:|
| 20 N | 2,3° | 2,0° | 1,11× |
| 50 N | 2,1° | 2,0° | 1,08× |
| 100 N | 1,8° | 1,8° | 1,01× |

A diferença encolhe conforme o aperto aperta. ⚠️ Isto vale porque a palma é PLACA PLANA.
Com pad esférico ou de ponto único, `condim=4` passaria a importar muito.

## A7 — histórico de observação: ADIADO, e o probe é inconclusivo

Fatos verificados:
- **`joint_torque` é `actuator_force`** (`common/observations.py:51`), o torque
  COMANDADO pelo PD. A carga não entra direto — entra pelo erro de posição que ela
  induz. Em regime, `kp·Δq = torque de carga`, então `joint_pos` E `joint_torque`
  codificam a carga num frame só; histórico só ajudaria no transiente.
- `history_length` existe em `ObservationTermCfg` (`observation_manager.py:57`) e
  **nenhuma task oficial do mjlab usa em termo de observação** — só em sensor de contato.
- Muda a largura da obs → Categoria C, invalida checkpoint.

⚠️ **Não consegui medir.** Sem política que segure a caixa, a ação nula domina: span de
9,5 N·m entre 1 e 5 kg contra desvio ENTRE ENVS de 8,8, e não-monotônico. Mesmo artefato
que a calibração já registrou para `arm_vel` e `table_contact`.

**Critério para reabrir:** se a taxa de sucesso cair conforme o eixo `peso` sobe enquanto
o eixo `altura` continua subindo, a inferência de carga é a suspeita.

## A8/A9 — sensores de contato: REJEITADO

`soft_landing` usa `sensor_data.force` INSTANTÂNEA, não `force_history`
(`velocity/mdp/rewards.py:366`). O único consumidor de histórico é o
`self_collision_cost`, e o nosso `self_collision` já tem `history_length=4` herdado.
Batemos o fabricante exatamente: `self_collision` = 4, `feet_ground_contact` = 0.

Sensor corpo↔chão não tem precedente no fabricante, e `fell_over` a 70° já cobre.

## Terreno irregular: ADIADO

`unitree_g1_rough_env_cfg` existe, mas exige `terrain_scan`, gerador de terreno e a
terminação `out_of_terrain_bounds` — e a prateleira mora no chão. Projeto, não ajuste.
Volta junto com o yaw largo do `reset_base`, que só faz sentido em terreno irregular
(ver g1-auditoria-mjlab-bloco2).

## Os dois "médios" que eu tinha listado: os DOIS eram falsos

**`giro = 360°` não é ângulo.** `commands.py:216` trata o último nível como
`topo = giro >= len(LEVELS)-1`, e a face alvo passa a ser sorteada em `TOPO_FUNDO` em vez
de `LATERAIS`. É código para "a face pode ser o topo ou o fundo", como o `tasks.py` já
documentava. Não é 360° literal.

**Altura 0,00 não quebra o `apoiada`.** Com o topo da prateleira exatamente no nível do
terreno, a caixa mantém **4 contatos com o slab** (e mais 4 com o terreno). O
`box_support` continua disparando, então o critério de sucesso do `reorientar` é
alcançável. Medido em modelo isolado varrendo o topo de 0,15 a 0,00.

## O que era real: colisão de chave de log

`vel_mdp.soft_landing` escreve `env.extras["log"]["Metrics/landing_force_mean"]` com
nome **fixo** (`velocity/mdp/rewards.py:373`). Instalamos dois termos — pés e impacto do
tronco contra a mesa. Os nomes dos TERMOS são distintos; a chave de LOG era a mesma, e o
`RewardManager` avalia na ordem de inserção. O `soft_landing_table` roda depois, então o
número publicado como "força de pouso" era sempre o do TRONCO.

Zero efeito em recompensa (os pesos são −1e−5 e −1e−4). É observabilidade, e é o que se
lê entre blocos. Corrigido com `R.soft_landing_rotulado`, que renomeia a chave depois da
chamada: `Metrics/landing_force_pes` e `Metrics/landing_force_mesa`.

Ver g1-auditoria-mjlab-bloco2, g1-multitask-orcamento-igual-por-tarefa.
