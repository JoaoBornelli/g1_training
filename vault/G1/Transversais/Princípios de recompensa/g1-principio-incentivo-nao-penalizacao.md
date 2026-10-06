---
tipo: achado
pai: "[[Princípios de recompensa]]"
estado: ativa
verificado: 2026-10-06
resumo: "principio de desenho de recompensa no g1_training, dado pelo usuario em 2026-08-25: toda acao precisa de INCENTIVO proprio, senao ficar parado custa menos que explorar; e nao se controla por PENALIZACAO — incentiva-se a acao correta em vez de punir a transgressao"
---
# g1-principio-incentivo-nao-penalizacao (original)

Corpo original da memória `g1-principio-incentivo-nao-penalizacao`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Enunciado pelo usuario em **2026-08-25** como aprendizado do `g1_multitask`, ao desenhar o
`g1_limpo`. Vale para todo termo de recompensa do projeto.

## As duas regras

**REGRA 1 — toda acao precisa de incentivo proprio.** Sem ele, ficar parado custa menos que
explorar, e o robo nao executa a tarefa.

**REGRA 2 — nao controlar por penalizacao.** E melhor um incentivo para a acao correta que
uma penalizacao pela transgressao.

## O caso POSITIVO: o `pegar` do g1_poc funcionou porque tinha VARIOS incentivos

Nao um. Varios, e todos positivos e continuos:

| incentivo | o que ele paga |
|---|---|
| `staged` = reaching x (1 + bringing) | aproximar a mao, e depois trazer a caixa |
| `squeeze` = tanh(min(F_esq, F_dir)/F_ref) | FORCA de aperto, contínua |
| `unload` = 1 − F_apoio/(m·g) | aliviar o apoio, contínua — a ponte |
| `precise_ori` | reduzir o Δθ |
| `sustentacao` | tempo NA condicao |
| `postura_ereta` | altura da pelve durante a pega |

Resultado medido: reorientava a caixa BEM e erguia ao alvo.
Ver g1-modulo-novo-heranca-comportamental.

## A instancia mais limpa da regra 2: o platô do grasp

O `grasp` BOOLEANO travou o `pegar` por **22 mil iteracoes**. O que destravou foi o
`unload`, que e o MESMO objetivo expresso de forma CONTINUA (força de apoio de 9,7 N para
0 N). Booleano nao tem gradiente; contínuo tem.
Ver g1-pegar-plato-do-grasp-booleano.

## O caso NEGATIVO: a locomocao do g1_poc nao andou porque ANDAR nao tinha incentivo

Medido na it 997 do bloco5:

```
POSITIVO que um ROBO IMOVEL colhe:
  upright 0,983 + pose 0,726        = 1,709 /s  = 63% de TODO o positivo
  track_lin (parado, por geometria) = 0,275     = 27% do termo de graca
  track_ang (parado)                = 0,856     contra 0,2027 MEDIDO tentando

PENALIDADE que domina:
  action_rate                       = 1,108 /s  = 87% de TODAS as penalidades
                                                  (piso de ruido puro = 1,227 /s)
```

Ou seja: a estatua era o otimo. E os tres termos que moldam a marcha valem **ZERO com o pe
plantado**, portanto nao ha gradiente para levantar o pe:

| termo | valor parado | por que |
|---|---|---|
| `feet_clearance` = Σ\|h−0,10\| x ‖v_pe‖ | 0 | v_pe = 0 |
| `feet_swing_height` | 0 | gateado em `first_contact` |
| `foot_slip` | 0 | precisa contato + velocidade |

**Os tres sao PENALIDADES, e todos moldam o passo DEPOIS que ele existe. Nenhum o CRIA.**
E a regra 1 e a regra 2 falhando juntas.
Ver g1-track-lin-parado-colhe-27pct, g1-poc-recompensa-loco-prefere-parar.

## Consequencia de desenho para o g1_limpo

1. Para CADA comportamento desejado, perguntar: existe um termo POSITIVO e CONTINUO que
   cresce quando o robo comeca a faze-lo, a partir do repouso? Se a derivada no repouso e
   zero, o comportamento nao nasce — ele so e moldado depois de nascer por acidente.
2. Andar precisa de um incentivo que a estatua NAO possa colher. `track_linear_velocity`
   sozinho nao serve: parado ja rende 27%.
3. Preferir contínuo a booleano. Booleano = platô.
4. Penalidade nao ensina o que fazer. Ela so limita COMO fazer, e so vale depois de o
   comportamento existir.
5. Auditar a razao positivo/negativo por tarefa antes de rodar. Se a maior penalidade e
   maior que o maior incentivo, o otimo local e nao agir.

## Ja aplicado antes desta sessao, e confirmado agora

g1-poc-auditoria-incentivos achou tres instancias da mesma classe: `reaching` com σ fixo
(sorte no nivel 3+), `postura_ereta` 3x fraca, e o `botar` **sem ninguem que pague soltar**.
Esse ultimo e a regra 1 pura: a acao "soltar" nao tinha incentivo.

Relacionado: g1-air-time-o-incentivo-desligado e a regra;
g1-poc-auditoria-incentivos, g1-pegar-plato-do-grasp-booleano,
g1-track-lin-parado-colhe-27pct, g1-poc-recompensa-loco-prefere-parar,
g1-postura-penhasco-std-standing, g1-modulo-novo-heranca-comportamental,
g1-lift-progress-alvo-e-a-caixa.
