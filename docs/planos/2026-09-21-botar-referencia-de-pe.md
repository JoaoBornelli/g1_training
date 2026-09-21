# BOTAR: a referência de orientação passa a ser a vertical

Plano escrito em 2026-09-21. Nada implementado. Ele sai da medição em
`docs/memoria/2026-09-21-botar-alinhado-pune-endireitar.md`, feita no `model_9100` da run
`zero02` com `registra_juntas` em três alturas de laje.

## O defeito, em uma frase

O BOTAR congela a direção pedida na normal da caixa **tombada nas mãos do robô** e depois
exige que ela não gire. A caixa chega tombada 29,7° (laje 0,55) e 52,2° (laje 0,35) e
termina a 0,0°, porque a laje a nivela. O giro que o `alinhado` reprova é o
**endireitamento**, e o `sigma_ori` trava no piso de 0,20 rad porque o congelamento zera
o próprio erro que define o σ.

## Restrição do dono

O `alinhado` fica. O robô real bota a caixa numa orientação pré-determinada. Mas esse
requisito só vale quando o REORIENTAR sair de inerte (`knobs.py:1049`), e hoje nada no
currículo ensina guinada.

⚠ POSIÇÃO DO DONO, 2026-09-21: **a guinada em torno de z não importa por enquanto.** A
caixa assentada na mesa com o lado de cima apontando para cima já é o resultado bom. O
estágio 1 abaixo não é um meio-caminho — ele é o requisito inteiro de hoje.

| estágio | referência do BOTAR | o fecho exige |
|---|---|---|
| este plano | a vertical do mundo | a caixa de pé |
| REORIENTAR vivo | a pose de destino da tarefa | de pé e com a face certa |

---

## Passo 0 — medir antes de mudar

O `registra_juntas` troca de fase por TEMPO, não pelo fecho. Os 29,7° e 52,2° saem de um
roteiro, não de um fecho real do PEGAR. Se no treino a caixa já chegar direita ao BOTAR,
este plano encolhe.

`metricas.media_por_estado` já recebe `grandeza` e já tem os casos `pelve_z` e
`tronco_incl` (`metricas.py:473-478`). Acrescentar um caso:

```python
elif grandeza == "caixa_incl":
    up = quat_apply(env.scene["box"].data.root_link_quat_w, self.ez)
    x = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
```

e um slot `caixa_na_pega` com `estados=(ESTADO_PEGAR_COM, ESTADO_BOTAR)`, no molde do
`tronco_na_pega` (`metricas.py:165`). Cerca de 8 linhas, e ela fica de sentinela depois.

**Decisão:** se o tombamento no fecho real do PEGAR ficar abaixo de 10°, o gargalo do
BOTAR é outro e este plano não é a prioridade.

---

## Passo 1 — o regime da face vira ternário

`_face_viva` é um `bool[n]` (`comando.py:438`). Ele **vira** um `int8[n]`, sem buffer
novo, com três constantes de módulo:

    FACE_VIVA      = 0   REORIENTAR: face marcada -> direção do robô, viva
    FACE_CONGELADA = 1   ANDAR, PEGAR, CARREGAR: face marcada -> normal congelada
    FACE_DE_PE     = 2   BOTAR: eixo z da CAIXA -> vertical do mundo

## Passo 2 — `_aplica_elo` escreve o regime, e congela menos

Em `comando.py:1544-1545`, as duas linhas atuais viram:

```python
elo = self._elo[ids]
self._regime_face[ids] = torch.where(
    elo == REORIENTAR, FACE_VIVA,
    torch.where(elo == BOTAR, FACE_DE_PE, FACE_CONGELADA))
self._congela_face(ids[self._regime_face[ids] == FACE_CONGELADA])
```

O BOTAR sai do congelamento. É esta linha que mata a referência tombada.

## Passo 3 — `_atualiza_face` escolhe o eixo e a direção

Em `comando.py:1844-1856`, o vetor medido deixa de ser sempre a face marcada:

```python
r = self._regime_face[ids].unsqueeze(-1)
de_pe = r == FACE_DE_PE
eixo_b = torch.where(de_pe, self._ez_b.expand(k, 3), self._face_b.expand(k, 3))
medido = quat_apply(self.caixa.data.root_link_quat_w[ids], eixo_b)
desejada = torch.where(r == FACE_VIVA, viva,
            torch.where(de_pe, self._ez_w.expand(k, 3), self._face_alvo_w[ids]))
```

`normal_w` passa a ser `medido`. As linhas do `ANG` e do `GIRO` (`comando.py:1859-1871`)
**não mudam**.

---

## O que NÃO muda, e é o ponto do plano

| lugar | por que fica igual |
|---|---|
| `_recalcula_sigmas` (`comando.py:1808`) | a mesma linha; ela passa a ler um `ANG` real em vez do zero fabricado |
| `precise_ori` (`recompensas.py:632`) | a mesma linha; lê o mesmo canal `ANG` |
| o fecho (`comando.py:1315`, `1356`) | a mesma linha; `tol_ang_deg` = 25° segue |
| `PesoPorEstado` e os pesos | nada |
| o canal `GIRO` | já carrega eixo × ângulo; passa a apontar para a vertical sozinho |

O `GIRO` importa: ele é a observação que diz **para que lado** girar. Sem ele um MLP sem
memória não aprende a reorientar. Ele já existe e já está no molde do ator.

---

## O gradiente, condição por condição, dentro do BOTAR

O fecho pede `perto ∧ alinhado ∧ apoiada`, sustentados por `sustenta_outros_s` = 0,5 s.

| condição | quem molda | σ | no ponto de operação |
|---|---|---|---|
| `perto` (0,10 m) | `precise_pos` | fixo 0,18 m | a 0,10 m: valor 0,734, derivada 0,045 por cm — **vivo** |
| `alinhado` (25°) | `precise_ori` | o tombo na abertura | na abertura: valor e⁻¹ = 0,368, derivada 0,025 por grau — **vivo depois deste plano** |
| `apoiada` (0,5·m·g) | `load` | contínuo na força | sobe de 0 a 1 conforme a laje assume o peso — **vivo dentro de `perto`** |
| sustentar 0,5 s | `renda_congelada` | — | paga no fecho |

### O que o `precise_ori` passa a valer

Com a coluna 2,0 do BOTAR e o peso base 1,0:

| momento | hoje | depois |
|---|---|---|
| abertura do BOTAR (tombo 29,7°) | 2,00 | 0,74 |
| pouso (tombo 0,0°) | 0,0004 | 2,00 |
| **saldo de endireitar** | **−2,00** | **+1,26** |

O mesmo comportamento que o robô já executa passa de multa máxima a prêmio. A virada é de
cerca de 3,3 no termo.

### Por que o σ sai do piso sozinho

Na abertura do BOTAR o erro deixa de ser zero por construção e passa a ser o tombo real.
`sigma_ori = (ANG × 1,0).clamp(min = 0,20)` devolve 0,52 rad e 0,91 rad nos dois casos
medidos. O kernel nasce em e⁻¹ = 0,368, que é a convenção de σ do projeto inteiro:
"σ é a distância inicial, e ali o kernel vale 0,368". Ninguém mexe no `sigma_ori`.

⚠ Caixa que chega direita dá σ no piso e kernel perto de 1,0, com derivada quase nula. É
o correto: aquele eixo já está resolvido e não precisa de gradiente.

---

## Contagem

| arquivo | antes → depois |
|---|---|
| `comando.py` | ~+10 / −2 linhas; um buffer TROCADO, nenhum acrescentado |
| `metricas.py` | +8 linhas (passo 0, sentinela permanente) |
| `smoke.py` | +3 verificações |

## O smoke

1. O regime por elo: REORIENTAR = VIVA, BOTAR = DE_PE, os outros três = CONGELADA.
2. Caixa de pé no BOTAR devolve `ANG` ≈ 0; caixa tombada 90° devolve `ANG` ≈ π/2.
3. `_congela_face` NÃO é chamado para o BOTAR — o `_face_alvo_w` dele não é lido.

## Riscos declarados

1. **A guinada fica livre** até o REORIENTAR sair de inerte. É a decisão do dono de
   2026-09-21, e o segundo estágio a recupera pelo mesmo campo.
2. **O desenho de depuração** lê `_face_b` direto (`comando.py:1915`). Ele passa a
   desenhar o vetor errado no BOTAR. Corrigir junto, ou o visualizador mente.
3. **O passo 0 pode derrubar o plano.** Se a caixa chegar direita ao BOTAR no treino, a
   mudança continua correta mas deixa de ser a prioridade.
