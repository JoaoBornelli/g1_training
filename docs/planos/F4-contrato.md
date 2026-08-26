# F4 — contrato de API e de chaves de log

**Escrito ANTES da implementação, de propósito.** Três frentes tocam arquivos disjuntos e
precisam concordar sem se falar. Este arquivo é a única fonte da verdade entre elas; quem
divergir dele está errado, mesmo que funcione.

---

## 1. `knobs.Cadeia` — a tabela

```python
@dataclass
class Cadeia:
    # ⚠ TETO DE 2 ELOS. A tupla de 1 elo é a cadeia curta da F3.
    #   índice 0: (PEGAR,)                 -> 1 elo
    #   índice 1: (REORIENTAR, PEGAR)
    #   índice 2: (PEGAR, CARREGAR)
    #   índice 3: (PEGAR, BOTAR)
    # O `pegar` aparece em TODAS: ele é o eixo, como 1º ou 2º elo. É daí que vem o
    # anti-esquecimento por construção — não se chega ao `botar` sem pegar.
    prob_por_nivel: tuple[tuple[float, ...], ...]   # [7 níveis × 4 cadeias], soma 1,0
    sustenta_pegar_s: float = 0.5
    sustenta_outros_s: float = 0.3
    carregar_s: float = 1.5        # PISO de tempo, não teto
```

As tuplas de elo vivem em `comando.CADEIAS`, junto da numeração dos elos.

## 2. `comando.AlvoCaixaCmd` — os buffers públicos

| atributo | forma | significado |
|---|---|---|
| `self._cadeia` | `(n,) long` | índice em `CADEIAS` |
| `self._passo` | `(n,) long` | 0 ou 1 — posição dentro da cadeia |
| `self._sust` | `(n,) float` | cronômetro de sustentação do elo corrente, em s |
| `self.avancou` | `(n,) bool` | `True` **só no passo** em que o elo avançou |
| `self.fechou` | `(n,) bool` | `True` quando o ÚLTIMO elo da cadeia fechou |

Métodos públicos novos:

```python
def elo_de(self, ids) -> LongTensor          # o elo corrente daqueles envs
def n_elos_da_cadeia(self, ids) -> LongTensor
def forca_avanco(self, ids) -> None          # ⚠ SÓ para o inspetor e o play
```

## 3. `AlvoCaixaCmdCfg` — os campos novos

```python
cadeia_forcada: int | None = None    # índice em CADEIAS. Inspetor e play.
prob_por_nivel: tuple[tuple[float, ...], ...] = ()
sustenta_pegar_s: float = 0.5
sustenta_outros_s: float = 0.3
carregar_s: float = 1.5
```

## 4. As chaves de log — CONTRATO com o `leitura.py`

Elas saem por `self.metrics` do termo de comando, e o `CommandManager.reset` as prefixa
com `Metrics/alvo_caixa/` (`command_manager.py:246`). **Nomes exatos:**

| chave | conteúdo |
|---|---|
| `Metrics/alvo_caixa/sucesso` | 1,0 se a cadeia completou no episódio |
| `Metrics/alvo_caixa/passo_final` | posição alcançada na cadeia (0 ou 1) |
| `Metrics/alvo_caixa/avancos` | quantos avanços de elo houve no episódio |
| `Metrics/alvo_caixa/fatia_cadeia` | 1,0 se o episódio era de cadeia de 2 elos |

⚠ Chave errada **não levanta erro**: a linha só não aparece e o bloco roda sem o portão.
Foi o que aconteceu com o `Policy/mean_noise_std` do `g1_poc`.

## 5. Invariantes que valem para todos

- **O avanço NÃO reseta e NÃO resampleia.** Ele roda dentro do `_update_command`, e é por
  isso que a pose já está fresca ali — ao contrário do reset.
- **Os σ são recalculados a CADA avanço**, contra a pose fresca, chamando o mesmo
  `_recalcula_sigmas` da F3. Com σ fixo de 0,40 rad um pedido de 90° dá `2,0e−7`.
- **Não existe corte de episódio por modo.** Um episódio de cadeia contém dois estados;
  `episode_length_s = 20,0` para todos. Quem gradua o tempo é o sustain por elo.
- **O teto do `BOTAR`** é `min(fundo_da_caixa − folga, botar_topo_teto)`. Sem ele a laje
  nasce DENTRO da caixa.
- **O one-hot acompanha o elo** sem nenhuma mudança: ele já é lido do canal `ELO` por passo.
- Todo `python` roda como
  `PYTHONPATH=<raiz do worktree> /home/joaobornelli/Documents/g1_training/.venv/bin/python`
  — o `.venv` é gitignored e não existe dentro do worktree.
