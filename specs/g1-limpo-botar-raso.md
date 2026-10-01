# G1 limpo — BOTAR raso primeiro (R4), como degrau de currículo

Estado: aprovado pelo dono ("sim, prepara o R4 em paralelo", 01/10). Origem:
`docs/relatorios/2026-09-30-plato-do-botar.md` (R4) e
`docs/relatorios/2026-10-01-monotonia-do-curriculo.md`. Enunciado: §1 "Ordem de aprendizado"
(tarefas aprendidas progressivamente), §3 "Progressão na cadeia", §4.5 BOTAR. Desvio declarado
do §2 "Lajes" (0 a 0,55 m): o degrau raso passa de 0,55 m, e volta ao §2 pelo `s_C`.

## 0. Problema (medido)

Na zero17 retomada com o lote K1b + R1 + K12 (it 4650 a 5180), 40–48% das C chegam ao BOTAR e
o fecho é de ~0,4% por visita (`s_C` 0,0015). As sondas de rede no checkpoint 5000 do resume:
V(alvo) − V(congelado) = −3,1; o ator comanda SUBIR as palmas em todo ponto da descida (+20 cm
com a caixa no alvo); V_CAUDA − V_BOTAR = −3,6. O crítico não vê valor em descer nem em fechar.

A laje do BOTAR nasce no topo da laje do PEGAR ±0,10 m (`comando.py:1838-1852`), p50 ≈ 0,43 m.
Com a caixa carregada a ~0,80 m e meia-aresta 0,10 m, a caixa apoia em `topo + 0,10`: a descida
típica é de ~25 cm, e o melhor caso do §2 (laje de 0,55 m) é de ~15 cm. O robô não explora uma
descida tão longa, nunca vê o fecho, e o crítico esquece o valor do fecho.

## 1. O que muda

Na abertura do BOTAR, em uma fração `f` dos envs, a laje nasce RASA: o topo é sorteado em
`[teto_raso − botar_delta_topo ; teto_raso]`, com `teto_raso = min(teto, botar_raso_topo_max)`.
O `teto` é o guarda físico que já existe (`fundo − botar_folga_laje`, no máximo
`_TOPO_TETO_FISICO`). O resto dos envs segue como hoje (`topo0 ± botar_delta_topo`).

A fração cai com o `s_C`: `f = botar_raso_frac × clamp(1 − s_C / botar_raso_s_c, 0, 1)`.
- `s_C` = 0 → `f` = 0,5. `s_C` ≥ 0,30 → `f` = 0 (volta ao §2).
- Sem o termo de currículo `forma` no cfg, `s_C` = 1,0 e `f` = 0: o R4 desliga (mesma
  convenção do `_resolve_p_c`).
- Equilíbrio esperado: se o raso fecha em ~80% e o fundo nunca fecha, `s_C` fica em ~0,17 e `f`
  em ~0,22. O degrau não some sozinho; só some quando o fundo também aprende. É o desenho.

Valores (knobs, `knobs.Alvo`, ao lado de `botar_delta_topo`):
- `botar_raso_frac` = 0,5.
- `botar_raso_s_c` = 0,30.
- `botar_raso_topo_max` = 0,64 m.
- A profundidade do sorteio raso REUSA `botar_delta_topo` (0,10 m): sem knob novo.

### Por que 0,64 m e não 0,68 m (medido)

Replay cinemático (`mj_kinematics`), DR do R1 (dx só afasta), 3000 aberturas por ponto, sem
dinâmica. Script (temporário): `.../scratchpad/r1_guarda/raso.py`. P(laje toca o robô), em %:

| pose | caixa z | topo 0,45 | 0,55 | 0,62 | 0,68 | 0,75 |
|---|---|---|---|---|---|---|
| fim do CARREGAR da zero17 | 0,795 | 18,7 | 0,0 | 0,0 | 0,0 (teto físico 0,645) | 0,0 (idem) |
| fim do PEGAR da zero17 (rumo −33°) | 0,905 | 62,4 | 24,0 | 2,9 | 0,0 | 0,0 |
| de pé no default | 0,95 | 2,4 | 0,5 | 0,0 | **30,1** | **41,2** |

O robô de pé com a caixa alta toca a laje de 0,68 m em 30% das aberturas (a borda perto da laje
fica na altura da barriga e do quadril). Com o teto do raso em 0,64 m, esse caso fica em ~0%. O
raso também é MAIS seguro que o sorteio de hoje (topo ~0,43): nas duas poses reais, 0,55 a 0,64 m
dá 0 a 3% de toque, contra 19–62% em 0,45 m.

### Limites conhecidos (aceitos)

- A tabela de IK de referência do BOTAR (`ik/ref_botar.npz`) cobre `h = topo + meia` até 0,68 m.
  Com topo de 0,58 a 0,64 m e meia 0,10 m, `h` passa de 0,68 em parte das aberturas, e o
  `forma_postural` usa a última linha da tabela (`recompensas.py:1019`, `clamp`): pelve 0,72 m,
  tronco ~22°. É uma referência inclinada demais para uma laje alta. O peso é pequeno
  (`forma_postural` paga ~0,14–0,20/s por ir aos 24°). Não se estende a tabela neste lote.
- O degrau passa do §2 em até 9 cm. Só enquanto `s_C` < 0,30.

## 2. Mudanças, arquivo a arquivo

1. `g1_limpo/knobs.py`, `class Alvo`, depois de `botar_folga_laje` (~linha 256): três campos,
   com comentário ⚠ curto cada (o porquê e a referência a esta spec):
   `botar_raso_frac: float = 0.5`, `botar_raso_s_c: float = 0.30`,
   `botar_raso_topo_max: float = 0.64`.
2. `g1_limpo/env_cfg.py` (~linhas 562-565, onde `botar_delta_topo` e `botar_folga_laje` passam
   do knob ao cfg do comando): os três campos novos, do mesmo jeito.
3. `g1_limpo/comando.py`
   - `AlvoCaixaCmdCfg` (~linhas 342-346): os três campos, com os mesmos valores default.
   - Função PURA de módulo `frac_botar_raso(s_c: float, frac: float, s_c_alvo: float) ->
     float` = `frac * min(max(1.0 - s_c / s_c_alvo, 0.0), 1.0)`, perto de `resolve_p_c`
     (procure onde ele é definido ou importado). Docstring curta em português. Exporte no
     `__all__` se `resolve_p_c` estiver lá.
   - No bloco `elif elo == BOTAR:` de `_aplica_elo` (~linhas 1830-1852), DEPOIS do cálculo do
     `teto` e do comentário do caso declarado, ANTES de `topo = ...`: sorteie
     `raso = torch.rand(k, device=d) < f`, com `f = frac_botar_raso(s_c, c.botar_raso_frac,
     c.botar_raso_s_c)` e `s_c` lido como em `_resolve_p_c` (`getattr(self._env, "limpo_forma",
     None)`; `float(st["s_C"])` se existir, senão `1.0`). O topo raso é
     `torch.clamp(teto, max=c.botar_raso_topo_max) - c.botar_delta_topo * torch.rand(k,
     device=d)`. O topo final é `torch.where(raso, topo_raso, torch.minimum(topo0 + dtopo,
     teto)).clamp(min=c.prateleira_topo_piso)`. Mantenha o comentário do caso declarado.
     Escreva a métrica `self.metrics["c_botar_raso"][m] = raso.float()`.
   - `__init__`: acrescente `"c_botar_raso"` à tupla de métricas do K12 (~linha 543-546).
   - Comentário ⚠ de 3–4 linhas no ponto da mudança: o porquê (descida longa, crítico sem
     valor de fechar), o desvio do §2 e a volta pelo `s_C`, e a referência a esta spec.
   - NÃO mexa no `dxy`, no `off_recuo`, no alvo, nem no resto do bloco.
4. `g1_limpo/smoke.py`, perto dos checks do lote de 01/10 (procure "R1 (lote 01/10"):
   a. Tabela da função pura: `frac_botar_raso(0.0, 0.5, 0.30) == 0.5`;
      `frac_botar_raso(0.15, 0.5, 0.30) == 0.25`; `frac_botar_raso(0.30, 0.5, 0.30) == 0.0`;
      `frac_botar_raso(1.0, 0.5, 0.30) == 0.0`; compare com tolerância 1e-9.
   b. Os três campos existem em `knobs.Alvo` e em `AlvoCaixaCmdCfg`, com o mesmo valor, e o
      `make_env_cfg` os entrega ao comando (compare `cfg.commands[...]` com o knob, no padrão
      dos checks de `botar_delta_topo` que já existem; procure por esse nome no smoke).
   c. Check de fonte: em `_aplica_elo`, `torch.clamp(teto, max=c.botar_raso_topo_max)` aparece
      DEPOIS de `teto = torch.clamp(` e ANTES de `topo = torch.where(`.
   d. A chave `"c_botar_raso"` está no `__init__` do comando (estenda o check das seis chaves
      do K12 para sete, sem afrouxar nada).
   e. PROCURE no smoke qualquer check existente que assuma que o topo da laje do BOTAR é
      `topo0 ± botar_delta_topo` ou que fica abaixo de `prateleira_topo_teto` (0,55). Se
      existir, ajuste-o para o novo comportamento SEM afrouxá-lo (ex.: condicione ao ramo
      não raso) e reporte o que mudou.
   Cada check tem de PODER FALHAR (sem `try/except` que engula erro).

## 3. Regras para quem implementa

- Toque só nos 4 arquivos acima. Nenhum comando `git`.
- Não rode o smoke, nem trecho dele, nem env do mjlab, nem import do pacote. A única execução
  permitida é, no FIM, `/home/joaobornelli/Documents/g1_training/.venv/bin/python -m py_compile`
  nos 4 arquivos.
- Estilo dos arquivos: comentários em português, ⚠ nos avisos, nomes em português.
- Não mude número da tabela por estado, peso, nem outro knob.
- Reporte: diff resumido por arquivo, contagem de linhas antes → depois, e todo ponto em que a
  spec não bateu com o código (com o que você decidiu).

## 4. Verificação

- Revisão do diff contra esta spec e o dono roda `python -m g1_limpo.smoke`.
- Depois do resume, no log:
  - `Metrics/alvo_caixa/c_botar_raso` ≈ `0,5 × (1 − s_C/0,30)` das aberturas do BOTAR;
  - `s_C` sai de ~0: esse é o sinal de sucesso do R4;
  - `c_chegou_botar` e `load` sobem; `Episode_Reward/contato_tronco` não sobe;
  - sondas de rede (V(alvo) − V(congelado), comando das palmas) num checkpoint novo.
- Se o `s_C` não sair de 0,01 em ~200 iterações com o R4, a hipótese "descida longa demais"
  está refutada: o próximo passo é K4 (piso do `σ_alcance`) e R3 (`upright` por estado).

## 5. Linhas estimadas (antes → depois)

| Arquivo | Antes → depois |
|---|---|
| comando.py | 2497 → ~2520 |
| knobs.py | 1603 → ~1613 |
| env_cfg.py | 973 → ~977 |
| smoke.py | 6257 → ~6280 |
