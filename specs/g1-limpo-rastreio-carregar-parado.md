# G1 limpo — rastreio ×1 no CARREGAR parado, em toda cadeia (K1b)

Estado: spec, aguarda "implementa". Data: 01/10. Substitui o R2
(`specs/g1-limpo-rastreio-carregar-elo.md`, commitado). Origem:
`docs/relatorios/2026-10-01-monotonia-do-curriculo.md` (V1, D5, conserto K1b).
Enunciado: §3 "Progressão na cadeia", §3 "Sem estátua", §4.4, §5 linha ANDAR ("rastreio
precisa de derivada só no movimento"), §6 ("multiplicador por posição na cadeia:
inobservável pelo crítico", rejeitado).

## 0. Problema (verificado)

O R2 põe o rastreio em ×1 só no elo CARREGAR aberto da cadeia C
(`carregar_elo_aberto` = `elo == CARREGAR ∧ cadeia == 2 ∧ ¬fechou`, `comando.py:167`).
Dois defeitos:

1. **Desvio pago.** A C que falha o `perto` no fim de uma espera vai à cauda CARREGAR com
   `fechou = True` (`comando.py:804-839`). A cauda fica fora do flag e paga ×3,5: 10/s a
   mais que o elo. Com o fecho raro de hoje, desviar rende até ~+22 a mais que avançar em
   2 s (γ 0,99). Nas fases 3–4 a cauda anda, mas herda twist zero por ~2,9 s, e o desvio
   segue pago.
2. **Multiplicador inobservável.** O flag depende da cadeia e do `fechou`. Nas fases 1–2 o
   elo da C e a cauda parada de B têm a mesma observação para o ator e o crítico
   (`observacoes.py:86-141`), com pesos ×1 e ×3,5. O §6 rejeita esse desenho.

## 1. O que muda

O peso dos dois rastreios no estado CARREGAR passa a depender do REGIME, que é
observável, e não da cadeia:

- estado CARREGAR ∧ regime parado → `knobs.Cadeia.rastreio_carregar_parado` = 1,0;
- estado CARREGAR ∧ regime andando → coluna CARREGAR da tabela (3,5), como hoje;
- todo outro estado → a tabela, como hoje.

Regime parado = o mesmo `standing` do `PosturaPorElo` (`recompensas.py:129-141`):
`limpo_twist_zerado > 0,5` ou `‖cmd_xy‖ + |cmd_wz| < walking_threshold` (0,05).

E a cauda da cadeia C desviada fica SEMPRE parada (`_carregar_parado = True`) em toda
fase. A cauda de B e de R segue a regra de fase de hoje (fase 1 parada; fase 2 90% parada;
fase ≥ 3 andando).

Efeito por caso:

| Caso | Hoje (R2) | Depois |
|---|---|---|
| Elo CARREGAR aberto da C | ×1 | ×1 |
| Cauda da C desviada | ×3,5 (parada ou andando) | ×1 (sempre parada) |
| Cauda parada de B/R (fases 1–2) | ×3,5 | ×1 |
| Cauda andando de B/R, comando de marcha | ×3,5 | ×3,5 |
| Cauda andando, envs `standing` do fabricante (comando zero) | ×3,5 | ×1 |

Contas do verificador: o BOTAR vence o desvio por 16 a 26 em 2 s. O fecho do PEGAR na F2
cai de +11,9..+24,9 para +4,6..+12,8, ainda positivo.

## 2. Mudanças, arquivo a arquivo

1. `g1_limpo/recompensas.py`
   - Função pequena nova, perto do `PosturaPorElo`: `regime_parado(env, command_name,
     walking_threshold) -> torch.Tensor` (bool), com o cálculo de `standing` que hoje está
     dentro de `PosturaPorElo.__call__` (linhas 129-141): soma das normas do comando, menor
     que o limiar, OU `limpo_twist_zerado > 0,5` quando o atributo existe.
   - `PosturaPorElo.__call__` passa a chamar `regime_parado`. O resultado numérico tem de
     ficar IGUAL (o comentário ⚠ "O REGIME VEM DO ESTADO" vai junto para a função).
   - `PesoPorEstado`: o parâmetro `carregar_elo` vira `carregar_parado` (o peso no regime
     parado). No `__call__`, se `self._w_par is not None`:
     `w = torch.where((env.limpo_estado == ESTADO_CARREGAR) & regime_parado(env,
     kw["command_name"], 0.05), torch.full_like(w, float(self._w_par)), w)`.
     O limiar 0,05 vem do mesmo lugar que o `walking_threshold` do `pose`
     (`env_cfg.py:724`); se ali for literal, crie uma constante e use-a nos dois.
   - Docstring do `PesoPorEstado`: troque o parágrafo ⚠ `carregar_elo` por um parágrafo ⚠
     `carregar_parado` (regime observável, só os dois rastreios, só o estado CARREGAR,
     referência a esta spec e ao §6).
2. `g1_limpo/comando.py`
   - REMOVER `carregar_elo_aberto` (linhas 155-167) e o nome no `__all__` (linha 73).
   - REMOVER `env.limpo_carregar_elo` do `__init__` (linhas 610-614) e a escrita em
     `_aplica_espera` (linhas 875-879), com os comentários.
   - No bloco da cauda de `_aplica_espera` (linhas ~826-837), DEPOIS do `if fase <= 1 /
     elif / else` que escreve `_carregar_parado`: a cadeia C desviada fica parada em toda
     fase. Uma linha, mais um comentário ⚠ curto com o porquê (desvio pago; spec K1b):
     `self._carregar_parado[vira_carregar[self._cadeia[vira_carregar] == 2]] = True`.
   - Atualize o comentário ⚠ "A cadeia C (2) desviada vai SEMPRE à cauda" para dizer que
     a cauda dela é parada.
3. `g1_limpo/knobs.py`
   - `Cadeia.rastreio_carregar_elo` → `rastreio_carregar_parado: float = 1.0`, com o
     comentário ⚠ reescrito (regime parado, observável).
   - Docstring do `PesoPorEstado`: o item "EXCEÇÃO ao 3,5" passa a descrever o regime
     parado em toda cadeia, com os números da seção 1.
4. `g1_limpo/env_cfg.py`, laço 3i (linhas ~825-827): a chave vira
   `_t.params["carregar_parado"] = k.cadeia.rastreio_carregar_parado`. Comentário
   atualizado.
5. `g1_limpo/smoke.py`
   - REMOVER os checks da tabela-verdade de `carregar_elo_aberto` (~1640-1650) e da ordem
     de escrita de `limpo_carregar_elo` (~1652-1656).
   - TROCAR o check dos parâmetros (~1399-1407): a chave é `carregar_parado`, igual a
     `k.cadeia.rastreio_carregar_parado`, só nos dois rastreios; ausente nos outros oito.
   - TROCAR o check da aritmética do wrapper (~1659-1684), sem env real, no mesmo padrão
     sintético de hoje. Quatro envs: `limpo_estado` = [CARREGAR, CARREGAR, CARREGAR,
     BOTAR]; `limpo_twist_zerado` = [0, 1, 0, 1]; comando `twist` = [[0,5; 0; 0],
     [0; 0; 0,3], [0; 0; 0,01], [0; 0; 0]]. Esperado com o knob: [3,5; 1,0; 1,0; coluna
     BOTAR]. Com 0,25: [3,5; 0,25; 0,25; coluna BOTAR]. Sem a chave: [3,5; 3,5; 3,5;
     coluna BOTAR]. O env 2 prova o `limpo_twist_zerado` com wz de rumo acima do limiar;
     o env 3 prova o ramo do comando abaixo do limiar.
   - Check NOVO: a cauda da cadeia C desviada fica com `_carregar_parado = True` em toda
     fase. Siga o padrão de env real que já existe perto da linha 3704 (`_tv._carregar_parado`);
     se esse padrão não alcançar a fase 3, use a mesma fixação de fase que o smoke já
     usa para o currículo de cadeia. O check tem de PODER FALHAR.
   - Os checks que já cobrem o `PosturaPorElo` (regime parado por `limpo_twist_zerado`,
     ~1795-1805) continuam passando sem mudança; NÃO os afrouxe.
6. `specs/g1-limpo-rastreio-carregar-elo.md`: uma linha de estado no topo, "Substituída
   por `g1-limpo-rastreio-carregar-parado.md` (01/10)". Nada mais.

## 3. Fora do escopo

- O twist herdado na entrada da cauda andando (K13). Ele continua: o trecho de ~2,9 s com
  vx = vy = 0 e laço de rumo paga ×3,5 quando |wz| ≥ 0,05. Spec própria depois do K12.
- Qualquer número da tabela por estado, e todo outro peso.

## 4. Regras para quem implementa

- Toque só nos 6 arquivos acima. Nenhum comando `git`.
- Não rode o smoke, nem trecho dele, nem env do mjlab, nem import do pacote. A única
  execução permitida é, no FIM, `/home/joaobornelli/Documents/g1_training/.venv/bin/python
  -m py_compile` nos 5 arquivos `.py`.
- Siga o estilo dos arquivos: comentários em português, ⚠ nos avisos, nomes em português.
- Reporte o diff resumido por arquivo e a contagem de linhas antes → depois.

## 5. Verificação

- Revisão do diff contra esta spec (revisor Opus separado).
- O dono roda `python -m g1_limpo.smoke`.
- Depois do resume: `Episode_Reward` dos dois rastreios cai; `s_cauda` não cai mais de 10%
  na fase 2; com o K12, a fração de C desviada no fim de cada espera cai.

## 6. Linhas estimadas

| Arquivo | Antes → depois |
|---|---|
| comando.py | 2479 → ~2461 |
| recompensas.py | 1354 → ~1358 |
| knobs.py | 1600 → 1600 |
| env_cfg.py | 973 → 973 |
| smoke.py | 6243 → ~6235 |
| Saldo | ≈ −22 |
