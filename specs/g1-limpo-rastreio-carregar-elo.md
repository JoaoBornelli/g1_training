# G1 limpo — rastreio ×1 no CARREGAR aberto da cadeia C (R2)

Estado: Substituída por `g1-limpo-rastreio-carregar-parado.md` (01/10). (Antes: aprovado pelo dono, "implementa", 01/10.) Origem: `docs/relatorios/2026-09-30-plato-do-botar.md`
(achado A4, opção R2 b). Enunciado: §3 "Sem estátua" e "Progressão na cadeia"; §4.4 Fecho.

## 0. Problema (medido)

No elo CARREGAR da cadeia C o comando de velocidade é zero (`comando.py:771`, `1465-1479`). A
coluna CARREGAR da tabela por estado paga o rastreio ×3,5 (`knobs.py:1311-1312`), pensado
para a cauda de B/R que ANDA. Parado com comando zero, o robô rastreia perfeitamente e
recebe 2,0 × 3,5 + 2,0 × 3,5 = 14/s. Fechar o CARREGAR leva à espera (rastreio ×1, os sete
termos em 0) e o rastreio não entra na renda congelada. Na conta descontada (γ 0,99),
fechar vale −0,8 a +5,4 contra ficar. Com rastreio ×1 no elo aberto, +15 a +21.

## 1. O que muda

Só no elo CARREGAR ABERTO da cadeia C — `elo interno == CARREGAR ∧ cadeia == 2 ∧ ¬fechou` —
os dois rastreios (`track_linear_velocity`, `track_angular_velocity`) usam o multiplicador
`knobs.Cadeia.rastreio_carregar_elo = 1.0` no lugar da coluna CARREGAR da tabela (3,5).

Ficam como estão:
- a cauda CARREGAR de B e R (fechou = True desde o PEGAR), parada ou andando: 3,5;
- a cadeia C desviada para a cauda (fechou = True): 3,5;
- todos os outros estados, termos e pesos; a tabela `knobs.PesoPorEstado` não muda.

## 2. Mudanças, arquivo a arquivo

1. `g1_limpo/comando.py`
   - Função PURA nova, logo depois de `estado_de_recompensa` (~linha 152), no mesmo estilo
     (docstring curta em português, com o porquê e a referência a esta spec):
     `carregar_elo_aberto(elo, cadeia, fechou) -> torch.Tensor` (bool) =
     `(elo == CARREGAR) & (cadeia == 2) & ~fechou`. O `2` é a cadeia C, como em
     `comando.py:1136` (`eh_c = cad == 2`); comente isso.
   - Exportar o nome no `__all__` do módulo (~linha 77).
   - No `__init__`, logo depois de `env.limpo_estado = torch.zeros(...)` (~linha 595):
     `env.limpo_carregar_elo = torch.zeros(n, device=d)` (float), com um comentário ⚠ de 2–3
     linhas no estilo vizinho.
   - Em `_aplica_espera`, logo DEPOIS de `self._env.limpo_estado.copy_(estado_de_recompensa(...))`
     (~linha 855-856), IN-PLACE como os vizinhos:
     `self._env.limpo_carregar_elo.copy_(carregar_elo_aberto(self._elo, self._cadeia, self.fechou).float())`.
     Mesma fase temporal do `limpo_estado`. NÃO escrever em `_update_command`.
2. `g1_limpo/knobs.py`
   - Em `class Cadeia` (~linha 1035), um campo novo com comentário ⚠ curto:
     `rastreio_carregar_elo: float = 1.0`. NÃO pôr em `Recompensa` nem em `PesoPorEstado`
     (os campos deles têm de ser termos de recompensa: `env_cfg.aplica_pesos` e o laço 3i
     afirmam isso).
   - No docstring de `class PesoPorEstado`, junto do item "Rastreio no CARREGAR = 3,5"
     (~linha 1269), um item novo: a exceção do elo aberto da cadeia C, o número medido
     (14/s, fechar ≈ 0) e a referência a esta spec.
3. `g1_limpo/env_cfg.py`, laço 3i (~linhas 808-826), dentro do `for`, depois de
   `_t.params["tabela"] = ...`:
   `if _nome in ("track_linear_velocity", "track_angular_velocity"):`
   `    _t.params["carregar_elo"] = k.cadeia.rastreio_carregar_elo`
   (o `assert "func" not in ... and "tabela" not in ...` da linha anterior continua).
4. `g1_limpo/recompensas.py`, `class PesoPorEstado` (~linhas 325-400)
   - `__init__`: `self._w_elo = cfg.params.get("carregar_elo")` (None nos outros oito termos).
   - `__call__`: acrescentar `carregar_elo=None` à assinatura e ao `del`. Calcular
     `w = self._t[env.limpo_estado]`; se `self._w_elo is not None`:
     `w = torch.where(env.limpo_carregar_elo > 0.5, torch.full_like(w, float(self._w_elo)), w)`;
     depois `r = self._f(env, **kw) * w`. O ramo do `rumo` fica como está.
   - Docstring: um parágrafo ⚠ curto com a exceção (só os dois rastreios, só o elo aberto
     da C, lido de `env.limpo_carregar_elo`, publicado na mesma fase do `limpo_estado`).
5. `g1_limpo/smoke.py` — checks novos, perto dos checks existentes que citam (procure os
   textos) "`estado_de_recompensa` só assume valores em range(10)" (~linha 1597) e
   "a tabela dos dois rastreios é a do `knobs.PesoPorEstado`" (~linha 1395). Cada check
   tem de PODER FALHAR (sem `try/except` que engula erro):
   a. Tabela-verdade de `CMD.carregar_elo_aberto` sobre todas as combinações de
      elo ∈ range(len(CMD.ELOS)), cadeia ∈ {−1, 0, 1, 2} e fechou ∈ {False, True}:
      verdadeiro SÓ em (CMD.CARREGAR, 2, False).
   b. `cfg.rewards[n].params["carregar_elo"] == k.cadeia.rastreio_carregar_elo` para os dois
      rastreios, e `"carregar_elo" not in cfg.rewards[n].params` para os outros oito termos
      de `k.peso_por_estado`.
   c. Ordem no fonte de `CMD.AlvoCaixaCmd._aplica_espera`:
      `index("limpo_estado.copy_(") < index("limpo_carregar_elo.copy_(")`, e
      `"limpo_carregar_elo" not in inspect.getsource(CMD.AlvoCaixaCmd._update_command)`.
   d. Aritmética do wrapper SEM env real: instancie `RC_.PesoPorEstado` com um cfg
      sintético (use `types.SimpleNamespace(params={...})` com `func` = uma função que
      devolve `torch.ones(n)`, `tabela` = `k.peso_por_estado.track_linear_velocity`,
      `carregar_elo` = `k.cadeia.rastreio_carregar_elo`) e um env sintético com
      `device="cpu"`, `limpo_estado` = `[ESTADO_CARREGAR, ESTADO_CARREGAR, ESTADO_BOTAR]` e
      `limpo_carregar_elo` = `[0., 1., 1.]`. Espera-se `[3.5, 1.0, 1.0×coluna BOTAR]`, isto é
      o flag só troca o peso onde vale 1, e a coluna BOTAR do rastreio é 1,0. Siga o padrão do
      env sintético que o smoke já usa perto da linha 716. Um segundo wrapper sem
      `carregar_elo` com o mesmo env devolve `[3.5, 3.5, 1.0]`.
   e. O check existente "o `PesoPorEstado` NÃO injeta `nome_do_comando`..." (~linha 1454)
      continua passando sem mudança; NÃO o afrouxe.

## 3. Regras para quem implementa

- Toque só nos 5 arquivos acima. Nenhum comando `git`.
- Não rode o smoke, nem trecho dele, nem env do mjlab, nem import do pacote entre edições.
  A única execução permitida é, no FIM, `/home/joaobornelli/Documents/g1_training/.venv/bin/python
  -m py_compile` nos 5 arquivos editados.
- Siga o estilo dos arquivos: comentários em português, ⚠ nos avisos, nomes em português.
- Não mude nenhum número da tabela por estado nem nenhum outro peso.
- Reporte: o diff resumido por arquivo e a contagem de linhas antes → depois de cada arquivo.

## 4. Verificação

- Revisão do diff contra esta spec (revisor separado).
- O dono roda `python -m g1_limpo.smoke`.
- Depois do resume: métrica por episódio C "chegou ao BOTAR" (R6) e o `Episode_Reward` dos
  dois rastreios na fase 4, que deve cair.

## 5. Linhas estimadas

comando +12, knobs +6, env_cfg +2, recompensas +8, smoke +40.
