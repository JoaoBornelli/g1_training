# G1 limpo — profundidade do BOTAR como currículo próprio (R4b)

Estado: aprovado pelo dono ("prepara o R4b", 01/10). Substitui a fração rasa do R4
(`specs/g1-limpo-botar-raso.md`). Enunciado: §1 "Ordem de aprendizado", §2 "Lajes" (altura de
cada laje sorteada de 0 a 0,55 m, de forma INDEPENDENTE), §4.5 BOTAR. Resolve o D17
(`docs/relatorios/2026-10-01-monotonia-do-curriculo.md`).

## 0. Problema (medido)

- A laje do BOTAR herda o topo da laje do PEGAR ±0,10 m (`comando.py:1838-1852`). A do PEGAR
  sai do nível: `U(topo_min[nível]; 0,55)`, `topo_min` = (0,55; 0,45; 0,30; 0,15; 0,04; …)
  (`knobs.py:319`). Com o nível médio em ~1,8, laje do BOTAR abaixo de 0,15 m aparece em
  menos de ~3% das aberturas (estimativa). O dono confirmou no play: "ele não chega a aprender
  a botar a caixa nas lajes no chão; sempre é sobre uma mesa".
- O R4 resolveu a entrada (`s_C` de 0 a 0,16 em 700 its), mas só desloca uma FRAÇÃO das
  aberturas para a laje rasa; o resto segue herdando o PEGAR.
- Plays de 01/10 (laje = a da pega): `6050` a 0,55 m para a caixa 2 cm acima e só solta na
  troca para ANDAR; `5800` a 0,30 m não pousa.

## 1. O que muda

Na abertura do BOTAR, TODA laje do BOTAR é sorteada por uma profundidade que cresce com o
`s_C`, sem herdar o topo do PEGAR:

    teto_raso = min(teto, botar_raso_topo_max)            # teto = fundo − folga (guarda física, já existe)
    a         = avanco_prof_botar(s_C, botar_prof_s_c)    # clamp(s_C / botar_prof_s_c, 0, 1)
    prof      = botar_delta_topo + a × (teto_raso − prateleira_topo_piso − botar_delta_topo)
    topo      = (teto_raso − prof × U(0; 1)).clamp(min = prateleira_topo_piso)

- `s_C` = 0: `prof` = 0,10 m, topo em [0,54; 0,64] (o raso de hoje).
- `s_C` = 0,16 (hoje): `a` = 0,32, `prof` ≈ 0,26, topo em [0,38; 0,64].
- `s_C` ≥ `botar_prof_s_c` (0,50): `prof` cobre até o chão, topo em [0,04; 0,64].
- Sem o termo de currículo `forma` no cfg, `s_C` = 1,0 → profundidade CHEIA (todas as alturas).
- O `topo0` (laje do PEGAR) e o `dtopo` deixam de ser usados no BOTAR.
- Se o `s_C` cai, a profundidade encolhe: o currículo se regula pelo sucesso do próprio BOTAR.
  O nível do PEGAR não muda (decisão do dono: sem nível por tarefa).

Knobs (`knobs.Alvo`):
- SAEM `botar_raso_frac` e `botar_raso_s_c`.
- ENTRA `botar_prof_s_c: float = 0.50`.
- FICA `botar_raso_topo_max` = 0,64 (o teto do sorteio; medido: acima disso o robô de pé toca a
  laje) e `botar_delta_topo` = 0,10 (agora é a profundidade mínima).

Postura (decisão do dono, 01/10): no BOTAR o robô usa a IK da mesma maneira que no PEGAR, e a
pose é praticamente a mesma para a mesma altura. JÁ É o código: os dois elos leem a mesma
`ik/ref_botar.npz`, pela altura da caixa no PEGAR e pela do alvo no BOTAR
(`recompensas.FormaPostural`). Nada muda aqui.

## 2. Métricas (sem peso)

Sai `c_botar_raso`. Entram, no mesmo `self.metrics` do comando:
- `c_botar_topo`: o topo sorteado, escrito na abertura do BOTAR (`[m] = topo`). No log,
  `c_botar_topo / c_chegou_botar` = topo médio das aberturas.
- `c_fechou_botar`: 1 no fecho terminal de um env da cadeia C (onde `sucesso` é escrito,
  `comando.py:~1755`, filtrando `self._cadeia[terminal] == 2`).
- `c_fecho_topo`: `env.limpo_topo` no mesmo instante, nos mesmos envs. No log,
  `c_fecho_topo / c_fechou_botar` = topo médio dos fechos. Se ele descer junto com o das
  aberturas, a laje baixa está aprendendo.

## 3. Mudanças, arquivo a arquivo

1. `g1_limpo/knobs.py` (`class Alvo`, ~linhas 257-264): troque os dois campos e reescreva o
   comentário ⚠ (o porquê, a fórmula curta, a referência a esta spec e ao §2/D17).
2. `g1_limpo/env_cfg.py` (~linhas 566-568): entregue `botar_prof_s_c` no lugar dos dois.
3. `g1_limpo/comando.py`
   - `frac_botar_raso` (~linha 265) vira `avanco_prof_botar(s_c: float, s_c_alvo: float) ->
     float` = `min(max(s_c / s_c_alvo, 0.0), 1.0)`. Atualize o `__all__` e a docstring.
   - `AlvoCaixaCmdCfg` (~linhas 355-357): troque os dois campos por `botar_prof_s_c: float =
     0.50`, com o comentário.
   - No bloco `elif elo == BOTAR:` (~linhas 1838-1876): o `topo0`, o `dtopo`, o sorteio `raso`
     e o `torch.where` saem; entra a fórmula da seção 1. MANTENHA o guarda `teto` e o
     comentário do caso declarado (o `clamp(min=prateleira_topo_piso)` continua). Reescreva o
     comentário ⚠ do R4 para o R4b (3-5 linhas). Escreva `self.metrics["c_botar_topo"][m] =
     topo`. NÃO mexa no `dxy`, no `off_recuo`, no alvo.
   - Na tupla de métricas do `__init__` (~linha 559): `c_botar_raso` sai; `c_botar_topo`,
     `c_fechou_botar`, `c_fecho_topo` entram.
   - No fecho terminal (~linha 1755): escreva as duas métricas de fecho para os envs de
     `terminal` com `self._cadeia == 2`.
4. `g1_limpo/smoke.py`
   - Tabela da função pura: `avanco_prof_botar(0.0, 0.5) == 0`, `(0.25, 0.5) == 0.5`,
     `(0.5, 0.5) == 1`, `(1.0, 0.5) == 1` (tolerância 1e-9). Substitui a do `frac_botar_raso`.
   - Igualdade dos knobs: `botar_prof_s_c` e `botar_raso_topo_max` em `knobs.Alvo`, no
     `AlvoCaixaCmdCfg` e entregues pelo `make_env_cfg` (substitui o laço dos três campos).
   - Check de fonte em `_aplica_elo`: `teto = torch.clamp(` vem antes de `avanco_prof_botar(`,
     e `topo0` NÃO aparece no bloco do BOTAR (o BOTAR não herda mais o PEGAR).
   - Chaves de métrica: troque `c_botar_raso` pelas três novas (o check passa a nove chaves).
   - Check 7 (~linhas 5805-5818): o par de checks do R4 sai. Entra: `topo <=
     min(teto, botar_raso_topo_max) + 1e-6` e `topo >= prateleira_topo_piso − 1e-6` em todos os
     envs. Calcule o limite inferior esperado pelo `s_C` que o env do smoke tem (se não houver
     `limpo_forma`, profundidade cheia) e cheque `topo >= teto_raso − prof − 1e-6`. Não afrouxe
     os outros checks do item 7.
   Cada check tem de PODER FALHAR.

## 4. Regras para quem implementa

- Toque só nos 4 arquivos. Nenhum comando `git`. Não rode smoke, env do mjlab nem import do
  pacote. No fim, só `/home/joaobornelli/Documents/g1_training/.venv/bin/python -m py_compile`
  nos 4 arquivos. Faça grep por `botar_raso_frac`, `botar_raso_s_c`, `frac_botar_raso` e
  `c_botar_raso` em `g1_limpo/` e garanta zero ocorrências.
- Estilo dos arquivos: comentários em português, ⚠ nos avisos. Não mude peso nem tabela.
- Reporte diff resumido por arquivo, contagem de linhas antes → depois, e todo ponto em que a
  spec não bateu com o código (com o que você decidiu).

## 5. Verificação

- Revisão do diff contra esta spec; o dono roda o smoke.
- No log, depois do resume: `c_botar_topo / c_chegou_botar` desce com o `s_C`;
  `c_fecho_topo / c_fechou_botar` desce junto; o `s_C` pode cair no começo (a profundidade
  de hoje passa de [0,54; 0,64] em 23% e herdado no resto para [0,38; 0,64] em todas) e deve
  voltar a subir.
- Play: laje baixa (0,15 e 0,30) com o checkpoint novo.
- `RUN` segue `zero17`: nenhum peso ou tabela muda.

## 6. Linhas estimadas

comando.py ~0 (troca o sorteio, +6 das métricas); knobs.py −1; env_cfg.py −1; smoke.py ~0.
