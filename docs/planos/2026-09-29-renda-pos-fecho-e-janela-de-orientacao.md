# Renda pós-fecho e janela de orientação (S1–S5)

Data: 2026-09-29. Base: `exp/g1-limpo-v2` @ `829ee03`. RUN nova: `zero12`, do zero.
Diagnóstico: `docs/relatorios/2026-09-29-tombo-da-caixa-achados-e-correcoes.md`.

Regra geral: **alterar código existente; nenhum termo novo; nenhum peso subido.**
Os implementadores NÃO rodam git, NÃO rodam simulador, smoke nem treino, e tocam SÓ os
arquivos do seu lote. Falha fora do lote é reportada, não consertada.

## Lotes (posse de arquivo, disjuntos)

| lote | dono | arquivos | itens |
|---|---|---|---|
| A | Sonnet | `g1_limpo/knobs.py`, `g1_limpo/recompensas.py` | S1, S3 |
| B | Sonnet | `g1_limpo/comando.py` | S2, S4 |
| C | Sonnet | `g1_limpo/kaggle/g1_limpo_zero_kaggle.ipynb`, `g1_limpo/kaggle/g1_limpo_zero_colab.ipynb` | S5 |
| D | main | `g1_limpo/smoke.py` | ajustes do smoke (teste não se delega) |

## S1 — tirar a renda de ficar parado do PEGAR_COM (lote A, `knobs.py`)

Na `PesoPorEstado` (colunas: ANDAR ESP_SEM ESP_COM REOR_SEM REOR_COM PEG_SEM PEG_COM
CARREGAR BOTAR CAUDA; PEG_COM é o índice 6):

- `track_linear_velocity` (~1263): índice 6 `1.0` → `0.0`.
- `track_angular_velocity` (~1264): índice 6 `1.0` → `0.0`.
- `pose` (~1265): índice 6 `4.0` → `1.0`.
- NÃO mexer em `forma_postural`, nem em nenhuma outra coluna ou linha.

Docstrings a reescrever (mesmo tamanho ou menor):
- `knobs.py` ~1226-1230 ("Rastreio nas outras colunas ..."): PEGAR_COM passa para a lista
  dos zeros. Motivo, uma frase: pairar no alvo pagava 25,0/s contra 16,5 a 23,2 depois do
  fecho (MEDIDO no `model_1600` da `zero11`); o freio `velocidade_por_regime` já contém a
  velocidade de junta parado, e o fator `rumo` cobre o giro.
- `knobs.py` ~1235-1240 ("`pose` em PEGAR_COM e ESPERA_COM = 4"): passa a ser só ESPERA_COM;
  no PEGAR_COM é 1, como no PEGAR_SEM; a `faixa_de_pose` cobra o punho junta a junta, e o
  ×4 encarecia 4× o punho, que é a junta que endireita a caixa.
- Invariante que o texto tem de citar: "o que falta fazer paga pelo menos o que já foi feito"
  (spec tabela-por-estado): com S1, pairar (18,6/s) fica abaixo da ESPERA (23,2) e do
  CARREGAR (~22,6).

## S3 — janela de orientação nos termos de aproximação (lote A, `recompensas.py`)

1. Criar o helper logo ANTES de `def precise_ori` (~622), movendo para ele o docstring do
   HÍBRIDO que hoje está no `precise_ori` (a medição do `model_1750` e os valores 1 / 0,61 /
   0,32 / 0):
   ```python
   def _alinha(env, nome: str, so_de_pe: bool = False) -> torch.Tensor:
       """<docstring do híbrido, movido do precise_ori> + um parágrafo:
       `so_de_pe=True` devolve 1 fora do regime `FACE_DE_PE`. No REORIENTAR (`FACE_VIVA`)
       a direção pedida aponta da caixa para o robô, e `staged`/`precise_pos` ali não têm
       mão no produto: o robô ganharia ANDANDO EM VOLTA da caixa. O `precise_ori` chama
       sem o flag e mede a face, que é a tarefa do REORIENTAR."""
       from g1_limpo.comando import ANG, FACE_DE_PE
       t = _t(env, nome)
       erro = env.command_manager.get_command(nome)[:, ANG]
       a = 0.5 * (1.0 - erro / math.pi) + 0.5 * torch.exp(
           -(erro / t.sigma_ori.clamp(min=1e-6)) ** 2)
       return torch.where(t._regime_face == FACE_DE_PE, a, torch.ones_like(a)) if so_de_pe else a
   ```
2. `precise_ori`: o corpo vira `return _alcancar(env, nome_do_comando) * _alinha(env, nome_do_comando)`.
   Docstring curto: "`alcançar × _alinha`. A face pedida no lugar." e o gate por `alcançar`.
3. `staged` (~608): `return alcanca * (1.0 + traz * _alinha(env, nome_do_comando, so_de_pe=True))`.
   Acrescentar ao docstring um parágrafo "⚠ × `_alinha` no `trazer` (29/09)": erguer a caixa
   tombada deixava de custar; de 17° a 60° a soma passa de +0,10 a −2,43/s, e erguer reto
   rende +1,41/s. O `alcançar` fica fora da janela: aproximar a mão da caixa na mesa não
   depende do tombo.
4. `precise_pos` (~619): `return torch.exp(-(d / sigma) ** 2) * _alinha(env, nome_do_comando, so_de_pe=True)`.
   Docstring: o aceite exige a caixa no alvo E de pé; no CARREGAR isto é o único gradiente
   contra o tombo (coluna `precise_ori` = 0 ali).
5. Assinaturas públicas (`staged`, `precise_pos`, `precise_ori`) NÃO mudam.
6. Invariante: com a caixa de pé (`ANG = 0`) os três valem exatamente o que valiam antes.

## S2 — quem não avança vai à cauda (lote B, `comando.py`)

- `_aplica_espera`, ~742: `cauda = acabou & ~tem_prox & ~ja_em_cauda`
  → `cauda = acabou & ~avanca & ~ja_em_cauda`.
- `avanca` é o tensor já calculado acima (com o portão `perto | _forcado` para quem já
  pegou). NÃO remover o portão, NÃO remover `_forcado`.
- Comentário novo acima da linha (≤ 5 linhas): quem fecha o PEGAR e falha o `perto` no fim
  da espera ficava no PEGAR com `fechou = True` e ganhava os sete ao vivo MAIS o congelado
  (32,3/s MEDIDO no `model_1600`, contra 20,7 no BOTAR). Agora vai à cauda CARREGAR, como a
  cadeia B; baixar a caixa na espera desvia da rota do BOTAR, que é perda.
- Docstring do `_aplica_espera` (~639-647): onde descreve o avanço e a reconferência do
  `perto`, acrescentar que quem falha vai à cauda (substituir, não empilhar).
- Invariantes: para `pegou = False` (REORIENTAR da cadeia R) nada muda; para o último elo
  (`tem_prox = False`) nada muda; o `concluiu` (~881) não muda (o env desviado fica no
  `_passo` 0 de 2, portanto conta como falha da cadeia C).
- `tem_prox` exclui o CARREGAR: sem isso o env desviado avançaria ao BOTAR quando a caixa
  voltasse ao peito (achado da revisão Opus).

## S4 — σ de orientação fixo no regime de pé (lote B, `comando.py`)

- `_recalcula_sigmas`, ~1843-1844:
  `de_pe, sig.clamp(min=float(np.deg2rad(c.tol_ang_deg))), sig)`
  → `de_pe, torch.full_like(sig, float(np.deg2rad(c.tol_ang_deg))), sig)`.
- Reescrever o docstring ~1800-1804 ("EXCEÇÃO: no regime FACE_DE_PE o piso ...") e o bloco
  ~1830-1837: no regime de pé o σ é a TOLERÂNCIA DO FECHO, fixa. Motivo: com `max(tombo
  inicial, tol)`, tombar a caixa na espera (os sete valem zero ali) alargava o σ de graça
  no CARREGAR e no BOTAR, e a S3 passa a ler o σ nos termos de aproximação. A metade linear
  do híbrido dá o gradiente de longe que o `max` existia para dar. O `FACE_VIVA` mantém
  `max(erro × fator, sigma_ori_min)`.

## S5 — notebooks (lote C)

Nos DOIS notebooks do zero (`g1_limpo_zero_kaggle.ipynb`, `g1_limpo_zero_colab.ipynb`),
editando o JSON com um script Python que troca strings exatas e confere `json.load` no fim:

1. `RUN    = \"zero11\"` → `RUN    = \"zero12\"`.
2. Logo depois do assert que menciona `"clone anterior a 28/09"` (célula 8), acrescentar:
   ```python
   assert KN.PesoPorEstado().pose[CMD.ESTADO_PEGAR_COM] == 1.0 and hasattr(
       __import__("g1_limpo.recompensas", fromlist=["_alinha"]), "_alinha"), \
       "clone anterior a 29/09: o PEGAR_COM ainda paga por ficar parado, ou erguer tombado paga aproximação"
   ```
   (`KN` e `CMD` já são importados na célula: `from g1_limpo import comando as CMD, knobs as KN`.)
3. SÓ no Kaggle, na impressão digital (célula 8, logo depois do `digital |= {"freio_fator": ...}`):
   ```python
   digital |= {f"tabela/{n}": json.loads(json.dumps(v.params["tabela"]))
               for n, v in rw.items() if "tabela" in v.params}
   ```
   (o `json.loads(json.dumps(...))` normaliza tupla → lista e dict de tuplas → dict de
   listas — `limite_de_junta` e `faixa_de_pose` têm `tabela` como DICT, e `[float(x) for x
   in ...]` quebraria neles; o formato final é o mesmo que `json.loads(arq.read_text())`
   devolve na retomada.) Comentário: as três tabelas (peso por estado, limite de junta,
   faixa de pose) mudam a recompensa sem mudar peso. Se o Colab tiver bloco `digital`, fazer
   o mesmo; se não tiver, não criar.

## D — smoke (main)

- `smoke.py` ~1554: `pose[PEGAR_COM] == 4.0` → `1.0` e o texto da checagem.
- ~1570: rastreio — PEGAR_COM entra na lista dos zeros; texto.
- ~4941-4946 (G1 1c): o `track_linear > 0` passa a ser testado em ESPERA_COM; PEGAR_COM `== 0`.
- item 6 (~5266-5321): metade dos envs com a caixa longe → no fim da espera `elo == CARREGAR`,
  `_passo == 0`; metade com a caixa no alvo → `elo == BOTAR`, `_passo == 1`; `avancos` só
  nos que avançaram.
- ~4172-4177: σ do BOTAR == tolerância (igualdade), e o comentário.

## Verificação (Opus, depois)

Plano contra implementação, arquivo a arquivo: diffs, invariantes acima, docstrings, e que
nada fora dos lotes mudou. O dono roda o smoke.
