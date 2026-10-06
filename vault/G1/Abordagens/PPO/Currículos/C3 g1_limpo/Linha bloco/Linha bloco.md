---
tipo: abordagem
pai: "[[C3 g1_limpo]]"
estado: validada
verificado: 2026-10-06
periodo: 26/08–18/09/26
resumo: "Um treino do zero e ~21 blocos em resume encadeado, cada um mudando a recompensa sobre uma política que já andava; termina no model_21700, base do melhor modelo."
cluster: c3-bloco
---
## Mapa da linha
| run | its | código (spec / commit) | o que testou | desfecho |
|---|---|---|---|---|
| bloco9 | 1800 → 5000 | v2.1 (`53f81e9`, 04/09) | gradientes v2.1 | cadeia fecha na it 2616; `sucesso` 0,574 na 5000 (só PEGAR) |
| bloco10 | 5000 → 6999 | provável v2.1 sem mudança | continuação | fonte do `model_5950` e `model_6999` (sondas de 08/09) |
| bloco11 | 6999 → ~7300 | v3.1 dois bits + F1/F2/G1/G2 | cadeia nova, alvo congelado, rastreio nos parados | arremesso da caixa no BOTAR; interrompida |
| bloco12 | 6999 → 7499 | v3.2 soltar-termina (`88fe9b5`) | soltar fora do alvo termina | arremesso some; robô segura a caixa até o time_out |
| bloco13 | 6999 → 9499 | v3.3 mão no alcançar (`206070b`) | `_alcancar` sem ≡ 1 no BOTAR | apoia (`load` 0 → 0,441/s), BOTAR não fecha (`de_pe`) |
| bloco14 | 9499 → ~10300 | v3.4 + v3.5 (`0234e7b` … `42c4453`) | BOTAR sem `de_pe`, cauda parada de pé, punho σ 0,30 | **BOTAR fecha**: `s_C` 0 → 0,195; parada pelo 3º estágio do envelope |
| bloco15 | ~10300 → ? | giro sem gingado + corte do 3º estágio + envelope ±1,6 (`d2ac2ac`, `80244b2`) | o robô girar | medições do `model_10200` geram a tabela por estado |
| bloco16 | → ~11322 | tabela por estado + dobradiça + punho σ 1,00 (`c64ee98`, `954ed94`, `bcd0e84`, `dbcc299`) | teto ≥ piso | Colab 8192 envs; "morreu na ~11322" |
| bloco17 | ~11322 → 14000 | + `faixa_de_pose` (`c9578ae`, `d74b93f`) | braço no batente | `model_14000` = tag `estavel-bloco17`; cadeia inteira |
| bloco19 (v3) | 13999/14000 → +472 | `amax` no `velocidade_por_regime` (`7d481d8`) | junta mais rápida | **colapso** em 472 its |
| bloco20 (sonda, v3) | 13999 → 14391 | correções M2–M6 (`f1e4e22`, `f7ae0f6`), alvo pinado 1,02 | as correções sobre a política pronta | `s_B` 0,62 → 0,86, `s_C` 0,484 → 0,002; inconclusiva |
| zero01 (Colab, 14/09) | 0 → 113 | v3 com `amax` só em standing (`25c9877`) | treino do zero | **colapso** em 113 its |
| bloco18 (v2 da tag) | 14000 → 16000 (META) | `9b6cc8f` (15/09): alvo 0,85–0,95, inércia real, `limite_de_junta` exponencial | correções mínimas sobre o estável | `model_15200`, `model_15999`; ver ambiguidade "bloco19" |

## Por que a linha bloco terminou e a linha zero começou
1. **14/09, a tentativa de corrigir o `model_14000` por resume falhou.** A sonda `bloco20` (correções
   M2–M6 sobre o 13999) derrubou `s_C` de 0,484 a 0,002 em ~390 its. Hipótese do dono: as correções
   tornaram inviável a única estratégia de pousar que a política conhecia (escorar), e reaprender por
   warm-start não acontece (handoff 14/09). É o mesmo argumento de 28/08 (vault `g1-limpo-mesa-contato-zero`):
   "desaprender custa mais que aprender certo".
2. **O plano do dia chama-se "Correções para o treino do zero"** (`51cc0ac`) e cria os notebooks
   `g1_limpo_zero` e `g1_limpo_zero_colab`. A `zero01` do Colab começou no mesmo dia e colapsou em 113
   its pelo `amax` (handoff).
3. **O pipeline foi dividido** (`96be97a`): o estável na tag `estavel-bloco17`, o experimental na
   `exp/g1-limpo-v3`.
4. **A linha bloco NÃO parou em 14–16/09.** Em 15/09 a v2 foi refeita a partir da tag com três mudanças
   pequenas (`9b6cc8f`, bloco18) e seguiu por resume: bloco20 (limite de pelve, 16/09), bloco21
   (`forma_postural`, 16–17/09), bloco22 (até o `model_24999`, 18/09). O relatório de 01/10 dá a linha
   bloco de 26/08 a 18/09 e a linha zero de 14/09 a 01/10 (`docs/relatorios/2026-10-01-aprendizados-zero17-para-v3.md` §5b).
5. **O que virou a balança para o zero (18/09):** o veredito do dono no `model_21700` (tronco deitado,
   braço dobrado, perna para fora) e a hipótese dele de que os hábitos vêm do resume, herdados de
   20 900 its com a rampa a 85% e sem a terminação (vault `g1-limpo-botar-veredito-do-dono-21700`). O
   notebook do zero (`618373f`, 16/09; RUN `zero01` corrigido em `c8bfaa0`, 17/09) é o teste.
6. **Lição registrada depois** (relatório 01/10 §5b): a linha bloco aprendeu porque recebeu as
   restrições depois da habilidade, em 21 blocos de resume; a linha zero recomeçou ~12 vezes (zero01 a
   zero17) e enfrenta todas as restrições na it 0. O `model_23250` (21700 refinado) é o melhor até 02/10.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.

## Filhos
<!-- filhos:inicio -->
- **Bloco 1** (abordagem · validada · 27/08/26) — Locomoção primeiro com fatia 0,95: razão de marcha 0,553 e portão aberto na it ~963; o PEGAR não fecha e o robô escora na mesa no play.
- **Bloco 10** (abordagem · parada · 05–08/09/26) — Continuação até a it 6999; os checkpoints 5950 e 6999 alimentam as sondas: 0 de 190 BOTAR fecham, caixa no hold a 1,025 m.
- **Bloco 11** (abordagem · rejeitada · 09/09/26) — Dois bits v3.1 (cadeias B, R, C, alvo congelado): o robô ATIRA a caixa no BOTAR; termo que paga estado da caixa tem de exigir a mão.
- **Bloco 12** (abordagem · rejeitada · 09/09/26) — Soltar fora do alvo termina: o arremesso some e o robô segura a caixa até o time_out; fechar não ganha nada.
- **Bloco 13** (abordagem · rejeitada · 09–10/09/26) — Mão no alcançar: o robô apoia (load 0 → 0,44/s), mas o BOTAR não fecha porque o de_pe reprova pelo joelho em 99,975%.
- **Bloco 14** (abordagem · validada · 10/09/26) — BOTAR sem de_pe e cauda parada de pé: o BOTAR fecha pela primeira vez (s_C 0 → 0,195); parada pelo 3º estágio de velocidade.
- **Bloco 15** (abordagem · parada · 10–11/09/26) — Giro sem gingado e envelope ±1,6 rad/s; o model_10200 gera a tabela por estado (teto do que falta ≥ piso do feito).
- **Bloco 16** (abordagem · parada · 11/09/26) — Tabela por estado, dobradiça de velocidade e punho σ 1,0; braço D no batente; morreu na ~11322 sem causa registrada.
- **Bloco 17** (abordagem · validada · 11–14/09/26) — faixa_de_pose: model_14000, tag estavel-bloco17, roda a cadeia inteira com três defeitos (juntas rápidas, escora na caixa, mãos tortas).
- **Bloco 18** (abordagem · validada · 15–16/09/26) — v2 refeita da tag: alvo 0,85–0,95, inércia real e limite_de_junta exponencial; model_15200 anda agachado com a caixa (pelve 0,564).
- **Bloco 19 v3 colapso do amax** (abordagem · rejeitada · 14/09/26) — Freio por amax em vez de mean: colapso em 472 its; o limiar carrega a redução para a qual foi calibrado.
- **Bloco 2** (abordagem · rejeitada · 27–28/08/26) — Multa −1,5 por tocar a mesa: o contato caiu e a manipulação caiu junto; multa que o robô pode pagar ele orça.
- **Bloco 20 v3 sonda** (abordagem · rejeitada · 14/09/26) — Correções M2–M6 sobre o 13999: s_B 0,62 → 0,86 mas s_C 0,484 → 0,002; reaprender por warm-start não aconteceu.
- **Bloco 3** (abordagem · rejeitada · 28/08/26) — Do zero com terminação de corpo inteiro: ~75% da manipulação morre na mesa; a auditoria achou três anti-hacks de geometria perdidos na reescrita.
- **Bloco 4** (abordagem · parada · 28–30/08/26) — Cadeia da pega consertada; ~46% da manipulação ainda termina na mesa; o run se perdeu com a sessão da Kaggle.
- **Bloco 5** (abordagem · parada · —) — Sem nenhum registro em git, specs, docs ou memória; talvez a sessão perdida ou a queda de internet de 1500 iterações.
- **Bloco 6** (abordagem · validada · 30–31/08/26) — Partição da mesa (94,7% é o tronco), piso da estátua (parado rendia 145 contra 102) e portão diluído; os consertos destravaram a pega.
- **Bloco 7** (abordagem · validada · 31/08–02/09/26) — Mesa vira multa em rampa e vantagem normalizada por elo: postura_ereta sai do zero e a queda cai de 51,9% para 0,9%.
- **Bloco 8** (abordagem · rejeitada · 03–04/09/26) — Contrato de troca v2 do zero: marcha madura na it 1503, mas a manipulação regride; a auditoria achou 10 buracos de gradiente.
- **Bloco 9** (abordagem · validada · 04–08/09/26) — Gradientes v2.1 em warm-start: o PEGAR fecha pela primeira vez na it 2616 e o sucesso chega a 0,574 na it 5000.
- **Blocos 20 a 22** (abordagem · validada · 16–18/09/26) — Limite de pelve, forma_postural contra a IK, sola e quadril; chegam ao model_21700, que faz tudo menos um BOTAR bom (tronco 65°, perna virada).
- **Resume com terminação de corpo inteiro** (abordagem · rejeitada · 28/08/26) — Terminação por tocar a mesa sobre o model_4999: apagou o único caminho que a política conhecia; desaprender custa mais que aprender certo.
- **Achados soltos da linha bloco** (achado · ativa · 26/08–16/09/26) — Achados datados que não pertencem a um run: Entity.data obsoleto, diluição do Episode_Reward, unload quase booleano, alcance refutado, escorar compra o fecho.
- **Checkpoints da linha bloco** (achado · ativa · 26/08–02/10/26) — Tabela dos checkpoints da linha bloco (4999 a 23250): origem, o que cada um faz no play e a fonte.
- **Lacunas da linha bloco** (pendencia · aberta · 06/10/26) — O que as fontes não confirmam: bloco 5, limites dos blocos 2, 6, 7, 10, 15, 16, nome bloco18 × bloco19, origem do ckp9100.
<!-- filhos:fim -->
