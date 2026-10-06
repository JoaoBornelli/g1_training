---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
resumo: "Auditoria de travas g1_poc (20/08): nivel lia a forma do episódio SEGUINTE (p_up=0,7·p); Risco 3 invertido (pico a 0,30 m); T1 da caixa caindo REFUTADA por mim"
cluster: c2
---
# g1-poc-auditoria-travas (original)

Corpo original da memória `g1-poc-auditoria-travas`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Auditoria por subagente em **2026-08-20** (passeio aleatório do nível simulado, FK sobre o modelo real do G1, matriz gate×gate). Complementa g1-poc-auditoria-incentivos.

## Confirmado e consertado no MACRO 1

- **T2 — a promoção de nível usava a forma do episódio SEGUINTE.** `nivel` vinha depois de `forma` no dict de currículo e `sorteia_forma` sobrescreve `poc_manipula`. Efeito: `p_up = 0,7·p` (ponto fixo em 0,714, não 0,5; nível médio 1,07 em p=0,5, não 3,0), e locomoção rebaixava o nível 70% das vezes — **um bloco com frac_locomocao=0,85 teria teto de nivel_medio 0,214 mesmo com manipulação perfeita**. Conserto: ordem `twist_ranges, nivel, forma, hinge, action_rate`. Regra geral: **quem lê a forma do episódio que ACABOU vem ANTES de `forma`; eventos leem a forma nova e rodam depois de todo o currículo.**
- **T3 — o portão "nivel_medio sai de zero" já vale hoje** com p=0,006 (0,0042 ≈ 17/4096 envs). Portão útil: `nivel_medio ≥ 1,0` (exige p≈0,5 pós-conserto) + histograma (`nivel_frac_0`, `nivel_frac_3mais`); `nivel_max` satura com um env sortudo.
- **T7 — gate do twist**: monotônico re-explodiria (resume além dos degraus + política que anda → 0→2 em 0,08 iteração com EMA medida no estágio 0). Conserto: descida com histerese (EMA < 0,8×alvo), teto de 1 degrau/12 iterações (≈3τ da EMA de 0,99), EMA sem os envs `is_standing_env` (parados até o time_out dariam 8% do alvo de graça). Estado não persiste no checkpoint (runner só salva `common_step_counter`) — recalibra em ~12 iters, declarado.
- **T9/T10 — postura_ereta**: rampa 0,45–0,65 estava saturada em 100% das pegas dos níveis 0–1 e MORTA em 33% do nível 4 (pelve na pega chega a 0,267 m); e sem gate de descarga o termo pagava +2,0/s no platô "encosta e fica de pé" do nível 0 (pelve máx na pega a 0,55 = 0,739 ≥ 0,65). Conserto: rampa em duas partes + gate `F_apoio < 0,2·m·g`.
- **T6 — nível 4 é parede de ALCANCE, não de força**: 39,7 N de palma bastam para 5 kg (μ_ef≥1,0 → precisa 24,5 N; e o `squeeze` DESSATURA: tanh(1,29)=0,86, derivada 0,26). Mas com jitter_x=0,20 fixo, 60% dos episódios do nível 4 nascem fora de alcance (poses só existem até x_rel≈0,45) e o twist zerado cobra 0,44/s pelo passo. Conserto: `jitter_x` na célula (0,08 nos níveis ≥4).

## REFUTADA (por mim, medição própria)

- **T1 "a caixa afastada cai na frente do robô"**: o auditor simulou SEM a prateleira. Medido no env real (120 passos, 5 kg de wrench, 100% locomoção): caixa fica em z=5,099, APOIADA na laje a 5 m. O sensor `apoio_caixa` prova que caixa↔laje colidem. `afasta_cena` correto. Lição: **subagente que "simulou" fora do env real precisa de contraprova no env real.**

## Invertido por medição (contingência, não mudança)

- **Risco 3 da §19**: o pico de `contato_ilegal` é no topo **0,30 m** (20–25% das poses de pega — a laje na altura da coxa), NÃO a 0,04 m (0–3%: é degrau, o corpo passa por cima). Se a escada travar em nivel_medio≈2 com `Episode_Termination/contato_ilegal` alto → prateleira x=0,50→0,60 (arrasta caixa_xy junto; bloco próprio).

## Decisões fechadas para a MACRO 2 (cadeias)

1. `load` (+2,0, espelho do unload) ativo só no elo `botar` + máscara do `squeeze` lá; sem isso o fecho tem saldo 0 (sorte).
2. `caixa_largada`: armar em `poc_pegou`; desarmar o ramo "escapou" quando a CADEIA fecha; mascarar na janela de sustentação do elo terminal (0,5 s; senão recuo de mão de 0,56 m/s termina o episódio). O gate `pegou & ~sucesso` é identicamente falso na cadeia de 1 elo.
3. `carregar` fecha SUSTENTADO 0,5 s (instantâneo com no_alvo≈57% = moeda → nível vira passeio sem deriva).
4. Bloco preparatório "segure e ande" antes da 4b: hoje **0,00%** das transições têm twist≠0 E caixa_valida=1, por construção.
5. `reorientar` = EMPURRAR a caixa apoiada (girar em mão é impossível: diagonal 45° = 0,283 m > vão 0,20 m; torque de escorregamento ≈0,49 N·m). Ator precisa de 3 canais de orientação da caixa (112→115) antes da 4a.
6. Ordem das correções do unload: máscara ao elo `pegar` ANTES de qualquer mexida em `poc_topo`; aliás ninguém observa `poc_topo` (o crítico lê a pose real da mesa) — reescrevê-lo é desnecessário.
7. ⚠ Docstring de `g1_training/common/robot.py` diz normal do pad "±Z local"; o código produz ±Y. `squeeze` usa ±Y com `.abs()` — certo, mas é mina para o `reorientar`.
8. **Sem canal de elo na observação** — a §21.1 fecha: "a política não tem fase, não sabe o que é cadeia"; o comando fica em 10 números. `pegar`×`botar` se separam pelo próprio alvo: com `botar_folga_laje = 0,05`, o alvo do botar sai em z ≤ 0,77 e o do pegar em 0,78–0,85 — faixas disjuntas.
9. **`face_normal_b` (3 canais, orientação atual da face no frame da base) → ator 112→115, crítico 125→128.** Entra POR ÚLTIMO nos dois grupos, de propósito: a cirurgia de warm-start vira append de 3 colunas zero na 1ª camada + mean 0/var 1 no normalizador (`g1_poc/expande_checkpoint.py`). A percepção do deploy já mede a grandeza (§21.2 "os medidos").

Relacionado: g1-poc-bloco1-treino, g1-mjlab-mutacao-em-treino.
