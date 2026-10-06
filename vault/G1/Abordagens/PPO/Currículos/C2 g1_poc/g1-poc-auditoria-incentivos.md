---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
resumo: "Auditoria de incentivos do g1_poc (20/08): reaching sem σ variável = sorte do nível 3+; postura_ereta subcalibrada 3×; botar sem termo que pague por soltar"
---
# g1-poc-auditoria-incentivos (original)

Corpo original da memória `g1-poc-auditoria-incentivos`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Auditoria por subagente em **2026-08-20**, com medição em CPU (16 envs) + cálculo sobre os 21 termos propostos. Complementa g1-postura-penhasco-std-standing e corrige g1-poc-bloco1-treino.

## Premissas da §8.3 que estavam erradas (medido)

- caixa→alvo no reset: **d₀ = 0,2561 m**, não 0,18 (σ do `bringing` 42% maior)
- `squeeze` no estado B: `tanh(1) = 0,7616`, não 1,00
- fundação no reset: **5,74/s**, não 6,0 (a entrega do navegador §11.1 custa 0,50/s no instante)

## Os achados, por gravidade

1. **`reaching` com σ = 0,20 fixo = sorte do nível 3 para cima.** Gradiente de aproximação por altura da prateleira: 2,64/m (0,55) → 0,027/m (0,15) → **0,0019/m (0,04)** — queda de 1391×. No nível 4 há um VALE de −0,84/s entre "em pé" e "mãos na caixa" (upright −0,59 + pose −0,28) e o farol do outro lado mede 1,85e-5. Conserto: `σ_reach = max(0,20 ; d_inicial palma→face)` — a MESMA correção que a §8.2 fez no `bringing`. Recupera 3,24/m no nível 4. Sem isto a tabela de células entrega cenas sem gradiente.
2. **`postura_ereta` (rampa 0,20, peso 1,0) subcalibrado ~3×.** No ponto de fecho: rampa 5,0/m contra `precise_pos` −16,21/m → saldo **−0,219/s** por subir os 1,55 cm que faltam (braços rígidos); com compensação de braço, +0,196/s. E a rampa 0,45–0,65 é MORTA no agachamento do nível 4 (pelve ~0,35). Conserto: rampa em DUAS partes (piso 0,30 m + trecho final íngreme ≥12,5/m).
3. **`botar` não tem quem pague por soltar.** `unload` cobra −2,0/s por apoiar; mascarado ao elo `pegar`, o saldo vira exatamente ZERO — fecha por sorte. Soltar de vez custa −8,38/s (perde staged/ori/squeeze/postura_ereta). Conserto: termo `load = clamp(F_apoio/m·g) × preensão × perto`, peso +2,0, ativo SÓ no elo `botar` — espelho do `unload`.
4. **`precise_ori` com σ = 0,40 rad fixo**: 45°→0,021, 90°→2,0e-7. O `reorientar` dos níveis 4–6 é sorte. Conserto: σ variável `max(0,40 ; Δθ_inicial)`, mesmo idioma.
5. **Sustentar 1,0 s não tem incentivo NENHUM** — 0,98 s e 1,00 s pagam o mesmo. Conserto: rampa `clamp(t/sustenta_s)`, peso ~0,5.
6. **Freios × tarefa:** hinge a −1,00 = −4,00/s e action_rate a −0,25 = −2,20/s; cada um sozinho supera o `unload` inteiro (+2,0/s). Gate por competência tem de cobrir os dois (Fase 5).
7. **`carregar`:** quem paga por andar são os `track_*` (4,0/s, destravam com o twist do elo). `std_walking` com caixa na mão é penhasco de −0,885/s (σ braço 0,15); `std_manipulando` anda mas com 1/7 da moldagem. Conserto MACRO 2: 5º dicionário `std_carregando` (pernas do walking + braços do manipulando) → pose 0,718, 0,404/rad no joelho.
8. **`base_lin_vel` fora do ator sem histórico → REVERTIDO em 20/08 (commit 562bd01), gatilho disparado.** Medido na it 5631: 530 iterações com ~30% das transições de andar (controlador entregou) e `dur_loco_ema` PIOROU (112→26), até os parados caindo; contraste: run com std 0,30 tinha dur 93. Voltou na receita do fabricante (canal + Unoise ±0,5, POR ÚLTIMO) → **ator 118**; crítico fica 128 (já tem a versão limpa). A escola "No-State-Estimation" exige histórico, que não temos — tirar o canal sem histórico deixa a velocidade irrecuperável num quadro. Deploy: o canal vem do estimador de bordo da Unitree; o ruído ±0,5 é a tolerância. `expande_checkpoint` migra 112|115→118.

## O que NÃO precisa mudar

O `unload` é o único termo com derivada não nula em toda a faixa, sem saturação. É o padrão que os consertos 1–5 copiam. A tabela A→D no nível 0 é monótona (erguer paga 38,8% mais no episódio, com os 21 termos).

## Da auditoria de viabilidade (mesmo dia)

- Spec MACRO 1 mecanicamente OK: 21 termos confirmados, `nivel_forcado` via `get_term_cfg` funciona, gate do twist funciona, NO-OP do nível 0 MEDIDO (topo 0,550–0,570, carga 1,0).
- Alias `env.poc_success` quebra no PRIMEIRO `_update_command` (rebind por `torch.maximum`); só a `caixa_largada` era cega — o `nivel_caixa` lê `cmd.episode_success` direto e nunca foi afetado. Depois do fix `copy_`, vigiar `Episode_Termination/caixa_largada` no 1º bloco.
- `env.poc_nivel` JÁ existe no primeiro reset (currículo roda inteiro antes dos eventos) — o `getattr(..., None)` é defensivo, não necessário.
- `Celulas.topo_min[4:] = 0,04` duplica `Cena.prateleira_topo_piso` — mesma laje, dois knobs; casar sempre.

Relacionado: g1-mjlab-mutacao-em-treino, g1-pegar-plato-do-grasp-booleano.
