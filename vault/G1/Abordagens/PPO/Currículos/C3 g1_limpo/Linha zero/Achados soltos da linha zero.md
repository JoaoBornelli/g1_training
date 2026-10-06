---
tipo: achado
pai: "[[Linha zero]]"
estado: ativa
verificado: 2026-10-06
periodo: 14/09–06/10/26
resumo: "Achados datados: escorar compra o fecho, alvo 1,02 circular, curso largo do hip_yaw, deriva de rumo, caixa_na_pega diluída, maior μ do par, NaN do cone."
cluster: c3-zero
---
### 14/09–23/09
- **14/09, escorar compra o fecho.** `apoiada = forca >= 0,5·m·g` dispara a 4,9 N com caixa de
  1 kg; o robô pesa ~343 N; fechar 1 s antes vale ~23 (cauda ~58/s contra BOTAR ~35/s) e
  escorar custava ~1/s de `upright` `[f7ae0f6] [handoff 14/09]`. Conserto só no ramo v3.
- **14/09, o alvo 1,02 era circular.** Veio de `caixa_b.z` p50 1,025 no HOLD do `model_6999`;
  o ombro fica a 1,074. Custo de braço 1,30 a 1,02, 0,80 a 0,85. Piso do alvo 0,80 vem do
  fecho (laje 0,57, caixa maior centro 0,70, `tol_pos` 0,10) `[f7ae0f6] [9b6cc8f]`.
- **14/09, quatro achados de arquitetura (não consertados):** episódio não acaba no sucesso
  (renda congelada vira anuidade; raiz da pressa); `renda_congelada` congela UM instante;
  `descarga` com `clamp(0,1)` cega acima de m·g; `squeeze` cego acima de ~3×F_ref (1,0000 a
  68,9 N) `[handoff 14/09] [plano 14/09 §4]`.
- **14/09, BOTAR é bang-bang e o alcance foi refutado.** `carrega085.csv` (`model_13999`): p50
  0,02 rad/s, pico 11,58; 84% dos passos congelados. IK alcança com margem 0,35 (laje 0,55) e
  0,25 (laje 0,05) (memória `g1-limpo-botar-e-bang-bang-nao-pressa`).
- **14/09, inércia real da caixa.** `dr.pseudo_inertia` em reset não corrompe mais a heap no
  mjlab 1.5.3 (A/B CPU, 8 envs); GPU não verificada `[handoff 14/09 alvo-e-inércia]`
  `[9b6cc8f]`.
- **15/09, batente é restrição mole.** `ankle_roll` chega a 150% do curso, `ankle_pitch` a
  120%, empurrados pelo contato (limitação 11) `[9b6cc8f]`.
- **17/09, curso largo.** `hip_yaw` (curso 316°) a 0,96 = 151° no BOTAR do `model_20500`;
  piora 17999 → 19300 → 20500 (−0,94 → −1,97 → −2,52 rad) `[a37300c]`; a rampa a partir de 40°
  foi paga 900 its (20900 → 21800) sem mover a pose `[b226853]`.
- **17/09, deriva de rumo.** `model_19300` −0,38 rad/s na pega, −20°/s no CARREGAR, +4,4°/s na
  marcha reta; a marcha reta natural tem |wz| mediano 0,12 rad/s `[3a4701c]`.
- **18/09, clip da ação rejeitado.** O corte de ±0,7 rad no `hip_yaw` foi revertido no mesmo
  dia `[7b1b5ac]`; motivo na memória: "o sinal só está sendo cortado após a decisão ser
  tomada".
- **18/09, terminação `|hip_yaw| > 1,0 rad` no resume (fora do git):** `s_C` 0,73 → 0,02 →
  0,6 em ~400 its (avaliação 02/10, tabela de terminações).
- **20/09, o `forma_postural` não tem o braço.** O robô copia pelve (0,53) e tronco (65°) da
  referência e mantém ombro→palma em 0,25 m contra 0,41–0,46 m (status Obsidian; nota do
  veredito).
- **21/09, aborto por perda de preensão (plano futuro).** O mergulho atrás da caixa caída é
  ótimo para a tabela; falta treinar a transição "ANDAR com comando zero em qualquer instante"
  `[c5b950c]`.
- **21/09, mão a 2,58 m/s.** `model_24999`: pegar p95 1,79, máx 2,58 m/s; 318 de 350 passos da
  pega custam zero; p99 de junta na pega 5,71 rad/s; 11,5% dos pares junta-passo passam de 1,5
  (status Obsidian).
- **22/09, contagem de termos não discrimina clone.** O assert de 29 termos caiu; eram 31. A
  ORDEM é o que discrimina (`renda_congelada` por último) `[8a457e4]`.
- **23/09, YAM lento por desenho.** Servo a 2 Hz, freio em `reward_curriculum` −0,01 → −0,1
  (it 500) → −1,0 (it 1000) (memória `g1-limpo-yam-lento-por-desenho`; `[plano 09-23 §1]`).
- **Velocidade de iteração medida.** Kaggle 4096 envs: 6,9 assumido, 7,45 medido; Colab 8192
  envs: 12,67 s/it na zero03 `[0682e6e] [1643fb3]`; zero01 piramidal: 12,1 s/it (notebook
  baixado).

### 28/09–06/10
| Data | Achado | Fonte |
|---|---|---|
| 28/09 | Um vetor normal de face lateral não muda quando a caixa gira em torno dele: o `ANG` antigo era cego ao giro palma-a-palma | V-ANG |
| 28/09 | O piso do currículo `abertos = buf.max()+1` mantém aberto todo nível já alcançado: o nível médio é equilíbrio de sorteio, não competência | REL-TOMBO achado J |
| 29/09 | Pads com `condim=3` não têm atrito de torção: empurrar a caixa contra o peito a endireita dentro da mão (latente, só no simulador) | REL-TOMBO achado G |
| 29/09 | Recaptura do eixo de cima com a caixa na mão via `_sigma_pendente` (defeito de 21/09 por outro caminho) | REL-TOMBO achado E |
| 29/09 | A impressão digital `pesos.json` era cega a janela, σ e portão (mudam a recompensa sem mudar peso) | REL-TOMBO achado I |
| 29/09 | O comentário do `velocidade_por_regime` estava errado em três pontos (regime pelo `twist`, dependência do elo, ANDAR) | `git:78d2af5` |
| 29/09 | O `caixa_na_pega` do log dilui-se entre os envs (marcava ~8° com a caixa deitada a 89°): julgar o tombo pelo play | `git:ce3d460` |
| 30/09 | O passeio ±1 com piso de 20% dá sucesso estacionário ≈ 0,5 − 0,16·(3 − nível médio): `s_B ≥ 0,50` só chega com nível ~3 | MEM:g1-limpo-piso-segura-o-s-b |
| 30/09 | Nível por tarefa REJEITADO pelo dono (o BOTAR vem sempre depois de um PEGAR na mesma caixa) | MEM:g1-limpo-piso-segura-o-s-b |
| 30/09 | Treino com impratio 2,0 e play com 1,0; horizonte efetivo do GAE ~0,34 s (γ 0,99, λ 0,95) | REL-PLATO §5 |
| 30/09 | O `PPOPorElo` agrupa a vantagem pelo elo interno: o grupo CARREGAR mistura o elo parado da C com a cauda de B que anda | REL-PLATO §5 |
| 30/09 | No BOTAR, descer paga +0,29 a +0,37/s por cm em todas as rotas; CoM com ~6 cm de margem no ponto congelado; nada encosta na laje | REL-PLATO A6 |
| 30/09 | Pega de berço: para descer pelos braços a mão tem de girar 35–57° sobre a face, e nenhum termo paga isso | REL-PLATO A7 |
| 30/09 | `CLAUDE.md` ganha as duas regras irrefutáveis (ação proposital; serve ao objetivo documentado) | `git:f8f7f10` |
| 01/10 | Agentes cegos erram a conclusão sem revisão: um concluiu "agachar é do ANDAR parado", e a observação idêntica prediz exatamente isso | REL-APR §7 |
| 01/10 | Dois arquivos `model_5000.pt` e `model_5000(1).pt` geravam CSVs indistinguíveis: daí a coluna `ckpt` | REL-APR §6 |
| 01/10 | O registrador recalcula o alvo do BOTAR a cada passo (no treino ele congela na abertura); o roteiro `c` reusa a laje da pega e cria alvo lateral | REL-APR §6 |
| 01/10 | A laje tem quatérnion identidade: com rumo de 30°, a quina fica a 0,09 m (o R1 não trata isso) | REL-MONO §3 C4c |
| 01/10 | No estágio 0 do `command_vel` (\|wz\| ≤ 0,5) o piso σ 0,707 faz a estátua colher 0,61–0,92 do rastreio angular | REL-MONO V2 |
| 02/10 | O checkpoint `.pt` não grava o commit; o commit é inferido pela hora do download e pelo `git log` (CERTO / PROVÁVEL) | V-CKPT |
| 03/10 | Logo depois de cada resume todas as EMAs de sucesso caem por ~1 duração de episódio (sucesso exige `time_out`), e o R4b encolhe a laje nesse vale | SESS-03/10 |
| 03/10 | O contador entre degraus do freio não vai no checkpoint: o freio fica parado ~400 its a cada sessão | SESS-03/10; REL-APR §5c |
| 05/10 | O MuJoCo usa o maior μ do par: DR só em pad e caixa é mascarada pelo pulso e pelo tronco em 1,0; `priority` resolve | REL-APR §3b |
| 05/10 | `noslip_iterations` não existe no MuJoCo Warp (`NotImplementedError`) | REL-DR §8 |
| 05/10 | Cone elíptico + impratio 10 → NaN: Hessiano indefinido (mujoco_warp #1657, fix ausente no 3.10.0.3), busca de linha esgotada (#3628) e warm-start que propaga o NaN no reset | V-NAN |
| 05/10 | Literatura: o teto de força do HDMI (G1 real) e a mola virtual do GentleHumanoid são precedentes do freio de aperto; "escorrega → aperta mais" sem tátil não tem precedente (números a confirmar nos PDFs) | REL-DR §7 |
| 06/10 | Ideia de testar o 24800 com BFM-Zero (tracking) e SONIC como executor do movimento | `vault/_sistema/inbox.md`; SESS-06/10 |

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
