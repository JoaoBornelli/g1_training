---
tipo: pendencia
pai: "[[Linha zero]]"
estado: aberta
verificado: 2026-10-06
periodo: 06/10/26
resumo: "O que as fontes não confirmam: fim da zero01 piramidal, zero02 × NoBatente, model_9100, linhagem do model_3100 e do 3450."
cluster: c3-zero
---
### zero01–zero08
1. **O destino da zero01 piramidal e a frase "a zero01 andou e pegou".** O único registro da
   3ª zero01 para na it 392 (`avancos` 0,014, `s_B` 0). O `[d4e5e4e]` diz que a zero01 do
   Colab morreu por NaN. A afirmação "andou e pegou" e "comprimiu as pernas com freio −2"
   vivem só em memória e documentos de 01–02/10, sem número nem checkpoint.
2. **Por que a zero02 sobreviveu à `NoBatente` e a zero03/04 não.** A zero02 tinha
   `VIGIA_BATENTE = True` com rampa 0,95 (notebook em `d4e5e4e`) e chegou à it ~11 191 com
   `s_B` 0,32; as fontes de 01–02/10 dizem que a `NoBatente` "matou a manipulação nas zero02–04".
   Diferenças candidatas (Kaggle 4096 contra Colab 8192; freio −2 contra −15; as sessões
   depois da 5477 talvez rodadas pelo notebook de resume, que não tem a terminação) não foram
   verificadas. Também não há log nem veredito final da zero02.
3. **Atribuição do `model_9100`.** A nota do vault de 21/09 e o plano de 21/09 dizem `zero02`;
   o registro de checkpoints de 01/10 diz `ckp9100` ≈ bloco13 de ~09/09 (PROVÁVEL). Os CSVs
   `ckp9100_m2_*_i2.csv` são de 21/09 e o `i2` sugere impratio 2 (só no zero), mas o `.pt`
   não está em `~/Downloads`.
4. **Iteração final e motivo de fim da zero05, zero06 e zero08.** Nenhum log ou checkpoint
   identificado; só os motivos de projeto nos commits.
5. **zero07: pelve a 59° (commit `6ab0258`) ou tronco a 67° com escoro (memória).** As duas
   grandezas não foram reconciliadas.
6. **O conserto do escoro (teto do `apoiada`, `~apoiada` no PEGAR, `upright` ×4) nunca entrou
   na linha zero** (ficou no ramo `exp/g1-limpo-v3`). Nenhum documento registra se isso foi
   decisão ou efeito colateral do bloco 18 partir da tag.

### zero09–refino
1. **Linhagem da zero17 e da zero15/16.**
   - A zero17 partiu do `model_3100` (memória e transcrição), mas nenhuma fonte diz de qual run ele saiu. Ele foi baixado em 29/09 às 19:27, antes do commit da zero15.
   - O `model_3450` é "zero15" na spec e "zero16" na memória.
   - A run do `model_3300` (play de postura) não é nomeada.
   - Não há registro de treino próprio da zero13 nem da zero16.
   - A relação de resume entre zero12 → zero13/14/15 não está escrita. A it ~3100 da zero12 e o `model_3100` podem ou não ser o mesmo arquivo.
2. **Números dos refinos de 03 a 05/10 só existem nas transcrições.**
   - As leituras do log de 23253 → 24393 e dos dois resumes do 24350 vêm de prints do TensorBoard comentados na sessão. Nenhum relatório no repo as guarda, salvo a tabela resumida do REL-APR §5c.
   - Os sha256 do `model_24350` (`5b19a35354`) e do `model_24800` (`c46e5b4f11`) só aparecem na coluna `ckpt` dos CSVs. Eles não estão no V-CKPT.
   - Não há checkpoint nem it final registrados para o segundo resume (parado após ~24669).
3. **Não medido, e decisivo para o porquê das falhas:**
   - em que elo acontecem o `caixa_largada` e o `fell_over` da zero17;
   - a causa do CARREGAR lento, girando e agachado;
   - a sonda do crítico sobre o ganho de chegar agachado;
   - uma linha zero com a sequência de afrouxamento da linha bloco, que separaria "tempo de refino" de "armadilhas do começo";
   - se a força de ~200 N vem da geometria da pega (hipótese).
4. **Menores:**
   - zero08/zero09 só aparecem por medidas citadas em commits e relatórios; as datas exatas dos treinos (26–27/09) não estão escritas.
   - O `model_23700`/`23750` (pico antes do freio) não tem play.
   - O `model_23800` testado pelo dono não tem veredito além da comparação com o 24350.
   - Os relatórios da v3 (inventário, avaliação, revisão da spec, DR para a v3) não foram lidos em detalhe neste levantamento.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
