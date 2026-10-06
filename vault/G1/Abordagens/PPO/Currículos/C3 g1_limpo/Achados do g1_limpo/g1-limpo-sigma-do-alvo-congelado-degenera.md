---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "σ = erro inicial é incompatível com alvo = atitude atual; no PEGAR isso deixava tombar a caixa de graça, medido em 53 a 59 graus"
cluster: c3
---
REGRA: `σ = erro inicial` só funciona com alvo INDEPENDENTE do robô. Alvo capturado da
pose corrente tem erro inicial zero por definição, então o σ cai no piso (0,20 rad) e o
kernel morre a partir de uns 20°.

No PEGAR a face congela com a caixa na mesa: tombar ao erguer era grátis (53 a 59° na `zero02`).

CONSERTADO 22/09: σ pelo `tol_ang_deg` = 25°. Desde 28/09 o regime congelado não existe; ver g1-limpo-ang-cego-ao-tombo.
σ 25° SOZINHO REPROVADO (28/09): `ANG` 66° na `zero08`, kernel 0,001.
Forma HÍBRIDA desde 28/09, todos os estados: `½(1 − Δθ/π) + ½ exp(−(Δθ/σ)²)`. Derivada
≥ 0,16/rad de 0 a 180°; 1,0/rad a 25°. Toda orientação precisa de gradiente.
⚠ `FACE_VIVA` fica fora do σ pela tolerância — alvo independente, erro de até 90°.

**Why:** o fecho e a recompensa passam a concordar sobre a régua.
**Detalhe:** `g1_limpo/ARQUITETURA.md` §8.6

Relacionado: g1-limpo-botar-alinhado-pune-endireitar
