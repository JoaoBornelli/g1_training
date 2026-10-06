---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
---
# g1-curriculo-congelamento-escala (original)

Corpo original da memória `g1-curriculo-congelamento-escala`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Medido 2026-08-05 com o harness ruidoso da S0 (`g1_multitask/sim_curriculo.py --ruido`,
`p` por célula constante no tempo, Bernoulli).

**1. O congelamento por viés do máximo corrido é efeito de POUCAS AMOSTRAS, não da regra.**
Varredura: `num_envs=8` → 2 células congeladas por viés; `num_envs=64` e `512` → zero.
A causa é o `_medir` (`curriculum.py:219`): ele alimenta a EMA com a **média dos envs no
nível**, não com uma amostra Bernoulli. Com `m` envs por nível, σ da EMA é
`√(p(1−p)/m) · √(α/(2−α))`. A S3 dimensiona o conserto para σ = 0,062, que pressupõe
`m ≈ 1`. Em `m = 32` dá 0,011, e o viés de 2,5–3σ não chega perto do `congela_queda` de
0,10. **O treino usa 4096 envs por rank** → o fenômeno se dilui sozinho.

**2. A célula `(PARADO, "push")` congelava em QUALQUER escala, por causa diferente.**
Ela media `sucesso.mean()` sobre todos os envs de **todas** as tarefas. Abrir uma
tarefa nova mais difícil derrubava a média global de verdade — medido: 0,90 → 0,696,
queda 0,206. Não era viés. Trocar o máximo por EMA lenta (o que a S3 faz) **não
resolveria**: a referência lenta converge para 0,90 e a queda de 0,206 continua
passando o limiar de 0,10.

O efeito real não era o congelamento (que é inócuo: o `_amostrar` pula `PUSH` no
`_dist`, e `T.AXES[PARADO]` é vazio, então o bloqueio por congelado não morde). O
efeito era `perf[push]` ser comparada com o limiar **absoluto** de 0,90 no
`_destravar_push` e no `_push_competente` — inatingível com população mista.

**CORRIGIDO em 2026-08-05, junto com a S3** (decisão do user): a célula passou a medir
só os envs cuja tarefa sorteada é `parado`. A população fica estável e a queda volta a
significar regressão. Na Fase 0 o comportamento é idêntico ao anterior, porque só o
`parado` está aberto. Depois da correção o harness da S0 dá **zero** congelamentos em
todos os `num_envs` (8, 64, 512), contra 3/1/1 antes.

**3. Intervalo entre destravamentos cresce 1,26×** da 1ª para a 2ª metade da run
(148 → 186 chamadas, `num_envs=8`). Registrado sem veredito — a S0 manda registrar e
seguir.

Contexto histórico: o congelamento espúrio observado em campo em 30/07 foi na mesma
célula de push, mas a causa registrada ali foi outra (prior otimista de 0,5 na EMA
inicial), e já está corrigida — ver comentário em `curriculum.py:101-108`.

Limitação do harness, registrada no código: dentro de uma tarefa o `p` efetivo é o
**mínimo** entre as células dela, então a célula que não é o mínimo mede o `p` da que é.

Ver g1-multitask-curriculum-design.
