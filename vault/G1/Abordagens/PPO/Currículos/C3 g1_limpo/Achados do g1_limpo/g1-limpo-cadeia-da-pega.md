---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "A cadeia mao->laterais->preensao->unload->alvo do g1_limpo: onde ela quebrava e o que a consertou"
---
O g1_limpo travou a pega por TRES anti-hacks perdidos na reescrita do g1_poc. Os tres
sao geometria, e nenhum e peso. Consertados em 2026-08-28, commit `917bf38`.

**1. O alcance era `min` das palmas contra uma esfera no centro.** Com `min`, uma mao
satura o kernel e a segunda nao tem gradiente. Mas o `squeeze` e `min` das FORCAS e
exige as duas. A cadeia ficava sem ponte entre "uma mao encosta" e "as duas apertam".
Hoje cada palma mira a SUA face lateral e o kernel usa a MEDIA.

**2. A lista da mesa estava invertida.** `.*_collision` cobria punho e cotovelo, que
tem de chegar perto do tampo. Nao cobria os pads, porque `add_pads_de_palma` apaga
`*_hand_collision` e os pads terminam em `_pad`. Aproximar terminava; escorar com a
palma era gratis.

**3. O `precise_ori` era inerte.** A direcao pedida era viva em todo elo, e o alvo se
movia com o ROBO. Hoje so o `REORIENTAR` pede direcao viva; os outros congelam a normal
da abertura e pagam por erguer SEM torcer.

**Why:** o gradiente existia em todo termo isolado. O que faltava era a LIGACAO entre
eles, e um elo sem gradiente anula os posteriores.

**How to apply:** ao ler pega travada, compare a cadeia termo a termo com o g1_poc
antes de mexer em peso.

Relacionado: g1-limpo-mesa-contato-zero, g1-pegar-plato-do-grasp-booleano
