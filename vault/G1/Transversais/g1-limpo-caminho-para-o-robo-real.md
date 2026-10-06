---
tipo: achado
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "o que falta para o treino do g1_limpo rodar no G1 físico — lidar, mãos Dex3 e DR agressiva; anotado pelo dono em 2026-09-16"
---
Como está hoje (bloco 21, 2026-09-16) o treino NÃO roda no robô real. Três frentes obrigatórias antes do modelo que vai para o físico:

1. **Lidar na observação.** O robô real precisa do lidar para saber onde é o chão; a sim dá o chão de graça.
2. **Mãos reais.** A sim usa dois pads de borracha (`cena.add_pads_de_palma`); o robô tem a Dex3 de três dedos. Muda a geometria de contato e os sensores de palma.
3. **DR agressiva.** Atrito das mãos, posição da caixa, erro de leitura da posição da caixa, entre outros. Sem isso o treino é intransferível.

**Why:** a política otimiza um mundo com chão conhecido, mão fictícia e caixa sem ruído.
**How to apply:** estas três vêm antes de qualquer refino de recompensa. A cápsula do punho (35 mm × pad 23 mm) some com a frente 2.

Relacionado: g1-limpo-forma-postural, g1-limpo-inercia-real-da-caixa, g1-limpo-batente-e-restricao-mole
