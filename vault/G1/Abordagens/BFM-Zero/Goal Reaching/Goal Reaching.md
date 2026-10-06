---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23/07–06/08/26
resumo: "Entregar a pose alvo por goal_inference, em 6 experimentos. Rejeitada: o BFM passa pela pose profunda e não a segura."
cluster: bfm
---
## Hipótese
Entregar a pose alvo (ângulos de junta) em vez de uma recompensa: `goal_inference` mapeia a pose direto para `z = project_z(B(pose))`, sem buffer, e o BFM cuida do equilíbrio.

## Método
Seis experimentos em sequência, cada um respondendo a falha do anterior: poses à mão (Teste de goal), pose gerada por IK (IK gera pose), viabilidade estática da pose (Viabilidade da pose), mapa comandado → alcançado (Mapa comandado-alcançado), malha fechada com overdrive (Goal servo) e a ferramenta por bloco (Goal track).

## Resultado
O BFM chega perto de poses em pé e de poses de braço. Nas poses profundas ele passa pela pose e não a segura, ou cai. A mão não fica abaixo de ~0,54 m de forma estável.

## Por que parou
`goal_inference` é um forward do encoder B mais normalização: não há otimização rumo ao alvo, não há realimentação, e a magnitude é descartada. A única alavanca é onde se põe o goal, e mover o goal gira o `z` em vez de empurrar mais. Causa de fundo: a pose não está no LaFAN.

## O que herda
- Uma pose de IK são 29 ângulos MAIS o quaternion da base. Passar só os ângulos dá outra pose.
- Antes de culpar o controlador, verificar se a pose é estaticamente viável.
- Medir o erro por bloco (pernas, cintura, braços), nunca a média dos 29.

## Filhos
<!-- filhos:inicio -->
- **Goal servo** (abordagem · rejeitada · 24/07/26) — Goal em malha fechada com overdrive integral. Rejeitado: mover o alvo gira o z em vez de empurrar mais.
- **Goal track** (abordagem · rejeitada · 06/08/26) — Ferramenta de pose por slider com erro por bloco agora × melhor; mostrou que o robô passa pela pose e vai embora.
- **IK gera pose** (abordagem · rejeitada · 23/07/26) — IK de corpo inteiro gera a pose de pega e o BFM executa; o BFM caía.
- **Teste de goal** (abordagem · rejeitada · 23–24/07/26) — Poses montadas à mão viram z; braços e frames do dataset funcionam, agachar não (a pose 6 era impossível).
- **Mapa comandado-alcançado** (achado · ativa · 24/07/26) — Mão comandada de 0,70 a 0,10 m: piso em ~0,54 m (meio abaixado) ou queda e robô deitado (pelve 0,38 m); sem nada entre os dois.
- **Viabilidade da pose** (achado · ativa · 24/07–06/08/26) — A pose 6 tinha o CoM 5,4 cm fora do apoio; ferramentas find_pose, pose_check, pose6_reach, reach10 separam falha de controle de pose impossível.
<!-- filhos:fim -->
