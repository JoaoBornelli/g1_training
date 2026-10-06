---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "Sub-projeto de percepção em unitree-g1-sim/Vision_training/ (decidido 2026-07-14)."
---
# g1-vision-cube-orientation (original)

Corpo original da memória `g1-vision-cube-orientation`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Sub-projeto de percepção em `unitree-g1-sim/Vision_training/` (decidido 2026-07-14).
Goal do robô-mãe: robô vira uma tag (face branca do cubo) pra câmera. Este
experimento é só o degrau de percepção.

**Goal deste experimento (travado):** 1 imagem RGB de um cubo na mesa → CNN simples
regride a ORIENTAÇÃO do cubo → desenhar gizmo (3 eixos) sobreposto pra validar no
olho. Só orientação — position e tag ficam FORA por enquanto.

**Decisões de desenho:**
- Loop REATIVO single-image (não temporal). Comportamentos futuros do robô: vejo
  tag→viro; não vejo→giro até achar. Robô adiado.
- Cubo liso = grupo de simetria octaédrico (24 rotações). User NÃO liga pro rótulo
  dos eixos ("x pra frente ou y pra cima tanto faz") → aceita a ambiguidade de 24.
  Gizmo gruda nas arestas em cada imagem, mas o eixo vermelho pula entre faces
  equivalentes ao trocar de imagem — comportamento CORRETO, não bug.
- **Loss = distância geodésica com mínimo sobre as 24 rotações** (senão o treino
  briga consigo mesmo). **Representação de rotação = 6D contínua** (Zhou 2019), não
  quaternion cru (dupla-cobertura/descontinuidade estraga regressão).
- Alvo = orientação no FRAME DA CÂMERA (R_cam_obj = R_cam_world·R_world_obj); câmera
  fixa no XML → dá pra compor. Gizmo sai direto em coords de vista.
- Modelo: ResNet18 pré-treinada + cabeça MLP → 6 números. N do dataset 500→~5000.
  Cropar/centralizar no cubo (pequeno no frame). Sucesso = overlay visual no test
  set + erro geodésico módulo-simetria (graus, mediana+histograma).
- **A tag é o ÚNICO landmark que quebra a simetria.** Ela volta num passo seguinte
  se quiser canonicalizar o gizmo (eixo fixo na face da tag) ou fechar o loop do
  robô. Manipulação (posicionar mãos) só precisa de orientação-módulo-simetria, que
  é a parte recuperável da geometria — a ambiguidade não atrapalha agarrar.

**RESULTADO — RESOLVIDO (2026-07-14).** Experimento FECHADO com sucesso:
`val geo med 1.4°`, linha CHECK confirma o `.pt` recarregado = 1.4°, overlay predito
gruda nas arestas no test set. Modelo final = **SmallCNN do zero** (4 conv stride-2 +
GAP + Linear→6, ~0.5M params), OUT=64, N=7500, 20 épocas, LR=1e-3 — treina em MINUTOS
em CPU local (ResNet18 em CPU era inviável, ~horas; user está em CPU, sem GPU local).

Bug pelo caminho (resolvido): o código salvava o modelo da ÚLTIMA época e o treino
divergia no fim → `.pt` era lixo (~40°, nível de chute) apesar do log mostrar 2° no
meio. FIX em `box_scene.py`: `train()` salva o MELHOR (não o último) + verificação
CHECK (recarrega o `.pt` e reavalia) + removido o save do último no `__main__`.
Também: `RESUME` flag (continua do melhor, sessão aditiva, seed do best pra não
degradar); rodar background com `python -u` (senão stdout bufferiza e log fica vazio).

Modelo final `modelo_orient.pt` é da SmallCNN (incompatível com o `.pt` ResNet antigo).
DOMAIN RANDOMIZATION adicionada (2026-07-14, tarde): trocar a cor da mesa quebrou o
modelo (decorou aparência). Adicionado ao box_scene.py: flag `DR` (randomize_scene:
cor de mesa/chão/cubo + luz por amostra) e flag `CAM_DR` (randomize_camera: pose da
câmera aleatória, mais perto ~0.45-0.9m, olhando pro cubo). IMPLICAÇÃO RESOLVIDA:
câmera variável quebra o recorte de câmera-fixa → agora cada label grava seu próprio
recorte `crop:[u,v,half]` (crop_uv_half: normalizado pela distância, cubo preenche
~fração constante). CubeDataset lê `crop` do label; live_test.py usa o mesmo helper.
`quat_cam` continua correto de graça (calculado da cam_xmat real do frame). Smoke test
(4 amostras) confirmou gizmo GT grudado + recorte enquadrando com câmera/cor variadas.
DATASET ANTIGO (sem chave `crop`) FICOU INCOMPATÍVEL → precisa regerar. Com DR o task
fica mais DIFÍCIL → esperar val err > 1.4° (era cena fixa fácil); o ganho é generalizar.
CANONICALIZAÇÃO VIA TAG (Opção A) — IMPLEMENTADA (2026-07-14, smoke ok): SmallCNN
virou backbone + 2 cabeças: head_orient (6D, loss simétrico, geometria mod-sim) +
head_tag (4 = [vis_logit, normal_xyz]). Labels da tag DERIVADOS do quat_cam (normal =
coluna-y de R_cam_obj; visível = dot(normal,-O_cam)>0, O_cam reconstruído do recorte)
→ SEM re-render, dataset atual serve. Perdas: sym_loss (todas) + BCE visibilidade
(todas) + cosseno da normal (mascarado só p/ visíveis). evaluate() reporta orient med
+ vis_acc + erro da normal(°). desenhar_predicoes mostra normal da tag como seta rosa
(cinza se não-visível). Tag mantida PEQUENA (0.02x0.01) a pedido do user → risco: ~8px
a 64, se vis_acc/normal patinarem, subir OUT ou aumentar tag. Smoke: fração visível
~0.40 (BCE balanceado). TREINADO (2026-07-14, CPU, 80 ép.): orient med 2.1°,
vis_acc 0.93, tag normal 2.6°, CHECK bate. Tag head funciona MESMO com tag pequena
(~8px); ~7% de erro de visibilidade = casos de raspão (tag de perfil, ambíguos).
Canonicalização entregue: orientação + tag visível? + direção da tag. Se quiser
vis_acc maior: tag maior / OUT maior. `real_test.py` criado (sim-to-real: webcam/vídeo,
recorte central manual, gizmo ortográfico) pra probe qualitativo — espera lacuna.

INSIGHT (memória de exploração do robô): a percepção é REATIVA (sem memória). "Quais
faces já olhei" é estado do CONTROLADOR, e a fonte é PROPRIOCEPÇÃO: quando o robô
segura a caixa, ela fica soldada à garra → orientação da caixa = cinemática da mão,
sabida sem visão. O vetor [-1/0/1] das 6 faces (ideia original do user, dia 1) é o
estado de crença do controlador (no frame da caixa), não a saída da rede — camada
certa. Visão só responde "tag apareceu agora?"; giro é comandado/integrado.
Próximo (futuro): position se as mãos precisarem; orquestrador FSM.
Colab (se quiser velocidade um dia): MUJOCO_GL=egl antes do import; flags via sed no
ARQUIVO (célula Python de assign não afeta subprocesso `!python`).

**Modo de trabalho:** aprendizado hands-on — eu guio passo a passo, user coda e
adapta. Não rodar por ele, não escrever spec. Ver feedback-como-responder,
feedback-como-trabalhar, feedback-quem-executa.
Base atual: `box_scene.py` (gera dataset, hoje loga face_cima) + `box_scene.xml`.
Relacionado: g1-lift-box-task (task-mãe de manipulação).
