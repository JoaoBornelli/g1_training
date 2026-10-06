---
tipo: achado
pai: "[[Controle determinístico]]"
estado: ativa
verificado: 2026-10-06
resumo: "G1 da UDESC simulado local — MuJoCo porque a máquina não tem NVIDIA; no robô real equilíbrio e marcha são serviço embutido, e os atuadores são torque puro"
---
Projeto do G1 da UDESC em `~/Documents/unitree-g1-sim/`. A máquina do user tem iGPU Intel e zero NVIDIA, então Isaac Sim é inviável e o caminho é MuJoCo em CPU.

Treino precisa GPU; inferência não. Uma política exportada em ONNX roda em CPU para sempre.

Não existe política de 29 DoF pré-treinada rodável em CPU. Os projetos distribuem o framework de treino, não o `.pt` — um checkpoint é inseparável do contrato de observação que o gerou.

No robô real são três camadas. Equilíbrio e marcha são serviço embutido proprietário (`LocoClient.StandUp()`, `Damp()`, `Move(vx,vy,yaw)`); os braços vão por `rt/arm_sdk` com campos `q`, `dq`, `kp`, `kd`, `tau`. Transferem do sim a percepção e o comando de braço. Não transferem equilíbrio e marcha.

Os atuadores do G1 são `motor`, torque puro. Não há PD de fábrica, então `ctrl = 0` é ragdoll e todo músculo é código do user.

**How to apply:** referencie junta e atuador por NOME (`model.actuator(i).name`), nunca por índice cru. O mapa é 0–11 pernas, 12–14 cintura, 15–28 braços.

Relacionado: g1-arm-ik-pd-control, g1-bfm-zero, feedback-quem-executa
