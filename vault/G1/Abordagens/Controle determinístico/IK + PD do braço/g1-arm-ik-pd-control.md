---
tipo: achado
pai: "[[IK + PD do braço]]"
estado: ativa
verificado: 2026-10-06
resumo: "Projeto pessoal unitree-g1-sim (MuJoCo), src/sim.py: mover a mão do G1 até um ponto XYZ."
---
# g1-arm-ik-pd-control (original)

Corpo original da memória `g1-arm-ik-pd-control`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Projeto pessoal `unitree-g1-sim` (MuJoCo), `src/sim.py`: mover a mão do G1 até um ponto XYZ.

**Método (correto, verificado 2026-07-10):** IK numérica por Jacobiano de diferença finita usando `mj_forward` como oráculo de FK (tip(q)→pos da mão no frame do torso), damped least squares (`J.T (JJ.T+λ²I)⁻¹ e`, λ=0.05), resolve UMA vez → ângulos alvo → PD de torque (`τ=kp(q*−q)+kd(−q̇)`, escrito em `data.ctrl`). É o mesmo esquema do exemplo oficial `deploy/deploy_mujoco/deploy_mujoco.py` (que na verdade é uma policy RL de LOCOMOÇÃO — mexe as 12 pernas, segura os braços parados; NÃO é exemplo de braço).

**Root cause do "braço chicoteava / mão ia pra baixo":** ganhos PD absurdos meus (cheguei a kp=1500, kd=100). Ganhos oficiais do G1 (de `deploy/deploy_real/configs/g1.yaml`, braço direito, ordem sh_pitch,sh_roll,sh_yaw,elbow,wr_roll,wr_pitch,wr_yaw): **kp=[100,100,50,50,20,20,20], kd=[2,2,2,2,1,1,1]**.

**Como a Unitree controla (deploy_real.py):** manda por motor `(q, qd, kp, kd, tau)` e o firmware do motor faz o PD `τ=kp(q*−q)+kd(qd*−q̇)+tau_ff`. Não manda torque do host — manda ângulo-alvo + ganhos. Mover braço = IK→ângulos→PD-no-motor.

**SOLUÇÃO FINAL (implementada 2026-07-10, verificada headless):** no `sim.py`, converter os atuadores do braço em **servos de posição** (espelha o motor real) + **integrador implicitfast**. Resultado: oscilação 0.0 em TODAS as juntas, vel=0, mão no alvo (~20mm de droop por gravidade, aceitável; some se subir kp ou add gravity-comp). Setup: `model.opt.integrator=mjINT_IMPLICITFAST`; por atuador do braço `gaintype=FIXED, biastype=AFFINE, gainprm=[kp,0,0], biasprm=[0,-kp,-kd]`; no loop `data.ctrl[ra_ctrl]=ra_target` (rampa suave até ra_goal). Euler explícito NÃO estabiliza (wrist_roll zunia ±6°); implicitfast resolve.

**Fatos do modelo (bench.xml):** atuadores são motor/torque (gainprm=1, força ilimitada); integrador Euler explícito, dt=0.002; base flutuante existe (qpos[7:], qvel[6:]); pelvis+pernas+cintura travadas por 16 equality constraints. IK só-posição = Jacobiano 3×7 (redundante); IK 6D (com orientação) pode pedir wrist_pitch>92° e estourar limite → preferir posição, e orientação só com alvo achável. Ver unitree-g1-sim-mujoco-not-isaac.

**Movimento do braço (07-11):** o jeito Unitree = **interpolação em ESPAÇO DE JUNTA** entre poses-chave (resolve IK 1× por pose → interpola os ângulos com ease senoidal → servo segue). Robusto, sem singularidade, sem droop dinâmico; a mão faz curva suave (não reta). É o `rt/arm_sdk` deles (comandam junta, não cartesiano; IK fica com o usuário/teleop). O `weight` (motor_cmd[29].q, 0→1) deles = engajar suave; espelhei com low-pass no comando. **Linha reta cartesiana** exige IK-por-quadro (fizemos): funciona mas é sensível — droop de gravidade (matável com feedforward `qfrc_applied=qfrc_bias`, = campo `tau` do motor real) + lag dinâmico do servo + inalcançabilidade se a linha sai do workspace. Alto nível deles p/ gestos = `G1ArmActionClient` (ações prontas: acenar/apertar mão).

**Estilo de trabalho aprendido aqui:** parei de chutar e rodei a IK/dinâmica **headless** pra achar o bug — fazer isso ANTES de editar sim.py. MAS: gastei ~20 sims headless perseguindo métrica de fidelidade de linha; pra exercício de aprendizado, implementar e deixar o USER ver no viewer é melhor que otimizar métrica offline. Ver feedback-como-responder.
