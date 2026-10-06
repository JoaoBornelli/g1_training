---
tipo: estudo
pai: "[[Estudos]]"
estado: ativa
verificado: 2026-10-06
periodo: 05/10/26
resumo: "Revisão de 05/10: ninguém dosa o aperto sem tátil; três mecanismos (teto de força no contato, mola virtual no PD, torque L2 pequeno); todo G1 real usa mão de borracha."
cluster: ppo
---
Revisão feita por agente em 05/10 para decidir o preço do aperto e a DR de atrito. Os papers foram lidos pelo HTML do arXiv com um resumidor: confirme cada número no paper antes de copiar.

## Como dosam o aperto

| Mecanismo | Fonte | Detalhe |
|---|---|---|
| Teto de força na recompensa de contato | [HDMI](https://arxiv.org/abs/2509.16757) (G1 real) | `exp(−‖p_eef − p_alvo‖/σ_pos) · max(exp((‖F‖ − F_lim)/σ_frc), 1)`; o limite existe para segurança no deploy |
| Limite de força no alvo do PD (mola virtual) | [GentleHumanoid](https://arxiv.org/abs/2511.04679) (G1, sem sensor de força) | `F = Kp·(x_alvo − x)` escalada acima do limite; 5 a 15 N; abraço a ~10 N contra > 20 N do RL comum |
| Penalidade de torque pequena | todos | L2 de 1e−7 ([Embracing Bulky](https://arxiv.org/abs/2509.13534)) a 1e−4 ([PhysHSI](https://arxiv.org/abs/2510.11072)); excesso sobre o limite −0,5 e −500 ([WoCoCo](https://arxiv.org/abs/2406.06005)) |

## Outros dados

- [Digit box](https://arxiv.org/abs/2310.03191): sim2real falhou por viés de orientação e caixa de 8 kg; corrigido com ajuste manual do CoM.
- ResMimic: G1 real com 4,5 kg, corpo encosta; contato sem teto (nota própria).
- [PhysHSI](https://arxiv.org/abs/2510.11072): G1 real, 0,6 a 3,6 kg, mão de borracha; falha acima de 3,6 kg por superaquecimento e deslize.
- [SplitAdapter](https://arxiv.org/abs/2606.03297): G1 real com luvas de alto atrito, 0,5 a 5,0 kg; adaptação à carga 26/27 contra 16/27.
- [Lin et al. 2025](https://arxiv.org/abs/2502.20396): GR-1, caixa bimanual; ajusta sim ao real em menos de 4 min.

## Leitura para o G1

O teto de força do HDMI é o freio de aperto que testamos no resume do 24350, com precedente em G1 real. A mola virtual do GentleHumanoid é a mesma ideia na ação e limita `Kp·Δx` antes de saturar os 25 N·m do ombro. Todos os trabalhos com G1 real usam borracha ou luva de alto atrito: motivou a decisão da palma de plástico com μ 0,3–0,5 na v3. Deslize sem tátil em humanoide, em tempo real: nada publicado.
