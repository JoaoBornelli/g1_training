---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
resumo: "SPEC S0-S15 do g1_multitask implementada 05/08 — as 4 decisões que divergiram da spec e os 2 números que ela não deu"
---
# g1-multitask-spec-validacao-2026-08 (original)

Corpo original da memória `g1-multitask-spec-validacao-2026-08`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

SPEC de 16 itens (S0–S15) para a rodada de validação do currículo, implementada em
2026-08-05. Branch pretendida `exp/curriculo-teste`, treino na Kaggle, PPO puro sem
BFM. Smoke passa 3/3; `sim_curriculo` reproduz o desenho.

S3 e S4 já estavam aplicadas no repo quando comecei; as outras 14 eu escrevi.

## As 4 decisões que divergiram da letra da spec

**S6 — histerese, não trava permanente.** A spec dá dois raios (0,10 dispara / 0,25
mantém) e também diz que `chegou` "trava pelo episódio". As duas coisas não fecham: com
trava, o raio de manutenção não teria uso. Implementei o *disparo* travado e a condição
reavaliada contra o raio folgado. Fecha o buraco de "chega, sai andando para longe, e
ainda pontua".

**S2 — o push sustentado escala pelo NÍVEL, não por `env.push_fator`.** O
`apply_body_impulse` do mjlab sorteia com `force_range` global e não expõe o wrench;
fator por env exigiria reimplementar os timers, e a S2 manda reusar. O aninhamento que
a spec quer continua valendo, porque o sorteio interno já cobre magnitudes pequenas.
O `empurrao` (velocidade) usa o fator por env normalmente.

**S1 — precisei escrever um evento que a spec não previu.** O `reset_scene_plr` escreve
o quaternion de `default_root_state`, ou seja identidade: trocar o `reset_box` por ele
apagaria o jitter de yaw de ±15° de que o `reset_segurando` depende. Criei
`jitter_cena` (a S12 depois o expandiu para o jitter da mesa).

**S8 — o `botar` perdeu gradiente inicial.** Sem a janela de pré-gatilho, o
`box_at_peito` deixa de valer para ele (não está no gate). Sobram `reaching` e
`box_at_prateleira`, que são contínuos. Confirmar em `Contrib/botar/*`.

## Os 2 números que a spec não deu, e a derivação

- **S12, yaw da mesa** ("um pouco de yaw"): usei ±15°, o mesmo `box_jitter_yaw_deg` já
  calibrado. Evita uma segunda escala angular para a mesma cena.
- **S13, faixa de atrito** ("perto do nominal"): μ ∈ (0,8, 1,2), ±20% do nominal 1,0
  (`common/box.py:26`). Conservador de propósito — randomização acrescenta variância ao
  `success_buf` e mais variância significa mais congelamento espúrio.

## O achado que pode invalidar a S9

`terminacao = -200` passa pelo `scale_by_dt` (`reward_manager.py:127`), portanto o custo
REAL de uma queda é **−4,0**, não −200. Um episódio de `parado` rende ~80 com o piso
ligado, então a queda custa ~5% disso. Se o −200 do ULC media penalidade efetiva, falta
um fator 50. A aceitação da S9 decide: se o comprimento médio do episódio não subir no
bloco de 300 iterações, o número está fraco demais.

## Duas armadilhas de teste que voltam a morder

1. **Buffer stale depois de escrita de pose.** Ler `root_link_pos_w` logo depois de
   `reset_scene_plr` traz a pose do episódio ANTERIOR. Sem `env.sim.forward()`, a caixa
   do `pegar` ia para `z = 0,035` com `desvio_xy = 1,50`.
2. **Checks de perfil não podem usar `env.step`.** Com ação nula o robô cai, o episódio
   termina dentro do `step`, e o reset re-sorteia o alvo forçado. Chamar
   `_update_command` direto torna o teste determinístico.

Ver g1-multitask-curriculum-design e g1-curriculo-congelamento-escala.
