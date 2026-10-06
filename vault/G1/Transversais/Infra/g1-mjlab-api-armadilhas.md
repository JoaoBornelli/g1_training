---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "robot.data.root_com_pos_w NÃO é o centro de massa do robô."
---
# g1-mjlab-api-armadilhas (original)

Corpo original da memória `g1-mjlab-api-armadilhas`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

`robot.data.root_com_pos_w` NÃO é o centro de massa do robô. É `xipos` da pelve, portanto gameável: o robô desloca o resto do corpo e a métrica não vê. O CoM real é `robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]`.

Qualquer entidade FIXED-BASE (sem freejoint) é auto-embrulhada em MOCAP pelo mjlab. Ela vira corpo cinemático, posicionável por env em runtime com `write_mocap_pose_to_sim`, e flutua em qualquer altura sem tocar o chão. É assim que se faz prateleira de altura variável por env — a alternativa, altura em tempo de compilação, não dá por env.

⚠ Entidade sem evento de reset fica na ORIGEM DO MUNDO para todos os envs.

`dr.body_mass` e `dr.pseudo_inertia` existem no módulo `dr` mas não são usados em nenhum treino do fabricante. Os dois CORROMPEM A HEAP (`corrupted size vs prev_size` no CPU, `CUDA illegal memory access` na GPU). As massas saem corretas e a corrupção vem no teardown.

**How to apply:** para peso variável, use força externa (`write_external_wrench_to_sim`) com `mode="reset"`. É o mecanismo testado. Publique a carga em kg, não em newtons — publicar newtons obriga o consumidor a desfazer a conta, e é assim que se erra um fator 9,81 em silêncio.

Mais quatro, medidas em 2026-08-26 montando a F1 do `g1_limpo`:

- **`SceneEntityCfg` só é resolvido se estiver em `params`** (`manager_base.py:141-145`). Como argumento default de função ele NUNCA resolve: `site_ids` fica `slice(None)` e o termo lê todos os sítios do robô em vez dos dois pés. Falhou por sorte de forma (6 contra 2); com contagem coincidente passaria medindo a coisa errada.
- **`reward_manager.py:174` só chama `reset()` em termo de classe que TENHA `reset`.** O `feet_swing_height` do fabricante não tem, portanto o `peak_heights` dele atravessa o fim do episódio e o pico do pé que estava no ar quando o robô caiu infla a métrica do episódio seguinte. Consertado por subclasse de 3 linhas.
- **O mjlab constrói termo de comando por `cfg.build(env)`** (`command_manager.py:268`), e NÃO por `class_type`. Um `class_type` numa subclasse de cfg é campo morto: o cfg passa, o manager chama o `build` herdado, e o treino roda com o termo do fabricante sem uma linha de erro.
- **`Policy/mean_noise_std` NÃO existe no rsl_rl 5.4.0** — o logger escreve `Policy/mean_std` (`rsl_rl/utils/logger.py:211`). A escada de corte do `g1_poc` usa o nome errado, portanto aquela linha nunca disparou. Prefixos reais: `Episode_Reward/<termo>`, `Episode_Metrics/<chave>`, `Metrics/<termo_de_comando>/<metrica>`, `Curriculum/<termo>`.

Mais tres, medidas em 2026-08-26 montando a F2:

- **`observation_manager.compute()` devolve CACHE** (`observation_manager.py:311`) a menos que se passe `update_history=True`. Num teste isso e' venenoso: le-se o passo anterior e o teste passa com o codigo errado. Aconteceu comigo ao provar que o one-hot e' por passo.
- **O normalizador do rsl_rl nao tem clamp:** `(x - _mean) / (_std + 1e-2)` (`rsl_rl/modules/normalization.py:48`). Canal constante em zero faz `_std -> 0`, e o primeiro 1,0 entra na rede como **100,0**. Consequencia de desenho: nenhum canal de one-hot pode ficar constante por muitas iteracoes se for acender depois.
- **A ordem de reset e' curriculo -> eventos -> comando**, e as tres linhas sao `curriculum_manager.compute` (`manager_based_rl_env.py:554`), `event_manager.apply(mode="reset")` (`:560`) e `command_manager.reset`, que resampleia (`:581`). O bloco de `manager.reset(env_ids)` de 566 em diante e' contabilidade/logging, nao execucao dos termos -- nao confundir. Quem precisa ser lido por evento E por comando tem de nascer no curriculo.

Mais uma, e ela e a mais caruda que apareceu (2026-08-26, achada por code review):

- **Termo de CURRICULO nao roda uma vez por iteracao de PPO — roda a cada passo em que ALGUM env reseta.** `curriculum_manager.compute` e chamado do `_reset_idx` (`manager_based_rl_env.py:554`), e o `step()` chama o `_reset_idx` sempre que ha env terminando. MEDIDO com 128 envs: 48,8% dos passos em 400 passos, e com 4096 envs tende a 100%. Portanto **`contador += 1` num termo de curriculo conta PASSOS**, e qualquer knob escrito em "iteracoes" erra por ~24x (o `num_steps_per_env`). No g1_limpo isso fazia a carencia de 200 iteracoes vencer em ~17 e a rampa de 396 colapsar em ~34.
  **How to apply:** derive a iteracao de `env.common_step_counter / num_steps_per_env`. Bonus: o mjlab JA persiste o `common_step_counter` no checkpoint (`mjlab/rl/runner.py:73`), logo a contagem fica resume-safe de graca. E se o termo dispara acao por cadencia, guarde a JANELA ja disparada — senao ele dispara ~24x por iteracao.

## O `std` é POR JUNTA, e `Policy/mean_std` é só a média (verificado 2026-08-27)

`rsl_rl/modules/distribution.py:166` faz `nn.Parameter(init_std * torch.ones(output_dim))`.
São **29 valores**, um por junta, aprendidos de forma independente. O `std_type="scalar"`
nomeia a PARAMETRIZAÇÃO (valor direto em vez de log), não o formato — é armadilha de nome.

Logo `Mean action std = 0,45` não diz nada sobre um subconjunto: os 14 canais de braço podem
estar em 0,8 com as pernas em 0,2. O mjlab loga só a média. Para ver os 29, leia
`std_param` do `state_dict` do checkpoint — não precisa de GPU nem de simulador.

E a entropia **pode subir**: o `std` é peso treinável (`learn_std=True`), `std_range` é
`(1e-6, 1e6)` sem piso útil, e `ppo.py:278` já paga `− entropy_coef × entropy` com
`entropy_coef = 0.01` na config do G1. A queda é EQUILÍBRIO (perda marginal de recompensa =
`entropy_coef`), não decaimento. O `schedule="adaptive"` ajusta a LR pelo KL, não a entropia.

Relacionado: g1-limpo-desenho, g1-leitura-log-e-kaggle, g1-mjlab-reset-buffers-obsoletos, g1-mjlab-mutacao-em-treino, g1-lift-box-task
