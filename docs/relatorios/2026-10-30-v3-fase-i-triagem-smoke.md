# Fase I: triagem das 21 falhas do smoke (30/10)

Entrada: `smoke2.txt` (142 ok / 21 falhas). Cada falha foi rodada de novo, isolada, em CPU, com scripts de
diagnóstico no scratchpad (`tri/d*.py`, `tri/fix*.py`). Cada conserto abaixo foi provado: a seção com o conserto
aplicado (por monkeypatch, sem editar o pacote) passou.

## Resumo

- **Do código: 2 das 21.** `eventos.py:242` (o empurrão quebra o treino em 5 s) e `observacoes.py:144` (a task
  `Mjlab-G1-V3-SemPercepcao` não monta). Os dois são uma linha cada; nenhum muda o comportamento esperado do robô.
- **Do check: 19 das 21.** Três falhas a mais estavam escondidas na s36 pela exceção do empurrão. As três também são
  do check.
- **Premissa errada do smoke, causa comum de 7 seções (s06, s18, s20, s21, s28, s29, s36).** Com o
  `Contrato.pd` (PD da Unitree; tornozelo kp 40), a ação zero NÃO segura o G1 de pé. Sem DR nenhuma (massa,
  viés, atraso e atrito zerados), o robô cai em ~1,1 s: tronco a 7,5° no passo 30, 38° no passo 50, `queda` no
  passo 52, igual nos 4 mundos (`tri/dpe.py semdr`). A conta: a rigidez dos dois tornozelos é 2 × 40 = 80 N·m/rad,
  e o pêndulo pede m·g·h ≈ 35 · 9,8 · 0,75 ≈ 257 N·m/rad. O comentário da s06 ("a ação zero deixa o robô
  parado") e a memória "De pé é grátis no mjlab" valem para o PD do mjlab, e não para a v3. Não é defeito: o PD é
  o do robô real (§6.2). O conserto é um auxiliar único do smoke, `_segura_base`, que segura a base antes de cada
  passo nas seções que medem outra coisa.
- **Arquivos:** `g1_v3/eventos.py` (1 linha), `g1_v3/observacoes.py` (1 linha), `g1_v3/smoke.py` (resto). Três
  lanes disjuntas.

## Tabela

| seção | check | causa | prova | conserto |
|---|---|---|---|---|
| 4 | um atuador por junta (0) | check | `smoke.py:183` chama `mj_name2id` com `left_hip_pitch_joint`, mas no modelo o nome é `robot/left_hip_pitch_joint`, então volta −1 e nenhum atuador casa (`tri/d04.py`). Com o prefixo: 29 achados, kp e kd todos certos | S1 |
| 6 | ≥ 90% da frente vista (0,405) | check | O robô cai na passagem (passos 33–38: `queda` nos 4 mundos, `tri/d06b.py`). O reset zera o mapa. No passo 28, antes da queda, a fração é 0,95–0,98. Com a base segura e a velocidade zero: 0,97–0,99 | S0, S2 |
| 6 | alturas vistas a ±2 cm | check | O ruído do lidar está ligado: σ_alcance 2 cm e 0,5° de inclinação a 2–3 m dão σ ≈ 2–3 cm por leitura. Medido: média de \|h\| 0,85–0,95 cm; 4–9% das células vistas passam de 2 cm; o máximo de \|h\|/σ_célula é 1,2–1,5 (σ_célula pela confiança, PD-6). O filtro é honesto; ±2 cm por célula é 1σ. O "205 de 210" da I §2.4 não tinha esse ruído | S2 |
| 9 | exceção `sacudida.__init__() … 'tarefas'` | check | `smoke.py:322` chama `termo.func` do `env.cfg.rewards`. O `RewardManager` faz `deepcopy(cfg)` e só instancia a classe na cópia (`manager_base.py:146-147`). Com `reward_manager.get_term_cfg`: todos finitos, vivo = 0,003 = 0,15·dt, contato_fora = 0 | S3 |
| 15 | exceção `KeyError: 'lidar'` | **código** | `observacoes.py:144` lê `K.percepcao.ligada` do `ATIVO` do módulo. A cena sai do `k` passado ao `make_env_cfg` (`env_cfg.py:93` → `cena.py:414`): com `ligada=False` a cena fica sem o sensor, mas a guarda ainda chama `_calcula_percepcao` (`observacoes.py:215`). O defeito não é só do smoke: a task registrada `Mjlab-G1-V3-SemPercepcao` (`tarefa.py:62`) quebra no `__init__`, e o `PERCEPCAO=0` do I5 (D13) não roda. Com o conserto: s15 imprime 212,8 / 203,8 ms por passo; a task monta, ator 565 (= 1 447 − 882), crítico 621 | C2 |
| 18 | soma do que paga positivo = 0 (1,87) | check | Duas falhas de montagem e um invariante errado. (1) `_assenta_fatia_b` não põe a base (`reset_base_por_fatia` fica de fora), e ela fica onde o ANDAR a sorteou (±0,5 m, rumo ±π), às vezes contra a laje. (2) O robô cai (z 0,787 → 0,449 em 50 passos, `tri/d18.py`). (3) O `progresso` é potencial (Ng 1999): o movimento passivo do PD sobe e desce o Φ, e cada subida paga. Com a base posta e segura, Φ oscila de 0,28 a 0,64 sozinho; a soma dos positivos ainda é 1,87, mas o líquido por mundo é −0,14 a −0,35, e nenhum outro termo paga positivo (`tri/fix.py s18`). Pairar soma < 0, como o E §3 "Progressão na cadeia" pede | S4, S0 |
| 20 | idade cresce ≥ 0,4 s com os braços (Δ ≈ 0) | check (premissa do plano E10/20) | Com `q_com_caixa` as palmas ficam ao lado da caixa (y = ±0,16 m), e a câmera vê a caixa entre os braços: 0–2 dos 9 pontos ocluídos, e a visibilidade pede ≥ 3 visíveis (`tri/d20.py`). Ver a caixa durante o carregar é o certo. Com a caixa atrás da base (fora do FOV) e a base segura: Δidade = 0,567 s em 0,6 s; os outros 3 checks seguem ok (`tri/d20f.py`) | S5, S0 |
| 21 | aviso depois de 0,5 s (nenhum) | check | Três camadas. (1) O robô cai: tronco 50° em 1,2 s. (2) Com a base segura, a cintura (kp 40) não segura os braços à frente: tronco +8° e palmas 6 cm fora em 0,36 s. O detector, com tolerâncias de 5° e 5 cm, não dispara, como deve. (3) A ação ignora o viés do encoder (a s04 mostra que o alvo é q_de_pé + escala·a − viés): \|q̇\| médio 0,21 > 0,2 no mundo 0. Com a base segura, as juntas reescritas e o viés somado à ação: aviso nos passos 24–25 (0,5 s) nos 4 mundos, nenhum antes (`tri/d21j.py`, `tri/fix.py s21`) | S6, S0 |
| 26 | z do alvo de soltura ± 1e-4 (4,42e-3) | check | `comando.py:490-497` (P22) soma ao `v3_alvo_w` o viés U[±vz] do alvo, e vz = 5 mm no degrau 0. 4,42 mm < 5 mm. Com a tolerância vz + 1e-4: passa | S7 |
| 27 | 5 cm: fundo a 5 cm do topo | check | A mesma causa: a pose da caixa sai do `v3_alvo_w` com o viés em z. Com o z do topo verdadeiro: as 8 checagens passam | S8 |
| 27 | 10 cm: fundo a 10 cm do topo | check | idem | S8 |
| 28 | aviso do BOTAR ≤ 1,5 s | check | Os mundos 0 e 2 caem antes do aviso: t_aviso = 0 quer dizer "não veio". Os mundos 1 e 3 dão aviso em 0,6 s e trocam 0,06 s e 0,22 s depois, dentro do limite. Com a base segura: as 4 checagens passam | S0, S9 |
| 28 | aviso não antes de 0,5 s | check | idem | S0, S9 |
| 28 | sai do BOTAR até atraso máx + margem | check | idem | S0, S9 |
| 29 | ANDAR final: time_out em todo mundo (−1) | check | O ANDAR final dura 1,5 s e o robô cai em ~1,1 s. A `queda` reinicia, e o episódio novo é o ANDAR de 20 s da fatia A. Com a base segura: time_out no passo 75 (1,5 s) nos 4 mundos, sem queda. O `fim_da_cadeia` está certo | S0, S10 |
| 29 | time_out no fim (± 2 passos) | check | idem | S0, S10 |
| 34 | por_degrau(laje_joelho_N) = 10 (50) | check | `smoke.py:1250` escreve `v3_degrau = 5` e depois chama `_abre_trecho_de`. O `_publica` (`comando.py:450-460`) reescreve o degrau com o do currículo (0). A semântica está certa: `no_degrau(ini, fim, 5, 5)` = fim; os pares do knob são (início; refino). Com a linha depois da abertura: os 3 valores certos, o degrau segue 5 depois de 5 passos, e sacudida e aperto ok | S11 |
| 34 | por_degrau(chegada) = 0,25 (1,0) | check | idem | S11 |
| 34 | por_degrau(sigma_pouso) = 0,35 (1,0) | check | idem | S11 |
| 36 | solref igual entre as geoms de um mundo | check | `_geoms_do_solref` (`smoke.py:1320`) lê o `SceneEntityCfg` do `env.cfg.events`, que o manager não resolve (ele resolve a cópia): `geom_ids = slice(None)`, 72 geoms em vez de 18. As 54 que o evento não toca ficam em 0,02 (`tri/d36.py`). O `eventos.solref` está certo: um u por mundo nas 18 geoms | S12 |
| 36 | exceção `Index put … Float … Double` | **código** | `eventos.py:242`: `episode_length_buf.to(float64) * step_dt` escrito num `v3_t_empurrao` float32 (`curriculo.py:75`). Dispara no primeiro empurrão (5 s) de todo treino e de todo env que passa de 250 passos | C1 |
| 36 (escondida) | wrench da caixa ≠ 0 no env 0 em 450 passos | check | O robô cai a cada ~1,1 s (8 resets em 450 passos). O `apply_body_impulse.reset` sorteia de novo o intervalo de 3–8 s (`mjlab events.py:584-605`), e o impulso nunca chega. Com a base segura: a força age no env 0, e os outros ficam em zero (`tri/d36c.py`) | S0, S12 |
| 36 (escondida) | contatos pé × terreno existem (0) | check | O regex `terrain$` não casa com nenhuma geom: os nomes são `terrain_0` … `terrain_1745`. O `tid` vazio também deixava "solref do terreno = 0,02" verdadeiro sem testar nada. Com `terrain_\d+` (1 746 geoms) e os pés no chão: 66 contatos, solref do contato = solref da geom do pé, terreno em 0,02 (`tri/d36e.py`) | S12 |
| 36 (escondida) | solref do contato = solref do pé | check | idem | S12 |

## Achados para o dono (fase I, sem conserto de código)

1. **A ação zero não segura o G1 de pé com o `Contrato.pd`.** Ele cai em ~1,1 s sem DR. A política tem de aprender
   o equilíbrio desde o passo 0; "de pé" não é de graça na v3. Isso vale para o play com `model_0.pt` e para todo
   teste que supõe o robô parado.
2. **O PD não segura a `q_com_caixa`.** Com a base segura, a cintura (kp 40, kd 5) deixa o tronco cair 8° e as
   palmas 6 cm em 0,36 s. O plano (E10, 21) pedia "medir na fase I": medido; o PD sozinho fica fora das
   tolerâncias do detector (I §4.2). A política tem de compensar com offset.
3. **Os braços em `q_com_caixa` não tapam a D435i.** A premissa do item 20 da E10 está errada: a caixa fica
   visível entre os braços. Para o real, isso é bom (a idade da caixa não cresce durante o CARREGAR).
4. **O mapa do lidar com ruído:** média de \|h\| < 1 cm no plano, e \|h\| ≤ 1,5σ da própria célula. A confiança
   do PD-6 é honesta.

## Consertos, por arquivo (lanes disjuntas)

Contagem: `eventos.py` 583 → 583; `observacoes.py` 383 → 383; `smoke.py` 1 543 → ~1 575 (+~32: o auxiliar
`_segura_base`, 9 linhas, troca 7 montagens que falhavam; o resto é a s06 e a s18, que medem mais certo).

### Lane A: `g1_v3/eventos.py`

**C1** (linha 242). Atende ao E §3 "Empurrão". Sem o conserto, o empurrão derruba o treino em 5 s; com ele, o
comportamento é o que o plano já descreve.

```python
# antes
    env.v3_t_empurrao[env_ids] = env.episode_length_buf[env_ids].to(torch.float64) * env.step_dt
# depois
    env.v3_t_empurrao[env_ids] = env.episode_length_buf[env_ids].to(env.v3_t_empurrao.dtype) * env.step_dt
```

### Lane B: `g1_v3/observacoes.py`

**C2** (linha 144). Muda só a task `SemPercepcao` (a corrida de custo do I5). A task principal fica igual.

```python
# antes
    if K.percepcao.ligada:
# depois
    if "lidar" in env.scene.sensors:  # a cena do cfg manda (o `k` do make_env_cfg), e não o ATIVO do módulo
```

### Lane C: `g1_v3/smoke.py`

**S0**: auxiliar novo, depois de `_zeros` (linha ~72).

```python
def _segura_base(env, pose):
    """Estado forçado: a base na `pose` (E,7) com velocidade zero, antes de cada passo.

    ⚠ Com o `Contrato.pd` (tornozelo kp 40) a ação zero NÃO segura o G1 de pé: sem DR ele cai em ~1,1 s (tronco a
    38° no passo 50, `queda` no 52; triagem da fase I). As seções que medem outra coisa com o robô parado seguram a base.
    """
    robo = env.scene["robot"]
    robo.write_root_link_pose_to_sim(pose)
    robo.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
```

**S1**: s04, linha 183.

```python
# antes
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
# depois
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{jn}")  # o modelo prefixa a entidade
```

**S2**: s06, três trechos.

```python
# antes (linhas 230-232)
        # parado: a ação zero deixa o robô parado (o `TwistV3._update_command` é `pass`, comando.py:440-441)
        for _ in range(20):
            env.step(_zeros(env))
# depois
        # parado: a base segura (S0); a ação zero sozinha não segura o G1 de pé
        pose0 = robo.data.root_link_pose_w.clone()
        for _ in range(20):
            _segura_base(env, pose0)
            env.step(_zeros(env))
```

```python
# antes (linhas 248-250)
            robo.write_root_link_pose_to_sim(pose)
            env.sim.forward()
            env.step(_zeros(env))
# depois
            _segura_base(env, pose)  # ⚠ sem a velocidade zero o robô cai no passo 33–38 e o reset zera o mapa
            env.sim.forward()
            env.step(_zeros(env))
```

```python
# antes (linhas 257-258)
        alt_ok = ((h[:, frente].abs() <= 0.02) | ~vista[:, frente]).all(dim=1)
        check("alturas das células vistas a ±2 cm", bool(alt_ok[plano].all()))
# depois
        # ⚠ o ruído do lidar está ligado (σ_alcance 2 cm; 0,5° a 2–3 m): ±2 cm por célula é 1σ. O σ de cada célula sai
        # da confiança (PD-6: c = σ_ref²/(σ_ref² + P)); a média fica nos ±2 cm do plano. Fase I: média 0,9 cm, máx 1,5σ.
        sig = (p.sigma_ref ** 2 * (1 / conf.clamp_min(1e-6) - 1)).sqrt()
        hv, vv = h[:, frente], vista[:, frente]
        dentro = ((hv.abs() <= 3 * sig[:, frente]) | ~vv).all(dim=1)
        media = (hv.abs() * vv).sum(dim=1) / vv.sum(dim=1).clamp_min(1)
        check("alturas das células vistas: |h| ≤ 3σ da célula e média ≤ 2 cm",
              bool((dentro & (media <= 0.02))[plano].all()), f"média {media.tolist()}")
```

**S3**: s09, linhas 321-323.

```python
# antes
        for nome, termo in env.cfg.rewards.items():
            valor = termo.func(env, **termo.params).float()
# depois
        # ⚠ o manager instancia os termos de classe (`sacudida`) numa CÓPIA do cfg; o `env.cfg.rewards` guarda a classe
        for nome in env.reward_manager.active_terms:
            termo = env.reward_manager.get_term_cfg(nome)
            valor = termo.func(env, **termo.params).float()
```

**S4**: `_assenta_fatia_b` (linhas 545-549) e s18.

```python
# antes (_assenta_fatia_b)
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    env.v3_fatia[:] = FATIA_B
    eventos.posiciona_cena(env, torch.arange(env.num_envs, device=env.device))
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
# depois
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    from g1_v3.knobs import ATIVO
    env.v3_fatia[:] = FATIA_B
    ids = torch.arange(env.num_envs, device=env.device)
    eventos.posiciona_cena(env, ids)
    # ⚠ como o reset do treino: a base atrás da caixa (`_base_da_pega`, E11 S4). Sem isto ela fica onde o ANDAR a
    # sorteou (±0,5 m, rumo ±π), às vezes contra a laje
    eventos.reset_base_por_fatia(env, ids, dict(ATIVO.cena.reset_base_andar), dict(ATIVO.cena.reset_base_manipula))
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
```

Com este auxiliar, a s17 (7 checks) e a s34 (6 checks, com S11) também passam (`tri/d17.py`).

```python
# antes (s18, linhas 625-637)
        _abre_trecho_de(env, "PEGAR")  # ⚠ força o trecho de PEGAR; o `avanca_trecho` pode sair dele antes de 50 passos
        zeros = _zeros(env)
        soma, fora = 0.0, 0
        for _ in range(50):
            env.step(zeros)
            fora += int((env.v3_tarefa != TAREFAS.index("PEGAR")).sum())
            for v in _por_termo(env).values():
                soma += float(v.clamp(min=0.0).sum())  # o que paga positivo neste passo, somado em mundos e termos
        check("50 passos no PEGAR (a precondição da seção)", fora == 0, f"{fora} mundo-passo fora do PEGAR")
        # ⚠ `progresso` e `rastreio` (c = 0: deslize ≤ 0, potencial.py:106) e `vivo`/`passo` (mascarados) não pagam
        check("soma dos termos que pagam positivo = 0 (ação zero)", soma <= 1e-6, f"soma {soma:.3g}")
# depois
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        _abre_trecho_de(env, "PEGAR")  # ⚠ força o trecho de PEGAR; o `avanca_trecho` pode sair dele antes de 50 passos
        zeros = _zeros(env)
        liquido, outros, fora = torch.zeros(env.num_envs, device=env.device), {}, 0
        for _ in range(50):
            _segura_base(env, pose0)
            env.step(zeros)
            fora += int((env.v3_tarefa != TAREFAS.index("PEGAR")).sum())
            for nome, v in _por_termo(env).items():
                if nome == "progresso":
                    liquido += v.to(liquido.device)
                elif float(v.clamp(min=0.0).sum()) > 0:
                    outros[nome] = outros.get(nome, 0.0) + float(v.clamp(min=0.0).sum())
        check("50 passos no PEGAR (a precondição da seção)", fora == 0, f"{fora} mundo-passo fora do PEGAR")
        check("nenhum termo além do `progresso` paga positivo (ação zero)", not outros, str(outros))
        # ⚠ Φ é potencial (Ng 1999): o PD passivo sobe e desce Φ, e cada subida paga. O que vale é a soma: com γ < 1,
        # pairar soma ≤ 0 (`potencial.progresso`). Fase I: positivos 1,87, líquido −0,14 a −0,35 por mundo.
        check("progresso líquido ≤ 0 em cada mundo (ação zero, base segura)", bool((liquido <= 1e-6).all()),
              str(liquido.tolist()))
```

**S5**: s20. A caixa fora do FOV substitui os braços à frente; os 3 checks da visão ficam.

```python
# antes (linhas 696-706, o repoe_caixa e o laço de 20 passos)
        def repoe_caixa():
            ...
            pose[:, :3] = robo.data.root_link_pos_w + quat_apply(robo.data.root_link_quat_w, ofs.expand(n, 3))
            ...
        zeros = _zeros(env)
        for _ in range(20):
            repoe_caixa()
            env.step(zeros)
# depois
        atras = torch.tensor([-0.45, 0.0, 0.0], device=dev)  # fora do FOV da D435i

        def repoe_caixa(o):
            # ⚠ força: a caixa é reposta a cada passo, senão cairia (0,4 s de queda livre são 0,8 m) antes do estimador
            pose = torch.zeros(n, 7, device=dev)
            pose[:, :3] = robo.data.root_link_pos_w + quat_apply(robo.data.root_link_quat_w, o.expand(n, 3))
            pose[:, 3] = 1.0
            caixa.write_root_link_pose_to_sim(pose)
            caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev))

        zeros = _zeros(env)
        pose0 = robo.data.root_link_pose_w.clone()
        for _ in range(20):
            _segura_base(env, pose0)
            repoe_caixa(ofs)
            env.step(zeros)
```

```python
# antes (linhas 727-740, do "braços na frente" ao último check)
        q = env.v3_ref.q_com_caixa(env.v3_meia).to(dev, torch.float32)
        ...
        check("braços à frente: a idade cresce ≥ 0,4 s em 0,6 s", ...)
# depois
        # ⚠ fase I: em `q_com_caixa` os braços NÃO tapam a câmera (palmas ao lado da caixa, 0–2 dos 9 pontos
        # ocluídos). A idade se mede sem leitura: a caixa atrás da base por 0,6 s.
        idade0 = s[:, 7].clone()
        for _ in range(30):
            _segura_base(env, pose0)
            repoe_caixa(atras)
            env.step(zeros)
        s2 = env.v3_estimador.saida_b(env.v3_odo, robo.data.root_link_quat_w)
        check("caixa fora do FOV: a idade cresce ≥ 0,4 s em 0,6 s", bool(((s2[:, 7] - idade0) >= 0.4).all()),
              f"Δ {(s2[:, 7] - idade0).tolist()}")
```

Os imports de `q_de_pe` e o nome da seção (`…_idade_cresce_com_bracos`) saem ou trocam; o `main()` chama a
seção pelo nome novo. Medido: Δ = 0,567 s nos 4 mundos.

**S6**: s21, linhas 770-773 (e `JUNTAS` no import da seção).

```python
# antes
        acao = (q - qde) / ATIVO.contrato.escala_acao  # segura os braços: o PD segue q_com_caixa
        avisos = []
        for _ in range(60):
            env.step(acao)
            avisos.append(env.v3_aviso.clone())
# depois
        # ⚠ o alvo do PD é q_de_pé + escala·a − viés (s04): a ação soma o viés para o alvo ser q_com_caixa
        idx = [robo.joint_names.index(j) for j in JUNTAS]
        acao = (q - qde + robo.data.encoder_bias[:, idx]) / ATIVO.contrato.escala_acao
        avisos = []
        for _ in range(60):
            # ⚠ fase I: a ação zero não segura a base (S0), e a cintura (kp 40) não segura os braços à frente
            # (tronco +8° em 0,36 s). A seção mede o relógio do detector: base e juntas reescritas a cada passo.
            _segura_base(env, pose)
            robo.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.step(acao)
            avisos.append(env.v3_aviso.clone())
```

**S7**: s26, linhas 902-906 (e `from g1_v3 import balanceador`, `from g1_v3.knobs import ATIVO`).

```python
# antes
        esperado = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia
        check("alvo de soltura: z = origem + v3_topo_destino + v3_meia (± 1e-4)",
              torch.allclose(env.v3_alvo_w[:, 2], esperado, atol=1e-4),
              f"{(env.v3_alvo_w[:, 2] - esperado).abs().max().item():.2e}")
# depois
        esperado = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia
        # ⚠ P22: o `_abre_botar` soma o viés do alvo U[±vz] (vz pelo degrau global; 5 mm no degrau 0)
        vz = balanceador.no_degrau(*ATIVO.estimador.vies_alvo_botar_z, env.v3_cur.degrau_global(),
                                   ATIVO.balanceador.degraus)
        dz = (env.v3_alvo_w[:, 2] - esperado).abs()
        check("alvo de soltura: z = origem + v3_topo_destino + v3_meia (± viés do alvo, P22)",
              bool((dz <= vz + 1e-4).all()), f"{dz.max().item():.2e} > {vz:.3f}")
```

**S8**: s27, linhas 922-923.

```python
# antes
            pose[:, :3] = env.v3_alvo_w
            pose[:, 2] += altura
# depois
            # ⚠ P22: o `v3_alvo_w` traz o viés do alvo (até 5 mm em z no degrau 0): o z sai do topo verdadeiro
            pose[:, :2] = env.v3_alvo_w[:, :2]
            pose[:, 2] = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia + altura
```

**S9**: s28.

```python
# antes (linhas 978-981)
        for p in range(math.ceil((1.5 + atraso_max + margem) / dt)):
            env.step(zeros)
# depois
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        for p in range(math.ceil((1.5 + atraso_max + margem) / dt)):
            _segura_base(env, pose0)  # S0: a ação zero cai em ~1,1 s, antes do aviso
            env.step(zeros)
```

**S10**: s29, o mesmo (linhas 1019-1020).

```python
# antes
        for p in range(math.ceil(2 * ATIVO.cadeia.andar_final_s / dt)):
            env.step(zeros)
# depois
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        for p in range(math.ceil(2 * ATIVO.cadeia.andar_final_s / dt)):
            _segura_base(env, pose0)  # S0: sem isto a `queda` (~1,1 s) chega antes do fim do ANDAR de 1,5 s
            env.step(zeros)
```

**S11**: s34, linhas 1250-1256.

```python
# antes
        env.v3_degrau[:] = 5
        # M2: ...
        _assenta_fatia_b(env)
        _le_caixa_valida(env)
        _abre_trecho_de(env, "PEGAR")
        env.v3_alvo_idx[:] = P3
# depois
        # M2: ...
        _assenta_fatia_b(env)
        _le_caixa_valida(env)
        _abre_trecho_de(env, "PEGAR")
        env.v3_alvo_idx[:] = P3
        env.v3_degrau[:] = 5  # ⚠ depois da abertura: o `_publica` escreve o degrau do currículo (0) no trecho novo
```

**S12**: s36, três trechos.

```python
# antes (_geoms_do_solref, linhas 1320 e 1324)
    ev = env.cfg.events["solref"]  # P17: o nome do evento no cfg (env_cfg.py:170)
    ...
    tid = [i for i in range(m.ngeom) if re.search("terrain$", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
# depois
    # ⚠ o manager resolve uma CÓPIA do cfg: no `env.cfg.events` o `geom_ids` fica slice(None) (as 72 geoms do robô)
    ev = env.event_manager.get_term_cfg("solref")
    ...
    # ⚠ as geoms do terreno se chamam terrain_0 … terrain_N; com `terrain$` o tid ficava vazio e o check, vazio
    tid = [i for i in range(m.ngeom) if re.fullmatch(r"terrain_\d+", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
```

```python
# antes (as 10 passadas do RFI e a linha 1384)
        for _ in range(10):
            env.step(_zeros(env))
        ...
        for _ in range(450):  # o intervalo da força é de 3 a 8 s (até 400 passos)
            env.v3_tarefa[0] = carregar
# depois
        for _ in range(10):
            env.step(_zeros(env))
        pose0 = robo.data.root_link_pose_w.clone()  # os pés no chão: o contato pé × terreno do fim se lê daqui
        ...
        for _ in range(450):  # o intervalo da força é de 3 a 8 s (até 400 passos)
            # ⚠ S0: sem a base segura o robô cai a cada ~1,1 s, e o reset sorteia de novo o intervalo do impulso
            # (`apply_body_impulse.reset`): a força nunca chega
            _segura_base(env, pose0)
            env.v3_tarefa[0] = carregar
```

## Ao implementador

- A ordem das lanes não importa. As 3 tocam arquivos diferentes.
- Depois dos consertos, rode o smoke inteiro: a triagem rodou cada seção isolada, com o conserto aplicado.
- `main()`: renomear a chamada da s20, se o nome da função mudar.
