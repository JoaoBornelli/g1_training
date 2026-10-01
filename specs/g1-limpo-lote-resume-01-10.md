# G1 limpo — lote do resume de 01/10: K1b + R1 + K12

Estado: aprovado pelo dono ("sim", 01/10). Origem:
`docs/relatorios/2026-10-01-monotonia-do-curriculo.md` e
`docs/relatorios/2026-09-30-plato-do-botar.md`.

Três partes, implementadas na ordem abaixo, num implementador só.

## Parte 1 — K1b

Implementar `specs/g1-limpo-rastreio-carregar-parado.md` inteira, como está escrita.

## Parte 2 — R1: a laje do BOTAR só se afasta

Enunciado: §3 Contato ("nada do robô toca a laje"); §4.5 Entrada. Decisão do dono (01/10):
a DR da laje fica (±0,10 em x, ±0,10 em y, topo ±0,10; alvo ±0,05); só o sorteio em x passa a
afastar a laje, dx ∈ [0; 0,10]. Medido: toque de 39% para 0,6% com o robô de pé.

- `g1_limpo/comando.py:1845`: hoje
  `dxy = c.botar_delta_xy * (2.0 * torch.rand(k, 2, device=d) - 1.0)`.
  Depois: o y fica simétrico; o x vira `c.botar_delta_xy * torch.rand(k, device=d)` (só
  positivo, longe do robô). Escreva com o mínimo de linhas (ex.: depois da linha atual,
  `dxy[:, 0] = dxy[:, 0].abs()`), com um comentário ⚠ de 2 linhas: o porquê (a laje nascia
  dentro da perna em 31–72% das aberturas) e a referência a este arquivo.
- Não mude `botar_delta_xy`, o alvo, o topo, nem o resto do bloco.
- `smoke.py`: se algum check existente supõe dx simétrico, ajuste-o; senão, um check de
  fonte de que o bloco do BOTAR contém o `.abs()` (ou a forma escolhida) no x.

## Parte 3 — K12: métricas sem peso por episódio da cadeia C

Enunciado: §7 item 6 (medir antes do próximo conserto). Nenhuma recompensa lê estas métricas.

Todas ficam em `self.metrics` do `AlvoCaixaCmd` (`comando.py:553-556`), são float, nascem
em zero e o `reset` do mjlab tira a média e zera, como as vizinhas. Flags valem 1 até o fim
do episódio.

| Chave | Quando vira 1 | Onde |
|---|---|---|
| `c_episodio` | todo passo em que `self._cadeia == 2` | no bloco por passo das métricas (`comando.py:~1671`), como `fatia_cadeia` |
| `c_chegou_carregar` | avanço de um env da C para o elo CARREGAR | no bloco do avanço de `_aplica_espera` (`comando.py:~775-797`), depois do `_elo` novo |
| `c_chegou_botar` | avanço de um env da C para o elo BOTAR | no mesmo bloco |
| `c_desvio_pegar` | env da C que vai à cauda vindo do elo PEGAR | no bloco da cauda (`comando.py:~808-839`), ANTES de `self._elo[vira_carregar] = CARREGAR`; leia o `_elo` anterior |
| `c_desvio_carregar` | env da C que vai à cauda vindo do elo CARREGAR | no mesmo lugar |
| `cauda_twist_herdado` | fração dos passos de cauda andando com comando linear < 0,05 m/s | veja abaixo |

`cauda_twist_herdado`: dois acumuladores por env, `self._n_cauda_anda` e
`self._n_cauda_herdado` (float, zerados no reset do env, no mesmo lugar onde o comando zera o
seu estado por env, perto de `comando.py:1253`). Em `_update_command`, DEPOIS de
`_zera_twist_nos_parados()` (`comando.py:1413`): máscara `anda = (elo == CARREGAR) &
~_carregar_parado & ~_soltou`; some 1 em `_n_cauda_anda[anda]`; some 1 em
`_n_cauda_herdado[anda & (‖vel_command_b[:, :2]‖ do twist < 0,05)]`; escreva
`metrics["cauda_twist_herdado"] = _n_cauda_herdado / _n_cauda_anda.clamp(min=1)`. Leia o twist
do mesmo jeito que `_zera_twist_nos_parados` lê (`command_manager.get_term(self.cfg.nome_do_twist)`).

Leitura no log (para o dono, não é código): razões `c_chegou_botar / c_episodio`,
`c_desvio_pegar / c_episodio`, `c_desvio_carregar / c_chegou_carregar`.

`smoke.py`: um check de que as seis chaves existem em `metrics` do comando, e um check de
fonte de que `c_desvio_*` é escrito antes de `self._elo[vira_carregar] = CARREGAR` em
`_aplica_espera`.

## Regras para quem implementa

- Arquivos: `g1_limpo/comando.py`, `recompensas.py`, `knobs.py`, `env_cfg.py`, `smoke.py`,
  `specs/g1-limpo-rastreio-carregar-elo.md` (só a linha de estado). Nenhum outro.
- Nenhum comando `git`. Não rode smoke, env do mjlab, nem import do pacote. No fim, só
  `/home/joaobornelli/Documents/g1_training/.venv/bin/python -m py_compile` nos 5 `.py`.
- Estilo dos arquivos: comentários em português, ⚠ nos avisos, nomes em português.
- Não mude nenhum número da tabela por estado nem outro peso.
- Reporte: diff resumido por arquivo, contagem de linhas antes → depois, e qualquer ponto
  em que a spec não bateu com o código (com o que você decidiu).
