#!/usr/bin/env bash
# Ponto de entrada do pod RunPod (D13 do plano; contrato da imagem em PD-22).
# A imagem clona o repo e chama este script. Só roda o treino: não instala pacote nem faz push.
set -euo pipefail
export MUJOCO_GL=egl

# Raiz do repo, a partir do próprio script (este arquivo fica em g1_v3/container/).
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN="${RUN:-}"
RETOMA="${RETOMA:-}"
ITERS="${ITERS:-30000}"
NUM_ENVS="${NUM_ENVS:-4096}"
PERCEPCAO="${PERCEPCAO:-1}"
SAIDA="${SAIDA:-/workspace/g1_v3}"
TENSORBOARD="${TENSORBOARD:-1}"
SECO="${SECO:-0}"

# 1. Entradas. RUN entra num glob e num nome de pasta: só caracteres seguros.
[ -n "$RUN" ] || { echo "falta RUN (nome da corrida)" >&2; exit 1; }
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN aceita só letras, números, '.', '-' e '_'" >&2; exit 1; }
case "$RETOMA" in
  0|1) ;;
  *) echo "RETOMA é obrigatório: 0 (corrida nova) ou 1 (retoma a última com checkpoint)" >&2; exit 1 ;;
esac

# 2. Checagens de GPU e versões. SECO=1 pula tudo (a máquina do teste não tem GPU).
# ⚠ rsl-rl-lib 5.4.0 é pinado de propósito: o RslRlModelCfg do 1.5.1 quebra com ele (notebook antigo, célula 2).
if [ "$SECO" != 1 ]; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  [ -n "$GPU" ] || { echo "nvidia-smi não viu GPU" >&2; exit 1; }
  python - <<'PY'
import importlib.metadata as md
import mujoco
import torch

assert torch.cuda.is_available(), "torch sem CUDA"
assert md.version("mjlab") == "1.5.3", md.version("mjlab")
assert md.version("rsl-rl-lib") == "5.4.0", md.version("rsl-rl-lib")
print("torch", torch.__version__, "| mujoco", mujoco.__version__, "| GPU", torch.cuda.get_device_name(0))
PY
  git -C "$RAIZ" log --oneline -1
fi

# 3. Pasta do experimento: $SAIDA/g1_v3 (experiment_name da D10). O mjlab nomeia cada corrida
# AAAA-MM-DD_HH-MM-SS_$RUN, então o glob casa só esse formato e não pega outro RUN com final igual.
EXP="$SAIDA/g1_v3"
shopt -s nullglob
PASTAS=("$EXP"/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"$RUN")

if [ "$RETOMA" = 0 ]; then
  # Corrida nova: não reaproveita uma pasta do mesmo RUN (nome novo a cada conjunto de mudanças).
  if [ "${#PASTAS[@]}" -gt 0 ]; then
    echo "já existe corrida $RUN em $EXP (${PASTAS[0]}): use outro RUN ou RETOMA=1" >&2
    exit 1
  fi
else
  # ⚠ "Mais nova" = a última em ordem de nome, que é a de data e hora mais recente, o mesmo critério
  # do mjlab (get_checkpoint_path ordena por nome). Sem mtime: copiar a pasta mudaria a ordem.
  # Pula pasta sem model_*.pt: sessão que caiu antes do primeiro checkpoint não é retomável.
  ESCOLHIDA=""
  for ((i = ${#PASTAS[@]} - 1; i >= 0; i--)); do
    if compgen -G "${PASTAS[i]}/model_*.pt" >/dev/null; then
      ESCOLHIDA="$(basename "${PASTAS[i]}")"
      break
    fi
  done
  [ -n "$ESCOLHIDA" ] || { echo "RETOMA=1 mas nenhuma corrida $RUN com checkpoint em $EXP" >&2; exit 1; }
fi

# 4. Comando. Sem percepção só no I5 (PERCEPCAO=0).
# ⚠ DÚVIDA: o comando não passa --agent.experiment-name; o g1_v3 vem do default da D10. Confirmar ao criar a config.
case "$PERCEPCAO" in
  1) TASK="Mjlab-G1-V3" ;;
  0) TASK="Mjlab-G1-V3-SemPercepcao" ;;
  *) echo "PERCEPCAO aceita 0 ou 1" >&2; exit 1 ;;
esac
ARGS=(python -m g1_v3.train "$TASK" --env.scene.num-envs "$NUM_ENVS" --agent.run-name "$RUN"
  --agent.max-iterations "$ITERS" --agent.logger tensorboard --log-root "$SAIDA")
if [ "$RETOMA" = 1 ]; then
  # ⚠ --agent.resume é bool do tyro: flag sem valor. --agent.load-run é regex: as âncoras impedem casar outra pasta.
  # ITERS vale para esta sessão: o rsl_rl soma à it do checkpoint (total = início + ITERS).
  ARGS+=(--agent.resume --agent.load-run "^${ESCOLHIDA}\$")
fi

if [ "$SECO" = 1 ]; then
  echo "${ARGS[*]}"
  exit 0
fi

# 5. TensorBoard em segundo plano, porta 6006, lendo a raiz dos logs.
if [ "$TENSORBOARD" = 1 ]; then
  tensorboard --logdir "$SAIDA" --port 6006 --bind_all &
fi

# 6. O treino substitui o script. cd na raiz para o `python -m g1_v3.train` achar o pacote.
cd "$RAIZ"
exec "${ARGS[@]}"
