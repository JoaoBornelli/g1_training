#!/usr/bin/env bash
# Empacota o repo para subir como Kaggle Dataset PRIVADO.
#
#   bash g1_multitask/kaggle/empacota.sh
#   -> /tmp/g1-multitask.zip
#
# Dataset e não `git clone` porque é reprodutível e não depende de internet habilitada
# na sessão (§15 do doc).
#
# O que fica FORA, e por quê:
#   .venv/                  ~4 GB, e é build da máquina local — inútil e quebraria
#   reference_checkpoints/  checkpoints da Lift; o multi-tarefa treina do zero
#                           (obs de 151 contra 132 = Categoria C, não carrega)
#   logs/ runs/ outputs/    saída de treino anterior
#   __pycache__/ *.pyc      artefato de build
#   *.npz *.png             rollouts e gráficos locais
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAIDA="${1:-/tmp/g1-multitask.zip}"
cd "$RAIZ"

rm -f "$SAIDA"
zip -r -q "$SAIDA" \
    g1_training g1_multitask README.md \
    -x '*/.venv/*' '*/__pycache__/*' '*.pyc' \
       '*/logs/*' '*/runs/*' '*/outputs/*' \
       '*.npz' '*.png' '*.pt'

echo "pacote: $SAIDA  ($(du -h "$SAIDA" | cut -f1))"
echo
echo "conteúdo, por diretório de topo:"
unzip -l "$SAIDA" | awk 'NR>3 && $4 ~ /\// {print $4}' | cut -d/ -f1 \
    | sort | uniq -c | sort -rn
echo
echo "Agora: kaggle.com -> Datasets -> New Dataset -> sobe o zip"
echo "       nome sugerido: g1-multitask   (privado)"
echo "       a Kaggle descompacta sozinha em /kaggle/input/g1-multitask/"
