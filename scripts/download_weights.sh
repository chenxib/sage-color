#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${HF_REPO:-chenxib/sage-color}"

if [[ -z "${HF_BIN:-}" ]]; then
  if command -v hf >/dev/null 2>&1; then
    HF_BIN="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN="huggingface-cli"
  else
    HF_BIN="hf"
  fi
fi

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "Cannot find '${HF_BIN}'. Install huggingface-hub first: pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p checkpoints

"${HF_BIN}" download "${HF_REPO}" \
  checkpoints/sage-color-final.pt \
  checkpoints/sage-color-grounding.pt \
  --local-dir .

echo "SAGE-Color checkpoints downloaded into checkpoints/."
