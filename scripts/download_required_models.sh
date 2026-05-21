#!/usr/bin/env bash
set -euo pipefail

HF_BIN="${HF_BIN:-hf}"
DOWNLOAD_SD35_FULL="${DOWNLOAD_SD35_FULL:-0}"
DOWNLOAD_CLEANDIFT_VAE="${DOWNLOAD_CLEANDIFT_VAE:-0}"
DOWNLOAD_DEPTH_ANYTHING="${DOWNLOAD_DEPTH_ANYTHING:-1}"
DOWNLOAD_SEGFORMER="${DOWNLOAD_SEGFORMER:-1}"
DOWNLOAD_MASK2FORMER="${DOWNLOAD_MASK2FORMER:-1}"

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "Cannot find '${HF_BIN}'. Install huggingface-hub first: pip install huggingface-hub" >&2
  exit 1
fi

mkdir -p model

if [[ "${DOWNLOAD_SD35_FULL}" == "1" ]]; then
  "${HF_BIN}" download stabilityai/stable-diffusion-3.5-medium \
    --local-dir model/stable-diffusion-3.5-medium
else
  "${HF_BIN}" download stabilityai/stable-diffusion-3.5-medium \
    --include "model_index.json" \
    --include "scheduler/*" \
    --include "transformer/*" \
    --include "vae/*" \
    --include "README.md" \
    --include "LICENSE.md" \
    --local-dir model/stable-diffusion-3.5-medium
fi

"${HF_BIN}" download facebook/dinov2-large \
  --local-dir model/dinov2-large

"${HF_BIN}" download google/siglip2-so400m-patch16-naflex \
  --local-dir model/siglip2-so400m-patch16-naflex

"${HF_BIN}" download CompVis/cleandift \
  cleandift_sd21_unet.safetensors \
  cleandift_sd21_depth_probe.safetensors \
  README.md \
  --local-dir model/cleandift

if [[ "${DOWNLOAD_DEPTH_ANYTHING}" == "1" ]]; then
  "${HF_BIN}" download depth-anything/Depth-Anything-V2-Base-hf \
    --local-dir model/depth-anything-v2-base
fi

if [[ "${DOWNLOAD_SEGFORMER}" == "1" ]]; then
  "${HF_BIN}" download nvidia/segformer-b0-finetuned-ade-512-512 \
    --include "config.json" \
    --include "preprocessor_config.json" \
    --include "model.safetensors" \
    --include "README.md" \
    --local-dir model/segformer-b0-ade
fi

if [[ "${DOWNLOAD_MASK2FORMER}" == "1" ]]; then
  "${HF_BIN}" download facebook/mask2former-swin-small-coco-panoptic \
    --include "config.json" \
    --include "preprocessor_config.json" \
    --include "model.safetensors" \
    --include "README.md" \
    --local-dir model/mask2former-swin-small-coco-panoptic
fi

if [[ "${DOWNLOAD_CLEANDIFT_VAE}" == "1" ]]; then
  "${HF_BIN}" download stabilityai/sd-vae-ft-mse \
    --local-dir model/sd-vae-ft-mse
fi

echo "Model download commands finished. See model/README.md for required paths."
