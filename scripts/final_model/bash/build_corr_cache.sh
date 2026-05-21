#!/usr/bin/env bash
set -euo pipefail

# Optional offline cache builder for full final model.
# Training/inference do not require this file, but saved caches can speed up
# repeated validation or inference. By default it uses DINOv2 + CleanDIFT +
# SigLIP2, matching full final model correspondence.

source scripts/resolve_runtime.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$("${PYTHON_BIN}" scripts/final_model/gpu_select.py)"
fi

TRAIN_JSONL="${TRAIN_JSONL:-datasets/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/corr_cache_final_1024}"
RESOLUTION="${RESOLUTION:-1024}"
GRID_SIZE="${GRID_SIZE:-}"
DINO_MODEL="${DINO_MODEL:-model/dinov2-large}"
VLM_MODEL="${VLM_MODEL:-model/siglip2-so400m-patch16-naflex}"
CLEANDIFT_UNET="${CLEANDIFT_UNET:-model/cleandift/cleandift_sd21_unet.safetensors}"
CLEANDIFT_VAE="${CLEANDIFT_VAE:-stabilityai/sd-vae-ft-mse}"
CLEANDIFT_FEATURE_KEY="${CLEANDIFT_FEATURE_KEY:-us6}"
CLEANDIFT_TIMESTEP="${CLEANDIFT_TIMESTEP:-261}"
CLEANDIFT_USE_TEXT_ENCODER="${CLEANDIFT_USE_TEXT_ENCODER:-0}"
DISABLE_CLEANDIFT="${DISABLE_CLEANDIFT:-0}"
LAMBDA_DINO="${LAMBDA_DINO:-0.5}"
LAMBDA_CLEAN="${LAMBDA_CLEAN:-0.5}"
DTYPE="${DTYPE:-bf16}"
START_INDEX="${START_INDEX:-0}"
MAX_ITEMS="${MAX_ITEMS:-}"
OVERWRITE="${OVERWRITE:-0}"
DEBUG_DIR="${DEBUG_DIR:-}"

cmd=(
  "${PYTHON_BIN}" scripts/final_model/build_corr_cache.py
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --resolution "${RESOLUTION}"
  --dino_model "${DINO_MODEL}"
  --vlm_model "${VLM_MODEL}"
  --cleandift_unet "${CLEANDIFT_UNET}"
  --cleandift_vae "${CLEANDIFT_VAE}"
  --cleandift_feature_key "${CLEANDIFT_FEATURE_KEY}"
  --cleandift_timestep "${CLEANDIFT_TIMESTEP}"
  --lambda_dino "${LAMBDA_DINO}"
  --lambda_clean "${LAMBDA_CLEAN}"
  --dtype "${DTYPE}"
  --start_index "${START_INDEX}"
)

if [[ -n "${GRID_SIZE}" ]]; then
  cmd+=(--grid_size "${GRID_SIZE}")
fi

if [[ "${CLEANDIFT_USE_TEXT_ENCODER}" == "1" ]]; then
  cmd+=(--cleandift_use_text_encoder)
fi

if [[ "${DISABLE_CLEANDIFT}" == "1" ]]; then
  cmd+=(--disable_cleandift)
fi

if [[ -n "${MAX_ITEMS}" ]]; then
  cmd+=(--max_items "${MAX_ITEMS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

if [[ -n "${DEBUG_DIR}" ]]; then
  cmd+=(--debug_dir "${DEBUG_DIR}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
