#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CHECKPOINT=outputs/.../checkpoint-100/color_edit_stage1.pt \
#   CONTENT_IMAGE=path/to/content.png \
#   REFERENCE_IMAGE=path/to/reference.png \
#   OUTPUT_IMAGE=outputs/sample.png \
#   bash scripts/stage1_training/bash/infer.sh
#
# Notes:
# - CORR_CACHE is optional. If it is empty, DINOv2 + CleanDIFT + SigLIP2 compute
#   correspondence online on the selected GPU.
# - SAVE_CORR_CACHE=path/to/cache.npz saves the online correspondence for reuse.
# - CLEANDIFT_VAE defaults to the checkpoint config, then stabilityai/sd-vae-ft-mse.
# - DISABLE_CLEANDIFT=1 runs the DINO+SigLIP ablation and is not full stage-1 model.
# - RESOLUTION is optional. If empty, the script uses the checkpoint resolution.
#   Set RESOLUTION=1024 or RESOLUTION=512 only when you want to override it.
# - NUM_INFERENCE_STEPS=28 is a normal-quality default; use 1-4 only for smoke tests.

source scripts/resolve_runtime.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$("${PYTHON_BIN}" scripts/stage1_training/gpu_select.py)"
fi

PRETRAINED_MODEL="${PRETRAINED_MODEL:-model/stable-diffusion-3.5-medium}"
CHECKPOINT="${CHECKPOINT:-outputs/stage1/checkpoint-100/color_edit_stage1.pt}"
CONTENT_IMAGE="${CONTENT_IMAGE:-datasets/validation/content.png}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-datasets/validation/reference.png}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-outputs/stage1/sample.png}"
CORR_CACHE="${CORR_CACHE:-}"
SAVE_CORR_CACHE="${SAVE_CORR_CACHE:-}"
DINO_MODEL="${DINO_MODEL:-model/dinov2-large}"
VLM_MODEL="${VLM_MODEL:-}"
CLEANDIFT_UNET="${CLEANDIFT_UNET:-model/cleandift/cleandift_sd21_unet.safetensors}"
CLEANDIFT_VAE="${CLEANDIFT_VAE:-}"
CLEANDIFT_FEATURE_KEY="${CLEANDIFT_FEATURE_KEY:-}"
CLEANDIFT_TIMESTEP="${CLEANDIFT_TIMESTEP:-}"
CLEANDIFT_USE_TEXT_ENCODER="${CLEANDIFT_USE_TEXT_ENCODER:-0}"
DISABLE_CLEANDIFT="${DISABLE_CLEANDIFT:-0}"

RESOLUTION="${RESOLUTION:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
REFERENCE_SCALE="${REFERENCE_SCALE:-1.0}"
CORR_SCALE="${CORR_SCALE:-1.0}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"

cmd=(
  "${PYTHON_BIN}" scripts/stage1_training/infer.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
  --checkpoint "${CHECKPOINT}"
  --content_image "${CONTENT_IMAGE}"
  --reference_image "${REFERENCE_IMAGE}"
  --output "${OUTPUT_IMAGE}"
  --dino_model "${DINO_MODEL}"
  --cleandift_unet "${CLEANDIFT_UNET}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --reference_scale "${REFERENCE_SCALE}"
  --corr_scale "${CORR_SCALE}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
)

if [[ -n "${CLEANDIFT_VAE}" ]]; then
  cmd+=(--cleandift_vae "${CLEANDIFT_VAE}")
fi

if [[ -n "${CLEANDIFT_FEATURE_KEY}" ]]; then
  cmd+=(--cleandift_feature_key "${CLEANDIFT_FEATURE_KEY}")
fi

if [[ -n "${CLEANDIFT_TIMESTEP}" ]]; then
  cmd+=(--cleandift_timestep "${CLEANDIFT_TIMESTEP}")
fi

if [[ "${CLEANDIFT_USE_TEXT_ENCODER}" == "1" ]]; then
  cmd+=(--cleandift_use_text_encoder)
fi

if [[ "${DISABLE_CLEANDIFT}" == "1" ]]; then
  cmd+=(--disable_cleandift)
fi

if [[ -n "${RESOLUTION}" ]]; then
  cmd+=(--resolution "${RESOLUTION}")
fi

if [[ -n "${CORR_CACHE}" ]]; then
  cmd+=(--corr_cache "${CORR_CACHE}")
fi

if [[ -n "${SAVE_CORR_CACHE}" ]]; then
  cmd+=(--save_corr_cache "${SAVE_CORR_CACHE}")
fi

if [[ -n "${VLM_MODEL}" ]]; then
  cmd+=(--vlm_model "${VLM_MODEL}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
