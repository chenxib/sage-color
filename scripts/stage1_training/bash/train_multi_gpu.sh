#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
#   bash scripts/stage1_training/bash/train_multi_gpu.sh
#
# Common smoke test:
#   CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
#   TRAIN_JSONL=datasets/train.jsonl \
#   OUTPUT_DIR=outputs/stage1-online-smoke-ddp \
#   RESOLUTION=256 LORA_RANK=16 MAX_TRAIN_STEPS=1 CHECKPOINTING_STEPS=1 NUM_WORKERS=0 \
#   bash scripts/stage1_training/bash/train_multi_gpu.sh
#
# Notes:
# - Training computes correspondence online on each training process with
#   DINOv2 + CleanDIFT dense matching plus SigLIP2 semantic gating.
#   No prebuilt .npz cache is required.
# - Online correspondence increases VRAM use because SD3.5, SD3 VAE, SD2.1 VAE,
#   CleanDIFT UNet, DINOv2, SigLIP2, and the adapter are resident on GPU.
# - CLEANDIFT_VAE defaults to stabilityai/sd-vae-ft-mse. The first run
#   may download the SD2.1 VAE unless you point it to a local SD2.1 folder.
# - DISABLE_CLEANDIFT=1 runs the DINO+SigLIP ablation and is not full stage-1 model.
# - NUM_PROCESSES should match the number of visible GPUs in CUDA_VISIBLE_DEVICES.
# - CHECKPOINTING_STEPS controls both checkpoint saves and validation_sample.png.
# - Set DISABLE_CHECKPOINT_VALIDATION=1 to skip validation image generation.
# - VALIDATION_CORR_CACHE is optional; when empty, validation also computes
#   correspondence online.

source scripts/resolve_runtime.sh

# Set this explicitly for multi-GPU training, for example: CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-model/stable-diffusion-3.5-medium}"
TRAIN_JSONL="${TRAIN_JSONL:-datasets/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage1-ddp}"
INIT_FROM_STAGE1_BASE_CHECKPOINT="${INIT_FROM_STAGE1_BASE_CHECKPOINT:-}"

RESOLUTION="${RESOLUTION:-1024}"
GRID_SIZE="${GRID_SIZE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAMW_FOREACH="${ADAMW_FOREACH:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
SEED="${SEED:-42}"

DINO_MODEL="${DINO_MODEL:-model/dinov2-large}"
VLM_MODEL="${VLM_MODEL:-model/siglip2-so400m-patch16-naflex}"
CLEANDIFT_UNET="${CLEANDIFT_UNET:-model/cleandift/cleandift_sd21_unet.safetensors}"
CLEANDIFT_VAE="${CLEANDIFT_VAE:-stabilityai/sd-vae-ft-mse}"
CLEANDIFT_FEATURE_KEY="${CLEANDIFT_FEATURE_KEY:-us6}"
CLEANDIFT_TIMESTEP="${CLEANDIFT_TIMESTEP:-261}"
CLEANDIFT_USE_TEXT_ENCODER="${CLEANDIFT_USE_TEXT_ENCODER:-0}"
DISABLE_CLEANDIFT="${DISABLE_CLEANDIFT:-0}"
TOP_M_REGIONS="${TOP_M_REGIONS:-2}"
TOP_K_SPARSE="${TOP_K_SPARSE:-16}"
LAMBDA_DINO="${LAMBDA_DINO:-0.5}"
LAMBDA_CLEAN="${LAMBDA_CLEAN:-0.5}"
LAMBDA_VLM_TOKEN="${LAMBDA_VLM_TOKEN:-0.15}"
LAMBDA_VLM_REGION="${LAMBDA_VLM_REGION:-0.25}"
LAMBDA_REGION_TOKEN="${LAMBDA_REGION_TOKEN:-0.20}"
TAU_SPARSE="${TAU_SPARSE:-0.07}"
TAU_REGION="${TAU_REGION:-0.10}"
LAMBDA_XY="${LAMBDA_XY:-0.05}"
KMEANS_ITERS="${KMEANS_ITERS:-10}"
NUM_GLOBAL_TOKENS="${NUM_GLOBAL_TOKENS:-32}"
NUM_REGIONS="${NUM_REGIONS:-24}"
REFERENCE_DROPOUT="${REFERENCE_DROPOUT:-0.05}"
CORR_DROPOUT="${CORR_DROPOUT:-0.10}"
REFERENCE_SCALE="${REFERENCE_SCALE:-1.0}"
CORR_SCALE="${CORR_SCALE:-1.0}"
CORR_WARMUP_STEPS="${CORR_WARMUP_STEPS:-5000}"
LOCAL_START_LAYER="${LOCAL_START_LAYER:-4}"
SPARSE_START_LAYER="${SPARSE_START_LAYER:-5}"

REFERENCE_RESAMPLER_DEPTH="${REFERENCE_RESAMPLER_DEPTH:-4}"
REFERENCE_RESAMPLER_HEADS="${REFERENCE_RESAMPLER_HEADS:-16}"
REFERENCE_RESAMPLER_DIM_HEAD="${REFERENCE_RESAMPLER_DIM_HEAD:-64}"

LORA_RANK="${LORA_RANK:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_BLOCKS="${LORA_BLOCKS:-}"
LORA_LAYERS="${LORA_LAYERS:-}"

VALIDATION_CONTENT_IMAGE="${VALIDATION_CONTENT_IMAGE:-datasets/validation/content.png}"
VALIDATION_REFERENCE_IMAGE="${VALIDATION_REFERENCE_IMAGE:-datasets/validation/reference.png}"
VALIDATION_CORR_CACHE="${VALIDATION_CORR_CACHE:-}"
VALIDATION_NUM_INFERENCE_STEPS="${VALIDATION_NUM_INFERENCE_STEPS:-4}"
VALIDATION_SEED="${VALIDATION_SEED:-42}"
DISABLE_CHECKPOINT_VALIDATION="${DISABLE_CHECKPOINT_VALIDATION:-0}"

train_args=(
  scripts/stage1_training/train.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --resolution "${RESOLUTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --mixed_precision "${MIXED_PRECISION}"
  --num_workers "${NUM_WORKERS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --seed "${SEED}"
  --dino_model "${DINO_MODEL}"
  --vlm_model "${VLM_MODEL}"
  --cleandift_unet "${CLEANDIFT_UNET}"
  --cleandift_vae "${CLEANDIFT_VAE}"
  --cleandift_feature_key "${CLEANDIFT_FEATURE_KEY}"
  --cleandift_timestep "${CLEANDIFT_TIMESTEP}"
  --top_m_regions "${TOP_M_REGIONS}"
  --top_k_sparse "${TOP_K_SPARSE}"
  --lambda_dino "${LAMBDA_DINO}"
  --lambda_clean "${LAMBDA_CLEAN}"
  --lambda_vlm_token "${LAMBDA_VLM_TOKEN}"
  --lambda_vlm_region "${LAMBDA_VLM_REGION}"
  --lambda_region_token "${LAMBDA_REGION_TOKEN}"
  --tau_sparse "${TAU_SPARSE}"
  --tau_region "${TAU_REGION}"
  --lambda_xy "${LAMBDA_XY}"
  --kmeans_iters "${KMEANS_ITERS}"
  --num_global_tokens "${NUM_GLOBAL_TOKENS}"
  --num_regions "${NUM_REGIONS}"
  --reference_resampler_depth "${REFERENCE_RESAMPLER_DEPTH}"
  --reference_resampler_heads "${REFERENCE_RESAMPLER_HEADS}"
  --reference_resampler_dim_head "${REFERENCE_RESAMPLER_DIM_HEAD}"
  --reference_dropout "${REFERENCE_DROPOUT}"
  --corr_dropout "${CORR_DROPOUT}"
  --reference_scale "${REFERENCE_SCALE}"
  --corr_scale "${CORR_SCALE}"
  --corr_warmup_steps "${CORR_WARMUP_STEPS}"
  --local_start_layer "${LOCAL_START_LAYER}"
  --sparse_start_layer "${SPARSE_START_LAYER}"
  --lora_rank "${LORA_RANK}"
  --lora_dropout "${LORA_DROPOUT}"
  --validation_content_image "${VALIDATION_CONTENT_IMAGE}"
  --validation_reference_image "${VALIDATION_REFERENCE_IMAGE}"
  --validation_num_inference_steps "${VALIDATION_NUM_INFERENCE_STEPS}"
  --validation_seed "${VALIDATION_SEED}"
  --gradient_checkpointing
)

if [[ -n "${GRID_SIZE}" ]]; then
  train_args+=(--grid_size "${GRID_SIZE}")
fi

if [[ "${ADAMW_FOREACH}" == "1" ]]; then
  train_args+=(--adamw_foreach)
fi

if [[ -n "${VALIDATION_CORR_CACHE}" ]]; then
  train_args+=(--validation_corr_cache "${VALIDATION_CORR_CACHE}")
fi

if [[ "${CLEANDIFT_USE_TEXT_ENCODER}" == "1" ]]; then
  train_args+=(--cleandift_use_text_encoder)
fi

if [[ "${DISABLE_CLEANDIFT}" == "1" ]]; then
  train_args+=(--disable_cleandift)
fi

if [[ -n "${INIT_FROM_STAGE1_BASE_CHECKPOINT}" ]]; then
  train_args+=(--init_from_stage1_base_checkpoint "${INIT_FROM_STAGE1_BASE_CHECKPOINT}")
fi

if [[ -n "${LORA_BLOCKS}" ]]; then
  train_args+=(--lora_blocks "${LORA_BLOCKS}")
fi

if [[ -n "${LORA_LAYERS}" ]]; then
  train_args+=(--lora_layers "${LORA_LAYERS}")
fi

if [[ "${DISABLE_CHECKPOINT_VALIDATION}" == "1" ]]; then
  train_args+=(--disable_checkpoint_validation)
fi

cmd=(
  "${ACCELERATE_BIN}" launch
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision "${MIXED_PRECISION}"
  "${train_args[@]}"
)

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
