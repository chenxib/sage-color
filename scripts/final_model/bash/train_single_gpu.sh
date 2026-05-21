#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/final_model/bash/train_single_gpu.sh
#
# Common overrides:
#   TRAIN_JSONL=datasets/train.jsonl \
#   INIT_FROM_STAGE1_CHECKPOINT=outputs/stage1/checkpoint-57000/color_edit_stage1.pt \
#   OUTPUT_DIR=outputs/final-model-online-smoke \
#   RESOLUTION=256 LORA_RANK=16 MAX_TRAIN_STEPS=1 CHECKPOINTING_STEPS=1 NUM_WORKERS=0 \
#   bash scripts/final_model/bash/train_single_gpu.sh
#
# Notes:
# - Training computes correspondence online on GPU with DINOv2 + CleanDIFT
#   dense matching plus SigLIP2 semantic gating.
#   No prebuilt .npz cache is required.
# - This loads SD3.5, SD3 VAE, SD2.1 VAE, CleanDIFT UNet, DINOv2, SigLIP2,
#   and the adapter on the selected GPU.
#   If you hit OOM, first reduce RESOLUTION or use fewer validation steps.
# - CLEANDIFT_VAE defaults to stabilityai/sd-vae-ft-mse. The first run
#   may download the SD2.1 VAE unless you point it to a local SD2.1 folder.
# - DISABLE_CLEANDIFT=1 runs the DINO+SigLIP ablation and is not full final model.
# - CHECKPOINTING_STEPS controls both checkpoint saves and validation_sample.png.
# - Set DISABLE_CHECKPOINT_VALIDATION=1 to skip validation image generation.
# - VALIDATION_CORR_CACHE is optional; when empty, validation also computes
#   correspondence online.
# - final model continues from a stage-1 checkpoint and adds the Stage-II Lab(a/b) color loss.

source scripts/resolve_runtime.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$("${PYTHON_BIN}" scripts/final_model/gpu_select.py)"
fi

PRETRAINED_MODEL="${PRETRAINED_MODEL:-model/stable-diffusion-3.5-medium}"
TRAIN_JSONL="${TRAIN_JSONL:-datasets/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/final-model}"
INIT_FROM_STAGE1_CHECKPOINT="${INIT_FROM_STAGE1_CHECKPOINT:-}"

if [[ -z "${INIT_FROM_STAGE1_CHECKPOINT}" ]]; then
  echo "ERROR: final model requires INIT_FROM_STAGE1_CHECKPOINT=/path/to/stage-1/color_edit_stage1.pt" >&2
  exit 2
fi

RESOLUTION="${RESOLUTION:-1024}"
GRID_SIZE="${GRID_SIZE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
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
DEPTH_MODEL="${DEPTH_MODEL:-model/depth-anything-v2-base}"
USE_DEPTH="${USE_DEPTH:-1}"
DISABLE_DEPTH="${DISABLE_DEPTH:-0}"
SEGMENTATION_MODEL="${SEGMENTATION_MODEL:-model/segformer-b0-ade}"
USE_SEGMENTATION="${USE_SEGMENTATION:-0}"
DISABLE_SEGMENTATION="${DISABLE_SEGMENTATION:-0}"
PANOPTIC_MODEL="${PANOPTIC_MODEL:-model/mask2former-swin-small-coco-panoptic}"
USE_PANOPTIC="${USE_PANOPTIC:-0}"
DISABLE_PANOPTIC="${DISABLE_PANOPTIC:-0}"
DISABLE_CIG="${DISABLE_CIG:-0}"
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
REFERENCE_DROPOUT="${REFERENCE_DROPOUT:-0.0}"
CORR_DROPOUT="${CORR_DROPOUT:-0.0}"
REFERENCE_SCALE="${REFERENCE_SCALE:-1.0}"
CORR_SCALE="${CORR_SCALE:-1.0}"
CORR_WARMUP_STEPS="${CORR_WARMUP_STEPS:-0}"
LOCAL_START_LAYER="${LOCAL_START_LAYER:-4}"
SPARSE_START_LAYER="${SPARSE_START_LAYER:-5}"
CIG_HIDDEN_DIM="${CIG_HIDDEN_DIM:-0}"
CIG_NUM_MASKS="${CIG_NUM_MASKS:-6}"
CIG_RUNTIME_START_STEP="${CIG_RUNTIME_START_STEP:-0}"
CIG_RUNTIME_RAMP_STEPS="${CIG_RUNTIME_RAMP_STEPS:-0}"
ANCHOR_START_STEP="${ANCHOR_START_STEP:-0}"
ANCHOR_RAMP_STEPS="${ANCHOR_RAMP_STEPS:-0}"
CIG_MAX_KEY_BIAS="${CIG_MAX_KEY_BIAS:-0.8}"
CIG_MAX_ANCHOR_SCALE="${CIG_MAX_ANCHOR_SCALE:-0.15}"
CIG_MAX_LOCAL_PROTECT="${CIG_MAX_LOCAL_PROTECT:-0.30}"
CIG_MAX_REGION_PROTECT="${CIG_MAX_REGION_PROTECT:-0.12}"
CIG_KEY_BIAS_SCALE="${CIG_KEY_BIAS_SCALE:-1.0}"
CIG_ANCHOR_SCALE="${CIG_ANCHOR_SCALE:-1.0}"
CIG_REF_PROTECT_SCALE="${CIG_REF_PROTECT_SCALE:-1.0}"

REFERENCE_RESAMPLER_DEPTH="${REFERENCE_RESAMPLER_DEPTH:-4}"
REFERENCE_RESAMPLER_HEADS="${REFERENCE_RESAMPLER_HEADS:-16}"
REFERENCE_RESAMPLER_DIM_HEAD="${REFERENCE_RESAMPLER_DIM_HEAD:-64}"

LORA_RANK="${LORA_RANK:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_BLOCKS="${LORA_BLOCKS:-}"
LORA_LAYERS="${LORA_LAYERS:-}"
USE_COLOR_LOSS="${USE_COLOR_LOSS:-1}"
COLOR_LOSS_WEIGHT="${COLOR_LOSS_WEIGHT:-0.05}"
COLOR_DECODE_RESOLUTION="${COLOR_DECODE_RESOLUTION:-0}"
COLOR_LOSS_SIGMA_POWER="${COLOR_LOSS_SIGMA_POWER:-2.0}"

VALIDATION_CONTENT_IMAGE="${VALIDATION_CONTENT_IMAGE:-datasets/validation/content.png}"
VALIDATION_REFERENCE_IMAGE="${VALIDATION_REFERENCE_IMAGE:-datasets/validation/reference.png}"
VALIDATION_CORR_CACHE="${VALIDATION_CORR_CACHE:-}"
VALIDATION_NUM_INFERENCE_STEPS="${VALIDATION_NUM_INFERENCE_STEPS:-4}"
VALIDATION_SEED="${VALIDATION_SEED:-42}"
DISABLE_CHECKPOINT_VALIDATION="${DISABLE_CHECKPOINT_VALIDATION:-0}"

cmd=(
  "${PYTHON_BIN}" scripts/final_model/train.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --init_from_stage1_checkpoint "${INIT_FROM_STAGE1_CHECKPOINT}"
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
  --depth_model "${DEPTH_MODEL}"
  --segmentation_model "${SEGMENTATION_MODEL}"
  --panoptic_model "${PANOPTIC_MODEL}"
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
  --cig_hidden_dim "${CIG_HIDDEN_DIM}"
  --cig_num_masks "${CIG_NUM_MASKS}"
  --cig_runtime_start_step "${CIG_RUNTIME_START_STEP}"
  --cig_runtime_ramp_steps "${CIG_RUNTIME_RAMP_STEPS}"
  --anchor_start_step "${ANCHOR_START_STEP}"
  --anchor_ramp_steps "${ANCHOR_RAMP_STEPS}"
  --cig_max_key_bias "${CIG_MAX_KEY_BIAS}"
  --cig_max_anchor_scale "${CIG_MAX_ANCHOR_SCALE}"
  --cig_max_local_protect "${CIG_MAX_LOCAL_PROTECT}"
  --cig_max_region_protect "${CIG_MAX_REGION_PROTECT}"
  --cig_key_bias_scale "${CIG_KEY_BIAS_SCALE}"
  --cig_anchor_scale "${CIG_ANCHOR_SCALE}"
  --cig_ref_protect_scale "${CIG_REF_PROTECT_SCALE}"
  --lora_rank "${LORA_RANK}"
  --lora_dropout "${LORA_DROPOUT}"
  --color_loss_weight "${COLOR_LOSS_WEIGHT}"
  --color_decode_resolution "${COLOR_DECODE_RESOLUTION}"
  --color_loss_sigma_power "${COLOR_LOSS_SIGMA_POWER}"
  --validation_content_image "${VALIDATION_CONTENT_IMAGE}"
  --validation_reference_image "${VALIDATION_REFERENCE_IMAGE}"
  --validation_num_inference_steps "${VALIDATION_NUM_INFERENCE_STEPS}"
  --validation_seed "${VALIDATION_SEED}"
  --gradient_checkpointing
)

if [[ -n "${GRID_SIZE}" ]]; then
  cmd+=(--grid_size "${GRID_SIZE}")
fi

if [[ "${ADAMW_FOREACH}" == "1" ]]; then
  cmd+=(--adamw_foreach)
fi

if [[ -n "${VALIDATION_CORR_CACHE}" ]]; then
  cmd+=(--validation_corr_cache "${VALIDATION_CORR_CACHE}")
fi

if [[ "${CLEANDIFT_USE_TEXT_ENCODER}" == "1" ]]; then
  cmd+=(--cleandift_use_text_encoder)
fi

if [[ "${DISABLE_CLEANDIFT}" == "1" ]]; then
  cmd+=(--disable_cleandift)
fi

if [[ "${USE_DEPTH}" == "0" ]]; then
  cmd+=(--no-use_depth)
fi

if [[ "${DISABLE_DEPTH}" == "1" ]]; then
  cmd+=(--disable_depth)
fi

if [[ "${USE_SEGMENTATION}" == "1" ]]; then
  cmd+=(--use_segmentation)
fi

if [[ "${DISABLE_SEGMENTATION}" == "1" ]]; then
  cmd+=(--disable_segmentation)
fi

if [[ "${USE_PANOPTIC}" == "1" ]]; then
  cmd+=(--use_panoptic)
fi

if [[ "${DISABLE_PANOPTIC}" == "1" ]]; then
  cmd+=(--disable_panoptic)
fi

if [[ "${DISABLE_CIG}" == "1" ]]; then
  cmd+=(--disable_cig)
fi

if [[ "${USE_COLOR_LOSS}" != "1" ]]; then
  cmd+=(--no-use_color_loss)
fi

if [[ -n "${LORA_BLOCKS}" ]]; then
  cmd+=(--lora_blocks "${LORA_BLOCKS}")
fi

if [[ -n "${LORA_LAYERS}" ]]; then
  cmd+=(--lora_layers "${LORA_LAYERS}")
fi

if [[ "${DISABLE_CHECKPOINT_VALIDATION}" == "1" ]]; then
  cmd+=(--disable_checkpoint_validation)
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
