# Color Grounding Training

This is the first pass in the recommended training recipe. It trains the
reference-guided color editing backbone and learns how to combine the content
latent with reference appearance tokens and correspondence-guided attention.

## Inputs

Each training JSONL row contains:

```json
{"content_image": "path/to/content.png", "reference_image": "path/to/reference.png", "target_image": "path/to/target.png"}
```

## Main Entrypoints

```text
scripts/stage1_training/train.py
scripts/stage1_training/infer.py
scripts/stage1_training/infer_jsonl.py
scripts/stage1_training/bash/train_single_gpu.sh
scripts/stage1_training/bash/train_multi_gpu.sh
scripts/stage1_training/bash/infer.sh
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_JSONL=datasets/train.jsonl \
OUTPUT_DIR=outputs/stage1 \
RESOLUTION=1024 \
TRAIN_BATCH_SIZE=2 \
LORA_RANK=128 \
MAX_TRAIN_STEPS=10000 \
CHECKPOINTING_STEPS=500 \
bash scripts/stage1_training/bash/train_single_gpu.sh
```

The checkpoint is saved as:

```text
outputs/stage1/checkpoint-<step>/color_edit_stage1.pt
```

## What This Stage Learns

- SD3.5 image latent conditioning with content-latent concatenation.
- Reference token construction from SigLIP2 features and Lab patch statistics.
- Dense correspondence from DINOv2 and CleanDIFT features.
- Global, region, and local sparse reference attention.

The model weights saved here are the required initialization for final model
training.
