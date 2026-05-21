# Final Model Training And Inference

This directory contains the final SAGE-Color model used for reference-guided
color editing. It initializes from the color-grounding checkpoint, keeps the
learned reference/correspondence path, and adds a content-only structure branch.

## Main Entrypoints

```text
scripts/final_model/train.py
scripts/final_model/infer.py
scripts/final_model/infer_jsonl.py
scripts/final_model/bash/train_single_gpu.sh
scripts/final_model/bash/train_multi_gpu.sh
scripts/final_model/bash/infer.sh
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_JSONL=datasets/train.jsonl \
INIT_FROM_STAGE1_CHECKPOINT=checkpoints/sage-color-grounding.pt \
OUTPUT_DIR=outputs/final-model \
RESOLUTION=1024 \
TRAIN_BATCH_SIZE=2 \
LORA_RANK=128 \
MAX_TRAIN_STEPS=10000 \
CHECKPOINTING_STEPS=500 \
LEARNING_RATE=2e-5 \
bash scripts/final_model/bash/train_single_gpu.sh
```

The final checkpoint is saved as:

```text
outputs/final-model/checkpoint-<step>/color_edit_final.pt
```

## Structure Branch

The content-only structure branch does not consume reference color, target
color, Lab chroma, correspondence labels, or DINO/SigLIP/CleanDIFT tokens. It
uses achromatic content statistics, depth, and optional segmentation/panoptic
structure priors. Its gates are initialized as no-op so the model starts from
the color-grounding behavior and learns structure preservation during
continuation training.

## Loss

```text
loss = flow_matching_loss + COLOR_LOSS_WEIGHT * Lab(a/b)_chroma_loss
```

The default `COLOR_LOSS_WEIGHT` is `0.05`.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT=checkpoints/sage-color-final.pt \
CONTENT_IMAGE=path/to/content.png \
REFERENCE_IMAGE=path/to/reference.png \
OUTPUT_IMAGE=outputs/sage-color/sample.png \
NUM_INFERENCE_STEPS=28 \
bash scripts/final_model/bash/infer.sh
```
