# SAGE-Color

**Semantic Appearance Grounding for Reference-Based Color Transfer**

SAGE-Color is a reference-based color transfer model built on Stable Diffusion
3.5 Medium. Given a content image and a reference image, it transfers the
reference image's palette, tone, contrast, and region-level color appearance
while preserving the content image's geometry, identity, layout, and fine
structure.

The core idea is to treat the reference image as **chromatic evidence**, not as a
spatial template. SAGE-Color is designed for cases where a user wants the color
language of a reference image while keeping the original content layout,
boundaries, identities, and fine details stable.

## Why SAGE-Color

- **Reference appearance without reference layout leakage.** The content image
  remains the spatial authority.
- **Semantic color grounding.** Global, regional, and local reference evidence is
  organized into a Semantic Color Gallery.
- **Structure preservation.** The Intrinsic Preservation Field uses content-only
  structure cues to attenuate unsafe reference residuals.
- **Single-checkpoint inference.** Normal usage only needs one content image, one
  reference image, and the released final checkpoint.
- **Training recipe included.** The repository includes the recommended
  two-stage recipe for researchers who want to reproduce or continue training.

## Links

- Project page source: [`docs/index.html`](docs/index.html)
- Code entrypoints: [`scripts/stage1_training`](scripts/stage1_training) and
  [`scripts/final_model`](scripts/final_model)
- Model weights: <https://huggingface.co/chenxib/sage-color>
- GitHub repository: <https://github.com/chenxib/sage-color>
- arXiv: to be updated after the arXiv identifier is assigned

## Quick Start

Python 3.11 and a CUDA-capable NVIDIA GPU are expected. The default mixed
precision is `bf16`.

```bash
git clone https://github.com/chenxib/sage-color.git
cd sage-color

conda env create -f environment.yml
conda activate zhuise-color-edit
bash scripts/bootstrap_external_diffusers.sh
pip install -r requirements.txt
```

Download the released SAGE-Color checkpoints:

```bash
pip install -U huggingface_hub
bash scripts/download_weights.sh
```

Download the required external backbones and feature extractors:

```bash
bash scripts/download_required_models.sh
```

Run inference:

```bash
CUDA_VISIBLE_DEVICES=0 \
CONTENT_IMAGE=path/to/content.png \
REFERENCE_IMAGE=path/to/reference.png \
OUTPUT_IMAGE=outputs/sage-color/sample.png \
bash scripts/final_model/bash/infer.sh
```

The default checkpoint path is:

```text
checkpoints/sage-color-final.pt
```

Override it when needed:

```bash
CHECKPOINT=/path/to/sage-color-final.pt bash scripts/final_model/bash/infer.sh
```

## Released Checkpoints

The checkpoints are hosted on Hugging Face:

<https://huggingface.co/chenxib/sage-color>

| File | Purpose |
| --- | --- |
| `checkpoints/sage-color-final.pt` | Final checkpoint for normal inference. |
| `checkpoints/sage-color-grounding.pt` | First-stage color-grounding checkpoint for continued final-stage training. |

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── docs/                         # static project page
├── model/README.md               # external model paths
├── checkpoints/README.md         # checkpoint download and placement notes
├── datasets/README.md            # JSONL data format
└── scripts/
    ├── bootstrap_external_diffusers.sh
    ├── download_required_models.sh
    ├── download_weights.sh
    ├── resolve_runtime.sh
    ├── stage1_training/          # reference color-grounding training code
    └── final_model/              # final training and inference code
```

Weights, datasets, checkpoints, generated outputs, and the local Diffusers
checkout are ignored by Git.

## Method Summary

SAGE-Color frames reference-based color transfer as **semantic appearance
grounding**. The reference image should control color appearance, but it should
not control geometry or layout.

The model separates the problem into three paths:

- **Dense Content Path:** concatenates the noisy target latent and the content
  latent, anchoring layout and geometry to the content image.
- **Semantic Color Gallery:** represents reference appearance as global,
  regional, and local chromatic evidence indexed by semantic correspondence.
- **Intrinsic Preservation Field:** derives color-free content structure cues
  from achromatic statistics, depth, and optional segmentation/panoptic priors
  to protect structure-sensitive regions.

## Data Format

Training uses JSONL. Each row should contain a content image, a reference image,
and a target image:

```json
{"content_image": "path/to/content.png", "reference_image": "path/to/reference.png", "target_image": "path/to/target.png"}
```

Batch inference also accepts:

```json
{"source": "path/to/content.png", "reference": "path/to/reference.png", "target": "path/to/target.png"}
```

Relative paths are resolved from the repository root.

## Recommended Training: Color Grounding

Single GPU:

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

Multi GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
TRAIN_JSONL=datasets/train.jsonl \
OUTPUT_DIR=outputs/stage1-ddp \
RESOLUTION=1024 \
TRAIN_BATCH_SIZE=2 \
LORA_RANK=128 \
MAX_TRAIN_STEPS=10000 \
CHECKPOINTING_STEPS=500 \
bash scripts/stage1_training/bash/train_multi_gpu.sh
```

This pass saves:

```text
outputs/stage1/checkpoint-<step>/color_edit_stage1.pt
```

## Recommended Training: Final Model

Continue from the color-grounding checkpoint:

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
COLOR_LOSS_WEIGHT=0.05 \
bash scripts/final_model/bash/train_single_gpu.sh
```

Multi GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
TRAIN_JSONL=datasets/train.jsonl \
INIT_FROM_STAGE1_CHECKPOINT=checkpoints/sage-color-grounding.pt \
OUTPUT_DIR=outputs/final-model-ddp \
RESOLUTION=1024 \
TRAIN_BATCH_SIZE=2 \
LORA_RANK=128 \
MAX_TRAIN_STEPS=10000 \
CHECKPOINTING_STEPS=500 \
LEARNING_RATE=2e-5 \
COLOR_LOSS_WEIGHT=0.05 \
bash scripts/final_model/bash/train_multi_gpu.sh
```

The final checkpoint is saved as:

```text
outputs/final-model/checkpoint-<step>/color_edit_final.pt
```

## Smoke Test

For a minimal smoke run on limited memory, set `RESOLUTION=128` or
`RESOLUTION=256`, `TRAIN_BATCH_SIZE=1`, `LORA_RANK=16`,
`MAX_TRAIN_STEPS=1`, `NUM_WORKERS=0`, and `DISABLE_CHECKPOINT_VALIDATION=1`.

## License And Dependencies

This project is released under the
[Creative Commons Attribution 4.0 International License](LICENSE).

This code release depends on third-party model licenses, including Stable
Diffusion 3.5 Medium and the feature extractors listed in
[`model/README.md`](model/README.md). Users are responsible for complying with
the licenses of those dependencies.

The Colorist-200K and Colorist-Bench-1K assets are described in the paper, but
full redistribution may be restricted by the authors' data-use agreements.

## Citation

The arXiv identifier and final citation will be added after public release.
