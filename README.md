# SAGE-Color

**Semantic Appearance Grounding for Reference-Based Color Transfer**

SAGE-Color is a reference-based color transfer model built on Stable Diffusion
3.5 Medium. Given a content image and an arbitrary reference image, it transfers
the reference palette, tone, contrast, and region-level chromatic appearance
while preserving the content image's geometry, identity, layout, and fine
structure.

This folder is the standalone release project distilled from the larger
research workspace. It intentionally contains the final two-stage recipe only:

1. **Stage I / v1.1 reference grounding** trains the reference-guided color
   editing backbone with dense content-latent conditioning, DINOv2 + CleanDIFT
   correspondence, SigLIP2 semantic gating, and global/region/local reference
   attention.
2. **Stage II / v3.5 final model** initializes from the Stage-I checkpoint,
   activates the content-only Intrinsic Preservation Field, and applies the
   Stage-II Lab(a/b) chroma calibration loss.

The earlier three-stage experimental plan is not part of this release.

## Links

- Project page: <https://chenxib.github.io/sage-color/>
- Code entrypoints: [`scripts/stage1_training`](scripts/stage1_training) and
  [`scripts/final_model`](scripts/final_model)
- GitHub repository: <https://github.com/chenxib/sage-color>
- arXiv: to be updated after the arXiv identifier is assigned

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── docs/                         # static project page
├── model/README.md               # expected external model-weight paths
├── datasets/README.md            # expected JSONL format and data notes
└── scripts/
    ├── bootstrap_external_diffusers.sh
    ├── download_required_models.sh
    ├── resolve_runtime.sh
    ├── stage1_training/          # Stage I / v1.1 release code
    └── final_model/              # Stage II / v3.5 final release code
```

Weights, datasets, checkpoints, generated outputs, and the local Diffusers
checkout are intentionally ignored by Git.

## Method Summary

The paper frames reference-based color transfer as **semantic appearance
grounding**: the reference image has chromatic authority, but not spatial
authority. SAGE-Color separates the problem into three paths:

- **Dense content path:** concatenates the noisy target latent and the content
  latent, anchoring layout and geometry to the content image.
- **Semantic Color Gallery:** represents reference appearance as global,
  regional, and local chromatic evidence indexed by semantic correspondence.
- **Intrinsic Preservation Field:** derives color-free content structure cues
  from achromatic statistics, depth, and optional segmentation/panoptic priors
  to protect structure-sensitive regions.

## Fresh Clone Setup

Python 3.11 and a CUDA-capable NVIDIA GPU are expected. The default mixed
precision is `bf16`. Run the setup commands from the repository root.

```bash
git clone https://github.com/chenxib/sage-color.git
cd sage-color

conda env create -f environment.yml
conda activate zhuise-color-edit
bash scripts/bootstrap_external_diffusers.sh
pip install -r requirements.txt
```

`environment.yml` only creates the Python environment. The editable Diffusers
install is intentionally kept in `requirements.txt`, because
`external/diffusers` does not exist until `scripts/bootstrap_external_diffusers.sh`
has run.

## Required External Models

The default paths are documented in [`model/README.md`](model/README.md). After
accepting the gated Stable Diffusion 3.5 Medium license on Hugging Face and
logging in with `hf auth login`, run:

```bash
bash scripts/download_required_models.sh
```

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

## Stage I Training

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

Stage I saves:

```text
outputs/stage1/checkpoint-<step>/color_edit_stage1.pt
```

## Stage II Final Training

Continue from the Stage-I checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_JSONL=datasets/train.jsonl \
INIT_FROM_STAGE1_CHECKPOINT=outputs/stage1/checkpoint-10000/color_edit_stage1.pt \
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
INIT_FROM_STAGE1_CHECKPOINT=outputs/stage1-ddp/checkpoint-10000/color_edit_stage1.pt \
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

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT=outputs/final-model/checkpoint-10000/color_edit_final.pt \
CONTENT_IMAGE=path/to/content.png \
REFERENCE_IMAGE=path/to/reference.png \
OUTPUT_IMAGE=outputs/final-model/sample.png \
NUM_INFERENCE_STEPS=28 \
bash scripts/final_model/bash/infer.sh
```

For a minimal smoke run on limited memory, set `RESOLUTION=128` or
`RESOLUTION=256`, `TRAIN_BATCH_SIZE=1`, `LORA_RANK=16`,
`MAX_TRAIN_STEPS=1`, `NUM_WORKERS=0`, and `DISABLE_CHECKPOINT_VALIDATION=1`.

## Project Page Assets

The static project page in [`docs/`](docs) includes lightweight preview images
derived only from figures used in the paper package. The original paper source
is intentionally not included in this code repository.

## License And Data Notes

This project is released under the
[Creative Commons Attribution 4.0 International License](LICENSE).

This code release depends on third-party model licenses, including Stable
Diffusion 3.5 Medium and the feature extractors listed in
[`model/README.md`](model/README.md). The Colorist-200K and Colorist-Bench-1K
assets are described in the paper, but full redistribution may be restricted by
the authors' data-use agreements.
