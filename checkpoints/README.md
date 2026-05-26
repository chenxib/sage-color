# Checkpoint Placement

This directory is reserved for released SAGE-Color checkpoints. Checkpoint files
are ignored by Git and distributed through Hugging Face:

<https://huggingface.co/chenxib/sage-color>

From the repository root, download them with:

```bash
pip install -U huggingface_hub
bash scripts/download_weights.sh
```

Use these default filenames if you want the wrapper scripts to work without
extra path overrides:

```text
checkpoints/sage-color-final.pt
checkpoints/sage-color-grounding.pt
```

`sage-color-final.pt` is the released model used for normal inference. Put the
final trained checkpoint at:

```text
checkpoints/sage-color-final.pt
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
CONTENT_IMAGE=path/to/content.png \
REFERENCE_IMAGE=path/to/reference.png \
OUTPUT_IMAGE=outputs/sage-color/sample.png \
bash scripts/final_model/bash/infer.sh
```

`sage-color-grounding.pt` is optional. It is only needed if you want to continue
the recommended two-stage training recipe from a prepared first-stage
checkpoint. Put that checkpoint at:

```text
checkpoints/sage-color-grounding.pt
```

Then the final-training wrapper can use it as its default initialization. If
your filenames differ, override them with:

```bash
CHECKPOINT=/path/to/final.pt bash scripts/final_model/bash/infer.sh
INIT_FROM_STAGE1_CHECKPOINT=/path/to/grounding.pt bash scripts/final_model/bash/train_single_gpu.sh
```
