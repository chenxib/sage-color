# External Model Weights

This repository does not commit model weights. Download them into this directory
with:

```bash
bash scripts/download_required_models.sh
```

Default required paths:

```text
model/stable-diffusion-3.5-medium
model/dinov2-large
model/siglip2-so400m-patch16-naflex
model/cleandift/cleandift_sd21_unet.safetensors
model/depth-anything-v2-base
```

Optional structure extractors:

```text
model/segformer-b0-ade
model/mask2former-swin-small-coco-panoptic
```

Stable Diffusion 3.5 Medium is gated on Hugging Face. Accept its license before
running the download helper, then authenticate with:

```bash
hf auth login
```

The CleanDIFT VAE defaults to `stabilityai/sd-vae-ft-mse` and can be downloaded
locally by running:

```bash
DOWNLOAD_CLEANDIFT_VAE=1 bash scripts/download_required_models.sh
```
