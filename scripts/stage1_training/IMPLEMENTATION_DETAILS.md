# Stage 1 Implementation Details

Stage 1 builds the reference-guided color editing backbone on top of Stable Diffusion 3.5 Medium.

## Architecture

The SD3.5 transformer input is expanded so the noisy target latent and the content latent are concatenated along channels. Text conditioning is replaced by zero prompt embeddings, so the model is driven by image conditions.

Reference appearance is injected through a correspondence-guided adapter:

```text
reference image
  -> SigLIP2 tokens
  -> Lab patch statistics
  -> global / region / local reference tokens
  -> attention processors inside the SD3.5 transformer
```

Dense correspondence is computed online by fusing DINOv2 and CleanDIFT features. SigLIP2 features provide semantic gating so local matches are less likely to follow only low-level texture.

## Trainable State

The SD3.5 base weights, VAE, DINOv2, SigLIP2, and CleanDIFT extractors are frozen. Training updates:

- the reference adapter,
- LoRA weights in selected SD3.5 transformer layers,
- correspondence-aware attention processor parameters,
- the expanded input projection used by content-latent concatenation.

## Checkpoint

Stage 1 saves a lightweight checkpoint:

```text
checkpoint-<step>/color_edit_stage1.pt
```

It contains:

```text
reference_adapter
transformer_trainable
args
step
```

The final training stage restores these parameters before adding the content-only structure branch.
