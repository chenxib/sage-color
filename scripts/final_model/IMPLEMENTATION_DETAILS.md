# Final Model Implementation Details

The final model continues from the stage-1 reference-guided editor and adds a content-only structure branch for better structure preservation.

## Initialization

Training requires:

```text
--init_from_stage1_checkpoint /path/to/color_edit_stage1.pt
```

The loader restores:

```text
reference_adapter
transformer_trainable
```

If the checkpoint already contains final-model structure state, the loader also restores:

```text
intrinsic_prior_tokenizer
corr_reference_processors
```

## No-Op Structure Start

The structure branch is designed not to disturb the stage-1 editor at initialization:

```text
key_bias_gate = 0
ref_protect_gate = 0
anchor_head final layer = 0
anchor_gate = 0.10
```

This lets the new branch receive gradients while keeping the initial forward behavior close to the stage-1 checkpoint.

## Structure Inputs

The structure branch only consumes content-derived, color-free signals:

```text
achromatic luminance statistics
edge and high-frequency statistics
Depth Anything structure
optional grayscale-fed SegFormer / Mask2Former structure
```

It does not consume reference image color, target image color, Lab chroma, correspondence labels, DINO tokens, SigLIP tokens, or CleanDIFT tokens.

## Checkpoint

The final checkpoint is:

```text
checkpoint-<step>/color_edit_final.pt
```

It contains:

```text
reference_adapter
intrinsic_prior_tokenizer
corr_reference_processors
transformer_trainable
args
step
```
