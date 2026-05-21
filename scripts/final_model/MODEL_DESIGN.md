# Model Design

The model is a reference-guided color editor. It takes a content image and a reference image, then generates an output that preserves content structure while adopting the reference color appearance.

## Conditioning Paths

```text
content latent path      provides the spatial structure condition
reference color path     injects reference appearance through attention
content structure path   protects structure using color-free content priors
```

## Reference Color Path

The reference image is encoded into global, region, and local reference tokens. These tokens combine SigLIP2 semantic features, Lab patch statistics, and dense correspondence. Inside the SD3.5 transformer, image queries attend to these reference keys and values through gated processors.

## Content Structure Path

The structure path is content-only and color-free. It produces intrinsic masks, protection masks, and additive anchors that modulate attention without directly injecting reference color.

At initialization, the new structure gates are no-op. The model therefore starts from the stage-1 editor and learns to use the structure branch during final training.

## Objective

```text
loss = flow_matching_loss + COLOR_LOSS_WEIGHT * Lab(a/b)_chroma_loss
```

No auxiliary structure loss is required.
