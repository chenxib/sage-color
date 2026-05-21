# Datasets

This directory is a placeholder for local training and evaluation manifests.
Image assets and generated caches are not committed.

Training JSONL rows:

```json
{"content_image": "path/to/content.png", "reference_image": "path/to/reference.png", "target_image": "path/to/target.png"}
```

Batch inference JSONL rows can also use:

```json
{"source": "path/to/content.png", "reference": "path/to/reference.png", "target": "path/to/target.png"}
```

Relative paths are resolved from the repository root. Colorist-200K and
Colorist-Bench-1K are described in the paper, but full redistribution may be
restricted by the authors' data-use agreements.

The wrapper scripts default to:

```text
datasets/train.jsonl
datasets/validation/content.png
datasets/validation/reference.png
```

Either place local files at those paths or override `TRAIN_JSONL`,
`VALIDATION_CONTENT_IMAGE`, `VALIDATION_REFERENCE_IMAGE`, `CONTENT_IMAGE`, and
`REFERENCE_IMAGE` when running the scripts.
