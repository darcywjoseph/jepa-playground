# jepa-playground
This repo serves as a playground for me to explore and implement a variety of jepa variants in an attempt to understand.

## The JEPA Family

### 1. I-JEPA — Image JEPA
**Paper:** [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (Assran et al., 2023)


### 2. V-JEPA — Video JEPA
**Paper:** [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08518) (Bardes et al., 2024)

### 3. V-JEPA 2
**Paper:** [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (Meta, 2025)

## Repo Layout

```
jepa-playground/
├── jepa/
│   ├── encoder.py
│   ├── predictor.py
│   ├── ema.py
│   └── masking/
│       ├── block.py
│       └── tube.py
├── variants/
│   ├── ijepa/
│   │   ├── model.py
│   │   └── train.py
│   ├── vjepa/
│   │   ├── model.py
│   │   └── train.py
│   └── vjepa2/
│       ├── model.py
│       └── train.py
├── tests/
├── experiments/
├── reference/
└── pyproject.toml
```
## Shared Architecture (`jepa/`)

All variants share a common skeleton.

```
jepa/
├── encoder.py
├── predictor.py
├── ema.py
└── masking/
    ├── block.py
    └── tube.py
```

## Setup

```bash
uv sync
uv run *
```

---

## Resources

- [LeCun's paper on world models](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [I-JEPA blog post](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [V-JEPA blog post](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
