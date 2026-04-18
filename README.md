[README.md](https://github.com/user-attachments/files/26852616/README.md)
# Image Classification: CNN vs ViT vs YOLO with Knowledge Distillation

This repository contains notebook-based experiments for image classification across two datasets using:
- Convolutional Neural Networks (CNN)
- Vision Transformers (ViT)
- YOLO for classification
- Knowledge Distillation (KD)

The main goal is to compare architecture families under a consistent pipeline and analyze the trade-off between accuracy and efficiency.

## Project Structure

```text
cnn_vit/
├── dataset-1/
│   ├── cnn/
│   ├── vit/
│   └── yolo-93.ipynb
└── dataset-2/
    ├── CNN/
    └── VIT/
```

## Datasets

### Dataset-1 (RealWaste subset)
- 9 classes
- Total images: 4,752
- Representative classes: cardboard, food organics, glass, metal, paper, plastic, textile, vegetation, miscellaneous trash

### Dataset-2 (Garbage subset)
- 10 classes
- Test samples: 1,226
- Classes: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash

## Implementations by Notebook

## Dataset-1

### CNN notebooks
- `dataset-1/cnn/efficientnet-b3-dataset-1.ipynb`
- `dataset-1/cnn/mobilenet-v3-large-dataset-1.ipynb`
- `dataset-1/cnn/resnet50-dataset-1.ipynb`

### ViT notebooks
- `dataset-1/vit/deit-tiny-patch16-224-dataset-1.ipynb`
- `dataset-1/vit/deit-base-patch16-224-dataset-1.ipynb`
- `dataset-1/vit/pvt-v2-b2-dataset-1.ipynb`
- `dataset-1/vit/pvt-v2-b3-dataset-1.ipynb`
- `dataset-1/vit/repvit-v2-dataset-1.ipynb`
- `dataset-1/vit/swin-transformer.ipynb`

### Knowledge Distillation notebook
- `dataset-1/vit/swin-transformer-with-kd.ipynb`

Teacher-student setup follows a Swin-based teacher transferring soft-label knowledge to a lighter student through a weighted CE + KL objective.

### YOLO notebook
- `dataset-1/yolo-93.ipynb`

YOLO is used in classification mode to benchmark fast single-stage feature extraction against CNN/ViT baselines.

## Dataset-2

### CNN notebooks
- `dataset-2/CNN/efficientnet-b3-data-2.ipynb`
- `dataset-2/CNN/mobilenet-v3-large-data-2.ipynb`
- `dataset-2/CNN/resnet50-data-2.ipynb`

### ViT notebooks
- `dataset-2/VIT/vit-large-patch16-224-dataset-2.ipynb`
- `dataset-2/VIT/pvt-v2-b3-dataset-2.ipynb`
- `dataset-2/VIT/pvt-v2-b5-dataset-2.ipynb`
- `dataset-2/VIT/repvit-dataset-2.ipynb`
- `dataset-2/VIT/swin-transformer -dataset-2.ipynb`

## Training Strategy

Experiments follow a common strategy to keep comparisons fair:
- Unified preprocessing and dataset split per dataset
- Transfer learning / pretrained backbones where available
- Standard classification objective with cross-entropy
- Architecture-specific optimizer strategy:
  - ViT models: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
  - CNN and YOLO backbones: SGD
- Cosine annealing learning-rate schedules used across model families
- Batch-based GPU training and test-time evaluation with per-class metrics

### Knowledge Distillation Strategy

The KD setup follows a standard teacher-student loss:

\[
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{CE}(y, \sigma(z_s))
+ (1-\alpha)\tau^2 \cdot \mathcal{L}_{KL}\left(\sigma\left(\frac{z_t}{\tau}\right), \sigma\left(\frac{z_s}{\tau}\right)\right)
\]

Where:
- `z_t`: teacher logits (Swin Transformer)
- `z_s`: student logits (lightweight classifier)
- `tau`: temperature for soft targets
- `alpha`: weight between hard-label CE and distillation KL loss

## Results

## Dataset-1 test accuracy

| Model | Notebook | Accuracy |
|---|---|---:|
| EfficientNet-B3 | `dataset-1/cnn/efficientnet-b3-dataset-1.ipynb` | 88.66% |
| MobileNetV3-Large | `dataset-1/cnn/mobilenet-v3-large-dataset-1.ipynb` | 92.65% |
| ResNet50 | `dataset-1/cnn/resnet50-dataset-1.ipynb` | 92.86% |
| YOLO (classification) | `dataset-1/yolo-93.ipynb` | 93.28% |
| DeiT-Tiny | `dataset-1/vit/deit-tiny-patch16-224-dataset-1.ipynb` | 89.50% |
| DeiT-Base | `dataset-1/vit/deit-base-patch16-224-dataset-1.ipynb` | 90.34% |
| PVT-v2-B2 | `dataset-1/vit/pvt-v2-b2-dataset-1.ipynb` | 91.81% |
| PVT-v2-B3 | `dataset-1/vit/pvt-v2-b3-dataset-1.ipynb` | 92.23% |
| RepViT-v2 | `dataset-1/vit/repvit-v2-dataset-1.ipynb` | 92.23% |
| Swin Transformer | `dataset-1/vit/swin-transformer.ipynb` | **96.22%** |
| Swin + KD | `dataset-1/vit/swin-transformer-with-kd.ipynb` | **96.22%** |

## Dataset-2 test accuracy

| Model | Notebook | Accuracy |
|---|---|---:|
| MobileNetV3-Large | `dataset-2/CNN/mobilenet-v3-large-data-2.ipynb` | 93.47% |
| ResNet50 | `dataset-2/CNN/resnet50-data-2.ipynb` | 95.43% |
| EfficientNet-B3 | `dataset-2/CNN/efficientnet-b3-data-2.ipynb` | 96.00% |
| ViT-Large | `dataset-2/VIT/vit-large-patch16-224-dataset-2.ipynb` | 93.96% |
| PVT-v2-B3 | `dataset-2/VIT/pvt-v2-b3-dataset-2.ipynb` | 95.35% |
| PVT-v2-B5 | `dataset-2/VIT/pvt-v2-b5-dataset-2.ipynb` | 94.86% |
| RepViT | `dataset-2/VIT/repvit-dataset-2.ipynb` | 92.25% |
| Swin Transformer | `dataset-2/VIT/swin-transformer -dataset-2.ipynb` | **97.96%** |

## Key Observations

- Swin Transformer gives the best accuracy on both datasets.
- YOLO (classification mode) is highly competitive on Dataset-1 while keeping a speed-oriented architecture.
- CNN baselines remain strong, especially on Dataset-2 where EfficientNet-B3 reaches 96.00%.
- KD setup demonstrates how transformer-level knowledge can be transferred to efficient students.

## Environment

Based on the experiment setup used in notebooks/thesis:
- Python + Jupyter Notebook
- PyTorch
- `timm`
- Ultralytics (YOLO)
- CUDA-enabled GPU recommended (experiments mention Tesla T4)

## How to Use

1. Install dependencies used in your notebooks (PyTorch, timm, ultralytics, sklearn, matplotlib, etc.).
2. Open the desired notebook from `dataset-1` or `dataset-2`.
3. Update dataset paths in the first setup cells.
4. Run training and evaluation cells.
5. Compare metrics with the benchmark tables above.


