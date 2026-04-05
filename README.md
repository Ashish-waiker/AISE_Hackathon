# AISEHack Phase 2 — Flood Detection (IBM)

Semi-supervised flood segmentation on multi-sensor satellite imagery using a U-Net ensemble with pseudo-labeling.

**Based on:** [Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](https://arxiv.org/abs/2107.08369) — Paul & Ganju, NeurIPS 2021 Workshop

---

## Overview

This notebook implements a 3-class semantic segmentation pipeline to detect floods from 6-channel satellite imagery. It combines supervised training on labeled data with iterative pseudo-label generation on unlabeled test patches, following the semi-supervised learning approach from the paper.

**Classes:**
- `0` — No flood
- `1` — Flood
- `2` — Permanent water body

---

## Input Data

| Source | Channels | Description |
|---|---|---|
| EOS-4 SAR | HH, HV | Synthetic aperture radar backscatter |
| Resourcesat-2 | Green, Red, NIR, SWIR | Multispectral optical bands |

- Patch size: `512 × 512` pixels
- Total input channels: **6**
- Data splits: `split/train.txt`, `split/val.txt`, `split/test.txt`

```
data/
├── image/           ← labeled + test patches (*_image.tif)
├── label/           ← ground truth masks (*_label.tif)
├── prediction/image ← unlabeled submission patches
└── split/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Pipeline

```
Labeled data
    │
    ▼
Train U-Net + U-Net++ ensemble (iteration 0)
    │
    ▼
Inference on test set with TTA → softmax probabilities
    │
    ▼
Filter high-confidence patches as pseudo-labels (conf > 0.7, pix_frac > 0.7)
    │
    ▼
Merge pseudo-labels + original labels → retrain (up to 3 cycles)
    │
    ▼
Final ensemble inference on submission set
    │
    ▼
Post-processing (confidence threshold + morphological cleanup)
    │
    ▼
Column-major RLE encoding → submission.csv
```

---

## Model Architecture

### Ensemble Members

| Model | Encoder | Role |
|---|---|---|
| U-Net | EfficientNet-B4 | Strong multi-scale features |
| U-Net++ | ResNet-50 | Dense skip connections, complementary bias |
| DeepLabV3+ | ResNet-50 | ASPP for multi-scale context |

All encoders are pretrained on ImageNet. The first convolution is patched to accept 6-channel input by tiling the 3-channel pretrained weights twice and halving their magnitude to preserve activation scale.

### Test-Time Augmentation

Dihedral Group D4 (8 transforms): horizontal flip, vertical flip, 4 × 90° rotations and their combinations. Predictions are averaged in probability space.

---

## Loss Function

```
Loss = 0.4 × Dice + 0.4 × Focal(γ=2) + 0.2 × Lovász-Softmax
```

| Component | Purpose |
|---|---|
| Multi-class Dice | Handles class imbalance |
| Focal loss (γ=2) | Up-weights hard, rare flood pixels |
| Lovász-Softmax | Directly optimises the IoU metric |

Class weights are computed from the actual pixel frequency in the training set (inverse frequency normalised), not hardcoded.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 6 |
| Optimizer | AdamW (weight decay 1e-4) |
| Initial LR (decoder) | 1e-3 |
| Initial LR (encoder) | 1e-4 (10× lower) |
| LR schedule | Linear warmup (5 epochs) → cosine decay |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Pseudo-label cycles | 3 |

### Data Augmentations (Training)

- Horizontal / vertical flips
- Random 90° rotations
- Shift, scale, rotate (±30°)
- Elastic transform
- Multiplicative noise (SAR speckle simulation)
- Grid distortion
- Random brightness / contrast
- Gaussian noise
- Coarse dropout (cutout)

---

## Semi-Supervised Pseudo-Labeling

Implements **Algorithm 1** from Paul & Ganju (2021):

1. Train on labeled data → infer on test set with ensemble + TTA
2. Accept patch as pseudo-label if: `fraction of pixels with max-confidence > 0.7` exceeds `0.6`
3. Merge accepted pseudo-labels with original training data → retrain
4. Repeat until IoU plateaus (delta < 0.002) or max cycles reached

> **Note:** The original paper used thresholds of 0.9/0.9. These are relaxed to 0.7/0.6 because early-stage models (flood IoU ~0.17) are not confident enough to pass the stricter filter — the cycle would never run.

---

## Post-Processing

Applied to ensemble softmax probabilities before RLE encoding:

1. **Water-body reclassification** — pixels predicted as water body with P(flood) > 0.25 are reclassified as flood (inundated land, not permanent water)
2. **Confidence threshold** — flood pixels with P(flood) < 0.35 are reassigned to the next-best class
3. **Morphological opening** — removes SAR speckle noise (isolated 1–2 pixel flood spots)
4. **Small region removal** — connected flood blobs smaller than 400 pixels are discarded
5. **Morphological closing** — fills small holes within flood regions

---

## Submission Format

Column-major (Fortran-order) run-length encoding of the **flood class only** (class == 1).

- Empty masks → `"0 0"`
- Format: `start1 length1 start2 length2 ...` (1-indexed, column-major)
- Output: `submission_final.csv` with columns `id`, `rle_mask`

---

## Dependencies

```
numpy==2.2.6
scipy==1.15.3
albumentations==1.4.18
segmentation-models-pytorch
ttach
rasterio
tifffile
torch >= 2.0
opencv-python
```

Install with:
```bash
pip install numpy==2.2.6 scipy==1.15.3 albumentations==1.4.18 \
    segmentation-models-pytorch ttach rasterio tifffile
```

> **Kaggle note:** Run the install cell first, then **restart the kernel** before running any other cells. The Kaggle environment ships conflicting numpy/scipy versions that must be replaced before albumentations can import.

---

## Notebook Cell Reference

| Cell | Description |
|---|---|
| 0 | Install dependencies |
| 1 | Imports & hyperparameter config |
| 2 | Data exploration + dataset-level normalisation stats |
| 3 | `FloodDataset` and `FloodTestDataset` classes |
| 4 | Augmentation pipeline + TTA definition |
| 5 | Stratified sampler (3× weight for flood patches) |
| 6 | Loss functions (Dice + Focal + Lovász) |
| 7 | Model definitions (U-Net, U-Net++, DeepLabV3+) |
| 8 | Training loop, metrics, early stopping |
| 9 | Initial training on labeled data (all 3 models) |
| 10 | Ensemble + TTA inference on test & submission sets |
| 11 | Pseudo-label filtering |
| 12 | Pseudo-label training cycle |
| 13 | IBM Prithvi foundation model (optional) |
| 14 | Temperature calibration + final inference |
| 15 | Post-processing (confidence threshold + morphology) |
| 16 | Train + val merge for final submission model |
| 17 | RLE encoding + submission CSV generation |
| 18 | Prediction visualisation |

---

## Optional: IBM Prithvi Foundation Model

Cell 13 includes code to swap in the [IBM Prithvi-EO-v2-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) geospatial foundation model as an additional ensemble member. Prithvi is pretrained on Sentinel-2 and HLS data — the same sensor family as Resourcesat-2 — making it domain-matched for this task.

```bash
pip install terratorch
```

Then uncomment the Prithvi block in Cell 13.

---

## Reference

```
@article{paul2021flood,
  title   = {Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning},
  author  = {Paul, Sayak and Ganju, Siddha},
  journal = {NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning},
  year    = {2021}
}
```