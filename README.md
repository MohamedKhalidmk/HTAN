# HTAN — Hyper TransAttUNet

**Hyper TransAttUNet: Manifold-Constrained Hyper-Connections for Medical Image Segmentation**

HTAN integrates Manifold-Constrained Hyper-Connections (mHC) into the TransAttUNet bottleneck, achieving consistent improvements across three medical imaging benchmarks.

---

## For MediLink Team

> You only need `inference.py` and `LangGraph_tool.py` from this repo.

**1. Clone and install**
```bash
git clone https://github.com/MohamedKhalidmk/HTAN.git
cd HTAN
pip install torch torchvision scipy opencv-python langchain-core langgraph pydantic huggingface_hub
```

**2. Create package init files**
```bash
touch configs/__init__.py datasets/__init__.py models/__init__.py \
      models/transattunet/__init__.py models/htan/__init__.py \
      models/baselines/__init__.py utils/__init__.py
```

**3. Download model weights**
```bash
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('saves/htan_2_n2', exist_ok=True)
hf_hub_download(repo_id='mohamedkhaledmk7/HTAN', filename='htan_2_n2/best_model.pth', local_dir='.')
print('Done.')
"
```

**4. Test inference**
```bash
python3 inference.py --image your_image.jpg
```

Output:
```
Tumor detected:     True
Confidence:         91.23%
Area:               12.50% (2.30 cm²)
Num lesions:        1
Largest diameter:   8.30 mm
Location:           center
Severity estimate:  moderate
```

**5. Add to your LangGraph agent**
```python
from LangGraph_tool import htan_segmentation_tool
tools = [...your_existing_tools..., htan_segmentation_tool]
```

---

## Results

All HTAN variants trained with AdamW + cosine LR schedule, seed 123.
Baselines on ISIC use SGD (faithful to original TransAttUNet paper protocol).
GlaS and Bowl use AdamW for all models.

### ISIC-2018 — Skin Lesion Segmentation
2594 dermoscopy images · 2074 train / 520 val · 256×256 · 100 epochs

| Model | Params | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|---|
| U-Net* | 17.3M | 87.76 | 79.08 | 95.10 | 86.20 | 90.81 |
| DoubleU-Net* (pretrained†) | 9.2M | 86.42 | 77.25 | 94.35 | 87.27 | 87.68 |
| Swin-UNet‡ | — | 89.72 | 82.90 | — | 90.32 | 92.04 |
| SegFormer‡ | — | 90.24 | 83.60 | — | 91.12 | 92.10 |
| MCTrans‡ | — | 90.35 | — | — | — | — |
| TransAttUNet_R* | 41.3M | 89.27 | 81.24 | 95.72 | 87.62 | 92.24 |
| HTAN_1 Hres-only (Ours) | 67M | 89.95 | 82.41 | 95.92 | 89.42 | 91.78 |
| HTAN_1 n=2 (Ours) | 47M | 90.15 | 82.71 | 96.07 | 89.12 | 92.36 |
| HTAN_1 n=4 (Ours) | 75M | 90.46 | 83.07 | 96.13 | 89.55 | 92.33 |
| HTAN_2 n=4 (Ours) | 129M | 90.06 | 82.62 | 96.07 | 88.75 | 92.77 |
| **HTAN_2 n=2 (Ours)** | **61M** | **90.32** | **82.93** | **96.02** | **89.77** | **92.00** |

### GlaS — Gland Segmentation (MICCAI 2015)
165 H&E histology images · 85 train / 80 test (fixed official split) · 128×128 · 150 epochs

| Model | Params | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|---|
| U-Net* | 17.3M | 90.27 | 82.77 | 90.55 | 89.36 | 91.54 |
| DoubleU-Net* (pretrained†) | 9.2M | 90.85 | 83.53 | 90.90 | 90.89 | 90.99 |
| TransAttUNet_R* | 41.3M | 88.10 | 79.10 | 87.35 | 94.28 | 82.86 |
| HTAN_1 Hres-only (Ours) | 54M | 91.05 | 83.96 | 91.04 | 91.59 | 90.63 |
| HTAN_1 n=4 (Ours) | 61M | 91.34 | 84.40 | 91.47 | 90.99 | 91.87 |
| HTAN_2 n=2 (Ours) | 51M | 91.50 | 84.65 | 91.58 | 92.21 | 90.99 |
| **HTAN_1 n=2 (Ours)** | **47M** | **91.67** | **84.94** | **91.71** | **91.71** | **91.76** |

### Bowl — Nuclei Segmentation (2018 Data Science Bowl)
670 fluorescence microscopy images · 80/10/10 split · 256×256 · 100 epochs

| Model | Params | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|---|
| TransAttUNet_R* | 41.3M | 91.07 | 83.81 | 96.94 | 91.90 | 90.56 |
| HTAN_1 Hres-only (Ours) | 67M | 91.23 | 84.07 | 97.03 | 89.54 | 93.26 |
| U-Net* | 17.3M | 92.01 | 85.37 | 97.30 | 91.02 | 93.27 |
| HTAN_1 n=4 (Ours) | 75M | 92.11 | 85.49 | 97.28 | 91.84 | 92.61 |
| HTAN_2 n=2 (Ours) | 61M | 92.13 | 85.54 | 97.28 | 92.09 | 92.36 |
| **HTAN_1 n=2 (Ours)** | **47M** | **92.14** | **85.55** | **97.31** | **91.74** | **92.74** |
| DoubleU-Net* (pretrained†) | 9.2M | 92.91 | 86.86 | 97.54 | 92.45 | 93.56 |

*\* Reproduced in our experimental setup.*
*† DoubleU-Net uses a frozen pretrained ImageNet VGG16 encoder — not training from scratch.*
*‡ Results from original publications.*

---

## Which weights should I use?

| Your task | Recommended weights | Dice |
|---|---|---|
| Skin lesion / dermoscopy | `htan_2_n2/best_model.pth` | 90.32% |
| Histology / gland segmentation | `htan_1_n2_glas/best_model.pth` | 91.67% |
| Cell nuclei / fluorescence microscopy | `htan_1_n2_bowl/best_model.pth` | 92.14% |
| General / unknown modality | `htan_2_n2/best_model.pth` | 90.32% |

---

## Model Variants

| Variant | Params | Description |
|---|---|---|
| `htan_2_n2` | 61M | 2 mHC blocks, n=2 streams. Best on ISIC. Recommended for general use. |
| `htan_1_n2` | 47M | 1 mHC block, n=2 streams. Best on GlaS and Bowl. |
| `htan_1_n4` | 75M | 1 mHC block, n=4 streams. Diminishing returns vs n=2. |
| `htan_2_n4` | 129M | 2 mHC blocks, n=4 streams. Over-parameterized. |
| `htan_1_hres_only` | 67M | Ablation — constrains H_res stream only. Weaker than full mHC. |
| `transattunet` | 41.3M | Baseline. Use for comparison only. |

---

## Weights on HuggingFace

All weights at [mohamedkhaledmk7/HTAN](https://huggingface.co/mohamedkhaledmk7/HTAN).

| Weight | Dataset | Dice |
|---|---|---|
| `htan_2_n2/best_model.pth` | ISIC | 90.32% |
| `htan_1_n2/best_model.pth` | ISIC | 90.15% |
| `htan_1_n4/best_model.pth` | ISIC | 90.46% |
| `htan_2_n4/best_model.pth` | ISIC | 90.06% |
| `htan_1_hres_only/best_model.pth` | ISIC | 89.95% |
| `transattunet/best_model.pth` | ISIC | 89.27% |
| `unet/best_model.pth` | ISIC | 87.76% |
| `doubleunet/best_model.pth` | ISIC | 86.42% |
| `htan_1_n2_glas/best_model.pth` | GlaS | 91.67% |
| `htan_2_n2_glas/best_model.pth` | GlaS | 91.50% |
| `htan_1_n4_glas/best_model.pth` | GlaS | 91.34% |
| `htan_1_hres_only_glas/best_model.pth` | GlaS | 91.05% |
| `unet_glas/best_model.pth` | GlaS | 90.27% |
| `doubleunet_glas/best_model.pth` | GlaS | 90.85% |
| `transattunet_glas/best_model.pth` | GlaS | 88.10% |
| `htan_1_n2_bowl/best_model.pth` | Bowl | 92.14% |
| `htan_2_n2_bowl/best_model.pth` | Bowl | 92.13% |
| `htan_1_n4_bowl/best_model.pth` | Bowl | 92.11% |
| `unet_bowl/best_model.pth` | Bowl | 92.01% |
| `doubleunet_bowl/best_model.pth` | Bowl | 92.91% |
| `htan_1_hres_only_bowl/best_model.pth` | Bowl | 91.23% |
| `transattunet_bowl/best_model.pth` | Bowl | 91.07% |

---

## Architecture

HTAN wraps the TransAttUNet bottleneck with a Manifold-Constrained Hyper-Connection (mHC) module:

```
Input → Encoder → [mHC Bottleneck] → Decoder → Output Mask
                        ↑
             SAA inside n parallel residual streams
             constrained to Birkhoff polytope via Sinkhorn-Knopp
```

The mHC module expands the bottleneck into n parallel residual streams, applies the Self-Aware Attention (TSA + GSA) across them, and aggregates back — enabling richer feature mixing without breaking the identity mapping property.

---

## Training

```bash
# Baselines
python3 train.py --model transattunet --dataset isic
python3 train.py --model transattunet --dataset glas
python3 train.py --model transattunet --dataset bowl

# HTAN variants
python3 train.py --model htan_2_n2 --dataset isic   # best on ISIC
python3 train.py --model htan_1_n2 --dataset glas   # best on GlaS
python3 train.py --model htan_1_n2 --dataset bowl   # best on Bowl

# All variants
python3 train.py --model htan_1_n2 --dataset isic
python3 train.py --model htan_1_n4 --dataset isic
python3 train.py --model htan_1_hres_only --dataset isic
python3 train.py --model htan_2_n4 --dataset isic

# Evaluate
python3 evaluate.py --model all --dataset isic
python3 evaluate.py --model all --dataset glas
python3 evaluate.py --model all --dataset bowl
```

Training resumes automatically if interrupted.

---

## Evaluate

```bash
python3 evaluate.py --model all --dataset isic
python3 evaluate.py --model all --dataset glas
python3 evaluate.py --model all --dataset bowl
python3 evaluate.py --model htan_2_n2 --dataset isic
```

---

## Project Structure

```
HTAN/
├── configs/              # Per-model training configs
├── datasets/             # ISIC, GlaS, Bowl dataset loaders
├── models/
│   ├── transattunet/     # TransAttUNet_R
│   ├── htan/             # HTAN variants + mHC module
│   └── baselines/        # U-Net, DoubleU-Net
├── utils/                # Metrics, losses, trainer
├── notebooks/            # Experiment results and visuals
├── inference.py          # Standalone inference
├── LangGraph_tool.py     # LangGraph tool for MediLink
├── train.py              # Training CLI
└── evaluate.py           # Evaluation CLI
```

---

## Contributing

```bash
git checkout -b your-feature
git push origin your-feature
# Open a Pull Request — direct pushes to main are not allowed
```

---

## Citation

```bibtex
@article{htan2026,
  title   = {HTAN: Hyper TransAttUNet with Manifold-Constrained Hyper-Connections for Medical Image Segmentation},
  author  = {Mohamed Khaled},
  year    = {2026}
}
```

## References

- TransAttUNet — Chen et al., IEEE T&M 2022. arXiv:2107.05274
- mHC — Xie et al., DeepSeek-AI, arXiv:2512.24880
- ISIC-2018 — Codella et al., ISIC 2018 Challenge
- GlaS — Warwick-QU Dataset, MICCAI 2015
- Bowl — Caicedo et al., Nature Methods 2019