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

The best model (HTAN_2_n2) is hosted on Hugging Face. Run once to download:
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

Claude receives structured JSON from every image analysis:
```json
{
  "tumor_detected": true,
  "confidence_score": 0.91,
  "tumor_area_percent": 12.5,
  "tumor_area_cm2": 2.3,
  "num_lesions": 1,
  "largest_lesion_diameter_mm": 8.3,
  "lesion_location": "center",
  "severity_estimate": "moderate"
}
```

> **Notes:** GPU recommended. Images auto-resized to 256×256. Model loads once and caches. Update `pixel_spacing_mm` with DICOM metadata for accurate physical measurements.

---

## Results

All models trained with AdamW + cosine LR schedule, seed 123. HTAN_2_n2 is the best performing variant across all datasets.

### ISIC-2018 — Skin Lesion Segmentation
2594 dermoscopy images, 2074 train / 520 val, 256×256.

| Model | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|
| TransAttUNet_R* | 89.27 | 81.24 | 95.72 | 87.62 | 92.24 |
| HTAN_1_n2 (Ours) | 90.15 | 82.71 | 96.07 | 89.12 | 92.36 |
| **HTAN_2_n2 (Ours)** | **90.32** | **82.93** | **96.02** | **89.76** | **92.00** |

### GlaS — Gland Segmentation (MICCAI 2015)
165 H&E histology images, 85 train / 80 test (fixed official split), 128×128.

| Model | Dice |
|---|---|
| TransAttUNet_R* | 90.37 |
| HTAN_1_n2 (Ours) | 90.64 |
| **HTAN_2_n2 (Ours)** | **90.78** |

### Bowl — Nuclei Segmentation (2018 Data Science Bowl)
670 fluorescence microscopy images, 80/10/10 split, 256×256.

| Model | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|
| TransAttUNet_R* | 91.07 | 83.81 | 96.94 | 91.90 | 90.56 |
| **HTAN_2_n2 (Ours)** | **92.13** | **85.54** | **97.28** | **92.09** | **92.36** |

*\* Reproduced using our experimental setup.*

---

## Architecture

HTAN wraps the TransAttUNet bottleneck with a Manifold-Constrained Hyper-Connection (mHC) module:

```
Input → Encoder → [mHC Bottleneck] → Decoder → Output Mask
                        ↑
             SAA inside n parallel residual streams
             constrained to Birkhoff polytope via Sinkhorn-Knopp
```

The mHC module expands the bottleneck into n parallel residual streams, applies the Self-Aware Attention (TSA + GSA) module across them, and aggregates back — enabling richer feature mixing without breaking the identity mapping property.

---

## Training

```bash
# Baselines
python3 train.py --model transattunet --dataset isic
python3 train.py --model transattunet --dataset glas
python3 train.py --model transattunet --dataset bowl

# HTAN variants
python3 train.py --model htan_2_n2 --dataset isic   # best
python3 train.py --model htan_1_n2 --dataset isic
python3 train.py --model htan_1_n4 --dataset isic
python3 train.py --model htan_1_hres_only --dataset isic  # ablation

# Evaluate
python3 evaluate.py --model all --dataset isic
python3 evaluate.py --model all --dataset glas
python3 evaluate.py --model all --dataset bowl
```

Training resumes automatically if interrupted.

---

## Weights on HuggingFace

All trained weights are available at [mohamedkhaledmk7/HTAN](https://huggingface.co/mohamedkhaledmk7/HTAN).

| Weight | Dataset | Dice |
|---|---|---|
| `htan_2_n2/best_model.pth` | ISIC | 90.32% |
| `htan_1_n2/best_model.pth` | ISIC | 90.15% |
| `htan_1_n4/best_model.pth` | ISIC | 90.14% |
| `htan_1_hres_only/best_model.pth` | ISIC | 89.95% |
| `transattunet/best_model.pth` | ISIC | 89.27% |
| `unet/best_model.pth` | ISIC | 87.76% |
| `doubleunet/best_model.pth` | ISIC | 86.42% |
| `htan_2_n2_glas/best_model.pth` | GlaS | 90.78% |
| `htan_1_n2_glas/best_model.pth` | GlaS | 90.64% |
| `transattunet_glas/best_model.pth` | GlaS | 90.37% |
| `htan_2_n2_bowl/best_model.pth` | Bowl | 92.13% |
| `transattunet_bowl/best_model.pth` | Bowl | 91.07% |

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
├── notebooks/            # Experiment results
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

- TransAttUNet — Chen et al., IEEE T&M 2022
- mHC — Xie et al., DeepSeek-AI, arXiv:2512.24880

---

## Datasets & Model Selection Guide

HTAN was trained and evaluated on three medical imaging datasets, each representing a different imaging modality. Depending on your use case, you should pick the right weights.

### Which weights should I use?

| Your task | Recommended weights |
|---|---|
| Skin lesion / dermoscopy | `htan_2_n2/best_model.pth` (ISIC) |
| Histology / gland segmentation | `htan_2_n2_glas/best_model.pth` (GlaS) |
| Cell nuclei / fluorescence microscopy | `htan_2_n2_bowl/best_model.pth` (Bowl) |
| General / unknown modality | `htan_2_n2/best_model.pth` (ISIC — largest training set) |

---

### ISIC-2018 — Skin Lesion Segmentation
- **What it is:** Dermoscopy images of skin lesions from the ISIC 2018 challenge
- **Task:** Binary segmentation — lesion vs background
- **Images:** 2594 total, RGB, resized to 256×256
- **Split:** 2074 train / 520 val (seed 123)
- **Best for:** Skin cancer screening, dermatology applications
- **HTAN Dice:** 90.32%

### GlaS — Gland Segmentation
- **What it is:** H&E stained histology images from the MICCAI 2015 Gland Segmentation Challenge (Warwick-QU dataset)
- **Task:** Binary segmentation — gland vs background
- **Images:** 165 total, RGB, resized to 128×128
- **Split:** 85 train / 80 test (official fixed split)
- **Best for:** Colon cancer pathology, histology analysis
- **HTAN Dice:** 90.78%

### Bowl — Nuclei Segmentation
- **What it is:** Fluorescence and brightfield microscopy images from the 2018 Data Science Bowl (Kaggle)
- **Task:** Binary segmentation — nuclei vs background (multiple masks per image merged into one)
- **Images:** 670 total, RGB, resized to 256×256
- **Split:** 80/10/10 train/val/test (seed 123)
- **Best for:** Cell biology, drug discovery, pathology automation
- **HTAN Dice:** 92.13%

---

### Model Variants Explained

| Variant | Params | Description |
|---|---|---|
| `htan_2_n2` | 61M | **Best overall.** 2 mHC blocks, n=2 expansion. Recommended for all tasks. |
| `htan_1_n2` | 47M | 1 mHC block, n=2 expansion. Good balance of accuracy and speed. |
| `htan_1_n4` | 75M | 1 mHC block, n=4 expansion. More parameters, similar accuracy to htan_1_n2. |
| `htan_1_hres_only` | 67M | Ablation variant — only constrains H_res stream. Weaker than full mHC. |
| `transattunet` | 41M | Baseline. Use for comparison only. |

**For MediLink:** Always use `htan_2_n2` — it's the best model across all datasets and modalities.