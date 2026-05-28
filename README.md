# HTAN — Hyper TransAttUNet

**Hyper TransAttUNet: Manifold-Constrained Hyper-Connections for Medical Image Segmentation**

HTAN integrates Manifold-Constrained Hyper-Connections (mHC) into the TransAttUNet bottleneck, achieving **90.32% Dice** on ISIC-2018 skin lesion segmentation — surpassing the TransAttUNet baseline (89.03%) across all metrics.

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

The best model (HTAN_2_n2, 90.32% Dice) is hosted on Hugging Face. Run once to download:
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

## Results (ISIC-2018)

Trained and evaluated on ISIC-2018 Task 1 (skin lesion segmentation) — 2594 dermoscopy images split into 2074 train / 520 val (seed 123). Images resized to 256×256. Augmentation: random affine, horizontal/vertical flip, color jitter.

| Model | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|
| TransAttUNet_R* | 89.03 | 80.94 | 95.62 | 87.59 | 91.84 |
| HTAN_1_n2 (Ours) | 90.22 | 82.75 | 96.06 | 88.93 | 92.65 |
| **HTAN_2_n2 (Ours)** | **90.32** | **82.93** | **96.02** | **89.76** | **92.00** |

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
# Baseline
python3 train.py --model transattunet --dataset isic

# HTAN variants
python3 train.py --model htan_2_n2 --dataset isic   # best
python3 train.py --model htan_1_n2 --dataset isic
python3 train.py --model htan_1_n4 --dataset isic
python3 train.py --model htan_1_hres_only --dataset isic  # ablation

# Baselines
python3 train.py --model unet       --dataset isic
python3 train.py --model doubleunet --dataset isic

# Evaluate
python3 evaluate.py --model all --dataset isic
```

Training resumes automatically if interrupted.

---

## Project Structure

```
HTAN/
├── configs/              # Per-model training configs
├── datasets/             # ISIC-2018 + 4 other dataset loaders
├── models/
│   ├── transattunet/     # Paper-faithful TransAttUNet_R
│   ├── htan/             # HTAN variants + mHC module
│   └── baselines/        # U-Net, DoubleU-Net
├── utils/                # Metrics, losses, trainer
├── notebooks/            # Results and visualizations
├── inference.py          # Standalone inference
├── LangGraph_tool.py          # LangGraph tool
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
@article{htan2025,
  title   = {HTAN: Hyper TransAttUNet with Manifold-Constrained Hyper-Connections for Medical Image Segmentation},
  author  = {Mohamed Khaled},
  year    = {2026}
}
```

## References

- TransAttUNet — Chen et al., IEEE T&M 2022
- mHC — Xie et al., DeepSeek-AI, arXiv:2512.24880