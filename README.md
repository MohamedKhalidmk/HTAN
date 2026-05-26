# HTAN — Hyper TransAttUNet

**Hyper TransAttUNet: Manifold-Constrained Hyper-Connections for Medical Image Segmentation**

HTAN integrates Manifold-Constrained Hyper-Connections (mHC) into the TransAttUNet bottleneck, achieving **90.22% Dice** on ISIC-2018 skin lesion segmentation — surpassing the TransAttUNet baseline (89.03%) across all metrics.

---

## For MediLink Team

You only need 2 files: `inference.py` and `htan_tool.py`

1. Clone the repo
2. Install dependencies:
```bash
pip install torch torchvision scipy opencv-python langchain-core langgraph pydantic
```
3. Get `best_model.pth` from MK and place it at:
```
saves/htan_1_n2/best_model.pth
```
4. In your LangGraph agent add:
```python
from htan_tool import htan_segmentation_tool
tools = [..., htan_segmentation_tool]
```

---

## Results (ISIC-2018)

| Model | Dice | IoU | ACC | REC | PRE |
|---|---|---|---|---|---|
| TransAttUNet_R* | 89.03 | 80.94 | 95.62 | 87.59 | 91.84 |
| **HTAN_1_n2 (Ours)** | **90.22** | **82.75** | **96.06** | **88.93** | **92.65** |

*Results marked with \* are reproduced using our experimental setup (seed 123, 2074/520 split).*

---

## Architecture

HTAN extends TransAttUNet by replacing the bottleneck with a **Manifold-Constrained Hyper-Connection (mHC)** module that:
- Expands the residual stream into n parallel streams
- Applies the SAA (Self-Aware Attention) module across streams
- Constrains the residual mixing matrix to the Birkhoff polytope via Sinkhorn-Knopp
- Aggregates streams back to the original channel dimension

```
Input → Encoder → [mHC Bottleneck] → Decoder → Output Mask
                        ↑
              SAA wrapped in mHC (n=2 streams)
```

---

## Setup

```bash
git clone https://github.com/MohamedKhalidmk/HTAN.git
cd HTAN
pip install torch torchvision Pillow requests numpy matplotlib scipy opencv-python
pip install langchain-core langgraph pydantic
touch configs/__init__.py datasets/__init__.py models/__init__.py \
      models/transattunet/__init__.py models/htan/__init__.py \
      models/baselines/__init__.py utils/__init__.py
```

---

## Inference

### Basic usage

```python
from inference import segment

result = segment(image_path="your_image.jpg")
print(result["tumor_detected"])        # True/False
print(result["tumor_area_percent"])    # e.g. 12.5
print(result["severity_estimate"])     # mild/moderate/severe/critical
print(result["lesion_location"])       # e.g. "center", "upper-left"
```

### CLI

```bash
python3 inference.py --image your_image.jpg
```

### Output structure

```json
{
  "tumor_detected": true,
  "confidence_score": 0.91,
  "tumor_area_percent": 12.5,
  "tumor_area_cm2": 2.3,
  "num_lesions": 1,
  "largest_lesion_diameter_mm": 8.3,
  "lesion_location": "center",
  "severity_estimate": "moderate",
  "lesion_details": [...]
}
```

---

## LangGraph Integration (MediLink)

```python
from htan_tool import htan_segmentation_tool, build_medilink_agent
from langchain_anthropic import ChatAnthropic

llm   = ChatAnthropic(model="claude-sonnet-4-20250514")
agent = build_medilink_agent(llm)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this image: /path/to/image.jpg"}]
})
```

The tool returns structured segmentation data that Claude uses to generate clinical summaries.

---

## Training

```bash
# TransAttUNet baseline
python3 train.py --model transattunet --dataset isic

# HTAN variants
python3 train.py --model htan_1_n2 --dataset isic   # 1 mHC block, n=2 (best)
python3 train.py --model htan_1_n4 --dataset isic   # 1 mHC block, n=4
python3 train.py --model htan_2_n2 --dataset isic   # 2 mHC blocks, n=2
python3 train.py --model htan_2_n4 --dataset isic   # 2 mHC blocks, n=4

# Baselines
python3 train.py --model unet       --dataset isic
python3 train.py --model doubleunet --dataset isic
```

Training resumes automatically from the last checkpoint if interrupted.

```bash
# Evaluate
python3 evaluate.py --model htan_1_n2 --dataset isic
python3 evaluate.py --model all       --dataset isic
```

---

## Project Structure

```
HTAN/
├── configs/          # Training configs per model
├── datasets/         # Dataset loaders (ISIC-2018 + 4 others)
├── models/
│   ├── transattunet/ # Paper-faithful TransAttUNet_R
│   ├── htan/         # HTAN variants + mHC module
│   └── baselines/    # U-Net, DoubleU-Net
├── utils/            # Metrics, losses, trainer
├── notebooks/        # Results notebook
├── inference.py      # Standalone inference script
├── htan_tool.py      # LangGraph tool wrapper
├── train.py          # Training CLI
└── evaluate.py       # Evaluation CLI
```

---

## Contributing

1. Create a new branch: `git checkout -b your-feature`
2. Make changes and push: `git push origin your-feature`
3. Open a Pull Request for review

Direct pushes to `main` are not allowed.

---

## Citation

```
@article{htan2025,
  title={HTAN: Hyper TransAttUNet with Manifold-Constrained Hyper-Connections 
         for Medical Image Segmentation},
  author={Mohamed Khalid},
  year={2025}
}
```

---

## References

- TransAttUNet: Chen et al., IEEE T&M 2022
- mHC: Xie et al., DeepSeek-AI, arXiv:2512.24880