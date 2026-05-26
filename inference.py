"""
inference.py — HTAN Medical Image Segmentation Inference

Usage:
    python3 inference.py --image path/to/image.jpg
    python3 inference.py --image path/to/image.jpg --checkpoint path/to/best_model.pth

Returns structured segmentation details for LLM consumption.
"""

import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from PIL import Image
import json
import cv2
from scipy import ndimage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/opt/dlami/nvme/HTAN/saves/htan_1_n2/best_model.pth"
IMG_SIZE           = 256
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# ISIC dermoscopy approximate pixel spacing (varies by device)
# This is an estimate — replace with actual DICOM pixel spacing if available
DEFAULT_PIXEL_SPACING_MM = 0.1   # 0.1mm per pixel at 256x256


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from models.htan.htan import HTAN_1
    model = HTAN_1(expansion_n=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size   # (W, H)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, original_size, image


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def predict_with_tta(model, img_tensor):
    with torch.no_grad():
        pred_orig = torch.sigmoid(model(img_tensor))
        pred_h    = torch.sigmoid(model(TF.hflip(img_tensor)))
        pred_v    = torch.sigmoid(model(TF.vflip(img_tensor)))

        pred_h = TF.hflip(pred_h)
        pred_v = TF.vflip(pred_v)

    return ((pred_orig + pred_h + pred_v) / 3.0).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Spatial location description
# ---------------------------------------------------------------------------
def describe_location(mask):
    """Returns human-readable spatial description of lesion location."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        return "no lesion detected"

    cy, cx = ys.mean() / h, xs.mean() / w

    v = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "central")
    hz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")

    if hz == "center" and v == "central":
        return "center"
    if hz == "center":
        return v
    if v == "central":
        return hz
    return f"{v}-{hz}"


# ---------------------------------------------------------------------------
# Severity estimation
# ---------------------------------------------------------------------------
def estimate_severity(area_percent, num_lesions):
    if area_percent == 0:
        return "none"
    if area_percent < 5:
        return "mild"
    if area_percent < 20:
        return "moderate"
    if area_percent < 40:
        return "severe"
    return "critical"


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
def segment(image_path, checkpoint_path=DEFAULT_CHECKPOINT,
            pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM, model=None):
    """
    Run HTAN segmentation and return structured output for LLM.

    Args:
        image_path (str):       Path to input image
        checkpoint_path (str):  Path to model weights
        pixel_spacing_mm (float): mm per pixel (for physical size estimation)
        model: pre-loaded model (optional, avoids reloading on repeated calls)

    Returns:
        dict: Structured segmentation results
    """
    if model is None:
        model = load_model(checkpoint_path)

    img_tensor, original_size, original_image = preprocess(image_path)

    # Predict with TTA
    prob_map = predict_with_tta(model, img_tensor)   # (256, 256) float [0,1]
    binary_mask = (prob_map > 0.5).astype(np.uint8)  # (256, 256) binary

    # Connected components — count individual lesions
    labeled, num_lesions = ndimage.label(binary_mask)

    # Area calculations
    total_px        = IMG_SIZE * IMG_SIZE
    lesion_px       = int(binary_mask.sum())
    area_percent    = round(lesion_px / total_px * 100, 2)
    area_mm2        = round(lesion_px * (pixel_spacing_mm ** 2), 2)
    area_cm2        = round(area_mm2 / 100, 3)

    # Per-lesion analysis
    lesion_details = []
    for i in range(1, num_lesions + 1):
        component = (labeled == i)
        comp_px   = int(component.sum())
        ys, xs    = np.where(component)
        h_span    = int(ys.max() - ys.min()) if len(ys) > 0 else 0
        w_span    = int(xs.max() - xs.min()) if len(xs) > 0 else 0
        diameter  = round(max(h_span, w_span) * pixel_spacing_mm, 2)

        lesion_details.append({
            "lesion_id":        i,
            "area_px":          comp_px,
            "area_percent":     round(comp_px / total_px * 100, 2),
            "diameter_mm":      diameter,
            "location":         describe_location(component.astype(np.uint8)),
        })

    # Sort by area descending
    lesion_details.sort(key=lambda x: x["area_px"], reverse=True)

    # Confidence score — mean probability in lesion region
    if lesion_px > 0:
        confidence = round(float(prob_map[binary_mask == 1].mean()), 4)
    else:
        confidence = 0.0

    # Largest lesion diameter
    largest_diameter = lesion_details[0]["diameter_mm"] if lesion_details else 0.0

    result = {
        # Detection
        "tumor_detected":           bool(lesion_px > 0),
        "confidence_score":         confidence,

        # Area
        "tumor_area_px":            lesion_px,
        "tumor_area_percent":       area_percent,
        "tumor_area_mm2":           area_mm2,
        "tumor_area_cm2":           area_cm2,

        # Morphology
        "num_lesions":              num_lesions,
        "largest_lesion_diameter_mm": largest_diameter,
        "lesion_location":          describe_location(binary_mask),
        "lesion_details":           lesion_details,

        # Clinical estimate
        "severity_estimate":        estimate_severity(area_percent, num_lesions),

        # Raw outputs
        "probability_map":          prob_map.tolist(),
        "binary_mask":              binary_mask.tolist(),
        "image_size":               list(original_size),
        "model":                    "HTAN_1_n2",
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True,  help="Path to input image")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pixel_spacing_mm", type=float, default=DEFAULT_PIXEL_SPACING_MM)
    args = parser.parse_args()

    result = segment(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        pixel_spacing_mm=args.pixel_spacing_mm
    )

    # Print clean summary
    print("\n" + "="*50)
    print("HTAN Segmentation Result")
    print("="*50)
    print(f"Tumor detected:     {result['tumor_detected']}")
    print(f"Confidence:         {result['confidence_score']:.2%}")
    print(f"Area:               {result['tumor_area_percent']}% ({result['tumor_area_cm2']} cm²)")
    print(f"Num lesions:        {result['num_lesions']}")
    print(f"Largest diameter:   {result['largest_lesion_diameter_mm']} mm")
    print(f"Location:           {result['lesion_location']}")
    print(f"Severity estimate:  {result['severity_estimate']}")
    print("="*50)

    # Save full JSON
    out_path = Path(args.image).stem + "_segmentation.json"
    with open(out_path, "w") as f:
        # Don't serialize the raw arrays in CLI output
        summary = {k: v for k, v in result.items()
                   if k not in ("probability_map", "binary_mask")}
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()