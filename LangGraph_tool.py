"""
htan_tool.py — LangGraph Tool Wrapper for HTAN Segmentation

Integrates HTAN medical image segmentation into MediLink's LangGraph pipeline.

Usage in LangGraph:
    from htan_tool import htan_segmentation_tool, HTANSegmentationInput
    tools = [htan_segmentation_tool]
"""

import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class HTANSegmentationInput(BaseModel):
    image_path: str = Field(
        description="Absolute path to the dermoscopy or medical image file to segment. "
                    "Supported formats: JPG, PNG, JPEG."
    )
    pixel_spacing_mm: Optional[float] = Field(
        default=0.1,
        description="Physical pixel spacing in mm per pixel. "
                    "Use DICOM metadata if available. Default 0.1mm."
    )
    checkpoint_path: Optional[str] = Field(
        default="saves/htan_2_n2/best_model.pth",
        description="Path to HTAN model weights. Use default unless testing a different variant."
    )


# ---------------------------------------------------------------------------
# Lazy model loader — loads once and caches
# ---------------------------------------------------------------------------
_model_cache = {}

def _get_model(checkpoint_path):
    if checkpoint_path not in _model_cache:
        from inference import load_model
        _model_cache[checkpoint_path] = load_model(checkpoint_path)
    return _model_cache[checkpoint_path]


# ---------------------------------------------------------------------------
# LangGraph Tool
# ---------------------------------------------------------------------------
@tool(args_schema=HTANSegmentationInput)
def htan_segmentation_tool(
    image_path: str,
    pixel_spacing_mm: float = 0.1,
    checkpoint_path: str = "saves/htan_2_n2/best_model.pth"
) -> str:
    """
    Analyzes a medical image using the HTAN (Hyper TransAttUNet) deep learning model
    to detect and segment skin lesions or tumors.

    Returns a structured clinical summary including:
    - Whether a tumor/lesion was detected
    - Lesion area (pixels, percentage, mm², cm²)
    - Number of distinct lesion regions
    - Largest lesion diameter in mm
    - Lesion spatial location (e.g., upper-left, center)
    - Severity estimate (none/mild/moderate/severe/critical)
    - Confidence score of the prediction

    Use this tool when a patient image needs to be analyzed for:
    - Skin lesion detection and measurement
    - Melanoma screening support
    - Pre-surgical lesion mapping
    - Tracking lesion changes over time
    """
    try:
        # Validate image exists
        if not Path(image_path).exists():
            return json.dumps({
                "error": f"Image not found: {image_path}",
                "suggestion": "Please provide the full absolute path to the image file."
            })

        # Run inference
        from inference import segment
        model = _get_model(checkpoint_path)
        result = segment(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            pixel_spacing_mm=pixel_spacing_mm,
            model=model
        )

        # Return LLM-friendly summary (exclude raw arrays)
        summary = {
            "tumor_detected":               result["tumor_detected"],
            "confidence_score":             result["confidence_score"],
            "tumor_area_percent":           result["tumor_area_percent"],
            "tumor_area_cm2":               result["tumor_area_cm2"],
            "num_lesions":                  result["num_lesions"],
            "largest_lesion_diameter_mm":   result["largest_lesion_diameter_mm"],
            "lesion_location":              result["lesion_location"],
            "severity_estimate":            result["severity_estimate"],
            "lesion_details":               result["lesion_details"],
            "model_used":                   result["model"],
            "image_analyzed":               image_path,
        }

        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "image_path": image_path
        })


# ---------------------------------------------------------------------------
# Example LangGraph agent setup
# ---------------------------------------------------------------------------
def build_medilink_agent(llm):
    """
    Example of how to integrate htan_segmentation_tool into MediLink's
    LangGraph agent.

    Args:
        llm: Your Claude Sonnet model instance

    Returns:
        Compiled LangGraph agent with HTAN tool
    """
    from langgraph.prebuilt import create_react_agent

    tools = [htan_segmentation_tool]

    system_prompt = """You are MediLink's medical AI assistant specializing in 
    dermatology image analysis. When a user provides an image path for analysis,
    use the htan_segmentation_tool to analyze it and provide a clear clinical 
    summary of the findings.
    
    When reporting results:
    - State clearly whether a lesion was detected
    - Report the lesion size in both percentage and cm²
    - Describe the location in plain language
    - Give the severity estimate with appropriate medical context
    - Always remind the user that AI analysis is assistive and not a substitute 
      for professional medical diagnosis
    """

    agent = create_react_agent(
        llm,
        tools=tools,
        state_modifier=system_prompt
    )

    return agent


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 htan_tool.py path/to/image.jpg")
        sys.exit(1)

    result = htan_segmentation_tool.invoke({
        "image_path": sys.argv[1]
    })
    print(result)