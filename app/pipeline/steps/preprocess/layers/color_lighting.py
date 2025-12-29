"""
Layer 1: Color & Lighting normalization.

This layer handles:
- Grayscale conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.utils import image_to_base64


def apply_color_lighting_layer(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
) -> tuple[np.ndarray, dict]:
    """
    Apply Layer 1: Color & Lighting normalization.
    
    This layer:
    1. Converts image to grayscale
    2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        cv_image: OpenCV image (BGR)
        config: Preprocessing configuration
        original_image: Original PIL image for reference
        
    Returns:
        Tuple of (processed cv_image, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Store grayscale output
    gray_pil = Image.fromarray(gray)
    gray_base64 = image_to_base64(gray_pil)
    phase_outputs["grayscale"] = {
        "image": gray_base64,
        "size": gray_pil.size,
    }
    
    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    clahe_output = clahe.apply(gray)
    
    # Store CLAHE output
    clahe_pil = Image.fromarray(clahe_output)
    clahe_base64 = image_to_base64(clahe_pil)
    phase_outputs["clahe"] = {
        "image": clahe_base64,
        "size": clahe_pil.size,
        "clip_limit": config.clahe_clip_limit,
        "tile_grid_size": config.clahe_tile_grid_size,
    }
    
    # Calculate some metadata for debugging
    phase_outputs["metadata"] = {
        "original_size": original_image.size,
        "grayscale_mean": float(np.mean(gray)),
        "grayscale_std": float(np.std(gray)),
        "clahe_mean": float(np.mean(clahe_output)),
        "clahe_std": float(np.std(clahe_output)),
    }
    
    # Return final output as BGR (convert grayscale back to BGR for consistency)
    final_bgr = cv2.cvtColor(clahe_output, cv2.COLOR_GRAY2BGR)
    
    return final_bgr, phase_outputs



