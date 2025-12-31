"""
Geometric normalization layer.

This layer handles:
- Deskew detection and correction
"""

import logging

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.geometric import (
    apply_rotation,
    detect_skew_hough,
    detect_skew_text,
)
from app.pipeline.steps.preprocess.utils import cv2_to_pil, image_to_base64

logger = logging.getLogger(__name__)


def apply_deskew(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
) -> tuple[np.ndarray, dict]:
    """
    Apply deskew detection and correction.
    
    This function:
    1. Detects skew angle using Hough Transform and/or text projection
    2. Applies rotation correction if angle exceeds threshold
    
    Args:
        cv_image: OpenCV image (BGR or grayscale)
        config: Preprocessing configuration
        original_image: Original PIL image for reference
        
    Returns:
        Tuple of (processed cv_image, phase_outputs_dict)
    """
    phase_outputs = {}
    original_size = cv_image.shape[:2][::-1]  # (width, height)
    
    # Convert to grayscale for processing
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
    
    deskew_outputs = {}
    deskewed_image = cv_image.copy()
    rotation_applied = False
    
    # Store original before deskew
    original_before_deskew = cv_image.copy()
    original_before_deskew_pil = cv2_to_pil(original_before_deskew)
    deskew_outputs["original_image"] = image_to_base64(original_before_deskew_pil)
    
    # Detect skew angle
    detected_angle = 0.0
    method_used = "none"
    hough_result = None
    text_result = None
    
    if config.deskew_method in ["hough", "auto"]:
        hough_angle, hough_confidence, hough_lines_img = detect_skew_hough(
            gray, config
        )
        hough_result = {
            "angle": float(hough_angle),
            "confidence": float(hough_confidence),
            "lines_visualization": image_to_base64(
                Image.fromarray(hough_lines_img)
            ) if hough_lines_img is not None else None,
        }
    
    if config.deskew_method in ["text", "auto"]:
        text_angle, text_confidence = detect_skew_text(gray, config)
        text_result = {
            "angle": float(text_angle),
            "confidence": float(text_confidence),
        }
    
    # Select best method with validation
    if config.deskew_method == "auto":
        if hough_result and text_result:
            # Use method with higher confidence
            if hough_result["confidence"] >= text_result["confidence"]:
                detected_angle = hough_result["angle"]
                method_used = "hough"
            else:
                detected_angle = text_result["angle"]
                method_used = "text"
        elif hough_result:
            detected_angle = hough_result["angle"]
            method_used = "hough"
        elif text_result:
            detected_angle = text_result["angle"]
            method_used = "text"
    elif config.deskew_method == "hough" and hough_result:
        detected_angle = hough_result["angle"]
        method_used = "hough"
    elif config.deskew_method == "text" and text_result:
        detected_angle = text_result["angle"]
        method_used = "text"
    
    # Apply rotation if angle exceeds threshold
    if abs(detected_angle) >= config.skew_threshold_degrees:
        deskewed_image = apply_rotation(cv_image, detected_angle)
        rotation_applied = True
        logger.debug(f"Applied rotation: {detected_angle:.2f}° using {method_used}")
    else:
        logger.debug(f"Skew angle {detected_angle:.2f}° below threshold, skipping rotation")
    
    # Store deskew outputs
    deskewed_pil = cv2_to_pil(deskewed_image)
    deskew_outputs["detected_angle"] = float(detected_angle)
    deskew_outputs["method_used"] = method_used
    deskew_outputs["rotation_applied"] = rotation_applied
    deskew_outputs["rotated_image"] = image_to_base64(deskewed_pil)
    if hough_result:
        deskew_outputs["hough_result"] = hough_result
    if text_result:
        deskew_outputs["text_result"] = text_result
    
    phase_outputs["deskew"] = deskew_outputs
    
    # Metadata
    phase_outputs["metadata"] = {
        "original_size": original_size,
        "final_size": deskewed_image.shape[:2][::-1],
        "rotation_applied": rotation_applied,
    }
    
    return deskewed_image, phase_outputs




