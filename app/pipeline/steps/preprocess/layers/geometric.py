"""
Layer 2: Geometric normalization.

This layer handles:
- Deskew detection and correction
- Border detection and padding
"""

import logging

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.geometric import (
    add_padding,
    apply_rotation,
    crop_to_content,
    detect_content_borders,
    detect_skew_hough,
    detect_skew_text,
)
from app.pipeline.steps.preprocess.utils import cv2_to_pil, image_to_base64

logger = logging.getLogger(__name__)


def apply_geometric_layer(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
) -> tuple[np.ndarray, dict]:
    """
    Apply Layer 2: Geometric normalization.
    
    This layer:
    1. Deskewing: Detects and corrects image rotation
    2. Border Detection & Padding: Detects content boundaries, crops, and adds padding
    
    Args:
        cv_image: OpenCV image (BGR)
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
    
    # Phase 2.1: Deskewing
    deskew_outputs = {}
    deskewed_image = cv_image.copy()
    rotation_applied = False
    
    if config.enable_geometric_layer:  # Check if deskewing is enabled
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
    deskewed_size = deskewed_image.shape[:2][::-1]  # (width, height)
    
    # Phase 2.2: Border Detection & Padding
    border_outputs = {}
    final_image = deskewed_image.copy()
    
    if config.enable_border_detection:
        # Detect content boundaries
        bounding_box = detect_content_borders(deskewed_image, config)
        border_outputs["bounding_box"] = bounding_box
        
        # Visualize bounding box
        bbox_vis = deskewed_image.copy()
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bbox_vis_pil = cv2_to_pil(bbox_vis)
        border_outputs["bounding_box_visualization"] = image_to_base64(bbox_vis_pil)
        
        # Crop to content
        if bounding_box:
            cropped_image = crop_to_content(deskewed_image, bounding_box)
            cropped_pil = cv2_to_pil(cropped_image)
            border_outputs["cropped_image"] = image_to_base64(cropped_pil)
            border_outputs["cropped_size"] = cropped_image.shape[:2][::-1]
            
            # Add padding
            padded_image, padding_size = add_padding(cropped_image, config)
            final_image = padded_image
            padded_pil = cv2_to_pil(padded_image)
            border_outputs["padded_image"] = image_to_base64(padded_pil)
            border_outputs["padding_size"] = int(padding_size)
            border_outputs["final_size"] = padded_image.shape[:2][::-1]
        else:
            # No content detected, just add padding to whole image
            padded_image, padding_size = add_padding(deskewed_image, config)
            final_image = padded_image
            padded_pil = cv2_to_pil(padded_image)
            border_outputs["padded_image"] = image_to_base64(padded_pil)
            border_outputs["padding_size"] = int(padding_size)
            border_outputs["final_size"] = padded_image.shape[:2][::-1]
    
    phase_outputs["border"] = border_outputs
    
    # Metadata
    phase_outputs["metadata"] = {
        "original_size": original_size,
        "deskewed_size": deskewed_size,
        "final_size": final_image.shape[:2][::-1],
        "rotation_applied": rotation_applied,
    }
    
    return final_image, phase_outputs



