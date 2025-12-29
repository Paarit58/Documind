"""
Border detection and padding utilities.

This module provides functions for detecting content boundaries and adding padding.
"""

import cv2
import numpy as np

from app.core.config import PreprocessingConfig


def detect_content_borders(
    cv_image: np.ndarray, config: PreprocessingConfig
) -> tuple[int, int, int, int] | None:
    """
    Detect content boundaries in image.
    
    Args:
        cv_image: OpenCV image
        config: Preprocessing configuration
        
    Returns:
        Bounding box as (x, y, width, height) or None if not detected
    """
    # Convert to grayscale if needed
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
    
    # Threshold to find non-background pixels
    # Assume white/light background
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find bounding box of all contours
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add small margin
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(cv_image.shape[1] - x, w + 2 * margin)
    h = min(cv_image.shape[0] - y, h + 2 * margin)
    
    return (x, y, w, h)


def crop_to_content(
    cv_image: np.ndarray, bounding_box: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop image to content bounding box.
    
    Args:
        cv_image: OpenCV image
        bounding_box: (x, y, width, height)
        
    Returns:
        Cropped image
    """
    x, y, w, h = bounding_box
    return cv_image[y : y + h, x : x + w]


def add_padding(
    cv_image: np.ndarray, config: PreprocessingConfig
) -> tuple[np.ndarray, int]:
    """
    Add white padding around image.
    
    Args:
        cv_image: OpenCV image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (padded_image, padding_size)
    """
    h, w = cv_image.shape[:2]
    
    # Calculate padding size
    padding_percent = config.border_padding_percent / 100.0
    padding_from_size = int(min(w, h) * padding_percent)
    
    # Clamp to min/max
    padding_size = max(
        config.border_min_padding_pixels,
        min(config.border_max_padding_pixels, padding_from_size),
    )
    
    # Add padding
    padded = cv2.copyMakeBorder(
        cv_image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    
    return padded, padding_size



