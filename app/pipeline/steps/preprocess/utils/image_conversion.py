"""
Image format conversion utilities.

Converts between PIL Image (RGB) and OpenCV (BGR) formats.
"""

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR).
    
    Args:
        pil_image: PIL Image (RGB)
        
    Returns:
        OpenCV image array (BGR)
    """
    # Convert PIL RGB to numpy array
    rgb_array = np.array(pil_image)
    
    # Convert RGB to BGR
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR) to PIL Image (RGB).
    
    Args:
        cv_image: OpenCV image array (BGR)
        
    Returns:
        PIL Image (RGB)
    """
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_array)
    
    return pil_image

