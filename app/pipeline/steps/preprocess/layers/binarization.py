"""
Layer 4: Binarization normalization.

This layer handles:
- Otsu's thresholding
- Sauvola's adaptive thresholding
"""

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.utils import image_to_base64


def apply_binarization_layer(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
) -> tuple[np.ndarray, dict]:
    """
    Apply Layer 4: Binarization normalization.
    
    This layer:
    1. Converts to grayscale (if needed)
    2. Applies Otsu's thresholding
    3. Applies Sauvola's thresholding
    4. Shows both outputs for comparison
    
    Args:
        cv_image: OpenCV image (BGR or grayscale)
        config: Preprocessing configuration
        original_image: Original PIL image for reference
        
    Returns:
        Tuple of (processed cv_image, phase_outputs_dict)
    """
    phase_outputs = {}
    original_size = cv_image.shape[:2][::-1]  # (width, height)
    
    # Phase 4.1: Ensure grayscale
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
    
    # Store grayscale image
    gray_pil = Image.fromarray(gray)
    phase_outputs["grayscale"] = {
        "image": image_to_base64(gray_pil),
        "size": gray_pil.size,
    }
    
    # Phase 4.2: Otsu's Thresholding
    otsu_outputs = {"enabled": False}
    otsu_binary = None
    
    if config.enable_otsu:
        # Map threshold type string to OpenCV constant
        threshold_type_map = {
            "BINARY": cv2.THRESH_BINARY,
            "BINARY_INV": cv2.THRESH_BINARY_INV,
            "TRUNC": cv2.THRESH_TRUNC,
            "TOZERO": cv2.THRESH_TOZERO,
            "TOZERO_INV": cv2.THRESH_TOZERO_INV,
        }
        threshold_type = threshold_type_map.get(
            config.otsu_threshold_type, cv2.THRESH_BINARY
        )
        
        # Apply Otsu's thresholding
        threshold_value, otsu_binary = apply_otsu_threshold(
            gray, config.otsu_max_value, threshold_type
        )
        
        # Calculate quality metric
        quality_metric = calculate_binarization_quality(otsu_binary)
        
        # Store original grayscale
        original_before_otsu_pil = Image.fromarray(gray)
        
        otsu_binary_pil = Image.fromarray(otsu_binary)
        otsu_outputs = {
            "enabled": True,
            "threshold_value": float(threshold_value),
            "threshold_type": config.otsu_threshold_type,
            "original_image": image_to_base64(original_before_otsu_pil),
            "binary_image": image_to_base64(otsu_binary_pil),
            "quality_metric": float(quality_metric),
        }
    
    phase_outputs["otsu"] = otsu_outputs
    
    # Phase 4.3: Sauvola's Thresholding
    sauvola_outputs = {"enabled": False}
    sauvola_binary = None
    
    if config.enable_sauvola:
        # Calculate parameters
        if (
            config.sauvola_window_size is not None
            and config.sauvola_k is not None
            and config.sauvola_r is not None
        ):
            window_size = config.sauvola_window_size
            k = config.sauvola_k
            r = config.sauvola_r
        else:
            window_size, k, r = calculate_sauvola_params(gray, config)
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply Sauvola's thresholding
        sauvola_binary = apply_sauvola_threshold(gray, window_size, k, r)
        
        # Calculate quality metric
        quality_metric = calculate_binarization_quality(sauvola_binary)
        
        # Store original grayscale
        original_before_sauvola_pil = Image.fromarray(gray)
        
        sauvola_binary_pil = Image.fromarray(sauvola_binary)
        sauvola_outputs = {
            "enabled": True,
            "window_size": int(window_size),
            "k": float(k),
            "r": float(r),
            "original_image": image_to_base64(original_before_sauvola_pil),
            "binary_image": image_to_base64(sauvola_binary_pil),
            "quality_metric": float(quality_metric),
        }
    
    phase_outputs["sauvola"] = sauvola_outputs
    
    # Phase 4.4: Output Selection
    final_image = cv_image.copy()
    method_used = "none"
    
    if config.binarization_output_method == "otsu" and otsu_outputs["enabled"] and otsu_binary is not None:
        # Convert binary to BGR for consistency
        if len(cv_image.shape) == 3:
            final_image = cv2.cvtColor(otsu_binary, cv2.COLOR_GRAY2BGR)
        else:
            final_image = otsu_binary
        method_used = "otsu"
    elif (
        config.binarization_output_method == "sauvola"
        and sauvola_outputs["enabled"]
        and sauvola_binary is not None
    ):
        # Convert binary to BGR for consistency
        if len(cv_image.shape) == 3:
            final_image = cv2.cvtColor(sauvola_binary, cv2.COLOR_GRAY2BGR)
        else:
            final_image = sauvola_binary
        method_used = "sauvola"
    elif config.binarization_output_method == "both":
        # Use Sauvola as default if both are enabled (changed from Otsu)
        if sauvola_outputs["enabled"] and sauvola_binary is not None:
            if len(cv_image.shape) == 3:
                final_image = cv2.cvtColor(sauvola_binary, cv2.COLOR_GRAY2BGR)
            else:
                final_image = sauvola_binary
            method_used = "sauvola"
        elif otsu_outputs["enabled"] and otsu_binary is not None:
            if len(cv_image.shape) == 3:
                final_image = cv2.cvtColor(otsu_binary, cv2.COLOR_GRAY2BGR)
            else:
                final_image = otsu_binary
            method_used = "otsu"
    
    # Metadata
    phase_outputs["metadata"] = {
        "original_size": original_size,
        "final_size": final_image.shape[:2][::-1],
        "method_used": method_used,
        "binarization_applied": method_used != "none",
    }
    
    return final_image, phase_outputs


def apply_otsu_threshold(
    gray_image: np.ndarray, max_value: int, threshold_type: int
) -> tuple[float, np.ndarray]:
    """
    Apply Otsu's thresholding method.
    
    Args:
        gray_image: Grayscale image
        max_value: Maximum value for thresholded pixels
        threshold_type: OpenCV threshold type constant
        
    Returns:
        Tuple of (threshold_value, binary_image)
    """
    threshold_value, binary = cv2.threshold(
        gray_image, 0, max_value, threshold_type | cv2.THRESH_OTSU
    )
    return threshold_value, binary


def apply_sauvola_threshold(
    gray_image: np.ndarray, window_size: int, k: float, r: float
) -> np.ndarray:
    """
    Apply Sauvola's thresholding method.
    
    Args:
        gray_image: Grayscale image
        window_size: Window size for local thresholding (must be odd)
        k: Sauvola parameter k
        r: Sauvola parameter r
        
    Returns:
        Binary image
    """
    h, w = gray_image.shape
    binary = np.zeros_like(gray_image, dtype=np.uint8)
    
    half_window = window_size // 2
    
    # Pad image to handle borders
    padded = cv2.copyMakeBorder(
        gray_image,
        half_window,
        half_window,
        half_window,
        half_window,
        cv2.BORDER_REPLICATE,
    )
    
    # Convert to float for calculations
    padded_float = padded.astype(np.float32)
    
    # Calculate integral images for fast mean and std calculation
    integral = cv2.integral(padded_float)
    integral_sq = cv2.integral(padded_float ** 2)
    
    for y in range(h):
        for x in range(w):
            # Window coordinates in padded image
            y1 = y
            x1 = x
            y2 = y + window_size
            x2 = x + window_size
            
            # Calculate mean and std using integral images
            area = window_size * window_size
            sum_val = (
                integral[y2, x2]
                - integral[y1, x2]
                - integral[y2, x1]
                + integral[y1, x1]
            )
            sum_sq = (
                integral_sq[y2, x2]
                - integral_sq[y1, x2]
                - integral_sq[y2, x1]
                + integral_sq[y1, x1]
            )
            
            mean = sum_val / area
            variance = (sum_sq / area) - (mean ** 2)
            std = np.sqrt(max(0, variance))
            
            # Calculate threshold
            threshold = mean * (1 + k * (std / r - 1))
            
            # Threshold pixel
            if padded[y + half_window, x + half_window] > threshold:
                binary[y, x] = 255
            else:
                binary[y, x] = 0
    
    return binary


def calculate_sauvola_params(
    gray_image: np.ndarray, config: PreprocessingConfig
) -> tuple[int, float, float]:
    """
    Calculate Sauvola parameters from image characteristics.
    
    Args:
        gray_image: Grayscale image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (window_size, k, r)
    """
    h, w = gray_image.shape
    image_size = max(h, w)
    
    # Calculate window size based on image size
    if image_size < 500:
        window_size = config.sauvola_auto_window_min
    elif image_size < 1500:
        # Interpolate between min and max
        ratio = (image_size - 500) / (1500 - 500)
        window_size = int(
            config.sauvola_auto_window_min
            + ratio * (config.sauvola_auto_window_max - config.sauvola_auto_window_min)
        )
    else:
        window_size = config.sauvola_auto_window_max
    
    # Ensure odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Clamp to valid range
    window_size = max(
        config.sauvola_auto_window_min,
        min(config.sauvola_auto_window_max, window_size),
    )
    
    # Calculate k based on image contrast
    contrast = float(np.std(gray_image))
    max_contrast = 128.0  # Approximate max std for 8-bit image
    contrast_ratio = min(1.0, contrast / max_contrast)
    
    if contrast_ratio < 0.3:
        # Low contrast
        k = config.sauvola_auto_k_min + 0.1
    elif contrast_ratio < 0.7:
        # Medium contrast
        k = (config.sauvola_auto_k_min + config.sauvola_auto_k_max) / 2.0
    else:
        # High contrast
        k = config.sauvola_auto_k_max - 0.1
    
    k = max(config.sauvola_auto_k_min, min(config.sauvola_auto_k_max, k))
    
    # Calculate r based on image brightness
    brightness = float(np.mean(gray_image))
    
    if brightness < 85:
        # Dark image
        r = config.sauvola_auto_r_min + (config.sauvola_auto_r_max - config.sauvola_auto_r_min) * 0.5
    elif brightness < 170:
        # Medium brightness
        r = (config.sauvola_auto_r_min + config.sauvola_auto_r_max) / 2.0
    else:
        # Bright image
        r = config.sauvola_auto_r_min + (config.sauvola_auto_r_max - config.sauvola_auto_r_min) * 0.3
    
    r = max(config.sauvola_auto_r_min, min(config.sauvola_auto_r_max, r))
    
    return int(window_size), float(k), float(r)


def calculate_binarization_quality(binary_image: np.ndarray) -> float:
    """
    Calculate binarization quality metric.
    
    Args:
        binary_image: Binary image (0 and 255)
        
    Returns:
        Quality metric (0-1, higher = better)
    """
    # Calculate contrast ratio (ratio of white to black pixels)
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)
    total_pixels = white_pixels + black_pixels
    
    if total_pixels == 0:
        return 0.0
    
    # Ideal ratio is around 0.5 (equal white and black)
    ratio = white_pixels / total_pixels
    # Quality is higher when ratio is closer to 0.5
    quality = 1.0 - abs(ratio - 0.5) * 2.0
    
    return max(0.0, min(1.0, quality))



