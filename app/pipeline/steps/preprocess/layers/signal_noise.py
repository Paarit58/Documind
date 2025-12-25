"""
Layer 3: Signal-to-Noise normalization.

This layer handles:
- Noise detection (auto or manual)
- Median blur filtering
- Bilateral filtering
"""

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.utils import cv2_to_pil, image_to_base64


def apply_signal_noise_layer(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
) -> tuple[np.ndarray, dict]:
    """
    Apply Layer 3: Signal-to-Noise normalization.
    
    This layer:
    1. Detects noise level (auto or manual)
    2. Applies median blur filtering
    3. Applies bilateral filtering
    4. Shows both outputs for comparison
    
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
    
    # Phase 3.1: Noise Detection
    noise_detection_outputs = {}
    noise_level = 0.0
    noise_score = 0.0
    auto_detected = False
    
    if config.enable_noise_auto_detection and config.noise_threshold is None:
        # Auto-detect noise level
        noise_level, noise_score = detect_noise_level(gray, config)
        auto_detected = True
        noise_detection_outputs["detection_method"] = config.noise_detection_method
    else:
        # Use manual threshold
        noise_level = config.noise_threshold if config.noise_threshold is not None else 0.0
        noise_score = noise_level  # Use threshold as score
        auto_detected = False
        noise_detection_outputs["detection_method"] = "manual"
    
    noise_detection_outputs["noise_level"] = float(noise_level)
    noise_detection_outputs["noise_score"] = float(noise_score)
    noise_detection_outputs["auto_detected"] = auto_detected
    
    phase_outputs["noise_detection"] = noise_detection_outputs
    
    # Phase 3.2: Noise Reduction
    final_image = cv_image.copy()
    method_used = "none"
    
    # Median Blur
    median_outputs = {"enabled": False}
    median_filtered_bgr = None
    if config.enable_median_blur:
        # Calculate kernel size
        if config.median_blur_kernel_size is not None:
            kernel_size = config.median_blur_kernel_size
        else:
            kernel_size = calculate_median_blur_params(noise_score, config)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Store original before filtering
        original_before_median = cv_image.copy()
        original_before_median_pil = cv2_to_pil(original_before_median)
        
        # Apply median blur
        median_filtered = apply_median_blur(gray, kernel_size)
        
        # Convert back to BGR if needed
        if len(cv_image.shape) == 3:
            median_filtered_bgr = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
        else:
            median_filtered_bgr = median_filtered
        
        # Calculate noise reduction metric
        noise_reduction_metric = calculate_noise_reduction_metric(
            gray, median_filtered
        )
        
        median_filtered_pil = cv2_to_pil(median_filtered_bgr)
        median_outputs = {
            "enabled": True,
            "kernel_size": int(kernel_size),
            "original_image": image_to_base64(original_before_median_pil),
            "filtered_image": image_to_base64(median_filtered_pil),
            "noise_reduction_metric": float(noise_reduction_metric),
        }
    
    phase_outputs["median_blur"] = median_outputs
    
    # Bilateral Filter
    bilateral_outputs = {"enabled": False}
    if config.enable_bilateral_filter:
        # Calculate parameters
        if (
            config.bilateral_d is not None
            and config.bilateral_sigma_color is not None
            and config.bilateral_sigma_space is not None
        ):
            d = config.bilateral_d
            sigma_color = config.bilateral_sigma_color
            sigma_space = config.bilateral_sigma_space
        else:
            d, sigma_color, sigma_space = calculate_bilateral_params(
                noise_score, config
            )
        
        # Ensure d is odd
        if d % 2 == 0:
            d += 1
        
        # Store original before filtering
        original_before_bilateral = cv_image.copy()
        original_before_bilateral_pil = cv2_to_pil(original_before_bilateral)
        
        # Apply bilateral filter
        bilateral_filtered = apply_bilateral_filter(
            cv_image, d, sigma_color, sigma_space
        )
        
        # Calculate noise reduction metric
        if len(cv_image.shape) == 3:
            gray_before = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY)
        else:
            gray_before = cv_image.copy()
            gray_after = bilateral_filtered.copy()
        
        noise_reduction_metric = calculate_noise_reduction_metric(
            gray_before, gray_after
        )
        
        bilateral_filtered_pil = cv2_to_pil(bilateral_filtered)
        bilateral_outputs = {
            "enabled": True,
            "d": int(d),
            "sigma_color": float(sigma_color),
            "sigma_space": float(sigma_space),
            "original_image": image_to_base64(original_before_bilateral_pil),
            "filtered_image": image_to_base64(bilateral_filtered_pil),
            "noise_reduction_metric": float(noise_reduction_metric),
        }
    
    phase_outputs["bilateral_filter"] = bilateral_outputs
    
    # Select final output based on config
    if config.signal_noise_output_method == "median" and median_outputs["enabled"] and median_filtered_bgr is not None:
        final_image = median_filtered_bgr
        method_used = "median"
    elif (
        config.signal_noise_output_method == "bilateral"
        and bilateral_outputs["enabled"]
    ):
        final_image = bilateral_filtered
        method_used = "bilateral"
    elif config.signal_noise_output_method == "both":
        # Use median as default if both are enabled
        if median_outputs["enabled"] and median_filtered_bgr is not None:
            final_image = median_filtered_bgr
            method_used = "median"
        elif bilateral_outputs["enabled"]:
            final_image = bilateral_filtered
            method_used = "bilateral"
    else:
        method_used = "none"
    
    # Metadata
    phase_outputs["metadata"] = {
        "original_size": original_size,
        "final_size": final_image.shape[:2][::-1],
        "method_used": method_used,
        "noise_reduced": method_used != "none",
    }
    
    return final_image, phase_outputs


def detect_noise_level(
    gray_image: np.ndarray, config: PreprocessingConfig
) -> tuple[float, float]:
    """
    Auto-detect noise level in image.
    
    Args:
        gray_image: Grayscale image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (noise_level, noise_score)
    """
    h, w = gray_image.shape
    
    variance_score = 0.0
    gradient_score = 0.0
    
    if config.noise_detection_method in ["variance", "both"]:
        # Divide image into patches and calculate local variance
        patch_size = min(16, min(h, w) // 4)  # Adaptive patch size
        if patch_size < 4:
            patch_size = 4
        
        variances = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray_image[y : y + patch_size, x : x + patch_size]
                variances.append(float(np.var(patch)))
        
        if variances:
            # Normalize variance score (0-1)
            max_variance = 255 * 255 / 12  # Theoretical max variance
            variance_score = min(1.0, np.mean(variances) / max_variance)
    
    if config.noise_detection_method in ["gradient", "both"]:
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate variance of gradient magnitude
        grad_variance = float(np.var(gradient_magnitude))
        
        # Normalize gradient score (0-1)
        max_grad_variance = (
            (255 * np.sqrt(2)) ** 2 / 12
        )  # Approximate max gradient variance
        gradient_score = min(1.0, grad_variance / max_grad_variance)
    
    # Combine scores
    if config.noise_detection_method == "both":
        noise_score = 0.6 * variance_score + 0.4 * gradient_score
    elif config.noise_detection_method == "variance":
        noise_score = variance_score
    else:  # gradient
        noise_score = gradient_score
    
    # Map score to noise level (0-1 scale)
    noise_level = noise_score
    
    return noise_level, noise_score


def calculate_median_blur_params(
    noise_score: float, config: PreprocessingConfig
) -> int:
    """
    Calculate median blur kernel size from noise score.
    
    Args:
        noise_score: Noise score (0-1)
        config: Preprocessing configuration
        
    Returns:
        Kernel size (odd number)
    """
    # Map noise score to kernel size
    if noise_score < 0.3:
        # Low noise
        kernel_size = config.median_blur_auto_min
    elif noise_score < 0.6:
        # Medium noise
        mid = (config.median_blur_auto_min + config.median_blur_auto_max) // 2
        # Ensure odd
        kernel_size = mid if mid % 2 == 1 else mid + 1
    else:
        # High noise
        kernel_size = config.median_blur_auto_max
    
    # Clamp to valid range
    kernel_size = max(config.median_blur_auto_min, min(config.median_blur_auto_max, kernel_size))
    
    # Ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return kernel_size


def calculate_bilateral_params(
    noise_score: float, config: PreprocessingConfig
) -> tuple[int, float, float]:
    """
    Calculate bilateral filter parameters from noise score.
    
    Args:
        noise_score: Noise score (0-1)
        config: Preprocessing configuration
        
    Returns:
        Tuple of (d, sigma_color, sigma_space)
    """
    # Map noise score to parameters
    if noise_score < 0.3:
        # Low noise
        d = config.bilateral_auto_d_min
        sigma_color = config.bilateral_auto_sigma_color_min
        sigma_space = config.bilateral_auto_sigma_space_min
    elif noise_score < 0.6:
        # Medium noise
        d = (config.bilateral_auto_d_min + config.bilateral_auto_d_max) // 2
        sigma_color = (
            config.bilateral_auto_sigma_color_min
            + config.bilateral_auto_sigma_color_max
        ) / 2.0
        sigma_space = (
            config.bilateral_auto_sigma_space_min
            + config.bilateral_auto_sigma_space_max
        ) / 2.0
    else:
        # High noise
        d = config.bilateral_auto_d_max
        sigma_color = config.bilateral_auto_sigma_color_max
        sigma_space = config.bilateral_auto_sigma_space_max
    
    # Ensure d is odd
    if d % 2 == 0:
        d += 1
    
    # Clamp to valid ranges
    d = max(config.bilateral_auto_d_min, min(config.bilateral_auto_d_max, d))
    sigma_color = max(
        config.bilateral_auto_sigma_color_min,
        min(config.bilateral_auto_sigma_color_max, sigma_color),
    )
    sigma_space = max(
        config.bilateral_auto_sigma_space_min,
        min(config.bilateral_auto_sigma_space_max, sigma_space),
    )
    
    return int(d), float(sigma_color), float(sigma_space)


def apply_median_blur(
    gray_image: np.ndarray, kernel_size: int
) -> np.ndarray:
    """
    Apply median blur filtering.
    
    Args:
        gray_image: Grayscale image
        kernel_size: Kernel size (must be odd)
        
    Returns:
        Filtered image
    """
    return cv2.medianBlur(gray_image, kernel_size)


def apply_bilateral_filter(
    cv_image: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    """
    Apply bilateral filtering.
    
    Args:
        cv_image: OpenCV image (BGR or grayscale)
        d: Filter diameter (must be odd)
        sigma_color: Color space sigma
        sigma_space: Coordinate space sigma
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(cv_image, d, sigma_color, sigma_space)


def calculate_noise_reduction_metric(
    before: np.ndarray, after: np.ndarray
) -> float:
    """
    Calculate noise reduction effectiveness metric.
    
    Args:
        before: Image before filtering
        after: Image after filtering
        
    Returns:
        Noise reduction metric (0-1, higher = more reduction)
    """
    # Calculate variance reduction
    var_before = float(np.var(before))
    var_after = float(np.var(after))
    
    if var_before == 0:
        return 0.0
    
    # Reduction percentage
    reduction = (var_before - var_after) / var_before
    
    # Normalize to 0-1 scale
    return max(0.0, min(1.0, reduction))

