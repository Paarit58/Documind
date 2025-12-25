"""
Deskew detection and correction.

This module provides functions for detecting and correcting image skew/rotation.
"""

import logging

import cv2
import numpy as np

from app.core.config import PreprocessingConfig

logger = logging.getLogger(__name__)


def detect_skew_hough(
    gray_image: np.ndarray, config: PreprocessingConfig
) -> tuple[float, float, np.ndarray | None]:
    """
    Detect skew angle using Hough Transform.
    
    Args:
        gray_image: Grayscale image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (angle_degrees, confidence, lines_visualization_image)
    """
    # Edge detection with adaptive thresholds
    # Use Otsu's method to find optimal thresholds
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(
        edges,
        rho=config.hough_rho_resolution,
        theta=config.hough_theta_resolution,
        threshold=config.hough_threshold,
    )
    
    if lines is None or len(lines) == 0:
        return 0.0, 0.0, None
    
    # Calculate angles from detected lines
    angles = []
    lines_vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    h, w = gray_image.shape
    
    for line in lines[:50]:  # Use more lines for better statistics
        rho, theta = line[0]
        
        # Convert theta (angle of normal) to line angle
        # For horizontal lines: theta ≈ 0° or 180° → line_angle ≈ 90°
        # For vertical lines: theta ≈ 90° → line_angle ≈ 0°
        # We want horizontal text lines, so we need theta ≈ 0° or 180°
        
        # Normalize theta to [0, π]
        if theta < 0:
            theta += np.pi
        if theta > np.pi:
            theta -= np.pi
        
        # Convert to degrees
        theta_deg = np.degrees(theta)
        
        # For text lines, theta should be close to 0° or 180° (horizontal)
        # If theta is close to 90°, it's a vertical line (not what we want)
        # Map to skew angle: if theta is near 0° or 180°, skew is 0°
        # If theta is near 90°, we don't want it
        
        # Filter out near-vertical lines (theta between 70° and 110°)
        if 70 <= theta_deg <= 110:
            continue  # Skip vertical lines
        
        # Calculate actual line angle from normal
        # Line angle = theta - 90° (for horizontal text)
        line_angle = theta_deg - 90.0
        
        # Normalize to [-45, 45] range
        if line_angle > 45:
            line_angle -= 90
        elif line_angle < -45:
            line_angle += 90
        
        angles.append(line_angle)
        
        # Draw line for visualization
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    if not angles:
        return 0.0, 0.0, None
    
    # Filter outliers using IQR (Interquartile Range)
    angles_array = np.array(angles)
    q1 = np.percentile(angles_array, 25)
    q3 = np.percentile(angles_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_angles = angles_array[(angles_array >= lower_bound) & (angles_array <= upper_bound)]
    
    if len(filtered_angles) == 0:
        filtered_angles = angles_array  # Fallback to all angles
    
    # Use median for robustness
    dominant_angle = float(np.median(filtered_angles))
    
    # Validate angle is reasonable (within ±10 degrees)
    if abs(dominant_angle) > 10.0:
        # If angle is too large, it's likely wrong - return 0
        logger.warning(f"Detected angle {dominant_angle:.2f}° seems too large, rejecting")
        return 0.0, 0.0, None
    
    # Calculate confidence based on consistency and number of lines
    angle_std = float(np.std(filtered_angles))
    consistency_score = max(0.0, min(1.0, 1.0 - (angle_std / 5.0)))  # Tighter std threshold
    line_count_score = min(1.0, len(filtered_angles) / 10.0)  # More lines = higher confidence
    confidence = 0.6 * consistency_score + 0.4 * line_count_score
    
    return dominant_angle, confidence, lines_vis


def detect_skew_text(
    gray_image: np.ndarray, config: PreprocessingConfig
) -> tuple[float, float]:
    """
    Detect skew angle using text orientation (projection profiles).
    
    Args:
        gray_image: Grayscale image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (angle_degrees, confidence)
    """
    h, w = gray_image.shape
    
    # Pre-filter: if image is too small or has low contrast, skip
    if min(h, w) < 100:
        return 0.0, 0.0
    
    # Check if image has enough text content
    # Use Otsu threshold to estimate text content
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_ratio = np.sum(binary > 0) / (h * w)
    
    if text_ratio < 0.05:  # Less than 5% text content
        return 0.0, 0.0
    
    # Try angles from -10 to 10 degrees (narrower range for better accuracy)
    # Use smaller step for fine-tuning
    angles_to_try = np.arange(-10, 11, max(1, config.text_projection_step // 2))
    best_angle = 0.0
    best_variance = 0.0
    variance_at_zero = 0.0
    
    center = (w // 2, h // 2)
    
    # First, calculate variance at 0° for comparison
    projection_zero = np.sum(gray_image, axis=1)
    variance_at_zero = float(np.var(projection_zero))
    best_variance = variance_at_zero
    
    for angle in angles_to_try:
        if angle == 0:
            continue  # Already calculated
        
        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_image, M, (w, h), borderValue=255)
        
        # Calculate horizontal projection profile
        projection = np.sum(rotated, axis=1)
        
        # Calculate variance (higher variance = more text lines = better alignment)
        variance = float(np.var(projection))
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    # Validate: if best angle doesn't improve variance significantly, reject it
    variance_improvement = (best_variance - variance_at_zero) / variance_at_zero if variance_at_zero > 0 else 0
    
    # Require at least 5% improvement to consider the angle valid
    if variance_improvement < 0.05:
        best_angle = 0.0
        best_variance = variance_at_zero
    
    # Validate angle is reasonable
    if abs(best_angle) > 10.0:
        logger.warning(f"Text projection detected angle {best_angle:.2f}° seems too large, rejecting")
        return 0.0, 0.0
    
    # Calculate confidence based on variance improvement
    # Normalize confidence (0-1 scale)
    max_possible_variance = h * 255 * 255 / 12  # Theoretical max variance
    base_confidence = min(1.0, best_variance / max_possible_variance) if max_possible_variance > 0 else 0.0
    
    # Boost confidence if variance improvement is significant
    improvement_boost = min(1.0, variance_improvement * 2.0)  # Scale improvement
    confidence = 0.7 * base_confidence + 0.3 * improvement_boost
    
    return best_angle, confidence


def apply_rotation(cv_image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Apply rotation to image.
    
    Args:
        cv_image: OpenCV image
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image
    """
    h, w = cv_image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    rotated = cv2.warpAffine(
        cv_image, M, (new_w, new_h), borderValue=(255, 255, 255)
    )
    
    return rotated

