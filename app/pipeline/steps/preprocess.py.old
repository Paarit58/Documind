"""
Global normalization preprocessing pipeline step.

This step implements a multi-layer preprocessing pipeline:
- Layer 1: Color & Lighting (Grayscale + CLAHE)
- Layer 2: Geometric (Deskew + Border removal)
- Layer 3: Signal-to-Noise (Median blur + Bilateral filtering)
- Layer 4: Binarization (Otsu + Sauvola adaptive thresholding)

Each layer produces intermediate outputs that are stored for UI display.
"""

import base64
import io
import logging

import cv2
import numpy as np
from PIL import Image

from app.core.config import OCRConfig, PreprocessingConfig
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class PreprocessPipelineStep(PipelineStep):
    """
    Pipeline step that performs global normalization preprocessing.
    
    This step processes images through multiple layers:
    1. Color & Lighting: Grayscale conversion + CLAHE
    2. Geometric: Deskew detection/correction + Border detection/padding
    3. Signal-to-Noise: Median blur + Bilateral filtering for noise reduction
    4. Binarization: Otsu's + Sauvola's adaptive thresholding for binary conversion
    
    Each phase produces intermediate outputs stored in context for UI display.
    """
    
    @property
    def name(self) -> str:
        """Get step name."""
        return "preprocess"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute preprocessing on the image.
        
        Args:
            context: Pipeline context with image and config
            
        Returns:
            Context with preprocessed image and phase outputs
        """
        logger.debug("Running preprocessing step")
        
        # Get preprocessing config (use defaults if None)
        preprocess_config = self._get_preprocessing_config(context.config)
        
        # Store original image
        original_image = context.image.copy()
        context.set_intermediate("original_image", original_image)
        
        try:
            # Work on a copy to avoid modifying the original
            processed_image = context.image.copy()
            
            # Convert PIL to OpenCV format (BGR)
            cv_image = self._pil_to_cv2(processed_image)
            
            # Initialize phase outputs dictionary
            phase_outputs = {}
            
            # Layer 1: Color & Lighting
            if preprocess_config.enable_color_lighting:
                cv_image, layer1_outputs = self._apply_color_lighting_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer1_color_lighting"] = layer1_outputs
                logger.debug("Applied Color & Lighting layer")
            
            # Layer 2: Geometric (Deskew + Border detection)
            if preprocess_config.enable_geometric_layer:
                cv_image, layer2_outputs = self._apply_geometric_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer2_geometric"] = layer2_outputs
                logger.debug("Applied Geometric layer")
            
            # Layer 3: Signal-to-Noise (Median blur + Bilateral filter)
            if preprocess_config.enable_signal_noise_layer:
                cv_image, layer3_outputs = self._apply_signal_noise_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer3_signal_noise"] = layer3_outputs
                logger.debug("Applied Signal-to-Noise layer")
            
            # Layer 4: Binarization (Otsu + Sauvola adaptive thresholding)
            if preprocess_config.enable_binarization_layer:
                cv_image, layer4_outputs = self._apply_binarization_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer4_binarization"] = layer4_outputs
                logger.debug("Applied Binarization layer")
            
            # Convert back to PIL (RGB)
            processed_image = self._cv2_to_pil(cv_image)
            
            # Update context with preprocessed image
            context.image = processed_image
            
            # Store phase outputs in context for extraction
            context.set_intermediate("preprocess_phase_outputs", phase_outputs)
            
            logger.debug("Preprocessing complete")
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {e}"
            logger.warning(error_msg, exc_info=True)
            context.add_error(error_msg)
            # Continue with original image if preprocessing fails
        
        finally:
            context.add_step(self.name)
        
        return context
    
    def _get_preprocessing_config(
        self, config: OCRConfig
    ) -> PreprocessingConfig:
        """Get preprocessing config, using defaults if None."""
        if config.preprocessing is None:
            return PreprocessingConfig()
        return config.preprocessing
    
    def _apply_color_lighting_layer(
        self,
        cv_image: np.ndarray,
        config: PreprocessingConfig,
        original_image: Image.Image,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply Layer 1: Color & Lighting normalization.
        
        This layer:
        1. Converts image to grayscale
        2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Applies light bilateral filtering to reduce noise (optional)
        
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
        gray_base64 = self._image_to_base64(gray_pil)
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
        
        # Store CLAHE output (before denoising)
        clahe_pil = Image.fromarray(clahe_output)
        clahe_base64 = self._image_to_base64(clahe_pil)
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
    
    def _apply_geometric_layer(
        self,
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
            original_before_deskew_pil = self._cv2_to_pil(original_before_deskew)
            deskew_outputs["original_image"] = self._image_to_base64(original_before_deskew_pil)
            
            # Detect skew angle
            detected_angle = 0.0
            method_used = "none"
            hough_result = None
            text_result = None
            
            if config.deskew_method in ["hough", "auto"]:
                hough_angle, hough_confidence, hough_lines_img = self._detect_skew_hough(
                    gray, config
                )
                hough_result = {
                    "angle": float(hough_angle),
                    "confidence": float(hough_confidence),
                    "lines_visualization": self._image_to_base64(
                        Image.fromarray(hough_lines_img)
                    ) if hough_lines_img is not None else None,
                }
            
            if config.deskew_method in ["text", "auto"]:
                text_angle, text_confidence = self._detect_skew_text(gray, config)
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
                deskewed_image = self._apply_rotation(cv_image, detected_angle)
                rotation_applied = True
                logger.debug(f"Applied rotation: {detected_angle:.2f}° using {method_used}")
            else:
                logger.debug(f"Skew angle {detected_angle:.2f}° below threshold, skipping rotation")
            
            # Store deskew outputs
            deskewed_pil = self._cv2_to_pil(deskewed_image)
            deskew_outputs["detected_angle"] = float(detected_angle)
            deskew_outputs["method_used"] = method_used
            deskew_outputs["rotation_applied"] = rotation_applied
            deskew_outputs["rotated_image"] = self._image_to_base64(deskewed_pil)
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
            bounding_box = self._detect_content_borders(deskewed_image, config)
            border_outputs["bounding_box"] = bounding_box
            
            # Visualize bounding box
            bbox_vis = deskewed_image.copy()
            if bounding_box:
                x, y, w, h = bounding_box
                cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bbox_vis_pil = self._cv2_to_pil(bbox_vis)
            border_outputs["bounding_box_visualization"] = self._image_to_base64(bbox_vis_pil)
            
            # Crop to content
            if bounding_box:
                cropped_image = self._crop_to_content(deskewed_image, bounding_box)
                cropped_pil = self._cv2_to_pil(cropped_image)
                border_outputs["cropped_image"] = self._image_to_base64(cropped_pil)
                border_outputs["cropped_size"] = cropped_image.shape[:2][::-1]
                
                # Add padding
                padded_image, padding_size = self._add_padding(cropped_image, config)
                final_image = padded_image
                padded_pil = self._cv2_to_pil(padded_image)
                border_outputs["padded_image"] = self._image_to_base64(padded_pil)
                border_outputs["padding_size"] = int(padding_size)
                border_outputs["final_size"] = padded_image.shape[:2][::-1]
            else:
                # No content detected, just add padding to whole image
                padded_image, padding_size = self._add_padding(deskewed_image, config)
                final_image = padded_image
                padded_pil = self._cv2_to_pil(padded_image)
                border_outputs["padded_image"] = self._image_to_base64(padded_pil)
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
    
    def _detect_skew_hough(
        self, gray_image: np.ndarray, config: PreprocessingConfig
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
    
    def _detect_skew_text(
        self, gray_image: np.ndarray, config: PreprocessingConfig
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
    
    def _apply_rotation(self, cv_image: np.ndarray, angle_degrees: float) -> np.ndarray:
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
    
    def _detect_content_borders(
        self, cv_image: np.ndarray, config: PreprocessingConfig
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
    
    def _crop_to_content(
        self, cv_image: np.ndarray, bounding_box: tuple[int, int, int, int]
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
    
    def _add_padding(
        self, cv_image: np.ndarray, config: PreprocessingConfig
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
    
    def _apply_signal_noise_layer(
        self,
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
            noise_level, noise_score = self._detect_noise_level(gray, config)
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
                kernel_size = self._calculate_median_blur_params(noise_score, config)
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Store original before filtering
            original_before_median = cv_image.copy()
            original_before_median_pil = self._cv2_to_pil(original_before_median)
            
            # Apply median blur
            median_filtered = self._apply_median_blur(gray, kernel_size)
            
            # Convert back to BGR if needed
            if len(cv_image.shape) == 3:
                median_filtered_bgr = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
            else:
                median_filtered_bgr = median_filtered
            
            # Calculate noise reduction metric
            noise_reduction_metric = self._calculate_noise_reduction_metric(
                gray, median_filtered
            )
            
            median_filtered_pil = self._cv2_to_pil(median_filtered_bgr)
            median_outputs = {
                "enabled": True,
                "kernel_size": int(kernel_size),
                "original_image": self._image_to_base64(original_before_median_pil),
                "filtered_image": self._image_to_base64(median_filtered_pil),
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
                d, sigma_color, sigma_space = self._calculate_bilateral_params(
                    noise_score, config
                )
            
            # Ensure d is odd
            if d % 2 == 0:
                d += 1
            
            # Store original before filtering
            original_before_bilateral = cv_image.copy()
            original_before_bilateral_pil = self._cv2_to_pil(original_before_bilateral)
            
            # Apply bilateral filter
            bilateral_filtered = self._apply_bilateral_filter(
                cv_image, d, sigma_color, sigma_space
            )
            
            # Calculate noise reduction metric
            if len(cv_image.shape) == 3:
                gray_before = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                gray_after = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY)
            else:
                gray_before = cv_image.copy()
                gray_after = bilateral_filtered.copy()
            
            noise_reduction_metric = self._calculate_noise_reduction_metric(
                gray_before, gray_after
            )
            
            bilateral_filtered_pil = self._cv2_to_pil(bilateral_filtered)
            bilateral_outputs = {
                "enabled": True,
                "d": int(d),
                "sigma_color": float(sigma_color),
                "sigma_space": float(sigma_space),
                "original_image": self._image_to_base64(original_before_bilateral_pil),
                "filtered_image": self._image_to_base64(bilateral_filtered_pil),
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
    
    def _detect_noise_level(
        self, gray_image: np.ndarray, config: PreprocessingConfig
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
    
    def _calculate_median_blur_params(
        self, noise_score: float, config: PreprocessingConfig
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
    
    def _calculate_bilateral_params(
        self, noise_score: float, config: PreprocessingConfig
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
    
    def _apply_median_blur(
        self, gray_image: np.ndarray, kernel_size: int
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
    
    def _apply_bilateral_filter(
        self,
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
    
    def _calculate_noise_reduction_metric(
        self, before: np.ndarray, after: np.ndarray
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
    
    def _apply_binarization_layer(
        self,
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
            "image": self._image_to_base64(gray_pil),
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
            threshold_value, otsu_binary = self._apply_otsu_threshold(
                gray, config.otsu_max_value, threshold_type
            )
            
            # Calculate quality metric
            quality_metric = self._calculate_binarization_quality(otsu_binary)
            
            # Store original grayscale
            original_before_otsu_pil = Image.fromarray(gray)
            
            otsu_binary_pil = Image.fromarray(otsu_binary)
            otsu_outputs = {
                "enabled": True,
                "threshold_value": float(threshold_value),
                "threshold_type": config.otsu_threshold_type,
                "original_image": self._image_to_base64(original_before_otsu_pil),
                "binary_image": self._image_to_base64(otsu_binary_pil),
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
                window_size, k, r = self._calculate_sauvola_params(gray, config)
            
            # Ensure window size is odd
            if window_size % 2 == 0:
                window_size += 1
            
            # Apply Sauvola's thresholding
            sauvola_binary = self._apply_sauvola_threshold(gray, window_size, k, r)
            
            # Calculate quality metric
            quality_metric = self._calculate_binarization_quality(sauvola_binary)
            
            # Store original grayscale
            original_before_sauvola_pil = Image.fromarray(gray)
            
            sauvola_binary_pil = Image.fromarray(sauvola_binary)
            sauvola_outputs = {
                "enabled": True,
                "window_size": int(window_size),
                "k": float(k),
                "r": float(r),
                "original_image": self._image_to_base64(original_before_sauvola_pil),
                "binary_image": self._image_to_base64(sauvola_binary_pil),
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
    
    def _apply_otsu_threshold(
        self, gray_image: np.ndarray, max_value: int, threshold_type: int
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
    
    def _apply_sauvola_threshold(
        self, gray_image: np.ndarray, window_size: int, k: float, r: float
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
    
    def _calculate_sauvola_params(
        self, gray_image: np.ndarray, config: PreprocessingConfig
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
    
    def _calculate_binarization_quality(self, binary_image: np.ndarray) -> float:
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
    
    def _image_to_base64(self, pil_image: Image.Image) -> str:
        """
        Convert PIL Image to base64 data URL.
        
        Args:
            pil_image: PIL Image
            
        Returns:
            Base64 data URL string
        """
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
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
    
    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
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
