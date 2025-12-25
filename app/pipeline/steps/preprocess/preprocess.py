"""
Global normalization preprocessing pipeline step.

This step implements a multi-layer preprocessing pipeline:
- Layer 1: Color & Lighting (Grayscale + CLAHE)
- Layer 2: Geometric (Deskew + Border removal)
- Layer 3: Signal-to-Noise (Median blur + Bilateral filtering)
- Layer 4: Binarization (Otsu + Sauvola adaptive thresholding)
- Layer 5: Structural Analysis (Form line detection - Phase 1: Kernel Definition)

Each layer produces intermediate outputs that are stored for UI display.
"""

import logging

import cv2
import numpy as np

from app.core.config import OCRConfig, PreprocessingConfig
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep
from app.pipeline.steps.preprocess.layers import (
    apply_binarization_layer,
    apply_color_lighting_layer,
    apply_geometric_layer,
    apply_signal_noise_layer,
    define_kernels,
    extract_morphological_masks,
    refine_grid_masks,
    generate_intelligent_mask,
    generate_final_clean_zone_mask,
)
from app.pipeline.steps.preprocess.utils import cv2_to_pil, pil_to_cv2

logger = logging.getLogger(__name__)


class PreprocessPipelineStep(PipelineStep):
    """
    Pipeline step that performs global normalization preprocessing.
    
    This step processes images through multiple layers:
    1. Color & Lighting: Grayscale conversion + CLAHE
    2. Geometric: Deskew detection/correction + Border detection/padding
    3. Signal-to-Noise: Median blur + Bilateral filtering for noise reduction
    4. Binarization: Otsu's + Sauvola's adaptive thresholding for binary conversion
    5. Structural Analysis: Form line detection (Phase 1: Kernel Definition)
    
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
            cv_image = pil_to_cv2(processed_image)
            
            # Initialize phase outputs dictionary
            phase_outputs = {}
            
            # Layer 1: Color & Lighting
            if preprocess_config.enable_color_lighting:
                cv_image, layer1_outputs = apply_color_lighting_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer1_color_lighting"] = layer1_outputs
                logger.debug("Applied Color & Lighting layer")
            
            # Layer 2: Geometric (Deskew + Border detection)
            if preprocess_config.enable_geometric_layer:
                cv_image, layer2_outputs = apply_geometric_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer2_geometric"] = layer2_outputs
                logger.debug("Applied Geometric layer")
            
            # Layer 3: Signal-to-Noise (Median blur + Bilateral filter)
            if preprocess_config.enable_signal_noise_layer:
                cv_image, layer3_outputs = apply_signal_noise_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer3_signal_noise"] = layer3_outputs
                logger.debug("Applied Signal-to-Noise layer")
            
            # Layer 4: Binarization (Otsu + Sauvola adaptive thresholding)
            if preprocess_config.enable_binarization_layer:
                cv_image, layer4_outputs = apply_binarization_layer(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["layer4_binarization"] = layer4_outputs
                logger.debug("Applied Binarization layer")
            
            # Layer 5: Structural Analysis (Phase 1: Kernel Definition)
            if preprocess_config.enable_structural_analysis:
                # Phase 1: Define kernels
                h_kernel, v_kernel, phase1_outputs = define_kernels(
                    cv_image, preprocess_config
                )
                phase_outputs["layer5_structural_analysis"] = {
                    "phase1_kernel_definition": phase1_outputs
                }
                logger.debug("Applied Structural Analysis Phase 1: Kernel Definition")
                # Note: Kernels will be stored in context for use in later phases
                context.set_intermediate("structural_horizontal_kernel", h_kernel)
                context.set_intermediate("structural_vertical_kernel", v_kernel)
                
                # Phase 2: Morphological Extraction
                h_mask, v_mask, phase2_outputs = extract_morphological_masks(
                    cv_image, h_kernel, v_kernel, preprocess_config
                )
                phase_outputs["layer5_structural_analysis"]["phase2_morphological_extraction"] = phase2_outputs
                logger.debug("Applied Structural Analysis Phase 2: Morphological Extraction")
                # Store masks in context for Phase 3 (Grid Intersection)
                context.set_intermediate("structural_horizontal_mask", h_mask)
                context.set_intermediate("structural_vertical_mask", v_mask)
                
                # Phase 3: Grid Intersection & Refinement
                refined_h_mask, refined_v_mask, phase3_outputs = refine_grid_masks(
                    h_mask, v_mask, cv_image, preprocess_config
                )
                phase_outputs["layer5_structural_analysis"]["phase3_grid_refinement"] = phase3_outputs
                logger.debug("Applied Structural Analysis Phase 3: Grid Intersection & Refinement")
                context.set_intermediate("structural_refined_horizontal_mask", refined_h_mask)
                context.set_intermediate("structural_refined_vertical_mask", refined_v_mask)
                
                # Extract detected boxes and comb fields from Phase 3 for visualization/statistics
                detected_boxes = phase3_outputs.get("detected_boxes", {}).get("boxes", []) if phase3_outputs.get("detected_boxes") else []
                comb_fields = phase3_outputs.get("comb_fields", {}).get("fields", []) if phase3_outputs.get("comb_fields") else []
                context.set_intermediate("structural_detected_boxes", detected_boxes)
                context.set_intermediate("structural_comb_fields", comb_fields)
                
                # Phase 4: Intelligent Mask Generation with Text Preservation
                initial_mask, cleaned_image, text_exclusion_mask, phase4_outputs = generate_intelligent_mask(
                    refined_h_mask, refined_v_mask, cv_image, preprocess_config
                )
                phase_outputs["layer5_structural_analysis"]["phase4_intelligent_mask"] = phase4_outputs
                logger.debug("Applied Structural Analysis Phase 4: Intelligent Mask Generation")
                
                # Phase 5: Final Mask Generation (Clean Zone Map)
                clean_zone_mask, phase5_outputs = generate_final_clean_zone_mask(
                    refined_h_mask, refined_v_mask, text_exclusion_mask, cv_image, preprocess_config,
                    detected_boxes=detected_boxes, comb_fields=comb_fields
                )
                phase_outputs["layer5_structural_analysis"]["phase5_final_mask_generation"] = phase5_outputs
                logger.debug("Applied Structural Analysis Phase 5: Final Mask Generation (Clean Zone Map)")
                
                # Apply final cleaning using the dilated clean zone mask
                # Determine background color
                if len(cv_image.shape) == 3:
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = cv_image.copy()
                unique_values, counts = np.unique(gray, return_counts=True)
                background_color = unique_values[np.argmax(counts)]
                
                # Create final cleaned image using clean zone mask
                cleaned_gray = gray.copy()
                cleaned_gray[clean_zone_mask > 0] = background_color
                
                # Convert back to original format
                if len(cv_image.shape) == 3:
                    cv_image = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)
                else:
                    cv_image = cleaned_gray
                
                # Store masks in context
                context.set_intermediate("structural_initial_mask", initial_mask)
                context.set_intermediate("structural_text_exclusion_mask", text_exclusion_mask)
                context.set_intermediate("structural_clean_zone_mask", clean_zone_mask)
                context.set_intermediate("structural_final_mask", clean_zone_mask)  # Keep for backward compatibility
            
            # Convert back to PIL (RGB)
            processed_image = cv2_to_pil(cv_image)
            
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

