"""
Global normalization preprocessing pipeline step.

This step implements a simplified preprocessing pipeline:
1. Grayscale conversion (always)
2. Deskew (optional)
3. Binarization (Sauvola method)
4. Comb field removal (exact working script logic)
"""

import logging

import cv2
import numpy as np

from app.core.config import OCRConfig, PreprocessingConfig
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep
from app.pipeline.steps.preprocess.layers import (
    apply_comb_field_removal,
    apply_deskew,
    apply_sauvola_binarization,
)
from app.pipeline.steps.preprocess.utils import cv2_to_pil, pil_to_cv2

logger = logging.getLogger(__name__)


class PreprocessPipelineStep(PipelineStep):
    """
    Pipeline step that performs global normalization preprocessing.
    
    This step processes images through a simplified pipeline:
    1. Grayscale conversion (always)
    2. Deskew detection/correction (optional)
    3. Binarization (Sauvola's adaptive thresholding)
    4. Comb field removal (exact working script logic)
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
            
            # Step 1: Grayscale conversion (always)
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            logger.debug("Applied grayscale conversion")
            
            # Step 2: Deskew (optional)
            if preprocess_config.enable_geometric_layer:
                cv_image, deskew_outputs = apply_deskew(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["deskew"] = deskew_outputs
                logger.debug("Applied deskew")
            
            # Step 3: Binarization (Sauvola only)
            # Save original before binarization for inpainting
            original_before_binarization = cv_image.copy()
            
            if preprocess_config.enable_binarization_layer:
                cv_image, binarization_outputs = apply_sauvola_binarization(
                    cv_image, preprocess_config, original_image
                )
                phase_outputs["binarization"] = binarization_outputs
                logger.debug("Applied Sauvola binarization")
            
            # Step 4: Comb field removal
            if preprocess_config.enable_comb_field_removal:
                cv_image, comb_removal_outputs = apply_comb_field_removal(
                    cv_image, preprocess_config, original_image, original_before_binarization
                )
                phase_outputs["comb_field_removal"] = comb_removal_outputs
                logger.debug("Applied comb field removal")
            
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

