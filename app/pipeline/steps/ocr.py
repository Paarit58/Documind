"""
OCR pipeline step.

This step delegates OCR processing to the registered engine
and stores the result in the pipeline context.
"""

import logging

from app.core.registry import get_registry
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class OCRPipelineStep(PipelineStep):
    """
    Pipeline step that performs OCR using the configured engine.
    
    This step:
    1. Retrieves the configured OCR engine from the registry
    2. Runs inference on the context's image
    3. Stores the result in the context
    
    The engine to use is determined by the config.engine field.
    """
    
    @property
    def name(self) -> str:
        """Get step name."""
        return "ocr"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute OCR on the image.
        
        Args:
            context: Pipeline context with image and config
            
        Returns:
            Context with OCR result
        """
        logger.debug(f"Running OCR step with engine: {context.config.engine}")
        
        try:
            # Get the configured engine
            registry = get_registry()
            engine = registry.get(context.config.engine)
            
            # Verify engine is loaded
            if not engine.is_loaded:
                raise RuntimeError(
                    f"Engine '{context.config.engine}' is registered but not loaded"
                )
            
            # Run inference
            result = engine.infer(context.image, context.config)
            
            # Store result
            context.result = result
            
            # Copy any errors from result to context
            if result.has_errors:
                for error in result.errors:
                    context.add_error(f"OCR: {error}")
            
            logger.debug(
                f"OCR complete: {len(result.text_blocks)} text blocks, "
                f"{result.metadata.processing_time_ms:.2f}ms"
            )
            
        except KeyError as e:
            error_msg = f"OCR engine not found: {e}"
            logger.error(error_msg)
            context.add_error(error_msg)
            
        except Exception as e:
            error_msg = f"OCR step failed: {e}"
            logger.error(error_msg, exc_info=True)
            context.add_error(error_msg)
        
        finally:
            context.add_step(self.name)
        
        return context

