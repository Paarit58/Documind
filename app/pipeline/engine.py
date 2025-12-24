"""
Pipeline engine for orchestrating processing steps.

The pipeline engine manages the execution of pipeline steps,
handling step ordering, error handling, and context flow.
"""

import base64
import io
import logging
import time
from typing import Any, Sequence

from PIL import Image

from app.core.config import OCRConfig
from app.core.types import OCRMetadata, OCRResult
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep
from app.pipeline.steps.ocr import OCRPipelineStep
from app.pipeline.steps.preprocess import PreprocessPipelineStep
from app.pipeline.steps.semantic import SemanticUnderstandingStep

logger = logging.getLogger(__name__)


class PipelineEngine:
    """
    Orchestrates the execution of pipeline steps.
    
    The engine:
    - Manages an ordered list of pipeline steps
    - Creates and flows context through steps
    - Handles errors gracefully
    - Provides timing and logging
    
    Usage:
        engine = PipelineEngine()
        engine.add_step(OCRPipelineStep())
        result = engine.run(image, config)
    """
    
    def __init__(self, steps: Sequence[PipelineStep] | None = None) -> None:
        """
        Initialize the pipeline engine.
        
        Args:
            steps: Optional initial list of steps
        """
        self._steps: list[PipelineStep] = list(steps) if steps else []
    
    def add_step(self, step: PipelineStep) -> None:
        """
        Add a step to the pipeline.
        
        Steps are executed in the order they are added.
        
        Args:
            step: Pipeline step to add
        """
        self._steps.append(step)
        logger.debug(f"Added pipeline step: {step.name}")
    
    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the pipeline by name.
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            True if step was removed, False if not found
        """
        for i, step in enumerate(self._steps):
            if step.name == step_name:
                self._steps.pop(i)
                logger.debug(f"Removed pipeline step: {step_name}")
                return True
        return False
    
    def clear_steps(self) -> None:
        """Remove all steps from the pipeline."""
        self._steps.clear()
    
    @property
    def steps(self) -> list[str]:
        """Get list of step names in execution order."""
        return [step.name for step in self._steps]
    
    def run(self, image: Image.Image, config: OCRConfig) -> OCRResult:
        """
        Execute the pipeline on an image.
        
        Args:
            image: Input image to process
            config: OCR configuration
            
        Returns:
            OCRResult from pipeline execution
        """
        start_time = time.perf_counter()
        
        # Create context
        context = PipelineContext(image=image, config=config)
        original_image = image
        
        logger.info(
            f"Starting pipeline with {len(self._steps)} steps: {self.steps}"
        )
        
        # Execute steps
        for i, step in enumerate(self._steps):
            try:
                if step.should_run(context):
                    logger.debug(f"Executing step: {step.name}")
                    context = step.run(context)
                    
                    # Collect step output if this is the last step or step produced output
                    is_last_step = (i == len(self._steps) - 1)
                    if is_last_step or context.get_intermediate(f"{step.name}_output"):
                        # Extract step output from context
                        step_output = self._extract_step_output(step.name, context, original_image)
                        if step_output:
                            context.set_intermediate(f"{step.name}_output", step_output)
                else:
                    logger.debug(f"Skipping step: {step.name}")
                    
            except Exception as e:
                error_msg = f"Step '{step.name}' failed: {e}"
                logger.error(error_msg, exc_info=True)
                context.add_error(error_msg)
                context.add_step(f"{step.name} (failed)")
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Pipeline complete in {total_time_ms:.2f}ms, "
            f"steps executed: {context.steps_executed}"
        )
        
        # Return result or create result with step outputs
        if context.has_result:
            # Add step results to existing result
            result = context.result
            step_results = self._collect_step_results(context)
            if step_results:
                result.raw_output["step_results"] = step_results
            return result
        
        # No OCR result - create result with step outputs
        step_results = self._collect_step_results(context)
        raw_output = {"step_results": step_results} if step_results else {}
        
        return OCRResult(
            metadata=OCRMetadata(
                engine="pipeline",
                model_variant=config.model_variant.value,
                processing_time_ms=total_time_ms,
            ),
            text_blocks=[],
            raw_output=raw_output,
            errors=context.errors if context.errors else [],
        )
    
    def _extract_step_output(
        self, 
        step_name: str, 
        context: PipelineContext, 
        original_image: Image.Image
    ) -> dict[str, Any] | None:
        """
        Extract output from a step for inclusion in results.
        
        This method is called for each step to extract its output.
        Steps can store their output in context.intermediate with
        a standard key pattern: {step_name}_output
        
        Args:
            step_name: Name of the step
            context: Pipeline context
            original_image: Original input image
            
        Returns:
            Step output dictionary or None
        """
        # Handle preprocessing step output
        if step_name == "preprocess":
            try:
                phase_outputs = context.get_intermediate("preprocess_phase_outputs", {})
                preprocessed_image = context.image
                
                # Convert final preprocessed image to base64
                img_buffer = io.BytesIO()
                preprocessed_image.save(img_buffer, format="PNG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                
                return {
                    "preprocessed_image": f"data:image/png;base64,{img_base64}",
                    "phase_outputs": phase_outputs,
                    "original_size": original_image.size,
                    "preprocessed_size": preprocessed_image.size,
                }
            except Exception as e:
                logger.warning(f"Failed to extract preprocessing output: {e}")
                return None
        
        # For other steps, check if they stored output in intermediate
        step_output = context.get_intermediate(f"{step_name}_output")
        if step_output:
            return step_output
        
        return None
    
    def _collect_step_results(self, context: PipelineContext) -> dict[str, Any]:
        """
        Collect results from all executed steps.
        
        Args:
            context: Pipeline context
            
        Returns:
            Dictionary mapping step names to their outputs
        """
        step_results = {}
        
        for step_name in context.steps_executed:
            # Clean step name (remove "(failed)" suffix if present)
            clean_name = step_name.split(" (")[0]
            
            # Try to get step output
            step_output = context.get_intermediate(f"{clean_name}_output")
            if step_output:
                step_results[clean_name] = step_output
        
        return step_results


def create_default_pipeline() -> PipelineEngine:
    """
    Create a pipeline with the default step configuration.
    
    The semantic understanding step is added conditionally based on
    the config, so we add it here and let the step's should_run() method
    decide whether to execute.
    
    Returns:
        Configured PipelineEngine instance
    """
    engine = PipelineEngine()
    # Add preprocessing step
    engine.add_step(PreprocessPipelineStep())
    # Temporarily disable OCR step for preprocessing testing
    # engine.add_step(OCRPipelineStep())
    # engine.add_step(SemanticUnderstandingStep())
    return engine

