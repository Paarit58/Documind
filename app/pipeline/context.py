"""
Pipeline context for carrying data through processing steps.

The context acts as a shared state container that flows through
the pipeline, accumulating results from each step.
"""

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.core.config import OCRConfig
from app.core.types import OCRResult


@dataclass
class PipelineContext:
    """
    Context object that flows through the pipeline.
    
    The context carries:
    - Input data (image, configuration)
    - Intermediate results from each step
    - The final OCR result
    - Any errors encountered during processing
    
    Each pipeline step receives the context, processes it,
    and returns it (possibly modified) for the next step.
    
    Usage:
        context = PipelineContext(image=my_image, config=my_config)
        context = step1.run(context)
        context = step2.run(context)
        result = context.result
    """
    
    # Input data
    image: Image.Image
    config: OCRConfig
    
    # Results
    result: OCRResult | None = None
    
    # Intermediate storage for step communication
    intermediate: dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: list[str] = field(default_factory=list)
    
    # Step execution tracking
    steps_executed: list[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """
        Add an error message to the context.
        
        Args:
            error: Error message to add
        """
        self.errors.append(error)
    
    def add_step(self, step_name: str) -> None:
        """
        Record that a step has been executed.
        
        Args:
            step_name: Name of the executed step
        """
        self.steps_executed.append(step_name)
    
    def set_intermediate(self, key: str, value: Any) -> None:
        """
        Store an intermediate value for later steps.
        
        Args:
            key: Identifier for the value
            value: Value to store
        """
        self.intermediate[key] = value
    
    def get_intermediate(self, key: str, default: Any = None) -> Any:
        """
        Retrieve an intermediate value.
        
        Args:
            key: Identifier for the value
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        return self.intermediate.get(key, default)
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0
    
    @property
    def has_result(self) -> bool:
        """Check if a result has been set."""
        return self.result is not None

