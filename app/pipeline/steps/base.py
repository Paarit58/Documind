"""
Abstract base class for pipeline steps.

Pipeline steps are the building blocks of the processing pipeline.
Each step performs a specific operation and passes the context
to the next step.
"""

from abc import ABC, abstractmethod

from app.pipeline.context import PipelineContext


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    
    Each pipeline step must implement the run() method which:
    - Receives a PipelineContext
    - Performs some processing
    - Returns the (possibly modified) context
    
    Steps should:
    - Be single-responsibility
    - Handle errors gracefully (add to context.errors)
    - Not modify the input image directly
    - Use context.intermediate for step communication
    
    Example:
        class PreprocessStep(PipelineStep):
            @property
            def name(self) -> str:
                return "preprocess"
            
            def run(self, context: PipelineContext) -> PipelineContext:
                # Do preprocessing
                context.set_intermediate("preprocessed", True)
                return context
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the unique name of this step.
        
        Returns:
            Step identifier string
        """
        pass
    
    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this pipeline step.
        
        Args:
            context: The pipeline context with input data
            
        Returns:
            The context, possibly modified with results
        """
        pass
    
    def should_run(self, context: PipelineContext) -> bool:
        """
        Determine if this step should run.
        
        Override this method to implement conditional step execution.
        Default implementation always returns True.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if the step should execute, False to skip
        """
        return True

