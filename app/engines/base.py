"""
Abstract base class for OCR engines.

This module defines the interface that all OCR engines must implement.
The interface is designed to be:
- Simple and focused on core OCR functionality
- Framework-agnostic (supports PyTorch, vLLM, etc.)
- Thread-safe for concurrent inference
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from app.core.config import OCRConfig
from app.core.types import OCRResult


class OCREngine(ABC):
    """
    Abstract base class for OCR engines.
    
    All OCR engine implementations must inherit from this class and
    implement the required abstract methods. This ensures consistent
    behavior across different engine implementations.
    
    The engine lifecycle is:
    1. Instantiation: Create engine with configuration
    2. Loading: Call load() to initialize model
    3. Inference: Call infer() for each image
    4. Shutdown: Engine cleanup (handled by garbage collection)
    
    Thread Safety:
        Implementations MUST be thread-safe. The infer() method may be
        called concurrently from multiple threads.
    
    Example:
        class MyEngine(OCREngine):
            def load(self) -> None:
                self._model = load_my_model()
                
            def infer(self, image, config) -> OCRResult:
                return self._model.process(image)
    """
    
    @abstractmethod
    def load(self) -> None:
        """
        Load and initialize the OCR model.
        
        This method should:
        - Load model weights from disk
        - Initialize any required resources
        - Prepare the model for inference
        
        This is called once at application startup. Implementations
        should fail fast with clear error messages if loading fails.
        
        Raises:
            RuntimeError: If model loading fails
            FileNotFoundError: If model files are not found
        """
        pass
    
    @abstractmethod
    def infer(self, image: Image.Image, config: OCRConfig) -> OCRResult:
        """
        Run OCR inference on an image.
        
        Args:
            image: PIL Image to process. The image should be in RGB format.
            config: OCR configuration for this inference request.
            
        Returns:
            OCRResult containing extracted text and metadata.
            
        Raises:
            RuntimeError: If inference fails
            ValueError: If the image format is invalid
            
        Note:
            This method must be thread-safe. Use appropriate locking
            if the underlying model is not thread-safe.
        """
        pass
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        """
        Get the unique identifier for this engine.
        
        Returns:
            A string identifier (e.g., "deepseek_torch", "tesseract")
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if the engine is loaded and ready for inference.
        
        Returns:
            True if the engine is ready, False otherwise
        """
        pass
    
    def get_info(self) -> dict[str, Any]:
        """
        Get engine information and status.
        
        Returns:
            Dictionary containing engine metadata
        """
        return {
            "engine_name": self.engine_name,
            "is_loaded": self.is_loaded,
        }
    
    def validate_image(self, image: Image.Image) -> None:
        """
        Validate that an image is suitable for OCR.
        
        Args:
            image: PIL Image to validate
            
        Raises:
            ValueError: If the image is invalid
        """
        if image is None:
            raise ValueError("Image cannot be None")
        
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Image dimensions cannot be zero")
        
        # Convert to RGB if necessary (handled by caller typically)
        if image.mode not in ("RGB", "L", "RGBA"):
            raise ValueError(f"Unsupported image mode: {image.mode}")

