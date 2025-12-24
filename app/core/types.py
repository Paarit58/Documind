"""
Core type definitions for the OCR platform.

This module defines the fundamental data structures used throughout the system.
These types form the stable output contract and should not change without versioning.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BBox:
    """
    Bounding box coordinates for a text region.
    
    Coordinates are in pixels, relative to the original image.
    Origin (0, 0) is at the top-left corner.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    
    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x1 < 0 or self.y1 < 0 or self.x2 < 0 or self.y2 < 0:
            raise ValueError("Bounding box coordinates must be non-negative")
        if self.x2 < self.x1:
            raise ValueError("x2 must be >= x1")
        if self.y2 < self.y1:
            raise ValueError("y2 must be >= y1")
    
    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return self.width * self.height
    
    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(frozen=True)
class TextBlock:
    """
    A single text block extracted from an image.
    
    Represents a contiguous region of text with optional
    location and confidence information.
    """
    text: str
    confidence: float = 1.0
    bbox: BBox | None = None
    
    def __post_init__(self) -> None:
        """Validate text block data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict() if self.bbox else None,
        }


@dataclass(frozen=True)
class OCRMetadata:
    """
    Metadata about the OCR processing.
    
    Contains information about the engine used and performance metrics.
    """
    engine: str
    model_variant: str
    processing_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "engine": self.engine,
            "model_variant": self.model_variant,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class OCRResult:
    """
    Complete OCR result conforming to the stable output contract.
    
    This is the primary output type for all OCR operations.
    The structure of this class is considered a public API contract.
    """
    metadata: OCRMetadata
    text_blocks: list[TextBlock] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """Concatenate all text blocks into a single string."""
        return "\n".join(block.text for block in self.text_blocks)
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing."""
        return len(self.errors) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation matching the API contract."""
        return {
            "metadata": self.metadata.to_dict(),
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "raw_output": self.raw_output,
            "errors": self.errors,
        }


# Type alias for image data (PIL Image will be used at runtime)
# This avoids importing PIL in the types module
ImageType = Any

