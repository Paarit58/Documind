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


@dataclass(frozen=True)
class DocumentElement:
    """
    A single document element with structured metadata.
    
    Represents individual elements extracted from a document such as
    titles, paragraphs, tables, formulas, charts, etc.
    """
    element_id: str
    element_type: str  # "title", "paragraph", "table", "formula", "chart", etc.
    content: str
    bbox: BBox | None = None
    confidence: float = 1.0
    reading_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate document element data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.reading_order < 0:
            raise ValueError("Reading order must be non-negative")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "content": self.content,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "reading_order": self.reading_order,
            "metadata": self.metadata,
        }
    
    def to_text_block(self) -> TextBlock:
        """Convert to TextBlock for backward compatibility."""
        return TextBlock(
            text=self.content,
            confidence=self.confidence,
            bbox=self.bbox,
        )


@dataclass
class DocumentSection:
    """
    A logical section of a document.
    
    Groups related document elements into sections such as
    header, body, footer, sidebar, etc.
    """
    section_id: str
    section_type: str  # "header", "body", "footer", "sidebar", etc.
    elements: list[DocumentElement] = field(default_factory=list)
    bbox: BBox | None = None
    reading_order: int = 0
    
    def __post_init__(self) -> None:
        """Validate document section data."""
        if self.reading_order < 0:
            raise ValueError("Reading order must be non-negative")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "section_id": self.section_id,
            "section_type": self.section_type,
            "elements": [elem.to_dict() for elem in self.elements],
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "reading_order": self.reading_order,
        }


@dataclass
class EnhancedOCRResult(OCRResult):
    """
    Enhanced OCR result with hierarchical document structure.
    
    Extends OCRResult with:
    - Document-level metadata (type, language)
    - Hierarchical structure (sections and elements)
    - Maintains backward compatibility with text_blocks
    """
    document_type: str = "general"  # "invoice", "form", "letter", "report", "general"
    language: str | None = None
    sections: list[DocumentSection] = field(default_factory=list)
    elements: list[DocumentElement] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation with enhanced structure."""
        base_dict = super().to_dict()
        base_dict.update({
            "document_type": self.document_type,
            "language": self.language,
            "sections": [section.to_dict() for section in self.sections],
            "elements": [elem.to_dict() for elem in self.elements],
        })
        return base_dict


# Type alias for image data (PIL Image will be used at runtime)
# This avoids importing PIL in the types module
ImageType = Any

