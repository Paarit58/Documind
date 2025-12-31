"""
Pydantic schemas for API request/response validation.

These schemas define the API contract and ensure type safety
at the API boundary.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import (
    DecodingConfig,
    ModelVariant,
    OCRConfig,
    PaddleOCRMode,
    PromptProfile,
    SemanticConfig,
)


class BBoxSchema(BaseModel):
    """Bounding box coordinates."""
    
    x1: int = Field(ge=0, description="Left x coordinate")
    y1: int = Field(ge=0, description="Top y coordinate")
    x2: int = Field(ge=0, description="Right x coordinate")
    y2: int = Field(ge=0, description="Bottom y coordinate")
    
    model_config = {"json_schema_extra": {"example": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}}}


class TextBlockSchema(BaseModel):
    """A single text block from OCR."""
    
    text: str = Field(description="Extracted text content")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    bbox: BBoxSchema | None = Field(
        default=None,
        description="Bounding box (null if not available)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Hello World",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 150, "y2": 45}
            }
        }
    }


class DocumentElementSchema(BaseModel):
    """A single document element with structured metadata."""
    
    element_id: str = Field(description="Unique identifier for the element")
    element_type: str = Field(description="Type of element (title, paragraph, table, formula, chart, etc.)")
    content: str = Field(description="Extracted text content")
    bbox: BBoxSchema | None = Field(default=None, description="Bounding box coordinates")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    reading_order: int = Field(ge=0, description="Reading order position")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "element_id": "elem_0",
                "element_type": "title",
                "content": "Document Title",
                "bbox": {"x1": 50, "y1": 100, "x2": 450, "y2": 150},
                "confidence": 0.95,
                "reading_order": 1,
                "metadata": {}
            }
        }
    }


class DocumentSectionSchema(BaseModel):
    """A logical section of a document."""
    
    section_id: str = Field(description="Unique identifier for the section")
    section_type: str = Field(description="Type of section (header, body, footer, sidebar, etc.)")
    elements: list[DocumentElementSchema] = Field(
        default_factory=list,
        description="Elements within this section"
    )
    bbox: BBoxSchema | None = Field(default=None, description="Overall section bounding box")
    reading_order: int = Field(ge=0, description="Reading order position")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "section_id": "section_0",
                "section_type": "header",
                "elements": [],
                "bbox": {"x1": 0, "y1": 0, "x2": 800, "y2": 200},
                "reading_order": 0
            }
        }
    }


class OCRMetadataSchema(BaseModel):
    """Metadata about OCR processing."""
    
    engine: str = Field(description="OCR engine used")
    model_variant: str = Field(description="Model variant/configuration")
    processing_time_ms: float = Field(
        ge=0,
        description="Processing time in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "engine": "DeepSeek-OCR",
                "model_variant": "gundam",
                "processing_time_ms": 1250.5
            }
        }
    }


class OCRResponse(BaseModel):
    """
    Standard OCR response format.
    
    This is the stable output contract for all OCR operations.
    Changes to this schema should be versioned.
    """
    
    metadata: OCRMetadataSchema = Field(description="Processing metadata")
    text_blocks: list[TextBlockSchema] = Field(
        default_factory=list,
        description="Extracted text blocks"
    )
    raw_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw model output (if requested)"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )
    # Enhanced structure fields (optional for backward compatibility)
    document_type: str | None = Field(
        default=None,
        description="Detected document type (invoice, form, letter, report, general)"
    )
    language: str | None = Field(
        default=None,
        description="Detected language"
    )
    sections: list[DocumentSectionSchema] = Field(
        default_factory=list,
        description="Document sections (hierarchical structure)"
    )
    elements: list[DocumentElementSchema] = Field(
        default_factory=list,
        description="All document elements (flat list)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "metadata": {
                    "engine": "DeepSeek-OCR",
                    "model_variant": "gundam",
                    "processing_time_ms": 1250.5
                },
                "text_blocks": [
                    {
                        "text": "# Document Title\n\nThis is the extracted text...",
                        "confidence": 1.0,
                        "bbox": None
                    }
                ],
                "raw_output": {},
                "errors": [],
                "document_type": "general",
                "language": "en",
                "sections": [],
                "elements": []
            }
        }
    }


class DecodingConfigSchema(BaseModel):
    """Decoding/generation parameters."""
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class SemanticConfigSchema(BaseModel):
    """Semantic understanding configuration."""
    
    enable: bool = Field(
        default=False,
        description="Enable semantic understanding step"
    )
    prompt: str | None = Field(
        default=None,
        description="Custom prompt for semantic analysis"
    )


class OCRConfigRequest(BaseModel):
    """
    Optional OCR configuration for requests.
    
    All fields have sensible defaults matching the standard configuration.
    """
    
    engine: str = Field(
        default="paddleocr_vl",  # Default to PaddleOCR-VL
        description="OCR engine to use"
    )
    model_variant: ModelVariant = Field(
        default=ModelVariant.GUNDAM,
        description="Model size/quality variant"
    )
    prompt_profile: PromptProfile = Field(
        default=PromptProfile.MARKDOWN,
        description="Prompt profile for OCR task"
    )
    decoding: DecodingConfigSchema = Field(
        default_factory=DecodingConfigSchema,
        description="Decoding parameters"
    )
    return_raw_output: bool = Field(
        default=True,
        description="Include raw model output"
    )
    paddleocr_mode: PaddleOCRMode | None = Field(
        default=None,
        description="PaddleOCR-VL mode override (document_parsing or element_recognition)"
    )
    semantic: SemanticConfigSchema = Field(
        default_factory=SemanticConfigSchema,
        description="Semantic understanding configuration"
    )
    
    def to_ocr_config(self) -> OCRConfig:
        """Convert to internal OCRConfig."""
        return OCRConfig(
            engine=self.engine,
            model_variant=self.model_variant,
            prompt_profile=self.prompt_profile,
            decoding=DecodingConfig(
                max_tokens=self.decoding.max_tokens,
                temperature=self.decoding.temperature,
            ),
            return_raw_output=self.return_raw_output,
            paddleocr_mode=self.paddleocr_mode,
            semantic=SemanticConfig(
                enable=self.semantic.enable,
                prompt=self.semantic.prompt,
            ),
        )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "engine": "deepseek_torch",
                "model_variant": "gundam",
                "prompt_profile": "markdown",
                "decoding": {"max_tokens": 4096, "temperature": 0.0},
                "return_raw_output": True
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    engine_loaded: bool = Field(description="Whether OCR engine is loaded")
    available_engines: list[str] = Field(description="List of registered engines")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "engine_loaded": True,
                "available_engines": ["deepseek_torch"]
            }
        }
    }


class ConfigDefaultsResponse(BaseModel):
    """Default configuration response."""
    
    engine: str
    model_variant: str
    prompt_profile: str
    decoding: DecodingConfigSchema
    return_raw_output: bool
    available_variants: list[str]
    available_profiles: list[str]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "engine": "deepseek_torch",
                "model_variant": "gundam",
                "prompt_profile": "markdown",
                "decoding": {"max_tokens": 4096, "temperature": 0.0},
                "return_raw_output": True,
                "available_variants": ["tiny", "small", "base", "large", "gundam"],
                "available_profiles": ["free_ocr", "markdown", "form", "table"]
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response format."""
    
    detail: str = Field(description="Error message")
    error_type: str = Field(description="Error type/category")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Invalid image format",
                "error_type": "validation_error"
            }
        }
    }

