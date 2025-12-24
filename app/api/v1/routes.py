"""
API v1 routes for OCR operations.

This module defines the HTTP endpoints for the OCR service.
"""

import io
import json
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from app.api.v1.schemas import (
    ConfigDefaultsResponse,
    DecodingConfigSchema,
    ErrorResponse,
    HealthResponse,
    OCRConfigRequest,
    OCRResponse,
    OCRMetadataSchema,
    TextBlockSchema,
    BBoxSchema,
)
from app.core.config import (
    ModelVariant,
    OCRConfig,
    PromptProfile,
    get_settings,
)
from app.core.registry import get_registry
from app.core.types import OCRResult
from app.pipeline.engine import create_default_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ocr"])

# Supported image formats
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}


def _result_to_response(result: OCRResult) -> OCRResponse:
    """Convert internal OCRResult to API response schema."""
    return OCRResponse(
        metadata=OCRMetadataSchema(
            engine=result.metadata.engine,
            model_variant=result.metadata.model_variant,
            processing_time_ms=result.metadata.processing_time_ms,
        ),
        text_blocks=[
            TextBlockSchema(
                text=block.text,
                confidence=block.confidence,
                bbox=BBoxSchema(
                    x1=block.bbox.x1,
                    y1=block.bbox.y1,
                    x2=block.bbox.x2,
                    y2=block.bbox.y2,
                ) if block.bbox else None,
            )
            for block in result.text_blocks
        ],
        raw_output=result.raw_output,
        errors=result.errors,
    )


def _validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check content type
    if file.content_type and file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format: {file.content_type}. "
                   f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )
    
    # Check file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension: {ext}. "
                       f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}",
            )


async def _load_image(file: UploadFile) -> Image.Image:
    """
    Load and validate image from upload.
    
    Args:
        file: Uploaded file
        
    Returns:
        PIL Image in RGB format
        
    Raises:
        HTTPException: If image cannot be loaded
    """
    settings = get_settings()
    
    try:
        contents = await file.read()
        
        # Check file size
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > settings.max_image_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({size_mb:.1f}MB) exceeds maximum "
                       f"({settings.max_image_size_mb}MB)",
            )
        
        # Load image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load image: {e}",
        )


def _parse_config(config_json: str | None) -> OCRConfig:
    """
    Parse configuration from JSON string.
    
    Args:
        config_json: JSON string or None
        
    Returns:
        OCRConfig instance
        
    Raises:
        HTTPException: If JSON is invalid
    """
    # registry = get_registry()
    # 
    # # Get default engine from registry if available
    # def get_default_engine() -> str:
    #     """Get the default engine from registry or use config default."""
    #     try:
    #         available_engines = registry.list_engines()
    #         if available_engines:
    #             # Use registry default if set, otherwise use first available
    #             return registry.default_engine_name or available_engines[0]
    #     except Exception:
    #         pass
    #     # Fallback to config default
    #     return "paddleocr_vl"
    # 
    # default_engine = get_default_engine()
    default_engine = "paddleocr_vl"
    
    if not config_json:
        # Use default engine from registry
        return OCRConfig(engine=default_engine)
    
    try:
        config_dict = json.loads(config_json)
        config_request = OCRConfigRequest.model_validate(config_dict)
        ocr_config = config_request.to_ocr_config()
        
        # # Validate engine exists, fallback to default if not
        # if not registry.has_engine(ocr_config.engine):
        #     available = registry.list_engines()
        #     if available:
        #         logger.warning(
        #             f"Requested engine '{ocr_config.engine}' not found. "
        #             f"Using default engine '{default_engine}' instead. "
        #             f"Available engines: {available}"
        #         )
        #         ocr_config.engine = default_engine
        #     else:
        #         raise HTTPException(
        #             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #             detail="No OCR engines available. Service is starting up.",
        #         )
        
        return ocr_config
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config JSON: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration: {e}",
        )


@router.post(
    "/ocr",
    response_model=OCRResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        413: {"model": ErrorResponse, "description": "Image too large"},
        415: {"model": ErrorResponse, "description": "Unsupported format"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Perform OCR on an image",
    description="Extract text from an uploaded image using the configured OCR engine.",
)
async def perform_ocr(
    image: Annotated[UploadFile, File(description="Image file to process")],
    config: Annotated[
        str | None,
        Form(description="Optional JSON configuration")
    ] = None,
) -> OCRResponse:
    """
    Perform OCR on an uploaded image.
    
    Args:
        image: Image file (JPEG, PNG, WEBP, TIFF)
        config: Optional JSON configuration string
        
    Returns:
        OCRResponse with extracted text and metadata
    """
    # Skip engine checks for preprocessing-only mode
    # registry = get_registry()
    # if not registry.list_engines():
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail="No OCR engines available. Service is starting up.",
    #     )
    
    # Validate image file
    _validate_image_file(image)
    
    # Load image
    pil_image = await _load_image(image)
    
    # Parse configuration
    ocr_config = _parse_config(config)
    
    # Skip engine existence check for preprocessing-only mode
    # if not registry.has_engine(ocr_config.engine):
    #     available = registry.list_engines()
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail=f"Engine '{ocr_config.engine}' not found. "
    #                f"Available: {available}",
    #     )
    
    # Run pipeline
    logger.info(
        f"Processing OCR request: engine={ocr_config.engine}, "
        f"variant={ocr_config.model_variant.value}"
    )
    
    try:
        pipeline = create_default_pipeline()
        result = pipeline.run(pil_image, ocr_config)
        
        return _result_to_response(result)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {e}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and engine status.",
)
async def health_check() -> HealthResponse:
    """
    Check service health.
    
    Returns:
        Health status including engine availability
    """
    registry = get_registry()
    engines = registry.list_engines()
    
    # Check if default engine is loaded
    engine_loaded = False
    if engines:
        try:
            default_engine = registry.get()
            engine_loaded = default_engine.is_loaded
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if engine_loaded else "degraded",
        engine_loaded=engine_loaded,
        available_engines=engines,
    )


@router.get(
    "/config/defaults",
    response_model=ConfigDefaultsResponse,
    summary="Get default configuration",
    description="Returns the default OCR configuration and available options.",
)
async def get_config_defaults() -> ConfigDefaultsResponse:
    """
    Get default configuration values and available options.
    
    Returns:
        Default configuration with available variants and profiles
    """
    default_config = OCRConfig()
    
    return ConfigDefaultsResponse(
        engine=default_config.engine,
        model_variant=default_config.model_variant.value,
        prompt_profile=default_config.prompt_profile.value,
        decoding=DecodingConfigSchema(
            max_tokens=default_config.decoding.max_tokens,
            temperature=default_config.decoding.temperature,
        ),
        return_raw_output=default_config.return_raw_output,
        available_variants=[v.value for v in ModelVariant],
        available_profiles=[p.value for p in PromptProfile],
    )

