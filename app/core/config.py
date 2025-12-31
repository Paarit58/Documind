"""
Configuration management for the OCR platform.

This module defines:
- AppSettings: Environment-based application configuration
- OCRConfig: Runtime request configuration for OCR operations
- Model variant mappings for DeepSeek-OCR
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelVariant(str, Enum):
    """
    DeepSeek-OCR model variants with different size/quality tradeoffs.
    
    Each variant corresponds to specific inference parameters:
    - tiny: Fastest, lowest quality (512x512)
    - small: Fast, good quality (640x640)
    - base: Balanced (1024x1024)
    - large: High quality, slower (1280x1280)
    - gundam: Optimized for documents with cropping (1024 base, 640 inference)
    """
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GUNDAM = "gundam"


class PromptProfile(str, Enum):
    """
    Predefined prompt profiles for different OCR tasks.
    
    Each profile optimizes the model output for specific use cases.
    """
    FREE_OCR = "free_ocr"
    MARKDOWN = "markdown"
    FORM = "form"
    TABLE = "table"


class PaddleOCRMode(str, Enum):
    """
    PaddleOCR-VL operation modes.
    
    - document_parsing: Full page-level parsing with layout understanding
    - element_recognition: Task-specific element recognition (OCR, table, formula, chart)
    """
    DOCUMENT_PARSING = "document_parsing"
    ELEMENT_RECOGNITION = "element_recognition"


# Mapping of model variants to DeepSeek-OCR parameters
VARIANT_PARAMS: dict[ModelVariant, dict[str, int | bool]] = {
    ModelVariant.TINY: {"base_size": 512, "image_size": 512, "crop_mode": False},
    ModelVariant.SMALL: {"base_size": 640, "image_size": 640, "crop_mode": False},
    ModelVariant.BASE: {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    ModelVariant.LARGE: {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    ModelVariant.GUNDAM: {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


# Mapping of prompt profiles to actual prompts
PROMPT_TEMPLATES: dict[PromptProfile, str] = {
    PromptProfile.FREE_OCR: "<image>\nFree OCR.",
    PromptProfile.MARKDOWN: "<image>\n<|grounding|>Convert the document to markdown.",
    PromptProfile.FORM: "<image>\n<|grounding|>Extract all form fields and their values.",
    PromptProfile.TABLE: "<image>\n<|grounding|>Extract tables in markdown format.",
}


class DecodingConfig(BaseModel):
    """Configuration for model decoding/generation parameters."""
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic)"
    )
    
    model_config = {"frozen": True}


class PreprocessingConfig(BaseModel):
    """
    Configuration for image preprocessing.
    
    Controls simplified preprocessing pipeline:
    1. Grayscale conversion (always)
    2. Deskew (optional)
    3. Binarization (Sauvola method)
    4. Comb field removal (exact working script logic)
    """
    
    # Deskew
    enable_geometric_layer: bool = Field(
        default=True,
        description="Enable deskew detection and correction"
    )
    skew_threshold_degrees: float = Field(
        default=0.5,
        ge=0.0,
        le=45.0,
        description="Minimum angle (degrees) to trigger skew correction"
    )
    deskew_method: str = Field(
        default="auto",
        description="Deskew detection method: 'hough', 'text', or 'auto' (try both)"
    )
    hough_rho_resolution: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Hough Transform rho resolution (pixels)"
    )
    hough_theta_resolution: float = Field(
        default=0.017453292519943295,  # np.pi/180
        ge=0.001,
        le=0.1,
        description="Hough Transform theta resolution (radians)"
    )
    hough_threshold: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Minimum votes for Hough line detection"
    )
    text_projection_step: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Step size (degrees) for projection profile analysis"
    )
    
    # Binarization (Sauvola only)
    enable_binarization_layer: bool = Field(
        default=True,
        description="Enable binarization using Sauvola's adaptive thresholding"
    )
    # Sauvola's Method
    enable_sauvola: bool = Field(
        default=True,
        description="Enable Sauvola's thresholding"
    )
    sauvola_window_size: int | None = Field(
        default=None,
        ge=3,
        le=101,
        description="Window size for local thresholding (None = auto-calculate, must be odd)"
    )
    sauvola_k: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sauvola parameter k (None = auto-calculate, typically 0.34)"
    )
    sauvola_r: float | None = Field(
        default=None,
        ge=0.0,
        le=255.0,
        description="Sauvola parameter r (None = auto-calculate, typically 128)"
    )
    sauvola_auto_window_min: int = Field(
        default=15,
        ge=3,
        le=101,
        description="Minimum window size for auto-detection (must be odd)"
    )
    sauvola_auto_window_max: int = Field(
        default=35,
        ge=3,
        le=101,
        description="Maximum window size for auto-detection (must be odd)"
    )
    sauvola_auto_k_min: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum k for auto-detection"
    )
    sauvola_auto_k_max: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum k for auto-detection"
    )
    sauvola_auto_r_min: float = Field(
        default=64.0,
        ge=0.0,
        le=255.0,
        description="Minimum r for auto-detection"
    )
    sauvola_auto_r_max: float = Field(
        default=192.0,
        ge=0.0,
        le=255.0,
        description="Maximum r for auto-detection"
    )
    
    # Comb Field Removal
    enable_comb_field_removal: bool = Field(
        default=True,
        description="Enable comb field removal using exact working script logic"
    )
    comb_k_h: int = Field(
        default=15,
        ge=3,
        le=50,
        description="Vertical kernel height for morphological opening"
    )
    comb_k_w: int = Field(
        default=15,
        ge=3,
        le=50,
        description="Horizontal kernel width for morphological opening"
    )
    comb_max_h: int = Field(
        default=70,
        ge=10,
        le=200,
        description="Maximum height for short vertical lines (pixels)"
    )
    comb_min_serial: int = Field(
        default=4,
        ge=2,
        le=20,
        description="Minimum number of lines in a row for seriality filtering"
    )
    comb_max_stroke_width: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Maximum stroke width for text rejection (pixels)"
    )
    comb_min_aspect_ratio: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Minimum aspect ratio for text rejection"
    )
    comb_min_long_intersections: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Minimum intersections required for long vertical lines to be considered grid-like"
    )
    
    model_config = {"frozen": True}


class SemanticConfig(BaseModel):
    """Configuration for semantic understanding step."""
    
    enable: bool = Field(
        default=False,
        description="Enable semantic understanding step"
    )
    prompt: str | None = Field(
        default=None,
        description="Custom prompt for semantic analysis (None = use default)"
    )
    
    model_config = {"frozen": True}


class OCRConfig(BaseModel):
    """
    Runtime configuration for OCR requests.
    
    This configuration can be passed per-request to customize
    OCR behavior without code changes.
    """
    
    engine: str = Field(
        default="paddleocr_vl",  # Default to PaddleOCR-VL (can be overridden)
        description="OCR engine identifier"
    )
    model_variant: ModelVariant = Field(
        default=ModelVariant.GUNDAM,
        description="Model size/quality variant"
    )
    prompt_profile: PromptProfile = Field(
        default=PromptProfile.MARKDOWN,
        description="Prompt profile for OCR task"
    )
    decoding: DecodingConfig = Field(
        default_factory=DecodingConfig,
        description="Decoding/generation parameters"
    )
    return_raw_output: bool = Field(
        default=True,
        description="Include raw model output in response"
    )
    preprocessing: PreprocessingConfig | None = Field(
        default=None,
        description="Preprocessing configuration (None = use defaults)"
    )
    paddleocr_mode: PaddleOCRMode | None = Field(
        default=None,
        description="PaddleOCR-VL mode override (None = use default from settings)"
    )
    semantic: SemanticConfig = Field(
        default_factory=SemanticConfig,
        description="Semantic understanding configuration"
    )
    
    model_config = {"frozen": True}
    
    def get_variant_params(self) -> dict[str, int | bool]:
        """Get DeepSeek-OCR parameters for the selected variant."""
        return VARIANT_PARAMS[self.model_variant]
    
    def get_prompt(self) -> str:
        """Get the actual prompt string for the selected profile."""
        return PROMPT_TEMPLATES[self.prompt_profile]


class AppSettings(BaseSettings):
    """
    Application-level settings loaded from environment variables.
    
    These settings are loaded once at startup and should not change
    during runtime.
    """
    
    # Model settings
    model_path: str = Field(
        default="/opt/models/deepseek-ocr",
        description="Path to the DeepSeek-OCR model directory"
    )
    qwen_model_id: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        description="HuggingFace model ID for Qwen3-VL engine"
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to run inference on"
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use flash attention if available"
    )
    use_quantization: bool = Field(
        default=True,
        description="Use 8-bit quantization to reduce memory usage (recommended for systems with <32GB RAM)"
    )
    
    # PaddleOCR-VL settings
    paddleocr_mode: Literal["document_parsing", "element_recognition"] = Field(
        default="element_recognition",
        description="Default PaddleOCR-VL mode"
    )
    
    # Ollama settings for semantic understanding
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="gemma2:2b",
        description="Ollama model name for semantic understanding"
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Limits
    max_image_size_mb: float = Field(
        default=20.0,
        ge=1.0,
        description="Maximum image size in MB"
    )
    request_timeout_seconds: int = Field(
        default=120,
        ge=10,
        description="Request timeout in seconds"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    @field_validator("device", mode="after")
    @classmethod
    def resolve_auto_device(cls, v: str) -> str:
        """Resolve 'auto' device to actual device."""
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Get cached application settings.
    
    Settings are loaded once and cached for the lifetime of the application.
    """
    return AppSettings()

