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
    
    Controls global normalization preprocessing layers:
    - Layer 1: Color & Lighting (Grayscale + CLAHE)
    - Layer 2: Geometric (Deskew + Border removal)
    - Layer 3: Signal-to-Noise (Median blur + Bilateral filtering)
    - Layer 4: Binarization (Otsu + Sauvola adaptive thresholding)
    """
    
    # Layer 1: Color & Lighting
    enable_color_lighting: bool = Field(
        default=True,
        description="Enable Color & Lighting layer (Grayscale + CLAHE)"
    )
    clahe_clip_limit: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="CLAHE clip limit for contrast enhancement"
    )
    clahe_tile_grid_size: tuple[int, int] = Field(
        default=(8, 8),
        description="CLAHE tile grid size (rows, cols) for local equalization"
    )
  
    bilateral_d: int = Field(
        default=7,
        ge=1,
        le=15,
        description="Diameter of pixel neighborhood for bilateral filter (odd number, typically 5-9)"
    )
    bilateral_sigma_color: float = Field(
        default=80.0,
        ge=0.0,
        le=200.0,
        description="Filter sigma in color space (larger = more color smoothing)"
    )
    bilateral_sigma_space: float = Field(
        default=70.0,
        ge=0.0,
        le=200.0,
        description="Filter sigma in coordinate space (larger = more spatial smoothing)"
    )
    
    # Layer 2: Geometric
    enable_geometric_layer: bool = Field(
        default=True,
        description="Enable Geometric layer (Deskew + Border detection)"
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
    # Border Detection & Padding
    enable_border_detection: bool = Field(
        default=True,
        description="Enable border detection and padding"
    )
    border_detection_threshold: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Threshold for edge detection (pixel intensity difference)"
    )
    border_padding_percent: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Padding percentage of image dimensions"
    )
    border_min_padding_pixels: int = Field(
        default=20,
        ge=0,
        le=200,
        description="Minimum padding in pixels"
    )
    border_max_padding_pixels: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum padding in pixels"
    )
    
    # Layer 3: Signal-to-Noise
    enable_signal_noise_layer: bool = Field(
        default=True,
        description="Enable Signal-to-Noise layer (Median blur + Bilateral filter)"
    )
    # Noise Detection
    enable_noise_auto_detection: bool = Field(
        default=True,
        description="Enable automatic noise level detection"
    )
    noise_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Manual noise threshold override (None = auto-detect)"
    )
    noise_detection_method: str = Field(
        default="variance",
        description="Noise detection method: 'variance', 'gradient', or 'both'"
    )
    # Median Blur
    enable_median_blur: bool = Field(
        default=False,
        description="Enable median blur filtering"
    )
    median_blur_kernel_size: int | None = Field(
        default=None,
        ge=3,
        le=15,
        description="Median blur kernel size (None = auto-calculate, must be odd)"
    )
    median_blur_auto_min: int = Field(
        default=3,
        ge=3,
        le=15,
        description="Minimum kernel size for auto-detection (must be odd)"
    )
    median_blur_auto_max: int = Field(
        default=3,
        ge=3,
        le=15,
        description="Maximum kernel size for auto-detection (must be odd)"
    )
    median_blur_auto_step: int = Field(
        default=2,
        ge=2,
        le=4,
        description="Step size for kernel size (must result in odd numbers)"
    )
    # Bilateral Filter
    enable_bilateral_filter: bool = Field(
        default=True,
        description="Enable bilateral filtering"
    )
    bilateral_d: int | None = Field(
        default=None,
        ge=1,
        le=15,
        description="Bilateral filter diameter (None = auto-calculate, must be odd)"
    )
    bilateral_sigma_color: float | None = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Bilateral filter color space sigma (None = auto-calculate)"
    )
    bilateral_sigma_space: float | None = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Bilateral filter coordinate space sigma (None = auto-calculate)"
    )
    bilateral_auto_d_min: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Minimum d for auto-detection (must be odd)"
    )
    bilateral_auto_d_max: int = Field(
        default=9,
        ge=1,
        le=15,
        description="Maximum d for auto-detection (must be odd)"
    )
    bilateral_auto_sigma_color_min: float = Field(
        default=50.0,
        ge=0.0,
        le=200.0,
        description="Minimum sigma color for auto-detection"
    )
    bilateral_auto_sigma_color_max: float = Field(
        default=100.0,
        ge=0.0,
        le=200.0,
        description="Maximum sigma color for auto-detection"
    )
    bilateral_auto_sigma_space_min: float = Field(
        default=50.0,
        ge=0.0,
        le=200.0,
        description="Minimum sigma space for auto-detection"
    )
    bilateral_auto_sigma_space_max: float = Field(
        default=100.0,
        ge=0.0,
        le=200.0,
        description="Maximum sigma space for auto-detection"
    )
    # Output Selection
    signal_noise_output_method: str = Field(
        default="bilateral",
        description="Output method: 'median', 'bilateral', or 'both' (show both, use median as final)"
    )
    
    # Layer 4: Binarization
    enable_binarization_layer: bool = Field(
        default=True,
        description="Enable Binarization layer (Otsu + Sauvola adaptive thresholding)"
    )
    binarization_method: str = Field(
        default="both",
        description="Binarization method: 'otsu', 'sauvola', or 'both' (show both, use otsu as final)"
    )
    # Otsu's Method
    enable_otsu: bool = Field(
        default=True,
        description="Enable Otsu's thresholding"
    )
    otsu_max_value: int = Field(
        default=255,
        ge=1,
        le=255,
        description="Maximum value for thresholded pixels"
    )
    otsu_threshold_type: str = Field(
        default="BINARY",
        description="Threshold type: 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV'"
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
    # Output Selection
    binarization_output_method: str = Field(
        default="sauvola",  # Changed from "otsu" to "sauvola"
        description="Output method: 'otsu', 'sauvola', or 'both' (show both, use sauvola as final)"
    )
    
    # Layer 5: Structural Analysis
    enable_structural_analysis: bool = Field(
        default=True,  # Start disabled for testing
        description="Enable Structural Analysis layer (form line detection and removal)"
    )
    # Phase 1: Kernel Definition
    structural_kernel_ratio: float = Field(
        default=40.0,
        ge=20.0,
        le=100.0,
        description="Kernel size ratio (1/N of image dimension, e.g., 40 = 1/40th)"
    )
    structural_kernel_min_size: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Minimum kernel size in pixels (safety limit to avoid too-small kernels)"
    )
    structural_kernel_max_size: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum kernel size in pixels (safety limit to avoid too-large kernels)"
    )
    
    # Phase 3: Grid Intersection & Refinement
    structural_min_line_length_ratio: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Minimum line length ratio (relative to image dimension) to keep"
    )
    structural_intersection_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum intersection size in pixels to consider as grid corner"
    )
    
    # Phase 3: Enhanced Grid Analysis
    structural_enable_junction_map: bool = Field(
        default=True,
        description="Enable junction map (wireframe) visualization"
    )
    structural_enable_comb_field_detection: bool = Field(
        default=True,
        description="Enable comb field detection from intersection points"
    )
    structural_intersection_cluster_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Distance threshold for clustering nearby intersections"
    )
    structural_comb_field_min_lines: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Minimum number of vertical lines for comb field detection"
    )
    structural_comb_field_pattern_detection: bool = Field(
        default=True,
        description="Enable enhanced pattern-based comb field detection (more accurate than intersection-based)"
    )
    structural_comb_field_vertical_tolerance: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Tolerance in pixels for vertical line detection near horizontal lines in comb fields"
    )
    structural_comb_field_min_vertical_coverage: float = Field(
        default=0.6,
        ge=0.3,
        le=1.0,
        description="Minimum percentage of height that vertical lines must cover to be considered part of comb field"
    )
    structural_comb_field_max_line_spacing_ratio: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Maximum spacing between vertical lines as ratio of image width (for pattern validation)"
    )
    structural_comb_field_extraction_padding: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Padding in pixels to add around comb field bounding boxes for complete line extraction"
    )
    structural_segment_context_radius: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Radius for context-aware segment filtering (pixels)"
    )
    
    # Phase 4: Text Preservation
    structural_enable_text_preservation: bool = Field(
        default=True,
        description="Enable intelligent text preservation (distinguish text from lines)"
    )
    structural_text_aspect_ratio_min: float = Field(
        default=0.2,
        ge=0.1,
        le=1.0,
        description="Minimum aspect ratio for text components (width/height or height/width)"
    )
    structural_text_aspect_ratio_max: float = Field(
        default=5.0,
        ge=2.0,
        le=10.0,
        description="Maximum aspect ratio for text components"
    )
    structural_text_min_area: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Minimum area in pixels for text components"
    )
    structural_text_max_area_ratio: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Maximum area ratio (relative to image) for text components"
    )
    structural_text_min_solidity: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Minimum solidity (filledness) for text components"
    )
    
    # Phase 5: Final Mask Generation (Clean Zone Map)
    structural_mask_dilation_kernel_size: int = Field(
        default=3,
        ge=3,
        le=7,
        description="Kernel size for mask dilation (must be odd, typically 3 or 5)"
    )
    structural_mask_dilation_iterations: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of dilation iterations (1 = ~1 pixel, 2 = ~2 pixels)"
    )
    structural_mask_dilation_pixels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Target dilation in pixels (for documentation/clarity, typically 1-2 pixels)"
    )
    
    # Comb Field Removal Mode
    structural_comb_field_removal_only: bool = Field(
        default=True,
        description="Remove only comb fields (checkbox fields), preserve other form lines. Useful for PaddleOCR-VL optimization."
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
        default="document_parsing",
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

