"""
DeepSeek-OCR engine implementation using PyTorch.

This module provides the concrete OCR engine implementation using
the DeepSeek-OCR model (deepseek-ai/DeepSeek-OCR) with PyTorch backend.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-OCR
"""

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from app.core.config import AppSettings, OCRConfig, get_settings
from app.core.types import OCRMetadata, OCRResult, TextBlock
from app.engines.base import OCREngine

logger = logging.getLogger(__name__)


class DeepSeekTorchEngine(OCREngine):
    """
    DeepSeek-OCR engine using PyTorch for inference.
    
    This engine loads the DeepSeek-OCR model from local disk and
    provides thread-safe inference using a mutex lock.
    
    Features:
    - Flash Attention 2 support (with fallback to eager)
    - GPU/CPU automatic device selection
    - BFloat16 precision for efficiency
    - Thread-safe inference
    
    Usage:
        engine = DeepSeekTorchEngine()
        engine.load()
        result = engine.infer(image, config)
    """
    
    ENGINE_NAME = "deepseek_torch"
    
    def __init__(self, settings: AppSettings | None = None) -> None:
        """
        Initialize the DeepSeek engine.
        
        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = self._settings.device
        self._lock = threading.Lock()
        self._loaded = False
        
    @property
    def engine_name(self) -> str:
        """Get the engine identifier."""
        return self.ENGINE_NAME
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    def load(self) -> None:
        """
        Load the DeepSeek-OCR model from disk.
        
        This method loads both the model and tokenizer, configures
        the attention implementation, and moves the model to the
        appropriate device.
        
        Raises:
            RuntimeError: If model loading fails
            FileNotFoundError: If model path doesn't exist
        """
        model_path = Path(self._settings.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}. "
                f"Please download the model or set DOCUMIND_MODEL_PATH."
            )
        
        logger.info(f"Loading DeepSeek-OCR model from: {model_path}")
        logger.info(f"Target device: {self._device}")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )
            
            # Determine attention implementation
            attn_impl = self._get_attention_implementation()
            logger.info(f"Using attention implementation: {attn_impl}")
            
            # Load model
            logger.info("Loading model weights...")
            self._model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation=attn_impl,
            )
            
            # Configure model for inference
            self._model = self._model.eval()
            
            # Move to device and set precision
            if self._device == "cuda":
                self._model = self._model.cuda().to(torch.bfloat16)
                logger.info("Model loaded on CUDA with bfloat16 precision")
            else:
                # CPU inference - use float32 for compatibility
                self._model = self._model.to(torch.float32)
                logger.info("Model loaded on CPU with float32 precision")
            
            self._loaded = True
            logger.info("DeepSeek-OCR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _get_attention_implementation(self) -> str:
        """
        Determine the best attention implementation.
        
        Returns:
            Attention implementation name ("flash_attention_2" or "eager")
        """
        if not self._settings.use_flash_attention:
            return "eager"
        
        if self._device != "cuda":
            logger.info("Flash attention requires CUDA, falling back to eager")
            return "eager"
        
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            logger.warning(
                "flash-attn not installed, falling back to eager attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            return "eager"
    
    def infer(self, image: Image.Image, config: OCRConfig) -> OCRResult:
        """
        Run OCR inference on an image.
        
        Args:
            image: PIL Image to process (RGB format)
            config: OCR configuration for this request
            
        Returns:
            OCRResult with extracted text and metadata
            
        Raises:
            RuntimeError: If engine is not loaded or inference fails
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")
        
        # Validate image
        self.validate_image(image)
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        start_time = time.perf_counter()
        
        try:
            # Get inference parameters from config
            variant_params = config.get_variant_params()
            prompt = config.get_prompt()
            
            # Save image to temporary file (required by DeepSeek-OCR API)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp_path = tmp.name
            
            try:
                # Thread-safe inference
                with self._lock:
                    raw_result = self._run_inference(
                        image_path=tmp_path,
                        prompt=prompt,
                        base_size=variant_params["base_size"],
                        image_size=variant_params["image_size"],
                        crop_mode=variant_params["crop_mode"],
                    )
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
            
            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Build result
            return self._build_result(
                raw_result=raw_result,
                config=config,
                processing_time_ms=processing_time_ms,
            )
            
        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Inference failed: {e}")
            
            # Return error result instead of raising
            return OCRResult(
                metadata=OCRMetadata(
                    engine="DeepSeek-OCR",
                    model_variant=config.model_variant.value,
                    processing_time_ms=processing_time_ms,
                ),
                text_blocks=[],
                raw_output={},
                errors=[str(e)],
            )
    
    def _run_inference(
        self,
        image_path: str,
        prompt: str,
        base_size: int,
        image_size: int,
        crop_mode: bool,
    ) -> Any:
        """
        Execute model inference.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt string for the model
            base_size: Base resolution for processing
            image_size: Target image size
            crop_mode: Whether to use crop mode
            
        Returns:
            Raw model output
        """
        # Create output directory for model (required by API)
        with tempfile.TemporaryDirectory() as output_dir:
            result = self._model.infer(
                self._tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False,
            )
        
        return result
    
    def _build_result(
        self,
        raw_result: Any,
        config: OCRConfig,
        processing_time_ms: float,
    ) -> OCRResult:
        """
        Build OCRResult from model output.
        
        Args:
            raw_result: Raw output from the model
            config: OCR configuration
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Structured OCRResult
        """
        # Build metadata
        metadata = OCRMetadata(
            engine="DeepSeek-OCR",
            model_variant=config.model_variant.value,
            processing_time_ms=processing_time_ms,
        )
        
        # Parse text from model output
        text_blocks = self._parse_text_blocks(raw_result)
        
        # Prepare raw output
        raw_output: dict[str, Any] = {}
        if config.return_raw_output:
            raw_output = self._serialize_raw_output(raw_result)
        
        return OCRResult(
            metadata=metadata,
            text_blocks=text_blocks,
            raw_output=raw_output,
            errors=[],
        )
    
    def _parse_text_blocks(self, raw_result: Any) -> list[TextBlock]:
        """
        Parse text blocks from model output.
        
        DeepSeek-OCR returns text output (often markdown format).
        We wrap this in a single TextBlock since the model doesn't
        provide per-region bounding boxes in the standard output.
        
        Args:
            raw_result: Raw model output
            
        Returns:
            List of TextBlock objects
        """
        # Handle different output types
        if raw_result is None:
            return []
        
        if isinstance(raw_result, str):
            text = raw_result.strip()
        elif isinstance(raw_result, dict):
            # Try common keys for text output
            text = raw_result.get("text", raw_result.get("output", ""))
            if isinstance(text, list):
                text = "\n".join(str(t) for t in text)
            text = str(text).strip()
        elif isinstance(raw_result, (list, tuple)):
            text = "\n".join(str(item) for item in raw_result)
        else:
            text = str(raw_result).strip()
        
        if not text:
            return []
        
        # DeepSeek-OCR doesn't provide confidence scores or bboxes
        # in its standard output, so we use defaults
        return [
            TextBlock(
                text=text,
                confidence=1.0,  # No confidence score available
                bbox=None,      # No bbox available in standard output
            )
        ]
    
    def _serialize_raw_output(self, raw_result: Any) -> dict[str, Any]:
        """
        Serialize raw model output for JSON response.
        
        Args:
            raw_result: Raw model output
            
        Returns:
            JSON-serializable dictionary
        """
        if raw_result is None:
            return {}
        
        if isinstance(raw_result, dict):
            return {k: self._make_serializable(v) for k, v in raw_result.items()}
        
        if isinstance(raw_result, str):
            return {"text": raw_result}
        
        if isinstance(raw_result, (list, tuple)):
            return {"items": [self._make_serializable(item) for item in raw_result]}
        
        return {"value": str(raw_result)}
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable format."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        if hasattr(obj, "tolist"):  # numpy arrays, torch tensors
            return obj.tolist()
        return str(obj)
    
    def get_info(self) -> dict[str, Any]:
        """Get engine information and status."""
        info = super().get_info()
        info.update({
            "model_path": self._settings.model_path,
            "device": self._device,
            "use_flash_attention": self._settings.use_flash_attention,
        })
        return info

