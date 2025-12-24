"""
Qwen3-VL OCR engine implementation.

This module provides the concrete OCR engine implementation using
the Qwen3-VL-2B-Instruct model for vision-language OCR tasks.

Reference: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
"""

import base64
import io
import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from app.core.config import AppSettings, OCRConfig, get_settings
from app.core.types import BBox, OCRMetadata, OCRResult, TextBlock
from app.engines.base import OCREngine

logger = logging.getLogger(__name__)


class QwenVLEngine(OCREngine):
    """
    Qwen3-VL OCR engine using transformers for inference.
    
    This engine loads the Qwen3-VL-2B-Instruct model from HuggingFace
    and provides thread-safe inference using a mutex lock.
    
    Features:
    - CPU-only device usage (optimized for local testing)
    - Float32 precision for CPU compatibility
    - Thread-safe inference
    - Chat-based prompt interface
    - Dynamic image resolution support
    - Lightweight model (~4.27GB) suitable for systems with limited RAM
    
    Usage:
        engine = QwenVLEngine()
        engine.load()
        result = engine.infer(image, config)
    """
    
    ENGINE_NAME = "qwen_vl"
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    
    def __init__(self, settings: AppSettings | None = None) -> None:
        """
        Initialize the Qwen VL engine.
        
        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"  # Force CPU for now
        self._lock = threading.Lock()
        self._loaded = False
        self._model_id = self._settings.qwen_model_id
    
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
        Load the Qwen3-VL model from HuggingFace.
        
        This method loads both the model and processor, configures
        the device, and prepares the model for inference.
        
        The 2B model is lightweight (~4.27GB) and suitable for CPU inference
        without quantization.
        
        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading Qwen3-VL model: {self._model_id}")
        logger.info(f"Target device: {self._device}")
        logger.info("Model size: ~4.27GB (2B parameters)")
        
        try:
            # Load processor
            logger.info("Loading processor...")
            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )
            
            # Load model (2B model is small enough for direct loading on CPU)
            logger.info("Loading model weights (CPU, float32)...")
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            
            # Configure model for inference
            self._model = self._model.eval()
            
            # Log device info
            try:
                model_device = next(self._model.parameters()).device.type
                logger.info(f"Model loaded on device: {model_device}")
            except Exception:
                logger.info("Model loaded successfully")
            
            logger.info("Qwen3-VL model loaded successfully")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
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
            # Convert PIL Image to temporary file (Qwen expects file:// path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp_path = tmp.name
            
            try:
                # Thread-safe inference
                with self._lock:
                    raw_result = self._run_inference(
                        image_path=tmp_path,
                        config=config,
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
            logger.error(f"Inference failed: {e}", exc_info=True)
            
            # Return error result instead of raising
            return OCRResult(
                metadata=OCRMetadata(
                    engine="Qwen3-VL",
                    model_variant="2B-Instruct",
                    processing_time_ms=processing_time_ms,
                ),
                text_blocks=[],
                raw_output={},
                errors=[str(e)],
            )
    
    def _run_inference(
        self,
        image_path: str,
        config: OCRConfig,
    ) -> str:
        """
        Execute model inference.
        
        Args:
            image_path: Path to the image file
            config: OCR configuration
            
        Returns:
            Generated text output
        """
        # Build prompt from config
        prompt = self._build_prompt(config)
        
        # Prepare messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process vision info (handles image formatting)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self._processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to CPU device
        if hasattr(inputs, "to"):
            inputs = inputs.to("cpu")
        else:
            # Handle dictionary of tensors
            inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            # Handle temperature: if 0.0, use greedy decoding (do_sample=False)
            # Otherwise, use sampling with the specified temperature
            if config.decoding.temperature == 0.0:
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=config.decoding.max_tokens,
                    do_sample=False,  # Greedy decoding for temperature=0.0
                )
            else:
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=config.decoding.max_tokens,
                    temperature=config.decoding.temperature,
                    do_sample=True,  # Sampling for temperature > 0.0
                )
        
        # Extract generated tokens (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Return first output (single image)
        return output_text[0] if output_text else ""
    
    def _build_prompt(self, config: OCRConfig) -> str:
        """
        Build prompt from OCR config.
        
        Maps PromptProfile to appropriate prompts for Qwen.
        Prompts are designed to request JSON output with bounding boxes.
        
        Args:
            config: OCR configuration
            
        Returns:
            Prompt string
        """
        # Base prompt requesting JSON format with bounding boxes
        json_format_instruction = """Return the result in JSON format with the following structure:
{
  "text_blocks": [
    {
      "text": "extracted text content",
      "bbox": {
        "x1": 0,
        "y1": 0,
        "x2": 100,
        "y2": 50
      },
      "confidence": 0.95
    }
  ]
}
Where bbox coordinates are in pixels (x1, y1 = top-left, x2, y2 = bottom-right)."""
        
        # Map existing prompt profiles to Qwen-friendly prompts
        prompt_mapping = {
            "free_ocr": f"""Extract all text from this image with bounding boxes.
{json_format_instruction}""",
            "markdown": f"""Convert the document to markdown format, preserving structure and formatting.
For each text block, provide its bounding box coordinates.
{json_format_instruction}""",
            "form": f"""Extract all form fields and their values from this document.
For each field, provide its bounding box coordinates.
{json_format_instruction}""",
            "table": f"""Extract all tables from this document in markdown format.
For each cell or text region, provide its bounding box coordinates.
{json_format_instruction}""",
        }
        
        profile = config.prompt_profile.value
        return prompt_mapping.get(profile, prompt_mapping["free_ocr"])
    
    def _build_result(
        self,
        raw_result: str,
        config: OCRConfig,
        processing_time_ms: float,
    ) -> OCRResult:
        """
        Build OCRResult from model output.
        
        Args:
            raw_result: Raw text output from the model
            config: OCR configuration
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Structured OCRResult
        """
        # Build metadata
        metadata = OCRMetadata(
            engine="Qwen3-VL",
            model_variant="2B-Instruct",
            processing_time_ms=processing_time_ms,
        )
        
        # Parse text from model output
        text_blocks = self._parse_text_blocks(raw_result)
        
        # Prepare raw output
        raw_output: dict[str, Any] = {}
        if config.return_raw_output:
            raw_output = {"text": raw_result}
        
        return OCRResult(
            metadata=metadata,
            text_blocks=text_blocks,
            raw_output=raw_output,
            errors=[],
        )
    
    def _parse_text_blocks(self, raw_result: str) -> list[TextBlock]:
        """
        Parse text blocks from model output.
        
        Qwen3-VL returns JSON with text blocks and bounding boxes.
        We parse the JSON and extract TextBlock objects with bboxes.
        
        Args:
            raw_result: Raw text output from model (should be JSON)
            
        Returns:
            List of TextBlock objects with bounding boxes
        """
        if not raw_result or not raw_result.strip():
            return []
        
        text = raw_result.strip()
        
        # Try to extract JSON from the response
        # The model might return JSON wrapped in markdown code blocks or plain JSON
        json_text = text
        
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(json_text)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Check if it has text_blocks array
                if "text_blocks" in data and isinstance(data["text_blocks"], list):
                    text_blocks = []
                    for block in data["text_blocks"]:
                        if isinstance(block, dict) and "text" in block:
                            text_content = str(block["text"])
                            confidence = float(block.get("confidence", 1.0))
                            
                            # Parse bounding box if available
                            bbox = None
                            if "bbox" in block and block["bbox"]:
                                bbox_data = block["bbox"]
                                if isinstance(bbox_data, dict):
                                    try:
                                        bbox = BBox(
                                            x1=int(bbox_data.get("x1", 0)),
                                            y1=int(bbox_data.get("y1", 0)),
                                            x2=int(bbox_data.get("x2", 0)),
                                            y2=int(bbox_data.get("y2", 0)),
                                        )
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Invalid bbox format: {e}")
                                        bbox = None
                            
                            text_blocks.append(
                                TextBlock(
                                    text=text_content,
                                    confidence=confidence,
                                    bbox=bbox,
                                )
                            )
                    
                    if text_blocks:
                        return text_blocks
                
                # If structure is different, try to extract text blocks directly
                # Some models might return a flat structure
                if "text" in data:
                    # Single text block
                    bbox = None
                    if "bbox" in data and data["bbox"]:
                        try:
                            bbox_data = data["bbox"]
                            bbox = BBox(
                                x1=int(bbox_data.get("x1", 0)),
                                y1=int(bbox_data.get("y1", 0)),
                                x2=int(bbox_data.get("x2", 0)),
                                y2=int(bbox_data.get("y2", 0)),
                            )
                        except (ValueError, TypeError):
                            bbox = None
                    
                    return [
                        TextBlock(
                            text=str(data["text"]),
                            confidence=float(data.get("confidence", 1.0)),
                            bbox=bbox,
                        )
                    ]
            
            # If it's a list, treat each item as a text block
            elif isinstance(data, list):
                text_blocks = []
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        bbox = None
                        if "bbox" in item and item["bbox"]:
                            try:
                                bbox_data = item["bbox"]
                                bbox = BBox(
                                    x1=int(bbox_data.get("x1", 0)),
                                    y1=int(bbox_data.get("y1", 0)),
                                    x2=int(bbox_data.get("x2", 0)),
                                    y2=int(bbox_data.get("y2", 0)),
                                )
                            except (ValueError, TypeError):
                                bbox = None
                        
                        text_blocks.append(
                            TextBlock(
                                text=str(item["text"]),
                                confidence=float(item.get("confidence", 1.0)),
                                bbox=bbox,
                            )
                        )
                
                if text_blocks:
                    return text_blocks
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from model output: {e}")
            logger.debug(f"Raw output: {text[:500]}")  # Log first 500 chars for debugging
        
        # Fallback: if JSON parsing fails, return as single text block
        # This maintains backward compatibility
        logger.info("Falling back to plain text parsing (no bounding boxes)")
        return [
            TextBlock(
                text=text,
                confidence=1.0,
                bbox=None,
            )
        ]
    
    def get_info(self) -> dict[str, Any]:
        """Get engine information and status."""
        info = super().get_info()
        info.update({
            "model_id": self._model_id,
            "device": self._device,
        })
        return info

