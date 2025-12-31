"""
PaddleOCR-VL OCR engine implementation.

This module provides the concrete OCR engine implementation using
PaddleOCR-VL for document parsing and element-level recognition.

Reference: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
"""

import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import AppSettings, OCRConfig, PaddleOCRMode, get_settings
from app.core.types import (
    BBox,
    DocumentElement,
    DocumentSection,
    EnhancedOCRResult,
    OCRMetadata,
    OCRResult,
    TextBlock,
)
from app.engines.base import OCREngine

logger = logging.getLogger(__name__)

# Structured prompt template for element recognition
# This prompt instructs the model to output JSON matching our DocumentElement format
STRUCTURED_OCR_PROMPT = """OCR: Extract all text from this document and return ONLY valid JSON in this exact format (no markdown, no explanations, just JSON):

{
  "elements": [
    {
      "element_id": "elem_0",
      "element_type": "title",
      "content": "Document Title",
      "bbox": [50, 100, 450, 150],
      "confidence": 0.95,
      "reading_order": 1
    },
    {
      "element_id": "elem_1",
      "element_type": "paragraph",
      "content": "Paragraph text content...",
      "bbox": [50, 160, 450, 200],
      "confidence": 0.92,
      "reading_order": 2
    }
  ]
}

Requirements:
- element_type must be one of: title, paragraph, table, formula, chart, text, header, footer, list
- bbox must be pixel coordinates [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
- reading_order must be sequential integers starting from 1
- confidence must be between 0.0 and 1.0
- Return ONLY the JSON object, no markdown code blocks, no explanations, no additional text"""


class PaddleOCRVLEngine(OCREngine):
    """
    PaddleOCR-VL OCR engine using PaddleOCR library.
    
    This engine supports both document parsing (page-level) and
    element-level recognition modes.
    
    Features:
    - Document parsing: Full page-level parsing with layout understanding
    - Element recognition: Task-specific recognition (OCR, table, formula, chart)
    - Bounding box extraction for all elements
    - Thread-safe inference
    - CPU-optimized (no GPU required)
    
    Usage:
        engine = PaddleOCRVLEngine()
        engine.load()
        result = engine.infer(image, config)
    """
    
    ENGINE_NAME = "paddleocr_vl"
    
    def __init__(self, settings: AppSettings | None = None) -> None:
        """
        Initialize the PaddleOCR-VL engine.
        
        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._pipeline: Any = None  # For document parsing (PaddleOCR library)
        self._transformers_model: Any = None  # For element recognition (transformers)
        self._transformers_processor: Any = None  # For element recognition (transformers)
        self._lock = threading.Lock()
        self._loaded = False
        self._mode = self._settings.paddleocr_mode
    
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
        Load the PaddleOCR-VL pipeline and optionally transformers model.
        
        This method initializes:
        - PaddleOCRVL pipeline for document parsing (always loaded)
        - Transformers model for element-level recognition (only if needed)
        
        For document parsing mode, only PaddleOCRVL is loaded (fast).
        Transformers model is only loaded if element recognition mode might be used.
        
        Raises:
            RuntimeError: If model loading fails
            ImportError: If PaddleOCR is not installed
        """
        logger.info("Loading PaddleOCR-VL pipeline...")
        logger.info(f"Default mode: {self._mode}")
        logger.info("Device: CPU (CPU-only deployment)")
        
        try:
            # Load PaddleOCR library for document parsing
            from paddleocr import PaddleOCRVL
            
            logger.info("Initializing PaddleOCRVL pipeline for document parsing...")
            self._pipeline = PaddleOCRVL()
            logger.info("PaddleOCRVL pipeline loaded successfully")
            
        except ImportError as e:
            error_msg = (
                "PaddleOCR is not installed. "
                "Install with: pip install paddlepaddle==3.2.1 && pip install -U 'paddleocr[doc-parser]'"
            )
            logger.error(f"{error_msg}: {e}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL: {e}")
            raise RuntimeError(f"PaddleOCR-VL loading failed: {e}") from e
        
        # Only load transformers model if element recognition mode might be used
        # For document parsing mode, skip transformers to improve performance
        default_mode = PaddleOCRMode(self._mode)
        if default_mode == PaddleOCRMode.ELEMENT_RECOGNITION:
            logger.info("Default mode is element recognition - loading transformers model...")
            self._load_transformers_model()
        else:
            logger.info(
                "Default mode is document parsing - skipping transformers model for better performance. "
                "Transformers will be loaded on-demand if element recognition is requested."
            )
            # Don't load transformers now - will be loaded lazily if needed
        
        self._loaded = True
        logger.info("PaddleOCR-VL engine loaded successfully")
    
    def _load_transformers_model(self) -> None:
        """
        Load transformers model for element-level recognition.
        
        This is called either at startup (if default mode is element recognition)
        or lazily when element recognition is actually requested.
        """
        # Check if already loaded
        if self._transformers_model is not None and self._transformers_processor is not None:
            logger.debug("Transformers model already loaded")
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            model_id = "PaddlePaddle/PaddleOCR-VL"
            logger.info(f"Loading transformers model for element recognition: {model_id}")
            
            # Check transformers version compatibility
            try:
                import transformers
                transformers_version = transformers.__version__
                logger.info(f"Transformers version: {transformers_version}")
                
                # PaddleOCR-VL model specifies transformers_version: "4.55.0" in config.json
                # Warn if version doesn't match (but don't fail, as it might still work)
                if transformers_version != "4.55.0":
                    logger.warning(
                        f"Transformers version {transformers_version} may not be compatible. "
                        "PaddleOCR-VL recommends transformers==4.55.0. "
                        "If you encounter rope_type KeyError, install: pip install transformers==4.55.0"
                    )
            except Exception as e:
                logger.warning(f"Could not determine transformers version: {e}")
            
            # Check torch availability
            logger.debug(f"PyTorch version: {torch.__version__}")
            logger.debug(f"CUDA available: {torch.cuda.is_available()}")
            
            # Determine device and dtype (following official example)
            if torch.cuda.is_available():
                dtype = torch.bfloat16
                device = "cuda"
                # Use flash_attention_2 for GPU if available (as per official example)
                attn_implementation = "flash_attention_2"
            else:
                dtype = torch.float32
                device = "cpu"
                attn_implementation = None  # CPU doesn't use flash attention
            
            # Load processor first (as per official example)
            logger.info("Loading processor...")
            self._transformers_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            logger.info("Processor loaded successfully")
            
            # Load model (following official example pattern)
            logger.info(f"Loading model on {device} with dtype {dtype}...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }
            
            # Add attention implementation for GPU (as per official example)
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation
            
            self._transformers_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            ).to(dtype=dtype, device=device).eval()
            
            logger.info(f"Transformers model loaded successfully on {device}")
            
        except ImportError as e:
            logger.error(
                f"Transformers not available for element recognition: {e}. "
                "Element recognition will fallback to document parsing. "
                "Install transformers: pip install transformers torch",
                exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Failed to load transformers model for element recognition: {e}. "
                "Element recognition will fallback to document parsing.",
                exc_info=True
            )
            # Log additional context
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
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
            # Determine mode (config override or default)
            mode = config.paddleocr_mode or PaddleOCRMode(self._mode)
            
            # Save image to temporary file (PaddleOCR expects file path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp_path = tmp.name
            
            try:
                # Thread-safe inference
                with self._lock:
                    if mode == PaddleOCRMode.DOCUMENT_PARSING:
                        raw_result = self._run_document_parsing(tmp_path)
                    else:
                        raw_result = self._run_element_recognition(tmp_path, config)
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
            
            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Build result (pass image for dimension calculation)
            return self._build_result(
                raw_result=raw_result,
                config=config,
                processing_time_ms=processing_time_ms,
                mode=mode,
                image=image,
            )
            
        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Inference failed: {e}", exc_info=True)
            
            # Return error result instead of raising
            return OCRResult(
                metadata=OCRMetadata(
                    engine="PaddleOCR-VL",
                    model_variant=f"{mode.value if 'mode' in locals() else 'unknown'}",
                    processing_time_ms=processing_time_ms,
                ),
                text_blocks=[],
                raw_output={},
                errors=[str(e)],
            )
    
    def _extract_json_from_output(self, text: str) -> dict[str, Any] | None:
        """
        Extract JSON from model output text.
        
        Handles various formats:
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Plain JSON responses
        - JSON with surrounding text
        
        Args:
            text: Raw text output from model
            
        Returns:
            Parsed JSON dictionary or None if extraction fails
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Try to extract JSON from markdown code blocks
        # Pattern: ```json ... ``` or ``` ... ```
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON from markdown code block")
        
        # Try plain ``` code blocks
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
                # Remove language identifier if present
                if json_text.startswith("json"):
                    json_text = json_text[4:].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON from code block")
        
        # Try to find JSON object directly
        # Look for opening brace
        start_idx = text.find("{")
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON object")
        
        # Try to find JSON array
        start_idx = text.find("[")
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(text)):
                if text[i] == "[":
                    bracket_count += 1
                elif text[i] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                try:
                    parsed = json.loads(json_text)
                    # If it's an array, wrap it in elements key
                    if isinstance(parsed, list):
                        return {"elements": parsed}
                    return parsed
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON array")
        
        # Last resort: try parsing entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to extract JSON from model output")
            return None
    
    def _parse_element_recognition_json(
        self, json_data: dict[str, Any], image: Image.Image
    ) -> list[DocumentElement]:
        """
        Parse JSON output from element recognition to DocumentElement objects.
        
        Args:
            json_data: Parsed JSON from model output
            image: Original image (for coordinate validation)
            
        Returns:
            List of DocumentElement objects
        """
        elements: list[DocumentElement] = []
        
        if not isinstance(json_data, dict):
            logger.warning("JSON data is not a dictionary")
            return elements
        
        # Extract elements array
        elements_data = json_data.get("elements", [])
        if not isinstance(elements_data, list):
            logger.warning("Elements data is not a list")
            return elements
        
        image_width = image.width if image else 1000
        image_height = image.height if image else 1000
        
        for idx, elem_data in enumerate(elements_data):
            if not isinstance(elem_data, dict):
                continue
            
            # Extract content
            content = elem_data.get("content", elem_data.get("text", ""))
            if not content:
                continue
            
            # Extract element type
            element_type = self._normalize_element_type(
                elem_data.get("element_type", elem_data.get("type", "text"))
            )
            
            # Extract element ID
            element_id = elem_data.get("element_id", f"elem_{idx}")
            
            # Extract confidence
            confidence = float(elem_data.get("confidence", elem_data.get("score", 1.0)))
            confidence = max(0.0, min(1.0, confidence))
            
            # Extract reading order
            reading_order = int(elem_data.get("reading_order", elem_data.get("order", idx + 1)))
            
            # Extract bounding box
            bbox = None
            bbox_data = elem_data.get("bbox", elem_data.get("bounding_box", elem_data.get("box")))
            if bbox_data:
                try:
                    if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                        coords = [float(x) for x in bbox_data[:4]]
                        
                        # Check if coordinates are normalized (0-1) or pixels
                        # If all values are between 0 and 1, assume normalized
                        if all(0 <= c <= 1 for c in coords):
                            # Convert normalized to pixels
                            x1 = int(coords[0] * image_width)
                            y1 = int(coords[1] * image_height)
                            x2 = int(coords[2] * image_width)
                            y2 = int(coords[3] * image_height)
                        else:
                            # Assume pixel coordinates
                            x1 = int(coords[0])
                            y1 = int(coords[1])
                            x2 = int(coords[2])
                            y2 = int(coords[3])
                        
                        # Validate coordinates
                        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                            if x2 <= image_width and y2 <= image_height:
                                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                            else:
                                # Clamp to image dimensions
                                x2 = min(x2, image_width)
                                y2 = min(y2, image_height)
                                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    elif isinstance(bbox_data, dict):
                        # Dict format
                        x1 = int(bbox_data.get("x1", bbox_data.get("left", 0)))
                        y1 = int(bbox_data.get("y1", bbox_data.get("top", 0)))
                        x2 = int(bbox_data.get("x2", bbox_data.get("right", image_width)))
                        y2 = int(bbox_data.get("y2", bbox_data.get("bottom", image_height)))
                        bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Failed to parse bbox: {e}")
                    bbox = None
            
            # Extract metadata
            metadata = {
                "source": "element_recognition",
                "original_type": elem_data.get("element_type", "text"),
            }
            
            elements.append(
                DocumentElement(
                    element_id=element_id,
                    element_type=element_type,
                    content=str(content),
                    bbox=bbox,
                    confidence=confidence,
                    reading_order=reading_order,
                    metadata=metadata,
                )
            )
        
        logger.debug(f"Parsed {len(elements)} elements from element recognition JSON")
        return elements
    
    def _normalize_element_type(self, element_type: str) -> str:
        """
        Normalize element type to standard values.
        
        Args:
            element_type: Raw element type from model output
            
        Returns:
            Normalized element type
        """
        if not element_type:
            return "text"
        
        element_type = element_type.lower().strip()
        
        # Map common variations to standard types
        type_mapping = {
            "title": "title",
            "heading": "title",
            "h1": "title",
            "h2": "title",
            "h3": "title",
            "paragraph": "paragraph",
            "para": "paragraph",
            "p": "paragraph",
            "table": "table",
            "formula": "formula",
            "equation": "formula",
            "chart": "chart",
            "graph": "chart",
            "text": "text",
            "header": "header",
            "footer": "footer",
            "list": "list",
            "bullet": "list",
            "numbered": "list",
        }
        
        # Direct match
        if element_type in type_mapping:
            return type_mapping[element_type]
        
        # Partial match
        for key, value in type_mapping.items():
            if key in element_type or element_type in key:
                return value
        
        # Default to text
        return "text"
    
    def _run_document_parsing(self, image_path: str) -> list[Any]:
        """
        Run full document parsing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of PaddleOCR result objects (or single object wrapped in list)
        """
        logger.debug("Running document parsing mode")
        output = self._pipeline.predict(image_path)
        
        # Ensure output is always a list for consistent processing
        if output is None:
            logger.warning("PaddleOCR predict returned None")
            return []
        if not isinstance(output, list):
            logger.debug(f"PaddleOCR returned non-list type: {type(output)}, wrapping in list")
            return [output]
        
        return output
    
    def _run_element_recognition(
        self, image_path: str, config: OCRConfig
    ) -> list[Any]:
        """
        Run element-level recognition using transformers with task-specific prompts.
        
        This uses the transformers library approach with specific prompts:
        - "OCR:" for text recognition
        - "Table Recognition:" for tables
        - "Formula Recognition:" for formulas
        - "Chart Recognition:" for charts
        
        Args:
            image_path: Path to the image file
            config: OCR configuration
            
        Returns:
            List of result objects (wrapped for consistent processing)
        """
        logger.debug(f"Running element recognition mode with profile: {config.prompt_profile.value}")
        
        # Check if transformers model is available, load lazily if needed
        if not self._transformers_model or not self._transformers_processor:
            logger.info(
                "Transformers model not loaded yet. Loading now for element recognition..."
            )
            self._load_transformers_model()
            
            # If still not loaded after attempt, fallback to document parsing
            if not self._transformers_model or not self._transformers_processor:
                logger.warning(
                    "Transformers model failed to load for element recognition. "
                    "Falling back to document parsing."
                )
                return self._run_document_parsing(image_path)
        
        # Use structured prompt for element recognition
        # This prompts the model to output JSON matching our DocumentElement format
        task_prompt = STRUCTURED_OCR_PROMPT
        logger.debug("Using structured OCR prompt for element recognition")
        
        try:
            import torch
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare messages with task-specific prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": task_prompt},
                    ]
                }
            ]
            
            # Apply chat template
            inputs = self._transformers_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            device = next(self._transformers_model.parameters()).device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            elif hasattr(inputs, "to"):
                inputs = inputs.to(device)
            
            # Get input_ids for trimming later
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            
            # Generate with increased token limit for structured JSON output
            with torch.no_grad():
                outputs = self._transformers_model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Increased for structured JSON output
                    do_sample=False,
                    use_cache=True,
                )
            
            # Extract generated tokens (remove input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, outputs)
            ]
            
            # Decode output
            generated_text = self._transformers_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )[0]
            
            logger.debug(f"Generated text length: {len(generated_text)}")
            
            # Store raw generated text for JSON parsing
            # We'll parse this in _build_result() to extract structured elements
            class ElementRecognitionResult:
                """Wrapper for element recognition results from transformers."""
                def __init__(self, text: str, task: str, engine_instance=None):
                    self.text = text
                    self.task = task
                    self._text_content = text.strip()
                    self._task_type = task.lower().replace(" recognition:", "").replace(":", "")
                    self._raw_json_text = text.strip()  # Store for JSON parsing
                    self._engine = engine_instance  # Reference to engine for JSON extraction
                
                def print(self) -> str:
                    """Return text representation."""
                    return self._text_content
                
                def save_to_json(self, save_path: str) -> None:
                    """Save result as JSON compatible with PaddleOCR format."""
                    import json
                    import os
                    # Try to extract JSON from response
                    json_data = self._extract_json_from_text(self._raw_json_text)
                    if json_data:
                        json_path = os.path.join(save_path, "result.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Fallback to simple structure
                        data = {
                            "text_blocks": [
                                {
                                    "text": self._text_content,
                                    "type": self._task_type,
                                    "confidence": 1.0,
                                }
                            ],
                            "task": self._task_type,
                        }
                        json_path = os.path.join(save_path, "result.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                
                def save_to_markdown(self, save_path: str) -> None:
                    """Save result as Markdown."""
                    import os
                    md_path = os.path.join(save_path, "result.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(self._text_content)
                
                def _extract_json_from_text(self, text: str) -> dict[str, Any] | None:
                    """Extract JSON from text (helper method)."""
                    if self._engine:
                        return self._engine._extract_json_from_output(text)
                    return None
            
            result_obj = ElementRecognitionResult(generated_text, task_prompt, self)
            logger.info(f"Element recognition completed: {len(generated_text)} characters extracted")
            return [result_obj]
            
        except Exception as e:
            logger.error(f"Element recognition with transformers failed: {e}", exc_info=True)
            logger.warning("Falling back to document parsing")
            # Fallback to document parsing
            return self._run_document_parsing(image_path)
    
    def _build_result(
        self,
        raw_result: list[Any],
        config: OCRConfig,
        processing_time_ms: float,
        mode: PaddleOCRMode,
        image: Image.Image | None = None,
    ) -> EnhancedOCRResult:
        """
        Build EnhancedOCRResult from PaddleOCR output with hierarchical structure.
        
        Args:
            raw_result: Raw output from PaddleOCR-VL
            config: OCR configuration
            processing_time_ms: Processing time in milliseconds
            mode: Operation mode used
            image: Original image (for dimension calculation)
            
        Returns:
            EnhancedOCRResult with hierarchical structure
        """
        # Build metadata
        metadata = OCRMetadata(
            engine="PaddleOCR-VL",
            model_variant=mode.value,
            processing_time_ms=processing_time_ms,
        )
        
        # Parse text blocks from PaddleOCR output (for backward compatibility)
        text_blocks = self._parse_paddleocr_output(raw_result)
        
        # Parse structured output to get DocumentElement objects
        elements: list[DocumentElement] = []
        sections: list[DocumentSection] = []
        
        # Check if this is element recognition mode (transformers output)
        if mode == PaddleOCRMode.ELEMENT_RECOGNITION:
            # Try to extract JSON from element recognition output
            for result_obj in raw_result:
                try:
                    if hasattr(result_obj, '_raw_json_text'):
                        # This is from element recognition - extract JSON from text
                        json_data = self._extract_json_from_output(result_obj._raw_json_text)
                        if json_data and isinstance(json_data, dict):
                            parsed_elements = self._parse_element_recognition_json(json_data, image)
                            if parsed_elements:
                                elements.extend(parsed_elements)
                                logger.info(f"Parsed {len(parsed_elements)} elements from element recognition JSON")
                                break
                    elif hasattr(result_obj, 'text'):
                        # Try to extract JSON from text attribute
                        json_data = self._extract_json_from_output(result_obj.text)
                        if json_data and isinstance(json_data, dict):
                            parsed_elements = self._parse_element_recognition_json(json_data, image)
                            if parsed_elements:
                                elements.extend(parsed_elements)
                                logger.info(f"Parsed {len(parsed_elements)} elements from element recognition text")
                                break
                except Exception as e:
                    logger.warning(f"Failed to parse element recognition JSON: {e}")
                    # Continue to try other result objects or fallback to document parsing
        
        # For document parsing mode, use existing JSON extraction
        if not elements:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for result_obj in raw_result:
                        if hasattr(result_obj, 'save_to_json'):
                            try:
                                result_obj.save_to_json(save_path=tmp_dir)
                                json_files = list(Path(tmp_dir).glob("*.json"))
                                if json_files:
                                    with open(json_files[0], 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    
                                    # Parse structured output from document parsing
                                    parsed_elements = self._parse_paddleocr_structured_output(json_data)
                                    elements.extend(parsed_elements)
                                    break  # Use first valid JSON
                            except Exception as e:
                                logger.debug(f"Failed to extract structured data: {e}")
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {e}")
        
        # If we got elements, group them into sections
        if elements:
            image_height = image.height if image else 1000  # Default fallback
            sections = self._group_elements_into_sections(elements, image_height)
        else:
            # Fallback: create elements from text blocks
            # This handles cases where JSON parsing failed or returned no elements
            logger.debug("No elements from structured parsing, creating from text blocks")
            for i, block in enumerate(text_blocks):
                elements.append(
                    DocumentElement(
                        element_id=f"elem_{i}",
                        element_type="text",
                        content=block.text,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        reading_order=i,
                        metadata={},
                    )
                )
            
            # If we have elements now, group them
            if elements:
                image_height = image.height if image else 1000
                sections = self._group_elements_into_sections(elements, image_height)
        
        # Prepare raw output
        raw_output: dict[str, Any] = {}
        if config.return_raw_output:
            raw_output = self._serialize_raw_output(raw_result)
            # Add structured data to raw output
            raw_output["structured_elements"] = [e.to_dict() for e in elements]
            raw_output["structured_sections"] = [s.to_dict() for s in sections]
        
        # Detect document type (basic heuristics)
        document_type = self._detect_document_type(elements)
        
        # Create enhanced result
        return EnhancedOCRResult(
            metadata=metadata,
            text_blocks=text_blocks,  # Backward compatibility
            raw_output=raw_output,
            errors=[],
            document_type=document_type,
            language=None,  # Could be enhanced with language detection
            sections=sections,
            elements=elements,
        )
    
    def _detect_document_type(self, elements: list[DocumentElement]) -> str:
        """
        Detect document type from elements (basic heuristics).
        
        Args:
            elements: List of document elements
            
        Returns:
            Detected document type
        """
        if not elements:
            return "general"
        
        # Count element types
        type_counts: dict[str, int] = {}
        for elem in elements:
            type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
        
        # Heuristics
        if type_counts.get("table", 0) > 2:
            return "report"  # Multiple tables suggest report
        elif any("invoice" in elem.content.lower() for elem in elements[:5]):
            return "invoice"
        elif any("form" in elem.content.lower() for elem in elements[:5]):
            return "form"
        elif any("dear" in elem.content.lower() or "sincerely" in elem.content.lower() for elem in elements):
            return "letter"
        else:
            return "general"
    
    def _parse_paddleocr_output(self, raw_result: list[Any]) -> list[TextBlock]:
        """
        Parse text blocks from PaddleOCR output.
        
        PaddleOCR-VL returns result objects with methods like:
        - .print() - formatted text output
        - .save_to_json() - JSON output with bounding boxes
        - .save_to_markdown() - Markdown output
        
        We extract text blocks with bounding boxes from the JSON format.
        
        Args:
            raw_result: List of PaddleOCR result objects
            
        Returns:
            List of TextBlock objects with bounding boxes
        """
        if not raw_result:
            logger.warning("PaddleOCR returned empty result")
            return []
        
        text_blocks: list[TextBlock] = []
        
        try:
            # Log the type of result we got for debugging
            logger.debug(f"PaddleOCR result type: {type(raw_result)}, length: {len(raw_result)}")
            
            # PaddleOCR-VL returns a list of result objects
            # Each result object has methods to export data
            for idx, result_obj in enumerate(raw_result):
                logger.debug(f"Processing result object {idx}, type: {type(result_obj)}")
                
                # Try multiple methods to extract data
                extracted = False
                
                # Method 1: Try save_to_json (most reliable for structured data)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        if hasattr(result_obj, 'save_to_json'):
                            logger.debug(f"Result object has save_to_json method")
                            result_obj.save_to_json(save_path=tmp_dir)
                            
                            # Find the JSON file in the directory
                            json_files = list(Path(tmp_dir).glob("*.json"))
                            if json_files:
                                json_path = json_files[0]
                                logger.debug(f"Found JSON file: {json_path}")
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                                
                                logger.debug(f"JSON data keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'not a dict'}")
                                
                                # Parse JSON structure
                                extracted_blocks = self._extract_text_blocks_from_json(json_data)
                                if extracted_blocks:
                                    text_blocks.extend(extracted_blocks)
                                    extracted = True
                                    logger.debug(f"Extracted {len(extracted_blocks)} text blocks from JSON")
                    except Exception as e:
                        logger.debug(f"save_to_json method failed: {e}")
                
                # Method 2: Try direct attribute access (result might have text/data attributes)
                if not extracted:
                    try:
                        # Check for common attribute names
                        if hasattr(result_obj, 'text'):
                            text = str(result_obj.text)
                            if text.strip():
                                text_blocks.append(
                                    TextBlock(text=text, confidence=1.0, bbox=None)
                                )
                                extracted = True
                                logger.debug("Extracted text from .text attribute")
                        elif hasattr(result_obj, 'content'):
                            text = str(result_obj.content)
                            if text.strip():
                                text_blocks.append(
                                    TextBlock(text=text, confidence=1.0, bbox=None)
                                )
                                extracted = True
                                logger.debug("Extracted text from .content attribute")
                        elif hasattr(result_obj, 'result'):
                            # Nested result
                            nested = result_obj.result
                            if isinstance(nested, str):
                                text_blocks.append(
                                    TextBlock(text=nested, confidence=1.0, bbox=None)
                                )
                                extracted = True
                                logger.debug("Extracted text from .result attribute")
                        elif hasattr(result_obj, 'to_dict'):
                            # Try to_dict() method
                            dict_data = result_obj.to_dict()
                            if isinstance(dict_data, dict):
                                extracted_blocks = self._extract_text_blocks_from_json(dict_data)
                                if extracted_blocks:
                                    text_blocks.extend(extracted_blocks)
                                    extracted = True
                                    logger.debug("Extracted text from to_dict() method")
                    except Exception as e:
                        logger.debug(f"Direct attribute access failed: {e}")
                
                # Method 3: Try print() method (fallback)
                if not extracted:
                    self._fallback_text_extraction(result_obj, text_blocks)
        
        except Exception as e:
            logger.error(f"Failed to parse PaddleOCR output: {e}", exc_info=True)
            # Fallback: create a single text block from raw result
            if raw_result:
                text = str(raw_result)
                if text.strip() and text != str(type(raw_result)):
                    text_blocks.append(
                        TextBlock(
                            text=text,
                            confidence=1.0,
                            bbox=None,
                        )
                    )
                    logger.debug("Created fallback text block from raw result")
        
        logger.info(f"Total text blocks extracted: {len(text_blocks)}")
        return text_blocks if text_blocks else []
    
    def _fallback_text_extraction(self, result_obj: Any, text_blocks: list[TextBlock]) -> None:
        """
        Fallback method to extract text from result object.
        
        Args:
            result_obj: PaddleOCR result object
            text_blocks: List to append text blocks to
        """
        try:
            # Try print() method
            if hasattr(result_obj, 'print'):
                try:
                    # print() might return a string or print to stdout
                    print_result = result_obj.print()
                    if print_result:
                        text = str(print_result)
                        if text.strip():
                            text_blocks.append(
                                TextBlock(
                                    text=text,
                                    confidence=1.0,
                                    bbox=None,  # No bbox in fallback
                                )
                            )
                            logger.debug("Extracted text using print() method")
                            return
                except Exception as e:
                    logger.debug(f"print() method failed: {e}")
            
            # Try save_to_markdown and read it
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    if hasattr(result_obj, 'save_to_markdown'):
                        result_obj.save_to_markdown(save_path=tmp_dir)
                        md_files = list(Path(tmp_dir).glob("*.md"))
                        if md_files:
                            with open(md_files[0], 'r', encoding='utf-8') as f:
                                text = f.read()
                            if text.strip():
                                text_blocks.append(
                                    TextBlock(
                                        text=text,
                                        confidence=1.0,
                                        bbox=None,
                                    )
                                )
                                logger.debug("Extracted text using save_to_markdown() method")
                                return
                except Exception as e:
                    logger.debug(f"save_to_markdown() method failed: {e}")
            
            # Last resort: string representation
            text = str(result_obj)
            if text.strip() and not text.startswith('<'):
                text_blocks.append(
                    TextBlock(
                        text=text,
                        confidence=1.0,
                        bbox=None,
                    )
                )
                logger.debug("Extracted text using string representation")
        except Exception as e:
            logger.warning(f"Fallback text extraction failed: {e}")
    
    def _extract_text_blocks_from_json(self, json_data: dict[str, Any]) -> list[TextBlock]:
        """
        Extract text blocks from PaddleOCR JSON output.
        
        Args:
            json_data: Parsed JSON data from PaddleOCR
            
        Returns:
            List of TextBlock objects
        """
        text_blocks: list[TextBlock] = []
        
        # PaddleOCR-VL JSON structure may vary
        # Primary format: parsing_res_list (PaddleOCR-VL document parsing)
        # Common patterns:
        # - Top-level "parsing_res_list" array (PaddleOCR-VL format)
        # - Top-level "text_blocks" or "blocks" array
        # - Each block has "text", "bbox" (coordinates), "confidence"
        # - May have separate arrays for tables, formulas, charts
        
        # Handle PaddleOCR-VL parsing_res_list format first
        if isinstance(json_data, dict) and "parsing_res_list" in json_data:
            parsing_list = json_data["parsing_res_list"]
            if isinstance(parsing_list, list):
                for block in parsing_list:
                    if not isinstance(block, dict):
                        continue
                    
                    # Extract content (PaddleOCR-VL uses block_content)
                    text = block.get("block_content", block.get("content", block.get("text", "")))
                    if not text:
                        continue
                    
                    # Extract confidence (may not be present, default to 1.0)
                    confidence = float(block.get("confidence", block.get("score", 1.0)))
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Extract bounding box (PaddleOCR-VL uses block_bbox)
                    bbox = None
                    bbox_data = block.get("block_bbox", block.get("bbox", block.get("box")))
                    if bbox_data:
                        try:
                            if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                                # Format: [x1, y1, x2, y2]
                                if isinstance(bbox_data[0], (int, float)):
                                    bbox = BBox(
                                        x1=int(bbox_data[0]),
                                        y1=int(bbox_data[1]),
                                        x2=int(bbox_data[2]),
                                        y2=int(bbox_data[3]),
                                    )
                                elif isinstance(bbox_data[0], (list, tuple)):
                                    # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    xs = [int(p[0]) for p in bbox_data if len(p) >= 2]
                                    ys = [int(p[1]) for p in bbox_data if len(p) >= 2]
                                    if xs and ys:
                                        bbox = BBox(
                                            x1=min(xs),
                                            y1=min(ys),
                                            x2=max(xs),
                                            y2=max(ys),
                                        )
                        except (ValueError, TypeError, KeyError) as e:
                            logger.debug(f"Failed to parse bbox: {e}")
                            bbox = None
                    
                    text_blocks.append(
                        TextBlock(
                            text=str(text),
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )
                
                logger.debug(f"Extracted {len(text_blocks)} blocks from parsing_res_list")
                return text_blocks
        
        def extract_from_array(arr: list[dict[str, Any]], element_type: str = "text") -> None:
            """Extract text blocks from an array of block dictionaries."""
            for block in arr:
                if not isinstance(block, dict):
                    continue
                
                # Extract text
                text = block.get("text", block.get("content", ""))
                if not text:
                    continue
                
                # Extract confidence
                confidence = float(block.get("confidence", block.get("score", 1.0)))
                confidence = max(0.0, min(1.0, confidence))
                
                # Extract bounding box
                bbox = None
                bbox_data = block.get("bbox", block.get("box", block.get("coordinates")))
                if bbox_data:
                    try:
                        if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                            # Format: [x1, y1, x2, y2] or [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            if isinstance(bbox_data[0], (int, float)):
                                # Simple [x1, y1, x2, y2] format
                                bbox = BBox(
                                    x1=int(bbox_data[0]),
                                    y1=int(bbox_data[1]),
                                    x2=int(bbox_data[2]),
                                    y2=int(bbox_data[3]),
                                )
                            elif isinstance(bbox_data[0], (list, tuple)):
                                # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                # Convert to axis-aligned bbox
                                xs = [int(p[0]) for p in bbox_data if len(p) >= 2]
                                ys = [int(p[1]) for p in bbox_data if len(p) >= 2]
                                if xs and ys:
                                    bbox = BBox(
                                        x1=min(xs),
                                        y1=min(ys),
                                        x2=max(xs),
                                        y2=max(ys),
                                    )
                        elif isinstance(bbox_data, dict):
                            # Dict format: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
                            bbox = BBox(
                                x1=int(bbox_data.get("x1", bbox_data.get("left", 0))),
                                y1=int(bbox_data.get("y1", bbox_data.get("top", 0))),
                                x2=int(bbox_data.get("x2", bbox_data.get("right", 0))),
                                y2=int(bbox_data.get("y2", bbox_data.get("bottom", 0))),
                            )
                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"Failed to parse bbox: {e}")
                        bbox = None
                
                text_blocks.append(
                    TextBlock(
                        text=str(text),
                        confidence=confidence,
                        bbox=bbox,
                    )
                )
        
        # Try common JSON structures
        if isinstance(json_data, dict):
            if "text_blocks" in json_data:
                extract_from_array(json_data["text_blocks"])
            elif "blocks" in json_data:
                extract_from_array(json_data["blocks"])
            elif "text" in json_data:
                # Single text block
                extract_from_array([json_data])
            elif "content" in json_data:
                # Content field
                content = json_data["content"]
                if isinstance(content, str):
                    text_blocks.append(
                        TextBlock(text=content, confidence=1.0, bbox=None)
                    )
                elif isinstance(content, list):
                    extract_from_array(content)
            else:
                # Try to find any array in the JSON
                for key, value in json_data.items():
                    if isinstance(value, list) and value:
                        # Check if it's an array of dicts with text
                        if isinstance(value[0], dict):
                            extract_from_array(value, element_type=key)
        elif isinstance(json_data, list):
            # Top-level array
            extract_from_array(json_data)
        elif isinstance(json_data, str):
            # Direct string
            if json_data.strip():
                text_blocks.append(
                    TextBlock(text=json_data, confidence=1.0, bbox=None)
                )
        
        return text_blocks
    
    def _normalize_element_type(self, block_label: str) -> str:
        """
        Normalize PaddleOCR block_label to standardized element type.
        
        Args:
            block_label: Raw block label from PaddleOCR
            
        Returns:
            Normalized element type
        """
        label_lower = block_label.lower() if block_label else ""
        
        # Map PaddleOCR labels to standardized types
        if label_lower in ("title", "heading", "head"):
            return "title"
        elif label_lower in ("paragraph", "text", "para"):
            return "paragraph"
        elif label_lower in ("table", "tbl"):
            return "table"
        elif label_lower in ("formula", "equation", "math"):
            return "formula"
        elif label_lower in ("chart", "figure", "image", "graph"):
            return "chart"
        elif label_lower == "header":
            return "header"
        elif label_lower == "footer":
            return "footer"
        elif label_lower in ("list", "item", "bullet"):
            return "list"
        else:
            return "text"  # Default fallback
    
    def _parse_paddleocr_structured_output(
        self, json_data: dict[str, Any]
    ) -> list[DocumentElement]:
        """
        Parse PaddleOCR-VL parsing_res_list into DocumentElement objects.
        
        Args:
            json_data: Parsed JSON data from PaddleOCR with parsing_res_list
            
        Returns:
            List of DocumentElement objects with all metadata
        """
        elements: list[DocumentElement] = []
        
        # Handle PaddleOCR-VL parsing_res_list format
        if isinstance(json_data, dict) and "parsing_res_list" in json_data:
            parsing_list = json_data["parsing_res_list"]
            if not isinstance(parsing_list, list):
                return elements
            
            for block in parsing_list:
                if not isinstance(block, dict):
                    continue
                
                # Extract content (PaddleOCR-VL uses block_content)
                content = block.get("block_content", block.get("content", block.get("text", "")))
                if not content:
                    continue
                
                # Extract element type from block_label
                block_label = block.get("block_label", block.get("type", "text"))
                element_type = self._normalize_element_type(block_label)
                
                # Extract element ID
                element_id = f"elem_{block.get('block_id', len(elements))}"
                
                # Extract confidence
                confidence = float(block.get("confidence", block.get("score", 1.0)))
                confidence = max(0.0, min(1.0, confidence))
                
                # Extract reading order
                reading_order = int(block.get("block_order", block.get("order", len(elements))))
                
                # Extract bounding box
                bbox = None
                bbox_data = block.get("block_bbox", block.get("bbox", block.get("box")))
                if bbox_data:
                    try:
                        if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                            if isinstance(bbox_data[0], (int, float)):
                                # Format: [x1, y1, x2, y2]
                                bbox = BBox(
                                    x1=int(bbox_data[0]),
                                    y1=int(bbox_data[1]),
                                    x2=int(bbox_data[2]),
                                    y2=int(bbox_data[3]),
                                )
                            elif isinstance(bbox_data[0], (list, tuple)):
                                # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                xs = [int(p[0]) for p in bbox_data if len(p) >= 2]
                                ys = [int(p[1]) for p in bbox_data if len(p) >= 2]
                                if xs and ys:
                                    bbox = BBox(
                                        x1=min(xs),
                                        y1=min(ys),
                                        x2=max(xs),
                                        y2=max(ys),
                                    )
                        elif isinstance(bbox_data, dict):
                            # Dict format
                            bbox = BBox(
                                x1=int(bbox_data.get("x1", bbox_data.get("left", 0))),
                                y1=int(bbox_data.get("y1", bbox_data.get("top", 0))),
                                x2=int(bbox_data.get("x2", bbox_data.get("right", 0))),
                                y2=int(bbox_data.get("y2", bbox_data.get("bottom", 0))),
                            )
                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"Failed to parse bbox: {e}")
                        bbox = None
                
                # Extract additional metadata
                metadata = {
                    "block_label": block_label,
                    "original_block_id": block.get("block_id"),
                }
                
                elements.append(
                    DocumentElement(
                        element_id=element_id,
                        element_type=element_type,
                        content=str(content),
                        bbox=bbox,
                        confidence=confidence,
                        reading_order=reading_order,
                        metadata=metadata,
                    )
                )
        
        logger.debug(f"Parsed {len(elements)} document elements from parsing_res_list")
        return elements
    
    def _group_elements_into_sections(
        self, elements: list[DocumentElement], image_height: int
    ) -> list[DocumentSection]:
        """
        Group document elements into logical sections.
        
        Args:
            elements: List of document elements
            image_height: Height of the image in pixels
            
        Returns:
            List of DocumentSection objects
        """
        if not elements:
            return []
        
        # Sort elements by reading order
        sorted_elements = sorted(elements, key=lambda e: e.reading_order)
        
        sections: list[DocumentSection] = []
        current_section_elements: list[DocumentElement] = []
        section_id_counter = 0
        
        # Threshold for vertical gap to start new section (20% of image height)
        vertical_gap_threshold = image_height * 0.2
        
        for i, element in enumerate(sorted_elements):
            if not element.bbox:
                # Elements without bbox go into current section
                current_section_elements.append(element)
                continue
            
            if not current_section_elements:
                # Start first section
                current_section_elements.append(element)
                continue
            
            # Check vertical gap from previous element
            prev_element = current_section_elements[-1]
            if prev_element.bbox:
                gap = element.bbox.y1 - prev_element.bbox.y2
                
                if gap > vertical_gap_threshold:
                    # Large gap - create new section
                    section = self._create_section(
                        current_section_elements, section_id_counter, image_height
                    )
                    sections.append(section)
                    section_id_counter += 1
                    current_section_elements = [element]
                else:
                    # Small gap - add to current section
                    current_section_elements.append(element)
            else:
                current_section_elements.append(element)
        
        # Add final section
        if current_section_elements:
            section = self._create_section(
                current_section_elements, section_id_counter, image_height
            )
            sections.append(section)
        
        logger.debug(f"Grouped {len(elements)} elements into {len(sections)} sections")
        return sections
    
    def _create_section(
        self, elements: list[DocumentElement], section_id: int, image_height: int
    ) -> DocumentSection:
        """
        Create a DocumentSection from a list of elements.
        
        Args:
            elements: List of elements for this section
            section_id: Unique section ID
            image_height: Height of the image
            
        Returns:
            DocumentSection object
        """
        # Calculate section bounding box (union of all element bboxes)
        section_bbox = None
        if elements and any(e.bbox for e in elements):
            bboxes = [e.bbox for e in elements if e.bbox]
            if bboxes:
                x1 = min(b.x1 for b in bboxes)
                y1 = min(b.y1 for b in bboxes)
                x2 = max(b.x2 for b in bboxes)
                y2 = max(b.y2 for b in bboxes)
                section_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        
        # Determine section type based on position
        section_type = "body"
        if section_bbox:
            y_center = (section_bbox.y1 + section_bbox.y2) / 2
            if y_center < image_height * 0.2:
                section_type = "header"
            elif y_center > image_height * 0.8:
                section_type = "footer"
        
        # Get reading order (minimum of element reading orders)
        reading_order = min((e.reading_order for e in elements), default=0)
        
        return DocumentSection(
            section_id=f"section_{section_id}",
            section_type=section_type,
            elements=elements,
            bbox=section_bbox,
            reading_order=reading_order,
        )
    
    def _serialize_raw_output(self, raw_result: list[Any]) -> dict[str, Any]:
        """
        Serialize raw PaddleOCR output for JSON response.
        
        Args:
            raw_result: Raw PaddleOCR result objects
            
        Returns:
            JSON-serializable dictionary
        """
        if not raw_result:
            return {}
        
        serialized: dict[str, Any] = {
            "paddleocr_results": [],
        }
        
        try:
            for result_obj in raw_result:
                # Try to get markdown representation
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        if hasattr(result_obj, 'save_to_markdown'):
                            result_obj.save_to_markdown(save_path=tmp_dir)
                            
                            # Find the markdown file in the directory
                            md_files = list(Path(tmp_dir).glob("*.md"))
                            if md_files:
                                md_path = md_files[0]
                                with open(md_path, 'r', encoding='utf-8') as f:
                                    markdown_content = f.read()
                                serialized["paddleocr_results"].append({
                                    "markdown": markdown_content,
                                })
                            else:
                                # Fallback to string representation
                                serialized["paddleocr_results"].append({
                                    "text": str(result_obj),
                                })
                        else:
                            # Fallback to string representation
                            serialized["paddleocr_results"].append({
                                "text": str(result_obj),
                            })
                    except Exception as e:
                        logger.debug(f"Failed to serialize result object: {e}")
                        serialized["paddleocr_results"].append({
                            "text": str(result_obj),
                            "error": str(e),
                        })
        except Exception as e:
            logger.warning(f"Failed to serialize PaddleOCR output: {e}")
            serialized["error"] = str(e)
        
        return serialized
    
    def get_info(self) -> dict[str, Any]:
        """Get engine information and status."""
        info = super().get_info()
        info.update({
            "mode": self._mode,
            "paddleocr_installed": self._pipeline is not None,
            "transformers_loaded": self._transformers_model is not None,
            "supports_element_recognition": self._transformers_model is not None,
        })
        return info

