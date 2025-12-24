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
from app.core.types import BBox, OCRMetadata, OCRResult, TextBlock
from app.engines.base import OCREngine

logger = logging.getLogger(__name__)


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
        Load the PaddleOCR-VL pipeline and transformers model.
        
        This method initializes:
        - PaddleOCRVL pipeline for document parsing
        - Transformers model for element-level recognition
        
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
        
        # Load transformers model for element-level recognition (optional)
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
        
        self._loaded = True
        logger.info("PaddleOCR-VL engine loaded successfully")
    
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
            
            # Build result
            return self._build_result(
                raw_result=raw_result,
                config=config,
                processing_time_ms=processing_time_ms,
                mode=mode,
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
        
        # Check if transformers model is available
        if not self._transformers_model or not self._transformers_processor:
            logger.warning(
                "Transformers model not loaded for element recognition. "
                "Falling back to document parsing."
            )
            return self._run_document_parsing(image_path)
        
        # Map prompt profiles to task-specific prompts
        # These match the HuggingFace demo prompts for element recognition
        PROMPTS = {
            "free_ocr": "OCR:",
            "markdown": "OCR:",  # Markdown is formatted OCR output
            "form": "OCR:",  # Forms are text extraction
            "table": "Table Recognition:",
        }
        
        # Note: For formula and chart recognition, you would use:
        # "Formula Recognition:" and "Chart Recognition:"
        # These can be added as new prompt profiles if needed
        
        task_prompt = PROMPTS.get(config.prompt_profile.value, "OCR:")
        logger.debug(f"Using task prompt: {task_prompt}")
        
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
            
            # Generate
            with torch.no_grad():
                outputs = self._transformers_model.generate(
                    **inputs,
                    max_new_tokens=1024,
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
            
            # Create a result object wrapper for consistent parsing
            # This mimics the PaddleOCR result object interface
            class ElementRecognitionResult:
                """Wrapper for element recognition results from transformers."""
                def __init__(self, text: str, task: str):
                    self.text = text
                    self.task = task
                    self._text_content = text.strip()
                    self._task_type = task.lower().replace(" recognition:", "").replace(":", "")
                
                def print(self) -> str:
                    """Return text representation."""
                    return self._text_content
                
                def save_to_json(self, save_path: str) -> None:
                    """Save result as JSON compatible with PaddleOCR format."""
                    import json
                    import os
                    # Create structure similar to PaddleOCR output
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
            
            result_obj = ElementRecognitionResult(generated_text, task_prompt)
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
    ) -> OCRResult:
        """
        Build OCRResult from PaddleOCR output.
        
        Args:
            raw_result: Raw output from PaddleOCR-VL
            config: OCR configuration
            processing_time_ms: Processing time in milliseconds
            mode: Operation mode used
            
        Returns:
            Structured OCRResult
        """
        # Build metadata
        metadata = OCRMetadata(
            engine="PaddleOCR-VL",
            model_variant=mode.value,
            processing_time_ms=processing_time_ms,
        )
        
        # Parse text blocks from PaddleOCR output
        text_blocks = self._parse_paddleocr_output(raw_result)
        
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
        # Common patterns:
        # - Top-level "text_blocks" or "blocks" array
        # - Each block has "text", "bbox" (coordinates), "confidence"
        # - May have separate arrays for tables, formulas, charts
        
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

