"""
Semantic understanding pipeline step.

This step uses Ollama + Gemma2B to analyze extracted OCR text and
layout information to generate semantic insights.
"""

import json
import logging
from typing import Any

import httpx

from app.core.config import AppSettings, get_settings
from app.pipeline.context import PipelineContext
from app.pipeline.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class SemanticUnderstandingStep(PipelineStep):
    """
    Pipeline step that performs semantic analysis using Ollama + Gemma2B.
    
    This step:
    1. Takes OCR results from previous steps
    2. Formats text blocks and bounding boxes into a prompt
    3. Calls Ollama API with Gemma2B model
    4. Stores semantic insights in context
    """
    
    def __init__(self, settings: AppSettings | None = None) -> None:
        """
        Initialize the semantic understanding step.
        
        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._ollama_url = self._settings.ollama_base_url
        self._ollama_model = self._settings.ollama_model
        self._client: httpx.AsyncClient | None = None
    
    @property
    def name(self) -> str:
        """Get step name."""
        return "semantic"
    
    def should_run(self, context: PipelineContext) -> bool:
        """
        Determine if semantic step should run.
        
        The step runs if:
        - Semantic understanding is enabled in config
        - OCR result exists and has text blocks
        - No critical errors occurred
        
        Args:
            context: Pipeline context
            
        Returns:
            True if step should execute, False to skip
        """
        # Check if semantic understanding is enabled
        if not context.config.semantic.enable:
            return False
        
        # Check if OCR result exists
        if not context.has_result:
            logger.debug("Skipping semantic step: No OCR result available")
            return False
        
        result = context.result
        if not result or not result.text_blocks:
            logger.debug("Skipping semantic step: No text blocks to analyze")
            return False
        
        # Check for critical errors
        if result.has_errors and len(result.text_blocks) == 0:
            logger.debug("Skipping semantic step: OCR failed")
            return False
        
        return True
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute semantic analysis on OCR results.
        
        Args:
            context: Pipeline context with OCR result
            
        Returns:
            Context with semantic insights added
        """
        logger.debug("Running semantic understanding step")
        
        if not context.has_result:
            logger.warning("No OCR result available for semantic analysis")
            return context
        
        result = context.result
        
        try:
            # Build semantic prompt from OCR results
            prompt = self._build_semantic_prompt(result, context.config)
            
            # Call Ollama API
            semantic_response = self._call_ollama(prompt)
            
            # Parse and store semantic insights
            semantic_data = self._parse_semantic_response(semantic_response)
            
            # Store in context intermediate
            context.set_intermediate("semantic_understanding", semantic_data)
            
            # Add to OCR result raw output
            # Note: OCRResult.raw_output is a dict, so we can modify it
            if result.raw_output is None:
                result.raw_output = {}
            result.raw_output["semantic_understanding"] = semantic_data
            
            logger.debug("Semantic understanding step completed successfully")
            
        except Exception as e:
            error_msg = f"Semantic understanding failed: {e}"
            logger.warning(error_msg, exc_info=True)
            # Don't fail the pipeline, just log the error
            context.add_error(error_msg)
        
        finally:
            context.add_step(self.name)
        
        return context
    
    def _build_semantic_prompt(
        self, result: Any, config: Any
    ) -> str:
        """
        Build prompt for semantic analysis from OCR results.
        
        Args:
            result: OCRResult object
            config: OCRConfig object
            
        Returns:
            Formatted prompt string
        """
        # Use custom prompt if provided, otherwise use default
        if config.semantic.prompt:
            base_prompt = config.semantic.prompt
        else:
            base_prompt = (
                "Analyze the following document content extracted via OCR and extract all information "
                "in a standard JSON format. The output must be valid JSON only, no additional text.\n\n"
                "Required JSON structure:\n"
                "{\n"
                '  "metadata": {\n'
                '    "document_type": "type of document (e.g., invoice, form, letter, receipt, etc.)",\n'
                '    "language": "detected language",\n'
                '    "page_count": number of pages if available,\n'
                '    "processing_date": "ISO 8601 date/time",\n'
                '    "confidence_score": 0.0-1.0 based on extraction quality\n'
                "  },\n"
                '  "summary": "brief summary of the document content",\n'
                '  "key_information": {\n'
                '    "dates": ["list of all dates found"],\n'
                '    "names": ["list of all person/organization names"],\n'
                '    "amounts": ["list of monetary amounts"],\n'
                '    "locations": ["list of addresses or locations"],\n'
                '    "identifiers": ["list of IDs, reference numbers, etc."]\n'
                "  },\n"
                '  "extracted_data": {\n'
                '    "key1": "value1",\n'
                '    "key2": "value2",\n'
                '    ...\n'
                "  },\n"
                '  "structure": {\n'
                '    "sections": ["list of document sections identified"],\n'
                '    "layout_type": "description of document layout"\n'
                "  }\n"
                "}\n\n"
                "Extract all relevant information from the document and structure it as key-value pairs in the "
                '"extracted_data" field. Include all important details such as names, dates, amounts, addresses, '
                "reference numbers, and any other structured data found in the document. "
                "Ensure the JSON is valid and complete."
            )
        
        # Format OCR text blocks
        text_content = []
        for i, block in enumerate(result.text_blocks, 1):
            text = block.text
            if block.bbox:
                text_content.append(
                    f"Block {i} (bbox: {block.bbox.x1},{block.bbox.y1}-{block.bbox.x2},{block.bbox.y2}): {text}"
                )
            else:
                text_content.append(f"Block {i}: {text}")
        
        full_text = "\n".join(text_content)
        
        # Combine prompt and content
        prompt = f"{base_prompt}\n\nDocument Content:\n{full_text}"
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        """
        Call Ollama API for semantic analysis.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Response dictionary from Ollama API
            
        Raises:
            httpx.RequestError: If API call fails
        """
        api_url = f"{self._ollama_url}/api/generate"
        
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        
        try:
            # Use sync client for now (can be made async if needed)
            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to connect to Ollama at {self._ollama_url}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API returned error: {e}")
            raise RuntimeError(f"Ollama API error: {e.response.status_code}") from e
    
    def _parse_semantic_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Parse semantic response from Ollama.
        
        Args:
            response: Raw response from Ollama API
            
        Returns:
            Structured semantic data dictionary
        """
        # Ollama API returns response in format:
        # {"model": "...", "created_at": "...", "response": "...", "done": true, ...}
        semantic_text = response.get("response", "").strip()
        
        # Initialize semantic data structure
        semantic_data: dict[str, Any] = {
            "model": response.get("model", self._ollama_model),
            "raw_response": semantic_text,
        }
        
        # Try to extract JSON from the response
        # The LLM might return JSON wrapped in markdown code blocks or plain JSON
        json_text = semantic_text
        
        # Remove markdown code blocks if present
        if "```json" in json_text:
            start = json_text.find("```json") + 7
            end = json_text.find("```", start)
            if end != -1:
                json_text = json_text[start:end].strip()
        elif "```" in json_text:
            start = json_text.find("```") + 3
            end = json_text.find("```", start)
            if end != -1:
                json_text = json_text[start:end].strip()
        
        # Try to find JSON object in the text
        if "{" in json_text:
            # Find the first { and try to extract complete JSON
            start_idx = json_text.find("{")
            if start_idx != -1:
                # Try to find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(json_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_text = json_text[start_idx:end_idx]
        
        # Try to parse as JSON
        if json_text.strip().startswith("{"):
            try:
                parsed = json.loads(json_text)
                # If parsing successful, use the structured data as primary output
                semantic_data.update(parsed)
                semantic_data["structured"] = True
                logger.debug("Successfully parsed structured JSON from semantic response")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from semantic response: {e}")
                # Fallback: store as text analysis
                semantic_data["structured"] = False
                semantic_data["analysis"] = semantic_text
        else:
            # No JSON found, store as text analysis
            semantic_data["structured"] = False
            semantic_data["analysis"] = semantic_text
        
        return semantic_data

