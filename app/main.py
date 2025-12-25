"""
DocuMind - Intelligent OCR Platform

Main application entry point with FastAPI setup and lifecycle management.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

from app import __version__
from app.api.v1 import router as v1_router
from app.core.config import get_settings
from app.core.registry import get_registry
# Temporarily disabled for preprocessing-only mode
# from app.engines.deepseek_torch import DeepSeekTorchEngine
# Temporarily disabled - Qwen3VLForConditionalGeneration not available in transformers==4.55.0
# from app.engines.qwen_vl import QwenVLEngine


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# def load_ocr_engine() -> None:
#     """
#     Load and register the OCR engine.
    
#     This function loads the DeepSeek-OCR model and registers it
#     with the global engine registry.
    
#     Raises:
#         RuntimeError: If engine loading fails
#     """
#     logger = logging.getLogger(__name__)
#     settings = get_settings()
#     registry = get_registry()
    
#     logger.info("Initializing OCR engine...")
#     logger.info(f"Model path: {settings.model_path}")
#     logger.info(f"Device: {settings.device}")
    
#     # Create and load engine
#     engine = DeepSeekTorchEngine(settings)
    
#     try:
#         engine.load()
#     except FileNotFoundError as e:
#         logger.error(f"Model files not found: {e}")
#         logger.error(
#             "Please download the model or set DOCUMIND_MODEL_PATH "
#             "environment variable to point to the model directory."
#         )
#         raise
#     except Exception as e:
#         logger.error(f"Failed to load OCR engine: {e}")
#         raise RuntimeError(f"Engine initialization failed: {e}") from e
    
#     # Register engine
#     registry.register(engine.engine_name, engine, set_default=True)
    
#     logger.info(f"OCR engine '{engine.engine_name}' loaded and registered")


# Temporarily disabled - Qwen3VLForConditionalGeneration not available in transformers==4.55.0
# def load_qwen_engine() -> None:
#     """
#     Load and register the Qwen3-VL OCR engine.
#     
#     This function loads the Qwen3-VL-2B-Instruct model from HuggingFace
#     and registers it with the global engine registry.
#     
#     Raises:
#         RuntimeError: If engine loading fails
#     """
#     logger = logging.getLogger(__name__)
#     settings = get_settings()
#     registry = get_registry()
#     
#     logger.info("Initializing Qwen3-VL OCR engine...")
#     logger.info(f"Model ID: {settings.qwen_model_id}")
#     logger.info("Device: CPU (optimized for local testing)")
#     
#     # Create and load engine
#     engine = QwenVLEngine(settings)
#     
#     try:
#         engine.load()
#     except Exception as e:
#         logger.error(f"Failed to load Qwen OCR engine: {e}")
#         raise RuntimeError(f"Qwen engine initialization failed: {e}") from e
#     
#     # Register engine as default (since it's the primary engine for now)
#     registry.register(engine.engine_name, engine, set_default=True)
#     
#     logger.info(f"Qwen OCR engine '{engine.engine_name}' loaded and registered")


def load_paddleocr_engine() -> None:
    """
    Load and register the PaddleOCR-VL OCR engine.
    
    This function loads the PaddleOCR-VL pipeline and registers it
    with the global engine registry. This is optional and won't
    fail startup if PaddleOCR is not available.
    
    Raises:
        RuntimeError: If engine loading fails (but caught and logged)
    """
    logger = logging.getLogger(__name__)
    settings = get_settings()
    registry = get_registry()
    
    logger.info("Initializing PaddleOCR-VL OCR engine...")
    logger.info(f"Default mode: {settings.paddleocr_mode}")
    logger.info("Device: CPU (CPU-only deployment)")
    
    try:
        from app.engines.paddleocr_vl import PaddleOCRVLEngine
        
        # Create and load engine
        engine = PaddleOCRVLEngine(settings)
        engine.load()
        
        # Register engine (set as default since Qwen is disabled)
        registry.register(engine.engine_name, engine, set_default=True)
        
        logger.info(f"PaddleOCR-VL engine '{engine.engine_name}' loaded and registered")
    except ImportError as e:
        logger.warning(
            f"PaddleOCR-VL not available (PaddleOCR not installed): {e}. "
            "Install with: pip install paddlepaddle==3.2.1 && pip install -U 'paddleocr[doc-parser]'"
        )
    except Exception as e:
        logger.warning(f"Failed to load PaddleOCR-VL engine: {e}. Continuing without it.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown operations:
    - Startup: Load OCR model, register engine
    - Shutdown: Cleanup resources
    """
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info(f"Starting DocuMind v{__version__}")
    
    # Load Qwen3-VL OCR engine
    # try:
    #     load_qwen_engine()
    # except Exception as e:
    #     logger.critical(f"Startup failed: {e}")
    #     # Re-raise to prevent app from starting in broken state
    #     raise
    
    # # Load PaddleOCR-VL engine (optional, won't fail startup)
    # try:
    #     load_paddleocr_engine()
    # except Exception as e:
    #     logger.warning(f"Failed to load PaddleOCR-VL engine: {e}")
    
    # Skip DeepSeek OCR engine loading for now
    # try:
    #     load_ocr_engine()
    # except Exception as e:
    #     logger.warning(f"Failed to load DeepSeek engine: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DocuMind...")
    
    # Clear registry (cleanup)
    registry = get_registry()
    registry.clear()
    
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    setup_logging()
    settings = get_settings()
    
    app = FastAPI(
        title="DocuMind",
        description="Intelligent OCR Platform - Self-hosted document understanding",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_type": "internal_error",
            },
        )
    
    # Include routers
    app.include_router(v1_router)
    
    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # UI endpoint
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def ui(request: Request) -> HTMLResponse:
        """Serve the OCR UI interface."""
        return templates.TemplateResponse("index.html", {"request": request})
    
    # API info endpoint
    @app.get("/api", include_in_schema=False)
    async def api_info() -> dict[str, str]:
        """API information endpoint."""
        return {
            "service": "DocuMind",
            "version": __version__,
            "docs": "/docs",
            "ui": "/",
        }
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=True,  # Disable reload in production
        log_level=settings.log_level.lower(),
    )

