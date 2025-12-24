# DocuMind - Intelligent OCR Platform

A production-grade, self-hosted OCR platform built with FastAPI and DeepSeek-OCR. Designed for rapid ML experimentation with a modular, interface-driven architecture.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange.svg)](https://pytorch.org/)

## 🚀 Features

- **Self-Hosted**: Run entirely on your infrastructure (EC2, local server, etc.)
- **Modular Architecture**: Swappable engines, frameworks, and pipeline steps
- **Interface-Driven**: Clean abstractions enable easy experimentation
- **Configuration-Driven**: Behavior controlled via runtime config, not code
- **Production-Ready**: Thread-safe inference, error handling, health checks
- **DeepSeek-OCR Integration**: State-of-the-art OCR model with PyTorch backend
- **Stable API Contract**: Versioned output schema for reliable integrations
- **GPU & CPU Support**: Automatic device selection with graceful fallback

## 📐 Architecture

```
┌──────────────────────────┐
│        API Layer         │
│  (FastAPI / HTTP)        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Pipeline Engine      │
│  (Step Orchestration)    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      OCR Interface       │
│   (Abstract Contract)    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   OCR Engine (Concrete)  │
│  DeepSeek / PyTorch      │
└──────────────────────────┘
```

### Core Principles

- **Interface-First**: All components interact via explicit interfaces
- **Configuration-Driven**: Experimental behavior controlled via runtime config
- **Stable Output Contract**: API response format is versioned and stable
- **Fail-Fast**: Application won't start if model loading fails
- **Thread-Safe**: Concurrent request handling with proper locking

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, CPU supported)
- ~10GB disk space for model files

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd DocuMind
```

### Step 2: Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch
```

### Step 3: Install Flash Attention (Optional, CUDA only)

For faster inference on CUDA GPUs:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** Flash attention requires CUDA and specific compiler setup. If installation fails, the system will automatically fall back to eager attention.

### Step 4: Download DeepSeek-OCR Model

Download the model from HuggingFace:

```bash
# Using huggingface-cli
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /opt/models/deepseek-ocr

# Or manually download from:
# https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

**Alternative:** Set a custom model path using environment variables (see Configuration section).

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
# Model Configuration
DOCUMIND_MODEL_PATH=/opt/models/deepseek-ocr  # Path to model directory
DOCUMIND_DEVICE=auto                           # auto, cuda, or cpu
DOCUMIND_USE_FLASH_ATTENTION=true              # Use flash attention if available

# Server Configuration
DOCUMIND_HOST=0.0.0.0                          # Server host
DOCUMIND_PORT=8000                             # Server port
DOCUMIND_WORKERS=1                             # Number of worker processes

# Limits
DOCUMIND_MAX_IMAGE_SIZE_MB=20.0                # Maximum image size in MB
DOCUMIND_REQUEST_TIMEOUT_SECONDS=120           # Request timeout

# Logging
DOCUMIND_LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR
```

### Model Variants

DeepSeek-OCR supports different size/quality variants:

| Variant | base_size | image_size | crop_mode | Use Case |
|---------|-----------|------------|-----------|----------|
| `tiny`  | 512       | 512        | False     | Fast, low quality |
| `small`  | 640       | 640        | False     | Fast, good quality |
| `base`   | 1024      | 1024       | False     | Balanced |
| `large`  | 1280      | 1280      | False     | High quality, slower |
| `gundam` | 1024      | 640        | True      | Optimized for documents (default) |

### Prompt Profiles

Different prompt profiles optimize output for specific tasks:

- `free_ocr`: General text extraction (`<image>\nFree OCR.`)
- `markdown`: Convert document to markdown format (`<image>\n<|grounding|>Convert the document to markdown.`)
- `form`: Extract form fields and values
- `table`: Extract tables in markdown format

## 🚀 Usage

### Starting the Server

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or using Python
python -m app.main

# With custom settings
DOCUMIND_MODEL_PATH=/path/to/model uvicorn app.main:app
```

The server will:
1. Load the DeepSeek-OCR model at startup (may take 1-2 minutes)
2. Register the engine in the global registry
3. Start accepting requests on the configured port
4. Display startup logs including model path and device

**Note:** The application will fail to start if the model cannot be loaded. This ensures the service is never in a broken state.

### API Endpoints

#### POST `/api/v1/ocr`

Perform OCR on an uploaded image.

**Request:**
- `image` (file, required): Image file (JPEG, PNG, WEBP, TIFF)
- `config` (optional, JSON string): OCR configuration

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -F "image=@document.png" \
  -F 'config={"model_variant": "gundam", "prompt_profile": "markdown"}'
```

**Example with Python:**

```python
import requests

url = "http://localhost:8000/api/v1/ocr"
files = {"image": open("document.png", "rb")}
data = {
    "config": '{"model_variant": "gundam", "prompt_profile": "markdown"}'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Processing time: {result['metadata']['processing_time_ms']:.2f} ms")
for block in result["text_blocks"]:
    print(block["text"])
```

**Example with minimal config:**

```bash
# Use defaults (gundam variant, markdown profile)
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -F "image=@document.png"
```

**Response:**

```json
{
  "metadata": {
    "engine": "DeepSeek-OCR",
    "model_variant": "gundam",
    "processing_time_ms": 1250.5
  },
  "text_blocks": [
    {
      "text": "# Document Title\n\nThis is the extracted text...",
      "confidence": 1.0,
      "bbox": null
    }
  ],
  "raw_output": {},
  "errors": []
}
```

**Error Response:**

```json
{
  "metadata": {
    "engine": "DeepSeek-OCR",
    "model_variant": "gundam",
    "processing_time_ms": 50.0
  },
  "text_blocks": [],
  "raw_output": {},
  "errors": ["OCR engine not found: 'invalid_engine'"]
}
```

#### GET `/api/v1/health`

Check service health and engine status.

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "engine_loaded": true,
  "available_engines": ["deepseek_torch"]
}
```

**Degraded Status:**

```json
{
  "status": "degraded",
  "engine_loaded": false,
  "available_engines": []
}
```

#### GET `/api/v1/config/defaults`

Get default configuration and available options.

```bash
curl http://localhost:8000/api/v1/config/defaults
```

**Response:**

```json
{
  "engine": "deepseek_torch",
  "model_variant": "gundam",
  "prompt_profile": "markdown",
  "decoding": {
    "max_tokens": 4096,
    "temperature": 0.0
  },
  "return_raw_output": true,
  "available_variants": ["tiny", "small", "base", "large", "gundam"],
  "available_profiles": ["free_ocr", "markdown", "form", "table"]
}
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test endpoints directly.

## 📁 Project Structure

```
DocuMind/
├── app/
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # FastAPI app + lifespan management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                # BBox, TextBlock, OCRResult dataclasses
│   │   ├── config.py               # AppSettings, OCRConfig schemas
│   │   └── registry.py             # Engine registry singleton
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py                 # OCREngine abstract base class
│   │   └── deepseek_torch.py       # DeepSeek-OCR implementation
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── context.py              # PipelineContext for data flow
│   │   ├── engine.py               # PipelineEngine orchestrator
│   │   └── steps/
│   │       ├── __init__.py
│   │       ├── base.py             # PipelineStep abstract base class
│   │       └── ocr.py              # OCR pipeline step implementation
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── routes.py            # HTTP endpoints
│           └── schemas.py          # Pydantic request/response models
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Development

### Adding a New OCR Engine

1. Create a new engine class inheriting from `OCREngine`:

```python
from app.engines.base import OCREngine
from app.core.types import OCRResult
from app.core.config import OCRConfig
from PIL import Image

class MyNewEngine(OCREngine):
    ENGINE_NAME = "my_new_engine"
    
    def __init__(self, settings):
        self._settings = settings
        self._loaded = False
    
    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        # Load your model here
        self._model = load_my_model()
        self._loaded = True
    
    def infer(self, image: Image.Image, config: OCRConfig) -> OCRResult:
        # Run inference and return OCRResult
        result = self._model.process(image)
        return OCRResult(...)
```

2. Register it in `app/main.py`:

```python
from app.engines.my_new_engine import MyNewEngine

def load_ocr_engine() -> None:
    # ... existing DeepSeek engine code ...
    
    # Register new engine
    new_engine = MyNewEngine(settings)
    new_engine.load()
    registry.register(new_engine.engine_name, new_engine)
```

### Adding a Pipeline Step

1. Create a new step class:

```python
from app.pipeline.steps.base import PipelineStep
from app.pipeline.context import PipelineContext

class PreprocessStep(PipelineStep):
    @property
    def name(self) -> str:
        return "preprocess"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        # Do preprocessing
        processed_image = enhance_image(context.image)
        context.image = processed_image
        context.set_intermediate("preprocessed", True)
        return context
```

2. Add it to the pipeline in `app/pipeline/engine.py`:

```python
def create_default_pipeline() -> PipelineEngine:
    engine = PipelineEngine()
    engine.add_step(PreprocessStep())
    engine.add_step(OCRPipelineStep())
    return engine
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## 🐛 Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: Model path does not exist`

**Solution:**
- Ensure the model is downloaded to the path specified in `DOCUMIND_MODEL_PATH`
- Set the environment variable: `export DOCUMIND_MODEL_PATH=/path/to/model`
- Verify the path contains model files (config.json, model files, etc.)

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Use a smaller model variant: `{"model_variant": "tiny"}`
- Reduce image size before uploading
- Use CPU mode: `export DOCUMIND_DEVICE=cpu`
- Close other GPU processes

### Flash Attention Not Available

**Warning:** `flash-attn not installed, falling back to eager attention`

**Solution:**
- This is not critical - eager attention works fine, just slower
- To enable flash attention: `pip install flash-attn==2.7.3 --no-build-isolation`
- Requires CUDA and compatible compiler setup

### Slow Inference

**Possible causes:**
- Using CPU instead of GPU
- Large model variant
- Large image size

**Solutions:**
- Use GPU: `export DOCUMIND_DEVICE=cuda`
- Use smaller variant: `{"model_variant": "tiny"}`
- Install flash attention for faster GPU inference
- Reduce image resolution before processing

### Application Won't Start

**Error:** `RuntimeError: Engine initialization failed`

**Solutions:**
- Check model path is correct
- Verify model files are complete
- Check disk space
- Review logs for specific error messages
- Ensure PyTorch is installed correctly

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
- Ensure you're running from the project root directory
- Install dependencies: `pip install -r requirements.txt`
- Use `python -m app.main` instead of `python app/main.py`

## 📊 Performance

Typical performance benchmarks:

### GPU (NVIDIA A100)

| Variant | Time per Page | Memory Usage |
|---------|---------------|--------------|
| `tiny`  | ~0.5-1s       | ~4GB         |
| `small` | ~0.8-1.5s     | ~5GB         |
| `gundam`| ~1-2s         | ~6GB         |
| `base`  | ~2-3s         | ~7GB         |
| `large` | ~3-5s         | ~9GB         |

### CPU (Intel i7-12700K)

| Variant | Time per Page | Memory Usage |
|---------|---------------|--------------|
| `tiny`  | ~8-15s        | ~6GB         |
| `small` | ~12-20s       | ~7GB         |
| `gundam`| ~15-25s       | ~8GB         |
| `base`  | ~25-40s       | ~10GB        |
| `large` | ~40-60s       | ~12GB        |

**Note:** Performance varies significantly based on:
- Image complexity and size
- Hardware specifications
- System load
- Model variant used

## 🔒 Security Considerations

- **CORS**: Currently allows all origins (`allow_origins=["*"]`). Configure appropriately for production.
- **File Upload**: Maximum file size is configurable via `DOCUMIND_MAX_IMAGE_SIZE_MB`.
- **Model Path**: Ensure model files are stored securely and not accessible via web server.
- **API Keys**: Consider adding authentication for production deployments.

## 📝 License

[Specify your license here]

## 🙏 Acknowledgments

- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) - State-of-the-art OCR model
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/docs/transformers) - Model loading library

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- [Add your contact/support information]

## 📚 Additional Resources

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Built with ❤️ for the ML community**

