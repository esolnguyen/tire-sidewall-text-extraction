# Tire Information Extraction System

Automatically extract structured information (manufacturer, model, size, DOT code, etc.) from tire sidewall images using a multi-stage pipeline combining computer vision and LLM analysis.

## Architecture

```
Input Image
    |
    v
[YOLO Segmentation] --> Detect tire & rim contours
    |
    v
[Polar-to-Rectangular Flattening] --> Flatten circular sidewall to rectangular strip
    |
    v
[Image Preprocessing] --> Enhance contrast (CLAHE / Linear Stretching / Histogram EQ)
    |
    v
[YOLO Text Detection] --> Detect text regions with bounding boxes
    |
    v
[TRBA Text Recognition] --> Recognize text from cropped regions
    |
    v
[Gemini LLM Extraction] --> Extract structured tire info from OCR + image
    |
    v
Output JSON (Manufacturer, Model, Size, Load/Speed, DOT, Special Markings)
```

## Project Structure

```
src/
  main.py                    # Entry point (uvicorn server)
  config.py                  # Centralized configuration
  exceptions.py              # Custom exception hierarchy
  api/
    __init__.py              # FastAPI app, lifespan, endpoints
    schemas.py               # Pydantic response models
  pipeline/
    core.py                  # TireImageProcessingPipeline
    types.py                 # PipelineResult, TextDetectionResult
  models/
    text_detection.py        # YOLO text detector
    text_recognition.py      # TRBA text recognizer
    trba/                    # TRBA architecture internals
      model.py               # TPS-ResNet-BiLSTM-Attention
      modules.py             # TPS, ResNet, Grid generator
      tokenizer.py           # Character tokenizer
  schemas/
    tire_info.py             # TireInfo, FieldWithBBox
  services/
    gemini.py                # Gemini LLM service + prompts
  utils/
    image_preprocessing.py   # CLAHE, Linear Stretching, HE
    tire_cropping.py         # Tire detection & sidewall flattening
    cv_utils.py              # Contour/bbox helpers, JSON I/O
    metrics.py               # Image quality metrics
scripts/                     # Evaluation & analysis scripts
```

## Setup

### Prerequisites

- Python 3.10+
- Model weights in `models/` directory:
  - `tire_det.pt`: YOLO tire/rim segmentation
  - `yolov11n_oclip_asp_c5.pt`: YOLO text detection
  - `text_rec.pth`: TRBA text recognition

### Installation

```bash
pip install -r requirements.txt
```

Or with evaluation dependencies:

```bash
pip install -e ".[eval]"
```

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Start the API server

```bash
cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Or using the wrapper script:

```bash
cd src && python main.py --port 8000 --reload
```

### API Endpoints

#### `POST /extract`

Upload a tire image and extract information.

**Parameters:**
- `image` (file): Tire image (JPEG, PNG, HEIC)
- `mode` (form): `pipeline` (default) or `llm_only`

**Modes:**
- `pipeline`: Full pipeline - detection, flattening, OCR, then LLM extraction
- `llm_only`: Send raw image directly to Gemini LLM, no preprocessing

**Example:**

```bash
curl -X POST http://localhost:8000/extract \
  -F "image=@tire_photo.jpg" \
  -F "mode=pipeline"
```

**Response:**

```json
{
  "mode": "pipeline",
  "tire_info": {
    "manufacturer": {"value": "Michelin", "source_bboxes": [[120, 50, 580, 130]]},
    "model": {"value": "Pilot Sport 4S", "source_bboxes": [...]},
    "size": {"value": "245/35ZR20", "source_bboxes": [...]},
    "load_speed": {"value": "95Y", "source_bboxes": [...]},
    "dot": {"value": "1023", "source_bboxes": [...]},
    "special_markings": [{"value": "XL", "source_bboxes": [...]}]
  },
  "text_detections": [...],
  "output_files": {...}
}
```

#### `GET /health`

Check if the server and models are loaded.

### Interactive Docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

All configuration is centralized in `src/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `PREPROCESSING_METHOD` | `clahe` | Image enhancement: `clahe`, `linear`, `histogram_eq`, `none` |
| `FLATTEN_OUTPUT_HEIGHT` | `640` | Flattened sidewall image height |
| `FLATTEN_OUTPUT_WIDTH` | `6400` | Flattened sidewall image width |
| `CONF_THRESHOLD` | `0.2` | Text detection confidence threshold |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for LLM extraction |
| `DEVICE` | `cpu` | Set `USE_CUDA=true` env var for GPU |

## Evaluation Scripts

Located in `scripts/`:

```bash
# Evaluate pipeline against ground truth
cd scripts && python evaluate.py

# Evaluate YOLO text detection
cd scripts && python eval_yolo.py

# Compare preprocessing methods visually
cd scripts && python compare_preprocessing.py --image path/to/image.jpg
```
