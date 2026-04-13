"""FastAPI application for tire information extraction."""

import os
import uuid
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from config import TireExtractionConfig
from pipeline.core import TireImageProcessingPipeline
from pipeline.types import TextDetectionResult
from schemas.tire_info import TireInfo
from exceptions import TireExtractionError, ImageLoadError
from api.schemas import (
    TextDetectionResponse,
    FieldWithBBoxResponse,
    TireInfoResponse,
    ExtractionMode,
    ExtractionResponse,
)

logger = logging.getLogger(__name__)

# ── Globals (set during API lifespan) ─────────────────────────────────────
pipeline: Optional[TireImageProcessingPipeline] = None
config: Optional[TireExtractionConfig] = None

UPLOAD_DIR = os.path.join("data", "uploads")
OUTPUT_DIR = os.path.join("data", "output")

# Ensure directories exist before StaticFiles mount
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, config
    config = TireExtractionConfig()
    logger.info("Loading pipeline models (this may take a moment)...")
    pipeline = TireImageProcessingPipeline(config)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Tire Information Extraction API",
    description="Upload a tire image to extract manufacturer, model, size, DOT and more.",
    lifespan=lifespan,
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ── Helpers ───────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic"}


def _field(f) -> FieldWithBBoxResponse:
    return FieldWithBBoxResponse(value=f.value, source_bboxes=f.source_bboxes)


def _tire_info_response(ti: TireInfo) -> TireInfoResponse:
    return TireInfoResponse(
        manufacturer=_field(ti.manufacturer),
        model=_field(ti.model),
        size=_field(ti.size),
        load_speed=_field(ti.load_speed),
        dot=_field(ti.dot),
        special_markings=[_field(m) for m in ti.special_markings],
    )


def _to_output_urls(output_files: Dict[str, str]) -> Dict[str, str]:
    url_files: Dict[str, str] = {}
    for key, abs_path in output_files.items():
        rel = os.path.relpath(abs_path, OUTPUT_DIR)
        url_files[key] = f"/outputs/{rel}"
    return url_files


def _text_detections_response(
    detections: List[TextDetectionResult],
) -> List[TextDetectionResponse]:
    return [
        TextDetectionResponse(text=d.text, bbox=d.bbox, confidence=d.confidence)
        for d in detections
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/extract")
async def extract_tire_info(
    image: UploadFile = File(...),
    mode: ExtractionMode = Form(ExtractionMode.pipeline),
):
    """Upload a tire image and get extracted information.

    **mode**:
    - `pipeline` — full pipeline: detection, flattening, OCR, then LLM.
    - `llm_only` — forward the raw image directly to LLM, no preprocessing.
    """
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex[:12]
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    try:
        contents = await image.read()
        with open(upload_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    job_output_dir = os.path.join(OUTPUT_DIR, job_id)

    try:
        if mode == ExtractionMode.llm_only:
            ti = pipeline.run_llm_only(upload_path)
            return ExtractionResponse(
                mode=mode.value,
                tire_info=_tire_info_response(ti),
                text_detections=[],
                output_files={},
            )

        result = pipeline.run_pipeline(
            upload_path, output_dir=job_output_dir, save_debug=True
        )
        return ExtractionResponse(
            mode=mode.value,
            tire_info=_tire_info_response(result.tire_info),
            text_detections=_text_detections_response(result.text_detections),
            output_files=_to_output_urls(result.output_files),
        )

    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TireExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": pipeline is not None}
