"""Streamlit application for Tire Information Extraction."""

import os
import sys
import tempfile

# Ensure working directory is project root so relative model paths resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import logging

import cv2
import numpy as np
import streamlit as st

from config import TireExtractionConfig
from pipeline.core import TireImageProcessingPipeline
from pipeline.types import PipelineResult
from schemas.tire_info import TireInfo

# Configure logging so pipeline logs appear in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)


class StreamlitLogHandler(logging.Handler):
    """Captures log records into a list for display in the UI."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


@st.cache_resource
def load_pipeline():
    """Load the pipeline once and cache it across reruns."""
    config = TireExtractionConfig()
    return TireImageProcessingPipeline(config)


def display_tire_info(tire_info: TireInfo):
    """Display extracted tire information in a structured layout."""
    st.subheader("Extracted Tire Information")

    fields = [
        ("Manufacturer", tire_info.manufacturer),
        ("Model", tire_info.model),
        ("Size", tire_info.size),
        ("Load/Speed", tire_info.load_speed),
        ("DOT", tire_info.dot),
    ]

    cols = st.columns(len(fields))
    for col, (label, field) in zip(cols, fields):
        col.metric(label, field.value if field.value else "N/A")

    if tire_info.special_markings:
        markings = ", ".join(m.value for m in tire_info.special_markings if m.value)
        st.metric("Special Markings", markings if markings else "None")

    # Detailed table with bounding box info
    with st.expander("Field Details (with bounding boxes)"):
        for label, field in fields:
            bbox_str = str(field.source_bboxes) if field.source_bboxes else "N/A"
            st.markdown(f"**{label}**: `{field.value}` — bboxes: `{bbox_str}`")
        for i, m in enumerate(tire_info.special_markings):
            bbox_str = str(m.source_bboxes) if m.source_bboxes else "N/A"
            st.markdown(f"**Marking {i+1}**: `{m.value}` — bboxes: `{bbox_str}`")


def display_text_detections(result: PipelineResult):
    """Display OCR text detection results."""
    if not result.text_detections:
        return

    st.subheader("Text Detections (OCR)")
    data = [
        {
            "Text": td.text,
            "Confidence": f"{td.confidence:.4f}",
            "BBox (x1,y1,x2,y2)": str(td.bbox),
        }
        for td in result.text_detections
    ]
    st.dataframe(data, width="stretch")


def display_output_images(output_files: dict):
    """Display all intermediate pipeline images."""
    if not output_files:
        return

    st.subheader("Pipeline Output Images")

    image_keys = [
        ("tire_rim_detection", "Tire & Rim Detection"),
        ("flattened", "Flattened Sidewall"),
        ("preprocessed", "Preprocessed Image"),
        ("detections_preprocessed", "Text Detections (Preprocessed)"),
        ("detections_flattened", "Text Detections (Flattened)"),
    ]

    for key, title in image_keys:
        if key in output_files and os.path.exists(output_files[key]):
            st.markdown(f"**{title}**")
            img = cv2.imread(output_files[key])
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")

    # Show individual text crops if available
    crop_dir = output_files.get("text_crops_dir")
    if crop_dir and os.path.isdir(crop_dir):
        crops = sorted(
            f for f in os.listdir(crop_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        if crops:
            with st.expander(f"Text Crops ({len(crops)} regions)"):
                cols = st.columns(min(len(crops), 5))
                for i, crop_file in enumerate(crops):
                    col = cols[i % len(cols)]
                    crop_img = cv2.imread(os.path.join(crop_dir, crop_file))
                    if crop_img is not None:
                        col.image(
                            cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB),
                            caption=crop_file,
                            width="stretch",
                        )


def main():
    st.set_page_config(page_title="Tire Info Extractor", layout="wide")
    st.title("Tire Information Extraction")

    # Sidebar controls
    mode = st.sidebar.radio(
        "Extraction Mode",
        options=["Pipeline", "LLM Only"],
        help=(
            "**Pipeline**: Detect tire -> flatten -> preprocess -> OCR -> LLM extraction.\n\n"
            "**LLM Only**: Send raw image directly to Gemini LLM."
        ),
    )

    save_debug = False
    if mode == "Pipeline":
        save_debug = st.sidebar.checkbox("Save debug crops", value=True)

    uploaded_file = st.file_uploader(
        "Upload a tire image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "heic", "heif"],
    )

    if uploaded_file is None:
        st.info("Upload a tire image to get started.")
        return

    # Show uploaded image
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_img is None:
        st.error("Failed to decode the uploaded image.")
        return

    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width="stretch")

    if not st.button("Extract", type="primary", width="stretch"):
        return

    # Save uploaded file to a temp path so the pipeline can load it
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Attach a handler to capture logs for the UI
    log_handler = StreamlitLogHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        with st.spinner("Loading models..."):
            pipeline = load_pipeline()

        if mode == "LLM Only":
            with st.spinner("Sending image to Gemini LLM..."):
                tire_info = pipeline.run_llm_only(tmp_path)
            display_tire_info(tire_info)
        else:
            with st.spinner("Running full pipeline (this may take a moment)..."):
                result = pipeline.run_pipeline(
                    tmp_path, save_debug=save_debug,
                )
            display_tire_info(result.tire_info)
            st.divider()
            display_text_detections(result)
            st.divider()
            display_output_images(result.output_files)

    except Exception as e:
        st.error(f"Extraction failed: {e}")
    finally:
        os.unlink(tmp_path)
        root_logger.removeHandler(log_handler)

    # Display captured logs
    if log_handler.records:
        with st.expander("Logs", expanded=True):
            st.code("\n".join(log_handler.records), language="log")


if __name__ == "__main__":
    main()
