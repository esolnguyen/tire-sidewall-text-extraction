"""Streamlit application for Tire Information Extraction."""

import os
import sys
import tempfile

# Ensure working directory is project root so relative model paths resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import io
import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from config import TireExtractionConfig
from pipeline.core import TireImageProcessingPipeline
from pipeline.types import TextDetectionResult
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


def display_text_detections(text_detections: list[TextDetectionResult]):
    """Display OCR text detection results."""
    if not text_detections:
        return

    st.subheader("Text Detections (OCR)")
    data = [
        {
            "Text": td.text,
            "Confidence": f"{td.confidence:.4f}",
            "BBox (x1,y1,x2,y2)": str(td.bbox),
        }
        for td in text_detections
    ]
    st.dataframe(data, width="stretch")


MODULE_DEFS = [
    {
        "title": "Module 1: Detect and segment tire & rim",
        "match": ["Detect tire & rim"],
        "subs": {},
    },
    {
        "title": "Module 2: Preprocess",
        "match": ["Flatten sidewall", "Preprocess"],
        "subs": {
            "Flatten sidewall": "Flatten Sidewall",
            "Preprocess": "Contrast Processing",
        },
    },
    {
        "title": "Module 3: Scene text detection and recognization",
        "match": ["Detect text regions", "Crop text regions", "Recognize text"],
        "subs": {
            "Detect text regions": "Text Detection",
            "Crop text regions": "Crop Text Regions",
            "Recognize text": "Text Recognition",
        },
    },
    {
        "title": "Module 4: Standardize and correct typo with LLM",
        "match": ["LLM extraction"],
        "subs": {},
    },
]


def find_module(event_name: str):
    """Return (module_index, match_key) for the given event name, or (None, None)."""
    for idx, module in enumerate(MODULE_DEFS):
        for key in module["match"]:
            if event_name.startswith(key):
                return idx, key
    return None, None


def render_module_markdown(idx: int, state: dict) -> str:
    module = MODULE_DEFS[idx]
    lines = [f"**{module['title']}** — {state['total']:.2f}s"]
    if module["subs"]:
        for key, dur in state["subs"]:
            label = module["subs"].get(key, key)
            lines.append(f"  - {label}: {dur:.2f}s")
    return "\n".join(lines)


IMAGE_TITLES = {
    "tire_rim_detection": "Tire & Rim Detection",
    "flattened": "Flattened Sidewall",
    "preprocessed": "Preprocessed Image",
    "detections_preprocessed": "Text Detections (Preprocessed)",
}


def render_output_file(container, key: str, path: str):
    """Render a single pipeline output file (image or crop directory)."""
    if key == "text_crops_dir":
        if not os.path.isdir(path):
            return
        crops = sorted(
            f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        if not crops:
            return
        with container.expander(f"Text Crops ({len(crops)} regions)"):
            cols = st.columns(min(len(crops), 5))
            for i, crop_file in enumerate(crops):
                col = cols[i % len(cols)]
                crop_img = cv2.imread(os.path.join(path, crop_file))
                if crop_img is not None:
                    col.image(
                        cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB),
                        caption=crop_file,
                        width="stretch",
                    )
        return

    title = IMAGE_TITLES.get(key, key)
    if not os.path.exists(path):
        return
    img = cv2.imread(path)
    if img is None:
        return
    container.markdown(f"**{title}**")
    container.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")


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

    # Show uploaded image (convert HEIC/HEIF to JPG bytes first)
    file_bytes = uploaded_file.read()
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext in (".heic", ".heif"):
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            jpg_buf = io.BytesIO()
            pil_image.save(jpg_buf, format="JPEG", quality=95)
            file_bytes = jpg_buf.getvalue()
            file_ext = ".jpg"
        except Exception as e:
            st.error(f"Failed to convert HEIC image: {e}")
            return

    nparr = np.frombuffer(file_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_img is None:
        st.error("Failed to decode the uploaded image.")
        return

    st.image(
        cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        width="stretch",
    )

    if not st.button("Extract", type="primary", width="stretch"):
        return

    # Save uploaded file to a temp path so the pipeline can load it
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Attach a handler to capture logs for the UI
    log_handler = StreamlitLogHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
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
            st.subheader("Pipeline Progress")
            status_area = st.container()
            module_placeholders = [status_area.empty() for _ in MODULE_DEFS]
            module_state = [{"total": 0.0, "subs": []} for _ in MODULE_DEFS]
            tire_info_area = st.container()
            text_detections_area = st.container()
            images_area = st.container()
            images_area.subheader("Pipeline Output Images")

            for event in pipeline.run_pipeline_streaming(
                tmp_path,
                save_debug=save_debug,
            ):
                mod_idx, match_key = find_module(event.name)
                if mod_idx is not None:
                    state = module_state[mod_idx]
                    state["total"] += event.duration
                    state["subs"].append((match_key, event.duration))
                    module_placeholders[mod_idx].markdown(
                        render_module_markdown(mod_idx, state)
                    )

                for key, path in event.output_files.items():
                    render_output_file(images_area, key, path)
                if event.text_detections:
                    with text_detections_area:
                        display_text_detections(event.text_detections)
                if event.tire_info is not None:
                    with tire_info_area:
                        display_tire_info(event.tire_info)
                        st.divider()

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
