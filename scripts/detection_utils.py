# Handles all pinniped detection logic using Roboflow API

import os
import numpy as np
from PIL import Image
from roboflow import Roboflow
import streamlit as st

from scripts.config import API_KEY, PROJECT, VERSION, CONF, OVERLAP
from scripts.exif_utils import extract_exif_metadata
from scripts.detections_struct import DetectionsLite  # our lightweight struct

def parse_roboflow_detections(result_json):
    xyxy, confidence, class_id = [], [], []
    for pred in result_json.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x_min, y_min = x - w / 2, y - h / 2
        x_max, y_max = x + w / 2, y + h / 2
        xyxy.append([x_min, y_min, x_max, y_max])
        confidence.append(pred["confidence"])
        class_id.append(0)  # single-class 'pinniped' as id 0

    if not xyxy:
        return DetectionsLite(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
        )

    return DetectionsLite(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.asarray(class_id, dtype=np.int32),
    )

@st.cache_resource(show_spinner="Loading Roboflow model…")
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT)
    return project.version(VERSION).model

def process_camera_image(img_source, model=None):
    """
    img_source: path (str/Path) or file-like (e.g., Streamlit UploadedFile).
    """
    if model is None:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT)
        model = project.version(VERSION).model

    # Normalize input to a file path so EXIF and Roboflow both work
    if hasattr(img_source, "read"):  # Streamlit UploadedFile or similar
        import tempfile
        img = Image.open(img_source).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, format="JPEG", quality=95)
        img_path = tmp_path
    else:
        img_path = str(img_source)

    capture_date, capture_time = extract_exif_metadata(img_path)
    result = model.predict(img_path, confidence=CONF, overlap=OVERLAP).json()
    detections = parse_roboflow_detections(result)
    pinniped_count = len(detections)

    return {
        "filename": os.path.basename(img_path),
        "capture_date": capture_date,
        "capture_time": capture_time,
        "pinniped_count": int(pinniped_count),
        "detections": detections,
        "raw_result": result,
    }
