# Handles all pinniped detection logic using Roboflow API

import numpy as np
import cv2
import os
from roboflow import Roboflow
from supervision import Detections
import streamlit as st
from scripts.config import API_KEY, PROJECT, VERSION, CONF, OVERLAP
from PIL import Image
from scripts.exif_utils import extract_exif_metadata

def parse_roboflow_detections(result_json):
    xyxy, confidence, class_id = [], [], []
    for pred in result_json.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x_min, y_min = x - w / 2, y - h / 2
        x_max, y_max = x + w / 2, y + h / 2
        xyxy.append([x_min, y_min, x_max, y_max])
        confidence.append(pred["confidence"])
        class_id.append(0)

    if not xyxy:
        return Detections(xyxy=np.zeros((0, 4)), confidence=np.array([]), class_id=np.array([]))

    return Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.array(class_id)
    )

@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT)
    return project.version(VERSION).model

def process_camera_image(img_path, model=None):
    """Process a single camera trap image for pinniped detection."""
    if model is None:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT)
        model = project.version(VERSION).model

    capture_date, capture_time = extract_exif_metadata(img_path)
    result = model.predict(img_path, confidence=CONF, overlap=OVERLAP).json()
    detections = parse_roboflow_detections(result)
    pinniped_count = len(detections.xyxy)

    return {
        'filename': os.path.basename(img_path),
        'capture_date': capture_date,
        'capture_time': capture_time,
        'pinniped_count': pinniped_count,
        'detections': detections,
        'raw_result': result
    }
