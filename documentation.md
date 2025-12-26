# Pinniped Camera App — Documentation

A lightweight Streamlit application for detecting pinnipeds (seals/sea lions) in camera trap images, visualizing detections, and exporting simple census summaries.

- Live app: https://psf-pinniped-camera-app.streamlit.app/
- Roboflow model deployed: https://universe.roboflow.com/eugene-7lac8/pinniped-detection-5vtha
- Code entrypoint: [app.py](app.py)
- Utilities: [scripts/detection_utils.py](scripts/detection_utils.py), [scripts/annotation_utils.py](scripts/annotation_utils.py), [scripts/exif_utils.py](scripts/exif_utils.py), [scripts/config.py](scripts/config.py)

## Overview

- Upload JPG/PNG images.
- Run the detection model and draw bounding boxes + labels.
- Read EXIF timestamps when available to enrich results.
- Review per-image counts and download a CSV summary.

## Features

- Image upload (jpg/jpeg/png)
- Automatic detection with confidence/overlap controls
- Visual annotations (boxes + class labels)
- EXIF `DateTimeOriginal` extraction (if present)
- Per-image counts + CSV export

## Architecture

- UI: [app.py](app.py) using Streamlit widgets, layout, and actions.
- Detection: [scripts/detection_utils.py](scripts/detection_utils.py) for model loading, inference, and result parsing.
- Annotation: [scripts/annotation_utils.py](scripts/annotation_utils.py) for drawing bounding boxes onto the annotated images.
- EXIF: [scripts/exif_utils.py](scripts/exif_utils.py) for timestamp extraction and metadata parsing.
- Config: [scripts/config.py](scripts/config.py) for thresholds, project settings, and app constants.

## Model Training

This detection model was trained on camera trap images collected from on‑land cameras at the Nanaimo, Cowichan, and Campbell sites. The full dataset lifecycle was managed on Roboflow:

- Dataset creation: Images uploaded to Roboflow and split into train, validation and test sets.
- Annotation: Pinniped bounding boxes labeled directly on the Roboflow platform.
- Training: Roboflow hosted training was used to iterate and improve the detector.
- Versioning: The model reached Version 6, which was selected for deployment.
- Deployment: The Version 6 inference endpoint is integrated into the Streamlit app via the configuration in [scripts/config.py](scripts/config.py) and secrets in `.streamlit/secrets.toml`.

## Workflow

1. Upload images (Streamlit app)
	- Users select one or more JPG/PNG images via the uploader.
	- Files are read into memory with their original filenames for reporting.

2. EXIF Extraction
	- For each image, [scripts/exif_utils.py](scripts/exif_utils.py) attempts to read `DateTimeOriginal` and other tags.
	- If EXIF is absent, capture time remains empty and the app proceeds.

3. Model Inference (Roboflow)
	- [scripts/detection_utils.py](scripts/detection_utils.py) loads configuration from [scripts/config.py](scripts/config.py) and the API key from `.streamlit/secrets.toml`.
	- Images are sent to the configured Roboflow project/version for inference.
	- The API returns bounding boxes (`x`, `y`, `width`, `height`), `class` labels, and `confidence` scores.

4. Parsing & Post‑Processing
	- Predictions are parsed and filtered using `CONF` (confidence) and `OVERLAP` (NMS) thresholds in [scripts/config.py](scripts/config.py).
	- Non‑max suppression reduces overlapping detections to the most confident boxes.

5. Annotation & Counts
	- [scripts/annotation_utils.py](scripts/annotation_utils.py) draws boxes and labels on the original image to produce an annotated preview.
	- Per‑image pinniped counts are computed from the parsed detections.

6. Display & Export (Streamlit)
	- Annotated images are rendered with `st.image`, and a table summarizes filename, capture time, optional location, and counts.
	- A CSV summary is generated and offered via `st.download_button` for quick census reporting.

```text
User
  │
  ├── Upload images (Streamlit)
  │       ↓
  ├── EXIF extraction (exif_utils)
  │       ↓
  ├── Inference (Roboflow via detection_utils)
  │       ↓
  ├── Parse + NMS (config thresholds)
  │       ↓
  ├── Annotate (annotation_utils)
  │       ↓
  └── Display + CSV (Streamlit UI)
```

## Steps to run the app locally using VS Code and deploy the app to streamlit cloud
(This is only required if changes are to be made to the app code)

1. Install Python 3.10+ and VS Code.
2. Open the repository folder in VS Code (File → Open Folder…).
3. Create a virtual environment using the integrated terminal:

```bash
python -m venv .venv
```

4. Select the interpreter: press `Ctrl+Shift+P` → “Python: Select Interpreter” → choose `.venv`.
5. Activate the environment and install dependencies in the terminal:

```bash
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

6. Add secrets if needed (e.g., Roboflow) by creating `.streamlit/secrets.toml`:

```toml
ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

7. Run the app from VS Code terminal:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Recommended extensions: “Python” (Microsoft) and “Pylance”.

### Configuration

Edit detection and app settings in [scripts/config.py](scripts/config.py):

- Confidence threshold (e.g., `CONF`)
- NMS/overlap threshold (e.g., `OVERLAP`)
- Optional project identifiers (e.g., `PROJECT`, `VERSION`)
- Any UI text or constant values

Secrets like `ROBOFLOW_API_KEY` are read from `.streamlit/secrets.toml` when available.

### Using the App

- Review annotated images and detection counts.
- Download the CSV summary for reporting.

### CSV Output (example columns)

- `filename`: original image file name
- `capture_datetime`: from EXIF `DateTimeOriginal` if available; empty if missing
- `location`: optional user-provided field from the UI
- `pinniped_count`: total detections per image

Columns may vary based on the current implementation in [app.py](app.py) and utilities.

### Deployment (Streamlit Cloud)

1. Push your repository to GitHub.
2. Create a new Streamlit app from the repo.
3. In the app settings, add secrets under “Advanced settings” → “Secrets”:

```toml
ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

4. Ensure `requirements.txt` is present and up to date.
5. Select the correct Python version if prompted (3.10+).
6. Deploy; the app will build and serve at a public URL.
