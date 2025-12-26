# Pinniped Camera App

A lightweight Streamlit app for detecting pinnipeds (seals/sea lions) in on-land camera images, visualizing detections, and exporting simple census summaries.

![Pinniped sample](https://raw.githubusercontent.com/riyaeliza123/psf-image/refs/heads/main/d006578131b8121d145ad74d4629ed357d7147e5fd9cab8afbf9d6f0.jpg)

## Deployed App

The app is deployed on Streamlit Cloud.

- Deployed URL: https://psf-pinniped-camera-app.streamlit.app/

## App flow

- Upload JPG/PNG images from camera traps.
- The app runs a detection model and draws bounding boxes on the images.
- EXIF timestamps (when available) are read from images to enrich results.
- Get per-image counts and download a CSV summary for quick census reporting.


## Project Structure

- [app.py](app.py): Streamlit UI and app workflow.
- [scripts/detection_utils.py](scripts/detection_utils.py): Model loading, inference, and parsing.
- [scripts/annotation_utils.py](scripts/annotation_utils.py): Drawing boxes/labels on images.
- [scripts/exif_utils.py](scripts/exif_utils.py): EXIF metadata extraction (timestamps, etc.).
- [scripts/config.py](scripts/config.py): Central configuration for thresholds and project info.
- [requirements.txt](requirements.txt): Python dependencies.
