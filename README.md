# Deepfake Detection Engine

A full-stack deepfake detection application that detects faces in an image, classifies each face as real or fake using a Vision Transformer, and optionally generates attention heatmaps to explain the model's decision.

## Overview

This project combines:

- a face detection model based on OpenCV DNN (Caffe SSD)
- a ViT-based classifier using PyTorch and timm
- attention rollout explainability for visual overlays
- a FastAPI backend for inference
- a Streamlit frontend for interactive image upload and result viewing

## Project structure

```text
.
├── configs/
│   └── config.yaml
├── deepfake/                  # local Python virtual environment
├── frontend/
│   └── app.py                 # Streamlit UI
├── models/
│   └── weights/
│       ├── deploy.prototxt
│       ├── res10_300x300_ssd_iter_140000.caffemodel
│       └── vit_small.pth
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── classifier.py          # model loading and prediction logic
│   ├── explainer.py           # attention heatmap generation
│   ├── face_extractor.py      # face detection logic
│   ├── pipeline.py            # end-to-end processing pipeline
│   └── __init__.py
├── tests/
│   ├── test_classifier.py
│   └── test_extractor.py
├── Dockerfile
├── README.md
├── requirements.txt
├── test_run.py
└── ...
```

## Model requirements

The system expects the following files to exist before running inference:

- `models/weights/deploy.prototxt`
- `models/weights/res10_300x300_ssd_iter_140000.caffemodel`
- `models/weights/vit_small.pth`

These paths are configured in `configs/config.yaml`.

## Setup

### 1) Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv deepfake
.\deepfake\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv deepfake
source deepfake/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download model weights

Download the required face detector and ViT weights and place them under `models/weights/`.

## Run the application

### Backend API

Start the FastAPI service:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API exposes:

- `GET /health` — checks whether the model is loaded
- `POST /predict` — accepts an uploaded image and returns face-level predictions

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -F "file=@example.jpg"
```

### Frontend UI

In a second terminal, start Streamlit:

```bash
streamlit run frontend/app.py
```

Then open the local URL shown by Streamlit in the browser.

The UI allows you to:

- upload an image
- optionally enable attention heatmap overlays
- view the original image and the model's explanation for each detected face
- see the predicted label and confidence score

## API response format

The backend returns a JSON object like:

```json
{
  "status": "success",
  "message": "Successfully processed 1 face(s).",
  "faces": [
    {
      "prediction": "Fake",
      "confidence": 0.96,
      "bbox": [x1, y1, x2, y2],
      "heatmap_base64": "...",
      "overlay_base64": "..."
    }
  ]
}
```

## Docker

This repo includes a Dockerfile that runs the FastAPI app inside a container.

Build the image:

```bash
docker build -t deepfake-api .
```

Run it:

```bash
docker run -p 7860:7860 deepfake-api
```

The container serves the API on port `7860`.

## Testing

You can run the included unit tests with:

```bash
pytest
```

## Notes

- The classifier uses the model name `vit_small_patch16_224` as defined in `configs/config.yaml`.
- The face detector uses a Caffe SSD model and returns bounding boxes above a configurable confidence threshold.
- If no face is detected, the pipeline returns an empty result list.
- The heatmap option is helpful for diagnosing what regions of an image the model focused on.

## Typical workflow

1. Activate the virtual environment.
2. Install project dependencies.
3. Ensure the three weight files are present.
4. Launch the backend with `uvicorn`.
5. Launch the Streamlit app.
6. Upload an image and analyze it.

## Troubleshooting

- If the backend fails to start, verify that `configs/config.yaml` points to valid model files.
- If the app cannot connect to the API, make sure the FastAPI server is running on port `8000`.
- If the model memory usage is high, reduce image size or run on a machine with adequate GPU/CPU resources.
- If you see no detections, try a higher-quality image with clearer visible faces.

