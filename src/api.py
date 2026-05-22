import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.pipeline import DeepfakePipeline

app = FastAPI(
    title="Deepfake Detection API", 
    description="ViT-based deepfake detection with Attention Rollout explainability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loading the model ones when the server starts
print("Loading model pipeline...")
try:
    pipeline = DeepfakePipeline(config_path="configs/config.yaml")
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load pipeline: {e}")
    pipeline = None

@app.get("/health")
async def health_check():
    """Cloud load balancers ping this endpoint to ensure the server is alive."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model failed to load.")
    return {"status": "healthy", "model_loaded": True}

# defining strict schemas for the api response
class FaceResult(BaseModel):
    prediction: str
    confidence: float
    bbox: tuple[int, int, int, int]
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None

class PredictResponse(BaseModel):
    status: str
    message: str
    faces: List[FaceResult]

# helper function 
def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Converts a raw OpenCV image array into a Base64 string for JSON transmission."""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

# The API Endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Generate and return attention heatmaps")
):
    """
    Accepts an image file, extracts faces, and predicts if they are real or fake.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: Pipeline offline.")

    # Read the uploaded file bytes into an OpenCV array
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not read the provided image file.")

    try:
        results = pipeline.run(img, include_explanation=explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal model processing error: {e}")

    formatted_faces = []
    for res in results:
        face_data = {
            "prediction": res["prediction"],
            "confidence": float(res["confidence"]),
            "bbox": res["bbox"]
        }

        if explain and "overlay" in res:
            face_data["heatmap_base64"] = encode_image_to_base64(res["heatmap"])
            face_data["overlay_base64"] = encode_image_to_base64(res["overlay"])

        formatted_faces.append(FaceResult(**face_data))

    return PredictResponse(
        status="success",
        message=f"Successfully processed {len(formatted_faces)} face(s).",
        faces=formatted_faces
    )