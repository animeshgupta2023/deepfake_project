# 🕵️ Deepfake Detection Engine
A full-stack, end-to-end deepfake detection microservice powered by a custom Vision Transformer (ViT) and Attention Rollout explainability.

## 🚀 Architecture
* **Frontend:** Streamlit web application
* **Backend:** FastAPI REST architecture
* **Face Extraction:** OpenCV DNN (Caffe)
* **Classification:** Custom fine-tuned `vit_small_patch16_224` (timm / PyTorch)
* **Explainability:** Custom Attention Rollout implementation for heatmaps

## ⚙️ How to Run Locally
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/deepfake_project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the model weights (https://drive.google.com/drive/folders/1SFeP4Uu9NY1o3Izr3ySBXLSPCclfxiXo?usp=sharing) and place them in `models/weights/` (See `configs/config.yaml` for expected paths).
4. Start the backend API: `uvicorn src.api:app --reload`
5. Start the frontend UI: `streamlit run frontend/app.py`

## 🐳 Docker Deployment
To run the containerized backend:
```bash
docker build -t deepfake-api .
docker run -p 8000:8000 deepfake-api