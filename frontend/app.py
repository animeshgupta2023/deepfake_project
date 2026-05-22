import streamlit as st
import requests
import base64
import io
from PIL import Image

# Configuration
API_URL = "http://127.0.0.1:8000/predict"

# Set up the webpage layout
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")
st.title("🕵️ Deepfake Detection Engine")
st.markdown("Upload an image to detect if the faces are **Real** or **Fake** using a Vision Transformer.")

# --- SIDEBAR ---
st.sidebar.header("Controls")
explain_mode = st.sidebar.checkbox("Generate Attention Heatmaps", value=True)
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "avif"])

# --- MAIN LAYOUT ---
if uploaded_file is not None:
    # Create two columns of equal width
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Original Image")
        # The image will now be constrained to the width of this left column
        st.image(uploaded_file, use_container_width=True)
        
        # Place the analyze button right under the original image
        analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        if analyze_button:
            with st.spinner("Analyzing faces with Vision Transformer..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"explain": str(explain_mode).lower()}
                
                try:
                    response = requests.post(API_URL, files=files, params=params)
                    response.raise_for_status() 
                    data = response.json()
                    
                    if data["status"] == "success":
                        faces = data["faces"]
                        
                        if len(faces) == 0:
                            st.warning("No faces were detected in this image.")
                        
                        for i, face in enumerate(faces):
                            st.markdown(f"### Face {i+1}: **{face['prediction']}**")
                            
                            # Confidence Progress Bar
                            conf = face['confidence']
                            if face['prediction'] == "Fake":
                                st.progress(conf, text=f"Fake Confidence: {conf*100:.1f}%")
                            else:
                                st.progress(conf, text=f"Real Confidence: {conf*100:.1f}%")
                            
                            # Decode and display the Base64 Heatmap Overlay
                            if explain_mode and face.get("overlay_base64"):
                                img_bytes = base64.b64decode(face["overlay_base64"])
                                img_pil = Image.open(io.BytesIO(img_bytes))
                                # The heatmap will be constrained to the right column
                                st.image(img_pil, caption=f"Attention Overlay (Face {i+1})", use_container_width=True)
                                
                    else:
                        st.error(f"API Error: {data.get('message', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to the backend API. Is your FastAPI server running on port 8000?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    # What the user sees before uploading anything
    st.info("👈 Please upload an image from the sidebar to begin.")