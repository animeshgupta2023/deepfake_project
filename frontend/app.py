import streamlit as st
import requests
import base64
import io
from PIL import Image

# --- Configuration ---
#API_URL = "http://127.0.0.1:8000/predict"
API_URL = "https://animeshgupta2022-deepfake-api.hf.space/predict"
MAX_FILE_SIZE_MB = 5 # Prevent server crashes from massive files

# --- Page Layout ---
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")
st.title("🕵️ Deepfake Detection Engine")
st.markdown("Upload an image to detect if the faces are **Real** or **Fake** using a Vision Transformer.")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
explain_mode = st.sidebar.checkbox("Generate Attention Heatmaps", value=True, 
                                   help="Shows exactly what the AI is looking at.")
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "avif"])

# --- Main Application Logic ---
if uploaded_file is not None:
    # 1. File Size Validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is too large ({file_size_mb:.1f}MB). Please upload an image smaller than {MAX_FILE_SIZE_MB}MB.")
    else:
        # Load the full image into memory ONCE
        full_image = Image.open(uploaded_file).convert("RGB")
        
        # Big analyze button at the top
        analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)
        st.markdown("---")

        if analyze_button:
            with st.spinner("Analyzing faces with Vision Transformer..."):
                # Prepare data for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"explain": str(explain_mode).lower()}
                
                try:
                    # Make the network request
                    response = requests.post(API_URL, files=files, params=params, timeout=30)
                    response.raise_for_status() 
                    data = response.json()
                    
                    if data["status"] == "success":
                        faces = data["faces"]
                        if len(faces) == 0:
                            st.warning("No faces were detected in this image.")
                        
                        # Loop through every face detected
                        for i, face in enumerate(faces):
                            # A. Text & Metrics above the images
                            st.markdown(f"### Face {i+1}: **{face['prediction']}**")
                            
                            conf = face['confidence']
                            if face['prediction'] == "Fake":
                                st.progress(conf, text=f"Fake Confidence: {conf*100:.1f}%")
                            else:
                                st.progress(conf, text=f"Real Confidence: {conf*100:.1f}%")
                            
                            st.write("") # Spacer
                            
                            # B. Create perfectly balanced image columns
                            img_col1, img_col2 = st.columns(2, gap="large")
                            
                            with img_col1:
                                # Left side: Full original image
                                st.image(full_image, caption="Original Image", use_container_width=True)

                            with img_col2:
                                if explain_mode and face.get("overlay_base64"):
                                    # 1. Decode the Base64 heatmap from the API
                                    img_bytes = base64.b64decode(face["overlay_base64"])
                                    heatmap_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    
                                    # 2. Create a fresh copy of the full image for the right side
                                    full_overlay_image = full_image.copy()
                                    
                                    # 3. Get the exact bounding box coordinates
                                    x1, y1, x2, y2 = face["bbox"]
                                    box_width = x2 - x1
                                    box_height = y2 - y1
                                    
                                    # 4. Resize the heatmap to perfectly fit the face's bounding box
                                    heatmap_resized = heatmap_pil.resize((box_width, box_height))
                                    
                                    # 5. Paste the glowing face back into the full image
                                    full_overlay_image.paste(heatmap_resized, (x1, y1))
                                    
                                    # Right side: Full image with perfectly localized overlay
                                    st.image(full_overlay_image, caption=f"Attention Overlay", use_container_width=True)
                                    
                            st.markdown("---") # Divider for the next face
                            
                    else:
                        st.error(f"API Error: {data.get('message', 'Unknown error')}")
                        
                except requests.exceptions.Timeout:
                    st.error("🚨 Request timed out. The model took too long to respond.")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to the backend. Ensure your FastAPI server is running on port 8000.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            # Preview state before the user clicks analyze
            st.subheader("Original Image Preview")
            # Create a center-constrained layout so tall images don't take over the screen
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.image(full_image, use_container_width=True)
                
else:
    # Empty state before any upload
    st.info("👈 Please upload an image from the sidebar to begin.")



# cloud link : https://deepfakeproject-agkzhcmdjejnoxpu9jwg6t.streamlit.app/