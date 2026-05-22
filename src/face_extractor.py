import cv2
import logging 
import numpy as np
from typing import List, Dict

class FaceExtractor:
    def __init__(self, prototxt_path, model_path, confidence_threshold):
        self.confidence_threshold = confidence_threshold
        try: 
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logging.info("Face detection model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading face detection model: {e}")
            raise
    
    def extract_faces(self, img):
        face_data = []
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = img[y1:y2, x1:x2]

                face_data.append({
                    "crop": face_crop,
                    "confidence": float(confidence),
                    "bbox": (x1, y1, x2, y2)
                })
                
        return face_data