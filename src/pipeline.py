import yaml
import cv2
import numpy as np
from typing import List, Dict
import torch

from src.face_extractor import FaceExtractor
from src.classifier import DeepfakeClassifier
from src.explainer import ViTExplainer

class DeepfakePipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        if self.config["model"]["device"] == "auto":
            self.config["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.extractor = FaceExtractor(
            prototxt_path=self.config['detector']['prototxt'],
            model_path=self.config['detector']['caffemodel'],
            confidence_threshold=self.config['detector']['confidence_threshold']
        )

        self.classifier = DeepfakeClassifier(
            weights_path=self.config['model']['weights_path'],
            model_name=self.config['model']['name'],
            device=self.config['model']['device']
        )
        
        self.explainer = ViTExplainer(
            model=self.classifier.model, 
            device=self.config['model']['device']
        )

    def run(self, image: np.ndarray, include_explanation: bool = False) -> List[Dict]:
        results = []
        faces = self.extractor.extract_faces(image)
        
        for face_data in faces:
            crop = face_data["crop"]

            tensor = self.classifier.preprocess(crop)
            
            pred_class, confidence = self.classifier.predict(tensor)
            
            result_dict = {
                "bbox": face_data["bbox"],
                "prediction": "Fake" if pred_class == 0 else "Real", # fake is 0 real is 1
                "confidence": confidence
            }
            
            if include_explanation:
                visuals = self.explainer.generate_heatmap(tensor, crop)
                result_dict["heatmap"] = visuals["heatmap"]
                result_dict["overlay"] = visuals["overlay"]
                
            results.append(result_dict)
            
        return results