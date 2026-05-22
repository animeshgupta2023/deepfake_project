import torch
import timm
import cv2
import logging
from PIL import Image
from torchvision import transforms
from typing import Tuple
import numpy as np 

class DeepfakeClassifier:
    ''' Classifies the image if fake or real '''
    def __init__(self, weights_path: str, model_name: str = 'vit_base_patch16_224', device: str ='cpu'):
        self.device = device
        self.model_name = model_name

        try:
            self.model = timm.create_model(model_name, pretrained=False, num_classes=2)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device).eval()
            logging.info(f"Successfully loaded model '{model_name}' with weights from '{weights_path}' on device '{device}'")

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, face_crop: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def predict(self, input_tensor: torch.Tensor) -> Tuple[int, float]:
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
        pred_class = torch.argmax(probs, dim=1).item() 
        confidence = probs[0][pred_class].item()
        
        return pred_class, float(confidence)