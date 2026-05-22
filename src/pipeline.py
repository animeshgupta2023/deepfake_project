import yaml
import cv2
import numpy as np
from typing import List, Tuple

from src.face_extractor import FaceExtractor
from src.classifier import DeepfakeClassifier
from src.explainer import VITExplainer

class DeepfakePipeline:
    def __init__(self, config_path: str = "configs/configs.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)