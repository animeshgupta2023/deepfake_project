import torch
import timm
from unittest.mock import patch
from src.classifier import DeepfakeClassifier

@patch('torch.load')
def test_classifier_predict_formats(mock_torch_load):
    # 1. Setup the Mock with random weights
    dummy_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
    dummy_state_dict = dummy_model.state_dict()
    mock_torch_load.return_value = dummy_state_dict

    # 2. Initialize the Classifier
    classifier = DeepfakeClassifier(
        weights_path="dummy_weights.pth", 
        model_name="vit_small_patch16_224", 
        device="cpu"
    )

    # 3. Create a dummy PyTorch Tensor (Batch=1, Channels=3, 224x224 pixels)
    dummy_tensor = torch.randn(1, 3, 224, 224)

    # 4. Run the pipeline
    pred_class, confidence = classifier.predict(dummy_tensor)

    # 5. Assertions
    assert isinstance(pred_class, int), "Prediction class should be an integer."
    assert pred_class in [0, 1], "Prediction should be either 0 (Fake) or 1 (Real)."
    assert isinstance(confidence, float), "Confidence should be a float."
    assert 0.0 <= confidence <= 1.0, "Confidence must be a percentage between 0 and 1."