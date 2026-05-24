import numpy as np
from unittest.mock import patch, MagicMock
from src.face_extractor import FaceExtractor

@patch('cv2.dnn.readNetFromCaffe')
def test_extract_faces_no_detections(mock_read_net):
    # 1. Setup the "Mock" (Fake) OpenCV Network
    mock_net = MagicMock()
    
    # Create a fake output tensor representing 0 detections 
    # OpenCV DNN face detector returns shape (1, 1, N, 7)
    fake_detections = np.zeros((1, 1, 1, 7))
    fake_detections[0, 0, 0, 2] = 0.1 # 10% confidence (below threshold)
    
    # Tell our fake network to return this tensor when forward() is called
    mock_net.forward.return_value = fake_detections
    mock_read_net.return_value = mock_net

    # 2. Initialize the Extractor (it will use the mock instead of real files)
    extractor = FaceExtractor(
        prototxt_path="dummy.prototxt", 
        model_path="dummy.caffemodel", 
        confidence_threshold=0.5
    )

    # 3. Create a dummy image (e.g., a 500x500 pitch black image)
    dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)

    # 4. Run the method being tested
    faces = extractor.extract_faces(dummy_image)

    # 5. Assertions (The actual test)
    assert isinstance(faces, list), "Extractor should return a list."
    assert len(faces) == 0, "Extractor should find 0 faces in a black image."