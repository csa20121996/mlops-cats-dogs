from src.dataset import preprocess_image
import numpy as np

def test_preprocess_shape():
    dummy = np.random.randint(0,255,(300,300,3),dtype=np.uint8)
    out = preprocess_image(dummy)
    assert out.shape == (3, 224, 224)
