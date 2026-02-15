import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "cats_dogs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
