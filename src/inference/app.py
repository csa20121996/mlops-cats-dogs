from fastapi import FastAPI, UploadFile, File
import logging
import time
from typing import Dict

import torch
from prometheus_client import Counter, Histogram, generate_latest
from PIL import Image
from torchvision import transforms
from io import BytesIO

from src.model import SimpleCNN
from src.config import MODEL_PATH, IMAGE_SIZE

app = FastAPI(title="Cats vs Dogs Inference Service")

# Logging (avoid dumping raw image bytes)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops-app")

# Metrics
REQUEST_COUNT = Counter("request_count", "Total prediction requests", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency (seconds)", ["endpoint"])

# Load model (state_dict saved by training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

CLASS_NAMES = ["Cat", "Dog"]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    conf, idx = torch.max(probs, 0)
    result = {
        "label": CLASS_NAMES[int(idx)],
        "confidence": float(conf),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

    logger.info(
        "predict label=%s confidence=%.4f latency_sec=%.4f filename=%s content_type=%s",
        result["label"], result["confidence"], latency, file.filename, file.content_type
    )
    return result


@app.get("/metrics")
def metrics():
    return generate_latest()