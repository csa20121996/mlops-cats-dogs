from fastapi import FastAPI, UploadFile, File
from src.inference.predict import predict_image
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency"
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)

    return response


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await predict_image(file)
