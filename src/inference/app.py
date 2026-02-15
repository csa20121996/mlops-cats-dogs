from fastapi import FastAPI, UploadFile, File
from src.inference.predict import predict_image

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await predict_image(file)
