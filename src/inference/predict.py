import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

from src.model import SimpleCNN
from src.config import MODEL_PATH, IMAGE_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


async def predict_image(file):

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    class_names = ["cat", "dog"]
    confidence, predicted = torch.max(probs, 1)

    return {
        "prediction": class_names[predicted.item()],
        "confidence": float(confidence.item())
    }
