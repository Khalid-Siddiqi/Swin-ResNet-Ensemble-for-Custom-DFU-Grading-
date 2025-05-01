from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
from timm import create_model
import io

app = FastAPI()

# Device and model setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
IMG_SIZE = 224

# Load class names
with open('class_names.txt', 'r') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load model
model = create_model('convnext_xlarge_in22k', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('best_convnext.pth', map_location=DEVICE))
model.to(DEVICE).eval()

# Transform
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            pred_class = outputs.argmax(dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][pred_class].item()

        return JSONResponse(content={
            "predicted_class": CLASS_NAMES[pred_class],
            "class_index": pred_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
