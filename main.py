from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torchvision.transforms as T
from timm import create_model
import numpy as np
import cv2
import io
from ultralytics import YOLO

app = FastAPI()

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load class names
with open('class_names.txt', 'r') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load ConvNeXt classification model
NUM_CLASSES = 4
IMG_SIZE = 224
classifier = create_model('convnext_xlarge_in22k', pretrained=False, num_classes=NUM_CLASSES)
classifier.load_state_dict(torch.load('best_convnext.pth', map_location=DEVICE))
classifier.to(DEVICE).eval()

# Load YOLO segmentation model
segmenter = YOLO("Instance_Segementation_Model.pt")

# Image transforms for classification
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # ---- Classification (ConvNeXt) ----
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = classifier(input_tensor)
            pred_class = outputs.argmax(dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][pred_class].item()
        predicted_label = CLASS_NAMES[pred_class]

        # ---- Segmentation (YOLO) ----
        results = segmenter.predict(np_image)
        annotated = np_image.copy()
        alpha = 0.5

        for result in results:
            for mask, box in zip(result.masks, result.boxes):
                mask_array = mask.data.cpu().numpy()
                if mask_array.shape[1:] != annotated.shape[:2]:
                    mask_array = cv2.resize(mask_array[0], (annotated.shape[1], annotated.shape[0]))
                overlay = np.zeros_like(annotated, dtype=np.uint8)
                overlay[mask_array.astype(bool)] = (255, 0, 0)
                annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Encode the final image as JPEG stream
        _, buffer = cv2.imencode('.jpg', annotated)
        stream = io.BytesIO(buffer)

        # Return as StreamingResponse with metadata in headers
        headers = {
            "X-DFU-Grade": predicted_label,
            "X-Class-Index": str(pred_class),
            "X-Confidence": str(round(confidence, 4))
        }

        return StreamingResponse(stream, media_type="image/jpeg", headers=headers)

    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)