# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from PIL import Image
# import torch
# import torchvision.transforms as T
# from timm import create_model
# import io

# app = FastAPI()

# # Device and model setup
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# NUM_CLASSES = 4
# IMG_SIZE = 224

# # Load class names
# with open('class_names.txt', 'r') as f:
#     CLASS_NAMES = [line.strip() for line in f.readlines()]

# # Load model
# model = create_model('convnext_xlarge_in22k', pretrained=False, num_classes=NUM_CLASSES)
# model.load_state_dict(torch.load('best_convnext.pth', map_location=DEVICE))
# model.to(DEVICE).eval()

# # Transform
# transform = T.Compose([
#     T.Resize(256),
#     T.CenterCrop(IMG_SIZE),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#         image = transform(image).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             outputs = model(image)
#             pred_class = outputs.argmax(dim=1).item()
#             confidence = torch.softmax(outputs, dim=1)[0][pred_class].item()

#         return JSONResponse(content={
#             "predicted_class": CLASS_NAMES[pred_class],
#             "class_index": pred_class,
#             "confidence": round(confidence, 4)
#         })

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
import io
import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from timm import create_model
from ultralytics import YOLO

app = FastAPI()

# ------------------- SETUP -----------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
NUM_CLASSES = 4

# Load class names
with open('class_names.txt', 'r') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load ConvNeXt classification model
clf_model = create_model('convnext_xlarge_in22k', pretrained=False, num_classes=NUM_CLASSES)
clf_model.load_state_dict(torch.load('best_convnext.pth', map_location=DEVICE))
clf_model.to(DEVICE).eval()

# Load YOLOv8 instance segmentation model
seg_model = YOLO("Instance_Segementation_Model.pt")

# Image Transform for ConvNeXt
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ------------------- API Endpoint -----------------------
@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    format: str = Query("base64", enum=["base64", "stream"])
):
    try:
        # Read the uploaded image bytes
        image_bytes = await file.read()

        # ========== CLASSIFICATION ==========
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = clf_model(image_tensor)
            pred_class_idx = outputs.argmax(dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][pred_class_idx].item()
            predicted_class = CLASS_NAMES[pred_class_idx]

        # ========== INSTANCE SEGMENTATION ==========
        np_image = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        results = seg_model.predict(original_image)

        annotated_image = original_image.copy()
        alpha = 0.5

        for result in results:
            for mask, box in zip(result.masks, result.boxes):
                mask_array = mask.data.cpu().numpy()
                if mask_array.shape[1:] != annotated_image.shape[:2]:
                    mask_array = cv2.resize(mask_array[0], (annotated_image.shape[1], annotated_image.shape[0]))
                overlay = np.zeros_like(annotated_image, dtype=np.uint8)
                overlay[mask_array.astype(bool)] = (255, 0, 0)
                annotated_image = cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Convert image to response format
        _, buffer = cv2.imencode('.jpg', annotated_image)
        io_buf = io.BytesIO(buffer)

        if format == "base64":
            base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            return JSONResponse(content={
                "predicted_class": predicted_class,
                "class_index": pred_class_idx,
                "confidence": round(confidence, 4),
                "segmented_image": base64_image
            })
        else:
            return StreamingResponse(io_buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
