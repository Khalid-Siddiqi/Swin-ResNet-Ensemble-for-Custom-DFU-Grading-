from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io

# Initialize FastAPI app
app = FastAPI()

# Load your trained Swin Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your model class and weights are ready
# Replace 'YourSwinModelClass' with your actual model class
from your_model_file import YourSwinModelClass  # Modify this

model = YourSwinModelClass()
model.load_state_dict(torch.load("swin_model.pth", map_location=device))
model.to(device)
model.eval()

# Define preprocessing (you should match training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Swin often uses 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        return JSONResponse(content={"prediction": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# pip install fastapi uvicorn python-multipart torch torchvision pillow
# uvicorn app:app --reload
