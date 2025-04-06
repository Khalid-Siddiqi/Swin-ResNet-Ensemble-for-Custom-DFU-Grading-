import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model
import os

# ---------- CONFIG ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 4
SWIN_WEIGHT = 0.7
RESNET_WEIGHT = 0.3
TTA_ROUNDS = 5
CLASS_NAMES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
MODEL_NAME = "swin_tiny_patch4_window7_224"

# ---------- TRANSFORMS ----------
tta_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- LOAD MODELS ----------
# Load Swin Transformer
swin_model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
swin_model.load_state_dict(torch.load("ensemble/swin_dfu_best.pth", map_location=DEVICE))
swin_model.to(DEVICE)
swin_model.eval()

# Load ResNet50
resnet_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model.load_state_dict(torch.load("ensemble/resnet_best.pth", map_location=DEVICE))
resnet_model.to(DEVICE)
resnet_model.eval()

# ---------- PREDICTION FUNCTION ----------
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")

    swin_sum = torch.zeros((1, NUM_CLASSES), device=DEVICE)
    resnet_sum = torch.zeros((1, NUM_CLASSES), device=DEVICE)

    for _ in range(TTA_ROUNDS):
        tta_img = tta_transforms(image).unsqueeze(0).to(DEVICE)

        swin_out = F.softmax(swin_model(tta_img), dim=1)
        resnet_out = F.softmax(resnet_model(tta_img), dim=1)

        swin_sum += swin_out
        resnet_sum += resnet_out

    swin_avg = swin_sum / TTA_ROUNDS
    resnet_avg = resnet_sum / TTA_ROUNDS

    ensemble_probs = (swin_avg * SWIN_WEIGHT + resnet_avg * RESNET_WEIGHT)
    _, predicted = torch.max(ensemble_probs, 1)

    pred_class = predicted.item()
    class_name = CLASS_NAMES[pred_class]
    return class_name, ensemble_probs.squeeze().cpu().numpy()

# ---------- USAGE ----------
# Replace with your image path
image_path = "/kaggle/working/test_image.jpg"  # or "/content/test_image.jpg" on Colab
predicted_class, probs = predict_image(image_path)

print(f"🧠 Predicted Class: {predicted_class}")
print(f"📊 Class Probabilities:")
for idx, p in enumerate(probs):
    print(f"  {CLASS_NAMES[idx]}: {p:.4f}")
