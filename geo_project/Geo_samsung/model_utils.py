import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os



IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Модель лежит в подпапке models рядом с этим файлом


# Путь к папке, где лежит этот файл (model_utils.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Модель лежит в подпапке models рядом с этим файлом
MODEL_PATH = os.path.join(BASE_DIR, "models", "geo_model_best.pth")

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def build_model():
    model = models.resnet152(weights="DEFAULT")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)
    return model

_model = None

def load_model():
    global _model
    if _model is None:
        _model = build_model()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    return tensor.to(DEVICE)

def predict(image: Image.Image):
    model = load_model()
    inp = preprocess_image(image)
    with torch.no_grad():
        out = model(inp)[0]
    lat_norm = out[0].item()
    sin_lon = out[1].item()
    cos_lon = out[2].item()
    lat = lat_norm * 90.0
    lon = math.degrees(math.atan2(sin_lon, cos_lon))
    return round(lat, 6), round(lon, 6)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return round(R * c, 3)