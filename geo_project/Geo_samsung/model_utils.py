import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np



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

def build_fig(image, true_lat, true_lon, pred_lat, pred_lon):
    img = np.array(image)
    error_km = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)

    fig = make_subplots(
        rows=1,
        cols=2,

        column_widths=[0.45, 0.55],

        specs=[
            [
                {"type": "image"},
                {"type": "scattergeo"}
            ]
        ],

        subplot_titles=(
            "Image",
            f"Error: {error_km:.1f} km"
        )
    )

    # ==================================================
    # IMAGE
    # ==================================================

    fig.add_trace(
        go.Image(z=img),
        row=1,
        col=1
    )

    # TRUE POINT
    fig.add_trace(
        go.Scattergeo(
            lon=[true_lon],
            lat=[true_lat],
            mode='markers',

            marker=dict(
                size=12,
                color="#00cc44"
            ),

            name='TRUE'
        ),
        row=1,
        col=2
    )

    # PRED POINT
    fig.add_trace(
        go.Scattergeo(
            lon=[pred_lon],
            lat=[pred_lat],
            mode='markers',

            marker=dict(
                size=12,
                color="#D32727"
            ),

            name='PRED'
        ),
        row=1,
        col=2
    )

    # ERROR LINE
    fig.add_trace(
        go.Scattergeo(
            lon=[true_lon, pred_lon],
            lat=[true_lat, pred_lat],
            mode='lines',
            line=dict(
                width=2,
                color="#000000"
            ),

            name='ERROR'
        ),
        row=1,
        col=2
    )

    # ==================================================
    # LAYOUT
    # ==================================================

    fig.update_geos(
        projection_type="natural earth",

        lataxis_range=[
            min(true_lat, pred_lat) - 10,
            max(true_lat, pred_lat) + 10,
        ],

        lonaxis_range=[
            min(true_lon, pred_lon) - 10,
            max(true_lon, pred_lon) + 10,
        ],

        showland=True,
        showcountries=True,
    )

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn'
    )

    return plot_html
