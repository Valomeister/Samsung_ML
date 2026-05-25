# train_geo_cnn.py
# CNN для предсказания latitude + longitude по изображению
# Longitude обучается корректно через sin/cos
# Метрика: ошибка в километрах (Haversine)
# GPU поддерживается автоматически

import os
import math
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# =====================================================
# CONFIG
# =====================================================

CSV_FILE   = r"datasets\images_part_b_subset\metadata.csv"
IMG_DIR    = r"datasets\images_part_b_subset\images"

BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
IMG_SIZE   = 224
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "models/doxer/geo_model_best.pth"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# HELPERS
# =====================================================

def deg2rad(x):
    return x * math.pi / 180.0


def lon_to_sincos(lon_deg):
    r = deg2rad(lon_deg)
    return math.sin(r), math.cos(r)


def sincos_to_lon(sin_v, cos_v):
    lon = math.degrees(math.atan2(sin_v, cos_v))
    return lon


def haversine_torch(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    a = torch.clamp(a, 0.0, 1.0)
    c = 2 * torch.asin(torch.sqrt(a + 1e-7))

    return R * c



# =====================================================
# DATASET
# =====================================================

class GeoDataset(Dataset):
    def __init__(self, csv_file, img_dir, train=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir

        if train:
            self.tfms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485,0.456,0.406],
                    [0.229,0.224,0.225]
                )
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485,0.456,0.406],
                    [0.229,0.224,0.225]
                )
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.tfms(img)

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        lat_norm = lat / 90.0  # normalize latitude to [-1, 1]

        sin_lon, cos_lon = lon_to_sincos(lon)

        target = torch.tensor(
            [lat_norm, sin_lon, cos_lon],
            dtype=torch.float32
        )

        return img, target


# =====================================================
# MODEL
# =====================================================

def build_model():
    model = models.resnet152(weights="DEFAULT")   # глубокий ResNet

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)

    return model


# =====================================================
# LOSS
# =====================================================

class GeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred: [lat, sin_lon, cos_lon]
        lat_loss = self.mse(pred[:,0], target[:,0])

        lon_loss = self.mse(pred[:,1:], target[:,1:])

        return lat_loss + lon_loss
    

class KmLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = [lat_norm, sin_lon, cos_lon]
        # target = [lat_norm, sin_lon, cos_lon]

        pred_lat = pred[:,0] * 90.0
        true_lat = target[:,0] * 90.0

        pred_lon = torch.atan2(pred[:,1], pred[:,2]) * 180.0 / math.pi
        true_lon = torch.atan2(target[:,1], target[:,2]) * 180.0 / math.pi

        return haversine_torch(
            pred_lat,
            pred_lon,
            true_lat,
            true_lon
        ).mean()


# =====================================================
# TRAIN
# =====================================================

def train():

    full_train = GeoDataset(CSV_FILE, IMG_DIR, train=True)
    full_val   = GeoDataset(CSV_FILE, IMG_DIR, train=False)

    n = len(full_train)
    n_train = int(n * 0.9)
    n_val = n - n_train

    train_idx, val_idx = random_split(range(n), [n_train, n_val])

    train_ds = torch.utils.data.Subset(full_train, train_idx.indices)
    val_ds   = torch.utils.data.Subset(full_val, val_idx.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = build_model().to(DEVICE)

    criterion = KmLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_loss = 1e9
    patience = 6
    bad_epochs = 0

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()

        train_loss = 0

        for imgs, targets in tqdm(train_loader):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                pred = model(imgs)
                loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if torch.isnan(loss):
                print("NaN loss detected.")

        train_loss /= len(train_loader)

        # ---------------- VAL ----------------
        model.eval()

        val_loss = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader):
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)

                pred = model(imgs)
                loss = criterion(pred, targets)

                val_loss += loss.item()


        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} km | "
        )

        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_FILE)
            print("Saved best model.")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break


if __name__ == "__main__":
    train()