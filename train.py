import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from dataset import AVECDataset
from fusion import FusionNet   # ← 네가 만든 모델


# ============================================================
# 1) Pearson Loss (1 - correlation)
# ============================================================
def pearson_loss(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()

    num = (pred * target).sum()
    denom = torch.sqrt((pred ** 2).sum()) * torch.sqrt((target ** 2).sum())
    denom = denom + 1e-8

    return 1 - num / denom


# ============================================================
# 2) Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# 3) Dataset 로드
# ============================================================
BASE = "/home/Pdanova/Dataset/Dataset/AVEC2014_AudioVisual"

train_data = AVECDataset(BASE, mode="Training")
dev_data   = AVECDataset(BASE, mode="Development")
test_data  = AVECDataset(BASE, mode="Testing")

print("\n=== Dataset Counts ===")
print("Training samples   :", len(train_data))
print("Development samples:", len(dev_data))
print("Testing samples    :", len(test_data))


# ============================================================
# 4) DataLoader
# ============================================================
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_data,   batch_size=4, shuffle=False, num_workers=0)


# ============================================================
# 5) Model / Optimizer
# ============================================================
model = FusionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
mse_loss = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

EPOCHS = 20
best_val_rmse = float("inf")

os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/best_fusion.pth"


# ============================================================
# 6) Training Loop
# ============================================================
for epoch in range(1, EPOCHS + 1):
    print(f"\n\n================ EPOCH {epoch} / {EPOCHS} ================")

    # ----------------------- TRAIN ----------------------------
    model.train()
    train_losses = []
    train_preds, train_gts = [], []

    loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")

    for batch in loop:
        if batch is None:
            continue

        frames, mfcc, labels = batch
        frames, mfcc, labels = frames.to(device), mfcc.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(frames, mfcc).squeeze()
            loss_mse = mse_loss(out, labels)
            loss_p = pearson_loss(out, labels)
            loss = loss_mse + 0.2 * loss_p

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        train_preds.extend(out.detach().cpu().numpy())
        train_gts.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    # ---- Train Metrics ----
    train_preds = np.array(train_preds)
    train_gts = np.array(train_gts)

    train_rmse = np.sqrt(np.mean((train_preds - train_gts) ** 2))
    train_mae = np.mean(np.abs(train_preds - train_gts))
    train_corr = pearsonr(train_preds, train_gts)[0]


    # ---------------------- VALIDATION ------------------------
    model.eval()
    val_preds, val_gts = [], []

    with torch.no_grad():
        for batch in dev_loader:
            if batch is None:
                continue

            frames, mfcc, labels = batch
            frames, mfcc = frames.to(device), mfcc.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(frames, mfcc).squeeze()

            val_preds.extend(out.detach().cpu().numpy())
            val_gts.extend(labels.detach().cpu().numpy())

    val_preds = np.array(val_preds)
    val_gts = np.array(val_gts)

    val_rmse = np.sqrt(np.mean((val_preds - val_gts) ** 2))
    val_mae = np.mean(np.abs(val_preds - val_gts))
    val_corr = pearsonr(val_preds, val_gts)[0]


    print(
        f"\n[Epoch {epoch}] "
        f"Train RMSE={train_rmse:.3f} MAE={train_mae:.3f} Corr={train_corr:.3f} || "
        f"Val RMSE={val_rmse:.3f} MAE={val_mae:.3f} Corr={val_corr:.3f}"
    )

    # ---------------------- BEST MODEL SAVE -------------------
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Best model updated! (Val RMSE={val_rmse:.3f})")


print("\n🎉 Training Finished!")
print(f"Best model saved to: {ckpt_path}")
