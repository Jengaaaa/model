import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import CONFIG
from data.dataset import AVECDataset
from models.fusion import FusionNet


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - 피어슨 상관계수
    """
    pred = pred - pred.mean()
    target = target - target.mean()

    num = (pred * target).sum()
    denom = torch.sqrt((pred**2).sum()) * torch.sqrt((target**2).sum())
    denom = denom + 1e-8

    return 1 - num / denom


def _build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    base_path = CONFIG["dataset"]["base_path"]
    tasks = CONFIG["dataset"].get("tasks", ["Freeform", "Northwind"])
    batch_size = CONFIG["training"]["batch_size"]
    num_workers = CONFIG["dataset"].get("num_workers", 0)

    train_data = AVECDataset(base_path, mode="Training", tasks=tasks)
    dev_data = AVECDataset(base_path, mode="Development", tasks=tasks)
    test_data = AVECDataset(base_path, mode="Testing", tasks=tasks)

    print("\n=== Dataset Counts ===")
    print("Training samples   :", len(train_data))
    print("Development samples:", len(dev_data))
    print("Testing samples    :", len(test_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, dev_loader


def train() -> None:
    """
    FusionNet 학습 엔트리 포인트.
    CONFIG 에 정의된 하이퍼파라미터와 경로를 사용한다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    epochs = CONFIG["training"]["epochs"]
    # config에서 문자열("5e-5")로 들어와도 항상 float 로 캐스팅
    learning_rate = float(CONFIG["training"]["learning_rate"])
    use_amp = CONFIG["training"].get("use_amp", True) and (device == "cuda")

    train_loader, dev_loader = _build_dataloaders()

    model = FusionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_rmse = float("inf")

    ckpt_dir = CONFIG["model"]["checkpoint_dir"]
    ckpt_name = CONFIG["model"]["checkpoint_name"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    for epoch in range(1, epochs + 1):
        print(f"\n\n================ EPOCH {epoch} / {epochs} ================")

        # ----------------------- TRAIN ----------------------------
        model.train()
        train_losses = []
        train_preds, train_gts = [], []

        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")

        for batch in loop:
            if batch is None:
                continue

            frames, mfcc, labels = batch
            frames, mfcc, labels = (
                frames.to(device),
                mfcc.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 출력/라벨을 항상 1D 벡터 형태로 맞춰서
                # batch_size=1일 때도 스칼라로 squeeze 되지 않도록 한다.
                out = model(frames, mfcc).view(-1)
                labels_vec = labels.view(-1)

                loss_mse = mse_loss(out, labels_vec)
                loss_p = pearson_loss(out, labels_vec)
                loss = loss_mse + 0.2 * loss_p

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            # numpy로 변환 후 리스트로 풀어서 extend
            train_preds.extend(out.detach().cpu().numpy().tolist())
            train_gts.extend(labels_vec.detach().cpu().numpy().tolist())

            loop.set_postfix(loss=loss.item())

        # ---- Train Metrics ----
        train_preds_arr = np.array(train_preds)
        train_gts_arr = np.array(train_gts)

        train_rmse = np.sqrt(np.mean((train_preds_arr - train_gts_arr) ** 2))
        train_mae = np.mean(np.abs(train_preds_arr - train_gts_arr))
        train_corr = pearsonr(train_preds_arr, train_gts_arr)[0]

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

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(frames, mfcc).view(-1)
                    labels_vec = labels.view(-1)

                val_preds.extend(out.detach().cpu().numpy().tolist())
                val_gts.extend(labels_vec.detach().cpu().numpy().tolist())

        # numpy 배열로 변환 (float)
        val_preds_arr = np.array(val_preds, dtype=float)
        val_gts_arr = np.array(val_gts, dtype=float)

        # NaN/inf 값 제거 후에만 메트릭 계산
        mask = np.isfinite(val_preds_arr) & np.isfinite(val_gts_arr)
        n_total = val_preds_arr.shape[0]
        n_valid = int(mask.sum())

        if n_valid == 0:
            print(
                "\n[Warning] Validation contains no finite values "
                f"(all NaN/inf). total={n_total}"
            )
            val_rmse = float("nan")
            val_mae = float("nan")
            val_corr = float("nan")
        else:
            # 일부 샘플이 NaN/inf 라서 걸러지더라도
            # 조용히(mask만 적용) 진행하고 추가 로그는 출력하지 않는다.
            vp = val_preds_arr[mask]
            vg = val_gts_arr[mask]

            val_rmse = np.sqrt(np.mean((vp - vg) ** 2))
            val_mae = np.mean(np.abs(vp - vg))

            # 상수 배열이면 pearsonr 가 nan을 줄 수 있으므로 방어적으로 처리
            try:
                val_corr = pearsonr(vp, vg)[0]
            except Exception as e:
                print(f"[Warning] pearsonr failed on validation data: {e}")
                val_corr = float("nan")

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

    print("\n Training Finished!")
    print(f"Best model saved to: {ckpt_path}")

