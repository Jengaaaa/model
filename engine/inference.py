import os
import subprocess
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import CONFIG
from data.dataset import AVECDataset
from data.audio_utils import load_mfcc_chunks
from data.video_utils import load_video_frames
from models.fusion import FusionNet


AUDIO_SR = CONFIG["audio"]["sample_rate"]
NUM_FRAMES = CONFIG["model"]["num_frames"]
CKPT_DIR = CONFIG["model"]["checkpoint_dir"]
CKPT_NAME = CONFIG["model"]["checkpoint_name"]
CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME)

_THR = CONFIG.get("inference", {}).get("depression_score_thresholds", {})
T_NORMAL = float(_THR.get("normal", 10.0))
T_MILD = float(_THR.get("mild", 20.0))
T_MODERATE = float(_THR.get("moderate", 30.0))
T_SEVERE = float(_THR.get("severe", 40.0))


def _load_model(device: str | None = None) -> Tuple[FusionNet, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = FusionNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"✔ Loaded checkpoint from {CKPT_PATH}")

    return model, device


def run_test(output_csv: str = "predictions_test.csv") -> List[dict]:
    """
    Testing 세트 전체에 대해 예측을 수행하고 CSV로 저장한다.
    """
    model, device = _load_model()

    base_path = CONFIG["dataset"]["base_path"]
    tasks = CONFIG["dataset"].get("tasks", ["Freeform", "Northwind"])

    test_dataset = AVECDataset(base_path, mode="Testing", tasks=tasks)
    loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    def predict_sample(frames, mfcc_chunks):
        frames = frames.to(device)
        preds = []

        with torch.no_grad():
            for mfcc in mfcc_chunks:
                mfcc = mfcc.unsqueeze(0).to(device)
                out = model(frames, mfcc)
                preds.append(out.item())

        return sum(preds) / len(preds)

    results: List[dict] = []

    for frames, mfcc_chunks, prefix in tqdm(loader_test, desc="[Predict] Testing"):
        mfcc_chunks = mfcc_chunks[0]  # unwrap list

        pred = predict_sample(frames, mfcc_chunks)

        results.append(
            {
                "prefix": prefix[0],
                "prediction": pred,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"✔ Saved {output_csv}")
    return results


def _extract_wav_from_mp4(mp4_path: str, wav_path: str) -> None:
    cmd = f'ffmpeg -y -i "{mp4_path}" -vn -ac 1 -ar {AUDIO_SR} "{wav_path}"'
    subprocess.call(cmd, shell=True)


def interpret_score(s: float) -> str:
    if s < T_NORMAL:
        return "정상 범위 (우울감 거의 없음)"
    elif s < T_MILD:
        return "경미한 우울 수준 가능성"
    elif s < T_MODERATE:
        return "중등도 우울 가능성"
    elif s < T_SEVERE:
        return "높은 우울 수준"
    else:
        return "매우 높은 우울 수준"


def predict_from_video(mp4_path: str, model_path: str | None = None) -> Tuple[float, str]:
    """
    단일 mp4 비디오 파일에 대해 우울 점수를 예측한다.
    """
    if model_path is not None:
        ckpt_path = model_path
    else:
        ckpt_path = CKPT_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load model
    print("[INFO] Loading model...")
    model = FusionNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✔ Model loaded from {ckpt_path}")

    # Extract WAV
    wav_path = mp4_path.replace(".mp4", ".wav")
    print("[INFO] Extracting audio...")
    _extract_wav_from_mp4(mp4_path, wav_path)

    # Load video
    print("[INFO] Loading video frames...")
    frames = load_video_frames(mp4_path, num_frames=NUM_FRAMES)
    frames = frames.unsqueeze(0).to(device)
    print("✔ Video loaded.")

    # Load audio MFCC
    print("[INFO] Loading MFCC chunks...")
    mfcc_chunks = load_mfcc_chunks(wav_path)
    print(f"✔ MFCC chunks: {len(mfcc_chunks)}")

    # Predict
    print("[INFO] Running prediction...")
    preds: List[float] = []

    with torch.no_grad():
        for chunk in tqdm(mfcc_chunks, desc="Predicting"):
            chunk = chunk.unsqueeze(0).to(device)
            out = model(frames, chunk)
            preds.append(out.item())

    score = sum(preds) / len(preds)
    level = interpret_score(score)

    print("\n===================================")
    print(f" Predicted Depression Score: {score:.2f}")
    print(f" Interpretation: {level}")
    print("===================================\n")

    # Remove WAV
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print("[INFO] Removed temp wav file.")

    return score, level

