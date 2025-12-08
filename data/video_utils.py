"""
data/video_utils.py
비디오 프레임 로딩 및 얼굴 크롭 유틸.
"""

import numpy as np
import cv2
import torch
from mtcnn import MTCNN
from PIL import Image

from configs.config import CONFIG


_VID_CFG = CONFIG["video"]
TARGET_SIZE = _VID_CFG["target_size"]
NUM_FRAMES = CONFIG["model"]["num_frames"]

_DETECTOR = MTCNN()


def crop_face_mtcnn(frame, target_size: int = TARGET_SIZE):
    """
    frame: RGB (H, W, 3)
    return: RGB (target_size, target_size, 3) or None
    """
    detections = _DETECTOR.detect_faces(frame)
    if len(detections) == 0:
        return None

    # 가장 큰 얼굴 하나 선택
    boxes = [d["box"] for d in detections]
    areas = [w * h for (_, _, w, h) in boxes]
    idx = int(np.argmax(areas))
    x, y, w, h = boxes[idx]

    x, y = max(0, x), max(0, y)
    face = frame[y : y + h, x : x + w]
    face = cv2.resize(face, (target_size, target_size))
    return face


def load_video_frames(path: str, num_frames: int = NUM_FRAMES, target_size: int = TARGET_SIZE):
    """
    path: 비디오 경로
    return: (T, 3, H, W) torch.FloatTensor, [0,1] 범위
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []

    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            # 프레임 읽기 실패 시 빈 이미지
            f = np.zeros((target_size, target_size, 3), np.uint8)
            frames.append(f)
            continue

        # BGR -> RGB
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        # 얼굴 crop 시도
        face = crop_face_mtcnn(f, target_size=target_size)
        if face is None:
            # 얼굴 못 찾으면 전체 프레임 resize
            face = cv2.resize(f, (target_size, target_size))

        frames.append(face)

    cap.release()

    frames = np.stack(frames)  # (T, H, W, 3)
    # (T, 3, H, W) + 정규화
    return torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0


def save_face_frames_from_video(
    video_path: str,
    out_dir: str,
    num_frames: int = NUM_FRAMES,
    target_size: int = TARGET_SIZE,
) -> None:
    """
    비디오에서 얼굴 crop된 프레임들을 이미지(jpg)로 저장.
      - video_path: 원본 mp4 경로
      - out_dir: 프레임을 저장할 디렉토리 (없으면 생성)
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    for j, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            f = np.zeros((target_size, target_size, 3), np.uint8)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            face = crop_face_mtcnn(f, target_size=target_size)
            if face is None:
                face = cv2.resize(f, (target_size, target_size))
            f = face

        img = Image.fromarray(f)
        img.save(os.path.join(out_dir, f"frame_{j:03d}.jpg"))

    cap.release()


def load_precomputed_frames(frames_dir: str, max_frames: int = NUM_FRAMES) -> torch.Tensor:
    """
    precompute 단계에서 저장한 프레임 이미지들을 읽어와 텐서로 변환.
      - frames_dir: frame_000.jpg ... 이 들어있는 디렉토리
      - max_frames: 사용할 프레임 수 (부족하면 마지막 프레임으로 반복)
    """
    files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))
    )
    if not files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    frames = []
    for i in range(max_frames):
        fname = files[min(i, len(files) - 1)]
        path = os.path.join(frames_dir, fname)
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        frames.append(arr)

    frames = np.stack(frames)  # (T, H, W, 3)
    return torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

