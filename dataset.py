import os
import cv2
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from mtcnn import MTCNN
detector = MTCNN()

def crop_face_mtcnn(frame, target_size=299):
    """
    frame: RGB (H, W, 3)
    return: RGB (target_size, target_size, 3) or None
    """
    detections = detector.detect_faces(frame)
    if len(detections) == 0:
        return None

    # 가장 큰 얼굴 하나 선택
    boxes = [d["box"] for d in detections]
    areas = [w * h for (_, _, w, h) in boxes]
    idx = int(np.argmax(areas))
    x, y, w, h = boxes[idx]

    x, y = max(0, x), max(0, y)
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (target_size, target_size))
    return face


def load_video_frames(path, num_frames=8, target_size=299):
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





def load_mfcc_chunks(path, n_mfcc=40, chunk_len=3000, hop_ratio=1.0):
    y, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)

    T = mfcc.shape[1]
    hop = int(chunk_len * hop_ratio)

    chunks = []
    for start in range(0, T, hop):
        end = start + chunk_len
        piece = mfcc[:, start:end]

        if piece.shape[1] < chunk_len:
            pad = chunk_len - piece.shape[1]
            piece = np.pad(piece, ((0, 0), (0, pad)), mode="constant")

        chunks.append(torch.tensor(piece, dtype=torch.float32))
        if end >= T:
            break

    return chunks


class AVECDataset(Dataset):
    def __init__(self, base_path, mode="Training", tasks=["Freeform", "Northwind"]):
        self.base_path = base_path
        self.mode = mode
        self.tasks = tasks

        # label directory
        if mode in ["Training", "Development"]:
            self.label_dir = os.path.join(
                base_path.replace("AVEC2014_AudioVisual", "Final_Labels"),
                f"{mode}_DepressionLabels"
            )
        else:
            self.label_dir = None

        # sample collection
        self.samples = []
        for task in tasks:
            video_dir = os.path.join(base_path, "Video", mode, task)
            if not os.path.isdir(video_dir):
                continue

            for f in os.listdir(video_dir):
                if f.endswith(f"_{task}_video.mp4"):
                    prefix = f.replace(f"_{task}_video.mp4", "")

                    # Training / Dev → check full-prefix label
                    if self.label_dir:
                        label_file = os.path.join(self.label_dir, f"{prefix}_Depression.csv")
                        if not os.path.exists(label_file):
                            continue

                    self.samples.append((task, prefix))

        self.samples = sorted(self.samples, key=lambda x: x[1])
        self.is_test = (mode == "Testing")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task, prefix = self.samples[idx]

        video_dir = os.path.join(self.base_path, "Video", self.mode, task)
        audio_dir = os.path.join(self.base_path, "Audio", self.mode, task)

        vpath = os.path.join(video_dir, f"{prefix}_{task}_video.mp4")
        apath = os.path.join(audio_dir, f"{prefix}_{task}_audio.wav")

        frames = load_video_frames(vpath)
        mfcc_chunks = load_mfcc_chunks(apath, chunk_len=3000)

        # Testing
        if self.is_test:
            return frames, mfcc_chunks, f"{task}_{prefix}"

        # Training / Dev (full prefix match)
        label_file = os.path.join(self.label_dir, f"{prefix}_Depression.csv")

        with open(label_file, "r") as f:
            label = float(f.read().strip())

        return frames, mfcc_chunks[0], torch.tensor(label, dtype=torch.float32)
