import torch
import torch.nn as nn
import timm
from dataset import load_video_frames, load_mfcc_chunks
from mtcnn import MTCNN
import cv2
import numpy as np

detector = MTCNN()  # MTCNN 얼굴 검출기 초기화

def crop_face_mtcnn(frame, target_size=299):  #얼굴 검출기
    # frame: (H, W, 3) BGR (cv2) 또는 RGB
    detections = detector.detect_faces(frame)
    if len(detections) == 0:
        return None  # 얼굴 없음
    
    # 가장 큰 얼굴 하나 선택
    boxes = [d["box"] for d in detections]
    areas = [w*h for (_, _, w, h) in boxes]
    idx = int(np.argmax(areas))
    x, y, w, h = boxes[idx]
    
    x, y = max(0, x), max(0, y)
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (target_size, target_size))
    return face


class TemporalAttention(nn.Module):
    """
    간단한 temporal attention:
    - 입력: x (B, T, D)
    - 출력: out (B, D), weight (B, T)
    """
    def __init__(self, dim, attn_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(dim, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        h = torch.tanh(self.fc1(x))          # (B, T, attn_dim)
        score = self.fc2(h).squeeze(-1)      # (B, T)
        weight = torch.softmax(score, dim=1) # (B, T)
        out = torch.sum(x * weight.unsqueeze(-1), dim=1)  # (B, D)
        return out, weight


class SimpleVideoEncoder(nn.Module):
    """
    - 입력: frames (B, T, 3, 299, 299)  ← dataset.load_video_frames 결과를 batch로 묶어서 사용
    - 처리:
        1) Inception-ResNet-V2 백본으로 프레임별 feature 추출 (1536)
        2) fc로 256차원으로 축소
        3) TemporalAttention으로 프레임 중요도 학습 후 가중합
    - 출력: video_embedding (B, 256)
    """
    def __init__(self, model_name="inception_resnet_v2", out_dim=256):
        super().__init__()

        # timm에서 백본 생성
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.classifier = nn.Identity()  # 분류층 제거

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, out_dim)  # 1536 -> 256

        # temporal attention 모듈 추가
        self.temporal_attn = TemporalAttention(out_dim)

    def forward(self, frames):
        """
        frames: (B, T, 3, 299, 299)
        """
        B, T, C, H, W = frames.shape

        # (B*T, 3, 299, 299)
        flat_frames = frames.view(B * T, C, H, W)

        # CNN feature 추출
        feat = self.backbone.forward_features(flat_frames)
        feat = self.pool(feat).view(B * T, -1)   # (B*T, 1536)
        feat = self.fc(feat)                     # (B*T, 256)

        # (B, T, 256)으로 reshape
        feat = feat.view(B, T, -1)

        # temporal attention으로 비디오 임베딩
        video_embedding, attn_weight = self.temporal_attn(feat)  # (B, 256), (B, T)

        return video_embedding