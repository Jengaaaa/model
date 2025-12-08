import torch
import torch.nn as nn
import timm


class TemporalAttention(nn.Module):
    """
    간단한 temporal attention:
    - 입력: x (B, T, D)
    - 출력: out (B, D), weight (B, T)
    """

    def __init__(self, dim: int, attn_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        h = torch.tanh(self.fc1(x))  # (B, T, attn_dim)
        score = self.fc2(h).squeeze(-1)  # (B, T)
        weight = torch.softmax(score, dim=1)  # (B, T)
        out = torch.sum(x * weight.unsqueeze(-1), dim=1)  # (B, D)
        return out, weight


class SimpleVideoEncoder(nn.Module):
    """
    - 입력: frames (B, T, 3, 299, 299)
    - 처리:
        1) Inception-ResNet-V2 백본으로 프레임별 feature 추출 (1536)
        2) fc로 256차원으로 축소
        3) TemporalAttention으로 프레임 중요도 학습 후 가중합
    - 출력: video_embedding (B, 256)
    """

    def __init__(self, model_name: str = "inception_resnet_v2", out_dim: int = 256):
        super().__init__()

        # timm에서 백본 생성
        self.backbone = timm.create_model(model_name, pretrained=True)
        # timm 모델에 따라 classifier 속성 이름이 다를 수 있으므로 방어적으로 처리
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, out_dim)  # 1536 -> 256

        # temporal attention 모듈 추가
        self.temporal_attn = TemporalAttention(out_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, 299, 299)
        """
        B, T, C, H, W = frames.shape

        # (B*T, 3, 299, 299)
        flat_frames = frames.view(B * T, C, H, W)

        # CNN feature 추출
        feat = self.backbone.forward_features(flat_frames)
        feat = self.pool(feat).view(B * T, -1)  # (B*T, 1536)
        feat = self.fc(feat)  # (B*T, 256)

        # (B, T, 256)으로 reshape
        feat = feat.view(B, T, -1)

        # temporal attention으로 비디오 임베딩
        video_embedding, _ = self.temporal_attn(feat)  # (B, 256), (B, T)

        return video_embedding


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)


class SimpleAudioEncoder(nn.Module):
    """
    temporal attention을 이용한 오디오 인코더
    입력: mfcc (B, n_mfcc, T) 또는 (n_mfcc, T)
    출력: (B, 256)
    """

    def __init__(self, n_mfcc: int = 40, out_dim: int = 256):
        super().__init__()

        # 1D CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.temporal_attn = TemporalAttention(dim=256)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        mfcc: (40, T) or (B, 40, T)
        """
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)  # (1, 40, T)

        x = self.cnn(mfcc)  # (B, 256, T)
        x = x.permute(0, 2, 1)  # (B, T, 256)  ← attention 입력

        x, _ = self.temporal_attn(x)  # (B, 256), (B, T)
        x = self.fc(x)  # (B, 256)
        return x

