import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class TemporalAttention(nn.Module):       #시간정보를 반영한 어텐션
    
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

class SimpleAudioEncoder(nn.Module):       #temporal attention을 이용한 오디오 인코더
    def __init__(self, n_mfcc=40, out_dim=256):
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

    def forward(self, mfcc):
        """
        mfcc: (40, T) or (B, 40, T)
        """
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)  # (1, 40, T)

        x = self.cnn(mfcc)           # (B, 256, T)
        x = x.permute(0, 2, 1)       # (B, T, 256)  ← attention 입력

        x, attn = self.temporal_attn(x)  # (B, 256), (B, T)
        x = self.fc(x)                   # (B, 256)
        return x