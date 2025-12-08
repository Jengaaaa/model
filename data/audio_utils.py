"""
data/audio_utils.py
공통 MFCC chunk 로딩 유틸.
하이퍼파라미터는 configs.CONFIG 에서 읽는다.
"""

import numpy as np
import torch
import librosa

from configs.config import CONFIG


_AUDIO_CFG = CONFIG["audio"]
SR = _AUDIO_CFG["sample_rate"]
N_MFCC = _AUDIO_CFG["n_mfcc"]
CHUNK_LEN = _AUDIO_CFG["chunk_len"]
HOP_RATIO = _AUDIO_CFG.get("hop_ratio", 1.0)
N_MELS = _AUDIO_CFG.get("n_mels", 128)
FMAX = _AUDIO_CFG.get("fmax", 8000)


def load_mfcc_chunks(
    path: str,
    n_mfcc: int = N_MFCC,
    chunk_len: int = CHUNK_LEN,
    hop_ratio: float = HOP_RATIO,
):
    """
    path: wav 파일 경로
    chunk_len: one chunk covers ~30s of MFCC (depending on hop size)
    hop_ratio: 1.0
    """

    y, sr = librosa.load(path, sr=SR)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)  # (n_mfcc, T)

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

