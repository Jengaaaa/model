import os

import torch
from torch.utils.data import Dataset

from configs.config import CONFIG
from data.video_utils import load_video_frames, load_precomputed_frames
from data.audio_utils import load_mfcc_chunks


class AVECDataset(Dataset):
    def __init__(self, base_path: str, mode: str = "Training", tasks=None):
        if tasks is None:
            tasks = CONFIG["dataset"].get("tasks", ["Freeform", "Northwind"])
        self.base_path = base_path
        self.mode = mode
        self.tasks = tasks

        # label directory
        if mode in ["Training", "Development"]:
            self.label_dir = os.path.join(
                base_path.replace("AVEC2014_AudioVisual", "Final_Labels"),
                f"{mode}_DepressionLabels",
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
                        label_file = os.path.join(
                            self.label_dir, f"{prefix}_Depression.csv"
                        )
                        if not os.path.exists(label_file):
                            continue

                    self.samples.append((task, prefix))

        self.samples = sorted(self.samples, key=lambda x: x[1])
        self.is_test = mode == "Testing"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task, prefix = self.samples[idx]

        video_dir = os.path.join(self.base_path, "Video", self.mode, task)
        audio_dir = os.path.join(self.base_path, "Audio", self.mode, task)
        frames_dir = os.path.join(self.base_path, "Frames", self.mode, task, prefix)

        vpath = os.path.join(video_dir, f"{prefix}_{task}_video.mp4")
        apath = os.path.join(audio_dir, f"{prefix}_{task}_audio.wav")

        # 1) precomputed 프레임이 있으면 우선 사용
        if os.path.isdir(frames_dir):
            frames = load_precomputed_frames(frames_dir)
        else:
            # 2) 아니면 비디오에서 직접 샘플링
            frames = load_video_frames(vpath)
        mfcc_chunks = load_mfcc_chunks(apath)

        # Testing
        if self.is_test:
            return frames, mfcc_chunks, f"{task}_{prefix}"

        # Training / Dev (full prefix match)
        label_file = os.path.join(self.label_dir, f"{prefix}_Depression.csv")

        with open(label_file, "r") as f:
            label = float(f.read().strip())

        return frames, mfcc_chunks[0], torch.tensor(label, dtype=torch.float32)

