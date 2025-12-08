import os
import subprocess

from tqdm import tqdm

from configs.config import CONFIG
from data.video_utils import save_face_frames_from_video


def extract_wav_from_mp4(mp4_path: str, wav_path: str, sample_rate: int) -> None:
    """
    mp4 영상에서 mono wav 로 오디오만 추출.
    """
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        wav_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main() -> None:
    base_path = CONFIG["dataset"]["base_path"]
    tasks = CONFIG["dataset"].get("tasks", ["Freeform", "Northwind"])
    sample_rate = CONFIG["audio"]["sample_rate"]

    modes = ["Training", "Development", "Testing"]

    for mode in modes:
        for task in tasks:
            video_dir = os.path.join(base_path, "Video", mode, task)
            audio_dir = os.path.join(base_path, "Audio", mode, task)
            frames_root = os.path.join(base_path, "Frames", mode, task)

            if not os.path.isdir(video_dir):
                continue

            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(frames_root, exist_ok=True)

            mp4_files = [
                f
                for f in os.listdir(video_dir)
                if f.endswith(f"_{task}_video.mp4")
            ]
            if not mp4_files:
                continue

            print(f"\n[{mode} - {task}] {len(mp4_files)} videos")

            for f in tqdm(mp4_files, desc=f"[{mode}/{task}] precompute"):
                prefix = f.replace(f"_{task}_video.mp4", "")
                vpath = os.path.join(video_dir, f)

                # 1) 오디오 추출
                wav_path = os.path.join(audio_dir, f"{prefix}_{task}_audio.wav")
                if not os.path.exists(wav_path):
                    extract_wav_from_mp4(vpath, wav_path, sample_rate)

                # 2) 얼굴 프레임 이미지 저장
                frames_dir = os.path.join(frames_root, prefix)
                # 이미 존재하면 스킵 (원하면 강제로 다시 만들도록 옵션 추가 가능)
                if not os.listdir(frames_dir) if os.path.isdir(frames_dir) else True:
                    save_face_frames_from_video(vpath, frames_dir)


if __name__ == "__main__":
    main()


