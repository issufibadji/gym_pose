"""Video IO utilities."""
import os
import cv2
import imageio
import numpy as np
from urllib.request import urlretrieve


SAMPLE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/squat.mp4"


def read_video(path: str):
    """Reads video and returns fps and list of RGB frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return fps, frames


def save_video(path: str, frames, fps: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)


def save_gif(path: str, frames, fps: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", fps=fps)


def download_sample(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample.mp4")
    if not os.path.exists(out_path):
        urlretrieve(SAMPLE_URL, out_path)
    return out_path
