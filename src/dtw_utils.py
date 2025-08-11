"""Utilities for handling keypoint sequences and DTW."""
import numpy as np
from dtaidistance import dtw_ndim


def center_scale(seq: np.ndarray) -> np.ndarray:
    """Center by hips and scale by torso length."""
    seq = np.array(seq)[..., :2]
    hips = seq[:, 11:13].mean(axis=1, keepdims=True)
    seq = seq - hips
    shoulders = seq[:, 5:7].mean(axis=1, keepdims=True)
    torso = np.linalg.norm(shoulders - hips, axis=-1, keepdims=True)
    torso[torso == 0] = 1.0
    seq = seq / torso[:, None, :]
    return seq


def flatten_series(seq: np.ndarray) -> np.ndarray:
    """Flatten a sequence of keypoints into 2D array [T, 34]."""
    seq = np.array(seq)
    return seq.reshape(seq.shape[0], -1)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute DTW distance between two sequences."""
    a = flatten_series(center_scale(seq_a))
    b = flatten_series(center_scale(seq_b))
    return dtw_ndim.distance(a, b)


def sliding_window_dtw(seq: np.ndarray, template: np.ndarray) -> float:
    """Slide template over seq and return best DTW distance."""
    tlen = len(template)
    best = float("inf")
    for i in range(0, len(seq) - tlen + 1):
        dist = dtw_distance(seq[i : i + tlen], template)
        if dist < best:
            best = dist
    return best
