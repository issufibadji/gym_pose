"""Utilities for MoveNet inference and drawing."""
from typing import Iterable
import numpy as np
import tensorflow as tf
import cv2

# Edges between keypoints as (start, end)
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8,10), (5, 6), (5,11),
    (6,12), (11,12), (11,13), (13,15),
    (12,14), (14,16)
]


def run_movenet(model: tf.types.experimental.GenericFunction, rgb: np.ndarray, input_size: int = 192) -> np.ndarray:
    """Runs MoveNet on an RGB image.

    Args:
        model: Loaded TF-Hub MoveNet model.
        rgb: RGB image ndarray.
        input_size: Model input size (192 or 256).

    Returns:
        Keypoints with shape [1, 1, 17, 3] (y, x, score).
    """
    image = tf.image.resize_with_pad(tf.expand_dims(rgb, axis=0), input_size, input_size)
    input_image = tf.cast(image, dtype=tf.int32)
    outputs = model(input_image)
    return outputs['output_0'].numpy()


def draw_skeleton(rgb: np.ndarray, kpts: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """Draws keypoints and edges on the image using OpenCV."""
    image = rgb.copy()
    h, w, _ = image.shape
    for y, x, c in kpts:
        if c > conf_thr:
            cv2.circle(image, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)
    for a, b in EDGES:
        y1, x1, c1 = kpts[a]
        y2, x2, c2 = kpts[b]
        if c1 > conf_thr and c2 > conf_thr:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    return image
