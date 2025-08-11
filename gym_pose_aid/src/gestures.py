"""Simple gesture detector for MoveNet keypoints."""
from collections import deque
import numpy as np


class SimpleGestureDetector:
    """Detects basic gestures using heuristic rules."""

    def __init__(self, smooth: int = 5, conf_thr: float = 0.3):
        self.buffer = deque(maxlen=smooth)
        self.conf_thr = conf_thr
        self.baseline_hip = None
        self.arm_state = False
        self.squat_state = False
        self.sit_state = False

    @staticmethod
    def _angle(a, b, c):
        """Returns angle ABC in degrees."""
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def update(self, kpts: np.ndarray):
        """Update detector with new keypoints.

        Args:
            kpts: Array (17,3) of keypoints (y,x,score).

        Returns:
            Dict with booleans for events: arm_raise, squat, sit_down.
        """
        self.buffer.append(kpts)
        avg = np.mean(self.buffer, axis=0)
        events = {"arm_raise": False, "squat": False, "sit_down": False}

        lw, rw = avg[9], avg[10]
        ls, rs = avg[5], avg[6]
        lh, rh = avg[11], avg[12]
        lk, rk = avg[13], avg[14]
        la, ra = avg[15], avg[16]

        if self.baseline_hip is None:
            self.baseline_hip = (lh[0] + rh[0]) / 2

        # Arm raise
        arm_condition = False
        if lw[2] > self.conf_thr and ls[2] > self.conf_thr and lw[0] < ls[0] - 0.05:
            arm_condition = True
        if rw[2] > self.conf_thr and rs[2] > self.conf_thr and rw[0] < rs[0] - 0.05:
            arm_condition = True
        if arm_condition and not self.arm_state:
            events["arm_raise"] = True
        self.arm_state = arm_condition

        # Squat
        angles = []
        if all(p[2] > self.conf_thr for p in [lh, lk, la]):
            angles.append(self._angle(lh[:2], lk[:2], la[:2]))
        if all(p[2] > self.conf_thr for p in [rh, rk, ra]):
            angles.append(self._angle(rh[:2], rk[:2], ra[:2]))
        hip_y = (lh[0] + rh[0]) / 2
        hip_drop = hip_y - self.baseline_hip
        squat_condition = angles and np.mean(angles) < 100 and hip_drop > 0.06
        if squat_condition and not self.squat_state:
            events["squat"] = True
        self.squat_state = squat_condition

        # Sit down
        hip_var = np.var([f[11:13, 0].mean() for f in self.buffer]) if len(self.buffer) == self.buffer.maxlen else np.inf
        sit_condition = angles and 80 < np.mean(angles) < 100 and hip_drop > 0.2 and hip_var < 1e-4
        if sit_condition and not self.sit_state:
            events["sit_down"] = True
        self.sit_state = sit_condition

        return events
