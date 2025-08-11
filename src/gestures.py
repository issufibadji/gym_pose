from collections import deque
import numpy as np

def _as_17x3(arr):
    """Normaliza kpts para shape (17,3) [y,x,score]. Aceita:
    (17,3), (1,17,3), (1,1,17,3)."""
    a = np.asarray(arr)
    if a.shape == (17, 3):
        return a
    if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == 17 and a.shape[2] == 3:
        return a[0]
    if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1 and a.shape[2] == 17 and a.shape[3] == 3:
        return a[0, 0]
    raise ValueError(f"Keypoints com shape inesperado: {a.shape}")

def _angle_xy(A, B, C):
    # recebe pontos em (x,y)
    BA = A - B
    BC = C - B
    denom = (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    cosang = np.dot(BA, BC) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

class SimpleGestureDetector:
    def __init__(self, smooth=5, conf_thr=0.3):
        self.q = deque(maxlen=smooth)  # guarda (y,x) normalizados
        self.hips_baseline = None
        self.conf_thr = conf_thr

    def update(self, kpts_any_shape):
        kp = _as_17x3(kpts_any_shape)         # (17,3) [y,x,score]
        conf = kp[:, 2]
        yx = kp[:, :2]                        # (y,x) em [0..1]
        self.q.append(yx)

        avg = np.mean(np.stack(self.q), axis=0)  # (17,2) [y,x]

        def ok(i): return conf[i] > self.conf_thr
        def y(i): return avg[i, 0]
        def x(i): return avg[i, 1]

        L_SH, R_SH = 5, 6
        L_WR, R_WR = 9, 10
        L_HIP, R_HIP = 11, 12
        L_KNE, R_KNE = 13, 14
        L_ANK, R_ANK = 15, 16

        # baseline do quadril após encher o buffer
        if self.hips_baseline is None and len(self.q) == self.q.maxlen:
            self.hips_baseline = (y(L_HIP) + y(R_HIP)) / 2

        # braço levantado: punho acima do ombro (y menor)
        arm_up = False
        for wr, sh in [(L_WR, L_SH), (R_WR, R_SH)]:
            if ok(wr) and ok(sh) and y(wr) < y(sh) - 0.05:
                arm_up = True

        squat = False
        sit_down = False

        need = [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
        if all(ok(i) for i in need):
            # converter (y,x) -> (x,y) p/ cálculo do ângulo
            A = avg[[L_HIP, L_KNE, L_ANK]][:, ::-1]
            B = avg[[R_HIP, R_KNE, R_ANK]][:, ::-1]
            la = _angle_xy(A[0], A[1], A[2])  # ângulo joelho esq
            ra = _angle_xy(B[0], B[1], B[2])  # ângulo joelho dir

            mid_hip = (y(L_HIP) + y(R_HIP)) / 2
            base = self.hips_baseline if self.hips_baseline is not None else mid_hip
            depth = (mid_hip - base)  # positivo quando desce

            # Agachamento: joelhos fecham + quadril desce
            if la < 100 and ra < 100 and depth > 0.06:
                squat = True

            # Senta: joelhos ~90 e quadril estável
            ys = [fr[[L_HIP, R_HIP], 0].mean() for fr in self.q]
            if len(ys) == self.q.maxlen and 70 <= la <= 110 and 70 <= ra <= 110 and np.std(ys) < 0.005:
                sit_down = True

        return {"arm_raise": arm_up, "squat": squat, "sit_down": sit_down}
