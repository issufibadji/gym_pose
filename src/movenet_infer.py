# src/movenet_infer.py
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Edges do MoveNet (17 keypoints)
EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

MODEL_URLS = {
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "thunder":   "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}

def load_movenet(model_name="lightning"):
    url = MODEL_URLS.get(model_name, MODEL_URLS["lightning"])
    model = hub.load(url)
    # Pegamos a função de inferência de forma robusta
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]
    else:
        # fallback (raro) – se for chamável direto
        def infer(x): return model(x)
    return infer  # retorna a função de inferência já pronta

def run_movenet(infer_fn, rgb_frame, input_size=192):
    """
    infer_fn: função retornada por load_movenet (signatures['serving_default'])
    rgb_frame: np.uint8 HxWx3 (RGB)
    Retorna: np.ndarray shape [1,1,17,3] (y, x, score) em [0..1]
    """
    # garante dtype/shape esperados
    img = tf.image.resize_with_pad(tf.expand_dims(rgb_frame, 0), input_size, input_size)
    img = tf.cast(img, tf.int32)
    outputs = infer_fn(img)
    # as chaves variam entre modelos, mas para MoveNet é 'output_0'
    key = list(outputs.keys())[0]
    return outputs[key].numpy()

def draw_skeleton(rgb, kpts, conf_thr=0.3):
    out = rgb.copy()
    h, w = out.shape[:2]
    pts = kpts[0,0,:,:]  # [17,3] (y,x,score)
    xy = np.stack([pts[:,1]*w, pts[:,0]*h], axis=1)  # (x,y)
    # linhas
    for i,j in EDGES:
        if pts[i,2] > conf_thr and pts[j,2] > conf_thr:
            cv2.line(out, tuple(np.int32(xy[i])), tuple(np.int32(xy[j])), (0,255,0), 2)
    # nós
    for i,(x,y) in enumerate(xy):
        if pts[i,2] > conf_thr:
            cv2.circle(out, (int(x),int(y)), 4, (255,0,0), -1)
    return out
