"""Webcam overlay demo using MoveNet."""
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MODEL_URLS = {
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}


def main():
    parser = argparse.ArgumentParser(description="Webcam MoveNet overlay")
    parser.add_argument("--model", choices=list(MODEL_URLS), default="lightning")
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--conf-thr", type=float, default=0.3)
    args = parser.parse_args()

    import cv2
    import tensorflow as tf  # noqa: F401
    import tensorflow_hub as hub
    from src.movenet_infer import run_movenet, draw_skeleton

    movenet = hub.load(MODEL_URLS[args.model])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts = run_movenet(movenet, rgb, input_size=args.input_size)[0, 0]
        out = draw_skeleton(rgb, kpts, conf_thr=args.conf_thr)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imshow("MoveNet", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
