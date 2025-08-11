"""Process a video with MoveNet and optionally log gestures."""
import argparse
import os
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MODEL_URLS = {
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}


def main():
    parser = argparse.ArgumentParser(description="Process video with MoveNet")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video (download sample if omitted or unreadable)"
    )
    parser.add_argument("--model", choices=list(MODEL_URLS), default="lightning")
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--conf-thr", type=float, default=0.3)
    parser.add_argument("--events", action="store_true", help="Save gesture events")
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    import tensorflow as tf  # noqa: F401
    import tensorflow_hub as hub
    from src.movenet_infer import run_movenet, draw_skeleton
    from src.io_utils import read_video, save_video, download_sample
    from src.gestures import SimpleGestureDetector

    if args.video is None:
        try:
            args.video = download_sample(args.out)
        except IOError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    movenet = hub.load(MODEL_URLS[args.model])
    try:
        fps, frames = read_video(args.video)
    except IOError:
        print(f"Failed to open {args.video}, downloading sample", file=sys.stderr)
        try:
            args.video = download_sample(args.out)
            fps, frames = read_video(args.video)
        except IOError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    detector = SimpleGestureDetector(conf_thr=args.conf_thr)
    out_frames = []
    events = []

    for i, frame in enumerate(frames):
        kpts = run_movenet(movenet, frame, input_size=args.input_size)[0, 0]
        out_frames.append(draw_skeleton(frame, kpts, conf_thr=args.conf_thr))
        if args.events:
            ev = detector.update(kpts)
            for name, flag in ev.items():
                if flag:
                    events.append({"t_sec": i / fps, "event": name})

    save_video(os.path.join(args.out, "pose_overlay.mp4"), out_frames, fps)
    if args.events and events:
        with open(os.path.join(args.out, "events.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_sec", "event"])
            writer.writeheader()
            writer.writerows(events)

    print("Saved results to", args.out)


if __name__ == "__main__":
    main()
