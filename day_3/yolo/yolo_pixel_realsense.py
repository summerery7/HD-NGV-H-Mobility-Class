import os
import sys
import time
import argparse

import numpy as np
import cv2
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from yolo_functions import load_yolo_model, DetectorThread, draw_detection


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--print-hz", type=float, default=2.0)
    ap.add_argument("--repo", default="ultralytics/yolov5")
    ap.add_argument("--source", default="github", choices=["github", "local"])
    return ap.parse_args()


def main():
    args = parse_args()

    model, names, device = load_yolo_model(
        args.weights, conf=args.conf, iou=args.iou, max_det=args.max_det,
        repo=args.repo, source=args.source, half=(not args.no_half))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)

    detector = DetectorThread(model, imgsz=args.imgsz)
    detector.start()

    print("[q] quit\n")

    frame_i = 0
    last_print = 0.0
    fps_t0, fps_n, disp_fps = time.time(), 0, 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_f = frames.get_color_frame()
            if not color_f:
                continue
            frame = np.asanyarray(color_f.get_data())

            frame_i += 1
            if frame_i % args.every == 0:
                detector.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            det = detector.get()

            now = time.time()
            do_print = (args.print_hz > 0 and now - last_print >= 1.0 / args.print_hz)

            if det is not None:
                for x1, y1, x2, y2, conf, cls in det:
                    uc = (x1 + x2) / 2.0
                    vc = (y1 + y2) / 2.0
                    cls_name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
                    label = "%s %.2f | u=%d v=%d" % (cls_name, conf, int(uc), int(vc))
                    if do_print:
                        print("%-12s conf=%.2f  center pixel (u, v) = (%d, %d)" %
                              (cls_name, conf, int(uc), int(vc)))
                    draw_detection(frame, x1, y1, x2, y2, uc, vc, label)

            if do_print:
                last_print = now

            fps_n += 1
            if now - fps_t0 >= 0.5:
                disp_fps = fps_n / (now - fps_t0)
                fps_t0, fps_n = now, 0
            cv2.putText(frame, "disp %.1f fps | infer %.0f ms (%s)"
                        % (disp_fps, detector.infer_ms, device),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("YOLO pixel", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        detector.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
