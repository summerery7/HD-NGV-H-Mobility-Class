#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from yolo_locate_functions import (
    load_xml_camera_matrix_dist,
    load_factory_extrinsic,
    load_yolo_model,
    DetectorThread,
    box_depth_mm,
    sample_depth_mm,
    deproject_pixel,
    color_to_depth_frame,
    draw_detection,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="YOLOv5 학습 가중치(.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence 임계값")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 임계값")
    ap.add_argument("--max-det", type=int, default=20, help="프레임당 최대 검출 수")
    ap.add_argument("--radius", type=int, default=4, help="중심 폴백 depth 창 반경[px]")
    ap.add_argument("--shrink", type=float, default=0.6,
                    help="박스 내부 depth 샘플 영역 비율(0~1)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="카메라 스트림 FPS")

    ap.add_argument("--imgsz", type=int, default=640,
                    help="추론 해상도. 낮출수록 빠름 (예: 416, 320)")
    ap.add_argument("--every", type=int, default=1,
                    help="N 프레임마다 1회 추론 요청. CPU 환경에서 2~3 권장")
    ap.add_argument("--no-half", action="store_true",
                    help="CUDA 에서도 FP16 을 쓰지 않음")
    ap.add_argument("--print-hz", type=float, default=2.0,
                    help="콘솔 좌표 출력 빈도[Hz]. 0 이면 출력 안 함")
    ap.add_argument("--params", default=os.path.join(SCRIPT_DIR, "camera_params.xml"),
                    help="캘리브 XML 경로 (camera_matrix + camera_distortion 사용)")
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
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    csp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    dsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    ci = csp.get_intrinsics()


    try:
        CAMK, CAMDIST = load_xml_camera_matrix_dist(args.params)
        print("=== 역투영 intrinsics = XML camera_matrix (%s) ===" % args.params)
        print("    fx=%.3f fy=%.3f cx=%.3f cy=%.3f" %
              (CAMK[0, 0], CAMK[1, 1], CAMK[0, 2], CAMK[1, 2]))
    except Exception as e:
        CAMK = CAMDIST = None
        print("=== XML 로드 실패(%s) -> factory intrinsics 로 역투영 ===" % e)


    R_c2d, t_c2d_mm, ext_src = load_factory_extrinsic(csp, dsp)
    print("=== color->depth extrinsic (%s) ===" % ext_src)
    print("t_c2d [mm] = [%.3f, %.3f, %.3f]  (baseline = %.3f mm)" %
          (t_c2d_mm[0], t_c2d_mm[1], t_c2d_mm[2], float(np.linalg.norm(t_c2d_mm))))


    detector = DetectorThread(model, imgsz=args.imgsz)
    detector.start()

    print("[q] 종료\n")

    frame_i = 0
    last_print = 0.0
    fps_t0, fps_n, disp_fps = time.time(), 0, 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_f = frames.get_color_frame()
            if not color_f:
                continue
            color = np.asanyarray(color_f.get_data())


            frame_i += 1
            if frame_i % args.every == 0:
                detector.submit(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))


            det = detector.get()


            depth_al = None
            if det is not None and len(det):
                depth_f = align.process(frames).get_depth_frame()
                if depth_f:
                    depth_al = np.asanyarray(depth_f.get_data())

            now = time.time()
            do_print = (args.print_hz > 0 and now - last_print >= 1.0 / args.print_hz)

            if det is not None:
                for x1, y1, x2, y2, conf, cls in det:
                    uc = (x1 + x2) / 2.0
                    vc = (y1 + y2) / 2.0

                    Z = box_depth_mm(depth_al, x1, y1, x2, y2, depth_scale, args.shrink)
                    if Z is None:
                        Z = sample_depth_mm(depth_al, uc, vc, depth_scale, args.radius)

                    cls_name = names[int(cls)] if int(cls) < len(names) else str(int(cls))

                    if Z is not None:
                        Xc, Yc, Zc = deproject_pixel(
                            uc, vc, Z, K=CAMK, dist=CAMDIST, factory_intrin=ci)
                        Xd, Yd, Zd = color_to_depth_frame(Xc, Yc, Zc, R_c2d, t_c2d_mm)
                        label = "%s %.2f | X%.0f Y%.0f Z%.0f mm" % (cls_name, conf, Xd, Yd, Zd)
                        if do_print:
                            print("%-12s conf=%.2f center=(%d,%d)  "
                                  "[depth] X=%.1f Y=%.1f Z=%.1f mm" %
                                  (cls_name, conf, int(uc), int(vc), Xd, Yd, Zd))
                    else:
                        label = "%s %.2f | depth N/A" % (cls_name, conf)

                    draw_detection(color, x1, y1, x2, y2, uc, vc, label)

            if do_print:
                last_print = now


            fps_n += 1
            if now - fps_t0 >= 0.5:
                disp_fps = fps_n / (now - fps_t0)
                fps_t0, fps_n = now, 0
            cv2.putText(color, "disp %.1f fps | infer %.0f ms (%s)"
                        % (disp_fps, detector.infer_ms, device),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("YOLO + depth", color)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        detector.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
