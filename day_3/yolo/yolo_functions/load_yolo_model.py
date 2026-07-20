#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def load_yolo_model(weights, conf=0.25, iou=0.45, max_det=20,
                    repo="ultralytics/yolov5", source="github", half=True):
    import torch

    print("YOLOv5 로드 중: %s (source=%s)" % (weights, source))
    model = torch.hub.load(repo, "custom", path=weights, source=source)

    model.conf = conf
    model.iou = iou
    model.max_det = max_det
    model.agnostic = False
    model.amp = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if half:
            model.model.half()
            print("FP16(half) 추론 활성화")
    else:

        torch.set_num_threads(torch.get_num_threads())
        print("!! CUDA 를 찾지 못해 CPU 로 추론합니다. 프레임 지연이 큽니다.")
        print("!! --imgsz 320 --every 3 같은 옵션으로 부담을 줄이세요.")

    model.eval()
    names = model.names
    print("사용 장치: %s / 클래스 수: %d" % (device, len(names)))
    return model, names, device
