import os
import cv2
import numpy as np
from time import sleep, time

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

from config_test import (
    PICAM_PREVIEW_SIZE,
    PICAM_CAPTURE_SIZE,
    MOTION_THRESHOLD,
    STABLE_THRESHOLD,
    STABLE_TIME,
    FOCUS_DELAY,
    USB_WIDTH,
    USB_HEIGHT,
    USB_FPS,
    TOC_MODEL_PATH,
    YOLO_CONF,
    YOLO_IMGSZ,
    USB_HOLD_TIME,
    EXPECTED_TOC_CLASSES,
    TOC_ORDER
)

def gray_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def motion_score(bg, current):
    diff = cv2.absdiff(bg, current)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return int(np.sum(thresh))

class PiCameraTest:
    def __init__(self, index, name, status_cb=print, preview_cb=None):
        self.index = index
        self.name = name
        self.status_cb = status_cb
        self.preview_cb = preview_cb
        self.cam = None

    def status(self, text):
        self.status_cb(f"{self.name}: {text}")

    def open(self):
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not available. Run this on Raspberry Pi with python3-picamera2.")

        self.status("Opening")

        self.cam = Picamera2(self.index)

        config = self.cam.create_preview_configuration(
            main={
                "size": PICAM_PREVIEW_SIZE,
                "format": "RGB888"
            }
        )

        self.cam.configure(config)
        self.cam.start()
        sleep(2)

        try:
            self.cam.set_controls({
                "AfMode": 2,
                "AfTrigger": 0
            })
        except Exception as e:
            print("AF warning:", e)

    def close(self):
        try:
            if self.cam is not None:
                self.cam.stop()
                self.cam.close()
        except Exception as e:
            print("Close warning:", e)

        self.cam = None
        self.status("Closed")

    def capture_preview_frame(self):
        return self.cam.capture_array()

    def wait_background(self, running_check=lambda: True):
        self.status("Learning Background")

        prev = None
        stable_start = None

        while running_check():
            frame = self.capture_preview_frame()

            if self.preview_cb:
                self.preview_cb(frame, False)

            gray = gray_from_frame(frame)

            if prev is None:
                prev = gray
                sleep(0.2)
                continue

            score = motion_score(prev, gray)
            print(f"{self.name} STABLE SCORE:", score)

            if score < STABLE_THRESHOLD:
                if stable_start is None:
                    stable_start = time()

                if time() - stable_start >= STABLE_TIME:
                    self.status("Background Ready")
                    return gray
            else:
                stable_start = None

            prev = gray
            sleep(0.2)

        return None

    def wait_object(self, bg, running_check=lambda: True):
        self.status("Waiting Object")

        while running_check():
            frame = self.capture_preview_frame()

            if self.preview_cb:
                self.preview_cb(frame, False)

            gray = gray_from_frame(frame)
            score = motion_score(bg, gray)
            print(f"{self.name} MOTION SCORE:", score)

            if score > MOTION_THRESHOLD:
                self.status("Object Detected")
                return True

            sleep(0.2)

        return False

    def focus_delay(self, running_check=lambda: True):
        self.status(f"Focus {FOCUS_DELAY:.1f} sec")

        start = time()

        while running_check() and time() - start < FOCUS_DELAY:
            frame = self.capture_preview_frame()
            if self.preview_cb:
                self.preview_cb(frame, False)
            sleep(0.2)

    def capture_file(self, path):
        self.status("Capture Full Image")

        still_config = self.cam.create_still_configuration(
            main={"size": PICAM_CAPTURE_SIZE}
        )

        self.cam.switch_mode_and_capture_file(still_config, path)
        self.status(f"Saved: {path}")
        return path

class UsbCameraTest:
    def __init__(self, device, name, status_cb=print, preview_cb=None):
        self.device = device
        self.name = name
        self.status_cb = status_cb
        self.preview_cb = preview_cb
        self.cap = None
        self.model = None
        self.names = {}

    def status(self, text):
        self.status_cb(f"{self.name}: {text}")

    def open(self):
        self.status(f"Opening /dev/video{self.device}")

        self.cap = cv2.VideoCapture(f"/dev/video{self.device}", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, USB_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera /dev/video{self.device}")

        self.load_yolo_if_available()

        start = time()
        while time() - start < 1.0:
            ret, frame = self.cap.read()
            if ret and self.preview_cb:
                self.preview_cb(frame, True)
            sleep(0.1)

    def load_yolo_if_available(self):
        if not os.path.exists(TOC_MODEL_PATH):
            self.status(f"YOLO model not found: {TOC_MODEL_PATH}")
            self.status("USB will capture direct, no YOLO ROI.")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(TOC_MODEL_PATH)
            self.names = self.model.names
            self.status(f"YOLO model loaded: {TOC_MODEL_PATH}")
        except Exception as e:
            self.status(f"YOLO load error: {e}")
            self.model = None

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        self.cap = None
        self.status("Closed")

    def _detect_toc_rois(self, frame):
        if self.model is None:
            return []

        results = self.model(
            frame,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.names[cls]

                if name not in EXPECTED_TOC_CLASSES:
                    continue

                detections.append({
                    "name": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2]
                })

        # ถ้า class ซ้ำ เลือก conf สูงสุด
        best_by_class = {}

        for det in detections:
            name = det["name"]
            if name not in best_by_class or det["conf"] > best_by_class[name]["conf"]:
                best_by_class[name] = det

        detections = list(best_by_class.values())

        detections = sorted(
            detections,
            key=lambda d: (
                TOC_ORDER.get(d["name"], 99),
                d["box"][1],
                d["box"][0]
            )
        )

        return detections

    def _draw_detections(self, frame, detections):
        draw = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            name = det["name"]
            conf = det["conf"]

            cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                draw,
                f"{name} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        return draw

    def capture_direct_or_yolo(self, path, running_check=lambda: True):
        """
        ถ้ามี YOLO:
        - detect
        - hold ตามเวลาที่ตั้งไว้
        - save full image
        - return image_path + detections เพื่อเอาไป crop OCR ตาม box

        ถ้าไม่มี YOLO:
        - save full image
        - return detections ว่าง
        """
        if self.model is None:
            self.status("Capture Direct Mode")
            while running_check():
                ret, frame = self.cap.read()
                if ret:
                    if self.preview_cb:
                        self.preview_cb(frame, True)
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    self.status(f"Saved: {path}")
                    return {
                        "image_path": path,
                        "detections": []
                    }
                sleep(0.2)
            return None

        first_detect_time = None
        last_frame = None
        last_detections = []

        self.status("Waiting TOC by YOLO")

        while running_check():
            ret, frame = self.cap.read()

            if not ret:
                self.status("Cannot Read Frame")
                sleep(0.2)
                continue

            detections = self._detect_toc_rois(frame)
            detected = len(detections) > 0

            draw = self._draw_detections(frame, detections)

            if self.preview_cb:
                self.preview_cb(draw, True)

            now = time()

            if detected:
                if first_detect_time is None:
                    first_detect_time = now

                last_frame = frame.copy()
                last_detections = detections

                hold = now - first_detect_time
                self.status(f"TOC Detected {hold:.1f}/{USB_HOLD_TIME:.1f}s | ROI={len(detections)}")

                if hold >= USB_HOLD_TIME:
                    cv2.imwrite(path, last_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    self.status(f"Saved: {path}")

                    return {
                        "image_path": path,
                        "detections": last_detections
                    }
            else:
                first_detect_time = None
                last_detections = []
                self.status("Waiting TOC")

            sleep(0.05)

        return None
