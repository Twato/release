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

# =========================================================
# T42 USB 4K CAMERA -> FIXED FOCUS 2K MODE + TOC OBB V2
# CAM3 uses a 4K USB camera, but the app captures 2K for OCR speed/stability.
# These values intentionally override config_test USB_WIDTH/HEIGHT/FPS.
# =========================================================
USB_CAPTURE_WIDTH = 2304
USB_CAPTURE_HEIGHT = 1296
USB_CAPTURE_FPS = 5
USB_CAPTURE_FOURCC = "MJPG"
USB_AUTOFOCUS_ENABLE = False
USB_FIXED_FOCUS_VALUE = None  # Set an integer only if your UVC camera supports manual focus.


# =========================================================
# T42 ISOLATED CAM3 SETTINGS
# This file is used only by T42_fixed_focus_2k_toc_obb_v2_isolated.py.
# It intentionally does not modify the original camera_test.py.
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TOC_MODEL_PATH = os.path.join(APP_DIR, "train_toc_obb_v2.pt")
USB_HOLD_TIME = 4.0
EXPECTED_TOC_CLASSES = {"toc", "toc1", "toc2", "toc3", "toc4", "toc5", "toc6"}
TOC_ORDER = {"toc": 0, "toc1": 1, "toc2": 2, "toc3": 3, "toc4": 4, "toc5": 5, "toc6": 6}
REQUIRE_ALL_TOC_CLASSES = True


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


def _tensor_to_list(value):
    try:
        return value.detach().cpu().numpy().tolist()
    except Exception:
        try:
            return value.cpu().numpy().tolist()
        except Exception:
            try:
                return value.tolist()
            except Exception:
                return value


def _obb_points_to_xyxy(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]

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

        # T39: Force USB 4K camera to work as 2K fixed-focus camera.
        # Set MJPG first, then resolution/FPS. This is usually required for high resolution via V4L2.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*USB_CAPTURE_FOURCC))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, USB_CAPTURE_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Disable autofocus for UVC cameras. Some cameras ignore these controls,
        # but calling them is safe. If manual focus is supported, set USB_FIXED_FOCUS_VALUE.
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if not USB_AUTOFOCUS_ENABLE else 1)
            if USB_FIXED_FOCUS_VALUE is not None:
                self.cap.set(cv2.CAP_PROP_FOCUS, int(USB_FIXED_FOCUS_VALUE))
        except Exception as e:
            print("USB focus control warning:", e)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera /dev/video{self.device}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.status(f"USB mode requested {USB_CAPTURE_WIDTH}x{USB_CAPTURE_HEIGHT}@{USB_CAPTURE_FPS}fps MJPG fixed-focus")
        self.status(f"USB mode actual {actual_w}x{actual_h}@{actual_fps:.1f}fps")

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
        """Detect TOC ROIs from YOLO bbox or YOLO OBB model.

        T42 supports train_toc_obb_v2.pt. OBB results are converted to normal
        xyxy boxes so the existing crop_by_box() flow can keep working.
        """
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
            names = getattr(self.model, "names", self.names) or {}

            # ---------- OBB model path ----------
            obb = getattr(r, "obb", None)
            if obb is not None:
                try:
                    n = len(obb)
                except Exception:
                    n = 0

                obb_points = None
                try:
                    if getattr(obb, "xyxyxyxy", None) is not None:
                        obb_points = _tensor_to_list(obb.xyxyxyxy)
                except Exception:
                    obb_points = None

                for i in range(n):
                    try:
                        cls = int(obb.cls[i].detach().cpu().item())
                        conf = float(obb.conf[i].detach().cpu().item())
                        raw_name = str(names.get(cls, cls))
                        name = raw_name.strip().lower()

                        if name not in EXPECTED_TOC_CLASSES:
                            continue

                        if getattr(obb, "xyxy", None) is not None:
                            xyxy = _tensor_to_list(obb.xyxy[i])
                        elif obb_points is not None:
                            xyxy = _obb_points_to_xyxy(obb_points[i])
                        else:
                            continue

                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append({
                            "name": name,
                            "conf": conf,
                            "box": [x1, y1, x2, y2],
                            "type": "obb",
                        })
                    except Exception as e:
                        print("TOC OBB parse warning:", e)

                # If OBB exists, do not also parse r.boxes from same result.
                continue

            # ---------- Normal bbox model path ----------
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    raw_name = str(names.get(cls, cls))
                    name = raw_name.strip().lower()

                    if name not in EXPECTED_TOC_CLASSES:
                        continue

                    detections.append({
                        "name": name,
                        "conf": conf,
                        "box": [x1, y1, x2, y2],
                        "type": "bbox",
                    })
                except Exception as e:
                    print("TOC bbox parse warning:", e)

        # If a class is detected more than once, keep highest confidence.
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
        เธ–เนเธฒเธกเธต YOLO:
        - detect
        - hold เธ•เธฒเธกเน€เธงเธฅเธฒเธ—เธตเนเธ•เธฑเนเธเนเธงเน
        - save full image
        - return image_path + detections เน€เธเธทเนเธญเน€เธญเธฒเนเธ crop OCR เธ•เธฒเธก box

        เธ–เนเธฒเนเธกเนเธกเธต YOLO:
        - save full image
        - return detections เธงเนเธฒเธ
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
            detected = (len(detections) >= len(EXPECTED_TOC_CLASSES)) if REQUIRE_ALL_TOC_CLASSES else (len(detections) > 0)

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
                self.status(f"TOC Detected {hold:.1f}/{USB_HOLD_TIME:.1f}s | ROI={len(detections)}/{len(EXPECTED_TOC_CLASSES)}")

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
                self.status(f"Waiting TOC | ROI={len(detections)}/{len(EXPECTED_TOC_CLASSES)}")

            sleep(0.05)

        return None
