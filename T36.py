import os
import sys
import json
import gc
import threading
from threading import Event
from datetime import datetime, timedelta
import subprocess
from time import sleep
import time
import traceback
import math
import shutil
import platform
import socket
import logging
import urllib.request
import urllib.error


import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from config_test import (
    CAPTURE_DIR,
    ROI_DIR,
    OCR_DIR,
    PICAM1_INDEX,
    PICAM2_INDEX,
    USB_DEVICE,
    PREVIEW_SIZE,
    PREVIEW_INTERVAL,
)

from camera_test import PiCameraTest, UsbCameraTest
import camera_test as camera_module
from roi_test import crop_center, crop_by_box, preprocess_ocr, save_roi_files
from ocr_test import run_easyocr, load_ocr

try:
    from picamera2 import Picamera2
    from libcamera import controls
except Exception:
    Picamera2 = None
    controls = None

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

# =========================================================
# T24 SYSTEM LOGGING / CRASH LOG
# This is separate from storage/job logs.
# - storage/          = production traceability data
# - system_logs/      = program runtime events
# - crash_logs/       = Python exception traceback
# =========================================================
APP_VERSION = "T36_FIXED_FIELD_MAP_RESULT_LAYOUT"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_LOG_DIR = os.path.join(APP_DIR, "system_logs")
CRASH_LOG_DIR = os.path.join(APP_DIR, "crash_logs")
os.makedirs(SYSTEM_LOG_DIR, exist_ok=True)
os.makedirs(CRASH_LOG_DIR, exist_ok=True)

SYSTEM_LOG_PATH = os.path.join(SYSTEM_LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")


def _safe_get_global(name, default=""):
    try:
        return globals().get(name, default)
    except Exception:
        return default


def get_runtime_context():
    """Collect current app state for logs without crashing if variables are not ready."""
    try:
        ctx = {
            "session_id": SESSION_ID,
            "version": APP_VERSION,
            "screen": _safe_get_global("current_screen_mode", "-"),
            "delivery_no": _safe_get_global("delivery_no", ""),
            "en_no": _safe_get_global("en_no", ""),
            "inner_index": _safe_get_global("current_index", 0),
            "inner_total": _safe_get_global("quantity", 0),
            "outer_index": _safe_get_global("current_outer_index", 0),
            "outer_total": _safe_get_global("outer_total", 0),
            "action": _safe_get_global("action_value", None),
            "running": _safe_get_global("running", False),
        }
        cam = _safe_get_global("current_camera", None)
        if cam is not None:
            ctx["current_camera"] = getattr(cam, "name", cam.__class__.__name__)
        else:
            ctx["current_camera"] = "-"
        return ctx
    except Exception:
        return {
            "session_id": SESSION_ID,
            "version": APP_VERSION,
            "context_error": True,
        }


def log_event(event, level="INFO", **detail):
    """Write one JSON-line style runtime log record and print a short line to terminal."""
    try:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "level": str(level).upper(),
            "event": event,
            "context": get_runtime_context(),
            "detail": detail or {},
        }
        line = json.dumps(record, ensure_ascii=False)
        with open(SYSTEM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"[{record['level']}] {event} {detail if detail else ''}", flush=True)
    except Exception as e:
        print("SYSTEM LOG ERROR:", e, event, detail, flush=True)


def get_pi_temp():
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_memory_info():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            lines = f.readlines()
        keep = {}
        for line in lines:
            if line.startswith(("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree")):
                k, v = line.split(":", 1)
                keep[k] = v.strip()
        return keep
    except Exception:
        return {}


def log_system_info():
    try:
        info = {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "opencv": getattr(cv2, "__version__", ""),
            "app_dir": APP_DIR,
            "system_log_path": SYSTEM_LOG_PATH,
            "temperature": get_pi_temp(),
            "memory": get_memory_info(),
        }
        log_event("PROGRAM_START", **info)
    except Exception as e:
        print("Log system info error:", e)


def write_crash_log(exc_type, exc_value, exc_traceback, source="main"):
    """Save crash traceback to crash_logs and also write a system log event."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_path = os.path.join(CRASH_LOG_DIR, f"crash_{ts}_{source}.log")

        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        ctx = get_runtime_context()

        lines = []
        lines.append("AI Camera Crash Log")
        lines.append("=" * 60)
        lines.append(f"time       : {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"session_id : {SESSION_ID}")
        lines.append(f"version    : {APP_VERSION}")
        lines.append(f"source     : {source}")
        lines.append("")
        lines.append("Context:")
        lines.append(json.dumps(ctx, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("System:")
        lines.append(json.dumps({
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "opencv": getattr(cv2, "__version__", ""),
            "temperature": get_pi_temp(),
            "memory": get_memory_info(),
        }, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("Traceback:")
        lines.append(tb_text)

        with open(crash_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log_event("CRASH_LOG_SAVED", level="ERROR", crash_path=crash_path, error=str(exc_value), source=source)
        return crash_path
    except Exception as e:
        print("WRITE CRASH LOG ERROR:", e, flush=True)
        return ""


def global_exception_hook(exc_type, exc_value, exc_traceback):
    write_crash_log(exc_type, exc_value, exc_traceback, source="sys")
    try:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    except Exception:
        pass


def thread_exception_hook(args):
    write_crash_log(args.exc_type, args.exc_value, args.exc_traceback, source=f"thread_{args.thread.name}")


sys.excepthook = global_exception_hook
try:
    threading.excepthook = thread_exception_hook
except Exception:
    pass

log_system_info()

# =========================================================
# T25 AUTO CAMERA RECOVERY
# Purpose:
# - If camera read/open/capture fails, try in-process camera recovery once.
# - If recovery succeeds, retry the same camera step automatically.
# - If recovery still fails, show error and let operator unplug/replug then press Reset Camera.
# This does NOT restart the app and does NOT reload OCR/YOLO.
# =========================================================
AUTO_CAMERA_RECOVERY_ENABLE = True
AUTO_CAMERA_RECOVERY_MAX_RETRY = 1
AUTO_CAMERA_RECOVERY_WAIT_SEC = 2.0


def auto_recover_camera(camera_name, reason="", wait_sec=AUTO_CAMERA_RECOVERY_WAIT_SEC):
    """Close current camera handle, run GC, wait, then allow caller to reopen same step."""
    global current_camera

    if not AUTO_CAMERA_RECOVERY_ENABLE:
        return False

    log_event(
        "AUTO_CAMERA_RECOVERY_START",
        level="WARNING",
        camera=camera_name,
        reason=str(reason),
        wait_sec=wait_sec,
    )

    try:
        if current_camera is not None:
            current_camera.close()
    except Exception as e:
        log_event("AUTO_CAMERA_RECOVERY_CLOSE_ERROR", level="ERROR", camera=camera_name, error=str(e))

    current_camera = None

    try:
        gc.collect()
    except Exception as e:
        log_event("AUTO_CAMERA_RECOVERY_GC_ERROR", level="ERROR", camera=camera_name, error=str(e))

    try:
        time.sleep(float(wait_sec))
    except Exception:
        pass

    log_event("AUTO_CAMERA_RECOVERY_READY_TO_RETRY", level="WARNING", camera=camera_name)
    return True


def should_auto_recover_error(error_text):
    text = str(error_text or "").lower()
    keywords = [
        "cannot read frame",
        "cannot read",
        "read frame",
        "camera",
        "capture",
        "video",
        "device",
        "libcamera",
        "v4l2",
        "timeout",
        "failed",
    ]
    return any(k in text for k in keywords)


def wait_operator_camera_fix(camera_name, message):
    """T26: After auto recovery fails, do not fail/close the job immediately.

    Keep the app on the current capture screen and wait for operator action:
    - unplug/replug camera
    - press Reset Camera or Restart The Process

    Returns:
    - "retry"  -> retry same stage
    - "cancel" -> cancel/fail current job
    """
    global running

    log_event("WAIT_OPERATOR_CAMERA_FIX", level="WARNING", camera=camera_name, message=str(message))

    try:
        root.after(0, lambda: set_status(f"{camera_name} error. Reconnect camera then press Reset Camera / Restart."))
        root.after(0, lambda: messagebox.showwarning(
            f"{camera_name} Error",
            f"{message}\n\nAuto recovery already tried and failed.\n"
            f"Please unplug/replug the camera, then press Reset Camera or Restart The Process."
        ))
    except Exception:
        pass

    # Clear previous action so a fresh button press is required.
    clear_action()

    while running and not action_event.is_set():
        sleep(0.1)

    action = action_value or "cancel"
    log_event("WAIT_OPERATOR_CAMERA_FIX_ACTION", level="WARNING", camera=camera_name, action=action)

    if action in ("camera_reset", "restart"):
        return "retry"

    return "cancel"


# =========================================================
# T27 USB CAMERA REDISCOVERY
# USB camera device can change after unplug/replug.
# Example:
#   before: /dev/video0
#   after : /dev/video1
# So CAM3 must not always trust USB_DEVICE.
# =========================================================
USB_SCAN_DEVICES = []
for _dev in [USB_DEVICE, 0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20]:
    if _dev not in USB_SCAN_DEVICES:
        USB_SCAN_DEVICES.append(_dev)


def find_available_usb_device():
    """Return the first /dev/videoN that can open and read one frame."""
    log_event("USB_SCAN_START", candidates=USB_SCAN_DEVICES)

    for dev in USB_SCAN_DEVICES:
        video_path = f"/dev/video{dev}"

        if not os.path.exists(video_path):
            continue

        cap = None
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
            if not cap.isOpened():
                log_event("USB_SCAN_SKIP", device=dev, path=video_path, reason="not_opened")
                continue

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 15)
            except Exception:
                pass

            ok = False
            frame = None

            # Try a few reads because some cameras need a short warm-up.
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.1)

            if ok and frame is not None:
                log_event("USB_SCAN_FOUND", device=dev, path=video_path, shape=str(getattr(frame, "shape", "")))
                return dev

            log_event("USB_SCAN_SKIP", device=dev, path=video_path, reason="open_but_no_frame")

        except Exception as e:
            log_event("USB_SCAN_ERROR", level="WARNING", device=dev, path=video_path, error=str(e))
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    log_event("USB_SCAN_NOT_FOUND", level="ERROR", candidates=USB_SCAN_DEVICES)
    return None


# =========================================================
# PI CAMERA MOTION THRESHOLD SETTINGS (T23)
# Values are taken from picam_motion_threshold_test.py.
# Change these numbers here if you retest CAM1/CAM2 threshold later.
# =========================================================
PICAM_MOTION_THRESHOLD = 8_000_000
PICAM_STABLE_TIME = 1.5
PICAM_FOCUS_DELAY = 0.5
PICAM_MIN_AREA = 25_000
PICAM_DIFF_THRESHOLD = 30
PICAM_BLUR_SIZE = 21

# Apply to camera_test module if that module uses these global values.
# This keeps T23 self-contained without requiring config_test.py changes.
try:
    camera_module.MOTION_THRESHOLD = PICAM_MOTION_THRESHOLD
    camera_module.STABLE_TIME = PICAM_STABLE_TIME
    camera_module.FOCUS_DELAY = PICAM_FOCUS_DELAY
    camera_module.MIN_AREA = PICAM_MIN_AREA
    camera_module.DIFF_THRESHOLD = PICAM_DIFF_THRESHOLD
    camera_module.BLUR_SIZE = PICAM_BLUR_SIZE
    # Some older camera_test versions used STABLE_THRESHOLD for the same motion score.
    camera_module.STABLE_THRESHOLD = PICAM_MOTION_THRESHOLD
    print("T27 Pi camera threshold applied:",
          "MOTION_THRESHOLD=", PICAM_MOTION_THRESHOLD,
          "STABLE_TIME=", PICAM_STABLE_TIME,
          "MIN_AREA=", PICAM_MIN_AREA,
          "DIFF_THRESHOLD=", PICAM_DIFF_THRESHOLD,
          "BLUR_SIZE=", PICAM_BLUR_SIZE)
except Exception as e:
    print("T23 threshold apply warning:", e)


# =========================================================
# T28 PI CAMERA OPTIMIZATION MERGED FROM T21
# Purpose:
# - Use 2304x1296 no-switch stream for CAM1/CAM2.
# - Keep CAM1/CAM2 open during the inner-box stage to avoid open/close delay.
# - Use sequential autofocus/capture like T21 because dual autofocus was slower.
# - Keep T27 storage/OCR/review/reset/recovery flow.
# =========================================================
USE_T21_OPTIMIZED_PI_CAMERAS = True

PI_CAPTURE_WIDTH = 2304
PI_CAPTURE_HEIGHT = 1296
PI_CAPTURE_FORMAT = "RGB888"
PI_JPEG_QUALITY = 95

PICAM1_FOCUS_DELAY_SEC = 1.5
PICAM2_FOCUS_DELAY_SEC = 1.0

PERSISTENT_PI_CAMERAS = True
PREWARM_CAM2_ON_START = True
PI_CAMERA_OPEN_SETTLE_SEC = 0.5
PI_AF_SETTLE_SEC = 0.15

# Motion detection runs on a resized 1280x720 frame so the old T27 threshold remains meaningful.
PI_MOTION_ANALYZE_SIZE = (1280, 720)

optimized_persistent_cam1 = None
optimized_persistent_cam2 = None
optimized_persistent_lock = threading.Lock()

# =========================================================
# T30 CAM1/CAM2 LABEL DETECTION + ROI OCR
# Purpose:
# - Keep T29 optimized auto capture / storage / review flow.
# - After CAM1/CAM2 full-image capture, detect label ROIs with YOLO.
# - Crop detected label ROI and send each ROI to EasyOCR instead of using center ROI only.
# - Fallback to center ROI if the detection model fails or detects nothing.
# =========================================================
USE_INNER_YOLO_ROI_OCR = True

CAM1_INNER_DETECTION_MODEL_PATH = "/home/toto1/AI_camera/models/TOA_model/train_toa_obb_v3.pt"
CAM2_INNER_DETECTION_MODEL_PATH = "/home/toto1/AI_camera/models/TOB_model/train_tob_obb_v3.pt"

INNER_DETECT_CONF = 0.50
INNER_DETECT_IMGSZ = 640
INNER_DETECT_PAD = 4

cam1_inner_detection_model = None
cam2_inner_detection_model = None
inner_detection_lock = threading.Lock()

# Class-name aliases. These are intentionally flexible because YOLO class names
# may be exported as vendor_box_id / vendor / vn / toa_lot / date_toa / etc.
INNER_FIELD_ALIASES = {
    "vendor_box_id": [
        "vendor_box_id", "vendor_boxid", "vendor_box", "vender_box_id", "vender_boxid",
        "vender_box", "box_id", "boxid", "box-id", "vn_box", "vn_box_id", "vendor", "boxid_vn", "box_id_vn", "boxidvn"
    ],
    "vendor_date_code": [
        "vendor_date_code", "vendor_date", "vender_date_code", "vender_date",
        "vn_date", "date_vendor", "date_vender", "datecode_vendor", "datecode_vn", "date_code_vn", "datecodevn"
    ],
    "toa_lot_no": [
        "toa_lot_no", "toa_lot", "lot_toa", "toa_lotno", "toa_lot_no", "toa", "lotno_toa", "lot_no_toa", "lotnotoa"
    ],
    "toa_date_code": [
        "toa_date_code", "toa_date", "date_toa", "datecode_toa", "date_code_toa", "datecodetoa"
    ],
    "tob_lot_no": [
        "tob_lot_no", "tob_lot", "lot_tob", "tob_lotno", "tob", "lotno_tob", "lot_no_tob", "lotnotob"
    ],
    "tob_date_code": [
        "tob_date_code", "tob_date", "date_tob", "datecode_tob", "date_code_tob", "datecodetob"
    ],
}


# T36: single source of truth for YOLO class -> result/validation field.
# Use exact model class names first to avoid substring bugs such as:
# DATECODE_TOA accidentally matching broad alias "toa" and becoming toa_lot_no.
INNER_CLASS_FIELD_MAP = {
    "boxid_vn": "vendor_box_id",
    "box_id_vn": "vendor_box_id",
    "boxidvn": "vendor_box_id",

    "datecode_vn": "vendor_date_code",
    "date_code_vn": "vendor_date_code",
    "datecodevn": "vendor_date_code",

    "lotno_toa": "toa_lot_no",
    "lot_no_toa": "toa_lot_no",
    "lotnotoa": "toa_lot_no",

    "datecode_toa": "toa_date_code",
    "date_code_toa": "toa_date_code",
    "datecodetoa": "toa_date_code",

    "lotno_tob": "tob_lot_no",
    "lot_no_tob": "tob_lot_no",
    "lotnotob": "tob_lot_no",

    "datecode_tob": "tob_date_code",
    "date_code_tob": "tob_date_code",
    "datecodetob": "tob_date_code",
}


def _normalize_class_key(name):
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_label_name(name):
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def map_inner_class_to_field(class_name, camera_key="cam1"):
    """T36: Map YOLO class name to result field using exact mapping first.

    Exact map prevents DATECODE_TOA -> toa_lot_no and DATECODE_TOB -> tob_lot_no bugs.
    Alias fallback is kept only for old models, with broad aliases handled after exact names.
    """
    n = _normalize_class_key(class_name)

    # 1) Exact class map from current v3 OBB models.
    exact = INNER_CLASS_FIELD_MAP.get(n)
    if exact:
        if camera_key == "cam2" and exact.startswith(("vendor", "toa")):
            return ""
        if camera_key == "cam1" and exact.startswith("tob"):
            return ""
        return exact

    # 2) Safe alias fallback for older model class names.
    # Prefer longer aliases first, so datecode_toa wins before broad words like toa.
    alias_pairs = []
    for field, aliases in INNER_FIELD_ALIASES.items():
        for alias in aliases:
            a = _normalize_class_key(alias)
            alias_pairs.append((len(a), field, a))
    alias_pairs.sort(reverse=True)

    for _, field, a in alias_pairs:
        if not a:
            continue
        if n == a or a in n:
            if camera_key == "cam2" and field.startswith(("vendor", "toa")):
                continue
            if camera_key == "cam1" and field.startswith("tob"):
                continue
            return field

    return ""


def resolve_existing_model_path(preferred_path, fallback_paths=None):
    """Use newest model path if it exists; otherwise fall back safely."""
    paths = [preferred_path] + list(fallback_paths or [])
    for path in paths:
        if path and os.path.exists(path):
            return path
    return preferred_path


def get_inner_model_path(camera_key):
    """T34: Prefer v3 model names, but fall back to v2 if the Pi still has v2 files only."""
    if camera_key == "cam1":
        return resolve_existing_model_path(
            CAM1_INNER_DETECTION_MODEL_PATH,
            [
                "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/train_toa_obb_v2.pt",
                "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/Models_detection_VNTOA_V1.pt",
            ],
        )
    return resolve_existing_model_path(
        CAM2_INNER_DETECTION_MODEL_PATH,
        [
            "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/train_tob_obb_v2.pt",
            "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/Models_detection_TOB_V1.pt",
        ],
    )


def get_inner_detection_model(camera_key):
    """Lazy-load and cache CAM1/CAM2 YOLO detection models."""
    global cam1_inner_detection_model, cam2_inner_detection_model
    if not USE_INNER_YOLO_ROI_OCR:
        return None

    with inner_detection_lock:
        if camera_key == "cam1":
            if cam1_inner_detection_model is None:
                from ultralytics import YOLO
                model_path = get_inner_model_path("cam1")
                log_event("INNER_YOLO_LOAD_START", camera="CAM1", path=model_path)
                model_path = get_inner_model_path("cam1")
                cam1_inner_detection_model = YOLO(model_path)
                log_event("INNER_YOLO_LOAD_OK", camera="CAM1", names=str(getattr(cam1_inner_detection_model, "names", {})))
            return cam1_inner_detection_model

        if camera_key == "cam2":
            if cam2_inner_detection_model is None:
                from ultralytics import YOLO
                model_path = get_inner_model_path("cam2")
                log_event("INNER_YOLO_LOAD_START", camera="CAM2", path=model_path)
                model_path = get_inner_model_path("cam2")
                cam2_inner_detection_model = YOLO(model_path)
                log_event("INNER_YOLO_LOAD_OK", camera="CAM2", names=str(getattr(cam2_inner_detection_model, "names", {})))
            return cam2_inner_detection_model

    return None


def load_inner_detection_models():
    """Preload CAM1/CAM2 detection models during startup."""
    if not USE_INNER_YOLO_ROI_OCR:
        return "disabled"
    msg = []
    try:
        get_inner_detection_model("cam1")
        msg.append("CAM1=OK")
    except Exception as e:
        msg.append(f"CAM1=ERROR:{e}")
    try:
        get_inner_detection_model("cam2")
        msg.append("CAM2=OK")
    except Exception as e:
        msg.append(f"CAM2=ERROR:{e}")
    return ", ".join(msg)


def _obb_points_to_xyxy(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def yolo_detect_inner_rois(image_path, camera_key):
    """T34: Run CAM1/CAM2 YOLO detection and support both OBB and normal bbox models.

    Important:
    - OBB models return result.obb, not result.boxes.
    - Previous T33 read only result.boxes, so OBB detections could be ignored.
    - Result page uses mapped fields from these detections to display ROI images and OCR values.
    """
    model = get_inner_detection_model(camera_key)
    if model is None:
        return []

    started = now_iso()
    t0 = time.time()
    detections = []
    try:
        results = model.predict(
            source=image_path,
            imgsz=INNER_DETECT_IMGSZ,
            conf=INNER_DETECT_CONF,
            verbose=False,
        )
        names = getattr(model, "names", {}) or {}

        for r in results or []:
            # ---------- OBB model ----------
            obb = getattr(r, "obb", None)
            if obb is not None:
                try:
                    n = len(obb)
                except Exception:
                    n = 0

                obb_points = None
                try:
                    if getattr(obb, "xyxyxyxy", None) is not None:
                        obb_points = obb.xyxyxyxy.detach().cpu().numpy().tolist()
                except Exception:
                    obb_points = None

                for i in range(n):
                    try:
                        cls_id = int(obb.cls[i].detach().cpu().item())
                        conf = float(obb.conf[i].detach().cpu().item())
                        if getattr(obb, "xyxy", None) is not None:
                            xyxy = obb.xyxy[i].detach().cpu().numpy().tolist()
                        elif obb_points is not None:
                            xyxy = _obb_points_to_xyxy(obb_points[i])
                        else:
                            continue

                        class_name = str(names.get(cls_id, cls_id))
                        item = {
                            "name": class_name,
                            "class_id": cls_id,
                            "conf": conf,
                            "box": [float(x) for x in xyxy],
                            "field_key": map_inner_class_to_field(class_name, camera_key),
                            "type": "obb",
                        }
                        if obb_points is not None:
                            item["obb_points"] = obb_points[i]
                        detections.append(item)
                    except Exception as e:
                        log_event("INNER_YOLO_OBB_PARSE_ERROR", level="WARNING", camera=camera_key, error=str(e))
                continue

            # ---------- Normal bbox model ----------
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            for b in boxes:
                try:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
                    class_name = str(names.get(cls_id, cls_id))
                    detections.append({
                        "name": class_name,
                        "class_id": cls_id,
                        "conf": conf,
                        "box": [float(x) for x in xyxy],
                        "field_key": map_inner_class_to_field(class_name, camera_key),
                        "type": "bbox",
                    })
                except Exception as e:
                    log_event("INNER_YOLO_BOX_PARSE_ERROR", level="WARNING", camera=camera_key, error=str(e))
    except Exception as e:
        log_event("INNER_YOLO_DETECT_ERROR", level="ERROR", camera=camera_key, image_path=image_path, error=str(e), traceback=traceback.format_exc())
        return []

    detections.sort(key=lambda d: (d.get("box", [0, 0, 0, 0])[1], d.get("box", [0, 0, 0, 0])[0]))
    log_event(
        "INNER_YOLO_DETECT_DONE",
        camera=camera_key,
        image_path=image_path,
        count=len(detections),
        duration_sec=round(time.time() - t0, 3),
        classes=[d.get("name") for d in detections],
        fields=[d.get("field_key") for d in detections],
        types=[d.get("type") for d in detections],
        started_at=started,
    )
    return detections


def process_inner_yolo_roi_ocr(image_path, camera_key):
    """Detect CAM1/CAM2 label ROIs, OCR each ROI, and return field-based OCR data.

    Returns a dict similar to process_center_roi_ocr(), but with:
      - fields: {vendor_box_id/toa_lot_no/...: OCR data}
      - detections: raw YOLO detections
      - fallback: center ROI data if model detects nothing or no field can be mapped
    """
    ocr_started = now_iso()
    log_event("INNER_YOLO_ROI_OCR_START", camera=camera_key, image_path=image_path)

    img = cv2.imread(image_path)
    if img is None:
        return process_center_roi_ocr(image_path, camera_key)

    detections = yolo_detect_inner_rois(image_path, camera_key)
    fields = {}
    by_class = {}
    unmapped = []

    for idx, det in enumerate(detections, start=1):
        name = str(det.get("name", f"roi{idx}")).lower()
        field_key = det.get("field_key", "")
        box = det.get("box", [])
        crop, fixed_box = crop_by_box(img, box, pad=INNER_DETECT_PAD)
        if crop is None or crop.size == 0:
            continue

        processed = preprocess_ocr(crop)
        prefix = f"{camera_key}_{field_key or name}_{idx}"
        roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)
        row_started = now_iso()
        ocr = run_easyocr(processed)
        row_finished = now_iso()

        item = {
            "name": field_key or name,
            "class_name": name,
            "field_key": field_key,
            "created_at": row_started,
            "updated_at": row_finished,
            "capture_time": file_time_iso(image_path),
            "ocr_started_at": row_started,
            "ocr_finished_at": row_finished,
            "ocr_duration_sec": seconds_between(row_started, row_finished),
            "source": f"{camera_key}_yolo_roi",
            "detect_conf": det.get("conf", None),
            "image_path": image_path,
            "roi_path": roi_path,
            "processed_path": processed_path,
            "box": fixed_box,
            "raw": ocr.get("raw", ""),
            "clean": ocr.get("clean", ""),
            "items": ocr.get("items", []),
        }
        by_class[name] = item
        if field_key:
            # Keep the highest-confidence detection if duplicate class/field appears.
            old = fields.get(field_key)
            if old is None or (item.get("detect_conf") or 0) > (old.get("detect_conf") or 0):
                fields[field_key] = item
        else:
            unmapped.append(item)

    # T36 strict result behavior:
    # Do NOT copy one detected ROI into missing fields.
    # If BOX ID / LOT / Date Code is not detected or not mapped, that result slot stays empty/NO IMAGE.
    # This makes the Result page show exactly what YOLO detection actually found.
    fallback = None

    ocr_finished = now_iso()
    primary = next(iter(fields.values()), fallback or {})
    result = {
        "name": camera_key,
        "created_at": ocr_started,
        "updated_at": ocr_finished,
        "capture_time": file_time_iso(image_path),
        "ocr_started_at": ocr_started,
        "ocr_finished_at": ocr_finished,
        "ocr_duration_sec": seconds_between(ocr_started, ocr_finished),
        "source": f"{camera_key}_yolo_detection_roi",
        "image_path": image_path,
        "roi_path": primary.get("roi_path", ""),
        "processed_path": primary.get("processed_path", ""),
        "box": primary.get("box", []),
        "raw": primary.get("raw", ""),
        "clean": primary.get("clean", ""),
        "items": primary.get("items", []),
        "fields": fields,
        "by_class": by_class,
        "unmapped": unmapped,
        "detections": detections,
        "fallback": fallback,
    }
    log_event(
        "INNER_YOLO_ROI_OCR_END",
        camera=camera_key,
        image_path=image_path,
        detection_count=len(detections),
        mapped_fields=list(fields.keys()),
        duration_sec=seconds_between(ocr_started, ocr_finished),
    )
    return result


def get_inner_field_data(data, field_key, fallback=None):
    """Return only the requested mapped detection/OCR field.

    T36: If a field is missing, return fallback if explicitly provided; otherwise return {}.
    This prevents the Result page from showing DateCode ROI in BOX ID / LotNo slots.
    """
    data = data or {}
    fields = data.get("fields") if isinstance(data, dict) else None
    if isinstance(fields, dict):
        return fields.get(field_key) or (fallback if fallback is not None else {})
    return fallback if fallback is not None else {}


def get_inner_field_clean(data, field_key, fallback=None):
    return get_clean_ocr(get_inner_field_data(data, field_key, fallback))

# =========================================================
# GLOBAL STATE
# =========================================================
running = False
last_preview_time = 0
current_camera = None

# Action event is used for OK/NEXT, Restart The Process, Cancel Job.
# action_value can be: None, "next", "restart", "camera_reset", "cancel"
action_event = Event()
action_value = None

debug_wait_counter = 0
current_review_context = "-"
current_screen_mode = "input"  # input, capture_inner, result_inner, capture_outer, result_outer, ready_toc, complete

check_labels = {}
capture_state_label = None
capture_state_dot = None
result_product_slots = {}

# Job data
en_no = ""
delivery_no = ""
item_no = "-"       # From Mock API / future production API.
pack_type = "Tray"  # Fixed for this test UI.
quantity = 0        # Inner Box total from API.
current_index = 0   # Current Inner Box index.
current_outer_index = 0
outer_total = 0     # ceil(quantity / 6)
inner_toa_lot_by_index = {}  # Backward-compatible cache: Inner Box no -> TOA Lot No OCR value

# =========================================================
# T31 MOCK API INTEGRATION
# Current flow:
# - Operator inputs/scans EN and Delivery No.
# - App sends Delivery No to Mock API.
# - API returns Item No, Quantity, and BOX ID list.
# - Vendor BOX ID may arrive in any order.
# - Each Inner Box validates Vendor BOX ID is included in API box_id_list.
# - Duplicate Vendor BOX ID inside the same job is NG.
# =========================================================
ENABLE_MOCK_API = True
MOCK_API_BASE_URL = "http://127.0.0.1:5000"
MOCK_API_PICKING_URL = MOCK_API_BASE_URL.rstrip("/") + "/api/picking"
MOCK_API_TIMEOUT_SEC = 5

picking_api_data = {}
box_id_list = []


def normalize_scan_text(value):
    """Normalize keyboard/scanner input. Scanner usually ends with Enter/Tab/Space."""
    text = str(value or "")
    text = text.replace("\r", "").replace("\n", "").replace("\t", "")
    return text.strip()


def api_post_json(url, payload, timeout=MOCK_API_TIMEOUT_SEC):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw or "{}")


def fetch_picking_from_api(delivery_value):
    """Return normalized picking data from Mock API or raise RuntimeError."""
    delivery_value = normalize_scan_text(delivery_value)
    if not delivery_value:
        raise RuntimeError("Delivery No is empty")

    started = now_iso()
    log_event("MOCK_API_PICKING_REQUEST", url=MOCK_API_PICKING_URL, delivery_no=delivery_value)

    try:
        res = api_post_json(MOCK_API_PICKING_URL, {"delivery_no": delivery_value})
    except urllib.error.URLError as e:
        log_event("MOCK_API_PICKING_CONNECTION_ERROR", level="ERROR", error=str(e), url=MOCK_API_PICKING_URL)
        raise RuntimeError(f"Cannot connect Mock API: {e}")
    except Exception as e:
        log_event("MOCK_API_PICKING_ERROR", level="ERROR", error=str(e), url=MOCK_API_PICKING_URL)
        raise RuntimeError(f"Mock API error: {e}")

    if not isinstance(res, dict) or not res.get("success"):
        message = res.get("message", "Delivery Not Found") if isinstance(res, dict) else "Invalid API response"
        log_event("MOCK_API_PICKING_NOT_FOUND", level="WARNING", delivery_no=delivery_value, response=res)
        raise RuntimeError(message)

    data = res.get("data", {}) or {}
    api_delivery = normalize_scan_text(data.get("delivery_no") or delivery_value)
    api_item = normalize_scan_text(data.get("item_no") or "-")
    api_box_ids = [normalize_scan_text(x) for x in (data.get("box_ids") or []) if normalize_scan_text(x)]

    try:
        api_qty = int(data.get("quantity", len(api_box_ids)))
    except Exception:
        api_qty = len(api_box_ids)

    if api_qty < 1:
        raise RuntimeError("API quantity is invalid")
    if api_box_ids and api_qty != len(api_box_ids):
        log_event(
            "MOCK_API_QTY_BOX_COUNT_MISMATCH",
            level="WARNING",
            delivery_no=api_delivery,
            quantity=api_qty,
            box_id_count=len(api_box_ids),
        )

    normalized = {
        "delivery_no": api_delivery,
        "item_no": api_item,
        "quantity": api_qty,
        "box_ids": api_box_ids,
        "raw_response": res,
        "api_url": MOCK_API_PICKING_URL,
        "requested_at": started,
        "received_at": now_iso(),
    }
    normalized["duration_sec"] = seconds_between(started, normalized["received_at"])
    log_event("MOCK_API_PICKING_OK", **{k: v for k, v in normalized.items() if k != "raw_response"})
    return normalized


def apply_picking_to_input_fields(data):
    """Fill UI fields from API data. Keep EN as operator input."""
    try:
        delivery_entry.delete(0, tk.END)
        delivery_entry.insert(0, data.get("delivery_no", ""))
        qty_entry.delete(0, tk.END)
        qty_entry.insert(0, str(data.get("quantity", "")))
    except Exception:
        pass


def get_api_box_id_list():
    """Return allowed BOX ID list from current API data/job data. Order is not fixed."""
    if box_id_list:
        return list(box_id_list)
    if job_data:
        return list(job_data.get("picking", {}).get("picking_box_ids", []) or [])
    return []


def get_used_vendor_box_ids(current_inner_no=None):
    """Return Vendor BOX IDs already accepted/saved in this job, excluding current inner record."""
    used = []
    if not job_data:
        return used
    for rec in job_data.get("inner_boxes", []) or []:
        if current_inner_no is not None and rec.get("sequence") == current_inner_no:
            continue
        value = rec.get("vendor_box_id", "")
        if value:
            used.append(value)
    return used


def box_id_allowed_any_order(vendor_box_id):
    """Check if OCR Vendor BOX ID exists in API box list, regardless of scanning order."""
    allowed = get_api_box_id_list()
    vendor_norm = normalize_text_for_compare(vendor_box_id)
    allowed_norm = [normalize_text_for_compare(x) for x in allowed]
    return bool(vendor_norm and vendor_norm in allowed_norm), allowed


VALIDATION_REASON_TEXT = {
    "RULE_1_API_BOX_ID_LIST_MISSING": "Vendor: API BOX ID list missing",
    "RULE_1_VENDOR_BOX_ID_OCR_EMPTY": "Vendor: BOX ID OCR empty",
    "RULE_1_VENDOR_BOX_ID_NOT_FOUND_IN_API_LIST": "Vendor: BOX ID not found in API list",
    "RULE_1_VENDOR_BOX_ID_DUPLICATE_IN_JOB": "Vendor: BOX ID duplicate in this job",
    "RULE_3_VENDOR_DATE_EMPTY": "Vendor: Date Code OCR empty",
    "RULE_3_TOA_DATE_EMPTY": "TOA: Date Code OCR empty",
    "RULE_3_TOB_DATE_EMPTY": "TOB: Date Code OCR empty",
    "RULE_3_DATE_CODE_NOT_MATCH": "Date Code: Vendor/TOA/TOB not match",
    "RULE_5_TOA_LOT_OCR_EMPTY": "TOA: Lot No OCR empty",
    "RULE_5_TOA_LOT_DUPLICATE_IN_JOB": "TOA: Lot No duplicate in this job",
    "RULE_6_TOB_LOT_OCR_EMPTY": "TOB: Lot No OCR empty",
    "RULE_6_TOB_LOT_DUPLICATE_IN_JOB": "TOB: Lot No duplicate in this job",
}


def validation_reason_label(code):
    return VALIDATION_REASON_TEXT.get(str(code), str(code))


def validation_error_summary(fail_reason):
    reasons = [validation_reason_label(x) for x in (fail_reason or [])]
    if not reasons:
        return "Validation PASS. Operator can press OK / NEXT."
    return "ERROR (Display Only): " + " | ".join(reasons) + "  | Operator can still press OK / NEXT."


def validation_status_from_rule(record, key):
    try:
        return (record.get("validation", {}) or {}).get(key, "WAIT")
    except Exception:
        return "WAIT"


def on_en_enter(event=None):
    """Scanner/keyboard Enter: move EN -> Delivery No."""
    try:
        en_entry.delete(0, tk.END)
        en_entry.insert(0, normalize_scan_text(en_entry.get()))
        delivery_entry.focus_set()
    except Exception:
        pass
    return "break"


def on_delivery_enter(event=None):
    """Scanner/keyboard Enter: normalize Delivery No and move focus to Start."""
    try:
        delivery_entry.delete(0, tk.END)
        delivery_entry.insert(0, normalize_scan_text(delivery_entry.get()))
        start_btn.focus_set()
    except Exception:
        pass
    return "break"

# =========================================================
# PHASE 2 DATA MODEL + STORAGE
# Phase 3/4/5 (Database / API / Audit Search) are intentionally not connected yet.
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = os.path.join(APP_DIR, "storage")

job_id = ""
job_started_at = ""
job_storage_dir = ""
job_data = {}

# V2 Attempt Management
delivery_storage_dir = ""
attempt_storage_dir = ""
attempt_no = 0
attempt_status = ""
process_attempt_counter = {}
current_process_attempt_dir = ""
current_process_attempt_meta = {}


def clean_value(value, empty=""):
    value = str(value or "").strip()
    if value == "(empty)" or value == "-":
        return empty
    return value


def normalize_key(value):
    return clean_value(value, "UNKNOWN").replace(" ", "").replace("/", "_").replace("\\", "_").upper()


def safe_folder_name(value, max_len=80):
    value = normalize_key(value)
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")
    name = "".join(safe).strip("_") or "UNKNOWN"
    return name[:max_len]


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def file_time_iso(path):
    try:
        if path and os.path.exists(path):
            return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    except Exception:
        pass
    return ""


def seconds_between(start_iso, end_iso=None):
    try:
        if not start_iso:
            return None
        end_iso = end_iso or now_iso()
        start_dt = datetime.fromisoformat(str(start_iso))
        end_dt = datetime.fromisoformat(str(end_iso))
        return round((end_dt - start_dt).total_seconds(), 3)
    except Exception:
        return None


def touch_dict_time(data, created_key="created_at"):
    if isinstance(data, dict):
        t = now_iso()
        data.setdefault(created_key, t)
        data["updated_at"] = t
    return data


def make_time_block(start_iso=None, end_iso=None):
    start_iso = start_iso or now_iso()
    end_iso = end_iso or ""
    return {
        "created_at": start_iso,
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_sec": seconds_between(start_iso, end_iso) if end_iso else None,
    }


def add_job_event(event, status="", detail=None):
    if not job_data:
        return
    item = {
        "time": now_iso(),
        "event": event,
    }
    if status:
        item["status"] = status
    if detail:
        item.update(detail)
    job_data.setdefault("events", []).append(item)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def copy_if_exists(src, dst_dir, new_name=None):
    if not src or not os.path.exists(src):
        return ""
    ensure_dir(dst_dir)
    filename = new_name or os.path.basename(src)
    dst = os.path.join(dst_dir, filename)
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception as e:
        print("Storage copy error:", src, "->", dst, e)
        return ""


def make_job_id(delivery):
    return datetime.now().strftime("JOB_%Y%m%d_%H%M%S_") + safe_folder_name(delivery, 40)



def get_delivery_dir(delivery=None):
    now = datetime.now()
    delivery = delivery or delivery_no or "UNKNOWN"
    return os.path.join(
        STORAGE_ROOT,
        now.strftime("%Y"),
        now.strftime("%m"),
        now.strftime("%d"),
        f"DELIVERY_{safe_folder_name(delivery)}",
    )


def find_next_attempt_no(delivery_dir):
    ensure_dir(delivery_dir)
    max_no = 0
    for name in os.listdir(delivery_dir):
        if not name.startswith("attempt_"):
            continue
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            max_no = max(max_no, int(parts[1]))
    return max_no + 1


def attempt_folder_name(no, status):
    return f"attempt_{no:03d}_{str(status or 'running').lower()}"


def write_json_file(path, data):
    ensure_dir(os.path.dirname(path))
    if isinstance(data, dict):
        # T7: every JSON saved by the system has updated_at/saved_at.
        data.setdefault("created_at", now_iso())
        data["updated_at"] = now_iso()
        data["saved_at"] = now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



# =========================================================
# GLOBAL SEARCH INDEX (T8)
# Single search file for 90-day audit lookup.
# Search keys:
# - Delivery No
# - Vendor BOX ID
# - TOA LOT NO
# - TOB LOT NO
# - TOC Delivery No
# - TOC LOT NO
# =========================================================
GLOBAL_SEARCH_INDEX_PATH = os.path.join(STORAGE_ROOT, "global_search_index.json")

# =========================================================
# STORAGE RETENTION / CLEANUP LOG (T21)
# Change only RETENTION_DAYS to control auto-delete.
#
# Examples:
#   RETENTION_DAYS = 90   -> delete folders older than 90 days
#   RETENTION_DAYS = 30   -> delete folders older than 30 days
#   RETENTION_DAYS = 7    -> delete folders older than 7 days
#   RETENTION_DAYS = 1    -> keep today + yesterday, delete older folders
#   RETENTION_DAYS = 0    -> keep only today's date folder, delete older folders
#   RETENTION_DAYS = -1   -> TEST / CLEAR ALL date folders under storage/YYYY/MM/DD
#
# Log folder:
#   storage/cleanup_logs/cleanup_YYYYMMDD_HHMMSS.json
#   storage/cleanup_logs/cleanup_YYYYMMDD_HHMMSS.txt
# =========================================================
RETENTION_DAYS = 90

# T22: Run cleanup repeatedly while the app is left open.
# Default = every 24 hours. For testing, you can set 1 or 0.1.
CLEANUP_INTERVAL_HOURS = 24
CLEANUP_INTERVAL_MS = int(CLEANUP_INTERVAL_HOURS * 60 * 60 * 1000)

CLEANUP_LOG_DIR = os.path.join(STORAGE_ROOT, "cleanup_logs")


def rel_storage_path(path):
    try:
        if not path:
            return ""
        return os.path.relpath(path, STORAGE_ROOT).replace("\\", "/")
    except Exception:
        return str(path or "").replace("\\", "/")


def read_json_safely(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def blank_global_index():
    return {
        "project": "AI-Based Validation System for Shipping Label Verification",
        "index_version": "T8_SINGLE_GLOBAL_SEARCH_INDEX",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "storage_root": STORAGE_ROOT,
        "record_count": 0,
        "delivery_nos": {},
        "box_ids": {},
        "toa_lots": {},
        "tob_lots": {},
        "toc_delivery_nos": {},
        "toc_lots": {},
    }


def add_index_entry(index_data, section, key_value, entry):
    key_value = clean_value(key_value, "")
    if not key_value:
        return
    key = normalize_text_for_compare(key_value)
    if not key:
        return
    item = dict(entry)
    item["value"] = key_value
    item["search_key"] = key
    item["updated_at"] = now_iso()

    bucket = index_data.setdefault(section, {})
    records = bucket.setdefault(key, [])

    # Avoid exact duplicate entries while allowing real duplicate BOX/LOT values across attempts.
    sig = (
        item.get("type"),
        item.get("delivery_no"),
        item.get("attempt_folder"),
        item.get("inner_ref"),
        item.get("toc_ref"),
        item.get("path"),
    )
    for old in records:
        old_sig = (
            old.get("type"),
            old.get("delivery_no"),
            old.get("attempt_folder"),
            old.get("inner_ref"),
            old.get("toc_ref"),
            old.get("path"),
        )
        if old_sig == sig:
            old.update(item)
            return
    records.append(item)


def derive_attempt_status_from_folder(folder_name):
    try:
        parts = str(folder_name or "").split("_")
        return parts[-1].upper() if len(parts) >= 3 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def build_entry_base(job, job_json_path):
    attempt_dir = os.path.dirname(job_json_path)
    delivery_dir = os.path.dirname(attempt_dir)
    attempt_folder = os.path.basename(attempt_dir)
    return {
        "job_id": job.get("job_id", ""),
        "operator_en": job.get("operator_en", ""),
        "delivery_no": job.get("picking", {}).get("picking_delivery_no", job.get("delivery_no", delivery_no)),
        "attempt_no": job.get("attempt_no", job.get("attempt", {}).get("attempt_no", "")),
        "attempt_folder": attempt_folder,
        "attempt_status": job.get("status", derive_attempt_status_from_folder(attempt_folder)),
        "attempt_start_time": job.get("start_time", job.get("job_started_at", "")),
        "attempt_end_time": job.get("end_time", job.get("job_finished_at", "")),
        "delivery_path": rel_storage_path(delivery_dir),
        "attempt_path": rel_storage_path(attempt_dir),
        "job_json": rel_storage_path(job_json_path),
    }


def rebuild_global_search_index():
    """
    Rebuild one global search index from all job.json files under storage/.
    This is intentionally simple and reliable for the current 90-day storage size.
    """
    ensure_dir(STORAGE_ROOT)
    index_data = blank_global_index()

    try:
        for root_dir, _, files in os.walk(STORAGE_ROOT):
            if "job.json" not in files:
                continue
            job_json_path = os.path.join(root_dir, "job.json")
            job = read_json_safely(job_json_path)
            if not isinstance(job, dict):
                continue

            base = build_entry_base(job, job_json_path)
            delivery_value = base.get("delivery_no", "")

            add_index_entry(index_data, "delivery_nos", delivery_value, {
                **base,
                "type": "DELIVERY_NO",
                "path": base.get("attempt_path", ""),
                "summary_path": rel_storage_path(os.path.join(os.path.dirname(root_dir), "delivery_summary.json")),
            })

            for rec in job.get("inner_boxes", []) or []:
                if not isinstance(rec, dict):
                    continue
                inner_ref = rec.get("inner_ref") or f"INNER_{int(rec.get('sequence', 0) or 0):03d}"
                inner_path = os.path.join(root_dir, "inner_boxes", inner_ref)
                common = {
                    **base,
                    "type": "INNER_BOX",
                    "inner_ref": inner_ref,
                    "sequence": rec.get("sequence", ""),
                    "path": rel_storage_path(inner_path),
                    "inner_summary": rel_storage_path(os.path.join(inner_path, "inner_summary.json")),
                    "ocr_result": rel_storage_path(os.path.join(rec.get("process_attempt_folder", ""), "ocr_result.json")) if rec.get("process_attempt_folder") else "",
                    "validation_result": rec.get("validation_result", ""),
                    "fail_reason": rec.get("fail_reason", []),
                    "record_start_time": rec.get("start_time", ""),
                    "record_end_time": rec.get("end_time", ""),
                }
                add_index_entry(index_data, "box_ids", rec.get("vendor_box_id", ""), {**common, "type": "VENDOR_BOX_ID"})
                add_index_entry(index_data, "toa_lots", rec.get("toa_lot_no", ""), {**common, "type": "TOA_LOT_NO"})
                add_index_entry(index_data, "tob_lots", rec.get("tob_lot_no", ""), {**common, "type": "TOB_LOT_NO"})

            for rec in job.get("outer_boxes", []) or []:
                if not isinstance(rec, dict):
                    continue
                toc_ref = rec.get("toc_ref") or f"TOC_{int(rec.get('outer_box_no', 0) or 0):03d}"
                toc_path = os.path.join(root_dir, "outer_toc", toc_ref)
                common = {
                    **base,
                    "type": "OUTER_TOC",
                    "toc_ref": toc_ref,
                    "outer_box_no": rec.get("outer_box_no", ""),
                    "path": rel_storage_path(toc_path),
                    "toc_summary": rel_storage_path(os.path.join(toc_path, "toc_summary.json")),
                    "ocr_result": rel_storage_path(os.path.join(rec.get("storage_folder", ""), "ocr_result.json")) if rec.get("storage_folder") else "",
                    "validation_result": rec.get("validation_result", ""),
                    "fail_reason": rec.get("fail_reason", []),
                    "matched_inner_boxes": rec.get("matched_inner_boxes", []),
                    "unmatched_lots": rec.get("unmatched_lots", []),
                    "record_start_time": rec.get("start_time", ""),
                    "record_end_time": rec.get("end_time", ""),
                }
                add_index_entry(index_data, "toc_delivery_nos", rec.get("toc_delivery_no", ""), {**common, "type": "TOC_DELIVERY_NO"})
                for lot in rec.get("toc_lot_no_list", []) or []:
                    add_index_entry(index_data, "toc_lots", lot, {**common, "type": "TOC_LOT_NO"})

        total = 0
        for section in ["delivery_nos", "box_ids", "toa_lots", "tob_lots", "toc_delivery_nos", "toc_lots"]:
            total += sum(len(v) for v in index_data.get(section, {}).values())
        index_data["record_count"] = total
        index_data["updated_at"] = now_iso()
        write_json_file(GLOBAL_SEARCH_INDEX_PATH, index_data)
        print("Saved global_search_index.json:", GLOBAL_SEARCH_INDEX_PATH)
    except Exception as e:
        print("Rebuild global_search_index error:", e)


def search_global_index(value):
    """Helper for future UI/API: returns all entries that match a BOX/LOT/Delivery value."""
    index_data = read_json_safely(GLOBAL_SEARCH_INDEX_PATH)
    if not isinstance(index_data, dict):
        return []
    key = normalize_text_for_compare(value)
    results = []
    for section in ["delivery_nos", "box_ids", "toa_lots", "tob_lots", "toc_delivery_nos", "toc_lots"]:
        results.extend(index_data.get(section, {}).get(key, []))
    return results



def write_cleanup_log(log_data):
    """Save cleanup log as JSON and TXT under storage/cleanup_logs/."""
    ensure_dir(CLEANUP_LOG_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(CLEANUP_LOG_DIR, f"cleanup_{ts}.json")
    txt_path = os.path.join(CLEANUP_LOG_DIR, f"cleanup_{ts}.txt")

    try:
        write_json_file(json_path, log_data)
    except Exception as e:
        print("Cleanup JSON log error:", e)

    try:
        lines = []
        lines.append("AI Camera Storage Cleanup Log")
        lines.append("=" * 40)
        lines.append(f"started_at       : {log_data.get('started_at', '')}")
        lines.append(f"finished_at      : {log_data.get('finished_at', '')}")
        lines.append(f"retention_days   : {log_data.get('retention_days', '')}")
        lines.append(f"delete_all_mode  : {log_data.get('delete_all_date_folders', False)}")
        lines.append(f"cutoff_date      : {log_data.get('cutoff_date', '')}")
        lines.append(f"storage_root     : {log_data.get('storage_root', '')}")
        lines.append(f"deleted_count    : {log_data.get('deleted_count', 0)}")
        lines.append(f"error_count      : {len(log_data.get('errors', []))}")
        lines.append("")
        lines.append("Deleted folders:")
        deleted_items = log_data.get("deleted", [])
        if deleted_items:
            for item in deleted_items:
                lines.append(f"- {item.get('path', '')} | date={item.get('folder_date', '')} | reason={item.get('reason', '')}")
        else:
            lines.append("- No old folders deleted.")
        lines.append("")
        lines.append("Errors:")
        errors = log_data.get("errors", [])
        if errors:
            for item in errors:
                lines.append(f"- {item.get('path', '')} | {item.get('error', '')}")
        else:
            lines.append("- No errors.")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\\n".join(lines))
    except Exception as e:
        print("Cleanup TXT log error:", e)

    print("Saved cleanup log:", json_path)
    print("Saved cleanup log:", txt_path)
    return json_path, txt_path


def auto_delete_old_storage(retention_days=None):
    """
    T21: Delete YYYY/MM/DD storage folders by RETENTION_DAYS.

    Change only RETENTION_DAYS at the config section above.

    Rules:
    - Only delete folders that match storage/YYYY/MM/DD.
    - Never delete cleanup_logs or global_search_index.json.
    - Save JSON/TXT cleanup log every time the app starts.
    - Rebuild global_search_index.json only when something was deleted.
    - RETENTION_DAYS < 0 means delete all YYYY/MM/DD folders for test/clear-all mode.
    """
    if retention_days is None:
        retention_days = RETENTION_DAYS

    retention_days = int(retention_days)
    delete_all_date_folders = retention_days < 0

    ensure_dir(STORAGE_ROOT)
    ensure_dir(CLEANUP_LOG_DIR)

    started_at = now_iso()
    cutoff = None if delete_all_date_folders else (datetime.now() - timedelta(days=retention_days))

    log_data = {
        "project": "AI-Based Validation System for Shipping Label Verification",
        "version": "T28_T21_CAMERA_OPTIMIZED_FULLAPP",
        "started_at": started_at,
        "finished_at": "",
        "retention_days": retention_days,
        "delete_all_date_folders": delete_all_date_folders,
        "cutoff_date": "ALL" if delete_all_date_folders else cutoff.date().isoformat(),
        "storage_root": STORAGE_ROOT,
        "cleanup_log_dir": CLEANUP_LOG_DIR,
        "deleted_count": 0,
        "deleted": [],
        "errors": [],
    }

    try:
        for year_name in sorted(os.listdir(STORAGE_ROOT)):
            year_dir = os.path.join(STORAGE_ROOT, year_name)

            if not os.path.isdir(year_dir) or not year_name.isdigit() or len(year_name) != 4:
                continue

            for month_name in sorted(os.listdir(year_dir)):
                month_dir = os.path.join(year_dir, month_name)
                if not os.path.isdir(month_dir) or not month_name.isdigit():
                    continue

                for day_name in sorted(os.listdir(month_dir)):
                    day_dir = os.path.join(month_dir, day_name)
                    if not os.path.isdir(day_dir) or not day_name.isdigit():
                        continue

                    try:
                        folder_date = datetime(int(year_name), int(month_name), int(day_name))
                    except Exception:
                        continue

                    if delete_all_date_folders or folder_date < cutoff:
                        try:
                            shutil.rmtree(day_dir)
                            log_data["deleted"].append({
                                "path": day_dir,
                                "relative_path": rel_storage_path(day_dir),
                                "folder_date": folder_date.date().isoformat(),
                                "reason": "delete_all_date_folders" if delete_all_date_folders else f"older_than_{retention_days}_days",
                                "deleted_at": now_iso(),
                            })
                            print("Auto-delete old storage:", day_dir)
                        except Exception as e:
                            log_data["errors"].append({
                                "path": day_dir,
                                "error": str(e),
                                "time": now_iso(),
                            })
                            print("Auto-delete old storage error:", day_dir, e)

                try:
                    if os.path.isdir(month_dir) and not os.listdir(month_dir):
                        os.rmdir(month_dir)
                except Exception as e:
                    log_data["errors"].append({
                        "path": month_dir,
                        "error": f"remove_empty_month_failed: {e}",
                        "time": now_iso(),
                    })

            try:
                if os.path.isdir(year_dir) and not os.listdir(year_dir):
                    os.rmdir(year_dir)
            except Exception as e:
                log_data["errors"].append({
                    "path": year_dir,
                    "error": f"remove_empty_year_failed: {e}",
                    "time": now_iso(),
                })

    except Exception as e:
        log_data["errors"].append({
            "path": STORAGE_ROOT,
            "error": str(e),
            "time": now_iso(),
        })
        print("Auto-delete old storage error:", e)

    log_data["deleted_count"] = len(log_data["deleted"])
    log_data["finished_at"] = now_iso()
    log_data["duration_sec"] = seconds_between(started_at, log_data["finished_at"])

    write_cleanup_log(log_data)

    if log_data["deleted_count"] > 0:
        rebuild_global_search_index()

    return [item["path"] for item in log_data["deleted"]]

def save_delivery_summary():
    if not delivery_storage_dir:
        return
    attempts = []
    try:
        for name in sorted(os.listdir(delivery_storage_dir)):
            if not name.startswith("attempt_"):
                continue
            job_path = os.path.join(delivery_storage_dir, name, "job.json")
            status = name.split("_")[-1].upper() if "_" in name else "UNKNOWN"
            attempt_item = {
                "attempt_folder": name,
                "status": status,
                "job_json": job_path if os.path.exists(job_path) else "",
            }
            if os.path.exists(job_path):
                try:
                    with open(job_path, "r", encoding="utf-8") as jf:
                        jd = json.load(jf)
                    attempt_item.update({
                        "attempt_no": jd.get("attempt_no", ""),
                        "operator_en": jd.get("operator_en", ""),
                        "start_time": jd.get("start_time", jd.get("job_started_at", "")),
                        "end_time": jd.get("end_time", jd.get("job_finished_at", "")),
                        "duration_sec": jd.get("duration_sec", None),
                        "overall_result": jd.get("summary", {}).get("overall_result", ""),
                    })
                except Exception:
                    pass
            attempts.append(attempt_item)
    except Exception as e:
        print("Delivery summary scan error:", e)

    summary = {
        "project": "AI-Based Validation System for Shipping Label Verification",
        "delivery_no": delivery_no,
        "operator_en": en_no,
        "updated_at": now_iso(),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    try:
        write_json_file(os.path.join(delivery_storage_dir, "delivery_summary.json"), summary)
        print("Saved delivery_summary.json")
    except Exception as e:
        print("Save delivery_summary error:", e)


def finalize_attempt_status(status, reason=""):
    """Rename attempt_XXX_running to attempt_XXX_completed/cancelled/failed without overwriting old attempts."""
    global attempt_storage_dir, attempt_status, job_storage_dir
    if not attempt_storage_dir:
        return

    status = str(status or "RUNNING").upper()
    attempt_status = status

    if job_data:
        finished = now_iso() if status != "RUNNING" else ""
        job_data["status"] = status
        job_data["job_finished_at"] = finished
        job_data["end_time"] = finished
        if finished:
            job_data["duration_sec"] = seconds_between(job_data.get("start_time") or job_data.get("job_started_at"), finished)
        job_data.setdefault("attempt", {})["status"] = status
        job_data["attempt"]["end_time"] = finished
        job_data["attempt"]["duration_sec"] = seconds_between(job_data["attempt"].get("start_time"), finished) if finished else None
        if reason:
            if status == "CANCELLED":
                job_data["cancel_reason"] = reason
            elif status == "FAILED":
                job_data["fail_reason"] = reason
        update_job_summary()
        save_job_json()

    parent = os.path.dirname(attempt_storage_dir)
    target = os.path.join(parent, attempt_folder_name(attempt_no, status))
    if os.path.abspath(target) != os.path.abspath(attempt_storage_dir):
        base_target = target
        suffix = 1
        while os.path.exists(target):
            # Should almost never happen because attempt_no is unique, but never overwrite.
            target = base_target + f"_dup{suffix}"
            suffix += 1
        try:
            os.rename(attempt_storage_dir, target)
            attempt_storage_dir = target
            job_storage_dir = target
        except Exception as e:
            print("Attempt rename error:", e)
    save_delivery_summary()
    rebuild_global_search_index()


def init_job_data():
    global job_id, job_started_at, job_storage_dir, job_data
    global delivery_storage_dir, attempt_storage_dir, attempt_no, attempt_status, process_attempt_counter

    job_started_at = now_iso()
    job_id = make_job_id(delivery_no)

    delivery_storage_dir = ensure_dir(get_delivery_dir(delivery_no))
    attempt_no = find_next_attempt_no(delivery_storage_dir)
    attempt_status = "RUNNING"
    attempt_storage_dir = ensure_dir(os.path.join(delivery_storage_dir, attempt_folder_name(attempt_no, attempt_status)))
    job_storage_dir = attempt_storage_dir
    process_attempt_counter = {}

    ensure_dir(os.path.join(job_storage_dir, "inner_boxes"))
    ensure_dir(os.path.join(job_storage_dir, "outer_toc"))

    job_data = {
        "project": "AI-Based Validation System for Shipping Label Verification",
        "design_version": "T36_FIXED_FIELD_MAP_RESULT_LAYOUT",
        "phase": "PHASE_2_DATA_MODEL_STORAGE_VALIDATION",
        "job_id": job_id,
        "status": "RUNNING",
        "attempt_no": attempt_no,
        "attempt_folder": os.path.basename(attempt_storage_dir),
        "created_at": job_started_at,
        "start_time": job_started_at,
        "end_time": "",
        "duration_sec": None,
        "job_started_at": job_started_at,
        "job_finished_at": "",
        "operator_en": en_no,
        "picking": {
            "picking_delivery_no": delivery_no,
            "picking_item_no": item_no,
            "picking_item_qty": quantity,
            "picking_box_ids": list(box_id_list),
            "picking_api": picking_api_data,
        },
        "summary": {
            "inner_box_total": quantity,
            "outer_box_total": outer_total,
            "pass_count": 0,
            "fail_count": 0,
            "overall_result": "WAIT",
        },
        "inner_boxes": [],
        "outer_boxes": [],
        "attempt": {
            "attempt_no": attempt_no,
            "status": "RUNNING",
            "created_at": job_started_at,
            "start_time": job_started_at,
            "end_time": "",
            "duration_sec": None,
        },
        "events": [
            {"time": job_started_at, "event": "JOB_STARTED", "status": "RUNNING"}
        ],
    }
    save_job_json()
    save_delivery_summary()
    rebuild_global_search_index()


def save_job_json():
    if not job_storage_dir or not job_data:
        return
    try:
        job_data["attempt_folder"] = os.path.basename(job_storage_dir)
        job_data["updated_at"] = now_iso()
        path = os.path.join(job_storage_dir, "job.json")
        write_json_file(path, job_data)
        print("Saved job.json:", path)
    except Exception as e:
        print("Save job.json error:", e)


def update_job_summary():
    if not job_data:
        return
    records = job_data.get("inner_boxes", []) + job_data.get("outer_boxes", [])
    pass_count = sum(1 for r in records if r.get("validation_result") == "PASS")
    fail_count = sum(1 for r in records if r.get("validation_result") == "FAIL")
    wait_count = sum(1 for r in records if r.get("validation_result") == "WAIT")

    if fail_count > 0:
        overall = "FAIL"
    elif wait_count > 0 or len(records) == 0:
        overall = "WAIT"
    else:
        overall = "PASS"

    job_data["summary"].update({
        "pass_count": pass_count,
        "fail_count": fail_count,
        "overall_result": overall,
    })


def save_record_json(record, folder, filename="ocr_result.json"):
    ensure_dir(folder)
    path = os.path.join(folder, filename)
    try:
        write_json_file(path, record)
        print("Saved record JSON:", path)
    except Exception as e:
        print("Save record error:", e)


def upsert_record(record_list, key_name, key_value, record):
    for i, old in enumerate(record_list):
        if old.get(key_name) == key_value:
            record_list[i] = record
            return
    record_list.append(record)


def get_clean_ocr(data):
    return clean_value((data or {}).get("clean", ""), "")


def collect_ocr_assets(data):
    data = data or {}
    return {
        "created_at": data.get("created_at", data.get("ocr_started_at", "")),
        "updated_at": data.get("updated_at", data.get("ocr_finished_at", "")),
        "capture_time": data.get("capture_time", file_time_iso(data.get("image_path", ""))),
        "ocr_started_at": data.get("ocr_started_at", ""),
        "ocr_finished_at": data.get("ocr_finished_at", ""),
        "ocr_duration_sec": data.get("ocr_duration_sec", None),
        "image_path": data.get("image_path", ""),
        "roi_path": data.get("roi_path", ""),
        "processed_path": data.get("processed_path", ""),
        "raw": data.get("raw", ""),
        "clean": data.get("clean", ""),
        "items": data.get("items", []),
        "box": data.get("box", []),
        "source": data.get("source", ""),
    }


def next_process_attempt_dir(scope, ref, final_parent=None):
    """Create a new process_attempt_XXX_running folder. Used for Reset Process history."""
    global current_process_attempt_dir, current_process_attempt_meta
    scope_key = f"{scope}:{ref}"
    process_attempt_counter[scope_key] = process_attempt_counter.get(scope_key, 0) + 1
    no = process_attempt_counter[scope_key]

    if final_parent:
        parent = final_parent
    elif scope == "inner":
        parent = os.path.join(job_storage_dir, "inner_boxes", ref)
    else:
        parent = os.path.join(job_storage_dir, "outer_toc", ref)

    folder = ensure_dir(os.path.join(parent, f"process_attempt_{no:03d}_running"))
    current_process_attempt_dir = folder
    started = now_iso()
    current_process_attempt_meta = {
        "scope": scope,
        "ref": ref,
        "process_attempt_no": no,
        "status": "RUNNING",
        "created_at": started,
        "started_at": started,
        "start_time": started,
        "finished_at": "",
        "end_time": "",
        "duration_sec": None,
        "folder": folder,
        "images": [],
        "actions": [
            {"time": started, "action": "PROCESS_ATTEMPT_STARTED", "status": "RUNNING"}
        ],
    }
    add_job_event("PROCESS_ATTEMPT_STARTED", "RUNNING", {"scope": scope, "ref": ref, "process_attempt_no": no})
    save_record_json(current_process_attempt_meta, folder, "process_attempt.json")
    return folder


def add_process_image(path, label):
    if not current_process_attempt_meta or not path:
        return
    current_process_attempt_meta.setdefault("images", []).append({"time": now_iso(), "label": label, "path": path, "capture_time": file_time_iso(path)})
    try:
        save_record_json(current_process_attempt_meta, current_process_attempt_meta["folder"], "process_attempt.json")
    except Exception:
        pass


def finalize_process_attempt(status="final", extra=None):
    """Rename process_attempt_XXX_running to process_attempt_XXX_final/reset/failed."""
    global current_process_attempt_dir, current_process_attempt_meta
    if not current_process_attempt_dir or not os.path.exists(current_process_attempt_dir):
        return ""

    status = str(status or "final").lower()
    meta = dict(current_process_attempt_meta or {})
    finished = now_iso()
    meta["status"] = status.upper()
    meta["finished_at"] = finished
    meta["end_time"] = finished
    meta["duration_sec"] = seconds_between(meta.get("start_time") or meta.get("started_at"), finished)
    meta.setdefault("actions", []).append({"time": finished, "action": f"PROCESS_ATTEMPT_{status.upper()}", "status": status.upper()})
    if extra:
        meta.update(extra)
    save_record_json(meta, current_process_attempt_dir, "process_attempt.json")
    add_job_event(f"PROCESS_ATTEMPT_{status.upper()}", status.upper(), {"scope": meta.get("scope"), "ref": meta.get("ref"), "process_attempt_no": meta.get("process_attempt_no"), "duration_sec": meta.get("duration_sec")})

    parent = os.path.dirname(current_process_attempt_dir)
    basename = os.path.basename(current_process_attempt_dir)
    if basename.endswith("_running"):
        target = os.path.join(parent, basename[:-8] + f"_{status}")
    else:
        target = current_process_attempt_dir + f"_{status}"

    if os.path.abspath(target) != os.path.abspath(current_process_attempt_dir):
        base_target = target
        suffix = 1
        while os.path.exists(target):
            target = base_target + f"_dup{suffix}"
            suffix += 1
        try:
            os.rename(current_process_attempt_dir, target)
            current_process_attempt_dir = target
            meta["folder"] = target
            save_record_json(meta, current_process_attempt_dir, "process_attempt.json")
        except Exception as e:
            print("Process attempt rename error:", e)
    final_dir = current_process_attempt_dir
    current_process_attempt_dir = ""
    current_process_attempt_meta = {}
    return final_dir


def relocate_process_attempt_to_parent(process_dir, target_parent):
    """Move process attempt from UNKNOWN/PENDING folder into final Vendor/TOC folder after OCR gives final key."""
    global current_process_attempt_dir, current_process_attempt_meta
    if not process_dir or not os.path.exists(process_dir):
        return ""
    ensure_dir(target_parent)
    target = os.path.join(target_parent, os.path.basename(process_dir))
    if os.path.abspath(process_dir) == os.path.abspath(target):
        return process_dir
    base_target = target
    suffix = 1
    while os.path.exists(target):
        target = base_target + f"_dup{suffix}"
        suffix += 1
    try:
        shutil.move(process_dir, target)
        if os.path.abspath(current_process_attempt_dir or "") == os.path.abspath(process_dir):
            current_process_attempt_dir = target
            if current_process_attempt_meta:
                current_process_attempt_meta["folder"] = target
                save_record_json(current_process_attempt_meta, target, "process_attempt.json")
        return target
    except Exception as e:
        print("Move process attempt error:", e)
        return process_dir


def build_inner_box_record(inner_no, cam1_data, cam2_data, process_dir=""):
    # T30: Prefer field-based YOLO ROI OCR from CAM1/CAM2.
    # Fallback behavior remains compatible with the old center-ROI OCR result.
    # V2.2 change:
    # - Inner folder is stable: INNER_001, INNER_002, ...
    # - Vendor/TOA/TOB evidence belongs inside each process_attempt folder.
    # - Do NOT create INNER_001_VENDOR_PENDING / UNKNOWN / VALUE anymore.
    vendor_box_data = get_inner_field_data(cam1_data, "vendor_box_id", {})
    vendor_date_data = get_inner_field_data(cam1_data, "vendor_date_code", {})
    toa_lot_data = get_inner_field_data(cam1_data, "toa_lot_no", {})
    toa_date_data = get_inner_field_data(cam1_data, "toa_date_code", {})
    tob_lot_data = get_inner_field_data(cam2_data, "tob_lot_no", {})
    tob_date_data = get_inner_field_data(cam2_data, "tob_date_code", {})

    vendor_box_id = get_clean_ocr(vendor_box_data)
    vendor_date_code = get_clean_ocr(vendor_date_data)
    toa_lot_no = get_clean_ocr(toa_lot_data)
    toa_date_code = get_clean_ocr(toa_date_data)
    tob_lot_no = get_clean_ocr(tob_lot_data)
    tob_date_code = get_clean_ocr(tob_date_data)

    fail_reason = []

    allowed_box_id_list = get_api_box_id_list()
    vendor_box_in_api, allowed_box_id_list = box_id_allowed_any_order(vendor_box_id)
    used_vendor_box_ids = get_used_vendor_box_ids(inner_no)
    vendor_box_is_duplicate = (
        bool(vendor_box_id)
        and normalize_text_for_compare(vendor_box_id) in [normalize_text_for_compare(x) for x in used_vendor_box_ids]
    )

    if not allowed_box_id_list:
        fail_reason.append("RULE_1_API_BOX_ID_LIST_MISSING")
    elif not vendor_box_id:
        fail_reason.append("RULE_1_VENDOR_BOX_ID_OCR_EMPTY")
    elif not vendor_box_in_api:
        fail_reason.append("RULE_1_VENDOR_BOX_ID_NOT_FOUND_IN_API_LIST")

    if vendor_box_is_duplicate:
        fail_reason.append("RULE_1_VENDOR_BOX_ID_DUPLICATE_IN_JOB")

    # T33 display-only validation: validate everything we can read, but do not block OK/NEXT.
    if not vendor_date_code:
        fail_reason.append("RULE_3_VENDOR_DATE_EMPTY")
    if not toa_date_code:
        fail_reason.append("RULE_3_TOA_DATE_EMPTY")
    if not tob_date_code:
        fail_reason.append("RULE_3_TOB_DATE_EMPTY")

    date_values = [vendor_date_code, toa_date_code, tob_date_code]
    if all(date_values) and len(set(normalize_text_for_compare(v) for v in date_values)) != 1:
        fail_reason.append("RULE_3_DATE_CODE_NOT_MATCH")

    previous_toa = [x.get("toa_lot_no", "") for x in job_data.get("inner_boxes", []) if x.get("sequence") != inner_no] if job_data else []
    previous_tob = [x.get("tob_lot_no", "") for x in job_data.get("inner_boxes", []) if x.get("sequence") != inner_no] if job_data else []

    if not toa_lot_no:
        fail_reason.append("RULE_5_TOA_LOT_OCR_EMPTY")
    elif normalize_text_for_compare(toa_lot_no) in [normalize_text_for_compare(x) for x in previous_toa]:
        fail_reason.append("RULE_5_TOA_LOT_DUPLICATE_IN_JOB")

    if not tob_lot_no:
        fail_reason.append("RULE_6_TOB_LOT_OCR_EMPTY")
    elif normalize_text_for_compare(tob_lot_no) in [normalize_text_for_compare(x) for x in previous_tob]:
        fail_reason.append("RULE_6_TOB_LOT_DUPLICATE_IN_JOB")

    validation_result = "FAIL" if fail_reason else "PASS"

    inner_ref = f"INNER_{inner_no:03d}"
    inner_folder = ensure_dir(os.path.join(job_storage_dir, "inner_boxes", inner_ref)) if job_storage_dir else ""

    # process_attempt is the owner of the OCR/evidence result for this round.
    # If process_dir is missing for any reason, fall back to inner_folder to avoid crashing.
    process_storage_folder = process_dir or inner_folder

    vendor_folder_name = f"Vendor_{safe_folder_name(vendor_box_id)}"
    toa_folder_name = f"TOA_{safe_folder_name(toa_lot_no)}"
    tob_folder_name = f"TOB_{safe_folder_name(tob_lot_no)}"

    vendor_folder = ensure_dir(os.path.join(process_storage_folder, vendor_folder_name)) if process_storage_folder else ""
    toa_folder = ensure_dir(os.path.join(process_storage_folder, toa_folder_name)) if process_storage_folder else ""
    tob_folder = ensure_dir(os.path.join(process_storage_folder, tob_folder_name)) if process_storage_folder else ""

    evidence = {
        "vendor": {
            "full_image": copy_if_exists(vendor_box_data.get("image_path", cam1_data.get("image_path", "")), vendor_folder, "full_vendor_toa.jpg"),
            "roi_image": copy_if_exists(vendor_box_data.get("roi_path", ""), vendor_folder, f"vendor_boxid_{safe_folder_name(vendor_box_id)}.jpg"),
            "processed_image": copy_if_exists(vendor_box_data.get("processed_path", ""), vendor_folder, "processed_vendor.jpg"),
        },
        "toa": {
            "full_image": copy_if_exists(toa_lot_data.get("image_path", cam1_data.get("image_path", "")), toa_folder, "full_vendor_toa.jpg"),
            "roi_image": copy_if_exists(toa_lot_data.get("roi_path", ""), toa_folder, f"toa_lot_{safe_folder_name(toa_lot_no)}.jpg"),
            "processed_image": copy_if_exists(toa_lot_data.get("processed_path", ""), toa_folder, "processed_toa.jpg"),
        },
        "tob": {
            "full_image": copy_if_exists(tob_lot_data.get("image_path", cam2_data.get("image_path", "")), tob_folder, "full_tob.jpg"),
            "roi_image": copy_if_exists(tob_lot_data.get("roi_path", ""), tob_folder, f"tob_lot_{safe_folder_name(tob_lot_no)}.jpg"),
            "processed_image": copy_if_exists(tob_lot_data.get("processed_path", ""), tob_folder, "processed_tob.jpg"),
        },
    }

    record = {
        "type": "inner_box",
        "inner_ref": inner_ref,
        "sequence": inner_no,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "start_time": (current_process_attempt_meta or {}).get("start_time", ""),
        "end_time": now_iso(),
        "duration_sec": seconds_between((current_process_attempt_meta or {}).get("start_time", ""), now_iso()),
        "process_attempt_no": (current_process_attempt_meta or {}).get("process_attempt_no", None),
        "operator_en": en_no,
        "delivery_no": delivery_no,
        "expected_vendor_box_id": "ANY_ORDER_IN_API_LIST",
        "allowed_vendor_box_ids": allowed_box_id_list,
        "used_vendor_box_ids_before_current": used_vendor_box_ids,
        "vendor_box_id": vendor_box_id,
        "vendor_date_code": vendor_date_code,
        "toa_lot_no": toa_lot_no,
        "toa_date_code": toa_date_code,
        "tob_lot_no": tob_lot_no,
        "tob_date_code": tob_date_code,
        "validation_result": validation_result,
        "validation_mode": "DISPLAY_ONLY_NOT_BLOCKING",
        "can_continue_even_if_fail": True,
        "fail_reason": fail_reason,
        "fail_reason_text": [validation_reason_label(x) for x in fail_reason],
        "validation": {
            "rule_1_vendor_box_id_in_api_list_any_order": "PASS" if "RULE_1_API_BOX_ID_LIST_MISSING" not in fail_reason and "RULE_1_VENDOR_BOX_ID_OCR_EMPTY" not in fail_reason and "RULE_1_VENDOR_BOX_ID_NOT_FOUND_IN_API_LIST" not in fail_reason else "FAIL",
            "rule_1_vendor_box_id_not_duplicate": "PASS" if "RULE_1_VENDOR_BOX_ID_DUPLICATE_IN_JOB" not in fail_reason else "FAIL",
            "rule_1_picking_box_id_eq_vendor_box_id": "PASS" if "RULE_1_API_BOX_ID_LIST_MISSING" not in fail_reason and "RULE_1_VENDOR_BOX_ID_OCR_EMPTY" not in fail_reason and "RULE_1_VENDOR_BOX_ID_NOT_FOUND_IN_API_LIST" not in fail_reason and "RULE_1_VENDOR_BOX_ID_DUPLICATE_IN_JOB" not in fail_reason else "FAIL",
            "rule_3_vendor_date_read": "PASS" if "RULE_3_VENDOR_DATE_EMPTY" not in fail_reason else "FAIL",
            "rule_3_toa_date_read": "PASS" if "RULE_3_TOA_DATE_EMPTY" not in fail_reason else "FAIL",
            "rule_3_tob_date_read": "PASS" if "RULE_3_TOB_DATE_EMPTY" not in fail_reason else "FAIL",
            "rule_3_date_code_match_all": "PASS" if "RULE_3_VENDOR_DATE_EMPTY" not in fail_reason and "RULE_3_TOA_DATE_EMPTY" not in fail_reason and "RULE_3_TOB_DATE_EMPTY" not in fail_reason and "RULE_3_DATE_CODE_NOT_MATCH" not in fail_reason else "FAIL",
            "rule_5_toa_lot_read": "PASS" if "RULE_5_TOA_LOT_OCR_EMPTY" not in fail_reason else "FAIL",
            "rule_5_toa_lot_not_duplicate": "PASS" if "RULE_5_TOA_LOT_OCR_EMPTY" not in fail_reason and "RULE_5_TOA_LOT_DUPLICATE_IN_JOB" not in fail_reason else "FAIL",
            "rule_6_tob_lot_read": "PASS" if "RULE_6_TOB_LOT_OCR_EMPTY" not in fail_reason else "FAIL",
            "rule_6_tob_lot_not_duplicate": "PASS" if "RULE_6_TOB_LOT_OCR_EMPTY" not in fail_reason and "RULE_6_TOB_LOT_DUPLICATE_IN_JOB" not in fail_reason else "FAIL",
        },
        "ocr": {
            "cam1_vendor_toa": collect_ocr_assets(cam1_data),
            "cam2_tob": collect_ocr_assets(cam2_data),
            "cam1_fields": {k: collect_ocr_assets(v) for k, v in (cam1_data.get("fields", {}) or {}).items()},
            "cam2_fields": {k: collect_ocr_assets(v) for k, v in (cam2_data.get("fields", {}) or {}).items()},
            "cam1_detections": cam1_data.get("detections", []),
            "cam2_detections": cam2_data.get("detections", []),
        },
        "evidence": evidence,
        "process_attempt_folder": process_storage_folder,
        "storage_folder": inner_folder,
    }

    # Summary file at INNER_001 level tells which OCR values are currently the latest/final view.
    inner_summary = {
        "inner_ref": inner_ref,
        "sequence": inner_no,
        "created_at": file_time_iso(os.path.join(inner_folder, "inner_summary.json")) or record["created_at"],
        "updated_at": now_iso(),
        "latest_start_time": record.get("start_time", ""),
        "latest_end_time": record.get("end_time", ""),
        "latest_duration_sec": record.get("duration_sec", None),
        "latest_vendor_box_id": vendor_box_id or "UNKNOWN",
        "latest_toa_lot_no": toa_lot_no or "UNKNOWN",
        "latest_tob_lot_no": tob_lot_no or "UNKNOWN",
        "latest_validation_result": validation_result,
        "latest_process_attempt_folder": process_storage_folder,
    }
    if inner_folder:
        save_record_json(inner_summary, inner_folder, "inner_summary.json")

    return record

def save_inner_record(record):
    if not job_data:
        return
    upsert_record(job_data["inner_boxes"], "sequence", record["sequence"], record)
    update_job_summary()
    save_record_json(record, record.get("process_attempt_folder") or record.get("storage_folder", job_storage_dir))
    save_job_json()
    save_delivery_summary()
    rebuild_global_search_index()


def get_inner_lot_map():
    lot_map = {}
    if not job_data:
        return lot_map
    for rec in job_data.get("inner_boxes", []):
        lot = rec.get("toa_lot_no", "")
        if lot:
            lot_map[normalize_text_for_compare(lot)] = {
                "inner_ref": rec.get("inner_ref", ""),
                "toa_lot_no": lot,
                "vendor_box_id": rec.get("vendor_box_id", ""),
            }
    return lot_map


def build_outer_box_record(toc_no, toc_total, item_map, fallback_data=None, process_dir=""):
    item_map = item_map or {}
    toc_delivery_no = get_clean_ocr(item_map.get("toc") or fallback_data or {})

    lot_list = []
    for name in ["toc1", "toc2", "toc3", "toc4", "toc5", "toc6"]:
        lot = get_clean_ocr(item_map.get(name, {}))
        if lot:
            lot_list.append(lot)

    lot_map = get_inner_lot_map()
    matched_inner_boxes = []
    unmatched_lots = []
    for lot in lot_list:
        matched = lot_map.get(normalize_text_for_compare(lot))
        if matched:
            matched_inner_boxes.append(matched["inner_ref"])
        else:
            unmatched_lots.append(lot)

    fail_reason = []
    if toc_delivery_no and delivery_no and normalize_text_for_compare(toc_delivery_no) != normalize_text_for_compare(delivery_no):
        fail_reason.append("RULE_2_PICKING_DELIVERY_NO_NOT_MATCH_TOC_DELIVERY_NO")
    if unmatched_lots:
        fail_reason.append("RULE_4_TOA_LOT_NOT_FOUND_IN_TOC_LOT_LIST")

    validation_result = "FAIL" if fail_reason else "PASS"
    toc_ref = f"TOC_{toc_no:03d}"
    toc_folder = ensure_dir(os.path.join(job_storage_dir, "outer_toc", toc_ref)) if job_storage_dir else ""
    if process_dir:
        # Keep all TOC / TOC1-TOC6 evidence inside the current process attempt.
        # Correct V2 storage:
        # outer_toc/TOC_001/process_attempt_XXX_running|reset|final/TOC, TOC1, ... TOC6
        process_dir = relocate_process_attempt_to_parent(process_dir, toc_folder)
    process_storage_folder = process_dir or toc_folder

    evidence = {}
    for name, data in item_map.items():
        # IMPORTANT: row folders must live inside process_attempt, not directly under TOC_001.
        row_folder = ensure_dir(os.path.join(process_storage_folder, name.upper())) if process_storage_folder else ""
        evidence[name] = {
            "full_image": copy_if_exists(data.get("image_path", ""), row_folder, f"full_{name}.jpg"),
            "roi_image": copy_if_exists(data.get("roi_path", ""), row_folder, f"roi_{name}_{safe_folder_name(data.get('clean', ''))}.jpg"),
            "processed_image": copy_if_exists(data.get("processed_path", ""), row_folder, f"processed_{name}.jpg"),
        }

    if fallback_data is not None and "toc" not in evidence:
        # Fallback is also evidence from this process attempt, so keep it inside process_attempt too.
        row_folder = ensure_dir(os.path.join(process_storage_folder, "TOC_FALLBACK")) if process_storage_folder else ""
        evidence["toc_fallback"] = {
            "full_image": copy_if_exists(fallback_data.get("image_path", ""), row_folder, "full_toc_fallback.jpg"),
            "roi_image": copy_if_exists(fallback_data.get("roi_path", ""), row_folder, "roi_toc_fallback.jpg"),
            "processed_image": copy_if_exists(fallback_data.get("processed_path", ""), row_folder, "processed_toc_fallback.jpg"),
        }

    process_final_dir = process_dir or ""

    return {
        "type": "outer_box",
        "toc_ref": toc_ref,
        "outer_box_no": toc_no,
        "outer_box_total": toc_total,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "start_time": (current_process_attempt_meta or {}).get("start_time", ""),
        "end_time": now_iso(),
        "duration_sec": seconds_between((current_process_attempt_meta or {}).get("start_time", ""), now_iso()),
        "process_attempt_no": (current_process_attempt_meta or {}).get("process_attempt_no", None),
        "operator_en": en_no,
        "delivery_no": delivery_no,
        "toc_delivery_no": toc_delivery_no,
        "toc_lot_no_list": lot_list,
        "matched_inner_boxes": matched_inner_boxes,
        "unmatched_lots": unmatched_lots,
        "validation_result": validation_result,
        "fail_reason": fail_reason,
        "validation": {
            "rule_2_picking_delivery_no_eq_toc_delivery_no": "PASS" if "RULE_2_PICKING_DELIVERY_NO_NOT_MATCH_TOC_DELIVERY_NO" not in fail_reason else "FAIL",
            "rule_4_toa_lot_exists_in_toc_lot_list": "PASS" if "RULE_4_TOA_LOT_NOT_FOUND_IN_TOC_LOT_LIST" not in fail_reason else "FAIL",
        },
        "ocr": {name: collect_ocr_assets(data) for name, data in item_map.items()},
        "fallback": collect_ocr_assets(fallback_data) if fallback_data else None,
        "evidence": evidence,
        "process_attempt_final": process_final_dir,
        "toc_folder": toc_folder,
        # Save ocr_result.json inside process_attempt so reset/final history is complete.
        "storage_folder": process_storage_folder,
    }


def save_outer_record(record):
    if not job_data:
        return
    upsert_record(job_data["outer_boxes"], "outer_box_no", record["outer_box_no"], record)
    update_job_summary()
    save_record_json(record, record.get("storage_folder", job_storage_dir))
    save_job_json()
    save_delivery_summary()
    rebuild_global_search_index()


def finish_job_data():
    log_event("JOB_COMPLETE")
    if not job_data:
        return
    job_data.setdefault("events", []).append({
        "time": now_iso(),
        "event": "JOB_COMPLETED",
        "status": "COMPLETED",
    })
    finalize_attempt_status("COMPLETED")


def fail_job_data(reason="process_failed"):
    log_event("JOB_FAILED", level="ERROR", reason=reason)
    if not job_data:
        return
    if current_process_attempt_dir:
        finalize_process_attempt("failed", {"reason": reason})
    job_data.setdefault("events", []).append({
        "time": now_iso(),
        "event": "JOB_FAILED",
        "status": "FAILED",
        "fail_reason": reason,
    })
    finalize_attempt_status("FAILED", reason)

def dbg(msg):
    try:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        th = threading.current_thread().name
        ev = action_event.is_set()
        print(f"[DBG {ts} {th} action={action_value} event={ev} running={running}] {msg}", flush=True)
        # T24: keep high-value debug events in system log, but avoid 1-second WAIT spam.
        if "still waiting" not in str(msg):
            log_event("DEBUG", level="DEBUG", message=str(msg), thread=th)
    except Exception as e:
        print("[DBG ERROR]", e, msg, flush=True)


def ui_call(name, func):
    try:
        dbg(f"UI_CALL START: {name}")
        func()
        dbg(f"UI_CALL END: {name}")
    except Exception:
        print(f"[UI ERROR] {name}", flush=True)
        traceback.print_exc()


def now_name(prefix):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.jpg")


def capture_path(prefix):
    return os.path.join(CAPTURE_DIR, now_name(prefix))


def current_process_alive():
    """Used by camera loops. False means current capture should stop/restart/camera-reset/cancel."""
    return running and action_value is None



# =========================================================
# T28 OPTIMIZED PI CAMERA ENGINE
# This replaces PiCameraTest only for CAM1/CAM2 capture flow.
# CAM3 still uses T27 USB rediscovery/recovery.
# =========================================================
class OptimizedPiCamera:
    def __init__(self, index, name, status_cb=None, preview_cb=None, focus_delay_sec=1.0):
        self.index = index
        self.name = name
        self.status_cb = status_cb
        self.preview_cb = preview_cb
        self.focus_delay_sec = float(focus_delay_sec)
        self.picam2 = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.capturing = False
        self.closed = False

    def _status(self, text):
        if callable(self.status_cb):
            try:
                self.status_cb(text)
            except Exception:
                pass
        else:
            print(text)

    def open(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available. Please install/use on Raspberry Pi.")

        if self.picam2 is not None:
            return

        self._status(f"{self.name} opening optimized 2K no-switch...")
        self.picam2 = Picamera2(camera_num=self.index)
        config = self.picam2.create_preview_configuration(
            main={"size": (PI_CAPTURE_WIDTH, PI_CAPTURE_HEIGHT), "format": PI_CAPTURE_FORMAT}
        )
        self.picam2.configure(config)

        # OCR-friendly image tuning. Safe defaults; change here if labels look over-sharpened.
        try:
            self.picam2.set_controls({
                "Sharpness": 1.2,
                "Contrast": 1.1,
            })
        except Exception as e:
            print(f"{self.name} image control warning:", e)

        self.picam2.start()
        self.closed = False
        time.sleep(PI_CAMERA_OPEN_SETTLE_SEC)
        self._status(f"{self.name} Ready optimized 2K {PI_CAPTURE_WIDTH}x{PI_CAPTURE_HEIGHT}")

    def _capture_array(self):
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")
        with self.lock:
            frame = self.picam2.capture_array()
        self.latest_frame = frame
        return frame

    def _preview(self, frame):
        if callable(self.preview_cb):
            try:
                self.preview_cb(frame, is_bgr=False)
            except TypeError:
                self.preview_cb(frame)
            except Exception as e:
                print(f"{self.name} preview callback error:", e)

    def _motion_frame(self, frame):
        try:
            small = cv2.resize(frame, PI_MOTION_ANALYZE_SIZE)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (PICAM_BLUR_SIZE, PICAM_BLUR_SIZE), 0)
            return gray
        except Exception:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return cv2.GaussianBlur(gray, (PICAM_BLUR_SIZE, PICAM_BLUR_SIZE), 0)

    def wait_background(self, keep_running):
        self._status(f"{self.name}: Learning background")
        bg = None
        start = time.time()
        while keep_running() and time.time() - start < 1.0:
            frame = self._capture_array()
            self._preview(frame)
            bg = self._motion_frame(frame)
            time.sleep(0.08)

        if bg is None:
            return None

        self._status(f"{self.name}: Background ready")
        return bg

    def wait_object(self, background, keep_running):
        self._status(f"{self.name}: Waiting object by motion")
        stable_start = None
        last_score = 0

        while keep_running():
            frame = self._capture_array()
            self._preview(frame)

            gray = self._motion_frame(frame)
            diff = cv2.absdiff(background, gray)
            _, th = cv2.threshold(diff, PICAM_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            try:
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area = sum(cv2.contourArea(c) for c in contours)
            except Exception:
                area = 0

            # T29 threshold fix:
            # The original Pi motion threshold was tuned from binary-mask SUM values
            # (0/255 image), not from raw countNonZero.
            # countNonZero max at 1280x720 is only ~0.92M, so it can never reach
            # PICAM_MOTION_THRESHOLD=8,000,000. Use sum/255-scale score instead.
            white_pixels = int(cv2.countNonZero(th))
            score = int(cv2.sumElems(th)[0])  # equivalent to white_pixels * 255
            last_score = score

            if score >= PICAM_MOTION_THRESHOLD:
                if stable_start is None:
                    stable_start = time.time()
                elapsed = time.time() - stable_start
                self._status(f"{self.name}: Object detected {elapsed:.1f}/{PICAM_STABLE_TIME:.1f}s | score={score} threshold={PICAM_MOTION_THRESHOLD}")
                if elapsed >= PICAM_STABLE_TIME:
                    return True
            else:
                stable_start = None
                self._status(f"{self.name}: Waiting object | score={score} threshold={PICAM_MOTION_THRESHOLD}")

            time.sleep(0.12)

        print(f"{self.name}: wait_object stopped, last_score={last_score}")
        return False

    def autofocus(self, keep_running=None):
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")

        self._status(f"{self.name}: Autofocus")
        t0 = time.time()
        af_ok = False

        try:
            if controls is not None:
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
            try:
                self.picam2.autofocus_cycle(wait=True)
            except TypeError:
                self.picam2.autofocus_cycle()
            af_ok = True
        except Exception as e:
            print(f"{self.name} autofocus_cycle(wait=True) failed, fallback timed trigger:", e)

        if not af_ok:
            try:
                if controls is not None:
                    self.picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Auto,
                        "AfTrigger": controls.AfTriggerEnum.Start,
                    })
            except Exception as e:
                print(f"{self.name} AfTrigger fallback failed:", e)

            end = time.time() + max(0.0, self.focus_delay_sec)
            while time.time() < end:
                if keep_running is not None and not keep_running():
                    break
                time.sleep(0.02)

        # Tiny settle only. Do not add another full 1.0/1.5 sec delay.
        end_settle = time.time() + PI_AF_SETTLE_SEC
        while time.time() < end_settle:
            if keep_running is not None and not keep_running():
                break
            time.sleep(0.02)

        elapsed = time.time() - t0
        print(f"{self.name} T28 autofocus total wait: {elapsed:.2f}s")
        return elapsed

    # Compatibility with old T27 calls.
    def focus_delay(self, keep_running):
        return self.autofocus(keep_running)

    def capture_file(self, path):
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")

        self.capturing = True
        try:
            frame = self._capture_array()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(PI_JPEG_QUALITY)])
            if not ok:
                raise RuntimeError(f"{self.name} cv2.imwrite failed: {path}")
            print(f"{self.name}: saved optimized 2K {path} ({frame.shape[1]}x{frame.shape[0]})")
            return path
        finally:
            self.capturing = False

    def close(self):
        try:
            self.closed = True
            self.capturing = True
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
        except Exception as e:
            print(f"{self.name} optimized close error:", e)
        self.picam2 = None
        self.capturing = False


def close_optimized_pi_cameras():
    global optimized_persistent_cam1, optimized_persistent_cam2, current_camera

    with optimized_persistent_lock:
        for cam in (optimized_persistent_cam1, optimized_persistent_cam2):
            try:
                if cam is not None:
                    cam.close()
            except Exception as e:
                print("close optimized persistent cam error:", e)

        optimized_persistent_cam1 = None
        optimized_persistent_cam2 = None

    try:
        if isinstance(current_camera, OptimizedPiCamera):
            current_camera = None
    except Exception:
        pass


def get_or_open_optimized_picam(cam_no):
    """cam_no: 1 = CAM1, 2 = CAM2."""
    global optimized_persistent_cam1, optimized_persistent_cam2

    if not USE_T21_OPTIMIZED_PI_CAMERAS:
        return None

    index = PICAM1_INDEX if cam_no == 1 else PICAM2_INDEX
    name = "CAM1" if cam_no == 1 else "CAM2"
    focus_delay = PICAM1_FOCUS_DELAY_SEC if cam_no == 1 else PICAM2_FOCUS_DELAY_SEC

    if not PERSISTENT_PI_CAMERAS:
        cam = OptimizedPiCamera(index=index, name=name, status_cb=set_status, preview_cb=update_preview, focus_delay_sec=focus_delay)
        cam.open()
        return cam

    with optimized_persistent_lock:
        if cam_no == 1:
            if optimized_persistent_cam1 is None or optimized_persistent_cam1.picam2 is None:
                optimized_persistent_cam1 = OptimizedPiCamera(index=index, name=name, status_cb=set_status, preview_cb=update_preview, focus_delay_sec=focus_delay)
                optimized_persistent_cam1.open()
            return optimized_persistent_cam1

        if optimized_persistent_cam2 is None or optimized_persistent_cam2.picam2 is None:
            optimized_persistent_cam2 = OptimizedPiCamera(index=index, name=name, status_cb=set_status, preview_cb=update_preview, focus_delay_sec=focus_delay)
            optimized_persistent_cam2.open()
        return optimized_persistent_cam2


def prewarm_optimized_cam2_background():
    if not USE_T21_OPTIMIZED_PI_CAMERAS or not PREWARM_CAM2_ON_START or not PERSISTENT_PI_CAMERAS:
        return

    def worker():
        try:
            if not running:
                return
            log_event("CAM2_PREWARM_START", camera="CAM2")
            get_or_open_optimized_picam(2)
            log_event("CAM2_PREWARM_OK", camera="CAM2")
            root.after(0, lambda: set_status("CAM2 prewarmed / optimized standby ready"))
        except Exception as e:
            log_event("CAM2_PREWARM_ERROR", level="WARNING", camera="CAM2", error=str(e))
            root.after(0, lambda e=e: set_status(f"CAM2 prewarm failed: {e}"))

    threading.Thread(target=worker, daemon=True, name="cam2_prewarm").start()


def is_retry_action():
    return action_value in ("restart", "camera_reset")


def retry_reason(default="operator_reset"):
    if action_value == "camera_reset":
        # T23: Strong in-process camera reset.
        # Keep OCR/YOLO models loaded, but give Picamera2/libcamera/V4L2 time
        # to release the current camera handle before the same step opens again.
        force_camera_release_for_reset()
        return "operator_camera_reset"
    return default


def force_camera_release_for_reset(wait_sec=2.0):
    global current_camera

    try:
        set_status(f"Reset camera: releasing camera handle {wait_sec:.1f}s...")
    except Exception:
        pass

    try:
        close_optimized_pi_cameras()
    except Exception as e:
        print("T28 reset close optimized pi cameras error:", e)

    try:
        if current_camera is not None:
            current_camera.close()
    except Exception as e:
        print("T23 reset close current_camera error:", e)

    current_camera = None

    try:
        gc.collect()
    except Exception as e:
        print("T23 gc.collect error:", e)

    try:
        time.sleep(float(wait_sec))
    except Exception:
        pass


def set_action(action):
    """Set user action: next, restart, camera_reset, or cancel."""
    global action_value, running, current_camera

    dbg(f"ACTION requested: {action}")
    log_event("USER_ACTION", action=action)
    action_value = action
    action_event.set()

    if action == "cancel":
        running = False

    # Force camera loops to unblock quickly.
    try:
        if current_camera is not None:
            current_camera.close()
    except Exception as e:
        print("set_action camera close error:", e)

    if action == "camera_reset":
        try:
            gc.collect()
        except Exception as e:
            print("set_action gc.collect error:", e)


def clear_action():
    global action_value
    action_value = None
    action_event.clear()


def reset_current_camera():
    """T24: Reset only the currently active camera/stage without cancelling the job.

    Flow:
    - signal current capture loop to stop
    - close current camera handle
    - gc.collect()
    - wait longer for Picamera2/libcamera/V4L2 to release
    - retry the same Inner/Outer index

    This does NOT restart the app and does NOT reload OCR/YOLO models.
    """
    if current_screen_mode not in ("capture_inner", "capture_outer"):
        try:
            messagebox.showinfo("Reset Camera", "Reset Camera is available only on the capture screen.")
        except Exception:
            pass
        return

    try:
        set_status("Reset current camera... close + gc + retry same step")
    except Exception:
        pass

    # set_action() closes the active camera immediately so camera loops unblock.
    # retry_reason() performs gc.collect() + 2s wait before the same stage opens again.
    log_event("CAMERA_RESET_REQUESTED")
    set_action("camera_reset")

def set_status(text):
    try:
        status_label.config(text=text)

        color = "#f1c40f"
        lower = text.lower()
        if "capture" in lower or "capturing" in lower or "opening" in lower:
            color = "#3498db"
        elif "ocr" in lower or "reading" in lower:
            color = "#e67e22"
        elif "ready" in lower:
            color = "#27ae60"
        elif "stop" in lower or "error" in lower or "cancel" in lower:
            color = "#c0392b"

        if capture_state_label is not None:
            capture_state_label.config(text=text)

        if capture_state_dot is not None:
            capture_state_dot.delete("all")
            capture_state_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

        root.update_idletasks()
    except Exception:
        pass

    print(text)
    if any(k in str(text).lower() for k in ["error", "reset", "opening", "saved", "cannot", "ocr", "ready", "capture"]):
        log_event("STATUS", status=str(text))


def set_step(text):
    try:
        step_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass
    print("STEP:", text)
    log_event("STEP", step=str(text))


def update_preview(frame, is_bgr=False):
    global last_preview_time

    now = time.time()
    if now - last_preview_time < PREVIEW_INTERVAL:
        return
    last_preview_time = now

    try:
        display = cv2.resize(frame, PREVIEW_SIZE)
        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk, text="")
    except Exception as e:
        print("Preview error:", e)


def make_thumbnail_image(image_path, size=(120, 60)):
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(image=Image.fromarray(img))
    except Exception as e:
        print("Thumbnail error:", e)
        return None


def show_page(page):
    page_name = getattr(page, "_debug_name", str(page))
    dbg(f"show_page -> {page_name}")
    for p in [page_input, page_search, page_capture, page_result, page_ready_toc, page_complete]:
        p.pack_forget()
    page.pack(fill="both", expand=True)
    try:
        root.update_idletasks()
    except Exception:
        pass


def get_current_count(mode):
    if mode == "outer":
        return current_outer_index, outer_total
    return current_index, quantity


def update_info(mode="inner"):
    idx, total = get_current_count(mode)
    box_title = "Outer Box" if mode == "outer" else "Inner Box"

    info_delivery.config(text=delivery_no or "-")
    info_item.config(text=item_no or "-")
    info_en.config(text=en_no or "-")
    info_box_title.config(text=box_title)
    info_box.config(text=f"{idx} / {total}" if total else "-")
    info_pack.config(text=pack_type)


def set_progress(mode="inner"):
    idx, total = get_current_count(mode)
    label = "Outer Box" if mode == "outer" else "Inner Box"
    progress_title_label.config(text=label)
    progress_label.config(text=f"{idx} / {total}" if total else "0 / 0")

    try:
        percent = int((idx / total) * 100) if total > 0 else 0
        progress_percent_label.config(text=f"{percent}%")
        progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)
    except Exception:
        pass

    root.update_idletasks()


def set_checklist(active=None, done=None, mode="inner"):
    """
    Checklist count rule:
    - TOA / TOB are Inner Box processes, so their count follows Inner Box.
      During Outer Box capture, TOA / TOB are already finished, so show quantity/quantity.
    - TOC is Outer Box process, so its count follows Outer Box.
      During Inner Box capture, show the first planned Outer Box count, e.g. 1/1 or 1/6.
    """
    done = done or []

    if mode == "outer":
        inner_count = f"{quantity}/{quantity}" if quantity else "-/-"
        toc_idx = current_outer_index if current_outer_index > 0 else 1
        toc_count = f"{toc_idx}/{outer_total}" if outer_total else "-/-"
    else:
        inner_count = f"{current_index}/{quantity}" if quantity else "-/-"
        toc_idx = 1 if outer_total > 0 else 0
        toc_count = f"{toc_idx}/{outer_total}" if outer_total else "-/-"

    for name, label in check_labels.items():
        count_text = toc_count if name == "TOC" else inner_count

        if name in done:
            label.config(text=f"[OK]  {name:<4} {count_text}", fg=GREEN, bg=GREEN_BG)
        elif active == name:
            label.config(text=f"[RUN] {name:<4} {count_text}", fg=YELLOW_TEXT, bg=YELLOW_BG)
        else:
            label.config(text=f"[ ]   {name:<4} {count_text}", fg=TEXT, bg=SOFT)


def setup_capture_page(mode="inner", active="TOA", done=None):
    global current_screen_mode
    current_screen_mode = "capture_outer" if mode == "outer" else "capture_inner"
    update_info(mode)
    set_progress(mode)
    set_checklist(active=active, done=done or [], mode=mode)
    show_page(page_capture)


# =========================================================
# OCR PROCESS HELPERS
# =========================================================
def process_center_roi_ocr(image_path, prefix):
    log_event("OCR_CENTER_START", image_path=image_path, prefix=prefix)
    ocr_started = now_iso()
    img = cv2.imread(image_path)
    if img is None:
        ocr_finished = now_iso()
        log_event("OCR_CENTER_ERROR", level="ERROR", prefix=prefix, image_path=image_path, error="cannot_read_image")
        return {
            "name": prefix,
            "created_at": ocr_started,
            "updated_at": ocr_finished,
            "capture_time": file_time_iso(image_path),
            "ocr_started_at": ocr_started,
            "ocr_finished_at": ocr_finished,
            "ocr_duration_sec": seconds_between(ocr_started, ocr_finished),
            "image_path": image_path,
            "error": "cannot_read_image",
            "raw": "",
            "clean": "",
        }

    crop, box = crop_center(img)
    processed = preprocess_ocr(crop)
    roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)
    ocr = run_easyocr(processed)
    ocr_finished = now_iso()
    log_event("OCR_CENTER_END", prefix=prefix, duration_sec=seconds_between(ocr_started, ocr_finished), raw=ocr.get("raw", ""), clean=ocr.get("clean", ""))

    return {
        "name": prefix,
        "created_at": ocr_started,
        "updated_at": ocr_finished,
        "capture_time": file_time_iso(image_path),
        "ocr_started_at": ocr_started,
        "ocr_finished_at": ocr_finished,
        "ocr_duration_sec": seconds_between(ocr_started, ocr_finished),
        "source": "center_roi",
        "image_path": image_path,
        "roi_path": roi_path,
        "processed_path": processed_path,
        "box": box,
        "raw": ocr["raw"],
        "clean": ocr["clean"],
        "items": ocr["items"],
    }


def process_yolo_roi_ocr(image_path, detections):
    log_event("OCR_YOLO_ROI_START", image_path=image_path, detection_count=len(detections or []))
    img = cv2.imread(image_path)
    if img is None:
        return []

    results = []
    for det in detections:
        row_started = now_iso()
        name = det["name"]
        box = det["box"]
        crop, fixed_box = crop_by_box(img, box, pad=0)
        if crop is None or crop.size == 0:
            continue

        processed = preprocess_ocr(crop)
        prefix = f"toc_{name}"
        roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)
        ocr = run_easyocr(processed)
        row_finished = now_iso()

        results.append({
            "name": name,
            "created_at": row_started,
            "updated_at": row_finished,
            "capture_time": file_time_iso(image_path),
            "ocr_started_at": row_started,
            "ocr_finished_at": row_finished,
            "ocr_duration_sec": seconds_between(row_started, row_finished),
            "source": "yolo_roi",
            "detect_conf": det["conf"],
            "image_path": image_path,
            "roi_path": roi_path,
            "processed_path": processed_path,
            "box": fixed_box,
            "raw": ocr["raw"],
            "clean": ocr["clean"],
            "items": ocr["items"],
        })

    log_event("OCR_YOLO_ROI_END", image_path=image_path, result_count=len(results))
    return results


def save_json(data, filename):
    path = os.path.join(OCR_DIR, filename)
    if isinstance(data, dict):
        data.setdefault("created_at", now_iso())
        data["updated_at"] = now_iso()
        data["saved_at"] = now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", path)


# =========================================================
# RESULT UI HELPERS
# =========================================================
def load_result_photo(image_path, size=(300, 86)):
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(image=Image.fromarray(img))
    except Exception as e:
        print("Result image load error:", e)
        return None


def update_result_image(label, image_path, size=(340, 70)):
    photo = load_result_photo(image_path, size=size)
    if photo is not None:
        label.imgtk = photo
        label.config(image=photo, text="")
    else:
        label.imgtk = None
        label.config(image="", text="NO IMAGE", fg="white", bg=DARK, font=("Arial", 10, "bold"))


def update_value_box(label, value):
    text = value or "(empty)"
    is_ok = text != "(empty)"
    label.config(text=text, bg=GREEN_BG if is_ok else "#f8f9fa", fg=GREEN if is_ok else "#7f8c8d")


def update_validation_box(label, value, status="WAIT"):
    text = value or "-"
    status = (status or "WAIT").upper()

    if status == "PASS":
        bg = GREEN_BG
        fg = GREEN
    elif status in ("NG", "FAIL"):
        bg = "#fdecea"
        fg = RED
    else:
        bg = "#f8f9fa"
        fg = "#7f8c8d"

    label.config(text=text, bg=bg, fg=fg)


def update_status_badge(label, status="WAIT"):
    status = (status or "WAIT").upper()

    if status == "PASS":
        text = "PASS"
        bg = GREEN
        fg = "white"
    elif status in ("NG", "FAIL"):
        text = "NG"
        bg = RED
        fg = "white"
    else:
        text = "WAIT"
        bg = "#f1c40f"
        fg = TEXT

    label.config(text=text, bg=bg, fg=fg)


def make_result_field(parent, field_title, value_title):
    """
    Result field for Inner Box validation.
    Layout:
    - ROI Image
    - OCR Result
    - Validation / DB Result
    - Status Badge
    """
    block = tk.Frame(parent, bg=CARD)
    block.pack(fill="x", pady=(0, 14))

    tk.Label(
        block,
        text=field_title,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 12, "bold")
    ).pack(anchor="w", pady=(0, 5))

    # T36: use a fixed-size frame for ROI preview.
    # Do not set Label(width=340, height=62) because Tkinter treats those as text units.
    # When image is missing, that made one result card expand across the whole UI.
    image_wrap = tk.Frame(block, bg=DARK, width=340, height=62)
    image_wrap.pack(fill="x", pady=(0, 7))
    image_wrap.pack_propagate(False)

    img_box = tk.Label(image_wrap, bg=DARK, fg="white", text="NO IMAGE", font=("Arial", 10, "bold"))
    img_box.pack(fill="both", expand=True)

    # OCR Result row
    ocr_row = tk.Frame(block, bg=CARD)
    ocr_row.pack(fill="x", pady=(0, 4))

    tk.Label(
        ocr_row,
        text=value_title,
        bg=CARD,
        fg=TEXT,
        font=("Arial", 12, "bold"),
        width=12,
        anchor="w"
    ).pack(side="left")

    value_box = tk.Label(
        ocr_row,
        text="-",
        bg="#f8f9fa",
        fg="#7f8c8d",
        font=("Arial", 17, "bold"),
        anchor="w",
        padx=10,
        pady=6
    )
    value_box.pack(side="left", fill="x", expand=True)

    # Validation / DB Result row
    validation_row = tk.Frame(block, bg=CARD)
    validation_row.pack(fill="x", pady=(0, 5))

    tk.Label(
        validation_row,
        text="Validate :",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 12, "bold"),
        width=12,
        anchor="w"
    ).pack(side="left")

    validation_box = tk.Label(
        validation_row,
        text="WAIT",
        bg="#f8f9fa",
        fg="#7f8c8d",
        font=("Arial", 14, "bold"),
        anchor="w",
        padx=10,
        pady=6
    )
    validation_box.pack(side="left", fill="x", expand=True)

    status_badge = tk.Label(
        block,
        text="WAIT",
        bg="#f1c40f",
        fg=TEXT,
        font=("Arial", 13, "bold"),
        padx=14,
        pady=6
    )
    status_badge.pack(anchor="w")

    return {
        "image": img_box,
        "value": value_box,
        "validation": validation_box,
        "status": status_badge,
    }


def make_product_result_section(parent, key, title, icon_key, field1, value1, field2, value2):
    card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
    card.pack(side="left", fill="both", expand=True, padx=6)

    title_row = tk.Frame(card, bg=CARD)
    title_row.pack(fill="x", pady=(0, 12))
    icon_label(title_row, icon_key, bg=CARD).pack(side="left", padx=(0, 8))
    tk.Label(title_row, text=title, bg=CARD, fg=TEXT, font=("Arial", 21, "bold")).pack(side="left")

    first = make_result_field(card, field1, value1)

    # Reserved area for future DB / validation detail fields.
    # Kept large because the operator page will later show extra validation data here.
    tk.Frame(card, bg=CARD, height=180).pack(fill="x")

    second = make_result_field(card, field2, value2)

    result_product_slots[key] = {"first": first, "second": second}


def update_result_field(field, image_path, ocr_value, validation_text="WAIT", status="WAIT"):
    update_result_image(field["image"], image_path, size=(340, 62))
    update_value_box(field["value"], ocr_value)
    update_validation_box(field["validation"], validation_text, status)
    update_status_badge(field["status"], status)


def update_product_result_section(
    key,
    first_image_path,
    first_value,
    first_validation="WAIT",
    first_status="WAIT",
    second_image_path=None,
    second_value=None,
    second_validation="WAIT",
    second_status="WAIT",
):
    section = result_product_slots.get(key)
    if not section:
        return

    if second_image_path is None:
        second_image_path = first_image_path
    if second_value is None:
        second_value = first_value

    update_result_field(section["first"], first_image_path, first_value, first_validation, first_status)
    update_result_field(section["second"], second_image_path, second_value, second_validation, second_status)


def mock_read_status(value):
    """Temporary test status until real DB / duplicate validation is connected."""
    return "PASS" if value and value != "(empty)" else "WAIT"


def mock_vendor_db_validation(value):
    if value and value != "(empty)":
        return f"DB Result : {value}"
    return "DB Result : WAIT DB"


def mock_duplicate_validation(value):
    if value and value != "(empty)":
        return "Duplicate : NOT FOUND"
    return "Duplicate : WAIT"


def mock_date_compare_validation(value, all_values):
    valid_values = [v for v in all_values if v and v != "(empty)"]
    if len(valid_values) < 3:
        return "Compare : WAIT", "WAIT"
    if len(set(valid_values)) == 1:
        return "Compare : MATCH ALL", "PASS"
    return "Compare : NOT MATCH", "NG"



def normalize_text_for_compare(value):
    return str(value or "").strip().replace(" ", "").upper()


def get_toc_row_validation(name, clean_text):
    """
    Temporary TOC validation UI for testing.
    - TOC row compares OCR value with Delivary No.
    - TOC1-TOC6 rows compare OCR value with stored TOA Lot No values from Inner Box results.
    """
    read_value = clean_text if clean_text and clean_text != "(empty)" else "-"
    name = str(name or "").lower()

    if read_value == "-":
        if name == "toc":
            return {
                "read": read_value,
                "label": "Delivary No",
                "match": delivery_no or "-",
                "status": "WAIT",
            }
        return {
            "read": read_value,
            "label": "Matched TOA",
            "match": "-",
            "status": "WAIT",
        }

    if name == "toc":
        target = delivery_no or "-"
        status = "PASS" if normalize_text_for_compare(read_value) == normalize_text_for_compare(target) else "NG"
        return {
            "read": read_value,
            "label": "Delivary No",
            "match": target,
            "status": status,
        }

    # TOC1-TOC6: compare with all TOA Lot No values collected during Inner Box review.
    read_norm = normalize_text_for_compare(read_value)
    matched = "-"
    for _, lot_value in sorted(inner_toa_lot_by_index.items()):
        if normalize_text_for_compare(lot_value) == read_norm:
            matched = lot_value
            break

    status = "PASS" if matched != "-" else "NG"
    return {
        "read": read_value,
        "label": "Matched TOA",
        "match": matched,
        "status": status,
    }


def make_validation_status_badge(parent, status):
    status = (status or "WAIT").upper()
    if status == "PASS":
        text, bg, fg = "PASS", GREEN, "white"
    elif status == "NG":
        text, bg, fg = "NG", RED, "white"
    else:
        text, bg, fg = "WAIT", "#f1c40f", TEXT

    return tk.Label(
        parent,
        text=text,
        bg=bg,
        fg=fg,
        font=("Arial", 10, "bold"),
        padx=10,
        pady=3,
        width=8,
    )


def clear_toc_cards():
    try:
        for w in toc_result_area.winfo_children():
            w.destroy()
    except Exception:
        pass


def add_toc_result_row(parent, row_no, title, image_data=None):
    data = image_data or {}
    clean_text = data.get("clean", "") or "(empty)"
    raw_text = data.get("raw", "") or ""
    source_text = data.get("source", "-")
    detect_conf = data.get("detect_conf", None)
    if detect_conf is not None:
        source_text = f"{source_text} / DET {detect_conf:.2f}"

    validation = get_toc_row_validation(title, clean_text)

    row = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=6)
    row.pack(fill="x", padx=12, pady=3)

    name_box = tk.Frame(row, bg=CARD, width=88)
    name_box.pack(side="left", fill="y", padx=(0, 10))
    name_box.pack_propagate(False)

    tk.Label(name_box, text=str(row_no), bg=BLUE, fg="white", font=("Arial", 10, "bold"), width=3, pady=2).pack(anchor="w")
    tk.Label(name_box, text=title.upper(), bg=CARD, fg=TEXT, font=("Arial", 12, "bold"), anchor="w").pack(anchor="w", pady=(4, 0))

    roi_path = data.get("roi_path", "")
    photo = load_result_photo(roi_path, size=(260, 56))

    image_wrap = tk.Frame(row, bg=DARK, width=260, height=56)
    image_wrap.pack(side="left", padx=(0, 14))
    image_wrap.pack_propagate(False)

    img_box = tk.Label(image_wrap, bg=DARK, fg="white", text="NO ROI", font=("Arial", 9, "bold"))
    img_box.pack(fill="both", expand=True)
    if photo is not None:
        img_box.imgtk = photo
        img_box.config(image=photo, text="")

    info_box = tk.Frame(row, bg=CARD)
    info_box.pack(side="left", fill="both", expand=True)

    top_line = tk.Frame(info_box, bg=CARD)
    top_line.pack(fill="x")
    tk.Label(top_line, text="OCR Result", bg=CARD, fg=MUTED, font=("Arial", 11, "bold")).pack(side="left")
    tk.Label(top_line, text=source_text, bg=CARD, fg=MUTED, font=("Arial", 8), anchor="e").pack(side="right")

    value_box = tk.Label(
        info_box,
        text=clean_text,
        bg=GREEN_BG if clean_text != "(empty)" else "#f8f9fa",
        fg=GREEN if clean_text != "(empty)" else "#7f8c8d",
        font=("Arial", 14, "bold"),
        anchor="w",
        padx=10,
        pady=4,
    )
    value_box.pack(fill="x", pady=(3, 2))

    tk.Label(info_box, text=f"RAW : {raw_text or '-'}", bg=CARD, fg=MUTED, font=("Arial", 9), anchor="w").pack(anchor="w")

    validation_row = tk.Frame(info_box, bg=CARD)
    validation_row.pack(fill="x", pady=(4, 0))

    tk.Label(
        validation_row,
        text=f"Read Value : {validation['read']}",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 10, "bold"),
        anchor="w",
        width=28,
    ).pack(side="left")

    tk.Label(
        validation_row,
        text=f"{validation['label']} : {validation['match']}",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 10, "bold"),
        anchor="w",
        width=34,
    ).pack(side="left", padx=(8, 8))

    tk.Label(
        validation_row,
        text="Result :",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 10, "bold"),
        anchor="w",
    ).pack(side="left", padx=(0, 6))

    make_validation_status_badge(validation_row, validation["status"]).pack(side="left")


def show_product_result(product_no, cam1_data, cam2_data):
    global current_review_context, current_screen_mode
    current_screen_mode = "result_inner"
    current_review_context = f"INNER BOX {product_no}/{quantity}"
    dbg(f"SHOW PRODUCT RESULT product={product_no}/{quantity}")

    show_page(page_result)
    result_title.config(text=f"OCR Result : Inner Box {product_no} / {quantity}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)
    result_hint.config(text="Review ROI image, OCR result, validation result, then press OK / NEXT to continue.")

    product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    toc_result_area.pack_forget()

    record = build_inner_box_record(product_no, cam1_data, cam2_data, process_dir=current_process_attempt_dir)
    save_inner_record(record)

    if record.get("validation_result") == "PASS":
        result_status_badge.config(text="PASS / REVIEW", bg=GREEN, fg="white")
    else:
        result_status_badge.config(text="NG / REVIEW", bg=RED, fg="white")
    result_hint.config(text=validation_error_summary(record.get("fail_reason", [])))
    log_event("INNER_VALIDATION_DISPLAY_ONLY", inner_no=product_no, result=record.get("validation_result"), fail_reason=record.get("fail_reason", []), can_continue=True)

    cam1_text = record.get("toa_lot_no", "") or "(empty)"
    cam2_text = record.get("tob_lot_no", "") or "(empty)"

    # Backward-compatible memory for existing TOC UI validation.
    if cam1_text and cam1_text != "(empty)":
        inner_toa_lot_by_index[product_no] = cam1_text

    vendor_box_roi = get_inner_field_data(cam1_data, "vendor_box_id", {}).get("roi_path", "")
    vendor_date_roi = get_inner_field_data(cam1_data, "vendor_date_code", {}).get("roi_path", "")
    toa_lot_roi = get_inner_field_data(cam1_data, "toa_lot_no", {}).get("roi_path", "")
    toa_date_roi = get_inner_field_data(cam1_data, "toa_date_code", {}).get("roi_path", "")
    tob_lot_roi = get_inner_field_data(cam2_data, "tob_lot_no", {}).get("roi_path", "")
    tob_date_roi = get_inner_field_data(cam2_data, "tob_date_code", {}).get("roi_path", "")

    vendor_box_status = "PASS" if record["validation"]["rule_1_picking_box_id_eq_vendor_box_id"] == "PASS" and record["vendor_box_id"] else "NG"
    toa_lot_status = "PASS" if record["validation"]["rule_5_toa_lot_not_duplicate"] == "PASS" and record["toa_lot_no"] else "NG"
    tob_lot_status = "PASS" if record["validation"]["rule_6_tob_lot_not_duplicate"] == "PASS" and record["tob_lot_no"] else "NG"

    date_status = record["validation"]["rule_3_date_code_match_all"]
    if date_status == "PASS":
        date_validation = "Compare : MATCH ALL"
    else:
        date_validation = "Compare : NOT MATCH"

    vendor_validation = f"API List : {'FOUND' if record['validation'].get('rule_1_vendor_box_id_in_api_list_any_order') == 'PASS' else 'NOT FOUND'} | Duplicate : {record['validation'].get('rule_1_vendor_box_id_not_duplicate', 'WAIT')}"
    toa_validation = record["validation"]["rule_5_toa_lot_not_duplicate"]
    tob_validation = record["validation"]["rule_6_tob_lot_not_duplicate"]

    update_product_result_section(
        "vendor",
        vendor_box_roi,
        record["vendor_box_id"] or "(empty)",
        f"BOX ID : {vendor_validation}",
        vendor_box_status,
        vendor_date_roi,
        record["vendor_date_code"] or "(empty)",
        date_validation,
        date_status,
    )

    update_product_result_section(
        "toa",
        toa_lot_roi,
        record["toa_lot_no"] or "(empty)",
        f"Duplicate : {toa_validation}",
        toa_lot_status,
        toa_date_roi,
        record["toa_date_code"] or "(empty)",
        date_validation,
        date_status,
    )

    update_product_result_section(
        "tob",
        tob_lot_roi,
        record["tob_lot_no"] or "(empty)",
        f"Duplicate : {tob_validation}",
        tob_lot_status,
        tob_date_roi,
        record["tob_date_code"] or "(empty)",
        date_validation,
        date_status,
    )

    # Keep old test JSON output too, so existing debugging folders still work.
    save_json(record, f"inner_box_{product_no:03d}.json")


def show_toc_result(toc_items, fallback_data=None, toc_no=1, toc_total=1):
    global current_review_context, current_screen_mode
    current_screen_mode = "result_outer"
    current_review_context = f"OUTER BOX {toc_no}/{toc_total}"
    dbg(f"SHOW TOC RESULT outer={toc_no}/{toc_total} items={len(toc_items or [])} fallback={fallback_data is not None}")

    show_page(page_result)
    result_title.config(text=f"OCR Result : Outer Box {toc_no} / {toc_total}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)

    product_result_area.pack_forget()
    toc_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    clear_toc_cards()

    result_hint.config(text="Review TOC ROI, OCR result, and validation result for each row.")

    item_map = {}
    for item in toc_items or []:
        name = str(item.get("name", "")).lower()
        item_map[name] = item

    expected_names = ["toc", "toc1", "toc2", "toc3", "toc4", "toc5", "toc6"]
    if fallback_data is not None and not item_map:
        item_map["toc"] = fallback_data

    record = build_outer_box_record(toc_no, toc_total, item_map, fallback_data=fallback_data, process_dir=current_process_attempt_dir)
    save_outer_record(record)

    for idx, name in enumerate(expected_names, start=1):
        add_toc_result_row(toc_result_area, row_no=idx, title=name, image_data=item_map.get(name))

    # Keep old test JSON output too, so existing debugging folders still work.
    save_json(record, f"outer_box_{toc_no:02d}.json")


# =========================================================
# JOB FLOW
# =========================================================
def start_job():
    global running, en_no, delivery_no, quantity, current_index, current_outer_index, outer_total, item_no, action_value, inner_toa_lot_by_index
    global picking_api_data, box_id_list

    if not preload_done_event.is_set():
        messagebox.showwarning("System Loading", "Please wait until startup preload is ready.")
        return

    en_no = normalize_scan_text(en_entry.get())
    delivery_no = normalize_scan_text(delivery_entry.get())

    if not en_no:
        messagebox.showwarning("Warning", "Please input EN")
        return
    if not delivery_no:
        messagebox.showwarning("Warning", "Please input Delivary NO")
        return

    # T31: Quantity and Item No come from Mock API by Delivery No.
    if ENABLE_MOCK_API:
        try:
            root.update_idletasks()
            set_status("Calling Mock API /api/picking...")
            picking_api_data = fetch_picking_from_api(delivery_no)
            delivery_no = picking_api_data.get("delivery_no", delivery_no)
            item_no = picking_api_data.get("item_no", "-") or "-"
            quantity = int(picking_api_data.get("quantity", 0))
            box_id_list = list(picking_api_data.get("box_ids", []) or [])
            apply_picking_to_input_fields(picking_api_data)
        except Exception as e:
            picking_api_data = {}
            box_id_list = []
            log_event("JOB_START_BLOCKED_BY_API", level="ERROR", delivery_no=delivery_no, error=str(e))
            messagebox.showerror("Mock API Error", f"Cannot start job.\n\n{e}")
            set_status(f"Mock API Error: {e}")
            return
    else:
        qty_text = normalize_scan_text(qty_entry.get())
        item_no = "-"
        if not qty_text.isdigit():
            messagebox.showwarning("Warning", "Please input Quantity")
            return
        quantity = int(qty_text)
        box_id_list = []
        picking_api_data = {}

    if quantity < 1 or quantity > 1000:
        messagebox.showwarning("Warning", "Quantity must be 1-1000")
        return

    if box_id_list and len(box_id_list) < quantity:
        messagebox.showwarning(
            "Warning",
            f"API BOX ID list has only {len(box_id_list)} item(s), but Quantity is {quantity}."
        )
        return

    outer_total = math.ceil(quantity / 6)
    inner_toa_lot_by_index = {}
    current_index = 0
    current_outer_index = 0
    running = True
    clear_action()
    init_job_data()
    log_event("JOB_START", en_no=en_no, delivery_no=delivery_no, item_no=item_no, quantity=quantity, outer_total=outer_total, box_id_count=len(box_id_list))

    if USE_T21_OPTIMIZED_PI_CAMERAS:
        close_optimized_pi_cameras()
        root.after(300, prewarm_optimized_cam2_background)

    dbg("START JOB")
    setup_capture_page(mode="inner", active="TOA", done=[])
    set_status("Loading OCR Model...")

    thread = threading.Thread(target=job_loop, daemon=True)
    thread.start()


def job_loop():
    global current_index, running

    # T9: OCR is preloaded at app startup. Calling load_ocr() again is safe/idempotent
    # in the existing ocr_test module, and protects us if startup preload failed.
    if not preload_done_event.is_set():
        root.after(0, lambda: set_status("Waiting startup preload..."))
        preload_done_event.wait(timeout=8)

    root.after(0, lambda: set_status("OCR Ready"))

    for i in range(1, quantity + 1):
        if not running:
            return

        current_index = i
        root.after(0, lambda i=i: setup_capture_page(mode="inner", active="TOA", done=[]))

        ok = process_product(i)
        if not ok:
            if action_value == "cancel":
                root.after(0, reset_to_input)
            else:
                fail_job_data("inner_process_failed")
            running = False
            return

        sleep(0.4)

    try:
        close_optimized_pi_cameras()
    except Exception as e:
        log_event("OPTIMIZED_PI_CLOSE_AFTER_INNER_ERROR", level="WARNING", error=str(e))

    running = False
    root.after(0, lambda: show_page(page_ready_toc))


def process_product(product_no):
    global current_camera

    while running:
        clear_action()
        inner_ref = f"INNER_{product_no:03d}"
        process_dir = next_process_attempt_dir("inner", inner_ref)
        root.after(0, lambda: setup_capture_page(mode="inner", active="TOA", done=[]))
        root.after(0, lambda: set_step("CAM1/CAM2 : Optimized 2K no-switch capture"))
        root.after(0, lambda: set_status("CAM1/CAM2 optimized capture starting"))

        cam1_path = os.path.join(process_dir, now_name(f"cam1_inner{product_no}"))
        cam2_path = os.path.join(process_dir, now_name(f"cam2_inner{product_no}"))

        # =====================================================
        # T28 optimized path from T21
        # - CAM1/CAM2 use 2304x1296 no-switch stream.
        # - CAM2 is prewarmed/opened in background.
        # - AF/capture are sequential because dual AF was slower.
        # - Keep all T27 reset/recovery/review/storage behavior.
        # =====================================================
        if USE_T21_OPTIMIZED_PI_CAMERAS:
            cam_retry = 0

            while running:
                cam1 = None
                cam2 = None

                try:
                    log_event("T28_OPTIMIZED_INNER_START", inner_no=product_no, retry=cam_retry)

                    # ===================== CAM1 =====================
                    root.after(0, lambda: setup_capture_page(mode="inner", active="TOA", done=[]))
                    root.after(0, lambda: set_step("CAM1 : Detect object by optimized 2K background"))
                    root.after(0, lambda: set_status("CAM1 optimized opening/standby"))

                    cam1 = get_or_open_optimized_picam(1)
                    current_camera = cam1
                    log_event("CAMERA_OPEN_OK", camera="CAM1", index=PICAM1_INDEX, retry=cam_retry, optimized=True, resolution=f"{PI_CAPTURE_WIDTH}x{PI_CAPTURE_HEIGHT}")

                    bg = cam1.wait_background(current_process_alive)
                    if bg is None:
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        raise RuntimeError("CAM1 cannot learn background")

                    detected = cam1.wait_object(bg, current_process_alive)
                    if not detected:
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        raise RuntimeError("CAM1 cannot detect object / read frame")

                    cam1.focus_delay(current_process_alive)
                    if action_value is not None or not running:
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        return False

                    cap_start = time.time()
                    cam1.capture_file(cam1_path)
                    log_event("CAPTURE_DONE", camera="CAM1", path=cam1_path, duration_sec=round(time.time() - cap_start, 3), retry=cam_retry, optimized=True)
                    add_process_image(cam1_path, "CAM1_VENDOR_TOA_FULL")

                    # ===================== CAM2 =====================
                    if not running:
                        return False

                    root.after(0, lambda: setup_capture_page(mode="inner", active="TOB", done=["TOA"]))
                    root.after(0, lambda: set_step("CAM2 : Optimized 2K direct capture"))
                    root.after(0, lambda: set_status("CAM2 optimized opening/standby"))

                    cam2 = get_or_open_optimized_picam(2)
                    current_camera = cam2
                    log_event("CAMERA_OPEN_OK", camera="CAM2", index=PICAM2_INDEX, retry=cam_retry, optimized=True, resolution=f"{PI_CAPTURE_WIDTH}x{PI_CAPTURE_HEIGHT}")

                    cam2.focus_delay(current_process_alive)
                    if action_value is not None or not running:
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        return False

                    cap_start = time.time()
                    cam2.capture_file(cam2_path)
                    log_event("CAPTURE_DONE", camera="CAM2", path=cam2_path, duration_sec=round(time.time() - cap_start, 3), retry=cam_retry, optimized=True)
                    add_process_image(cam2_path, "CAM2_TOB_FULL")

                    log_event("T28_OPTIMIZED_INNER_CAPTURE_OK", inner_no=product_no, retry=cam_retry)
                    current_camera = None
                    break

                except Exception as e:
                    log_event("T28_OPTIMIZED_INNER_ERROR", level="ERROR", error=str(e), traceback=traceback.format_exc(), retry=cam_retry)

                    try:
                        close_optimized_pi_cameras()
                    except Exception as ce:
                        log_event("T28_OPTIMIZED_CLOSE_ERROR", level="ERROR", error=str(ce))

                    current_camera = None

                    if is_retry_action():
                        finalize_process_attempt("reset", {"reason": retry_reason()})
                        break

                    if cam_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and should_auto_recover_error(str(e)):
                        if auto_recover_camera("CAM1/CAM2", str(e)):
                            cam_retry += 1
                            root.after(0, lambda: set_status("CAM1/CAM2 optimized auto recovery retry..."))
                            continue

                    op_action = wait_operator_camera_fix("CAM1/CAM2", f"{e}")
                    if op_action == "retry":
                        finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_optimized_inner_error")})
                        break
                    return False

            if is_retry_action():
                continue

        # =====================================================
        # Legacy fallback path from T27
        # Keep this block available by setting USE_T21_OPTIMIZED_PI_CAMERAS = False.
        # =====================================================
        else:
            # ===================== CAM1 =====================
            cam1_retry = 0
            while running:
                cam1 = PiCameraTest(index=PICAM1_INDEX, name="CAM1", status_cb=set_status, preview_cb=update_preview)
                current_camera = cam1
                try:
                    log_event("CAMERA_OPEN_START", camera="CAM1", index=PICAM1_INDEX, retry=cam1_retry)
                    cam1.open()
                    log_event("CAMERA_OPEN_OK", camera="CAM1", index=PICAM1_INDEX, retry=cam1_retry)

                    bg = cam1.wait_background(current_process_alive)
                    if bg is None:
                        cam1.close()
                        current_camera = None
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        if cam1_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and auto_recover_camera("CAM1", "background_not_ready"):
                            cam1_retry += 1
                            continue
                        op_action = wait_operator_camera_fix("CAM1", "CAM1 cannot learn background.")
                        if op_action == "retry":
                            finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_background_fail")})
                            break
                        return False

                    detected = cam1.wait_object(bg, current_process_alive)
                    if not detected:
                        cam1.close()
                        current_camera = None
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        if cam1_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and auto_recover_camera("CAM1", "object_detect_failed"):
                            cam1_retry += 1
                            continue
                        op_action = wait_operator_camera_fix("CAM1", "CAM1 cannot detect/read frame.")
                        if op_action == "retry":
                            finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_detect_fail")})
                            break
                        return False

                    cam1.focus_delay(current_process_alive)
                    if action_value is not None or not running:
                        cam1.close()
                        current_camera = None
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        return False

                    cap_start = time.time()
                    cam1.capture_file(cam1_path)
                    log_event("CAPTURE_DONE", camera="CAM1", path=cam1_path, duration_sec=round(time.time() - cap_start, 3), retry=cam1_retry)
                    add_process_image(cam1_path, "CAM1_VENDOR_TOA_FULL")
                    break

                except Exception as e:
                    log_event("CAM1_ERROR", level="ERROR", error=str(e), traceback=traceback.format_exc(), retry=cam1_retry)
                    try:
                        cam1.close()
                    except Exception:
                        pass
                    current_camera = None

                    if is_retry_action():
                        finalize_process_attempt("reset", {"reason": retry_reason()})
                        break

                    if cam1_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and should_auto_recover_error(str(e)):
                        if auto_recover_camera("CAM1", str(e)):
                            cam1_retry += 1
                            root.after(0, lambda: set_status("CAM1 auto recovery retry..."))
                            continue

                    op_action = wait_operator_camera_fix("CAM1", f"{e}")
                    if op_action == "retry":
                        finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_cam1_error")})
                        break
                    return False

                finally:
                    try:
                        cam1.close()
                        log_event("CAMERA_CLOSE", camera="CAM1")
                    except Exception as e:
                        log_event("CAMERA_CLOSE_ERROR", level="ERROR", camera="CAM1", error=str(e))
                    current_camera = None

            if is_retry_action():
                continue

            # ===================== CAM2 =====================
            if not running:
                return False

            root.after(0, lambda: setup_capture_page(mode="inner", active="TOB", done=["TOA"]))
            root.after(0, lambda: set_step("CAM2 : Capture directly"))
            root.after(0, lambda: set_status("CAM2 opening"))

            cam2_retry = 0
            while running:
                cam2 = PiCameraTest(index=PICAM2_INDEX, name="CAM2", status_cb=set_status, preview_cb=update_preview)
                current_camera = cam2
                try:
                    log_event("CAMERA_OPEN_START", camera="CAM2", index=PICAM2_INDEX, retry=cam2_retry)
                    cam2.open()
                    log_event("CAMERA_OPEN_OK", camera="CAM2", index=PICAM2_INDEX, retry=cam2_retry)

                    cam2.focus_delay(current_process_alive)
                    if action_value is not None or not running:
                        cam2.close()
                        current_camera = None
                        if is_retry_action():
                            finalize_process_attempt("reset", {"reason": retry_reason()})
                            break
                        return False

                    cap_start = time.time()
                    cam2.capture_file(cam2_path)
                    log_event("CAPTURE_DONE", camera="CAM2", path=cam2_path, duration_sec=round(time.time() - cap_start, 3), retry=cam2_retry)
                    add_process_image(cam2_path, "CAM2_TOB_FULL")
                    break

                except Exception as e:
                    log_event("CAM2_ERROR", level="ERROR", error=str(e), traceback=traceback.format_exc(), retry=cam2_retry)
                    try:
                        cam2.close()
                    except Exception:
                        pass
                    current_camera = None

                    if is_retry_action():
                        finalize_process_attempt("reset", {"reason": retry_reason()})
                        break

                    if cam2_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and should_auto_recover_error(str(e)):
                        if auto_recover_camera("CAM2", str(e)):
                            cam2_retry += 1
                            root.after(0, lambda: set_status("CAM2 auto recovery retry..."))
                            continue

                    op_action = wait_operator_camera_fix("CAM2", f"{e}")
                    if op_action == "retry":
                        finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_cam2_error")})
                        break
                    return False

                finally:
                    try:
                        cam2.close()
                        log_event("CAMERA_CLOSE", camera="CAM2")
                    except Exception as e:
                        log_event("CAMERA_CLOSE_ERROR", level="ERROR", camera="CAM2", error=str(e))
                    current_camera = None

            if is_retry_action():
                continue

        # ===================== OCR + REVIEW =====================
        if not running:
            return False
        if is_retry_action():
            finalize_process_attempt("reset", {"reason": retry_reason()})
            continue

        if USE_INNER_YOLO_ROI_OCR:
            root.after(0, lambda: set_step("OCR : YOLO ROI CAM1/CAM2"))
            root.after(0, lambda: set_status("OCR reading from CAM1/CAM2 YOLO ROI"))
            cam1_data = process_inner_yolo_roi_ocr(cam1_path, "cam1")
            cam2_data = process_inner_yolo_roi_ocr(cam2_path, "cam2")
        else:
            root.after(0, lambda: set_step("OCR : Center ROI CAM1/CAM2"))
            root.after(0, lambda: set_status("OCR reading"))
            cam1_data = process_center_roi_ocr(cam1_path, "cam1")
            cam2_data = process_center_roi_ocr(cam2_path, "cam2")

        clear_action()
        root.after(0, lambda: setup_capture_page(mode="inner", active=None, done=["TOA", "TOB"]))
        root.after(0, lambda: show_product_result(product_no, cam1_data, cam2_data))

        action = wait_review_action(f"Inner Box {product_no}/{quantity}")
        if action == "next":
            finalize_process_attempt("final", {"reason": "operator_ok_next"})
            return True
        if action == "restart":
            finalize_process_attempt("reset", {"reason": "operator_reset_after_review"})
            continue
        return False

    return False


def continue_toc():
    global running, current_outer_index

    running = True
    current_outer_index = 0
    clear_action()
    log_event("CONTINUE_TOC", outer_total=outer_total)
    dbg("CONTINUE TOC")

    setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"])
    thread = threading.Thread(target=toc_loop, daemon=True)
    thread.start()


def toc_loop():
    global running, current_outer_index

    dbg(f"TOC LOOP START quantity={quantity} outer_total={outer_total}")

    for toc_no in range(1, outer_total + 1):
        if not running:
            return

        current_outer_index = toc_no
        root.after(0, lambda: setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"]))

        ok = process_toc(toc_no=toc_no, toc_total=outer_total)
        if not ok:
            if action_value == "cancel":
                root.after(0, reset_to_input)
            else:
                fail_job_data("outer_process_failed")
            running = False
            return

        sleep(0.4)

    running = False
    finish_job_data()
    root.after(0, lambda: show_page(page_complete))


def process_toc(toc_no=1, toc_total=1):
    global current_camera

    while running:
        clear_action()
        toc_ref = f"TOC_{toc_no:03d}"
        process_dir = next_process_attempt_dir("outer", toc_ref)
        root.after(0, lambda: setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"]))
        root.after(0, lambda: set_step(f"CAM3 : Outer Box {toc_no}/{toc_total} YOLO detect -> capture -> crop ROI -> OCR"))
        root.after(0, lambda: set_status(f"CAM3 scanning USB device Outer Box {toc_no}/{toc_total}"))

        toc_path = os.path.join(process_dir, now_name(f"cam3_outer{toc_no}"))

        capture_result = None
        cam3_retry = 0

        while running:
            cam3_device = find_available_usb_device()
            if cam3_device is None:
                op_action = wait_operator_camera_fix("CAM3", "No readable USB camera found. Please unplug/replug USB camera.")
                if op_action == "retry":
                    finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_usb_not_found")})
                    break
                return False

            cam3 = UsbCameraTest(device=cam3_device, name="CAM3", status_cb=set_status, preview_cb=update_preview)
            current_camera = cam3

            try:
                log_event("CAMERA_OPEN_START", camera="CAM3", device=cam3_device, retry=cam3_retry)
                cam3.open()
                log_event("CAMERA_OPEN_OK", camera="CAM3", device=cam3_device, retry=cam3_retry)

                yolo_capture_start = time.time()
                capture_result = cam3.capture_direct_or_yolo(toc_path, current_process_alive)

                if not capture_result:
                    log_event("CAM3_CAPTURE_EMPTY", level="WARNING", retry=cam3_retry)
                    cam3.close()
                    current_camera = None

                    if is_retry_action():
                        finalize_process_attempt("reset", {"reason": retry_reason()})
                        break

                    if cam3_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and auto_recover_camera("CAM3", "capture_result_empty_or_cannot_read_frame"):
                        cam3_retry += 1
                        root.after(0, lambda: set_status("CAM3 auto recovery retry..."))
                        continue

                    op_action = wait_operator_camera_fix("CAM3", "CAM3 cannot read frame / no capture result.")
                    if op_action == "retry":
                        finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_cam3_empty")})
                        break
                    return False

                log_event(
                    "CAM3_CAPTURE_RESULT",
                    duration_sec=round(time.time() - yolo_capture_start, 3),
                    detection_count=len(capture_result.get("detections", [])),
                    image_path=capture_result.get("image_path", ""),
                    retry=cam3_retry,
                )
                break

            except Exception as e:
                log_event("CAM3_ERROR", level="ERROR", error=str(e), traceback=traceback.format_exc(), retry=cam3_retry)
                try:
                    cam3.close()
                except Exception:
                    pass
                current_camera = None

                if is_retry_action():
                    finalize_process_attempt("reset", {"reason": retry_reason()})
                    break

                if cam3_retry < AUTO_CAMERA_RECOVERY_MAX_RETRY and should_auto_recover_error(str(e)):
                    if auto_recover_camera("CAM3", str(e)):
                        cam3_retry += 1
                        root.after(0, lambda: set_status("CAM3 auto recovery retry..."))
                        continue

                op_action = wait_operator_camera_fix("CAM3", f"{e}")
                if op_action == "retry":
                    finalize_process_attempt("reset", {"reason": retry_reason("operator_camera_fix_after_cam3_error")})
                    break
                return False

            finally:
                try:
                    cam3.close()
                    log_event("CAMERA_CLOSE", camera="CAM3")
                except Exception as e:
                    log_event("CAMERA_CLOSE_ERROR", level="ERROR", camera="CAM3", error=str(e))
                current_camera = None

        if is_retry_action():
            continue

        if not running:
            return False

        if not capture_result:
            return False

        image_path = capture_result["image_path"]
        add_process_image(image_path, "CAM3_TOC_FULL")
        detections = capture_result.get("detections", [])

        root.after(0, lambda: set_step(f"Outer Box {toc_no}/{toc_total} OCR : YOLO ROI count = {len(detections)}"))
        root.after(0, lambda: set_status(f"Outer Box {toc_no}/{toc_total} OCR reading from YOLO ROI"))

        toc_items = process_yolo_roi_ocr(image_path, detections)
        fallback = None
        if not toc_items:
            log_event("CAM3_NO_YOLO_ROI_OCR_FALLBACK", level="WARNING", image_path=image_path)
            root.after(0, lambda: set_status("No YOLO ROI OCR, fallback center ROI"))
            fallback = process_center_roi_ocr(image_path, f"toc_center_{toc_no}")

        clear_action()
        root.after(0, lambda: setup_capture_page(mode="outer", active=None, done=["TOA", "TOB", "TOC"]))
        root.after(0, lambda: show_toc_result(toc_items, fallback, toc_no, toc_total))

        action = wait_review_action(f"Outer Box {toc_no}/{toc_total}")
        if action == "next":
            finalize_process_attempt("final", {"reason": "operator_ok_next"})
            return True
        if action == "restart":
            finalize_process_attempt("reset", {"reason": "operator_reset_after_review"})
            continue
        return False

    return False


def wait_review_action(context="-"):
    global debug_wait_counter

    debug_wait_counter += 1
    wait_id = debug_wait_counter
    dbg(f"WAIT[{wait_id}] ENTER context={context}")
    last_log = time.time()

    while running and not action_event.is_set():
        now = time.time()
        if now - last_log >= 1.0:
            dbg(f"WAIT[{wait_id}] still waiting context={context}")
            last_log = now
        sleep(0.1)

    action = action_value or "cancel"
    dbg(f"WAIT[{wait_id}] EXIT context={context} action={action} running={running}")
    return action


def on_result_ok():
    dbg(f"OK/NEXT pressed on {current_review_context}")
    set_action("next")


def restart_process():
    # Keep current Inner Box / Outer Box index. Only redo current capture + OCR.
    dbg(f"Restart The Process pressed on {current_screen_mode}")
    set_action("restart")


def cancel_job():
    log_event("CANCEL_JOB_REQUESTED")
    dbg("Cancel Job pressed")
    # Design V2: Cancel ends entire job, but existing attempt/images/OCR must be saved.
    if current_process_attempt_dir:
        finalize_process_attempt("reset", {"reason": "operator_cancel_before_final"})
    if job_data:
        job_data["cancelled_at"] = now_iso()
        job_data.setdefault("events", []).append({
            "time": job_data["cancelled_at"],
            "event": "JOB_CANCELLED",
            "status": "CANCELLED",
            "cancel_reason": "operator_cancel",
        })
        finalize_attempt_status("CANCELLED", "operator_cancel")
    set_action("cancel")
    root.after(0, reset_to_input)


def reset_to_input():
    global running, current_index, current_outer_index, delivery_no, en_no, quantity, outer_total, inner_toa_lot_by_index, job_id, job_started_at, job_storage_dir, job_data
    global delivery_storage_dir, attempt_storage_dir, attempt_no, attempt_status, process_attempt_counter, current_process_attempt_dir, current_process_attempt_meta
    global item_no, picking_api_data, box_id_list

    running = False
    clear_action()
    current_index = 0
    current_outer_index = 0
    quantity = 0
    outer_total = 0
    inner_toa_lot_by_index = {}
    delivery_no = ""
    en_no = ""
    item_no = "-"
    picking_api_data = {}
    box_id_list = []
    job_id = ""
    job_started_at = ""
    job_storage_dir = ""
    job_data = {}
    delivery_storage_dir = ""
    attempt_storage_dir = ""
    attempt_no = 0
    attempt_status = ""
    process_attempt_counter = {}
    current_process_attempt_dir = ""
    current_process_attempt_meta = {}

    try:
        close_optimized_pi_cameras()
    except Exception as e:
        log_event("OPTIMIZED_PI_CLOSE_RESET_INPUT_ERROR", level="WARNING", error=str(e))

    try:
        en_entry.delete(0, tk.END)
        delivery_entry.delete(0, tk.END)
        qty_entry.delete(0, tk.END)
    except Exception:
        pass

    show_page(page_input)


def exit_app():
    log_event("APP_EXIT_REQUESTED")
    global running
    if running and job_data:
        if current_process_attempt_dir:
            finalize_process_attempt("reset", {"reason": "app_exit"})
        job_data.setdefault("events", []).append({
            "time": now_iso(),
            "event": "APP_EXIT_DURING_JOB",
            "status": "CANCELLED",
            "cancel_reason": "app_exit",
        })
        finalize_attempt_status("CANCELLED", "app_exit")
    running = False
    try:
        close_optimized_pi_cameras()
    except Exception as e:
        log_event("OPTIMIZED_PI_CLOSE_EXIT_ERROR", level="WARNING", error=str(e))

    try:
        if current_camera is not None:
            current_camera.close()
    except Exception:
        pass
    root.destroy()


def on_close():
    exit_app()


# =========================================================
# DAILY STORAGE CLEANUP SCHEDULER (T22)
# =========================================================
daily_cleanup_running = False


def schedule_storage_cleanup():
    """Run auto_delete_old_storage() every CLEANUP_INTERVAL_HOURS while app is open.
    Uses root.after so Tkinter UI is not blocked by a 24-hour sleep.
    The actual cleanup runs in a background thread.
    """

    def run_cleanup_worker():
        global daily_cleanup_running
        try:
            print("=" * 60)
            print("T27 DAILY STORAGE CLEANUP START", now_iso())
            deleted = auto_delete_old_storage()
            print(f"T27 DAILY STORAGE CLEANUP DONE. Deleted {len(deleted)} day folder(s).")
            print("=" * 60)
            try:
                root.after(0, lambda: set_preload_status(detail=f"Daily cleanup done. Deleted {len(deleted)} day folder(s).", cleanup="OK"))
            except Exception:
                pass
        except Exception as e:
            print("T27 DAILY STORAGE CLEANUP ERROR:", e)
            try:
                root.after(0, lambda e=e: set_preload_status(detail=f"Daily cleanup error: {e}", cleanup="ERROR"))
            except Exception:
                pass
        finally:
            daily_cleanup_running = False

    def trigger_cleanup():
        global daily_cleanup_running
        if not daily_cleanup_running:
            daily_cleanup_running = True
            threading.Thread(target=run_cleanup_worker, daemon=True, name="daily_storage_cleanup").start()
        root.after(CLEANUP_INTERVAL_MS, trigger_cleanup)

    root.after(CLEANUP_INTERVAL_MS, trigger_cleanup)


# =========================================================
# STARTUP PRELOAD (T9)
# Load slow components before operator starts a job.
# =========================================================
preload_done_event = Event()
preload_status = {
    "state": "WAIT",
    "detail": "Waiting startup preload...",
    "ocr": "WAIT",
    "yolo": "WAIT",
    "index": "WAIT",
    "cleanup": "WAIT",
}


def set_preload_status(state=None, detail=None, ocr=None, yolo=None, index=None, cleanup=None):
    if state is not None:
        preload_status["state"] = state
    if detail is not None:
        preload_status["detail"] = detail
    if ocr is not None:
        preload_status["ocr"] = ocr
    if yolo is not None:
        preload_status["yolo"] = yolo
    if index is not None:
        preload_status["index"] = index
    if cleanup is not None:
        preload_status["cleanup"] = cleanup

    text = (
        f"Startup: {preload_status['state']} | "
        f"OCR={preload_status['ocr']} | "
        f"YOLO={preload_status['yolo']} | "
        f"Index={preload_status['index']} | "
        f"Cleanup={preload_status['cleanup']}\n"
        f"{preload_status['detail']}"
    )

    def ui_update():
        try:
            preload_label.config(text=text)
        except Exception:
            pass

    try:
        root.after(0, ui_update)
    except Exception:
        pass

    print(text)


def preload_yolo_model_if_available():
    """Best-effort YOLO preload.

    The current T8 flow loads YOLO inside camera_test/UsbCameraTest.
    Different camera_test versions may expose the model in different names,
    so this function tries common preload hooks without breaking old code.
    """
    # 1) Preferred: explicit preload function in camera_test.py.
    for fn_name in [
        "preload_yolo",
        "preload_yolo_model",
        "load_yolo",
        "load_yolo_model",
        "load_model",
        "get_yolo_model",
    ]:
        fn = getattr(camera_module, fn_name, None)
        if callable(fn):
            fn()
            return f"OK via camera_test.{fn_name}()"

    # 2) Class-level preload hook if future UsbCameraTest supports it.
    for fn_name in ["preload_model", "preload_yolo", "load_model"]:
        fn = getattr(UsbCameraTest, fn_name, None)
        if callable(fn):
            fn()
            return f"OK via UsbCameraTest.{fn_name}()"

    # 3) Constructor-only warmup. This should not open the camera; it only helps
    # if a future version loads YOLO in __init__.
    try:
        _ = UsbCameraTest(device=USB_DEVICE, name="CAM3_PRELOAD", status_cb=None, preview_cb=None)
        return "WARM constructor only"
    except Exception as e:
        return f"SKIP constructor warmup: {e}"


def startup_preload_worker():
    log_event("STARTUP_PRELOAD_START")
    set_preload_status("LOADING", "Auto-delete old storage...", cleanup="RUN")
    try:
        deleted = auto_delete_old_storage()
        set_preload_status(detail=f"Auto-delete done. Deleted {len(deleted)} day folder(s). Cleanup log saved. RETENTION_DAYS={RETENTION_DAYS}", cleanup="OK")
    except Exception as e:
        set_preload_status(detail=f"Auto-delete error: {e}", cleanup="ERROR")

    set_preload_status("LOADING", "Rebuilding global search index...", index="RUN")
    try:
        rebuild_global_search_index()
        set_preload_status(detail="Search index ready.", index="OK")
    except Exception as e:
        set_preload_status(detail=f"Search index error: {e}", index="ERROR")

    set_preload_status("LOADING", "Loading OCR model...", ocr="RUN")
    try:
        load_ocr()
        set_preload_status(detail="OCR model ready.", ocr="OK")
    except Exception as e:
        set_preload_status(detail=f"OCR preload error: {e}", ocr="ERROR")

    set_preload_status("LOADING", "Preloading YOLO models: CAM1/CAM2 labels + CAM3 TOC...", yolo="RUN")
    try:
        inner_msg = load_inner_detection_models()
        yolo_msg = preload_yolo_model_if_available()
        set_preload_status(detail=f"YOLO preload: Inner={inner_msg}; CAM3={yolo_msg}", yolo="OK")
    except Exception as e:
        set_preload_status(detail=f"YOLO preload skipped/error: {e}", yolo="SKIP")

    preload_done_event.set()
    log_event("STARTUP_PRELOAD_DONE", status=dict(preload_status))
    set_preload_status("READY", "System ready. Operator can start job now.")
    try:
        root.after(0, lambda: start_btn.config(state="normal", text="START TEST"))
    except Exception:
        pass


def start_startup_preload():
    try:
        start_btn.config(state="disabled", text="LOADING MODELS...")
    except Exception:
        pass
    thread = threading.Thread(target=startup_preload_worker, daemon=True, name="startup_preload")
    thread.start()


# =========================================================
# AUDIT SEARCH UI HELPERS (T9)
# =========================================================
audit_results_cache = []


def short_audit_label(entry, idx):
    ref = entry.get("inner_ref") or entry.get("toc_ref") or "-"
    return f"{idx:03d} | {entry.get('type', '-')} | {entry.get('value', '-')} | Delivery {entry.get('delivery_no', '-')} | {ref} | {entry.get('attempt_status', '-')}"


def audit_search_now():
    global audit_results_cache
    query = audit_entry.get().strip()
    audit_listbox.delete(0, tk.END)
    audit_detail_text.delete("1.0", tk.END)

    if not query:
        audit_detail_text.insert(tk.END, "Please input Delivery No / BOX ID / LOT No / TOC value.")
        return

    # Rebuild first so search covers the latest completed/cancelled attempts.
    rebuild_global_search_index()
    results = search_global_index(query)
    audit_results_cache = results

    audit_count_label.config(text=f"Result : {len(results)} record(s)")

    if not results:
        audit_detail_text.insert(tk.END, f"No audit record found for: {query}\n")
        audit_detail_text.insert(tk.END, f"Index file: {GLOBAL_SEARCH_INDEX_PATH}")
        return

    for i, entry in enumerate(results[:500], start=1):
        audit_listbox.insert(tk.END, short_audit_label(entry, i))

    if len(results) > 500:
        audit_listbox.insert(tk.END, f"... more {len(results) - 500} records not shown")

    audit_listbox.selection_set(0)
    audit_listbox.event_generate("<<ListboxSelect>>")


def audit_on_select(event=None):
    audit_detail_text.delete("1.0", tk.END)
    sel = audit_listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    if idx >= len(audit_results_cache):
        return

    entry = audit_results_cache[idx]
    audit_detail_text.insert(tk.END, json.dumps(entry, ensure_ascii=False, indent=2))


def get_selected_audit_path():
    sel = audit_listbox.curselection()
    if not sel:
        return ""
    idx = sel[0]
    if idx >= len(audit_results_cache):
        return ""
    entry = audit_results_cache[idx]
    path = entry.get("path") or entry.get("attempt_path") or entry.get("delivery_path") or ""
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.join(STORAGE_ROOT, path.replace("/", os.sep))
    return path


def audit_open_selected_path():
    path = get_selected_audit_path()
    if not path:
        messagebox.showwarning("Audit Lookup", "Please select a record first.")
        return
    if not os.path.exists(path):
        messagebox.showwarning("Audit Lookup", f"Path not found:\n{path}")
        return

    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showinfo("Audit Lookup", f"Path:\n{path}\n\nOpen error:\n{e}")


def open_audit_page():
    audit_entry.delete(0, tk.END)
    audit_count_label.config(text="Result : 0 record(s)")
    audit_listbox.delete(0, tk.END)
    audit_detail_text.delete("1.0", tk.END)
    audit_detail_text.insert(tk.END, "Input Delivery No / BOX ID / LOT No / TOC value, then press SEARCH.")
    show_page(page_search)


# =========================================================
# UI
# =========================================================
root = tk.Tk()

def tk_report_callback_exception(exc_type, exc_value, exc_traceback):
    write_crash_log(exc_type, exc_value, exc_traceback, source="tkinter_callback")
    try:
        messagebox.showerror("Program Error", f"{exc_type.__name__}: {exc_value}")
    except Exception:
        pass

root.report_callback_exception = tk_report_callback_exception
root.title("AI Camera Station - Mock API - T36 Fixed Field Map Result Layout")
root.geometry("1200x720")
root.resizable(False, False)

# ===================== THEME =====================
BG = "#eef3f7"
CARD = "#ffffff"
TEXT = "#17202a"
MUTED = "#6c7a89"
BLUE = "#0b5cab"
BLUE_DARK = "#084b8a"
RED = "#c0392b"
RED_DARK = "#922b21"
BORDER = "#dfe6e9"
SOFT = "#f7f9fa"
GREEN = "#229954"
GREEN_BG = "#eafaf1"
YELLOW_TEXT = "#b9770e"
YELLOW_BG = "#fff3cd"
DARK = "#17202a"

root.configure(bg=BG)

# ===================== ICONS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets", "icons")
icons = {}


def load_icon_file(filename, size=(28, 28)):
    path = os.path.join(ICON_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        try:
            img = img.resize(size, Image.Resampling.LANCZOS)
        except Exception:
            img = img.resize(size)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print("Icon load error:", path, e)
        return None


def load_icons():
    icons["lot"] = load_icon_file("assignment.png")
    icons["delivery"] = load_icon_file("description.png")
    icons["barcode"] = load_icon_file("label.png")
    icons["progress"] = load_icon_file("analytics.png")
    icons["pack"] = load_icon_file("package.png")
    icons["camera"] = load_icon_file("photo.png")
    icons["status"] = load_icon_file("dashboard.png")
    icons["check"] = load_icon_file("checklist.png")
    icons["setting"] = load_icon_file("settings.png")
    icons["info"] = load_icon_file("info.png")
    icons["cancel"] = load_icon_file("cancel.png")
    icons["stop"] = load_icon_file("stop.png")
    icons["refresh"] = load_icon_file("refresh.png")


def icon_label(parent, key, bg=CARD):
    icon = icons.get(key)
    if icon is not None:
        return tk.Label(parent, image=icon, bg=bg)
    return tk.Label(parent, text=key.upper()[:4], bg=SOFT, fg=BLUE, font=("Arial", 8, "bold"), width=5, height=2)


def title_with_icon(parent, icon_key, text, bg=CARD):
    frame = tk.Frame(parent, bg=bg)
    frame.pack(fill="x")
    icon_label(frame, icon_key, bg=bg).pack(side="left", padx=(0, 10))
    tk.Label(frame, text=text, bg=bg, fg=TEXT, font=("Arial", 22, "bold")).pack(side="left")
    return frame


def flat_button(parent, text, bg, fg="white", command=None, icon_key=None, width=None):
    kwargs = {
        "text": text,
        "bg": bg,
        "fg": fg,
        "activebackground": RED_DARK if bg == RED else BLUE_DARK,
        "activeforeground": "white",
        "font": ("Arial", 10, "bold"),
        "relief": "flat",
        "command": command,
        "cursor": "hand2",
        "padx": 10,
        "pady": 5,
    }
    if width is not None:
        kwargs["width"] = width
    icon = icons.get(icon_key) if icon_key else None
    if icon is not None:
        kwargs["image"] = icon
        kwargs["compound"] = "left"
    return tk.Button(parent, **kwargs)


def add_action_button(parent, text, bg, fg="white", command=None, width_px=190, height_px=44, frame_bg=None):
    """Create same-size action buttons so Restart / Cancel / OK are aligned."""
    if frame_bg is None:
        frame_bg = parent.cget("bg")
    wrap = tk.Frame(parent, bg=frame_bg, width=width_px, height=height_px)
    wrap.pack(side="left", padx=(0, 8))
    wrap.pack_propagate(False)

    active_bg = RED_DARK if bg == RED else ("#1e8449" if bg == "#27ae60" else "#aeb4b8")
    btn = tk.Button(
        wrap,
        text=text,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground="white" if fg == "white" else fg,
        font=("Arial", 14, "bold"),
        relief="flat",
        cursor="hand2",
        command=command,
        padx=0,
        pady=0,
    )
    btn.pack(fill="both", expand=True)
    return btn


load_icons()

top = tk.Frame(root, bg=BG, height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

# Placeholder labels
info_delivery = tk.Label(root)
info_item = tk.Label(root)
info_en = tk.Label(root)
info_box_title = tk.Label(root)
info_box = tk.Label(root)
info_pack = tk.Label(root)
progress_title_label = tk.Label(root)
progress_label = tk.Label(root)
progress_percent_label = tk.Label(root)
status_label = tk.Label(root)
step_label = tk.Label(root)

# =========================================================
# PAGE 1: INPUT
# =========================================================
page_input = tk.Frame(root, bg=BG)

input_card = tk.Frame(page_input, bg=CARD, padx=70, pady=38, highlightbackground=BORDER, highlightthickness=1)
input_card.place(relx=0.5, rely=0.50, anchor="center", width=760, height=600)

tk.Label(input_card, text="Start Test Job", font=("Arial", 30, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 16))


def make_input(parent, title):
    tk.Label(parent, text=title, bg=CARD, fg=TEXT, font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 5))
    ent = tk.Entry(parent, font=("Arial", 24), relief="solid", borderwidth=1)
    ent.pack(fill="x", ipady=8, pady=(0, 24))
    return ent


en_entry = make_input(input_card, "EN")
delivery_entry = make_input(input_card, "Delivary NO")
qty_entry = make_input(input_card, "Quantity (Auto from API)")

en_entry.bind("<Return>", on_en_enter)
delivery_entry.bind("<Return>", on_delivery_enter)
qty_entry.bind("<Return>", lambda e: start_job())

tk.Label(
    input_card,
    text="Scanner/Keyboard: EN → Enter → Delivery No → Enter → START. Quantity is loaded from Mock API.",
    bg=CARD,
    fg=MUTED,
    font=("Arial", 10, "bold"),
    anchor="w",
).pack(fill="x", pady=(0, 8))

preload_label = tk.Label(
    input_card,
    text="Startup: WAIT | OCR=WAIT | YOLO=WAIT | Index=WAIT | Cleanup=WAIT",
    bg=SOFT,
    fg=TEXT,
    font=("Arial", 10, "bold"),
    justify="left",
    anchor="w",
    padx=10,
    pady=8,
)
preload_label.pack(fill="x", pady=(0, 10))

start_btn = tk.Button(input_card, text="START TEST", font=("Arial", 21, "bold"), bg=BLUE, fg="white", activebackground=BLUE_DARK, activeforeground="white", relief="flat", command=start_job)
start_btn.pack(fill="x", ipady=10, pady=(4, 8))
tk.Button(input_card, text="SEARCH / AUDIT LOOKUP", font=("Arial", 18, "bold"), bg="#34495e", fg="white", activebackground="#2c3e50", activeforeground="white", relief="flat", command=open_audit_page).pack(fill="x", ipady=8, pady=(0, 8))
tk.Button(input_card, text="EXIT APP", font=("Arial", 21, "bold"), bg=RED, fg="white", activebackground=RED_DARK, activeforeground="white", relief="flat", command=exit_app).pack(fill="x", ipady=9)

# =========================================================
# PAGE SEARCH: AUDIT LOOKUP
# =========================================================
page_search = tk.Frame(root, bg=BG)

search_header = tk.Frame(page_search, bg=BG)
search_header.pack(fill="x", padx=16, pady=(14, 8))

tk.Label(search_header, text="Search / Audit Lookup", font=("Arial", 28, "bold"), bg=BG, fg=TEXT).pack(side="left")
tk.Button(search_header, text="BACK", font=("Arial", 14, "bold"), bg="#bfc3c7", fg=TEXT, relief="flat", command=lambda: show_page(page_input), padx=20, pady=8).pack(side="right")

search_card = tk.Frame(page_search, bg=CARD, padx=18, pady=14, highlightbackground=BORDER, highlightthickness=1)
search_card.pack(fill="x", padx=16, pady=(0, 10))

tk.Label(search_card, text="Search key (Delivery No / BOX ID / TOA LOT / TOB LOT / TOC)", bg=CARD, fg=TEXT, font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 6))
search_row = tk.Frame(search_card, bg=CARD)
search_row.pack(fill="x")
audit_entry = tk.Entry(search_row, font=("Arial", 22), relief="solid", borderwidth=1)
audit_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 10))
tk.Button(search_row, text="SEARCH", font=("Arial", 16, "bold"), bg=BLUE, fg="white", activebackground=BLUE_DARK, activeforeground="white", relief="flat", command=audit_search_now, padx=22, pady=7).pack(side="left", padx=(0, 8))
tk.Button(search_row, text="OPEN PATH", font=("Arial", 16, "bold"), bg="#27ae60", fg="white", activebackground="#1e8449", activeforeground="white", relief="flat", command=audit_open_selected_path, padx=18, pady=7).pack(side="left")

audit_entry.bind("<Return>", lambda e: audit_search_now())
audit_count_label = tk.Label(search_card, text="Result : 0 record(s)", bg=CARD, fg=MUTED, font=("Arial", 12, "bold"))
audit_count_label.pack(anchor="w", pady=(8, 0))

search_body = tk.Frame(page_search, bg=BG)
search_body.pack(fill="both", expand=True, padx=16, pady=(0, 14))

list_frame = tk.Frame(search_body, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=8)
list_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
tk.Label(list_frame, text="Matched Records", bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w", pady=(0, 6))
list_scroll = tk.Scrollbar(list_frame)
list_scroll.pack(side="right", fill="y")
audit_listbox = tk.Listbox(list_frame, font=("Consolas", 11), yscrollcommand=list_scroll.set)
audit_listbox.pack(fill="both", expand=True)
list_scroll.config(command=audit_listbox.yview)
audit_listbox.bind("<<ListboxSelect>>", audit_on_select)

detail_frame = tk.Frame(search_body, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=8)
detail_frame.pack(side="right", fill="both", expand=True)
tk.Label(detail_frame, text="Audit Detail / Evidence Path", bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w", pady=(0, 6))
detail_scroll = tk.Scrollbar(detail_frame)
detail_scroll.pack(side="right", fill="y")
audit_detail_text = tk.Text(detail_frame, font=("Consolas", 10), wrap="word", yscrollcommand=detail_scroll.set)
audit_detail_text.pack(fill="both", expand=True)
detail_scroll.config(command=audit_detail_text.yview)

# =========================================================
# PAGE 2/4: CAPTURE
# =========================================================
page_capture = tk.Frame(root, bg=BG)
capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=14, pady=14)

# Section 1 header
header_frame = tk.Frame(capture_container, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=30, pady=36)
header_frame.pack(fill="x", pady=(0, 12))

header_top = tk.Frame(header_frame, bg=CARD)
header_top.pack(fill="x")

info_left = tk.Frame(header_top, bg=CARD)
info_left.pack(side="left", fill="x", expand=True)

header_buttons = tk.Frame(header_top, bg=CARD)
header_buttons.pack(side="right")

add_action_button(header_buttons, "Reset Camera", bg="#dfe6e9", fg=TEXT, command=reset_current_camera, width_px=170, height_px=58)
add_action_button(header_buttons, "Restart The Process", bg="#bfc3c7", fg=TEXT, command=restart_process, width_px=190, height_px=58)
add_action_button(header_buttons, "Cancel Job", bg=RED, fg="white", command=cancel_job, width_px=170, height_px=58)
add_action_button(header_buttons, "OK / NEXT", bg="#27ae60", fg="white", command=on_result_ok, width_px=170, height_px=58)

info_row1 = tk.Frame(info_left, bg=CARD)
info_row1.pack(fill="x", pady=(0, 28))
info_row2 = tk.Frame(info_left, bg=CARD)
info_row2.pack(fill="x")


def header_info_inline(parent, icon_key, title, value_width=18):
    """Header row 1: icon + title and value on the same line."""
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 74))

    icon_label(box, icon_key, bg=CARD).pack(side="left", padx=(0, 8))

    tk.Label(
        box,
        text=f"{title} :",
        bg=CARD,
        fg=MUTED,
        font=("Arial", 22, "bold"),
        anchor="w"
    ).pack(side="left")

    val = tk.Label(
        box,
        text="-",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 25, "bold"),
        width=value_width,
        anchor="w"
    )
    val.pack(side="left", padx=(10, 0))
    return val


def header_info_box(parent, icon_key, title, value_width=16):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 70))

    icon_label(box, icon_key, bg=CARD).pack(side="left", padx=(0, 8), anchor="n")

    text_box = tk.Frame(box, bg=CARD)
    text_box.pack(side="left")

    tk.Label(
        text_box,
        text=title,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 21, "bold"),
        anchor="w"
    ).pack(anchor="w")

    val = tk.Label(
        text_box,
        text="-",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 25, "bold"),
        width=value_width,
        anchor="w"
    )
    val.pack(anchor="w", pady=(6, 0))
    return val


info_delivery = header_info_inline(info_row1, "delivery", "Delivary No", 18)
info_item = header_info_inline(info_row1, "barcode", "Item No", 10)
info_en = header_info_box(info_row2, "barcode", "EN", 10)

inner_box_container = tk.Frame(info_row2, bg=CARD)
inner_box_container.pack(side="left", padx=(0, 70))
icon_label(inner_box_container, "progress", bg=CARD).pack(side="left", padx=(0, 8), anchor="n")
info_box_text = tk.Frame(inner_box_container, bg=CARD)
info_box_text.pack(side="left")
info_box_title = tk.Label(info_box_text, text="Inner Box", bg=CARD, fg=MUTED, font=("Arial", 21, "bold"), anchor="w")
info_box_title.pack(anchor="w")
info_box = tk.Label(info_box_text, text="-", bg=CARD, fg=TEXT, font=("Arial", 25, "bold"), width=10, anchor="w")
info_box.pack(anchor="w", pady=(6, 0))

info_pack = header_info_box(info_row2, "pack", "Pack Type", 10)

# Body
body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

# Section 2 capture left
capture_left = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=14, width=800)
capture_left.pack(side="left", fill="both", expand=False, padx=(0, 10))
capture_left.pack_propagate(False)

title_with_icon(capture_left, "camera", "Capture Images", bg=CARD)
tk.Label(capture_left, text="Preview", bg=CARD, fg=MUTED, font=("Arial", 14, "bold")).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(capture_left, bg=DARK, width=760, height=430, highlightbackground=DARK, highlightthickness=2)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)

preview_label = tk.Label(preview_frame, text="Camera Preview", bg=DARK, fg="white", font=("Arial", 20))
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)

capture_state_dot = tk.Canvas(preview_overlay, width=14, height=14, bg=DARK, highlightthickness=0)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill="#f1c40f", outline="#f1c40f")

capture_state_label = tk.Label(preview_overlay, text="Waiting Object", bg=DARK, fg="white", font=("Arial", 14, "bold"), padx=2, pady=2)
capture_state_label.pack(side="left")

reset_btn = tk.Button(capture_left, text="Reset Current Camera", bg="#dfe6e9", fg=TEXT, activebackground="#cfd8dc", activeforeground=TEXT, font=("Arial", 13, "bold"), relief="flat", height=2, cursor="hand2", command=reset_current_camera)
if icons.get("refresh") is not None:
    reset_btn.config(image=icons["refresh"], compound="left")
reset_btn.pack(fill="x", pady=(8, 0))

# Section 3 right status
capture_right = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=22, pady=18)
capture_right.pack(side="right", fill="both", expand=True)

title_with_icon(capture_right, "status", "Status", bg=CARD)

product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))
icon_label(product_header, "pack", bg=CARD).pack(side="left", padx=(0, 8))
progress_title_label = tk.Label(product_header, text="Inner Box", bg=CARD, fg=TEXT, font=("Arial", 18, "bold"))
progress_title_label.pack(side="left")

product_row = tk.Frame(capture_right, bg=CARD)
product_row.pack(fill="x")
progress_label = tk.Label(product_row, text="0 / 0", bg=CARD, fg=TEXT, font=("Arial", 19, "bold"))
progress_label.pack(side="left")
progress_percent_label = tk.Label(product_row, text="0%", bg=CARD, fg=TEXT, font=("Arial", 17, "bold"))
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(capture_right, bg="#c4c4c8", height=26)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)
progress_bar_fill = tk.Frame(progress_bar_bg, bg=BLUE)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

check_header = tk.Frame(capture_right, bg=CARD)
check_header.pack(fill="x", pady=(4, 8))
icon_label(check_header, "check", bg=CARD).pack(side="left", padx=(0, 8))
tk.Label(check_header, text="Checklist", bg=CARD, fg=TEXT, font=("Arial", 18, "bold")).pack(side="left")

checklist_frame = tk.Frame(capture_right, bg=CARD)
checklist_frame.pack(fill="x")


def make_check_item(name):
    item = tk.Label(checklist_frame, text=f"[ ]   {name}", bg=SOFT, fg=TEXT, font=("Arial", 17, "bold"), anchor="w", padx=18, pady=16, relief="flat")
    item.pack(fill="x", pady=5)
    check_labels[name] = item


make_check_item("TOA")
make_check_item("TOB")
make_check_item("TOC")

step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="x", pady=(16, 10))

step_header = tk.Frame(step_status_box, bg=SOFT)
step_header.pack(fill="x")
icon_label(step_header, "setting", bg=SOFT).pack(side="left", padx=(0, 8))
tk.Label(step_header, text="Step", bg=SOFT, fg="#34495e", font=("Arial", 14, "bold")).pack(side="left")

step_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 14), wraplength=360, justify="left")
step_label.pack(anchor="w", pady=(2, 8))

status_detail_header = tk.Frame(step_status_box, bg=SOFT)
status_detail_header.pack(fill="x")
icon_label(status_detail_header, "info", bg=SOFT).pack(side="left", padx=(0, 8))
tk.Label(status_detail_header, text="Status", bg=SOFT, fg="#34495e", font=("Arial", 14, "bold")).pack(side="left")

status_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 14), wraplength=360, justify="left")
status_label.pack(anchor="w", pady=(2, 0))

# =========================================================
# PAGE 3/5: RESULT
# =========================================================
page_result = tk.Frame(root, bg=BG)

result_header = tk.Frame(page_result, bg=BG)
result_header.pack(fill="x", padx=16, pady=(12, 8))

result_left_header = tk.Frame(result_header, bg=BG)
result_left_header.pack(side="left")

result_title = tk.Label(result_left_header, text="OCR Result", font=("Arial", 28, "bold"), bg=BG, fg=TEXT)
result_title.pack(side="left")

result_status_badge = tk.Label(result_left_header, text="WAIT REVIEW", bg="#f1c40f", fg=TEXT, font=("Arial", 14, "bold"), padx=16, pady=7)
result_status_badge.pack(side="left", padx=14)

result_button_frame = tk.Frame(result_header, bg=BG)
result_button_frame.pack(side="right")

cancel_result_btn = add_action_button(result_button_frame, "Cancel Job", bg=RED, fg="white", command=cancel_job, width_px=205, height_px=58, frame_bg=BG)
restart_result_btn = add_action_button(result_button_frame, "Restart The Process", bg="#bfc3c7", fg=TEXT, command=restart_process, width_px=205, height_px=58, frame_bg=BG)
ok_next_btn = add_action_button(result_button_frame, "OK / NEXT", bg="#27ae60", fg="white", command=on_result_ok, width_px=205, height_px=58, frame_bg=BG)

result_hint = tk.Label(page_result, text="Review ROI image and OCR result, then press OK / NEXT to continue.", bg=BG, fg=MUTED, font=("Arial", 14))
result_hint.pack(anchor="w", padx=18, pady=(0, 10))

product_result_area = tk.Frame(page_result, bg=BG)
product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))

make_product_result_section(product_result_area, key="vendor", title="RESULT : Vendor Label", icon_key="delivery", field1="BOX ID", value1="BOX ID :", field2="Date Code", value2="Date Code :")
make_product_result_section(product_result_area, key="toa", title="RESULT : TOA Label", icon_key="barcode", field1="Lot No", value1="Lot No :", field2="Date Code", value2="Date Code :")
make_product_result_section(product_result_area, key="tob", title="RESULT : TOB Label", icon_key="pack", field1="Lot No", value1="Lot No :", field2="Date Code", value2="Date Code :")

toc_result_area = tk.Frame(page_result, bg=BG)

# =========================================================
# READY TOC
# =========================================================
page_ready_toc = tk.Frame(root, bg=BG)

toc_card = tk.Frame(page_ready_toc, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
toc_card.place(relx=0.5, rely=0.45, anchor="center", width=560, height=300)

tk.Label(toc_card, text="Inner Box Test Completed", font=("Arial", 21, "bold"), bg=CARD, fg=GREEN).pack(pady=15)
tk.Label(toc_card, text="Press continue to test CAM3 Outer Box / TOC YOLO ROI OCR", font=("Arial", 13), bg=CARD, fg="#34495e", wraplength=460).pack(pady=8)
tk.Button(toc_card, text="CONTINUE TO OUTER BOX", bg=BLUE, fg="white", font=("Arial", 14, "bold"), width=24, relief="flat", command=continue_toc).pack(pady=20)
tk.Button(toc_card, text="CANCEL JOB", bg=RED, fg="white", font=("Arial", 12, "bold"), width=24, relief="flat", command=cancel_job).pack(pady=(0, 8))

# =========================================================
# COMPLETE
# =========================================================
page_complete = tk.Frame(root, bg=BG)

complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=520, height=260)

tk.Label(complete_card, text="TEST COMPLETED", font=("Arial", 24, "bold"), bg=CARD, fg=GREEN).pack(pady=25)
tk.Button(complete_card, text="OK", bg=BLUE, fg="white", font=("Arial", 14, "bold"), width=16, relief="flat", command=reset_to_input).pack(pady=10)

# Debug page names
page_input._debug_name = "page_input"
page_search._debug_name = "page_search"
page_capture._debug_name = "page_capture"
page_result._debug_name = "page_result"
page_ready_toc._debug_name = "page_ready_toc"
page_complete._debug_name = "page_complete"

root.protocol("WM_DELETE_WINDOW", on_close)
show_page(page_input)
log_event("UI_READY")
start_startup_preload()
schedule_storage_cleanup()
log_event("TK_MAINLOOP_START")
root.mainloop()
log_event("TK_MAINLOOP_END")
