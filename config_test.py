import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CAPTURE_DIR = os.path.join(OUTPUT_DIR, "captures")
ROI_DIR = os.path.join(OUTPUT_DIR, "roi")
OCR_DIR = os.path.join(OUTPUT_DIR, "ocr")

# =====================
# CAMERA
# =====================
PICAM1_INDEX = 0
PICAM2_INDEX = 1
USB_DEVICE = 17

PICAM_PREVIEW_SIZE = (640, 480)
PICAM_CAPTURE_SIZE = (1280, 720)

USB_WIDTH = 1280
USB_HEIGHT = 720
USB_FPS = 15

PREVIEW_SIZE = (640, 360)
PREVIEW_INTERVAL = 0.3

# =====================
# PI CAMERA DETECT
# =====================
MOTION_THRESHOLD = 50000
STABLE_THRESHOLD = 70000
STABLE_TIME = 2.0
FOCUS_DELAY = 1.5

# =====================
# ROI TEST FOR PICAM 1/2
# =====================
CENTER_ROI_W = 320
CENTER_ROI_H = 200

# OCR scale
OCR_SCALE = 2

# =====================
# OCR
# =====================
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_."

# =====================
# TOC YOLO MODEL
# ตอนนี้คุณวาง TOC_V2.pt ไว้ใน /home/toto/AI_CAMERA_TEST
# ถ้า path ไม่ตรงให้แก้ตรงนี้
# =====================
TOC_MODEL_PATH = os.path.join(BASE_DIR, "TOC_V2.pt")

YOLO_CONF = 0.40
YOLO_IMGSZ = 640
USB_HOLD_TIME = 4.0

# class ที่คาดว่าจะเจอใน TOC model
EXPECTED_TOC_CLASSES = [
    "toc",
    "toc1",
    "toc2",
    "toc3",
    "toc4",
    "toc5",
    "toc6"
]

TOC_ORDER = {
    "toc": 0,
    "toc1": 1,
    "toc2": 2,
    "toc3": 3,
    "toc4": 4,
    "toc5": 5,
    "toc6": 6
}
