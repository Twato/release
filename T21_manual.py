import os
import math
import time
import threading
import subprocess
import queue
from datetime import datetime

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from config_test import (
    PICAM1_INDEX,
    PICAM2_INDEX,
    USB_DEVICE,
    PREVIEW_SIZE,
    PREVIEW_INTERVAL,
    FOCUS_DELAY,
)

# เนเธเนเธเธญเธเน€เธ”เธดเธก เน€เธเธทเนเธญเนเธซเน CAM3 YOLO เธ—เธณเธเธฒเธเน€เธซเธกเธทเธญเธเนเธเธฃเนเธเธฃเธก T8
import camera_test as camera_test_module
from camera_test import UsbCameraTest
from roi_test import crop_by_box

try:
    from picamera2 import Picamera2
    from libcamera import controls
except Exception:
    Picamera2 = None
    controls = None

# =========================================================
# CONFIG
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_SAVE_ROOT = os.path.join(APP_DIR, "manual_capture")
os.makedirs(MANUAL_SAVE_ROOT, exist_ok=True)

FOCUS_DELAY_SEC = FOCUS_DELAY
# Split autofocus delay for Pi cameras.
# CAM1 reads Vendor + TOA, so it gets a little more settle time.
# CAM2 reads TOB only, so it can be faster.
PICAM1_FOCUS_DELAY_SEC = 1.5
PICAM2_FOCUS_DELAY_SEC = 1.0
USB_SCAN_DEVICES = [USB_DEVICE, 0, 1, 2, 3, 4, 5]

# USB CAM3 resolution stays the same as T8 (used by OpenCV / YOLO capture)
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# CAM3 YOLO preload.
# Keep USB scan/reset behavior safe: this only preloads the model, not the USB device.
YOLO_PRELOAD_ENABLED = True
CAM3_YOLO_MODEL_PATH = os.path.join(APP_DIR, "TOC_V2.pt")

# T15 experimental no-switch mode for Pi CAM1/CAM2.
# Run the sensor/main stream at 2K all the time, then resize the preview in the UI.
# This removes Preview 720p -> Still 2K -> Preview 720p mode switching delay.
PI_PREVIEW_WIDTH = 2304
PI_PREVIEW_HEIGHT = 1296

# Pi CAM1/CAM2 real saved capture resolution: 2K.
# Same as preview/main stream in T15, so capture does not need switch_mode.
PI_CAPTURE_WIDTH = 2304
PI_CAPTURE_HEIGHT = 1296

# Performance tuning
SAVE_JPEG_QUALITY = 95
PREVIEW_STOP_WAIT_SEC = 0.05
CAM_SWITCH_WAIT_SEC = 0.05
CAM3_HANDOFF_WAIT_SEC = 0.20
COMPLETE_OVERLAY_MS = 3000
COMPLETE_OVERLAY_WAIT_SEC = 3.00

# Experimental optimization: keep Pi cameras open during one job.
# This reduces CAM1 -> CAM2 switching time, but uses more RAM/CSI resources.
PERSISTENT_PI_CAMERAS = True
PREWARM_CAM2_ON_START = True

PREVIEW_W, PREVIEW_H = PREVIEW_SIZE

# =========================================================
# STATE
# =========================================================
running = False
quantity = 0
outer_total = 0
inner_index = 1
outer_index = 1
current_stage = "input"   # input, cam1, cam2, cam3, complete
current_camera = None
current_job_dir = ""
last_preview_time = 0

capture_lock = threading.Lock()
preview_stop = threading.Event()
auto_cam3_capture_on_ready = False
selected_usb_device = None  # kept for compatibility, but CAM3 is rescanned for reset safety.

# Preloaded YOLO model for CAM3.
yolo_preload_lock = threading.Lock()
yolo_preload_done = threading.Event()
yolo_preload_error = None
preloaded_yolo_model = None
_original_camera_test_yolo = getattr(camera_test_module, "YOLO", None)
_original_ultralytics_yolo = None

# Persistent camera handles for speed test version.
persistent_cam1 = None
persistent_cam2 = None
persistent_lock = threading.Lock()

# Background image saving queue. Capture flow does not need to wait for JPG disk write.
save_queue = queue.Queue()


# =========================================================
# COLORS / UI THEME เนเธเธฅเนเน€เธเธตเธขเธ T8 เน€เธ”เธดเธก
# =========================================================
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
GREEN_DARK = "#1e8449"
GREEN_BG = "#eafaf1"
YELLOW = "#f1c40f"
DARK = "#17202a"
GRAY = "#bfc3c7"

# =========================================================
# HELPERS
# =========================================================
def now_name(prefix, ext="jpg"):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.{ext}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save_image_worker():
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        path, frame_rgb, quality = item
        try:
            # Convert and write in background so the next camera stage can start earlier.
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            print(f"Background saved: {path} at {frame_rgb.shape[1]}x{frame_rgb.shape[0]} Q={quality}")
        except Exception as e:
            print(f"Background save error for {path}: {e}")
        finally:
            save_queue.task_done()


save_worker_thread = threading.Thread(target=_save_image_worker, daemon=True)
save_worker_thread.start()


def queue_save_rgb(path, frame_rgb, quality=SAVE_JPEG_QUALITY):
    # Copy the frame before returning because Picamera2/OpenCV buffers may be reused.
    save_queue.put((path, frame_rgb.copy(), quality))


def wait_for_pending_saves(timeout_sec=3.0):
    end = time.time() + float(timeout_sec)
    while time.time() < end:
        if save_queue.unfinished_tasks == 0:
            return True
        time.sleep(0.05)
    print(f"Warning: pending background saves = {save_queue.unfinished_tasks}")
    return False


def _same_model_path(a, b):
    try:
        aa = os.path.abspath(str(a))
        bb = os.path.abspath(str(b))
        return aa == bb or os.path.basename(aa) == os.path.basename(bb)
    except Exception:
        return False


def _cached_camera_test_yolo(model_path, *args, **kwargs):
    """Return the preloaded YOLO model when camera_test asks for the TOC model."""
    global preloaded_yolo_model
    try:
        if preloaded_yolo_model is not None and _same_model_path(model_path, CAM3_YOLO_MODEL_PATH):
            print(f"CAM3: using preloaded YOLO model: {model_path}")
            return preloaded_yolo_model
    except Exception as e:
        print("YOLO cache path check error:", e)

    if _original_camera_test_yolo is not None and _original_camera_test_yolo is not _cached_camera_test_yolo:
        return _original_camera_test_yolo(model_path, *args, **kwargs)

    if _original_ultralytics_yolo is not None:
        return _original_ultralytics_yolo(model_path, *args, **kwargs)

    from ultralytics import YOLO
    return YOLO(model_path, *args, **kwargs)


def patch_yolo_loaders():
    """Patch both camera_test.YOLO and ultralytics.YOLO so CAM3 reuses the preloaded model.
    This covers camera_test implementations that import YOLO at module level or inside open().
    """
    global _original_ultralytics_yolo
    try:
        import ultralytics
        if _original_ultralytics_yolo is None:
            _original_ultralytics_yolo = getattr(ultralytics, "YOLO", None)
        ultralytics.YOLO = _cached_camera_test_yolo
    except Exception as e:
        print("ultralytics YOLO patch error:", e)

    try:
        camera_test_module.YOLO = _cached_camera_test_yolo
    except Exception as e:
        print("camera_test YOLO patch error:", e)

def preload_yolo_model_background():
    """Load CAM3 YOLO once at program startup, before the workflow reaches CAM3."""
    global preloaded_yolo_model, yolo_preload_error
    if not YOLO_PRELOAD_ENABLED:
        yolo_preload_done.set()
        return

    with yolo_preload_lock:
        if preloaded_yolo_model is not None:
            yolo_preload_done.set()
            return
        try:
            print(f"CAM3 YOLO preload starting: {CAM3_YOLO_MODEL_PATH}")
            try:
                root.after(0, lambda: set_status("Loading CAM3 YOLO model in background...", BLUE))
            except Exception:
                pass

            # Load once using the real YOLO constructor, then patch all later YOLO loads.
            real_yolo = _original_camera_test_yolo
            if real_yolo is None:
                import ultralytics
                global _original_ultralytics_yolo
                if _original_ultralytics_yolo is None:
                    _original_ultralytics_yolo = getattr(ultralytics, "YOLO", None)
                real_yolo = _original_ultralytics_yolo
            preloaded_yolo_model = real_yolo(CAM3_YOLO_MODEL_PATH)

            # Patch camera_test.YOLO and ultralytics.YOLO so UsbCameraTest.open() reuses this model.
            patch_yolo_loaders()

            print(f"CAM3 YOLO preloaded: {CAM3_YOLO_MODEL_PATH}")
            try:
                root.after(0, lambda: set_status("CAM3 YOLO model ready", GREEN))
            except Exception:
                pass
        except Exception as e:
            yolo_preload_error = e
            print("CAM3 YOLO preload error:", e)
            try:
                root.after(0, lambda e=e: set_status(f"CAM3 YOLO preload failed: {e}", RED))
            except Exception:
                pass
        finally:
            yolo_preload_done.set()


def start_yolo_preload_thread():
    threading.Thread(target=preload_yolo_model_background, daemon=True).start()


def attach_preloaded_yolo_to_cam3(cam3):
    """Best-effort injection for UsbCameraTest implementations that keep model on self."""
    if preloaded_yolo_model is None:
        return
    for attr in ("model", "yolo_model", "det_model", "toc_model"):
        try:
            if not hasattr(cam3, attr) or getattr(cam3, attr) is None:
                setattr(cam3, attr, preloaded_yolo_model)
        except Exception:
            pass


def safe_close_camera():
    global current_camera
    try:
        if current_camera is not None:
            if hasattr(current_camera, "close"):
                current_camera.close()
            elif hasattr(current_camera, "stop"):
                current_camera.stop()
    except Exception as e:
        print("Camera close error:", e)
    current_camera = None


def is_persistent_pi_camera(cam):
    return PERSISTENT_PI_CAMERAS and cam is not None and (cam is persistent_cam1 or cam is persistent_cam2)


def release_current_camera_for_stage_switch():
    """Close only non-persistent cameras. Persistent Pi cameras stay open for the whole job."""
    global current_camera
    try:
        if current_camera is not None and not is_persistent_pi_camera(current_camera):
            if hasattr(current_camera, "close"):
                current_camera.close()
            elif hasattr(current_camera, "stop"):
                current_camera.stop()
    except Exception as e:
        print("Camera release error:", e)
    current_camera = None


def close_persistent_pi_cameras():
    global persistent_cam1, persistent_cam2, current_camera
    with persistent_lock:
        for cam in (persistent_cam1, persistent_cam2):
            try:
                if cam is not None:
                    cam.close()
            except Exception as e:
                print("Persistent camera close error:", e)
        persistent_cam1 = None
        persistent_cam2 = None
        if current_camera is not None and not hasattr(current_camera, "cap"):
            current_camera = None


def get_or_open_persistent_picam(cam_no):
    """cam_no: 1 for CAM1, 2 for CAM2. Opens once and reuses during the job."""
    global persistent_cam1, persistent_cam2
    if not PERSISTENT_PI_CAMERAS:
        if cam_no == 1:
            cam = ManualPiCamera(PICAM1_INDEX, "CAM1", PICAM1_FOCUS_DELAY_SEC)
        else:
            cam = ManualPiCamera(PICAM2_INDEX, "CAM2", PICAM2_FOCUS_DELAY_SEC)
        cam.open()
        return cam

    with persistent_lock:
        if cam_no == 1:
            if persistent_cam1 is None or persistent_cam1.picam2 is None:
                persistent_cam1 = ManualPiCamera(PICAM1_INDEX, "CAM1", PICAM1_FOCUS_DELAY_SEC)
                persistent_cam1.open()
            return persistent_cam1
        else:
            if persistent_cam2 is None or persistent_cam2.picam2 is None:
                persistent_cam2 = ManualPiCamera(PICAM2_INDEX, "CAM2", PICAM2_FOCUS_DELAY_SEC)
                persistent_cam2.open()
            return persistent_cam2


def prewarm_cam2_background():
    if not running or not PREWARM_CAM2_ON_START or not PERSISTENT_PI_CAMERAS:
        return

    def worker():
        try:
            print("CAM2 prewarm opening in background...")
            get_or_open_persistent_picam(2)
            root.after(0, lambda: set_status("CAM2 prewarmed / standby ready", GREEN))
        except Exception as e:
            print("CAM2 prewarm error:", e)
            root.after(0, lambda e=e: set_status(f"CAM2 prewarm failed: {e}", YELLOW))

    threading.Thread(target=worker, daemon=True).start()


def stop_preview_loop():
    preview_stop.set()
    time.sleep(PREVIEW_STOP_WAIT_SEC)


def set_status(text, color=None):
    print(text)
    try:
        status_label.config(text=text)
        capture_state_label.config(text=text)
        if color is None:
            lower = text.lower()
            if "ready" in lower:
                color = GREEN
            elif "error" in lower or "not found" in lower or "fail" in lower:
                color = RED
            elif "capture" in lower or "focus" in lower or "opening" in lower:
                color = BLUE
            else:
                color = YELLOW
        capture_state_dot.delete("all")
        capture_state_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)
        root.update_idletasks()
    except Exception:
        pass


def set_step(text):
    print("STEP:", text)
    try:
        step_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass


def resize_to_fit(frame, max_width, max_height):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    max_width = max(int(max_width), 20)
    max_height = max(int(max_height), 20)
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h))


def update_preview(frame, is_bgr=False):
    global last_preview_time
    now = time.time()
    if now - last_preview_time < PREVIEW_INTERVAL:
        return
    last_preview_time = now

    try:
        if frame is None:
            return

        # HMI responsive preview: fit the current preview frame instead of fixed 760x430.
        preview_w = preview_frame.winfo_width() - 8
        preview_h = preview_frame.winfo_height() - 8
        if preview_w < 50 or preview_h < 50:
            preview_w = int(root.winfo_width() * 0.45)
            preview_h = int(root.winfo_height() * 0.45)

        display = resize_to_fit(frame, preview_w, preview_h)
        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk, text="")
    except Exception as e:
        print("Preview error:", e)


def show_page(page):
    for p in [page_input, page_capture, page_complete]:
        p.pack_forget()
    page.pack(fill="both", expand=True)
    root.update_idletasks()


def show_capture_complete_overlay(title, detail, duration_ms=1000):
    """Show a large COMPLETE badge in the middle of the capture page, then hide automatically."""
    try:
        overlay_complete_title.config(text=title)
        overlay_complete_detail.config(text=detail)
        overlay_complete.place(relx=0.5, rely=0.5, anchor="center")
        overlay_complete.lift()
        root.after(duration_ms, lambda: overlay_complete.place_forget())
    except Exception as e:
        print("complete overlay error:", e)


def update_header():
    if current_stage in ("cam1", "cam2"):
        box_title = "Inner Box"
        box_value = f"{inner_index} / {quantity}"
        stage_text = "CAM1 : Vendor + TOA" if current_stage == "cam1" else "CAM2 : TOB"
        progress_title_label.config(text="Inner Box")
        progress_label.config(text=box_value)
        percent = int((inner_index / max(quantity, 1)) * 100)
    elif current_stage == "cam3":
        box_title = "Outer Box"
        box_value = f"{outer_index} / {outer_total}"
        stage_text = "CAM3 : USB / TOC YOLO"
        progress_title_label.config(text="Outer Box")
        progress_label.config(text=box_value)
        percent = int((outer_index / max(outer_total, 1)) * 100)
    else:
        box_title = "-"
        box_value = "-"
        stage_text = "-"
        percent = 0

    info_box_title.config(text=box_title)
    info_box.config(text=box_value)
    info_stage.config(text=stage_text)
    progress_percent_label.config(text=f"{percent}%")
    progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)


def get_inner_dir(index):
    return ensure_dir(os.path.join(current_job_dir, "Inner", f"{index:03d}"))


def get_outer_dir(index):
    return ensure_dir(os.path.join(current_job_dir, "Outer", f"{index:03d}"))


def create_job_folder():
    global current_job_dir
    name = datetime.now().strftime("manual_%Y%m%d_%H%M%S")
    current_job_dir = ensure_dir(os.path.join(MANUAL_SAVE_ROOT, name))
    ensure_dir(os.path.join(current_job_dir, "Inner"))
    ensure_dir(os.path.join(current_job_dir, "Outer"))
    return current_job_dir

# =========================================================
# PI CAMERA MANUAL WRAPPER
# =========================================================
class ManualPiCamera:
    def __init__(self, index, name, focus_delay_sec=None):
        self.index = index
        self.name = name
        self.focus_delay_sec = FOCUS_DELAY_SEC if focus_delay_sec is None else float(focus_delay_sec)
        self.picam2 = None
        self.latest_frame = None
        self.thread = None
        self.preview_config = None
        self.camera_lock = threading.Lock()
        self.capturing = False

    def open(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available. Please install/use on Raspberry Pi.")

        self.picam2 = Picamera2(camera_num=self.index)

        # T15 no-switch mode:
        # Main stream is already 2304x1296. Preview is resized by update_preview().
        # Capture uses capture_array() from the same stream, so no switch_mode is needed.
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": (PI_CAPTURE_WIDTH, PI_CAPTURE_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(self.preview_config)
        self.picam2.start()
        time.sleep(0.5)

    def ensure_preview_mode(self):
        """T15 no-switch mode: already running in 2K main stream, nothing to switch."""
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")
        return

    def start_preview(self):
        def loop():
            while not preview_stop.is_set() and self.picam2 is not None:
                try:
                    if self.capturing:
                        time.sleep(0.03)
                        continue
                    with self.camera_lock:
                        if self.picam2 is None or self.capturing:
                            continue
                        frame = self.picam2.capture_array()
                    self.latest_frame = frame
                    root.after(0, lambda f=frame: update_preview(f, is_bgr=False))
                except Exception as e:
                    print(f"{self.name} preview error:", e)
                    break
                time.sleep(0.02)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def autofocus(self, wait_sec=None):
        """Run real autofocus and do not add another full delay after it.

        T19 behavior:
        - Use Picamera2 autofocus_cycle(wait=True) so the lens can finish focusing.
        - Do NOT force-stop AF at 1.0/1.5s like T18.
        - Do NOT add another 1.0/1.5s delay after AF is done.
        - Add only a tiny settle delay so the lens is stable before capture.
        """
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")

        settle_sec = 0.15
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
            print(f"{self.name} autofocus_cycle(wait=True) failed, fallback to timed AF trigger:", e)

        # Fallback only if blocking AF is unavailable.
        if not af_ok:
            fallback_wait = float(self.focus_delay_sec if wait_sec is None else wait_sec)
            try:
                if controls is not None:
                    self.picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Auto,
                        "AfTrigger": controls.AfTriggerEnum.Start,
                    })
            except Exception as e:
                print(f"{self.name} AfTrigger fallback failed:", e)
            end_time = time.time() + max(0.0, fallback_wait)
            while time.time() < end_time:
                if preview_stop.is_set():
                    break
                time.sleep(0.02)

        # Tiny lens settle only, not another focus delay.
        end_settle = time.time() + settle_sec
        while time.time() < end_settle:
            if preview_stop.is_set():
                break
            time.sleep(0.02)

        print(f"{self.name} real AF total wait: {time.time() - t0:.2f}s (no extra delay)")

    def capture_file(self, path, return_to_preview=True):
        if self.picam2 is None:
            raise RuntimeError(f"{self.name} not opened")

        self.capturing = True
        try:
            with self.camera_lock:
                # T15: capture directly from the already-running 2K stream.
                # No switch_mode_and_capture_array(), so no 1280->2304->1280 delay/flicker.
                frame = self.picam2.capture_array()

            self.latest_frame = frame
            queue_save_rgb(path, frame, SAVE_JPEG_QUALITY)
            print(f"{self.name}: queued save {path} at {frame.shape[1]}x{frame.shape[0]} (no switch)")
            return path
        finally:
            self.capturing = False

    def close(self):
        try:
            self.capturing = True
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
        except Exception as e:
            print(f"{self.name} close error:", e)
        self.picam2 = None
        self.capturing = False

# =========================================================
# USB CAMERA PREVIEW WRAPPER (CAM3)
# =========================================================
class ManualUsbPreview:
    def __init__(self, device, name="CAM3"):
        self.device = device
        self.name = name
        self.cap = None
        self.thread = None
        self.latest_frame = None

    def open(self):
        self.cap = cv2.VideoCapture(f"/dev/video{self.device}", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        if not self.cap.isOpened():
            raise RuntimeError(f"{self.name}: cannot open /dev/video{self.device}")

        ok = False
        frame = None
        for _ in range(10):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.08)
        if not ok or frame is None:
            self.close()
            raise RuntimeError(f"{self.name}: cannot read frame from /dev/video{self.device}")
        self.latest_frame = frame

    def start_preview(self):
        def loop():
            bad_count = 0
            ready_announced = False
            while not preview_stop.is_set() and self.cap is not None:
                try:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        bad_count += 1
                        if bad_count >= 20:
                            root.after(0, lambda: set_status(f"{self.name}: Signal lost / Cannot Read Frame. Reconnect then press RESET CURRENT CAMERA", RED))
                            break
                        time.sleep(0.08)
                        continue
                    bad_count = 0
                    self.latest_frame = frame
                    if not ready_announced:
                        ready_announced = True
                        root.after(0, lambda: set_status(f"{self.name} Ready on /dev/video{self.device}. Auto capture will start", GREEN))
                    root.after(0, lambda f=frame: update_preview(f, is_bgr=True))
                except Exception as e:
                    print(f"{self.name} preview error:", e)
                    root.after(0, lambda e=e: set_status(f"{self.name}: Preview error {e}", RED))
                    break
                time.sleep(0.02)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            print(f"{self.name} close error:", e)
        self.cap = None

# =========================================================
# CAMERA OPEN / RESET
# =========================================================
def open_current_stage_camera():
    global current_camera, last_preview_time
    stop_preview_loop()
    release_current_camera_for_stage_switch()
    preview_stop.clear()
    last_preview_time = 0
    update_header()
    show_page(page_capture)

    try:
        if current_stage == "cam1":
            set_step(f"Inner Box {inner_index}/{quantity} : CAM1 preview. Press CAPTURE once, CAM1 then CAM2 will autofocus/capture")
            set_status("CAM1 opening/standby...")
            cam = get_or_open_persistent_picam(1)
            cam.ensure_preview_mode()
            current_camera = cam
            cam.start_preview()
            set_status(f"CAM1 Ready. No-switch 2K stream / sequential AF ready {PI_CAPTURE_WIDTH}x{PI_CAPTURE_HEIGHT}")

        elif current_stage == "cam2":
            set_step(f"Inner Box {inner_index}/{quantity} : CAM2 auto capture")
            set_status("CAM2 opening/standby...")
            cam = get_or_open_persistent_picam(2)
            current_camera = cam
            # CAM2 is normally auto-captured, so no preview is started here.
            set_status(f"CAM2 Ready Standby. No-switch 2K capture {PI_CAPTURE_WIDTH}x{PI_CAPTURE_HEIGHT}")

        elif current_stage == "cam3":
            global selected_usb_device
            set_step(f"Outer Box {outer_index}/{outer_total} : CAM3 YOLO direct capture")
            set_status("CAM3 scanning USB devices...")
            dev = selected_usb_device if selected_usb_device is not None else find_usb_device()
            if dev is None:
                preview_label.config(image="", text="USB Camera not found\nReconnect USB camera and press RESET CURRENT CAMERA")
                set_status("USB Camera not found. Reconnect and press RESET CURRENT CAMERA.", RED)
                selected_usb_device = None
                return
            selected_usb_device = dev
            preview_label.config(image="", text="CAM3 YOLO Direct Capture")
            set_status(f"CAM3 selected /dev/video{dev}. YOLO capture will start")
            maybe_auto_capture_cam3()

    except Exception as e:
        set_status(f"Open camera error: {e}", RED)
        messagebox.showerror("Camera Error", str(e))


def reset_current_camera():
    """Reset เน€เธเธเธฒเธฐเธเธฅเนเธญเธ/เธเธฑเนเธเธ•เธญเธเธเธฑเธเธเธธเธเธฑเธ เนเธกเนเธขเนเธญเธเธเธฅเธฑเธเนเธเน€เธฃเธดเนเธก Job เนเธซเธกเน"""
    global auto_cam3_capture_on_ready, selected_usb_device
    auto_cam3_capture_on_ready = False
    if current_stage == "cam3":
        selected_usb_device = None
    if current_stage not in ("cam1", "cam2", "cam3"):
        return
    set_status("Reset current camera... closing and rescanning device")
    stop_preview_loop()
    if current_stage in ("cam1", "cam2"):
        close_persistent_pi_cameras()
    else:
        safe_close_camera()
    try:
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    current_cam_holder["cam"] = None
    time.sleep(0.3)  # เนเธซเน OS เธชเธฃเนเธฒเธ /dev/video เนเธซเธกเนเธซเธฅเธฑเธเธ–เธญเธ”เน€เธชเธตเธขเธ
    open_current_stage_camera()


def get_v4l2_usb_groups():
    """
    เธญเนเธฒเธเธเธฒเธเธเธณเธชเธฑเนเธ v4l2-ctl --list-devices เนเธฅเนเธงเธเธฑเธ”เธเธฅเธธเนเธกเน€เธเนเธ:
    [{"name": "Rapoo Camera...", "videos": [0, 18]}, ...]

    เน€เธซเธ•เธธเธเธฅ: USB camera เธ–เธญเธ”เน€เธชเธตเธขเธเนเธซเธกเนเนเธฅเนเธง /dev/video เธญเธฒเธเน€เธเธฅเธตเนเธขเธ
    เน€เธเนเธ /dev/video0 -> /dev/video1 เนเธ•เน v4l2-ctl เธเธฐเธเธญเธเธเธทเนเธญเธเธฅเนเธญเธเธเธฃเธดเธเน€เธชเธกเธญ
    """
    groups = []
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=2,
        )
        print("========== v4l2-ctl --list-devices ==========")
        print(out)
        print("=============================================")
    except Exception as e:
        print("v4l2-ctl read error:", e)
        return groups

    current = None
    for raw in out.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Device header เน€เธเนเธ: Rapoo Camera: Rapoo Camera (usb-xhci-hcd.0-1):
        if not line.startswith("\t") and line.endswith(":"):
            current = {"name": line[:-1], "videos": []}
            groups.append(current)
            continue

        if "/dev/video" in line and current is not None:
            try:
                dev_no = int(line.strip().replace("/dev/video", ""))
                current["videos"].append(dev_no)
            except Exception:
                pass

    return groups


def _open_read_once(dev, backend=cv2.CAP_V4L2):
    cap = cv2.VideoCapture(f"/dev/video{dev}", backend)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            return False

        # USB camera เธเธฒเธเธ•เธฑเธงเธ•เนเธญเธ warm up เธซเธฅเธฒเธข frame เธซเธฅเธฑเธเธ–เธญเธ”เน€เธชเธตเธขเธเนเธซเธกเน
        for _ in range(25):
            ret, frame = cap.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                return True
            time.sleep(0.06)
        return False
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _can_read_usb_device(dev):
    """เธฅเธญเธเธญเนเธฒเธ frame เธเธฃเธดเธ เนเธกเนเนเธเนเนเธเนเน€เธเนเธเธงเนเธฒเน€เธเธดเธ” device เนเธ”เน"""
    # เธฅเธญเธ V4L2 เธเนเธญเธ เธ–เนเธฒเนเธกเนเนเธ”เนเธเนเธญเธข fallback เน€เธเนเธ CAP_ANY
    if _open_read_once(dev, cv2.CAP_V4L2):
        return True
    try:
        if _open_read_once(dev, cv2.CAP_ANY):
            return True
    except Exception:
        pass
    return False


def find_usb_device():
    """
    เธซเธฒ USB Camera เธเธฒเธ v4l2-ctl --list-devices เน€เธ—เนเธฒเธเธฑเนเธเน€เธเนเธเธซเธฅเธฑเธ
    เน€เธเธทเนเธญเนเธกเนเนเธเน€เธเธดเธ” device เธเธญเธ Pi เน€เธเนเธ /dev/video20-35 เธ—เธตเนเธ—เธณเนเธซเน select() timeout / เธเนเธฒเธ

    เธซเธฅเธฑเธเธเธฒเธฃ:
    - เน€เธฅเธทเธญเธเน€เธเธเธฒเธฐเธเธฅเธธเนเธกเธ—เธตเนเธเธทเนเธญเน€เธเนเธ Rapoo/USB camera
    - เนเธเน /dev/video เน€เธฅเธเธ•เนเธณเธชเธธเธ”เธเธญเธเธเธฅเธธเนเธกเธเธฑเนเธเธเนเธญเธ เน€เธเนเธ /dev/video0 เธซเธฃเธทเธญ /dev/video1
    - เนเธกเนเธฅเธญเธ scan pispbe/rp1-cfe/rpi-hevc-dec
    - เนเธกเนเธฅเธญเธ /dev/video18 เธ–เนเธฒเธกเธต /dev/video0 เธซเธฃเธทเธญ /dev/video1 เน€เธเธฃเธฒเธฐ 18 เธกเธฑเธเน€เธเนเธ metadata node
    """
    groups = get_v4l2_usb_groups()
    candidates = []

    for g in groups:
        name = g.get("name", "")
        lname = name.lower()
        videos = sorted(g.get("videos", []))

        is_usb_camera = (
            "rapoo" in lname
            or "usb" in lname
            or ("camera" in lname and "rp1" not in lname and "pisp" not in lname and "rpi" not in lname)
        )
        is_pi_pipeline = (
            "pisp" in lname
            or "rp1-cfe" in lname
            or "rpi-hevc" in lname
            or "platform:" in lname and "usb" not in lname
        )
        if not is_usb_camera or is_pi_pipeline:
            continue

        low_nodes = [v for v in videos if v < 18]
        high_nodes = [v for v in videos if v >= 18]
        for dev in low_nodes + high_nodes:
            candidates.append((dev, name, "v4l2"))

    # fallback เน€เธเธเธฒเธฐ manual list เน€เธฅเธเธ•เนเธณ เน เน€เธ—เนเธฒเธเธฑเนเธ เนเธกเน glob เธ—เธฑเนเธเธฃเธฐเธเธ
    for dev in USB_SCAN_DEVICES:
        candidates.append((dev, "manual_low_device", "manual"))

    seen = set()
    tried = []
    for dev, name, source in candidates:
        if dev in seen:
            continue
        seen.add(dev)
        if dev >= 18:
            # เธฅเธญเธ high node เน€เธเธเธฒเธฐเธเธฃเธ“เธตเนเธกเนเธกเธต low node เธญเนเธฒเธเนเธ”เนเธเธฃเธดเธ เน เนเธ•เนเธขเธฑเธเนเธกเนเธเนเธฒเธกเนเธเธฃเธญเธเธเธตเนเธ–เนเธฒเน€เธเนเธ candidate เธชเธธเธ”เธ—เนเธฒเธข
            pass
        ok = _can_read_usb_device(dev)
        print(f"USB scan {source}: /dev/video{dev} ({name}) read={ok}")
        tried.append(f"/dev/video{dev}")
        if ok:
            set_status(f"CAM3 selected /dev/video{dev} from {source}", BLUE)
            return dev

    set_status("USB Camera not found/readable from v4l2-ctl list", RED)
    print("USB scan failed. Tried:", tried)
    return None

def maybe_auto_capture_cam3():
    """เธซเธฅเธฑเธ CAM1/CAM2 เธเธฃเธ เนเธซเน CAM3 เน€เธฃเธดเนเธก YOLO capture เธญเธฑเธ•เนเธเธกเธฑเธ•เธด เนเธกเนเธ•เนเธญเธเธเธ” CAPTURE เธญเธตเธเธเธฃเธฑเนเธ"""
    global auto_cam3_capture_on_ready
    if current_stage != "cam3" or not running or not auto_cam3_capture_on_ready:
        return
    auto_cam3_capture_on_ready = False

    def delayed_capture():
        if running and current_stage == "cam3":
            set_status("CAM3 auto capture starting...", BLUE)
            capture_current()

    # เธซเธเนเธงเธเธเธดเธ”เธซเธเนเธญเธขเนเธซเน preview / device settle เธซเธฅเธฑเธเน€เธเธดเธ”เธเธฅเนเธญเธ
    root.after(1200, delayed_capture)


# =========================================================
# CAPTURE FLOW
# =========================================================
def start_job():
    global running, quantity, outer_total, inner_index, outer_index, current_stage, auto_cam3_capture_on_ready, selected_usb_device
    qty_text = qty_entry.get().strip()
    if not qty_text.isdigit():
        messagebox.showwarning("Warning", "Please input Quantity")
        return
    quantity = int(qty_text)
    if quantity < 1 or quantity > 1000:
        messagebox.showwarning("Warning", "Quantity must be 1-1000")
        return

    create_job_folder()
    outer_total = math.ceil(quantity / 6)
    inner_index = 1
    outer_index = 1
    running = True
    current_stage = "cam1"
    auto_cam3_capture_on_ready = False
    selected_usb_device = None
    close_persistent_pi_cameras()
    open_current_stage_camera()
    root.after(300, prewarm_cam2_background)


def capture_current():
    if not running:
        return
    if not capture_lock.acquire(blocking=False):
        return
    threading.Thread(target=_capture_current_worker, daemon=True).start()


def _capture_current_worker():
    try:
        if current_stage in ("cam1", "cam2"):
            capture_pi_stage()
        elif current_stage == "cam3":
            capture_cam3_stage()
    finally:
        try:
            capture_lock.release()
        except Exception:
            pass


def capture_pi_stage():
    """
    T21 Flow:
    - Based on T20/T19 no-switch 2K stream for CAM1/CAM2.
    - Keep CAM1/CAM2 persistent/open at 2304x1296.
    - Autofocus one camera at a time, like the older T8 behavior, because dual AF made both cameras slower.
    - Capture CAM1 then CAM2 without any resolution switch.
    - Save images in background threads.
    - COMPLETE overlay remains for 3 seconds.
    """
    global current_stage, inner_index, running, auto_cam3_capture_on_ready, current_camera, last_preview_time

    if current_stage != "cam1":
        set_status("Please capture from CAM1. CAM1 then CAM2 will autofocus/capture sequentially.", YELLOW)
        return

    folder = get_inner_dir(inner_index)
    cam1_path = os.path.join(folder, "cam1_vendor_toa.jpg")
    cam2_path = os.path.join(folder, "cam2_tob.jpg")

    cam1 = None
    cam2 = None
    try:
        set_status("Preparing CAM1 + CAM2 for sequential autofocus/capture...", BLUE)

        cam1 = get_or_open_persistent_picam(1)
        cam2 = get_or_open_persistent_picam(2)
        current_camera = cam1

        # Pause preview capture_array() while AF/capture is running.
        cam1.capturing = True
        cam2.capturing = True

        root.after(0, lambda: set_step(f"Inner Box {inner_index}/{quantity} : CAM1 then CAM2 autofocus/capture"))

        t_all = time.time()

        # ===================== CAM1 =====================
        set_status("CAM1 autofocus...", BLUE)
        t0 = time.time()
        cam1.autofocus(PICAM1_FOCUS_DELAY_SEC)
        t1 = time.time()
        print(f"CAM1 T21 sequential autofocus done in {t1 - t0:.2f}s")

        set_status("CAM1 capturing 2K...", BLUE)
        t0 = time.time()
        cam1.capture_file(cam1_path, return_to_preview=False)
        t1 = time.time()
        print(f"CAM1 T21 sequential capture queued in {t1 - t0:.2f}s")

        # ===================== CAM2 =====================
        set_status("CAM2 autofocus...", BLUE)
        t0 = time.time()
        cam2.autofocus(PICAM2_FOCUS_DELAY_SEC)
        t1 = time.time()
        print(f"CAM2 T21 sequential autofocus done in {t1 - t0:.2f}s")

        set_status("CAM2 capturing 2K...", BLUE)
        t0 = time.time()
        cam2.capture_file(cam2_path, return_to_preview=False)
        t1 = time.time()
        print(f"CAM2 T21 sequential capture queued in {t1 - t0:.2f}s")

        print(f"CAM1+CAM2 T21 sequential AF/capture total: {time.time() - t_all:.2f}s")

        cam1.capturing = False
        cam2.capturing = False

        set_status(f"Queued CAM1 + CAM2 saves for Inner Box {inner_index}/{quantity}", GREEN)

    except Exception as e:
        try:
            if cam1 is not None:
                cam1.capturing = False
            if cam2 is not None:
                cam2.capturing = False
        except Exception:
            pass
        set_status(f"CAM1/CAM2 sequential capture error: {e}", RED)
        root.after(0, lambda e=e: messagebox.showerror("CAM1/CAM2 Error", str(e)))
        return
    finally:
        # Stop current preview loop after capture. In persistent mode, keep both cameras open.
        stop_preview_loop()
        if not PERSISTENT_PI_CAMERAS:
            safe_close_camera()

    completed_no = inner_index
    root.after(0, lambda n=completed_no: show_capture_complete_overlay("✓ COMPLETE", f"Inner Box {n} / {quantity}", COMPLETE_OVERLAY_MS))
    time.sleep(COMPLETE_OVERLAY_WAIT_SEC)

    inner_index += 1
    if inner_index <= quantity:
        current_stage = "cam1"
        root.after(0, open_current_stage_camera)
    else:
        current_stage = "cam3"
        auto_cam3_capture_on_ready = True
        root.after(0, open_current_stage_camera)

def save_yolo_rois(image_path, detections, outer_dir):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    roi_count = 0
    roi_dir = ensure_dir(os.path.join(outer_dir, "roi"))

    for det in detections or []:
        name = str(det.get("name", "roi")).lower()
        box = det.get("box", [])
        crop, fixed_box = crop_by_box(img, box, pad=0)
        if crop is None or crop.size == 0:
            continue
        roi_path = os.path.join(roi_dir, f"{name}.jpg")
        cv2.imwrite(roi_path, crop)
        roi_count += 1
    return roi_count


def capture_cam3_stage():
    global outer_index, current_stage, running, auto_cam3_capture_on_ready
    patch_yolo_loaders()
    folder = get_outer_dir(outer_index)
    image_path = os.path.join(folder, "cam3_toc_full.jpg")

    # CAM3 is rescanned every time for reset safety.
    # Reason: during production the USB camera may be unplugged/replugged and /dev/video can change.
    # Do not use cached USB device here.
    global selected_usb_device
    selected_usb_device = None
    try:
        if isinstance(current_camera, ManualUsbPreview) and current_camera.latest_frame is not None:
            dev = current_camera.device
        else:
            dev = find_usb_device()
    except Exception:
        dev = find_usb_device()

    if dev is None:
        set_status("USB Camera not found. Reconnect and press RESET CURRENT CAMERA.", RED)
        messagebox.showerror("CAM3 Error", "USB Camera not found. Please reconnect USB camera and press RESET CURRENT CAMERA.")
        root.after(0, open_current_stage_camera)
        return

    # เธซเนเธฒเธกเน€เธเธดเธ”เธ—เธ”เธชเธญเธ /dev/video เน€เธ”เธดเธกเธเนเธณเนเธเธเธ“เธฐเธ—เธตเน preview เธขเธฑเธเน€เธเธดเธ”เธญเธขเธนเน
    # เน€เธเธฃเธฒเธฐ OpenCV/V4L2 เธเธ Pi เธเธฒเธเธเธฃเธฑเนเธเธเธฐเธ—เธณเนเธซเน stream เธเนเธฒเธเธซเธฃเธทเธญ timeout
    # เธ–เนเธฒ preview เธกเธต latest_frame เนเธเธฅเธงเนเธฒ device เธเธตเนเธญเนเธฒเธเนเธ”เนเนเธฅเนเธง เธเธถเธ release เนเธฅเนเธงเนเธซเน YOLO เน€เธเธดเธ”เธ•เนเธญ
    try:
        if isinstance(current_camera, ManualUsbPreview) and current_camera.latest_frame is None:
            set_status("CAM3 preview has no frame. Auto reopen camera and retry capture...", RED)
            auto_cam3_capture_on_ready = True
            root.after(0, open_current_stage_camera)
            return
    except Exception:
        pass

    # If CAM3 preview is active, release it before YOLO opens the same /dev/video.
    # In optimized flow we normally skip CAM3 preview, so this usually adds no delay.
    if isinstance(current_camera, ManualUsbPreview):
        stop_preview_loop()
        safe_close_camera()
        time.sleep(CAM3_HANDOFF_WAIT_SEC)

    cam3 = None
    try:
        set_step(f"CAM3 Outer Box {outer_index}/{outer_total} : YOLO detect/capture, no OCR")
        if YOLO_PRELOAD_ENABLED and not yolo_preload_done.is_set():
            set_status("Waiting CAM3 YOLO preload to finish...", BLUE)
            yolo_preload_done.wait(timeout=15.0)

        set_status(f"CAM3 YOLO opening /dev/video{dev}...")
        cam3 = UsbCameraTest(device=dev, name="CAM3", status_cb=set_status, preview_cb=update_preview)
        attach_preloaded_yolo_to_cam3(cam3)
        current_cam_holder["cam"] = cam3
        cam3.open()
        result = cam3.capture_direct_or_yolo(image_path, lambda: running and current_stage == "cam3")
        cam3.close()
        current_cam_holder["cam"] = None

        if not result:
            # Auto retry instead of stopping and waiting for another CAPTURE click.
            set_status("CAM3 capture failed. Auto reopen camera and retry capture...", RED)
            auto_cam3_capture_on_ready = True
            root.after(0, open_current_stage_camera)
            return

        full_path = result.get("image_path", image_path)
        detections = result.get("detections", []) or []
        roi_count = save_yolo_rois(full_path, detections, folder)
        set_status(f"Saved CAM3 full + ROI count: {roi_count}", GREEN)
        completed_outer = outer_index
        root.after(0, lambda n=completed_outer: show_capture_complete_overlay("✓ COMPLETE", f"Outer Box {n} / {outer_total}", COMPLETE_OVERLAY_MS))
        time.sleep(COMPLETE_OVERLAY_WAIT_SEC)
    except Exception as e:
        try:
            if cam3 is not None:
                cam3.close()
        except Exception:
            pass
        current_cam_holder["cam"] = None
        # CAM3 error can happen when USB stream is handed from preview to YOLO.
        # Do not wait for the operator to press CAPTURE again.
        # Re-open the same CAM3 stage and auto-run capture again after preview is ready.
        set_status(f"CAM3 error: {e}. Auto reopen camera and retry capture...", RED)
        auto_cam3_capture_on_ready = True
        root.after(0, open_current_stage_camera)
        return

    outer_index += 1
    if outer_index <= outer_total:
        current_stage = "cam3"
        auto_cam3_capture_on_ready = True
        root.after(0, open_current_stage_camera)
    else:
        running = False
        safe_close_camera()
        root.after(0, show_complete)

# เนเธเน dict เธเธตเนเน€เธเธทเนเธญเนเธซเน CAM3 worker close เนเธ”เนเนเธ”เธขเนเธกเนเธเธเธเธฑเธ ManualPiCamera state
current_cam_holder = {"cam": None}


def show_complete():
    wait_for_pending_saves(5.0)
    complete_title.config(text="MANUAL CAPTURE COMPLETED")
    complete_detail.config(text=f"Inner: {quantity} boxes | Outer: {outer_total} boxes\nSaved at:\n{current_job_dir}")
    show_page(page_complete)


def reset_to_input():
    global running, current_stage, quantity, outer_total, inner_index, outer_index, current_job_dir, auto_cam3_capture_on_ready, selected_usb_device
    running = False
    auto_cam3_capture_on_ready = False
    selected_usb_device = None
    wait_for_pending_saves(2.0)
    stop_preview_loop()
    close_persistent_pi_cameras()
    safe_close_camera()
    try:
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    current_cam_holder["cam"] = None
    current_stage = "input"
    quantity = 0
    outer_total = 0
    inner_index = 1
    outer_index = 1
    current_job_dir = ""
    qty_entry.delete(0, tk.END)
    show_page(page_input)


def exit_app():
    reset_running = False
    wait_for_pending_saves(2.0)
    try:
        stop_preview_loop()
        close_persistent_pi_cameras()
        safe_close_camera()
        cam = current_cam_holder.get("cam")
        if cam is not None:
            cam.close()
    except Exception:
        pass
    root.destroy()


# =========================================================
# TOUCHSCREEN NUMBER PAD
# =========================================================
def qty_insert(value):
    try:
        qty_entry.focus_set()
        current = qty_entry.get().strip()
        if len(current) >= 4:
            return
        qty_entry.delete(0, tk.END)
        qty_entry.insert(0, current + str(value))
    except Exception as e:
        print("qty_insert error:", e)


def qty_backspace():
    try:
        qty_entry.focus_set()
        current = qty_entry.get().strip()
        qty_entry.delete(0, tk.END)
        qty_entry.insert(0, current[:-1])
    except Exception as e:
        print("qty_backspace error:", e)


def qty_clear():
    try:
        qty_entry.focus_set()
        qty_entry.delete(0, tk.END)
    except Exception as e:
        print("qty_clear error:", e)

# =========================================================
# UI BUILD
# =========================================================
root = tk.Tk()
root.title("AI Camera Manual Capture - Dataset Mode T15 2K No-Switch Pi + YOLO Preload")
# Fullscreen responsive layout for small HMI/Raspberry Pi display
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
root.configure(bg=BG)

# ---------- PAGE INPUT ----------
page_input = tk.Frame(root, bg=BG)
input_card = tk.Frame(page_input, bg=CARD, padx=28, pady=18, highlightbackground=BORDER, highlightthickness=1)
input_card.place(relx=0.5, rely=0.50, anchor="center", relwidth=0.92, relheight=0.90)

tk.Label(input_card, text="Manual Dataset Capture", font=("Arial", 22, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 10))
tk.Label(input_card, text="Quantity / Inner Box", bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w", pady=(0, 6))
qty_entry = tk.Entry(input_card, font=("Arial", 26), relief="solid", borderwidth=1, justify="center")
qty_entry.pack(fill="x", ipady=6, pady=(0, 12))

# On-screen keypad for touchscreen HMI.
keypad_frame = tk.Frame(input_card, bg=CARD)
keypad_frame.pack(fill="both", expand=True, pady=(0, 12))

key_rows = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
    ["CLR", "0", "DEL"],
]
for r, row in enumerate(key_rows):
    keypad_frame.grid_rowconfigure(r, weight=1)
    for c, key in enumerate(row):
        keypad_frame.grid_columnconfigure(c, weight=1)
        if key == "CLR":
            cmd = qty_clear
            bg = GRAY
            fg = TEXT
        elif key == "DEL":
            cmd = qty_backspace
            bg = GRAY
            fg = TEXT
        else:
            cmd = lambda v=key: qty_insert(v)
            bg = SOFT
            fg = TEXT
        tk.Button(
            keypad_frame,
            text=key,
            font=("Arial", 18, "bold"),
            bg=bg,
            fg=fg,
            activebackground="#cfd8dc",
            activeforeground=TEXT,
            relief="flat",
            command=cmd,
        ).grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

tk.Button(input_card, text="START MANUAL CAPTURE", font=("Arial", 15, "bold"), bg=BLUE, fg="white", activebackground=BLUE_DARK, activeforeground="white", relief="flat", command=start_job).pack(fill="x", ipady=8, pady=(2, 8))
tk.Button(input_card, text="EXIT APP", font=("Arial", 15, "bold"), bg=RED, fg="white", activebackground=RED_DARK, activeforeground="white", relief="flat", command=exit_app).pack(fill="x", ipady=8)

# ---------- PAGE CAPTURE ----------
page_capture = tk.Frame(root, bg=BG)
capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=6, pady=6)

header_frame = tk.Frame(capture_container, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=10)
header_frame.pack(fill="x", pady=(0, 6))

header_top = tk.Frame(header_frame, bg=CARD)
header_top.pack(fill="x")
info_left = tk.Frame(header_top, bg=CARD)
info_left.pack(side="left", fill="x", expand=True)
header_buttons = tk.Frame(header_top, bg=CARD)
header_buttons.pack(side="right")

def action_button(parent, text, bg, fg="white", command=None, width=130, height=42):
    wrap = tk.Frame(parent, bg=CARD, width=width, height=height)
    wrap.pack(side="left", padx=(0, 8))
    wrap.pack_propagate(False)
    btn = tk.Button(wrap, text=text, bg=bg, fg=fg, font=("Arial", 11, "bold"), relief="flat", cursor="hand2", command=command, activebackground=BLUE_DARK if bg == BLUE else (RED_DARK if bg == RED else GREEN_DARK), activeforeground="white")
    btn.pack(fill="both", expand=True)
    return btn

action_button(header_buttons, "RESET", GRAY, fg=TEXT, command=reset_current_camera, width=110)
action_button(header_buttons, "CAPTURE", GREEN, command=capture_current, width=120)
action_button(header_buttons, "EXIT", RED, command=exit_app, width=80)

info_row = tk.Frame(info_left, bg=CARD)
info_row.pack(fill="x")

def header_info_box(parent, title, width=14):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 18))
    tk.Label(box, text=title, bg=CARD, fg=MUTED, font=("Arial", 16, "bold"), anchor="w").pack(anchor="w")
    val = tk.Label(box, text="-", bg=CARD, fg=TEXT, font=("Arial", 15, "bold"), width=width, anchor="w")
    val.pack(anchor="w", pady=(6, 0))
    return val

info_box_wrap = tk.Frame(info_row, bg=CARD)
info_box_wrap.pack(side="left", padx=(0, 18))
info_box_title = tk.Label(info_box_wrap, text="Inner Box", bg=CARD, fg=MUTED, font=("Arial", 16, "bold"), anchor="w")
info_box_title.pack(anchor="w")
info_box = tk.Label(info_box_wrap, text="-", bg=CARD, fg=TEXT, font=("Arial", 15, "bold"), width=12, anchor="w")
info_box.pack(anchor="w", pady=(6, 0))
info_stage = header_info_box(info_row, "Stage", 24)

body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

capture_left = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=8)
capture_left.pack(side="left", fill="both", expand=True, padx=(0, 6))

tk.Label(capture_left, text="Capture Images", bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w")
tk.Label(capture_left, text="Preview", bg=CARD, fg=MUTED, font=("Arial", 11, "bold")).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(capture_left, bg=DARK, highlightbackground=DARK, highlightthickness=2)
preview_frame.pack(fill="both", expand=True, pady=(2, 6))
preview_label = tk.Label(preview_frame, text="Camera Preview", bg=DARK, fg="white", font=("Arial", 14))
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

# Center COMPLETE overlay for each successful inner/outer capture.
overlay_complete = tk.Frame(preview_frame, bg=GREEN, padx=28, pady=24, highlightbackground="white", highlightthickness=2)
overlay_complete_title = tk.Label(overlay_complete, text="✓ COMPLETE", bg=GREEN, fg="white", font=("Arial", 30, "bold"))
overlay_complete_title.pack(pady=(0, 8))
overlay_complete_detail = tk.Label(overlay_complete, text="-", bg=GREEN, fg="white", font=("Arial", 20, "bold"))
overlay_complete_detail.pack()

preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)
capture_state_dot = tk.Canvas(preview_overlay, width=14, height=14, bg=DARK, highlightthickness=0)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill=YELLOW, outline=YELLOW)
capture_state_label = tk.Label(preview_overlay, text="Waiting", bg=DARK, fg="white", font=("Arial", 11, "bold"), padx=2, pady=2)
capture_state_label.pack(side="left")

preview_button_row = tk.Frame(capture_left, bg=CARD)
preview_button_row.pack(fill="x", pady=(8, 0))

reset_preview_btn = tk.Button(
    preview_button_row,
    text="RESET CURRENT CAMERA",
    bg=GRAY,
    fg=TEXT,
    activebackground="#aeb4b8",
    activeforeground=TEXT,
    font=("Arial", 11, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
    command=reset_current_camera,
)
reset_preview_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

capture_preview_btn = tk.Button(
    preview_button_row,
    text="CAPTURE",
    bg=GREEN,
    fg="white",
    activebackground=GREEN_DARK,
    activeforeground="white",
    font=("Arial", 11, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
    command=capture_current,
)
capture_preview_btn.pack(side="left", fill="x", expand=True)

capture_right = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=8)
capture_right.pack(side="right", fill="both", expand=True)

tk.Label(capture_right, text="Status", bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(anchor="w")

product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))
progress_title_label = tk.Label(product_header, text="Inner Box", bg=CARD, fg=TEXT, font=("Arial", 15, "bold"))
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

step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="both", expand=True, pady=(16, 10))

tk.Label(step_status_box, text="Step", bg=SOFT, fg="#34495e", font=("Arial", 16, "bold")).pack(anchor="w")
step_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 10), wraplength=360, justify="left")
step_label.pack(anchor="w", pady=(4, 18))

tk.Label(step_status_box, text="Status", bg=SOFT, fg="#34495e", font=("Arial", 16, "bold")).pack(anchor="w")
status_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 10), wraplength=360, justify="left")
status_label.pack(anchor="w", pady=(4, 0))

hint = tk.Label(step_status_box, text="Manual Mode:\n- CAM1/CAM2 เนเธกเนเธกเธต background detection\n- Press Capture once: CAM1/CAM2 autofocus and capture together\n- CAM3 เนเธเน YOLO เน€เธ”เธดเธก เนเธ•เนเนเธกเน OCR\n- Reset Current Camera เนเธกเนเน€เธฃเธดเนเธกเธเธฒเธเนเธซเธกเน", bg=SOFT, fg=MUTED, font=("Arial", 13), justify="left", wraplength=360)
hint.pack(anchor="w", pady=(28, 0))

# ---------- PAGE COMPLETE ----------
page_complete = tk.Frame(root, bg=BG)
complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=680, height=330)
complete_title = tk.Label(complete_card, text="MANUAL CAPTURE COMPLETED", font=("Arial", 15, "bold"), bg=CARD, fg=GREEN)
complete_title.pack(pady=(10, 16))
complete_detail = tk.Label(complete_card, text="-", font=("Arial", 13), bg=CARD, fg=TEXT, wraplength=600, justify="center")
complete_detail.pack(pady=(0, 22))
tk.Button(complete_card, text="OK", bg=BLUE, fg="white", font=("Arial", 16, "bold"), width=18, relief="flat", command=reset_to_input).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", exit_app)
show_page(page_input)
start_yolo_preload_thread()
root.mainloop()
