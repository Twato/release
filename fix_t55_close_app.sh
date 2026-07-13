#!/usr/bin/env bash
set -u

PROJECT_DIR="/home/toto/AI_CAMERA_TEST_YOLO_ROI"
MAIN_FILE="$PROJECT_DIR/T55.py"

if [[ ! -f "$MAIN_FILE" ]]; then
    echo "[ERROR] Not found: $MAIN_FILE"
    exit 1
fi

cp "$MAIN_FILE" "$MAIN_FILE.backup_before_exit_fix"

python3 - "$MAIN_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

old = "    root.destroy\n"
new = "    root.destroy()\n"

if old not in text:
    print("[WARNING] Exact 'root.destroy' line was not found.")
    print("Please inspect exit_app() manually.")
    sys.exit(2)

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("[OK] Changed root.destroy to root.destroy()")
PY

# Stop the hidden old Main process and its launcher so the flock lock is released.
pkill -f "$PROJECT_DIR/T55.py" 2>/dev/null || true
pkill -f "$PROJECT_DIR/run_ai_camera_main.sh" 2>/dev/null || true

# Stop old Mock API process left from the previous launcher run.
pkill -f "$PROJECT_DIR/Mock_API_V1/app.py" 2>/dev/null || true

rm -f /tmp/ai_camera_main_app.lock

echo "[OK] Old hidden process and Mock API were stopped."
echo "[OK] You can now open AI Camera Main again."
