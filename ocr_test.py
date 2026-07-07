import easyocr
import re

from config_test import ALLOWLIST

_reader = None

def load_ocr():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader

def clean_text(text):
    text = text.strip()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    return re.sub(r"[^A-Za-z0-9\-_.]", "", text)

def run_easyocr(image):
    reader = load_ocr()

    results = reader.readtext(
        image,
        allowlist=ALLOWLIST,
        detail=1,
        paragraph=False,
        decoder="beamsearch"
    )

    raw = ""
    items = []

    for bbox, text, conf in results:
        raw += text
        items.append({
            "text": text,
            "conf": float(conf)
        })

    return {
        "raw": raw,
        "clean": clean_text(raw),
        "items": items
    }
