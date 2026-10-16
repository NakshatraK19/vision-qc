"""Vercel Python Serverless Function: /api/inspect

Accepts POST JSON with either:
  - mode = "synth": {seed, n_parts, defect_rate, tilt, noise, lighting, rectify, max_chip_depth_mm, tol_scale}
  - mode = "image": {image_b64, rectify, max_chip_depth_mm, tol_scale}

Runs the VisionQC OpenCV pipeline and returns:
  - annotated_b64, raw_b64, binary_b64 (data URLs)
  - counts, calibration, elapsed_ms, rows, catalogue
"""

from __future__ import annotations

import base64
import copy
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionqc import Inspector, load_config
from visionqc.synth import PlacedPart, random_scene, render_scene, tilt

_BASE_CFG = load_config(ROOT / "config" / "parts.yaml")


def _encode_png(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def run_inspection(payload: dict) -> dict:
    cfg = copy.deepcopy(_BASE_CFG)

    rectify = bool(payload.get("rectify", True))
    cfg["calibration"]["rectify"] = rectify

    if "max_chip_depth_mm" in payload:
        cfg["inspection"]["max_chip_depth_mm"] = float(payload["max_chip_depth_mm"])

    tol_scale = float(payload.get("tol_scale", 1.0))
    for p in cfg["parts"].values():
        p["tol_mm"] = round(p["tol_mm"] * tol_scale, 3)

    mode = payload.get("mode", "synth")
    if mode == "image" and payload.get("image_b64"):
        raw_b64 = payload["image_b64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
        image_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode uploaded image")
        # Cap resolution to 1280px wide for fast serverless execution
        h, w = image_bgr.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            image_bgr = cv2.resize(image_bgr, (1280, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        seed = int(payload.get("seed", 3))
        n_parts = max(1, min(12, int(payload.get("n_parts", 7))))
        defect_rate = float(payload.get("defect_rate", 0.45))
        tilt_strength = float(payload.get("tilt", 0.03))
        noise_sigma = float(payload.get("noise", 4.0))
        lighting = float(payload.get("lighting", 0.25))

        _, truth = random_scene(cfg, seed=seed, n_parts=n_parts, defect_rate=defect_rate)
        placed = [
            PlacedPart(t["kind"], t["cx_mm"], t["cy_mm"], t["angle_deg"], t["defect"])
            for t in truth
        ]
        image_bgr, _ = render_scene(
            placed,
            cfg,
            seed=seed,
            noise_sigma=noise_sigma,
            lighting=lighting,
        )
        if tilt_strength > 0:
            image_bgr, _ = tilt(image_bgr, strength=tilt_strength, seed=seed)

    inspector = Inspector(cfg)
    report = inspector.inspect(image_bgr)
    cal = report.calibration

    binary_bgr = cv2.cvtColor(report.binary, cv2.COLOR_GRAY2BGR)

    return {
        "counts": report.counts,
        "total_parts": len(report.parts),
        "elapsed_ms": round(report.elapsed_ms, 1),
        "fps": round(1000.0 / max(report.elapsed_ms, 1e-3), 1),
        "calibration": {
            "px_per_mm": round(cal.px_per_mm, 3) if cal.is_metric else None,
            "source": cal.source,
            "is_metric": cal.is_metric,
            "notes": report.notes,
        },
        "rows": report.rows(),
        "catalogue": cfg["parts"],
        "images": {
            "annotated": _encode_png(report.annotated),
            "raw": _encode_png(image_bgr),
            "binary": _encode_png(binary_bgr),
        },
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        try:
            res = run_inspection({"mode": "synth", "seed": 3})
            self._send_json(200, res)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            res = run_inspection(payload)
            self._send_json(200, res)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
