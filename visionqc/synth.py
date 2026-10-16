"""Synthetic inspection scenes with exact ground truth.

Real line data is expensive to label, so we render scenes of catalogue parts on a
"conveyor" with a printed ArUco marker, inject known defects, then add the things that
make real images hard: uneven lighting, sensor noise, optical blur and (optionally)
camera tilt. Because every dimension is known, measurement accuracy is testable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

DEFECTS = ("chip", "missing_hole", "oversize", "undersize", "bad_hole")
BG_LEVEL = (42, 44, 47)        # dark belt (BGR)
PART_LEVEL = (188, 192, 198)   # machined aluminium / steel


@dataclass
class PlacedPart:
    kind: str
    cx_mm: float
    cy_mm: float
    angle_deg: float = 0.0
    defect: str | None = None

    @property
    def expected_status(self) -> str:
        return "FAIL" if self.defect else "PASS"


def _hole_offsets(spec: dict[str, Any], n: int) -> list[tuple[float, float]]:
    """Hole centres in the part frame (mm). Long axis = x."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    span = spec["length_mm"] * 0.6 if spec["shape"] == "rect" else spec["diameter_mm"] * 0.5
    return [(-span / 2 + span * i / (n - 1), 0.0) for i in range(n)]


def _footprint_radius_mm(spec: dict[str, Any]) -> float:
    if spec["shape"] == "circle":
        return spec["diameter_mm"] / 2 + 1.5
    return math.hypot(spec["length_mm"], spec["width_mm"]) / 2 + 1.5


def render_scene(
    parts: list[PlacedPart],
    cfg: dict[str, Any],
    px_per_mm: float = 5.0,
    size: tuple[int, int] = (1280, 720),
    marker_origin_mm: tuple[float, float] = (8.0, 8.0),
    marker_id: int = 7,
    seed: int = 0,
    noise_sigma: float = 4.0,
    lighting: float = 0.25,
    supersample: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    w, h = size
    S = px_per_mm * supersample                       # supersampled px per mm
    canvas = np.zeros((h * supersample, w * supersample, 3), np.uint8)
    canvas[:] = BG_LEVEL
    SHIFT = 4
    F = 1 << SHIFT

    def P(x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(x_mm * S * F)), int(round(y_mm * S * F))

    def to_world(p: PlacedPart, dx: float, dy: float) -> tuple[float, float]:
        a = math.radians(p.angle_deg)
        return (p.cx_mm + dx * math.cos(a) - dy * math.sin(a),
                p.cy_mm + dx * math.sin(a) + dy * math.cos(a))

    truth: list[dict[str, Any]] = []
    for p in parts:
        spec = cfg["parts"][p.kind]
        delta = {"oversize": 1.6, "undersize": -1.6}.get(p.defect, 0.0)
        tone = tuple(int(np.clip(c + rng.normal(0, 6), 0, 255)) for c in PART_LEVEL)

        if spec["shape"] == "circle":
            d = spec["diameter_mm"] + delta
            cv2.circle(canvas, P(p.cx_mm, p.cy_mm), int(round(d / 2 * S * F)), tone, -1, cv2.LINE_AA, SHIFT)
            dims = {"diameter_mm": d}
        else:
            L, W = spec["length_mm"] + delta, spec["width_mm"] + delta
            box = cv2.boxPoints(((p.cx_mm * S, p.cy_mm * S), (L * S, W * S), p.angle_deg))
            cv2.fillPoly(canvas, [np.round(box * F).astype(np.int32)], tone, cv2.LINE_AA, SHIFT)
            dims = {"length_mm": L, "width_mm": W}

        holes = spec.get("holes", {"count": 0})
        n = holes.get("count", 0)
        offsets = _hole_offsets(spec, n)
        if p.defect == "missing_hole" and offsets:
            offsets = offsets[:-1]
        hole_d = holes.get("diameter_mm", 0) + (1.3 if p.defect == "bad_hole" else 0.0)
        for dx, dy in offsets:
            hx, hy = to_world(p, dx, dy)
            cv2.circle(canvas, P(hx, hy), int(round(hole_d / 2 * S * F)), BG_LEVEL, -1, cv2.LINE_AA, SHIFT)

        if p.defect == "chip":
            if spec["shape"] == "circle":
                r_edge = spec["diameter_mm"] / 2
                theta = rng.uniform(0, 2 * math.pi)
                ex, ey = to_world(p, r_edge * math.cos(theta), r_edge * math.sin(theta))
            else:
                ex, ey = to_world(p, rng.uniform(-0.2, 0.2) * spec["length_mm"], spec["width_mm"] / 2)
            cv2.circle(canvas, P(ex, ey), int(round(3.0 * S * F)), BG_LEVEL, -1, cv2.LINE_AA, SHIFT)

        truth.append({**asdict(p), "expected_status": p.expected_status,
                      "center_px": (p.cx_mm * px_per_mm, p.cy_mm * px_per_mm),
                      "hole_count": len(offsets), "hole_diameter_mm": hole_d if offsets else None, **dims})

    # Printed marker with a white quiet zone.
    marker_mm = cfg["calibration"]["marker_mm"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["calibration"]["aruco_dict"]))
    side = int(round(marker_mm * S))
    quiet = side // 4
    mk = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    x0, y0 = int(round(marker_origin_mm[0] * S)), int(round(marker_origin_mm[1] * S))
    canvas[y0 - quiet:y0 + side + quiet, x0 - quiet:x0 + side + quiet] = 245
    canvas[y0:y0 + side, x0:x0 + side] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Uneven illumination: a linear ramp in a random direction.
    if lighting > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, 2 * math.pi)
        ramp = (xx * math.cos(a) + yy * math.sin(a))
        ramp = (ramp - ramp.min()) / max(np.ptp(ramp), 1e-6)
        img *= (1 - lighting / 2 + lighting * ramp)[..., None]

    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    if noise_sigma > 0:
        img += rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), truth


def random_scene(
    cfg: dict[str, Any],
    seed: int = 0,
    n_parts: int = 7,
    defect_rate: float = 0.45,
    px_per_mm: float = 5.0,
    size: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Random non-overlapping layout of catalogue parts with random defects."""
    rng = np.random.default_rng(seed)
    field_w, field_h = size[0] / px_per_mm, size[1] / px_per_mm
    kinds = list(cfg["parts"].keys())
    marker_box = (0, 0, 8 + cfg["calibration"]["marker_mm"] + 12, 8 + cfg["calibration"]["marker_mm"] + 12)
    placed: list[tuple[PlacedPart, float]] = []

    for _ in range(4000):
        if len(placed) >= n_parts:
            break
        kind = kinds[int(rng.integers(len(kinds)))]
        spec = cfg["parts"][kind]
        r = _footprint_radius_mm(spec) + 2
        cx, cy = rng.uniform(r + 3, field_w - r - 3), rng.uniform(r + 3, field_h - r - 3)
        if cx - r < marker_box[2] and cy - r < marker_box[3]:
            continue
        if any(math.hypot(cx - q.cx_mm, cy - q.cy_mm) < r + qr + 3 for q, qr in placed):
            continue
        defect = None
        if rng.random() < defect_rate:
            options = [d for d in DEFECTS if d not in ("missing_hole", "bad_hole")
                       or spec.get("holes", {}).get("count", 0) > 0]
            defect = options[int(rng.integers(len(options)))]
        placed.append((PlacedPart(kind, float(cx), float(cy), float(rng.uniform(0, 180)), defect), r))

    return render_scene([p for p, _ in placed], cfg, px_per_mm=px_per_mm, size=size, seed=seed)


def tilt(image: np.ndarray, strength: float = 0.08, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a camera that is not perpendicular to the belt. Returns (image, H)."""
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inward = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])   # keep the whole scene in frame
    jitter = rng.uniform(0, strength, (4, 2)) * np.float32([w, h]) * inward
    dst = (src + jitter).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=BG_LEVEL)
    return out, H
