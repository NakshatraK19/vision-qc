"""Draw inspection results on the image (operator-facing overlay)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .pipeline import InspectionReport

# BGR
COLORS = {
    "PASS": (80, 200, 60),
    "FAIL": (40, 40, 230),
    "UNKNOWN": (0, 165, 255),
    "PARTIAL": (160, 160, 160),
    "UNCALIBRATED": (200, 120, 255),
}
HOLE_COLOR = (255, 200, 0)
MARKER_COLOR = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, scale, color, thickness=1, bg=(20, 20, 20)):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - base - 2))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return th + base + 6


def draw_report(image: np.ndarray, report: "InspectionReport") -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    s = max(0.45, min(w, h) / 1100)            # font scale tracks resolution
    t = max(1, int(round(s * 2)))

    cal = report.calibration
    if cal.marker_quad is not None:
        q = cal.marker_quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [q], True, MARKER_COLOR, t, cv2.LINE_AA)
        top = cal.marker_quad[np.argmin(cal.marker_quad[:, 1])]
        _label(out, "scale ref", (top[0], top[1] - 6), s * 0.8, MARKER_COLOR)

    occupied: list[tuple[int, int, int, int]] = [(0, 0, w, int(42 * s))]
    for p in report.parts:
        px, py, pbw, pbh = cv2.boundingRect(p.contour)
        occupied.append((px, py, px + pbw, py + pbh))

    for p in report.parts:
        color = COLORS.get(p.status, (255, 255, 255))
        cv2.drawContours(out, [p.contour], -1, color, t + 1, cv2.LINE_AA)
        for hole in p.holes:
            cv2.drawContours(out, [hole], -1, HOLE_COLOR, t, cv2.LINE_AA)

        m = p.measurement
        x, y, bw, bh = cv2.boundingRect(p.contour)
        if m is not None and m.chip_point_px is not None and any("chip" in d for d in p.defects):
            cv2.circle(out, m.chip_point_px, int(14 * s) + 6, COLORS["FAIL"], t + 1, cv2.LINE_AA)

        lines = [f"#{p.id} {p.part_type}  {p.status}"]
        if m is not None:
            if m.shape == "circle":
                dims = f"D {m.diameter_mm:.2f} mm"
            else:
                dims = f"{m.length_mm:.2f} x {m.width_mm:.2f} mm"
            if m.hole_diameters_mm:
                dims += "  holes " + ",".join(f"{d:.1f}" for d in m.hole_diameters_mm)
            lines.append(dims)
        lines += p.defects[:2]

        line_h = int(22 * s + 8)
        block_h = len(lines) * line_h
        block_w = max(
            cv2.getTextSize(ln, FONT, s * (0.75 if i == 0 else 0.62), t if i == 0 else 1)[0][0]
            for i, ln in enumerate(lines)
        ) + 8

        candidates = [
            (x, y - 8 - (len(lines) - 1) * line_h),           # above
            (x, y + bh + line_h),                              # below
            (x + bw + 10, y + line_h),                         # right
            (max(4, x - block_w - 8), y + line_h),             # left
        ]

        def _overlap_score(cand):
            cx, cy = cand
            bx0, by0 = max(0, cx - 2), max(0, cy - line_h)
            bx1, by1 = min(w, bx0 + block_w), min(h, by0 + block_h)
            pen = 0
            if cx < 4 or cy - line_h < 4 or bx1 >= w - 4 or by1 >= h - 4:
                pen += 50000
            for ox0, oy0, ox1, oy1 in occupied:
                ix = max(0, min(bx1, ox1) - max(bx0, ox0))
                iy = max(0, min(by1, oy1) - max(by0, oy0))
                pen += ix * iy
            return pen

        lx, ty = min(candidates, key=_overlap_score)
        occupied.append((lx - 2, ty - line_h, lx + block_w, ty - line_h + block_h))
        for i, line in enumerate(lines):
            if i == 0:
                _label(out, line, (lx, ty), s * 0.75, color, t)
            else:
                _label(out, line, (lx, ty), s * 0.62, (235, 235, 235), 1)
            ty += line_h

    c = report.counts
    total = len(report.parts)
    ppm = f"{cal.px_per_mm:.2f} px/mm" if cal.is_metric else "NO SCALE"
    hud = (f"VisionQC  parts {total}  PASS {c['PASS']}  FAIL {c['FAIL']}  "
           f"UNKNOWN {c['UNKNOWN']}  |  {ppm} ({cal.source})  |  {report.elapsed_ms:.0f} ms")
    _label(out, hud, (10, int(30 * s) + 10), s * 0.8, (255, 255, 255), t, bg=(0, 0, 0))
    for i, note in enumerate(report.notes):
        _label(out, note, (10, int(30 * s) + 10 + (i + 1) * int(30 * s + 8)), s * 0.65, (0, 215, 255))
    return out
