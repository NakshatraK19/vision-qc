"""End-to-end inspection pipeline: calibrate -> segment -> measure -> classify -> check."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .annotate import draw_report
from .calibration import Calibration, MarkerCalibrator, valid_region_mask
from .rules import check, classify
from .vision import Measurement, measure, segment

PASS, FAIL, UNKNOWN, PARTIAL, UNCALIBRATED = "PASS", "FAIL", "UNKNOWN", "PARTIAL", "UNCALIBRATED"


@dataclass
class PartResult:
    id: int
    part_type: str
    status: str
    defects: list[str]
    measurement: Measurement | None
    contour: np.ndarray
    holes: list[np.ndarray]

    def to_row(self) -> dict[str, Any]:
        m = self.measurement
        return {
            "id": self.id,
            "part_type": self.part_type,
            "status": self.status,
            "shape": m.shape if m else "",
            "diameter_mm": _r(m.diameter_mm) if m else None,
            "length_mm": _r(m.length_mm) if m else None,
            "width_mm": _r(m.width_mm) if m else None,
            "holes": len(m.hole_diameters_mm) if m else None,
            "hole_dia_mm": ";".join(f"{d:.2f}" for d in m.hole_diameters_mm) if m else "",
            "chip_depth_mm": _r(m.chip_depth_mm) if m else None,
            "center_x_px": _r(m.center_px[0], 1) if m else None,
            "center_y_px": _r(m.center_px[1], 1) if m else None,
            "defects": " | ".join(self.defects),
        }


@dataclass
class InspectionReport:
    image: np.ndarray                  # working (possibly rectified) image
    annotated: np.ndarray
    binary: np.ndarray
    calibration: Calibration
    parts: list[PartResult]
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {PASS: 0, FAIL: 0, UNKNOWN: 0, PARTIAL: 0, UNCALIBRATED: 0}
        for p in self.parts:
            out[p.status] = out.get(p.status, 0) + 1
        return out

    def rows(self) -> list[dict[str, Any]]:
        return [p.to_row() for p in self.parts]


class Inspector:
    """Stateless for single images; with persist_calibration=True it reuses the last
    marker calibration for a few frames (useful on a live feed when a hand covers the marker)."""

    def __init__(self, cfg: dict[str, Any], persist_calibration: bool = False, max_stale_frames: int = 60):
        self.cfg = cfg
        cal = cfg["calibration"]
        self.calibrator = MarkerCalibrator(cal["marker_mm"], cal["aruco_dict"])
        self.persist = persist_calibration
        self.max_stale = max_stale_frames
        self._last: dict[str, Any] | None = None
        self._stale = 0

    # ------------------------------------------------------------------ public
    def inspect(self, image: np.ndarray) -> InspectionReport:
        t0 = time.perf_counter()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cal_cfg, seg_cfg, ins_cfg = self.cfg["calibration"], self.cfg["segmentation"], self.cfg["inspection"]

        work, cal = self.calibrator.calibrate(image, rectify=cal_cfg["rectify"],
                                              fallback_px_per_mm=cal_cfg["fallback_px_per_mm"])
        work, cal = self._maybe_reuse_calibration(image, work, cal)

        valid = valid_region_mask(work.shape[:2], cal.homography, image.shape[:2])
        notes = list(cal.notes)

        if not cal.is_metric:
            # Still segment (in a nominal 1 px/mm space) so the operator sees *something*.
            blobs, binary = segment(work, valid, 1.0, seg_cfg["blur_ksize"], seg_cfg["morph_kernel"],
                                    seg_cfg["polarity"], 400.0, 25.0)
            parts = [PartResult(i + 1, "-", UNCALIBRATED, ["no scale reference"], None, b.contour, b.holes)
                     for i, b in enumerate(blobs)]
        else:
            ppm = cal.px_per_mm
            blobs, binary = segment(work, valid, ppm, seg_cfg["blur_ksize"], seg_cfg["morph_kernel"],
                                    seg_cfg["polarity"], seg_cfg["min_part_area_mm2"],
                                    seg_cfg["min_hole_area_mm2"], exclude_quad=cal.marker_quad)
            blobs.sort(key=lambda b: (cv2.boundingRect(b.contour)[1] // 50, cv2.boundingRect(b.contour)[0]))
            parts = [self._evaluate(i + 1, b, ppm, ins_cfg, binary) for i, b in enumerate(blobs)]

        elapsed = (time.perf_counter() - t0) * 1000
        report = InspectionReport(work, work, binary, cal, parts, elapsed, notes)
        report.annotated = draw_report(work, report)
        return report

    # ------------------------------------------------------------------ helpers
    def _evaluate(self, idx: int, blob, ppm: float, ins_cfg: dict[str, Any], binary: np.ndarray) -> PartResult:
        if blob.partial:
            return PartResult(idx, "-", PARTIAL, ["part cut off by frame edge"], None, blob.contour, blob.holes)
        meas = measure(blob, ppm, binary)
        name, _ = classify(meas, self.cfg["parts"], ins_cfg["unknown_size_ratio"])
        if name is None:
            return PartResult(idx, "unknown", UNKNOWN, ["does not match any catalogue part"],
                              meas, blob.contour, blob.holes)
        defects = check(meas, self.cfg["parts"][name], ins_cfg["max_chip_depth_mm"])
        return PartResult(idx, name, FAIL if defects else PASS, defects, meas, blob.contour, blob.holes)

    def _maybe_reuse_calibration(self, image, work, cal: Calibration):
        if not self.persist:
            return work, cal
        if cal.source.startswith("marker"):
            self._last = {"cal": cal, "shape": image.shape[:2],
                          "size": (work.shape[1], work.shape[0])}
            self._stale = 0
            return work, cal
        if self._last and self._stale < self.max_stale and self._last["shape"] == image.shape[:2]:
            self._stale += 1
            last: Calibration = self._last["cal"]
            if last.homography is not None:
                work = cv2.warpPerspective(image, last.homography, self._last["size"])
            reused = Calibration(last.px_per_mm, last.source + " (held)", last.marker_quad,
                                 last.homography, notes=[f"marker hidden - holding calibration ({self._stale})"])
            return work, reused
        return work, cal


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(float(v), nd)
