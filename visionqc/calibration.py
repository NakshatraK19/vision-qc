"""Metric calibration from an ArUco marker of known size.

The marker gives us two things:
  1. scale  - pixels per millimetre, from the marker's side length in pixels;
  2. pose   - a homography that maps the (possibly tilted) image plane to a true
              top-down view, so measurements stay correct when the camera is not
              perfectly perpendicular to the inspection surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Refuse to rectify if the corrected image would be absurdly large (extreme tilt / bad detection).
_MAX_RECTIFY_AREA_RATIO = 3.0


@dataclass
class Calibration:
    px_per_mm: float | None
    source: str                               # "marker", "marker+rectified", "fallback", "none"
    marker_quad: np.ndarray | None = None     # 4x2 marker corners in the *working* image
    homography: np.ndarray | None = None      # original -> working image (None if not rectified)
    notes: list[str] = field(default_factory=list)

    @property
    def is_metric(self) -> bool:
        return self.px_per_mm is not None and self.px_per_mm > 0


class MarkerCalibrator:
    def __init__(self, marker_mm: float, dict_name: str = "DICT_4X4_50"):
        if not hasattr(cv2.aruco, dict_name):
            raise ValueError(f"unknown ArUco dictionary {dict_name!r}")
        self.marker_mm = float(marker_mm)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Return the 4x2 corners (TL, TR, BR, BL) of the largest marker, or None."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # The detector misses markers whose quiet zone touches the frame edge; a dark pad fixes that.
        pad = 32
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        corners, ids, _ = self._detector.detectMarkers(padded)
        if ids is None or len(corners) == 0:
            return None
        quads = [c.reshape(4, 2).astype(np.float64) - pad for c in corners]
        return max(quads, key=lambda q: cv2.contourArea(q.astype(np.float32)))

    @staticmethod
    def side_px(quad: np.ndarray) -> float:
        return float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))

    def calibrate(
        self,
        image: np.ndarray,
        rectify: bool = True,
        fallback_px_per_mm: float | None = None,
    ) -> tuple[np.ndarray, Calibration]:
        """Return (working_image, calibration). The working image is rectified when possible."""
        quad = self.detect(image)
        if quad is None:
            if fallback_px_per_mm:
                return image, Calibration(float(fallback_px_per_mm), "fallback",
                                          notes=["no marker found - using fallback scale"])
            return image, Calibration(None, "none", notes=["no marker found and no fallback scale"])

        side = self.side_px(quad)
        ppm = side / self.marker_mm
        if not rectify:
            return image, Calibration(ppm, "marker", marker_quad=quad)

        warped = self._rectify(image, quad, side)
        if warped is None:
            return image, Calibration(ppm, "marker", marker_quad=quad,
                                      notes=["rectification skipped (extreme perspective)"])
        out, H = warped
        quad_w = cv2.perspectiveTransform(quad.reshape(1, 4, 2), H).reshape(4, 2)
        return out, Calibration(ppm, "marker+rectified", marker_quad=quad_w, homography=H)

    @staticmethod
    def _rectify(image: np.ndarray, quad: np.ndarray, side: float):
        """Warp so the marker becomes an axis-aligned square of `side` pixels.

        Keeping roughly the same resolution means px_per_mm == side / marker_mm holds
        everywhere on the inspection plane after warping.
        """
        h, w = image.shape[:2]
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float64)
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32)).astype(np.float64)

        img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(img_corners, H).reshape(4, 2)
        min_xy = mapped.min(axis=0)
        max_xy = mapped.max(axis=0)
        out_w, out_h = np.ceil(max_xy - min_xy).astype(int)
        if out_w <= 0 or out_h <= 0 or out_w * out_h > _MAX_RECTIFY_AREA_RATIO * w * h:
            return None

        T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=np.float64)
        H = T @ H
        warped = cv2.warpPerspective(image, H, (int(out_w), int(out_h)),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return warped, H


def valid_region_mask(shape: tuple[int, int], homography: np.ndarray | None,
                      src_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Mask (255 = real image data) of the working image; excludes warp padding."""
    h, w = shape
    if homography is None or src_shape is None:
        return np.full((h, w), 255, np.uint8)
    sh, sw = src_shape
    ones = np.full((sh, sw), 255, np.uint8)
    return cv2.warpPerspective(ones, homography, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
