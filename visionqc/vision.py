"""Foreground segmentation and per-part geometric measurement."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _hole_area_px(hole: np.ndarray, binary: np.ndarray | None) -> float:
    """Background pixels enclosed by a hole contour (falls back to a Steiner-corrected polygon)."""
    if binary is None:
        return max(cv2.contourArea(hole) - 0.5 * cv2.arcLength(hole, True) + np.pi / 4, 0.0)
    x, y, w, h = cv2.boundingRect(hole)
    roi = binary[y:y + h, x:x + w]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [hole - np.array([x, y])], -1, 255, -1)
    return float(np.count_nonzero((mask > 0) & (roi == 0)))


def _fit_circle(pts: np.ndarray, iters: int = 4, reject_px: float = 1.5):
    """Least-squares (Kasa) circle fit; iteratively drops points far from the circle (chips)."""
    keep = np.ones(len(pts), bool)
    cx = cy = r = 0.0
    for _ in range(iters):
        x, y = pts[keep, 0], pts[keep, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x * x + y * y
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
        resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        new_keep = resid < reject_px
        if new_keep.sum() < max(8, 0.5 * len(pts)) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return (float(cx), float(cy)), r


def _fit_rect(pts: np.ndarray, corner_margin: float = 0.12, reject_px: float = 1.5):
    """Fit a line to each of the four edges; return (length, width) in px incl. the 0.5 px offset."""
    box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))).astype(np.float64)
    lines = []
    dists = []
    for i in range(4):
        p0, p1 = box[i], box[(i + 1) % 4]
        seg = p1 - p0
        L = np.linalg.norm(seg)
        u = seg / max(L, 1e-9)
        n = np.array([-u[1], u[0]])
        rel = pts - p0
        dists.append(np.abs(rel @ n))
        lines.append((p0, u, n, L))
    side = np.argmin(np.stack(dists), axis=0)

    fitted = []
    for i, (p0, u, n, L) in enumerate(lines):
        rel = pts - p0
        t = rel @ u
        sel = (side == i) & (t > corner_margin * L) & (t < (1 - corner_margin) * L)
        sp = pts[sel]
        if len(sp) < 6:
            return None
        for _ in range(3):
            vx, vy, x0, y0 = cv2.fitLine(sp.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            nn = np.array([-vy, vx], dtype=np.float64)
            res = np.abs((sp - [x0, y0]) @ nn)
            ok = res < reject_px
            if ok.all() or ok.sum() < 6:
                break
            sp = sp[ok]
        fitted.append((np.array([x0, y0], dtype=np.float64), nn, sp.mean(axis=0)))

    def gap(a, b):
        (pa, na, ma), (pb, nb, mb) = fitted[a], fitted[b]
        return 0.5 * (abs((ma - pb) @ nb) + abs((mb - pa) @ na)) + 1.0   # +0.5 px per edge

    d02, d13 = gap(0, 2), gap(1, 3)
    return max(d02, d13), min(d02, d13)


@dataclass
class Blob:
    """One segmented part with its holes, in working-image pixels."""
    contour: np.ndarray
    holes: list[np.ndarray]
    partial: bool = False          # touches the image / warp edge -> cannot be measured reliably


@dataclass
class Measurement:
    shape: str                     # "circle" | "rect"
    center_px: tuple[float, float]
    area_mm2: float
    diameter_mm: float | None = None
    length_mm: float | None = None
    width_mm: float | None = None
    angle_deg: float = 0.0
    hole_diameters_mm: list[float] = field(default_factory=list)
    hole_centers_px: list[tuple[float, float]] = field(default_factory=list)
    chip_depth_mm: float = 0.0
    chip_point_px: tuple[int, int] | None = None
    extent: float = 0.0            # area / min-area-rect area (rect ~1.0, circle ~0.785)
    circularity: float = 0.0       # 4*pi*A / P^2


# --------------------------------------------------------------------------- segmentation

def segment(
    image: np.ndarray,
    valid_mask: np.ndarray,
    px_per_mm: float,
    blur_ksize: int = 5,
    morph_kernel: int = 5,
    polarity: str = "auto",
    min_part_area_mm2: float = 60.0,
    min_hole_area_mm2: float = 4.0,
    exclude_quad: np.ndarray | None = None,
) -> tuple[list[Blob], np.ndarray]:
    """Threshold the image into parts and holes. Returns (blobs, binary_mask)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = max(1, int(blur_ksize) | 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    # Otsu only on real pixels so warp padding doesn't skew the threshold.
    valid_px = blurred[valid_mask > 0]
    thr, _ = cv2.threshold(valid_px.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = np.where(blurred > thr, 255, 0).astype(np.uint8)

    if polarity == "dark_parts" or (polarity == "auto" and _border_is_bright(binary, valid_mask)):
        binary = cv2.bitwise_not(binary)
    binary = cv2.bitwise_and(binary, valid_mask)

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Pixels within 2 px of the image / warp edge: a part touching them is cut off.
    framed = valid_mask.copy()
    framed[[0, -1], :] = 0
    framed[:, [0, -1]] = 0
    safe = cv2.erode(framed, np.ones((5, 5), np.uint8))

    min_part_px = min_part_area_mm2 * px_per_mm ** 2
    min_hole_px = min_hole_area_mm2 * px_per_mm ** 2
    exclusion = _expand_quad(exclude_quad, 1.6) if exclude_quad is not None else None

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    blobs: list[Blob] = []
    if hierarchy is None:
        return blobs, binary
    hierarchy = hierarchy[0]
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:            # skip holes here; collected via their parent
            continue
        if cv2.contourArea(cnt) < min_part_px:
            continue
        if exclusion is not None and _centroid_inside(cnt, exclusion):
            continue                         # the calibration marker itself
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_hole_px:
                holes.append(contours[child])
            child = hierarchy[child][0]
        pts = cnt.reshape(-1, 2)
        partial = bool(np.any(safe[pts[:, 1], pts[:, 0]] == 0))
        blobs.append(Blob(contour=cnt, holes=holes, partial=partial))
    return blobs, binary


def _border_is_bright(binary: np.ndarray, valid_mask: np.ndarray) -> bool:
    """If most of the image border is foreground, the parts are darker than the background."""
    edge = cv2.morphologyEx(valid_mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
    edge[[0, -1], :] = valid_mask[[0, -1], :]
    edge[:, [0, -1]] = valid_mask[:, [0, -1]]
    sel = edge > 0
    return bool(sel.any() and binary[sel].mean() > 127)


def _expand_quad(quad: np.ndarray, factor: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return ((quad - c) * factor + c).astype(np.float32)


def _centroid_inside(cnt: np.ndarray, quad: np.ndarray) -> bool:
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return False
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0


# --------------------------------------------------------------------------- measurement

def measure(blob: Blob, px_per_mm: float, binary: np.ndarray | None = None,
            rect_extent_threshold: float = 0.88) -> Measurement:
    """Measure one part with sub-pixel, outlier-robust geometric fits.

    Contour vertices sit on the *centres* of boundary pixels: on average 0.5 px inside the
    real edge for the outer boundary and 0.5 px outside it for holes. At 5 px/mm that is a
    0.2 mm diameter error - half a typical tolerance - so every fit is offset by 0.5 px.

      * circles / holes: iterative least-squares circle fit, rejecting points >1.5 px off
        the circle (that is how a chip is ignored when measuring diameter);
      * rectangles: one robust line fit per edge, size = distance between opposite edges.
    """
    cnt = blob.contour
    area_px = cv2.contourArea(cnt)
    perim_px = cv2.arcLength(cnt, True)
    (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(cnt)
    extent = area_px / max(rw * rh, 1e-9)
    circularity = 4 * np.pi * area_px / max(perim_px ** 2, 1e-9)

    m = cv2.moments(cnt)
    center = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (rcx, rcy)
    shape = "rect" if extent >= rect_extent_threshold else "circle"

    meas = Measurement(
        shape=shape,
        center_px=(float(center[0]), float(center[1])),
        area_mm2=(area_px + perim_px * 0.5) / px_per_mm ** 2,
        extent=float(extent),
        circularity=float(circularity),
    )

    pts = cnt.reshape(-1, 2).astype(np.float64)
    if shape == "circle":
        _, r = _fit_circle(pts)
        meas.diameter_mm = float(2 * (r + 0.5)) / px_per_mm
    else:
        dims = _fit_rect(pts)
        if dims is None:                               # degenerate contour: fall back to the box
            dims = (max(rw, rh) + 1.0, min(rw, rh) + 1.0)
        meas.length_mm = float(dims[0]) / px_per_mm
        meas.width_mm = float(dims[1]) / px_per_mm
        meas.angle_deg = float(angle if rw >= rh else angle + 90.0) % 180.0

    for hole in blob.holes:
        hpts = hole.reshape(-1, 2).astype(np.float64)
        if len(hpts) >= 8:
            (hx, hy), hr = _fit_circle(hpts)
            meas.hole_diameters_mm.append(float(2 * (hr - 0.5)) / px_per_mm)
            meas.hole_centers_px.append((float(hx), float(hy)))
        else:
            a = _hole_area_px(hole, binary)
            meas.hole_diameters_mm.append(float(np.sqrt(4 * a / np.pi)) / px_per_mm)

    depth_px, point = _deepest_convexity_defect(cnt)
    meas.chip_depth_mm = depth_px / px_per_mm
    meas.chip_point_px = point
    return meas


def _deepest_convexity_defect(cnt: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    """Depth (px) and location of the deepest dent in the outer edge (a chip / notch)."""
    try:
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 4:
            return 0.0, None
        defects = cv2.convexityDefects(cnt, np.sort(hull, axis=0))
    except cv2.error:
        return 0.0, None
    if defects is None or len(defects) == 0:
        return 0.0, None
    defects = np.asarray(defects).reshape(-1, 4)   # OpenCV 4: (N,1,4); OpenCV 5: (N,4)
    idx = int(np.argmax(defects[:, 3]))
    _, _, far, depth_fixed = defects[idx]
    x, y = cnt[int(far)][0]
    return depth_fixed / 256.0, (int(x), int(y))
