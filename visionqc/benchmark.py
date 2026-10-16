"""Score the pipeline against synthetic ground truth."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .pipeline import Inspector
from .synth import random_scene, tilt


def _map_points(pts: np.ndarray, *Hs: np.ndarray | None) -> np.ndarray:
    out = pts.reshape(1, -1, 2).astype(np.float64)
    for H in Hs:
        if H is not None:
            out = cv2.perspectiveTransform(out, H)
    return out.reshape(-1, 2)


def run_benchmark(cfg: dict[str, Any], n_scenes: int = 50, seed: int = 0,
                  tilt_strength: float = 0.0, match_px: float = 15.0) -> dict[str, Any]:
    inspector = Inspector(cfg)
    stats: dict[str, Any] = defaultdict(int)
    errors: dict[str, list[float]] = defaultdict(list)
    defect_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    times: list[float] = []

    for i in range(n_scenes):
        img, truth = random_scene(cfg, seed=seed + i)
        H_tilt = None
        if tilt_strength > 0:
            img, H_tilt = tilt(img, tilt_strength, seed=seed + i)
        rep = inspector.inspect(img)
        times.append(rep.elapsed_ms)
        if not rep.calibration.is_metric:
            stats["scenes_uncalibrated"] += 1
            continue

        det_centers = np.array([p.measurement.center_px if p.measurement else
                                cv2.minEnclosingCircle(p.contour)[0] for p in rep.parts], dtype=np.float64)
        gt_centers = _map_points(np.array([t["center_px"] for t in truth]), H_tilt, rep.calibration.homography)
        used: set[int] = set()

        for t, c in zip(truth, gt_centers):
            stats["parts"] += 1
            if len(det_centers) == 0:
                stats["missed"] += 1
                continue
            dist = np.linalg.norm(det_centers - c, axis=1)
            j = int(np.argmin(dist))
            if dist[j] > match_px or j in used:
                stats["missed"] += 1
                continue
            used.add(j)
            p = rep.parts[j]
            stats["type_correct"] += int(p.part_type == t["kind"])
            stats["status_correct"] += int(p.status == t["expected_status"])
            if t["defect"]:
                defect_hits[t["defect"]][1] += 1
                defect_hits[t["defect"]][0] += int(p.status == "FAIL")
            else:
                stats["good_parts"] += 1
                stats["false_rejects"] += int(p.status != "PASS")

            m = p.measurement
            if m is None or p.part_type != t["kind"]:
                continue
            if "diameter_mm" in t and m.diameter_mm is not None:
                errors["diameter"].append(m.diameter_mm - t["diameter_mm"])
            if "length_mm" in t and m.length_mm is not None:
                errors["length"].append(m.length_mm - t["length_mm"])
                errors["width"].append(m.width_mm - t["width_mm"])
            if t.get("hole_diameter_mm") and len(m.hole_diameters_mm) == t["hole_count"]:
                errors["hole"].extend(d - t["hole_diameter_mm"] for d in m.hole_diameters_mm)

        stats["false_detections"] += len(rep.parts) - len(used)

    n = max(stats["parts"], 1)
    return {
        "scenes": n_scenes,
        "tilt_strength": tilt_strength,
        "parts": stats["parts"],
        "detection_rate": 1 - stats["missed"] / n,
        "false_detections": stats["false_detections"],
        "type_accuracy": stats["type_correct"] / n,
        "status_accuracy": stats["status_correct"] / n,
        "false_reject_rate": stats["false_rejects"] / max(stats["good_parts"], 1),
        "defect_recall": {k: v[0] / v[1] for k, v in sorted(defect_hits.items()) if v[1]},
        "defect_counts": {k: v[1] for k, v in sorted(defect_hits.items())},
        "error_mm": {k: {"bias": float(np.mean(v)), "mae": float(np.mean(np.abs(v))),
                         "max": float(np.max(np.abs(v))), "n": len(v)} for k, v in errors.items() if v},
        "ms_per_frame": {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95))},
        "scenes_uncalibrated": stats["scenes_uncalibrated"],
    }


def format_benchmark(r: dict[str, Any]) -> str:
    lines = [
        f"Scenes: {r['scenes']}   parts: {r['parts']}   tilt: {r['tilt_strength']}",
        f"Detection rate      {r['detection_rate']:.1%}   (false detections: {r['false_detections']})",
        f"Part-type accuracy  {r['type_accuracy']:.1%}",
        f"PASS/FAIL accuracy  {r['status_accuracy']:.1%}",
        f"False-reject rate   {r['false_reject_rate']:.1%}  (good parts wrongly failed)",
        "Defect recall:      " + ", ".join(f"{k} {v:.0%} (n={r['defect_counts'][k]})"
                                           for k, v in r["defect_recall"].items()),
        "Measurement error (mm):",
    ]
    for k, e in r["error_mm"].items():
        lines.append(f"  {k:<9} bias {e['bias']:+.3f}   MAE {e['mae']:.3f}   max {e['max']:.3f}   (n={e['n']})")
    lines.append(f"Latency: {r['ms_per_frame']['mean']:.1f} ms mean, {r['ms_per_frame']['p95']:.1f} ms p95")
    if r["scenes_uncalibrated"]:
        lines.append(f"WARNING: {r['scenes_uncalibrated']} scenes had no calibration")
    return "\n".join(lines)
