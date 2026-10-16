"""Match measured parts against the catalogue and apply tolerance / defect rules."""

from __future__ import annotations

from typing import Any

from .vision import Measurement


def classify(meas: Measurement, parts: dict[str, Any], unknown_ratio: float) -> tuple[str | None, float]:
    """Return (catalogue_name or None, relative size error) for the closest same-shape part."""
    best_name, best_err = None, float("inf")
    for name, spec in parts.items():
        if spec["shape"] != meas.shape:
            continue
        if meas.shape == "circle":
            err = abs(meas.diameter_mm - spec["diameter_mm"]) / spec["diameter_mm"]
        else:
            err = max(abs(meas.length_mm - spec["length_mm"]) / spec["length_mm"],
                      abs(meas.width_mm - spec["width_mm"]) / spec["width_mm"])
        # Hole count is a strong tie-breaker between similarly sized parts.
        if len(meas.hole_diameters_mm) != spec["holes"].get("count", 0):
            err += 0.05
        if err < best_err:
            best_name, best_err = name, err
    if best_name is None or best_err > unknown_ratio:
        return None, best_err
    return best_name, best_err


def check(meas: Measurement, spec: dict[str, Any], max_chip_depth_mm: float) -> list[str]:
    """Return a list of human-readable defects (empty list == PASS)."""
    defects: list[str] = []
    tol = spec["tol_mm"]

    if meas.shape == "circle":
        _dim(defects, "diameter", meas.diameter_mm, spec["diameter_mm"], tol)
    else:
        _dim(defects, "length", meas.length_mm, spec["length_mm"], tol)
        _dim(defects, "width", meas.width_mm, spec["width_mm"], tol)

    holes = spec.get("holes", {"count": 0})
    expected = holes.get("count", 0)
    found = len(meas.hole_diameters_mm)
    if found < expected:
        defects.append(f"missing hole: found {found}/{expected}")
    elif found > expected:
        defects.append(f"unexpected hole: found {found}/{expected}")
    if expected and "diameter_mm" in holes:
        for d in meas.hole_diameters_mm:
            if abs(d - holes["diameter_mm"]) > holes.get("tol_mm", tol):
                defects.append(f"hole dia {d:.2f} mm (spec {holes['diameter_mm']:.1f}+/-{holes.get('tol_mm', tol)})")

    if meas.chip_depth_mm > max_chip_depth_mm:
        defects.append(f"edge chip {meas.chip_depth_mm:.2f} mm deep (max {max_chip_depth_mm})")
    return defects


def _dim(defects: list[str], label: str, value: float, nominal: float, tol: float) -> None:
    if abs(value - nominal) > tol:
        kind = "oversize" if value > nominal else "undersize"
        defects.append(f"{label} {value:.2f} mm {kind} (spec {nominal:.1f}+/-{tol})")
