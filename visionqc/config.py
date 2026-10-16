"""Configuration loading with sensible defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "parts.yaml"

_DEFAULTS: dict[str, Any] = {
    "calibration": {
        "marker_mm": 30.0,
        "aruco_dict": "DICT_4X4_50",
        "rectify": True,
        "fallback_px_per_mm": None,
    },
    "segmentation": {
        "blur_ksize": 5,
        "morph_kernel": 5,
        "polarity": "auto",
        "min_part_area_mm2": 60.0,
        "min_hole_area_mm2": 4.0,
    },
    "inspection": {
        "max_chip_depth_mm": 0.8,
        "unknown_size_ratio": 0.20,
    },
    "parts": {},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and fill in defaults for anything missing."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    cfg = _merge(_DEFAULTS, user_cfg)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for name, part in cfg["parts"].items():
        shape = part.get("shape")
        if shape == "circle":
            required = ("diameter_mm", "tol_mm")
        elif shape == "rect":
            required = ("length_mm", "width_mm", "tol_mm")
        else:
            raise ValueError(f"part '{name}': shape must be 'circle' or 'rect', got {shape!r}")
        missing = [k for k in required if k not in part]
        if missing:
            raise ValueError(f"part '{name}' is missing {missing}")
        part.setdefault("holes", {"count": 0})
    if cfg["segmentation"]["polarity"] not in ("auto", "bright_parts", "dark_parts"):
        raise ValueError("segmentation.polarity must be auto | bright_parts | dark_parts")
