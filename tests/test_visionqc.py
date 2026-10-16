"""Automated test suite for the VisionQC inspection pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from visionqc import Inspector, load_config
from visionqc.synth import PlacedPart, render_scene, tilt


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_nominal_parts_pass_and_dimensions_are_accurate(cfg):
    placed = [
        PlacedPart("washer_m16", cx_mm=70.0, cy_mm=70.0, angle_deg=0.0, defect=None),
        PlacedPart("bracket_plate", cx_mm=155.0, cy_mm=65.0, angle_deg=25.0, defect=None),
        PlacedPart("spacer_disc", cx_mm=215.0, cy_mm=100.0, angle_deg=0.0, defect=None),
    ]
    img, _ = render_scene(placed, cfg, seed=10)
    report = Inspector(cfg).inspect(img)

    assert report.calibration.is_metric
    assert abs(report.calibration.px_per_mm - 5.0) < 0.08
    assert len(report.parts) == 3
    assert all(p.status == "PASS" for p in report.parts)

    by_type = {p.part_type: p.measurement for p in report.parts}
    assert abs(by_type["washer_m16"].diameter_mm - 40.0) < 0.25
    assert abs(by_type["washer_m16"].hole_diameters_mm[0] - 17.0) < 0.25
    assert abs(by_type["bracket_plate"].length_mm - 60.0) < 0.30
    assert abs(by_type["bracket_plate"].width_mm - 30.0) < 0.25
    assert abs(by_type["spacer_disc"].diameter_mm - 25.0) < 0.25


@pytest.mark.parametrize("defect_kind", ["chip", "missing_hole", "oversize", "undersize", "bad_hole"])
def test_each_defect_mode_is_caught(cfg, defect_kind):
    placed = [PlacedPart("washer_m16", cx_mm=120.0, cy_mm=80.0, angle_deg=15.0, defect=defect_kind)]
    img, _ = render_scene(placed, cfg, seed=42)
    report = Inspector(cfg).inspect(img)

    assert len(report.parts) == 1
    part = report.parts[0]
    assert part.part_type == "washer_m16"
    assert part.status == "FAIL"
    assert len(part.defects) >= 1


def test_perspective_tilt_rectification_preserves_pass(cfg):
    placed = [
        PlacedPart("washer_m16", cx_mm=90.0, cy_mm=80.0, defect=None),
        PlacedPart("spacer_disc", cx_mm=170.0, cy_mm=85.0, defect=None),
    ]
    img, _ = render_scene(placed, cfg, seed=99)
    tilted, _ = tilt(img, strength=0.06, seed=99)
    report = Inspector(cfg).inspect(tilted)

    assert report.calibration.source == "marker+rectified"
    assert len(report.parts) == 2
    assert all(p.status == "PASS" for p in report.parts)
