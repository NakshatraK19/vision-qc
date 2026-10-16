"""Interactive Streamlit dashboard for VisionQC."""

from __future__ import annotations

import copy
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from visionqc import Inspector, load_config
from visionqc.synth import random_scene, tilt

st.set_page_config(page_title="VisionQC — Industrial Machine Vision", layout="wide")

st.title("VisionQC — Automated Optical Inspection & Metrology")
st.caption(
    "ArUco homography calibration • Sub-pixel Kåsa circle & Huber edge fitting • "
    "Convexity-defect chip localization • Real-time PASS/FAIL tolerance gating"
)

base_cfg = load_config()
cfg = copy.deepcopy(base_cfg)

with st.sidebar:
    st.header("1. Scene Source")
    mode = st.radio("Input mode", ["Synthetic Conveyor Generator", "Upload Image", "Webcam Snapshot"])

    if mode == "Synthetic Conveyor Generator":
        seed = st.slider("Scene Seed", 0, 200, 7)
        n_parts = st.slider("Parts Count", 1, 10, 7)
        defect_rate = st.slider("Defect Rate", 0.0, 1.0, 0.45, 0.05)
        tilt_strength = st.slider("Camera Tilt (Perspective)", 0.0, 0.10, 0.03, 0.01)
        noise_sigma = st.slider("Sensor Noise (σ)", 0.0, 12.0, 4.0, 0.5)
        lighting = st.slider("Lighting Gradient", 0.0, 0.50, 0.25, 0.05)

    st.header("2. Inspection Tolerances")
    rectify = st.checkbox("Perspective Rectification (ArUco Homography)", value=True)
    cfg["calibration"]["rectify"] = rectify
    max_chip = st.slider(
        "Max Edge Chip Depth (mm)",
        0.3,
        2.5,
        float(cfg["inspection"]["max_chip_depth_mm"]),
        0.1,
    )
    cfg["inspection"]["max_chip_depth_mm"] = max_chip

    tol_scale = st.slider("Dimensional Tolerance Multiplier", 0.5, 2.0, 1.0, 0.1)
    for p in cfg["parts"].values():
        p["tol_mm"] = round(p["tol_mm"] * tol_scale, 3)

image_bgr: np.ndarray | None = None
ground_truth = None

if mode == "Synthetic Conveyor Generator":
    img, ground_truth = random_scene(
        cfg,
        seed=seed,
        n_parts=n_parts,
        defect_rate=defect_rate,
    )
    # Re-render with user noise/lighting settings if adjusted
    from visionqc.synth import PlacedPart, render_scene

    placed = [
        PlacedPart(
            t["kind"],
            t["cx_mm"],
            t["cy_mm"],
            t["angle_deg"],
            t["defect"],
        )
        for t in ground_truth
    ]
    img, ground_truth = render_scene(
        placed,
        cfg,
        seed=seed,
        noise_sigma=noise_sigma,
        lighting=lighting,
    )
    if tilt_strength > 0:
        img, _ = tilt(img, strength=tilt_strength, seed=seed)
    image_bgr = img

elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload conveyor image (.png, .jpg)", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif mode == "Webcam Snapshot":
    cam_file = st.camera_input("Capture frame (place a 30mm ArUco DICT_4X4_50 marker in view)")
    if cam_file is not None:
        file_bytes = np.frombuffer(cam_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.info("Select or capture an image to run inspection.")
    st.stop()

inspector = Inspector(cfg)
report = inspector.inspect(image_bgr)
counts = report.counts

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Parts Detected", len(report.parts))
m2.metric("PASS", counts["PASS"])
m3.metric("FAIL", counts["FAIL"])
m4.metric(
    "Scale Calibration",
    f"{report.calibration.px_per_mm:.2f} px/mm" if report.calibration.is_metric else "Uncalibrated",
)
m5.metric("Frame Latency", f"{report.elapsed_ms:.1f} ms ({1000.0 / max(report.elapsed_ms, 1e-3):.0f} FPS)")

tab_overlay, tab_compare, tab_mask = st.tabs(
    ["Annotated Inspection View", "Raw vs. Rectified", "Binary Segmentation Mask"]
)

with tab_overlay:
    st.image(cv2.cvtColor(report.annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

with tab_compare:
    c1, c2 = st.columns(2)
    c1.subheader("Raw Camera Input")
    c1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    c2.subheader("Rectified Working Plane + Overlay")
    c2.image(cv2.cvtColor(report.annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

with tab_mask:
    st.image(report.binary, caption="Otsu + Morphological Opening/Closing Mask", use_container_width=True)

st.subheader("Per-Part Metrology & Defect Report")
df = pd.DataFrame(report.rows())
if not df.empty:
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Inspection CSV", data=csv_bytes, file_name="visionqc_report.csv", mime="text/csv")
else:
    st.warning("No parts segmented in frame.")

with st.expander("Part Catalogue Specification (config/parts.yaml)"):
    st.json(cfg["parts"])
