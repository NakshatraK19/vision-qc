"""Command-line interface.

    python -m visionqc generate  --out samples --n 8
    python -m visionqc inspect   samples --out results
    python -m visionqc live      --camera 0
    python -m visionqc marker    --out marker_30mm.png
    python -m visionqc benchmark --scenes 100 --tilt 0.06
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import load_config
from .pipeline import Inspector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cmd_generate(args, cfg):
    from .synth import random_scene, tilt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img, truth = random_scene(cfg, seed=args.seed + i)
        if args.tilt:
            img, _ = tilt(img, args.tilt, seed=args.seed + i)
        name = f"scene_{i:02d}"
        cv2.imwrite(str(out / f"{name}.png"), img)
        (out / f"{name}.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {args.n} scenes (+ ground-truth JSON) to {out}/")


def cmd_inspect(args, cfg):
    src = Path(args.path)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    if not files:
        sys.exit(f"no images found in {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inspector = Inspector(cfg)
    rows, totals = [], {"PASS": 0, "FAIL": 0, "UNKNOWN": 0, "PARTIAL": 0, "UNCALIBRATED": 0}

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  skip {f.name}: unreadable")
            continue
        rep = inspector.inspect(img)
        cv2.imwrite(str(out / f"{f.stem}_inspected.png"), rep.annotated)
        for k, v in rep.counts.items():
            totals[k] += v
        for r in rep.rows():
            rows.append({"image": f.name, **r})
        c = rep.counts
        print(f"  {f.name:<22} parts {len(rep.parts):>2}  PASS {c['PASS']:>2}  FAIL {c['FAIL']:>2}  "
              f"UNKNOWN {c['UNKNOWN']}  {rep.elapsed_ms:6.1f} ms  [{rep.calibration.source}]")
        for p in rep.parts:
            if p.status != "PASS":
                print(f"      #{p.id} {p.part_type}: {p.status} - {'; '.join(p.defects)}")

    if rows:
        with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    total = sum(totals.values())
    print(f"\n{total} parts inspected: " + ", ".join(f"{k} {v}" for k, v in totals.items() if v))
    if total:
        print(f"yield (PASS / measured): {totals['PASS'] / max(totals['PASS'] + totals['FAIL'], 1):.1%}")
    print(f"annotated images + report.csv -> {out}/")


def cmd_live(args, cfg):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.no_rectify:
        cfg["calibration"]["rectify"] = False
    inspector = Inspector(cfg, persist_calibration=True)
    win = "VisionQC live  [q] quit  [s] snapshot  [b] binary view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_binary, fps, last = False, 0.0, time.perf_counter()
    snap_dir = Path(args.snapshots)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        rep = inspector.inspect(frame)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        view = cv2.cvtColor(rep.binary, cv2.COLOR_GRAY2BGR) if show_binary else rep.annotated.copy()
        cv2.putText(view, f"{fps:4.1f} FPS", (view.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("b"):
            show_binary = not show_binary
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(snap_dir / f"{stamp}_raw.png"), frame)
            cv2.imwrite(str(snap_dir / f"{stamp}_inspected.png"), rep.annotated)
            print(f"snapshot saved: {snap_dir / stamp}_*.png")
    cap.release()
    cv2.destroyAllWindows()


def cmd_marker(args, cfg):
    """Render a print-ready ArUco marker at the configured physical size."""
    mm = cfg["calibration"]["marker_mm"]
    dpi = args.dpi
    px = int(round(mm / 25.4 * dpi))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["calibration"]["aruco_dict"]))
    marker = cv2.aruco.generateImageMarker(d, args.id, px)
    margin = px // 2
    page = np.full((px + 2 * margin + 60, px + 2 * margin), 255, np.uint8)
    page[margin:margin + px, margin:margin + px] = marker
    cv2.putText(page, f"VisionQC ref  {cfg['calibration']['aruco_dict']} id {args.id}  side {mm:g} mm",
                (10, px + 2 * margin + 35), cv2.FONT_HERSHEY_SIMPLEX, dpi / 600, 0, 1, cv2.LINE_AA)
    out = Path(args.out)
    # Embed DPI so the printer reproduces the physical size (needs Pillow; falls back to OpenCV).
    try:
        from PIL import Image
        Image.fromarray(page).save(out, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(str(out), page)
    print(f"marker saved to {out} - print at 100% scale ({dpi} DPI) and verify the black square "
          f"measures {mm:g} mm. If it doesn't, set calibration.marker_mm to the measured value.")


def cmd_benchmark(args, cfg):
    from .benchmark import format_benchmark, run_benchmark
    res = run_benchmark(cfg, n_scenes=args.scenes, seed=args.seed, tilt_strength=args.tilt)
    print(format_benchmark(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="visionqc", description="Industrial part inspection with OpenCV")
    ap.add_argument("--config", help="path to parts.yaml (default: config/parts.yaml)")
    ap.add_argument("--version", action="version", version=f"visionqc {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render synthetic inspection scenes with ground truth")
    g.add_argument("--out", default="samples")
    g.add_argument("--n", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tilt", type=float, default=0.0, help="simulate camera tilt (0-0.12)")

    i = sub.add_parser("inspect", help="inspect an image or a folder of images")
    i.add_argument("path")
    i.add_argument("--out", default="results")

    lv = sub.add_parser("live", help="real-time inspection from a webcam")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--width", type=int, default=1280)
    lv.add_argument("--height", type=int, default=720)
    lv.add_argument("--no-rectify", action="store_true", help="skip perspective correction (faster)")
    lv.add_argument("--snapshots", default="snapshots")

    mk = sub.add_parser("marker", help="create a printable calibration marker")
    mk.add_argument("--out", default="marker.png")
    mk.add_argument("--id", type=int, default=7)
    mk.add_argument("--dpi", type=int, default=300)

    b = sub.add_parser("benchmark", help="score accuracy against synthetic ground truth")
    b.add_argument("--scenes", type=int, default=50)
    b.add_argument("--seed", type=int, default=1000)
    b.add_argument("--tilt", type=float, default=0.0)
    b.add_argument("--json", help="also write results to this JSON file")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"generate": cmd_generate, "inspect": cmd_inspect, "live": cmd_live,
     "marker": cmd_marker, "benchmark": cmd_benchmark}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
