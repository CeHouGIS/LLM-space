"""Audit crop-edge effects without altering the original V0.1 results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .analyze import DATA_ROOT
from .v01 import write_csv


CASE_ID = "Austin_d0g3XZ59cg25iKdi1i8q4w_0"
THRESHOLDS = (0, 10, 20, 30)


def edge_margin(box, output_size):
    x1, y1, x2, y2 = box
    width, height = output_size
    return min(x1, y1, width - x2, height - y2)


def audit_rows(manifest):
    views = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    original = {(view["image_id"], obj["object_id"]): edge_margin(obj["bbox"], view["output_size"])
                for view in views if view["transformation"] == "original" for obj in view["objects"]}
    rows = []
    for view in views:
        for obj in view["objects"]:
            key = view["image_id"], obj["object_id"]
            if key not in original:
                raise ValueError(f"Missing original object for {key}")
            margin = edge_margin(obj["bbox"], view["output_size"])
            rows.append({"scene_id": view["scene_id"], "image_id": view["image_id"],
                         "object_id": obj["object_id"], "category": obj["object_category"],
                         "split": view["split"], "annotation_status": view.get("annotation_status", "unspecified"),
                         "transformation": view["transformation"],
                         "crop_xyxy": json.dumps(view["crop_xyxy"]),
                         "bbox_xyxy": json.dumps(obj["bbox"]),
                         "edge_margin_px": margin, "original_edge_margin_px": original[key],
                         "pass_20px_both": int(min(margin, original[key]) >= 20)})
    return rows, views


def scene_ci(frame, metric, seed, repeats=300):
    groups = [part[metric].to_numpy() for _, part in frame.groupby("scene_id")]
    if len(groups) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(len(groups), size=len(groups))
        values.append(np.concatenate([groups[i] for i in indices]).mean())
    return np.percentile(values, [2.5, 97.5])


def sensitivity(audit, trajectories):
    columns = ["image_id", "object_id", "transformation", "edge_margin_px", "original_edge_margin_px"]
    merged = trajectories.merge(audit[columns], on=["image_id", "object_id", "transformation"],
                                how="left", validate="many_to_one")
    if merged["edge_margin_px"].isna().any():
        raise ValueError("Trajectory rows lack matching crop audit records")
    results = []
    for threshold in THRESHOLDS:
        filtered = merged[(merged["edge_margin_px"] >= threshold) &
                          (merged["original_edge_margin_px"] >= threshold) &
                          (merged["transformation"] != "original")]
        for (stage, seed, condition, transform), part in filtered.groupby(
                ["stage", "seed", "condition", "transformation"], sort=True):
            lo, hi = scene_ci(part, "equivariance_error", seed)
            results.append({"margin_threshold_px": threshold, "stage": stage, "seed": seed,
                            "condition": condition, "transformation": transform,
                            "n_objects": len(part), "n_scenes": part["scene_id"].nunique(),
                            "equivariance_mae": part["equivariance_error"].mean(),
                            "equiv_ci_low": lo, "equiv_ci_high": hi,
                            "coordinate_mae": part["coordinate_error"].mean(),
                            "direction_consistency": part["direction_consistent"].mean()})
    return results


def crop_figure(views, destination):
    chosen = {view["transformation"]: view for view in views if view["image_id"] == CASE_ID}
    base = chosen["original"]
    fig, ax = plt.subplots(figsize=(7, 6))
    with Image.open(base["source_image_path"]) as image:
        ax.imshow(image.convert("RGB"))
    for name, color, label in (
            ("original", "#1b8a8f", "Original crop"),
            ("shift_right", "#d44c35", "Crop window right"),
            ("zoom_in", "#9470aa", "Zoom-in crop")):
        left, top, right, bottom = chosen[name]["crop_xyxy"]
        ax.add_patch(Rectangle((left, top), right-left, bottom-top,
                               fill=False, edgecolor=color, linewidth=2.5, label=label))
    crop = base["crop_xyxy"]
    out_w, out_h = base["output_size"]
    for obj in base["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        source_box = (crop[0] + x1 * (crop[2]-crop[0]) / out_w,
                      crop[1] + y1 * (crop[3]-crop[1]) / out_h,
                      crop[0] + x2 * (crop[2]-crop[0]) / out_w,
                      crop[1] + y2 * (crop[3]-crop[1]) / out_h)
        ax.add_patch(Rectangle(source_box[:2], source_box[2]-source_box[0],
                               source_box[3]-source_box[1], fill=False,
                               edgecolor="#242424", linewidth=1.5, linestyle="--"))
        ax.text(source_box[0], source_box[1]-3, obj["object_id"], fontsize=8,
                color="#242424", weight="bold")
    ax.set(xlim=(0, 400), ylim=(300, 0), title="Austin source image: crop windows and reviewed object boxes")
    ax.legend(loc="lower right", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/pilot200_final/transformations.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/v0_1")
    args = parser.parse_args()
    records, views = audit_rows(args.manifest)
    audit = pd.DataFrame(records)
    trajectories = pd.read_csv(args.output / "metrics/e2_latent_trajectories.csv")
    sensitivity_rows = sensitivity(audit, trajectories)
    write_csv(args.output / "metrics/crop_edge_audit.csv", records)
    write_csv(args.output / "metrics/e2_edge_margin_sensitivity.csv", sensitivity_rows)
    crop_figure(views, args.output / "figures/fig_crop_windows_austin.png")
    print(json.dumps({"all_object_views": len(audit), "test_object_views": int((audit.split == "test").sum()),
                      "test_views_below_20px": int(((audit.split == "test") & (audit.edge_margin_px < 20)).sum())}))


if __name__ == "__main__":
    main()
