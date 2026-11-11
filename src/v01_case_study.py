"""Render a reviewed held-out street scene with measured V0.1 predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import Image

from .analyze import DATA_ROOT


IMAGE_ID = "Austin_d0g3XZ59cg25iKdi1i8q4w_0"
ORDER = ("original", "mirror", "shift_left", "shift_right",
         "shift_up", "shift_down", "zoom_in", "zoom_out")
NAMES = {"original": "Original", "mirror": "Mirror", "shift_left": "Crop left",
         "shift_right": "Crop right", "shift_up": "Crop up", "shift_down": "Crop down",
         "zoom_in": "Zoom in", "zoom_out": "Zoom out"}
COLORS = {"tree_1": "#e24b34", "tree_2": "#1687a0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/pilot200_final/transformations.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/v0_1")
    args = parser.parse_args()
    views = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["image_id"] == IMAGE_ID:
                views[row["transformation"]] = row
    if set(ORDER) - set(views) or any(view["annotation_status"] != "visually_reviewed" for view in views.values()):
        raise ValueError("The chosen reviewed test scene is incomplete or changed")
    data = pd.read_csv(args.output / "metrics/e2_latent_trajectories.csv")
    data = data[(data.image_id == IMAGE_ID) & (data.seed == 42) & (data.stage == "encoder_31")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, name in zip(axes.flat, ORDER):
        view = views[name]
        with Image.open(view["image_path"]) as source:
            ax.imshow(source.convert("RGB"))
        for obj in view["objects"]:
            oid = obj["object_id"]
            color = COLORS[oid]
            x1, y1, x2, y2 = obj["bbox"]
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, max(10, y1-4), oid, fontsize=8, color="white", weight="bold",
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 2})
            subset = data[(data.object_id == oid) & (data.transformation == name)]
            if not {"real", "position_only"}.issubset(set(subset.condition)):
                raise ValueError(f"Missing predictions for {name}/{oid}")
            for condition, marker, size in (("real", "o", 54), ("position_only", "+", 90)):
                point = subset[subset.condition == condition].iloc[0]
                ax.scatter(point.latent_x * 400, point.latent_y * 300, marker=marker, s=size,
                           color=color, facecolors="none" if marker == "o" else color,
                           linewidths=2, zorder=5)
            ax.scatter((x1+x2)/2, (y1+y2)/2, marker="x", s=55, c=color, linewidths=2, zorder=6)
        ax.set_title(NAMES[name], fontsize=11)
        if name in ("shift_right", "zoom_in"):
            margin = min(views[name]["objects"][0]["bbox"][0],
                         400 - views[name]["objects"][0]["bbox"][2])
            ax.text(8, 285, f"Edge-sensitive: {margin:.1f} px", fontsize=9, color="#a92820",
                    weight="bold", bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none", "pad": 2})
        ax.set_xlim(0, 400)
        ax.set_ylim(300, 0)
        ax.axis("off")
    fig.suptitle("One reviewed held-out Austin street scene | two tracked trees", fontsize=15)
    fig.text(.5, .025, "Box / x: reviewed object and center     o: Encoder 31 decoded center     +: Token-position baseline",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.09, wspace=.04, hspace=.30)
    destination = args.output / "figures/fig_case_reviewed_austin.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
