"""Figures and compact summaries from measured V0.1 CSV files."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def line_compare(frame, path, title, metric, group="condition"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, part in frame.groupby(group):
        summary = part.groupby("transformation")[metric].mean().sort_index()
        ax.plot(summary.index, summary.values, marker="o", label=name)
    ax.set(title=title, ylabel=metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    save(fig, path)


def trajectory_panels(frame, path, title, transformations, cases=4):
    frame = frame[(frame.seed == 42) & (frame.stage == "encoder_31") & (frame.condition == "real")]
    frame = frame[frame.transformation.isin(["original"] + transformations)]
    groups = [(key, group) for key, group in frame.groupby(["image_id", "object_id"])
              if set(transformations).issubset(set(group.transformation))]
    groups.sort(key=lambda item: item[1].coordinate_error.mean())
    if not groups:
        return
    chosen = groups[:max(1, cases // 2)] + groups[-max(1, cases // 2):]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 3.8), squeeze=False)
    colors = {"original": "#222222", "shift_left": "#d05f42", "shift_right": "#de9a33",
              "shift_up": "#2d8b84", "shift_down": "#577abd", "mirror": "#c08a35",
              "zoom_in": "#6b63a8", "zoom_out": "#ad69a0"}
    for ax, (key, group) in zip(axes[0], chosen):
        group = group.sort_values("transformation")
        for _, row in group.iterrows():
            color = colors.get(row.transformation, "#555")
            ax.scatter(row.true_x, row.true_y, marker="x", color=color, s=55)
            ax.scatter(row.latent_x, row.latent_y, marker="o", facecolors="none", edgecolors=color, s=55)
            ax.plot([row.true_x, row.latent_x], [row.true_y, row.latent_y], color=color, alpha=.5,
                    label=row.transformation.replace("shift_", "") if ax is axes[0][0] else None)
        ax.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", title=f"{key[0][:14]} / {key[1]}", xlabel="x", ylabel="y")
    axes[0][0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(title + "  (x=true, o=decoded)", fontsize=11)
    save(fig, path)


def heatmap(frame, path, title, column, index, values):
    matrix = frame.pivot_table(index=index, columns=column, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, len(matrix.columns)), max(4, len(matrix.index) * .55)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    fig.colorbar(im, ax=ax, label=values)
    ax.set_title(title)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/v0_1")
    args = parser.parse_args()
    m, f = args.output / "metrics", args.output / "figures"
    f.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(m / "e2_latent_trajectories.csv")
    controls = pd.read_csv(m / "e1_position_leakage.csv")
    equivariance = pd.read_csv(m / "e2_equivariance_comparison.csv")
    transfer = pd.read_csv(m / "e3_category_transfer.csv")
    disentangle = pd.read_csv(m / "e4_disentanglement.csv")
    trajectory_panels(trajectory, f / "fig_e2_horizontal_vertical_trajectories.png",
                      "Horizontal and vertical object trajectories", ["shift_left", "shift_right", "shift_up", "shift_down"])
    trajectory_panels(trajectory, f / "fig_e2_mirror_transform.png", "Mirror object trajectories", ["mirror"])
    subset = controls[controls.subset.isna() & controls.condition.isin(["position_only", "real", "position_residual"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    for condition, part in subset.groupby("condition"):
        values = part.groupby("stage").coordinate_r2.mean()
        ax.plot(values.index, values.values, marker="o", label=condition)
    ax.set(title="Position-controlled residual decoding", ylabel="Held-out coordinate R2")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save(fig, f / "fig_e1_residual_coordinate_information.png")
    line_compare(equivariance[equivariance.stage == "encoder_31"],
                 f / "fig_e2_equivariance_controls.png", "Matched trajectory controls", "equivariance_mae")
    computed = transfer[(transfer.status == "computed") & (transfer.stage.isin(["encoder_31", "llm_14"]))]
    heatmap(computed[computed.condition == "real"], f / "fig_e3_transfer_matrix.png",
            "Unseen category MAE", "stage", "task", "unseen_mae")
    focus = disentangle[(disentangle.stage == "encoder_31") & (disentangle.variant == "projected")]
    heatmap(focus, f / "fig_e4_spatial_semantic_overlap.png", "Overlap with semantic probe subspace",
            "k", "subspace", "coord_semantic_overlap")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, part in focus.groupby("subspace"):
        summary = part.groupby("k")[["coordinate_r2", "category_balanced_accuracy"]].mean()
        ax.plot(summary.coordinate_r2, summary.category_balanced_accuracy, marker="o", label=name)
    ax.set(title="Spatial and semantic decoding tradeoff", xlabel="Coordinate R2",
           ylabel="Category balanced accuracy")
    ax.legend(fontsize=8)
    save(fig, f / "fig_e4_spatial_semantic_tradeoff.png")
    heatmap(focus, f / "fig_e4_cross_decoding.png", "Category balanced accuracy in candidate subspace",
            "k", "subspace", "category_balanced_accuracy")
    cross = disentangle[(disentangle.subspace == "coordinate") & (disentangle.variant == "projected")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, color in (("coordinate_r2", "#317d7a"), ("category_balanced_accuracy", "#c66f3b")):
        values = cross.groupby("stage")[metric].mean()
        ax.plot(values.index, values.values, marker="o", label=metric, color=color)
    ax.set(title="Cross-stage candidate subspace organization", ylabel="Score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save(fig, f / "fig_cross_stage_organization.png")
    composition = m / "e2_transformation_composition.csv"
    if composition.exists():
        rows = pd.read_csv(composition)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels, values = [], []
        for (stage, transform), part in rows.groupby(["stage", "transformation"]):
            labels.append(stage + "\n" + transform.replace("_then_", "+"))
            values.append(part.composition_mae.mean())
        ax.bar(labels, values, color="#c08a35")
        ax.set(title="Actual two-step image transformation", ylabel="Composition MAE")
        ax.tick_params(axis="x", rotation=40)
        save(fig, f / "fig_e2_transformation_composition.png")


if __name__ == "__main__":
    main()
