"""Independent scene-held-out linear probes and spatial/semantic comparisons."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .token_alignment import overlap_mask, token_boxes


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0")
SEEDS = (17, 42, 73)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def load_features(directory):
    data = defaultdict(list)
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["metadata_json"]))
            for stage in payload.files:
                if stage == "metadata_json":
                    continue
                features = payload[stage].astype(np.float32)
                for index, obj in enumerate(meta["objects"]):
                    data[stage].append({
                        "x": features[index], "y": np.asarray(obj["normalized_center"], dtype=np.float32),
                        "category": obj["object_category"], "object_id": obj["object_id"],
                        "image_id": meta["image_id"], "scene_id": meta["scene_id"],
                        "split": meta["split"], "transformation": meta["transformation"],
                        "crop": meta["crop_xyxy"], "mirror": meta["mirror"],
                        "bbox": obj["bbox"], "grid": meta["image_grid_thw"],
                        "image_size": meta["output_size"],
                    })
    if not data:
        raise ValueError(f"No cached features found in {directory}")
    for stage, rows in data.items():
        splits = defaultdict(set)
        for row in rows:
            splits[row["scene_id"]].add(row["split"])
        if any(len(groups) != 1 for groups in splits.values()):
            raise ValueError(f"Scene leakage in {stage}")
    return data


def filter_quality(data, min_original_box_side):
    if min_original_box_side <= 0:
        return data
    filtered = {}
    for stage, rows in data.items():
        valid_objects = {(row["image_id"], row["object_id"]): row["scene_id"] for row in rows
                         if row["transformation"] == "original" and
                         min(row["bbox"][2] - row["bbox"][0], row["bbox"][3] - row["bbox"][1])
                         >= min_original_box_side}
        scene_counts = Counter(valid_objects.values())
        valid_scenes = {scene for scene, count in scene_counts.items() if count >= 2}
        filtered[stage] = [row for row in rows
                           if row["scene_id"] in valid_scenes and
                           (row["image_id"], row["object_id"]) in valid_objects]
    return filtered


def xy(rows):
    return np.stack([row["x"] for row in rows]), np.stack([row["y"] for row in rows])


def preprocessor(train_x, components):
    if components <= 0:
        return StandardScaler().fit(train_x)
    count = min(components, len(train_x) - 1, train_x.shape[1])
    if count < 1:
        raise ValueError("Insufficient training examples for matched PCA probe")
    return make_pipeline(StandardScaler(), PCA(n_components=count, svd_solver="auto", random_state=42)).fit(train_x)


def fit_ridge(train, val, seed, components):
    train_x, train_y = xy(train)
    scaler = preprocessor(train_x, components)
    train_x = scaler.transform(train_x)
    val_x, val_y = xy(val)
    val_x = scaler.transform(val_x)
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene_id"] for row in train})
    sampled = rng.choice(scenes, size=len(scenes), replace=True)
    indices = np.concatenate([[i for i, row in enumerate(train) if row["scene_id"] == scene] for scene in sampled])
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(train_x[indices], train_y[indices])
        error = mean_absolute_error(val_y, model.predict(val_x))
        if best is None or error < best[0]:
            best = (error, alpha, model)
    return scaler, best[2], best[1]


def scores(y, predicted):
    return {
        "coordinate_mae": float(mean_absolute_error(y, predicted)),
        "coordinate_r2": float(r2_score(y, predicted)) if len(y) >= 2 else float("nan"),
        "x_r2": float(r2_score(y[:, 0], predicted[:, 0])) if len(y) >= 2 else float("nan"),
        "y_r2": float(r2_score(y[:, 1], predicted[:, 1])) if len(y) >= 2 else float("nan"),
    }


def bootstrap_ci(rows, predictions, metric, seed=42, repetitions=300):
    scenes = sorted({row["scene_id"] for row in rows})
    if len(scenes) < 2:
        return float("nan"), float("nan")
    groups = {scene: np.array([i for i, row in enumerate(rows) if row["scene_id"] == scene]) for scene in scenes}
    rng = np.random.default_rng(seed)
    values = []
    truth = np.stack([row["y"] for row in rows])
    for _ in range(repetitions):
        ids = np.concatenate([groups[scene] for scene in rng.choice(scenes, size=len(scenes), replace=True)])
        values.append(metric(truth[ids], predictions[ids]))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def scene_scalar_ci(rows, field, seed=42, repetitions=300):
    scenes = sorted({row["scene_id"] for row in rows})
    if len(scenes) < 2:
        return float("nan"), float("nan")
    grouped = {scene: [row[field] for row in rows if row["scene_id"] == scene] for scene in scenes}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        values = [value for scene in rng.choice(scenes, size=len(scenes), replace=True)
                  for value in grouped[scene]]
        samples.append(float(np.mean(values)))
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def map_prediction(point, original_crop, target_crop, mirror):
    source_x = original_crop[0] + point[0] * (original_crop[2] - original_crop[0])
    source_y = original_crop[1] + point[1] * (original_crop[3] - original_crop[1])
    x = (source_x - target_crop[0]) / (target_crop[2] - target_crop[0])
    y = (source_y - target_crop[1]) / (target_crop[3] - target_crop[1])
    return np.array([1 - x if mirror else x, y])


def patch_position_readout(row):
    _, h, w = row["grid"]
    rectangles = token_boxes(h, w, row["image_size"])
    selected = rectangles[overlap_mask(rectangles, row["bbox"])]
    if not len(selected):
        raise ValueError("Position baseline has no overlapping patch")
    width, height = row["image_size"]
    return np.array([(selected[:, 0] + selected[:, 2]).mean() / (2 * width),
                     (selected[:, 1] + selected[:, 3]).mean() / (2 * height)])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def coordinate_analysis(stage, rows, components):
    train = [row for row in rows if row["split"] == "train" and row["transformation"] == "original"]
    val = [row for row in rows if row["split"] == "val" and row["transformation"] == "original"]
    test = [row for row in rows if row["split"] == "test"]
    if len(train) < 4 or len(val) < 2 or len({row["scene_id"] for row in test}) < 2:
        raise ValueError(f"Insufficient independent train/val/test scenes for {stage}")
    test_x, test_y = xy(test)
    position_predictions = np.stack([patch_position_readout(row) for row in test])
    probe_rows, equiv_rows, relative_rows = [], [], []
    for seed in SEEDS:
        scaler, model, alpha = fit_ridge(train, val, seed, components)
        predictions = model.predict(scaler.transform(test_x))
        random = np.random.default_rng(seed + 1000)
        probe_dim = model.coef_.shape[1]
        random_train = random.standard_normal((len(train), probe_dim)).astype(np.float32)
        random_test = random.standard_normal((len(test), probe_dim)).astype(np.float32)
        random_probe = Ridge(alpha=alpha).fit(random_train, np.stack([r["y"] for r in train]))
        random_predictions = random_probe.predict(random_test)
        mean_coordinate = np.stack([row["y"] for row in train]).mean(axis=0)
        for transform in sorted({row["transformation"] for row in test}):
            indices = [i for i, row in enumerate(test) if row["transformation"] == transform]
            if not indices:
                continue
            subset = [test[i] for i in indices]
            metric = scores(test_y[indices], predictions[indices])
            lower, upper = bootstrap_ci(subset, predictions[indices], lambda y, p: mean_absolute_error(y, p), seed)
            r2_lower, r2_upper = bootstrap_ci(subset, predictions[indices],
                                              lambda y, p: r2_score(y, p) if len(y) >= 2 else float("nan"), seed)
            probe_rows.append({"stage": stage, "seed": seed, "transformation": transform,
                               "category": "all", "n_objects": len(indices), "alpha": alpha,
                               "probe_dim": probe_dim,
                               **metric, "mae_ci_low": lower, "mae_ci_high": upper,
                               "r2_ci_low": r2_lower, "r2_ci_high": r2_upper,
                               "patch_position_mae": float(mean_absolute_error(test_y[indices], position_predictions[indices])),
                               "patch_position_r2": float(r2_score(test_y[indices], position_predictions[indices])),
                               "random_feature_mae": float(mean_absolute_error(test_y[indices], random_predictions[indices])),
                               "mean_baseline_mae": float(mean_absolute_error(test_y[indices],
                                                      np.tile(mean_coordinate, (len(indices), 1))))})
        for category in sorted({row["category"] for row in test}):
            indices = [i for i, row in enumerate(test) if row["category"] == category]
            probe_rows.append({"stage": stage, "seed": seed, "transformation": "all",
                               "category": category, "n_objects": len(indices), "alpha": alpha,
                               "probe_dim": probe_dim,
                               **scores(test_y[indices], predictions[indices])})
        original = {(row["image_id"], row["object_id"]): (row, predictions[i])
                    for i, row in enumerate(test) if row["transformation"] == "original"}
        by_transform = defaultdict(list)
        pairs_by_view = defaultdict(list)
        for index, row in enumerate(test):
            pairs_by_view[(row["image_id"], row["transformation"])].append((row, predictions[index]))
            if row["transformation"] == "original":
                continue
            key = (row["image_id"], row["object_id"])
            if key in original:
                base, base_pred = original[key]
                expected = map_prediction(base_pred, base["crop"], row["crop"], row["mirror"])
                by_transform[row["transformation"]].append((row, predictions[index], expected))
        for transform, triples in by_transform.items():
            errors = [np.abs(pred - expected).mean() for _, pred, expected in triples]
            error_rows = [{"scene_id": row["scene_id"], "error": error}
                          for (row, _, _), error in zip(triples, errors)]
            ci_low, ci_high = scene_scalar_ci(error_rows, "error", seed)
            equiv_rows.append({"stage": stage, "seed": seed, "transformation": transform,
                               "n_objects": len(errors), "equivariance_mae": float(np.mean(errors)),
                               "equivariance_ci_low": ci_low, "equivariance_ci_high": ci_high,
                               "coordinate_mae": float(np.mean([np.abs(pred - row["y"]).mean()
                                                                  for row, pred, _ in triples]))})
        for (image_id, transform), pairs in pairs_by_view.items():
            for i, (a, pa) in enumerate(pairs):
                for b, pb in pairs[i + 1:]:
                    truth = a["y"] - b["y"]
                    prediction = pa - pb
                    relative_rows.append({"stage": stage, "seed": seed, "image_id": image_id,
                                          "scene_id": a["scene_id"],
                                          "transformation": transform, "method": "absolute_difference",
                                          "relative_mae": float(np.abs(truth - prediction).mean()),
                                          "left_right_correct": int(np.sign(truth[0]) == np.sign(prediction[0])),
                                          "above_below_correct": int(np.sign(truth[1]) == np.sign(prediction[1]))})
    return probe_rows, equiv_rows, relative_rows


def pair_rows(rows, split, original_only=False):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == split and (not original_only or row["transformation"] == "original"):
            grouped[(row["image_id"], row["transformation"])].append(row)
    result = []
    for (image_id, transformation), objects in grouped.items():
        for index, a in enumerate(objects):
            for b in objects[index + 1:]:
                result.append({"x": np.concatenate([a["x"], b["x"]]), "y": a["y"] - b["y"],
                               "scene_id": a["scene_id"], "image_id": image_id,
                               "transformation": transformation})
    return result


def direct_pair_analysis(stage, rows, components):
    train = pair_rows(rows, "train", original_only=True)
    val = pair_rows(rows, "val", original_only=True)
    test = pair_rows(rows, "test")
    if len(train) < 3 or len(val) < 1 or len(test) < 1:
        return []
    results = []
    test_x, test_y = xy(test)
    for seed in SEEDS:
        scaler, model, _ = fit_ridge(train, val, seed, components)
        predictions = model.predict(scaler.transform(test_x))
        for row, truth, pred in zip(test, test_y, predictions):
            results.append({"stage": stage, "seed": seed, "image_id": row["image_id"],
                            "scene_id": row["scene_id"],
                            "transformation": row["transformation"], "method": "direct_pair",
                            "relative_mae": float(np.abs(truth - pred).mean()),
                            "left_right_correct": int(np.sign(truth[0]) == np.sign(pred[0])),
                            "above_below_correct": int(np.sign(truth[1]) == np.sign(pred[1]))})
    return results


def row_basis(weights):
    _, singular, vectors = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-7)) if len(singular) else 0
    return vectors[:rank].T


def organization_analysis(stage, rows, seed, components):
    train = [r for r in rows if r["split"] == "train" and r["transformation"] == "original"]
    test = [r for r in rows if r["split"] == "test" and r["transformation"] == "original"]
    scenes = sorted({r["scene_id"] for r in train})
    np.random.default_rng(seed).shuffle(scenes)
    group_a, group_b = set(scenes[:len(scenes) // 2]), set(scenes[len(scenes) // 2:])
    basis_rows = [r for r in train if r["scene_id"] in group_a]
    eval_rows = [r for r in train if r["scene_id"] in group_b]
    if len(basis_rows) < 4 or len(eval_rows) < 4 or len(test) < 2:
        raise ValueError(f"Insufficient scenes for independent cross-decoding at {stage}")
    bx, by = xy(basis_rows)
    ex, ey = xy(eval_rows)
    tx, ty = xy(test)
    scaler = preprocessor(bx, components)
    bx, ex, tx = [scaler.transform(x) for x in (bx, ex, tx)]
    labels = sorted({r["category"] for r in basis_rows + eval_rows + test})
    basis_categories = np.array([labels.index(r["category"]) for r in basis_rows])
    eval_categories = np.array([labels.index(r["category"]) for r in eval_rows])
    test_categories = np.array([labels.index(r["category"]) for r in test])
    spatial = Ridge(alpha=100).fit(bx, by)
    semantic = Ridge(alpha=100).fit(bx, np.eye(len(labels))[basis_categories])
    spatial_basis = row_basis(spatial.coef_)
    semantic_basis = row_basis(semantic.coef_)
    angles = np.degrees(subspace_angles(spatial_basis, semantic_basis)).tolist()
    overlap = float(np.linalg.norm(spatial_basis.T @ semantic_basis, ord="fro")**2 /
                    min(spatial_basis.shape[1], semantic_basis.shape[1]))
    subspace = [{"stage": stage, "seed": seed, "probe_dim": bx.shape[1],
                 "spatial_rank": spatial_basis.shape[1],
                 "semantic_rank": semantic_basis.shape[1], "principal_angles_degrees": json.dumps(angles),
                 "subspace_overlap": overlap, "n_basis_objects": len(basis_rows)}]
    cross = []
    for name, projection in (
        ("full", lambda x: x),
        ("spatial", lambda x: x @ spatial_basis @ spatial_basis.T),
        ("semantic", lambda x: x @ semantic_basis @ semantic_basis.T),
        ("spatial_removed", lambda x: x - x @ spatial_basis @ spatial_basis.T),
    ):
        coord = Ridge(alpha=100).fit(projection(ex), ey)
        category_probe = Ridge(alpha=100).fit(projection(ex), np.eye(len(labels))[eval_categories])
        predicted = coord.predict(projection(tx))
        classified = np.argmax(category_probe.predict(projection(tx)), axis=1)
        cross.append({"stage": stage, "seed": seed, "representation": name, "n_test_objects": len(test),
                      **scores(ty, predicted),
                      "category_balanced_accuracy": float(balanced_accuracy_score(test_categories, classified))})
    return subspace, cross


def category_transfer(stage, rows, seed, components):
    source = {"Car", "Tree"}
    target = {"Pole", "Traffic Sign"}
    train = [r for r in rows if r["split"] == "train" and r["transformation"] == "original"
             and r["category"] in source]
    test = [r for r in rows if r["split"] == "test" and r["transformation"] == "original"
            and r["category"] in target]
    if len(train) < 4 or len(test) < 2:
        return {"stage": stage, "seed": seed, "status": "insufficient_category_examples",
                "n_train": len(train), "n_test": len(test)}
    train_x, train_y = xy(train)
    test_x, test_y = xy(test)
    scaler = preprocessor(train_x, components)
    scenes = sorted({r["scene_id"] for r in train})
    sampled = np.random.default_rng(seed).choice(scenes, size=len(scenes), replace=True)
    indices = np.concatenate([[i for i, r in enumerate(train) if r["scene_id"] == scene] for scene in sampled])
    model = Ridge(alpha=100).fit(scaler.transform(train_x)[indices], train_y[indices])
    return {"stage": stage, "seed": seed, "status": "computed", "n_train": len(train), "n_test": len(test),
            "train_categories": ",".join(sorted(source)), "test_categories": ",".join(sorted(target)),
            **scores(test_y, model.predict(scaler.transform(test_x)))}


def figures(output, coordinate, equivariance, overlap, cross):
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    original = [r for r in coordinate if r["transformation"] == "original" and r["category"] == "all"]
    def stage_order(name):
        if name == "encoder_patch":
            return (0, -1)
        if name.startswith("encoder_"):
            return (0, int(name.split("_")[-1]))
        if name == "merger":
            return (1, 0)
        return (2, int(name.split("_")[-1]))

    stages = sorted({r["stage"] for r in original}, key=stage_order)
    values = [np.mean([r["coordinate_r2"] for r in original if r["stage"] == stage]) for stage in stages]
    fig, ax = plt.subplots(figsize=(max(7, len(stages) * 0.9), 4))
    ax.bar(stages, values, color="#277b82")
    ax.set(ylabel="Held-out coordinate R2", title="Original-view coordinate decoding")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_rq1_layer_coordinate_r2.png", dpi=160)
    fig.savefig(figures_dir / "fig_rq3_coordinate_transfer.png", dpi=160)
    plt.close(fig)
    if equivariance:
        labels = sorted({r["transformation"] for r in equivariance})
        values = [np.mean([r["equivariance_mae"] for r in equivariance if r["transformation"] == label])
                  for label in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#cf694b")
        ax.set(ylabel="Equivariance MAE", title="Geometric consistency on held-out scenes")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_rq1_equivariance.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(7, len(overlap) * 0.9), 4))
    ax.bar(stages, [np.mean([r["subspace_overlap"] for r in overlap if r["stage"] == stage])
                    for stage in stages], color="#bc8b37")
    ax.set(ylabel="Normalized row-space overlap", title="Spatial and semantic probe subspaces")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_rq2_subspace_overlap.png", dpi=160)
    plt.close(fig)
    for filename, metric in (("fig_rq2_cross_decoding.png", "coordinate_r2"),
                             ("fig_rq3_semantic_spatial_changes.png", "category_balanced_accuracy")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for representation in sorted({r["representation"] for r in cross}):
            values = [np.mean([r[metric] for r in cross if r["stage"] == stage
                               and r["representation"] == representation])
                      for stage in stages]
            ax.plot(stages, values, marker="o", label=representation)
        ax.set(ylabel=metric.replace("_", " "), title=metric.replace("_", " ").title())
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_ROOT / "outputs/v0/features")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "outputs/v0")
    parser.add_argument("--min-original-box-side", type=float, default=0,
                        help="Keep only scenes with two or more original-view boxes this many pixels wide and tall")
    parser.add_argument("--probe-components", type=int, default=128,
                        help="Matched PCA dimension fitted on training scenes; use 0 for full embeddings")
    args = parser.parse_args()
    data = filter_quality(load_features(args.features), args.min_original_box_side)
    coordinate, equivariance, relative, overlap, cross, transfer = [], [], [], [], [], []
    for stage, rows in data.items():
        probe, equi, relations = coordinate_analysis(stage, rows, args.probe_components)
        coordinate.extend(probe)
        equivariance.extend(equi)
        relative.extend(relations)
        relative.extend(direct_pair_analysis(stage, rows, args.probe_components))
        for seed in SEEDS:
            subspace, decoding = organization_analysis(stage, rows, seed, args.probe_components)
            overlap.extend(subspace)
            cross.extend(decoding)
        transfer.extend(category_transfer(stage, rows, seed, args.probe_components) for seed in SEEDS)
        print(f"Analyzed {stage}: {len(rows)} object views", flush=True)
    metrics = args.output / "metrics"
    for name, rows in (
        ("coordinate_probe_results.csv", coordinate), ("equivariance_results.csv", equivariance),
        ("relative_relation_results.csv", relative), ("subspace_overlap.csv", overlap),
        ("cross_decoding.csv", cross), ("category_transfer.csv", transfer),
        ("cross_stage_coordinates.csv", coordinate), ("cross_stage_organization.csv", cross),
    ):
        write_csv(metrics / name, rows)
    for name, rows, grouping, fields in (
        ("coordinate_probe_summary.csv", coordinate, ("stage", "transformation", "category"),
         ("coordinate_mae", "coordinate_r2", "x_r2", "y_r2", "random_feature_mae",
          "mean_baseline_mae", "patch_position_mae", "patch_position_r2")),
        ("equivariance_summary.csv", equivariance, ("stage", "transformation"),
         ("equivariance_mae", "coordinate_mae")),
        ("subspace_overlap_summary.csv", overlap, ("stage",), ("subspace_overlap",)),
        ("cross_decoding_summary.csv", cross, ("stage", "representation"),
         ("coordinate_mae", "coordinate_r2", "category_balanced_accuracy")),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in grouping)].append(row)
        summary_rows = []
        for key, records in groups.items():
            summary = dict(zip(grouping, key))
            summary["n_seeds"] = len(records)
            for field in fields:
                values = np.array([float(row[field]) for row in records if field in row], dtype=np.float64)
                if len(values):
                    summary[f"{field}_mean"] = float(np.nanmean(values))
                    summary[f"{field}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(summary)
        write_csv(metrics / name, summary_rows)
    relative_groups = defaultdict(lambda: defaultdict(list))
    for row in relative:
        key = (row["stage"], row["transformation"], row["method"])
        relative_groups[key][row["seed"]].append(row)
    relative_summary = []
    for key, seeds in relative_groups.items():
        summary = dict(zip(("stage", "transformation", "method"), key))
        summary["n_seeds"] = len(seeds)
        summary["n_pairs"] = sum(len(group) for group in seeds.values()) // len(seeds)
        for field in ("relative_mae", "left_right_correct", "above_below_correct"):
            values = [np.mean([float(row[field]) for row in group]) for group in seeds.values()]
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_low, ci_high = scene_scalar_ci(next(iter(seeds.values())), "relative_mae")
        summary["relative_mae_ci_low"] = ci_low
        summary["relative_mae_ci_high"] = ci_high
        relative_summary.append(summary)
    write_csv(metrics / "relative_relation_summary.csv", relative_summary)
    figures(args.output, coordinate, equivariance, overlap, cross)
    summary = {"stages": list(data), "min_original_box_side": args.min_original_box_side,
               "probe_components": args.probe_components,
               "object_views_per_stage": {s: len(r) for s, r in data.items()},
               "scene_count_per_stage": {s: len({x["scene_id"] for x in r}) for s, r in data.items()},
               "splits": {s: dict(Counter(r["split"] for r in rows)) for s, rows in data.items()}}
    (args.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
