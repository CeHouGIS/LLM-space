"""StreetSpatial V0.1 controls on frozen, object-pooled V0 features."""

import argparse
import csv
import hashlib
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
from sklearn.preprocessing import StandardScaler

from .analyze import DATA_ROOT, SEEDS, bootstrap_ci, map_prediction, scores
from .build_dataset import map_bbox
from .token_alignment import overlap_mask, token_boxes


STAGES = ("encoder_patch", "encoder_0", "encoder_16", "encoder_31", "merger", "llm_14")
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def position_features(meta, obj, stage):
    """Use precisely the token mask used to pool this object's cached state."""
    _, height, width = meta["image_grid_thw"]
    boxes = token_boxes(height, width, meta["output_size"], merged=stage in ("merger", "llm_14"))
    indices = np.flatnonzero(overlap_mask(boxes, obj["bbox"]))
    if not len(indices):
        raise ValueError(f"No selected tokens: {meta['image_id']}/{obj['object_id']}")
    selected = boxes[indices]
    size = np.asarray(meta["output_size"], dtype=np.float32)
    centers = (selected[:, :2] + selected[:, 2:]) / (2 * size)
    fourier = np.concatenate([f(2 * np.pi * centers * frequency).mean(axis=0)
                              for frequency in (1, 2, 4, 8) for f in (np.sin, np.cos)])
    geometry = np.concatenate([centers.mean(axis=0), centers.std(axis=0), centers.min(axis=0),
                               centers.max(axis=0), [len(indices) / len(boxes)], fourier])
    return geometry.astype(np.float32), indices, len(boxes)


def load_cache(directory, stages=STAGES):
    data = {stage: [] for stage in stages}
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No V0 feature cache in {directory}")
    audit = Counter()
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["metadata_json"]))
            if meta["embedding_alignment_max_abs_error"] > 0.01 or meta["image_token_count"] != 154:
                raise ValueError(f"Invalid token alignment in {path}")
            audit[(meta["split"], meta["annotation_status"])] += 1
            for stage in stages:
                if stage not in payload or len(payload[stage]) != len(meta["objects"]):
                    raise ValueError(f"Missing or mismatched {stage} in {path}")
                for index, obj in enumerate(meta["objects"]):
                    pos, ids, token_count = position_features(meta, obj, stage)
                    row = {"x": payload[stage][index].astype(np.float32), "p": pos,
                           "indices": ids, "n_tokens": token_count, "y": np.asarray(obj["normalized_center"]),
                           "bbox": obj["bbox"], "image_size": meta["output_size"],
                           "scene_id": meta["scene_id"], "image_id": meta["image_id"],
                           "object_id": obj["object_id"], "category": obj["object_category"],
                           "split": meta["split"], "status": meta["annotation_status"],
                           "city": meta["city_id"], "transformation": meta["transformation"],
                           "crop": meta["crop_xyxy"], "mirror": meta["mirror"]}
                    data[stage].append(row)
    for stage, rows in data.items():
        scenes = defaultdict(set)
        identities = defaultdict(set)
        for row in rows:
            scenes[row["scene_id"]].add(row["split"])
            identities[(row["image_id"], row["object_id"])].add(row["category"])
        if any(len(splits) != 1 for splits in scenes.values()):
            raise ValueError(f"Scene leakage in {stage}")
        if any(len(categories) != 1 for categories in identities.values()):
            raise ValueError(f"Object category changed between views in {stage}")
    return data, audit


def arrays(rows, field):
    return np.stack([row[field] for row in rows])


def scene_sample(rows, seed):
    scenes = sorted({row["scene_id"] for row in rows})
    chosen = np.random.default_rng(seed).choice(scenes, len(scenes), replace=True)
    return np.concatenate([[i for i, row in enumerate(rows) if row["scene_id"] == scene]
                           for scene in chosen])


def design(train, val, test, field, components=0):
    a, b, c = [arrays(rows, field) for rows in (train, val, test)]
    scaler = StandardScaler().fit(a)
    a, b, c = [scaler.transform(x) for x in (a, b, c)]
    if components and a.shape[1] > components:
        pca = PCA(n_components=min(components, len(a) - 1), svd_solver="randomized", random_state=42).fit(a)
        a, b, c = [pca.transform(x) for x in (a, b, c)]
    return a, b, c


def fit(a, y, b, val_y, seed, indices=None):
    if indices is None:
        indices = np.arange(len(a))
    chosen = min(ALPHAS, key=lambda alpha: mean_absolute_error(
        val_y, Ridge(alpha=alpha).fit(a[indices], y[indices]).predict(b)))
    return Ridge(alpha=chosen).fit(a[indices], y[indices]), chosen


def ci(rows, predicted, seed):
    low, high = bootstrap_ci(rows, predicted, mean_absolute_error, seed=seed, repetitions=300)
    return {"mae_ci_low": low, "mae_ci_high": high}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(path, rows, key, metric, title, ylabel):
    labels = list(dict.fromkeys(row[key] for row in rows))
    values = [np.mean([row[metric] for row in rows if row[key] == label]) for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * .8), 4))
    ax.bar(labels, values, color="#317d7a")
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def e1(stage, rows, seed):
    train = [r for r in rows if r["split"] == "train" and r["transformation"] == "original"]
    val = [r for r in rows if r["split"] == "val" and r["transformation"] == "original"]
    test = [r for r in rows if r["split"] == "test" and r["transformation"] == "original"]
    y, vy, ty = [arrays(group, "y") for group in (train, val, test)]
    indices = scene_sample(train, seed)
    p = design(train, val, test, "p")
    h = design(train, val, test, "x", 128)
    results, predictions = [], {}

    mean = y[indices].mean(axis=0)
    category_means = {}
    for category in {r["category"] for r in train}:
        examples = [train[i] for i in indices if train[i]["category"] == category]
        category_means[category] = arrays(examples, "y").mean(axis=0) if examples else mean
    predictions["mean"] = np.tile(mean, (len(test), 1))
    predictions["category_mean"] = np.stack([category_means.get(r["category"], mean) for r in test])
    predictions["oracle_patch"] = np.stack([r["p"][:2] for r in test])
    predictions["oracle_bbox"] = ty.copy()

    for name, xs in (("position_only", p), ("real", h),
                     ("position_plus_real", tuple(np.concatenate((a, b), axis=1) for a, b in zip(p, h)))):
        model, _ = fit(xs[0], y, xs[1], vy, seed, indices)
        predictions[name] = model.predict(xs[2])

    # First remove the component of the representation predictable from known object geometry.
    residualizer = Ridge(alpha=100).fit(p[0][indices], h[0][indices])
    residual = tuple(a - residualizer.predict(b) for a, b in zip(h, p))
    model, _ = fit(residual[0], y, residual[1], vy, seed, indices)
    predictions["position_residual"] = model.predict(residual[2])

    for tied in (False, True):
        def random_rows(group):
            vectors = []
            for row in group:
                key = f"{seed}:{stage}:{'tied' if tied else row['image_id'] + row['transformation']}"
                rng = np.random.default_rng(stable_seed(key))
                token_vectors = rng.standard_normal((row["n_tokens"], 128)).astype(np.float32)
                vectors.append(token_vectors[row["indices"]].mean(axis=0))
            return np.stack(vectors)
        ra, rb, rc = [random_rows(group) for group in (train, val, test)]
        scaler = StandardScaler().fit(ra)
        ra, rb, rc = [scaler.transform(a) for a in (ra, rb, rc)]
        model, _ = fit(ra, y, rb, vy, seed, indices)
        predictions["random_position_tied" if tied else "random_view_tokens"] = model.predict(rc)

    for name, predicted in predictions.items():
        results.append({"stage": stage, "seed": seed, "condition": name, "n_scenes": len({r['scene_id'] for r in test}),
                        "n_objects": len(test), "reviewed_objects": sum(r["status"] == "visually_reviewed" for r in test),
                        **scores(ty, predicted), **ci(test, predicted, seed)})
        for status in ("visually_reviewed", "machine_proposal_unreviewed"):
            chosen = [i for i, row in enumerate(test) if row["status"] == status]
            if chosen:
                subset = [test[i] for i in chosen]
                results.append({"stage": stage, "seed": seed, "condition": name, "subset": status,
                                "n_scenes": len({r["scene_id"] for r in subset}), "n_objects": len(chosen),
                                **scores(ty[chosen], predicted[chosen]),
                                **ci(subset, predicted[chosen], seed)})
    base = next(r for r in results if r["condition"] == "position_only")
    scene_groups = {scene: np.array([i for i, row in enumerate(test) if row["scene_id"] == scene])
                    for scene in sorted({row["scene_id"] for row in test})}
    scene_ids = list(scene_groups)
    for row in results:
        if "subset" not in row:
            row["incremental_r2_vs_position"] = row["coordinate_r2"] - base["coordinate_r2"]
            row["paired_mae_delta_vs_position"] = row["coordinate_mae"] - base["coordinate_mae"]
            random = np.random.default_rng(seed)
            differences = []
            for _ in range(300):
                selected = np.concatenate([scene_groups[scene] for scene in random.choice(scene_ids, len(scene_ids), replace=True)])
                differences.append(mean_absolute_error(ty[selected], predictions[row["condition"]][selected])
                                   - mean_absolute_error(ty[selected], predictions["position_only"][selected]))
            row["paired_delta_ci_low"], row["paired_delta_ci_high"] = np.percentile(differences, [2.5, 97.5])
    return results, {"train": train, "val": val, "test": test, "p": p, "h": h,
                     "residual": residual, "predictions": predictions}


def e2(stage, fitted, seed):
    train, val, test = [fitted[name] for name in ("train", "val", "test")]
    # Fit on original views only, then apply the same mapping to every held-out transform.
    all_test = fitted["all_test"]
    y, vy = arrays(train, "y"), arrays(val, "y")
    indices = scene_sample(train, seed)
    trajectories, summaries = [], []
    for condition, field, dim in (("real", "x", 128), ("position_only", "p", 0),
                                  ("pca_2", "x", 2), ("random_2", "x", 128),
                                  ("candidate_2", "x", 128)):
        a, b, c = design(train, val, all_test, field, dim)
        if condition == "random_2":
            random_basis = np.linalg.qr(np.random.default_rng(seed).standard_normal((a.shape[1], 2)))[0]
            a, b, c = [x @ random_basis for x in (a, b, c)]
        elif condition == "candidate_2":
            direction = Ridge(alpha=100).fit(a[indices], y[indices]).coef_
            candidate_basis = np.linalg.svd(direction, full_matrices=False)[2][:2].T
            a, b, c = [x @ candidate_basis for x in (a, b, c)]
        model, _ = fit(a, y, b, vy, seed, indices)
        pred = model.predict(c)
        original = {(r["image_id"], r["object_id"]): (r, pred[i]) for i, r in enumerate(all_test)
                    if r["transformation"] == "original"}
        for row, point in zip(all_test, pred):
            base = original.get((row["image_id"], row["object_id"]))
            if base is None:
                raise ValueError(f"Missing original identity {row['image_id']}/{row['object_id']}")
            expected = map_prediction(base[1], base[0]["crop"], row["crop"], row["mirror"])
            actual = row["y"]
            axis = 0 if row["transformation"] in ("mirror", "shift_left", "shift_right") else 1 \
                   if row["transformation"] in ("shift_up", "shift_down") else None
            delta_actual, delta_pred = actual - base[0]["y"], point - base[1]
            if row["transformation"] == "original":
                direction_ok = 1
            elif axis is None:
                moving = np.abs(delta_actual) > .01
                direction_ok = int(np.all(np.sign(delta_actual[moving]) == np.sign(delta_pred[moving]))) if moving.any() else 1
            else:
                direction_ok = int(np.sign(delta_actual[axis]) == np.sign(delta_pred[axis]))
            trajectories.append({"stage": stage, "seed": seed, "condition": condition,
                "scene_id": row["scene_id"], "image_id": row["image_id"], "object_id": row["object_id"],
                "category": row["category"], "annotation_status": row["status"],
                "transformation": row["transformation"], "true_x": actual[0], "true_y": actual[1],
                "latent_x": point[0], "latent_y": point[1], "expected_x": expected[0], "expected_y": expected[1],
                "coordinate_error": np.abs(point - actual).mean(), "equivariance_error": np.abs(point - expected).mean(),
                "direction_consistent": direction_ok})
        for transform in sorted({r["transformation"] for r in all_test}):
            subset = [r for r in trajectories if r["stage"] == stage and r["seed"] == seed
                      and r["condition"] == condition and r["transformation"] == transform]
            if not subset:
                continue
            scenes = sorted({r["scene_id"] for r in subset})
            rng = np.random.default_rng(seed)
            samples = []
            for _ in range(300):
                sampled = rng.choice(scenes, len(scenes), replace=True)
                samples.append(np.mean([r["equivariance_error"] for scene in sampled for r in subset if r["scene_id"] == scene]))
            summaries.append({"stage": stage, "seed": seed, "condition": condition, "transformation": transform,
                              "n_objects": len(subset), "n_scenes": len(scenes),
                              "coordinate_mae": np.mean([r["coordinate_error"] for r in subset]),
                              "equivariance_mae": np.mean([r["equivariance_error"] for r in subset]),
                              "direction_consistency": np.mean([r["direction_consistent"] for r in subset]),
                              "equiv_ci_low": np.percentile(samples, 2.5), "equiv_ci_high": np.percentile(samples, 97.5)})
    return trajectories, summaries


def e3(stage, rows, seed):
    train = [r for r in rows if r["split"] == "train" and r["transformation"] == "original"]
    val = [r for r in rows if r["split"] == "val" and r["transformation"] == "original"]
    test = [r for r in rows if r["split"] == "test" and r["transformation"] == "original"]
    categories = sorted({r["category"] for r in rows})
    tasks = [("Car+Tree", {"Car", "Tree"}, {"Pole", "Traffic Sign"})]
    tasks += [(f"leave_{category}_out", set(categories) - {category}, {category}) for category in categories]
    result = []
    for task, source, target in tasks:
        a = [r for r in train if r["category"] in source]
        b = [r for r in val if r["category"] in source]
        seen_pool = [r for r in test if r["category"] in source]
        unseen = [r for r in test if r["category"] in target]
        if min(map(len, (a, b, seen_pool, unseen))) < 3:
            result.append({"stage": stage, "seed": seed, "task": task, "status": "insufficient_examples",
                           "n_train": len(a), "n_seen": len(seen_pool), "n_unseen": len(unseen)})
            continue
        # Predefined nearest-neighbor match on city, center and box geometry; reuse is reported.
        def metadata(row):
            box = np.asarray(row["bbox"])
            size = np.asarray(row["image_size"])
            span = (box[2:] - box[:2]) / size
            return np.concatenate([row["y"], span, [np.log(max(span.prod(), 1e-6)) / 10]])
        seen = []
        for target_row in unseen:
            same_city = [r for r in seen_pool if r["city"] == target_row["city"]]
            pool = same_city if same_city else seen_pool
            seen.append(min(pool, key=lambda row: np.linalg.norm(metadata(row) - metadata(target_row))))
        # Evaluation uses exactly the same target objects across conditions.
        all_test = seen + unseen
        for condition in ("position_only", "real", "position_residual", "candidate_2"):
            if condition == "position_only":
                train_x, val_x, test_x = design(a, b, all_test, "p", 0)
            else:
                train_x, val_x, test_x = design(a, b, all_test, "x", 128)
                if condition == "position_residual":
                    train_p, val_p, test_p = design(a, b, all_test, "p", 0)
                    residualizer = Ridge(alpha=100).fit(train_p, train_x)
                    train_x, val_x, test_x = [x - residualizer.predict(p)
                        for x, p in ((train_x, train_p), (val_x, val_p), (test_x, test_p))]
                elif condition == "candidate_2":
                    direction = Ridge(alpha=100).fit(train_x, arrays(a, "y")).coef_
                    basis = np.linalg.svd(direction, full_matrices=False)[2][:2].T
                    train_x, val_x, test_x = [x @ basis for x in (train_x, val_x, test_x)]
            model, _ = fit(train_x, arrays(a, "y"), val_x, arrays(b, "y"), seed, scene_sample(a, seed))
            predicted = model.predict(test_x)
            n = len(seen)
            seen_mae = mean_absolute_error(arrays(seen, "y"), predicted[:n])
            unseen_mae = mean_absolute_error(arrays(unseen, "y"), predicted[n:])
            confidence = ci(unseen, predicted[n:], seed)
            lo, hi = confidence["mae_ci_low"], confidence["mae_ci_high"]
            result.append({"stage": stage, "seed": seed, "task": task, "condition": condition,
                           "status": "computed", "n_train": len(a), "n_seen": n, "n_unseen": len(unseen),
                           "n_unique_matched_seen": len({(r["image_id"], r["object_id"]) for r in seen}),
                           "same_city_match_rate": np.mean([x["city"] == z["city"] for x, z in zip(seen, unseen)]),
                           "mean_matching_distance": np.mean([np.linalg.norm(metadata(x)-metadata(z)) for x,z in zip(seen,unseen)]),
                           "n_unseen_scenes": len({r["scene_id"] for r in unseen}),
                           "reviewed_unseen": sum(r["status"] == "visually_reviewed" for r in unseen),
                           "seen_mae": seen_mae, "unseen_mae": unseen_mae, "transfer_gap": unseen_mae - seen_mae,
                           "unseen_r2": r2_score(arrays(unseen, "y"), predicted[n:]),
                           "unseen_mae_ci_low": lo, "unseen_mae_ci_high": hi})
    return result


def orthonormal(matrix, k):
    return np.linalg.svd(matrix, full_matrices=False)[2][:k].T


def e4(stage, rows, seed, dimensions=(2, 4, 8, 16, 32)):
    originals = [r for r in rows if r["split"] == "train" and r["transformation"] == "original"]
    scenes = sorted({r["scene_id"] for r in originals})
    np.random.default_rng(seed).shuffle(scenes)
    discovery_scenes = set(scenes[:len(scenes) // 2])
    discovery = [r for r in originals if r["scene_id"] in discovery_scenes]
    evaluation = [r for r in originals if r["scene_id"] not in discovery_scenes]
    test = [r for r in rows if r["split"] == "test" and r["transformation"] == "original"]
    if {r["scene_id"] for r in discovery} & {r["scene_id"] for r in evaluation + test}:
        raise ValueError("Subspace discovery shares scenes with evaluation")
    a, b, c = design(discovery, evaluation, test, "x", 128)
    labels = sorted({r["category"] for r in discovery + evaluation + test})
    label_id = {name: i for i, name in enumerate(labels)}
    category_y = lambda group: np.eye(len(labels))[[label_id[r["category"]] for r in group]]
    coord = Ridge(alpha=100).fit(a, arrays(discovery, "y"))
    semantic = Ridge(alpha=100).fit(a, category_y(discovery))
    coordinate_basis = orthonormal(coord.coef_, 2)
    semantic_basis = orthonormal(semantic.coef_, min(len(labels), 4))
    pca_basis = orthonormal(a, max(dimensions))
    rng = np.random.default_rng(seed)
    random_basis = np.linalg.qr(rng.standard_normal((a.shape[1], max(dimensions))))[0]
    # The transform-sensitive basis is learned only from discovery scenes and identities.
    all_train = [r for r in rows if r["split"] == "train" and r["scene_id"] in discovery_scenes]
    scaler = StandardScaler().fit(arrays(discovery, "x"))
    pca = PCA(n_components=a.shape[1], svd_solver="randomized", random_state=42).fit(scaler.transform(arrays(discovery, "x")))
    transformed = pca.transform(scaler.transform(arrays(all_train, "x")))
    original_lookup = {(r["image_id"], r["object_id"]): transformed[i]
                       for i, r in enumerate(all_train) if r["transformation"] == "original"}
    differences = defaultdict(list)
    for row, feature in zip(all_train, transformed):
        if row["transformation"] != "original":
            base = original_lookup.get((row["image_id"], row["object_id"]))
            if base is not None:
                kind = "shift" if row["transformation"].startswith("shift") else "zoom" if row["transformation"].startswith("zoom") else "mirror"
                differences[kind].append(feature - base)
    bases = {"coordinate": coordinate_basis, "pca": pca_basis, "random": random_basis,
             "semantic": semantic_basis}
    for kind, diff in differences.items():
        if len(diff) > max(dimensions):
            bases[f"transform_{kind}"] = orthonormal(np.stack(diff), max(dimensions))
    result = []
    for k in dimensions:
        for name, basis in bases.items():
            if name == "coordinate" and k != 2:
                continue
            if name == "semantic" and k != semantic_basis.shape[1]:
                continue
            q = basis[:, :min(k, basis.shape[1])]
            for variant, transform in (("projected", lambda x: x @ q),
                                       ("removed", lambda x: x - x @ q @ q.T)):
                ex, tx = transform(b), transform(c)
                spatial = Ridge(alpha=100).fit(ex, arrays(evaluation, "y"))
                semantic_probe = Ridge(alpha=100).fit(ex, category_y(evaluation))
                predicted = spatial.predict(tx)
                predicted_labels = semantic_probe.predict(tx).argmax(axis=1)
                result.append({"stage": stage, "seed": seed, "subspace": name, "variant": variant,
                               "k": q.shape[1], "n_discovery_scenes": len(discovery_scenes),
                               "n_evaluation_scenes": len(scenes) - len(discovery_scenes),
                               "n_test_scenes": len({r["scene_id"] for r in test}), "n_test": len(test),
                               "coordinate_mae": mean_absolute_error(arrays(test, "y"), predicted),
                               "coordinate_r2": r2_score(arrays(test, "y"), predicted),
                               "category_balanced_accuracy": balanced_accuracy_score(
                                   [label_id[r["category"]] for r in test], predicted_labels),
                               "coord_semantic_overlap": np.linalg.norm(q.T @ semantic_basis, ord="fro")**2 / q.shape[1],
                               "coord_semantic_min_angle": np.min(np.degrees(subspace_angles(q, semantic_basis))),
                               **ci(test, predicted, seed)})
    return result


def audit_manifest(path, cache, output):
    views = [json.loads(line) for line in path.open(encoding="utf-8")]
    split = defaultdict(set)
    missing = []
    invalid = []
    originals = {(view["image_id"], view["scene_id"]): view for view in views if view["transformation"] == "original"}
    for view in views:
        split[view["scene_id"]].add(view["split"])
        if not Path(view["image_path"]).is_file():
            missing.append(view["image_path"])
        original = {(obj["object_id"], obj["object_category"]) for obj in view["objects"]}
        if len(original) != len(view["objects"]):
            invalid.append(view["image_id"])
        base = originals.get((view["image_id"], view["scene_id"]))
        if base is None or view["split"] != base["split"]:
            invalid.append(view["image_id"])
            continue
        by_id = {obj["object_id"]: obj for obj in base["objects"]}
        for obj in view["objects"]:
            source = by_id.get(obj["object_id"])
            if source is None or source["object_category"] != obj["object_category"]:
                invalid.append(view["image_id"])
                continue
            # Compose through source-image pixels using exact recorded crop windows.
            first = np.asarray(source["bbox"])
            left, top, right, bottom = base["crop_xyxy"]
            width, height = base["output_size"]
            if base["mirror"]:
                first = np.asarray([width-first[2], first[1], width-first[0], first[3]])
            source_box = [left + first[0]*(right-left)/width, top + first[1]*(bottom-top)/height,
                          left + first[2]*(right-left)/width, top + first[3]*(bottom-top)/height]
            expected = map_bbox(source_box, view["crop_xyxy"], view["output_size"], view["mirror"])
            if not np.allclose(expected, obj["bbox"], atol=1e-3):
                invalid.append(view["image_id"])
    if any(len(values) != 1 for values in split.values()) or missing or invalid:
        raise ValueError(f"Audit failed: scene leakage={sum(len(x)>1 for x in split.values())}, missing={len(missing)}, invalid={len(invalid)}")
    expected_stems = {f"{view['image_id']}__{view['transformation']}" for view in views}
    actual_stems = {item.stem for item in cache.glob("*.npz")}
    if expected_stems != actual_stems:
        raise ValueError(f"Feature cache differs from manifest: missing={len(expected_stems-actual_stems)}, extra={len(actual_stems-expected_stems)}")
    run_config = json.loads((cache / "run_config.json").read_text(encoding="utf-8"))
    if run_config["neutral_prompt"] != "Describe the image." or run_config["model_manifest"]["model_id"] != "Qwen/Qwen2.5-VL-7B-Instruct":
        raise ValueError("V0 feature cache model or prompt does not match V0.1 protocol")
    result = {"n_views": len(views), "n_scenes": len(split), "split_scenes": dict(Counter(next(iter(x)) for x in split.values())),
              "annotation_status_views": dict(Counter(v.get("annotation_status", "unspecified") for v in views)),
              "image_paths_readable": True, "feature_cache_complete": True,
              "note": "Machine proposals are not human ground truth; no additional manual review was performed."}
    (output / "manifests" / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_ROOT / "outputs/v0/pilot200_features")
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/pilot200_final/transformations.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/v0_1")
    args = parser.parse_args()
    for folder in ("metrics", "figures", "manifests", "logs"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(args.manifest, args.features, args.output)
    data, _ = load_cache(args.features)
    metric_dir, figure_dir = args.output / "metrics", args.output / "figures"
    e1_rows, e2_rows, e2_summary, e3_rows, e4_rows = [], [], [], [], []
    for stage, rows in data.items():
        for seed in SEEDS:
            controls, fitted = e1(stage, rows, seed)
            e1_rows.extend(controls)
            fitted["all_test"] = [r for r in rows if r["split"] == "test"]
            trajectory, equivariance = e2(stage, fitted, seed)
            e2_rows.extend(trajectory)
            e2_summary.extend(equivariance)
            e3_rows.extend(e3(stage, rows, seed))
            e4_rows.extend(e4(stage, rows, seed))
        print(f"V0.1 completed {stage}", flush=True)
    for name, rows in (("e1_position_leakage", e1_rows), ("e2_latent_trajectories", e2_rows),
                       ("e2_equivariance_comparison", e2_summary), ("e3_category_transfer", e3_rows),
                       ("e4_disentanglement", e4_rows)):
        write_csv(metric_dir / f"{name}.csv", rows)
    cross = [r for r in e2_summary if r["stage"] in ("encoder_16", "encoder_31", "merger", "llm_14")]
    write_csv(metric_dir / "cross_stage_v0_1.csv", cross)
    plot_bars(figure_dir / "fig_e1_position_baselines.png", [r for r in e1_rows if r["stage"] == "encoder_31" and "subset" not in r],
              "condition", "coordinate_mae", "Encoder 31 position controls", "Coordinate MAE")
    plot_bars(figure_dir / "fig_e1_residual_coordinate_information.png",
              [r for r in e1_rows if "subset" not in r and r["condition"] in ("position_only", "real", "position_residual")],
              "stage", "coordinate_r2", "Coordinate decoding by stage", "R2")
    plot_bars(figure_dir / "fig_e2_equivariance_comparison.png",
              [r for r in e2_summary if r["stage"] == "encoder_31" and r["condition"] == "real"],
              "transformation", "equivariance_mae", "Encoder 31 held-out equivariance", "MAE")
    plot_bars(figure_dir / "fig_e3_transfer_gap.png", [r for r in e3_rows if r.get("status") == "computed" and r["stage"] == "encoder_31"],
              "task", "transfer_gap", "Encoder 31 category transfer", "MAE gap")
    plot_bars(figure_dir / "fig_e4_spatial_semantic_tradeoff.png",
              [r for r in e4_rows if r["stage"] == "encoder_31" and r["variant"] == "projected"],
              "subspace", "coordinate_r2", "Independent subspace test", "R2")
    plot_bars(figure_dir / "fig_cross_stage_equivariance.png",
              [r for r in cross if r["condition"] == "real" and r["transformation"] == "mirror"],
              "stage", "equivariance_mae", "Mirror equivariance across stages", "MAE")
    (args.output / "manifests" / "run_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
