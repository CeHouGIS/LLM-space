"""Audit every cached feature against the finalized view manifest."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .build_dataset import read_jsonl


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/pilot200_final/transformations.jsonl")
    parser.add_argument("--features", type=Path, default=DATA_ROOT / "outputs/v0/pilot200_features")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "outputs/v0/pilot200_audit.json")
    args = parser.parse_args()
    views = read_jsonl(args.manifest)
    expected = {f"{view['image_id']}__{view['transformation']}.npz": view for view in views}
    observed = {path.name: path for path in args.features.glob("*.npz")}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Feature inventory mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    stages = defaultdict(lambda: {"count": 0, "dimensions": set(), "dtypes": set(), "max_abs": 0.0})
    image_token_counts = Counter()
    max_alignment_error = 0.0
    scenes_by_split = defaultdict(set)
    for filename, view in expected.items():
        scenes_by_split[view["scene_id"]].add(view["split"])
        with np.load(observed[filename], allow_pickle=False) as file:
            meta = json.loads(str(file["metadata_json"]))
            for field in ("image_id", "scene_id", "split", "transformation", "annotation_status"):
                if meta[field] != view[field]:
                    raise ValueError(f"Metadata mismatch in {filename}: {field}")
            if len(meta["objects"]) != len(view["objects"]):
                raise ValueError(f"Object count mismatch in {filename}")
            t, h, w = meta["image_grid_thw"]
            if meta["image_token_count"] != t * h * w // 4:
                raise ValueError(f"Image token count mismatch in {filename}")
            image_token_counts[meta["image_token_count"]] += 1
            max_alignment_error = max(max_alignment_error, meta["embedding_alignment_max_abs_error"])
            for stage in file.files:
                if stage == "metadata_json":
                    continue
                values = file[stage]
                if values.ndim != 2 or values.shape[0] != len(view["objects"]) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid feature tensor in {filename}: {stage}")
                stats = stages[stage]
                stats["count"] += 1
                stats["dimensions"].add(values.shape[1])
                stats["dtypes"].add(str(values.dtype))
                stats["max_abs"] = max(stats["max_abs"], float(np.max(np.abs(values.astype(np.float32)))))
    if any(len(value) != 1 for value in scenes_by_split.values()):
        raise ValueError("Scene leakage detected")
    report = {
        "status": "passed", "views": len(views), "scenes": len(scenes_by_split),
        "image_token_counts": dict(image_token_counts), "max_embedding_alignment_abs_error": max_alignment_error,
        "stages": {name: {"count": stats["count"], "dimensions": sorted(stats["dimensions"]),
                          "dtypes": sorted(stats["dtypes"]), "max_abs": stats["max_abs"]}
                   for name, stats in stages.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
