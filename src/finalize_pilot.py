"""Correct broad pole labels and choose a preregistered label-balanced scene split."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_dataset import read_jsonl, split_scenes, validate_annotations, write_jsonl


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
CORE_CATEGORIES = ("Car", "Tree", "Pole", "Traffic Sign")


def corrected_annotations(rows):
    corrected = []
    for record in rows:
        row = record.copy()
        row["objects"] = []
        for old in record["objects"]:
            obj = old.copy()
            if record["annotation_status"] != "visually_reviewed" and obj["object_category"] == "Lamp Post":
                obj["object_category"] = "Pole"
                obj["annotation_source"] = "Grounding DINO tiny pole candidate; function unverified"
            row["objects"].append(obj)
        corrected.append(row)
    return corrected


def balanced_split(rows):
    # Choose using labels only, before feature extraction or outcome inspection.
    for seed in range(42, 1042):
        split = split_scenes(rows, seed, [0.6, 0.2, 0.2])
        counts = {name: Counter(obj["object_category"] for row in rows if split[row["scene_id"]] == name
                                for obj in row["objects"])
                  for name in ("train", "val", "test")}
        if (all(counts["train"][category] >= 8 for category in CORE_CATEGORIES)
                and all(counts[name][category] >= 4 for name in ("val", "test")
                        for category in CORE_CATEGORIES)
                and counts["train"]["Lamp Post"] >= 2):
            return seed, split, counts
    raise ValueError("Cannot find a pre-outcome scene split with sufficient category support")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DATA_ROOT / "annotations/pilot200_annotations.jsonl")
    parser.add_argument("--views", type=Path, default=DATA_ROOT / "splits/pilot200/transformations.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "splits/pilot200_final")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = corrected_annotations(read_jsonl(args.annotations))
    validate_annotations(records)
    seed, split, categories = balanced_split(records)
    by_id = {row["image_id"]: row for row in records}
    views = read_jsonl(args.views)
    for view in views:
        record = by_id[view["image_id"]]
        mapping = {obj["object_id"]: obj for obj in record["objects"]}
        view["objects"] = [
            {**obj, "object_category": mapping[obj["object_id"]]["object_category"],
             "annotation_source": mapping[obj["object_id"]]["annotation_source"]}
            for obj in view["objects"]
        ]
        view["annotation_status"] = record["annotation_status"]
        view["split"] = split[view["scene_id"]]
    annotations = []
    for row in records:
        width, height = row["image_size"]
        for obj in row["objects"]:
            box = obj["bbox"]
            annotations.append({k: row[k] for k in ("image_id", "scene_id", "city_id", "image_path")}
                               | {**obj, "normalized_center": [(box[0] + box[2]) / (2 * width),
                                                                (box[1] + box[3]) / (2 * height)]})
    write_jsonl(args.output / "annotations.jsonl", annotations)
    write_jsonl(args.output / "transformations.jsonl", views)
    for name in ("train", "val", "test"):
        write_jsonl(args.output / f"{name}.jsonl", [view for view in views if view["split"] == name])
    summary = {"scene_count": len(records), "view_count": len(views), "split_seed": seed,
               "scene_split_counts": dict(Counter(split.values())),
               "category_split_counts": {name: dict(count) for name, count in categories.items()},
               "review_status": dict(Counter(row["annotation_status"] for row in records)),
               "label_correction": "Unreviewed Lamp Post proposals relabeled Pole",
               "transformed_images_reused_from": str(args.views.parent / "transformed")}
    (args.output / "pilot_finalization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
