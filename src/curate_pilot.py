"""Select a 200-scene exploratory pilot from detector proposals with geometry checks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from .build_dataset import make_view, read_jsonl, validate_annotations, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")
DEFAULT_PROPOSALS = [
    DATA_ROOT / "annotations/pilot200_proposals.jsonl",
    DATA_ROOT / "annotations/extra150_proposals.jsonl",
    DATA_ROOT / "annotations/extra100_proposals.jsonl",
]


def eligible(record, config):
    objects = [obj for obj in record["objects"]
               if make_view({**record, "objects": [obj]}, "original", config)[0] is not None]
    if len(objects) < 2:
        return None
    objects = sorted(objects, key=lambda obj: obj["detection_score"], reverse=True)[:3]
    candidate = {**record, "objects": objects, "annotation_status": "machine_proposal_unreviewed"}
    return candidate


def rank(record):
    scores = sorted((obj["detection_score"] for obj in record["objects"]), reverse=True)
    categories = {obj["object_category"] for obj in record["objects"]}
    return sum(scores[:2]) + 0.04 * len(categories) + (0.05 if "Traffic Sign" in categories else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, action="append")
    parser.add_argument("--reviewed", type=Path, default=DATA_ROOT / "annotations/smoke10_reviewed.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "annotations/pilot200_annotations.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v0.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reviewed = read_jsonl(args.reviewed)
    if len(reviewed) > args.count:
        raise ValueError("More reviewed scenes than requested pilot size")
    selected_ids = {row["scene_id"] for row in reviewed}
    seen = set(selected_ids)
    candidates = []
    for path in args.proposal or DEFAULT_PROPOSALS:
        for record in read_jsonl(path):
            sid = record["scene_id"]
            if sid in seen:
                continue
            seen.add(sid)
            candidate = eligible(record, config)
            if candidate:
                candidates.append(candidate)
    if len(candidates) + len(reviewed) < args.count:
        raise ValueError(f"Only {len(candidates) + len(reviewed)} eligible scenes, requested {args.count}")
    candidates.sort(key=lambda row: (-rank(row), row["image_id"]))
    picked = list(reviewed)
    cities = sorted({row["city_id"] for row in reviewed + candidates})
    minimum_per_city = min(30, args.count // len(cities))
    for city in cities:
        current = sum(row["city_id"] == city for row in picked)
        needed = max(0, minimum_per_city - current)
        city_candidates = [row for row in candidates if row["city_id"] == city and row["scene_id"] not in
                           {picked_row["scene_id"] for picked_row in picked}]
        picked.extend(city_candidates[:needed])
    picked_ids = {row["scene_id"] for row in picked}
    picked.extend(row for row in candidates if row["scene_id"] not in picked_ids)
    picked = picked[:args.count]
    validate_annotations(picked)
    write_jsonl(args.output, picked)
    summary = {
        "scenes": len(picked), "review_status": dict(Counter(row["annotation_status"] for row in picked)),
        "cities": dict(Counter(row["city_id"] for row in picked)),
        "categories": dict(Counter(obj["object_category"] for row in picked for obj in row["objects"])),
        "candidate_pool": len(candidates), "selection": "original-crop validity, city minimum, then detector confidence",
    }
    (args.output.parent / "pilot200_curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
