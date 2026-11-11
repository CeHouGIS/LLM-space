"""Materialize the visually reviewed smoke selection without altering proposals."""

import argparse
import json
from pathlib import Path

import yaml

from .build_dataset import TRANSFORMS, make_view, read_jsonl, validate_annotations, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=DATA_ROOT / "annotations/pilot200_proposals.jsonl")
    parser.add_argument("--selection", type=Path, default=ROOT / "configs/smoke_review.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v0.yaml")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "annotations/smoke10_reviewed.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    chosen = json.loads(args.selection.read_text(encoding="utf-8"))
    proposals = {r["image_id"]: r for r in read_jsonl(args.proposals)}
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for image_id, object_ids in chosen.items():
        if image_id not in proposals:
            raise ValueError(f"Missing image proposal: {image_id}")
        row = proposals[image_id].copy()
        available = {obj["object_id"]: obj for obj in row["objects"]}
        if len(object_ids) < 2 or len(object_ids) > 3 or not set(object_ids) <= set(available):
            raise ValueError(f"Invalid object selection for {image_id}")
        row["objects"] = [{**available[oid], "annotation_source": "Grounding DINO tiny; visually reviewed"}
                          for oid in object_ids]
        row["annotation_status"] = "visually_reviewed"
        for transformation in TRANSFORMS:
            view, reason = make_view(row, transformation, config)
            if view is None:
                raise ValueError(f"{image_id}: {transformation} invalid: {reason}")
        rows.append(row)
    validate_annotations(rows)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} reviewed scenes to {args.output}")


if __name__ == "__main__":
    main()
