"""Verify transformed boxes/relations and render per-scene visual audit sheets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .build_dataset import read_jsonl, relative_relations


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")


def check_view(view):
    width, height = view["output_size"]
    with Image.open(view["image_path"]) as image:
        if image.size != (width, height):
            raise ValueError(f"Incorrect output size: {view['image_path']}")
    for obj in view["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-bounds box: {view['image_path']} {obj['object_id']}")
        expected = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height)]
        if any(abs(a - b) > 1e-6 for a, b in zip(expected, obj["normalized_center"])):
            raise ValueError(f"Incorrect normalized center: {view['image_path']} {obj['object_id']}")
    if relative_relations(view["objects"]) != view["relations"]:
        raise ValueError(f"Incorrect pair labels: {view['image_path']}")


def check_scene(views):
    by_name = {view["transformation"]: view for view in views}
    original = by_name.get("original")
    if original is None:
        raise ValueError("Scene has no original view")
    baseline = {(r["object_a"], r["object_b"]): r for r in original["relations"]}
    for view in views:
        for rel in view["relations"]:
            key = rel["object_a"], rel["object_b"]
            old = baseline[key]
            horizontal = ("right" if old["horizontal"] == "left" else "left"
                          if old["horizontal"] == "right" else "aligned")
            expected_horizontal = horizontal if view["mirror"] else old["horizontal"]
            if rel["horizontal"] != expected_horizontal or rel["vertical"] != old["vertical"]:
                raise ValueError(f"Relation order changed incorrectly: {view['image_id']} {view['transformation']}")


def draw_sheet(views, output):
    tile_w, tile_h = 400, 326
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(views):
        x, y = index % 4 * tile_w, index // 4 * tile_h
        with Image.open(view["image_path"]) as source:
            tile = source.convert("RGB")
        marks = ImageDraw.Draw(tile)
        for obj in view["objects"]:
            marks.rectangle(obj["bbox"], outline="red", width=2)
            marks.text((obj["bbox"][0], obj["bbox"][1]), obj["object_id"], fill="red")
        sheet.paste(tile, (x, y + 20))
        draw.text((x + 4, y + 2), view["transformation"], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "splits/smoke10_reviewed/transformations.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "splits/smoke10_reviewed/review_sheets")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()
    views = read_jsonl(args.manifest)
    by_scene = defaultdict(list)
    for view in views:
        check_view(view)
        by_scene[view["scene_id"]].append(view)
    for scene, group in by_scene.items():
        check_scene(group)
        if not args.no_sheets:
            draw_sheet(group, args.output / f"{scene}.jpg")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"scenes": len(by_scene), "views": len(views),
               "transformations": dict(Counter(view["transformation"] for view in views)),
               "status": "geometry_and_relation_checks_passed"}
    (args.output.parent / "view_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
