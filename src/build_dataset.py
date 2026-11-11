"""Prepare real street-view annotation templates and geometry-controlled views."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = (
    "original", "mirror", "shift_left", "shift_right", "shift_up",
    "shift_down", "zoom_in", "zoom_out",
)
HEADINGS = {"0", "90", "180", "270"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scene_id(image_id):
    # GoogleSV IDs end with a camera heading; all headings of one panorama stay together.
    stem, separator, heading = image_id.rpartition("_")
    return stem if separator and heading in HEADINGS else image_id


def city_id(image_id):
    return image_id.split("_", 1)[0]


def sample_manifest(source, count, seed, excluded_scenes=None):
    excluded_scenes = set(excluded_scenes or [])
    by_scene = {}
    for row in read_jsonl(source):
        image_id = row["id"]
        path = Path(row["image"])
        if path.is_file() and scene_id(image_id) not in excluded_scenes:
            by_scene.setdefault(scene_id(image_id), []).append((image_id, path))
    if len(by_scene) < count:
        raise ValueError(f"Only {len(by_scene)} readable scenes; requested {count}")
    rng = random.Random(seed)
    selected_scenes = rng.sample(sorted(by_scene), count)
    records = []
    for sid in selected_scenes:
        image_id, path = sorted(by_scene[sid], key=lambda pair: pair[0])[0]
        with Image.open(path) as image:
            width, height = image.size
        records.append({
            "image_id": image_id, "scene_id": sid, "city_id": city_id(image_id),
            "image_path": str(path), "image_size": [width, height], "objects": [],
        })
    return records


def validate_annotations(records):
    seen = set()
    for record in records:
        image_id = record["image_id"]
        if image_id in seen:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen.add(image_id)
        if record["scene_id"] != scene_id(image_id):
            raise ValueError(f"Inconsistent scene_id: {image_id}")
        path = Path(record["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        if list(record["image_size"]) != [width, height]:
            raise ValueError(f"Image dimensions changed: {image_id}")
        if len(record["objects"]) > 3:
            raise ValueError(f"More than three objects in {image_id}")
        object_ids = set()
        for obj in record["objects"]:
            oid = obj["object_id"]
            if oid in object_ids:
                raise ValueError(f"Duplicate object_id in {image_id}: {oid}")
            object_ids.add(oid)
            if not isinstance(obj.get("object_category"), str) or not obj["object_category"].strip():
                raise ValueError(f"Missing category: {image_id}/{oid}")
            box = obj["bbox"]
            if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid bbox: {image_id}/{oid}: {box}")


def split_scenes(records, seed, ratios):
    if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("split_ratios must be three nonnegative values summing to one")
    scenes = sorted({row["scene_id"] for row in records})
    random.Random(seed).shuffle(scenes)
    n_train = round(len(scenes) * ratios[0])
    n_val = round(len(scenes) * ratios[1])
    split_by_scene = {}
    for idx, sid in enumerate(scenes):
        split_by_scene[sid] = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
    return split_by_scene


def crop_for(name, size, config):
    width, height = size
    base_ratio = config["base_crop_ratio"]
    out_w, out_h = config["output_size"]
    scale = min(width / out_w, height / out_h) * base_ratio
    base_w, base_h = round(out_w * scale), round(out_h * scale)
    if name == "zoom_in":
        crop_w, crop_h = round(base_w * config["zoom_in_ratio"]), round(base_h * config["zoom_in_ratio"])
    elif name == "zoom_out":
        crop_w, crop_h = round(base_w * config["zoom_out_ratio"]), round(base_h * config["zoom_out_ratio"])
    else:
        crop_w, crop_h = base_w, base_h
    left, top = round((width - crop_w) / 2), round((height - crop_h) / 2)
    dx, dy = round(base_w * config["shift_ratio"]), round(base_h * config["shift_ratio"])
    if name == "shift_left":
        left -= dx
    elif name == "shift_right":
        left += dx
    elif name == "shift_up":
        top -= dy
    elif name == "shift_down":
        top += dy
    box = [left, top, left + crop_w, top + crop_h]
    if crop_w <= 0 or crop_h <= 0 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        return None
    return box


def map_bbox(box, crop, output_size, mirror=False):
    left, top, right, bottom = crop
    out_w, out_h = output_size
    x1 = (box[0] - left) * out_w / (right - left)
    y1 = (box[1] - top) * out_h / (bottom - top)
    x2 = (box[2] - left) * out_w / (right - left)
    y2 = (box[3] - top) * out_h / (bottom - top)
    return [out_w - x2, y1, out_w - x1, y2] if mirror else [x1, y1, x2, y2]


def make_view(record, name, config):
    size = record["image_size"]
    crop = crop_for(name, size, config)
    if crop is None:
        return None, "outside_image"
    if any(not (crop[0] <= box[0] < box[2] <= crop[2] and crop[1] <= box[1] < box[3] <= crop[3])
           for box in (obj["bbox"] for obj in record["objects"])):
        return None, "object_clipped"
    out_w, out_h = config["output_size"]
    objects = []
    for obj in record["objects"]:
        mapped = map_bbox(obj["bbox"], crop, (out_w, out_h), name == "mirror")
        center = [(mapped[0] + mapped[2]) / (2 * out_w), (mapped[1] + mapped[3]) / (2 * out_h)]
        objects.append({**obj, "bbox": mapped, "normalized_center": center})
    return {
        "image_id": record["image_id"], "scene_id": record["scene_id"],
        "city_id": record["city_id"], "source_image_path": record["image_path"],
        "annotation_status": record.get("annotation_status", "unspecified"),
        "transformation": name, "crop_xyxy": crop, "mirror": name == "mirror",
        "source_size": size, "output_size": [out_w, out_h], "objects": objects,
    }, None


def relative_relations(objects):
    relations = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            dx = a["normalized_center"][0] - b["normalized_center"][0]
            dy = a["normalized_center"][1] - b["normalized_center"][1]
            relations.append({
                "object_a": a["object_id"], "object_b": b["object_id"],
                "delta": [dx, dy],
                "horizontal": "left" if dx < 0 else "right" if dx > 0 else "aligned",
                "vertical": "above" if dy < 0 else "below" if dy > 0 else "aligned",
            })
    return relations


def render_view(view, destination):
    with Image.open(view["source_image_path"]) as source:
        image = source.convert("RGB").crop(tuple(view["crop_xyxy"]))
    if view["mirror"]:
        image = ImageOps.mirror(image)
    image = image.resize(tuple(view["output_size"]), Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=95)


def make_contact_sheets(records, directory, page_size=25):
    directory.mkdir(parents=True, exist_ok=True)
    width, height = 210, 185
    columns = 5
    for start in range(0, len(records), page_size):
        page = records[start:start + page_size]
        rows = (len(page) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * width, rows * height), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, record in enumerate(page):
            x, y = (idx % columns) * width, (idx // columns) * height
            with Image.open(record["image_path"]) as source:
                thumbnail = ImageOps.contain(source.convert("RGB"), (200, 150))
            sheet.paste(thumbnail, (x + 5, y + 5))
            draw.text((x + 5, y + 158), f"{start + idx:03d} {record['image_id'][:22]}", fill="black")
        sheet.save(directory / f"page_{start // page_size + 1:02d}.jpg", quality=90)


def run_sample(args, config):
    if args.annotations.exists():
        raise FileExistsError(f"Annotation file already exists: {args.annotations}")
    excluded = {row["scene_id"] for path in (args.exclude or []) for row in read_jsonl(path)}
    records = sample_manifest(args.source or config["source_manifest"], args.count or config["sample_size"],
                              args.seed if args.seed is not None else config["seed"], excluded)
    write_jsonl(args.annotations, records)
    make_contact_sheets(records, args.contact_sheets)
    print(f"Sampled {len(records)} real scenes; annotation template: {args.annotations}")


def run_build(args, config):
    records = read_jsonl(args.annotations)
    validate_annotations(records)
    labeled = [r for r in records if len(r["objects"]) >= 2]
    if not labeled:
        raise ValueError("No scenes with at least two annotated objects; annotate and review the template first")
    split = split_scenes(labeled, config["seed"], config["split_ratios"])
    views, skipped, rows = [], Counter(), []
    for record in labeled:
        for obj in record["objects"]:
            box = obj["bbox"]
            rows.append({k: record[k] for k in ("image_id", "scene_id", "city_id", "image_path")}
                        | {**obj, "normalized_center": [(box[0] + box[2]) / (2 * record["image_size"][0]),
                                                         (box[1] + box[3]) / (2 * record["image_size"][1])]})
        for name in TRANSFORMS:
            view, reason = make_view(record, name, config)
            if reason:
                skipped[f"{name}:{reason}"] += 1
                continue
            view["split"] = split[record["scene_id"]]
            view["relations"] = relative_relations(view["objects"])
            view["image_path"] = str(args.output / "transformed" / f"{record['image_id']}__{name}.jpg")
            render_view(view, Path(view["image_path"]))
            views.append(view)
    if not views:
        raise ValueError("All views were skipped; inspect object boxes and crop settings")
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "annotations.jsonl", rows)
    write_jsonl(args.output / "transformations.jsonl", views)
    for label in ("train", "val", "test"):
        write_jsonl(args.output / f"{label}.jsonl", [v for v in views if v["split"] == label])
    summary = {"labeled_scenes": len(labeled), "views": len(views), "splits": dict(Counter(v["split"] for v in views)),
               "skipped": dict(skipped), "config": config}
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_review(args):
    records = read_jsonl(args.annotations)
    validate_annotations(records)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not record["objects"]:
            continue
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in record["objects"]:
            draw.rectangle(obj["bbox"], outline="red", width=2)
            draw.text((obj["bbox"][0], obj["bbox"][1]), f"{obj['object_id']} {obj['object_category']}", fill="red")
        image.save(out / f"{record['image_id']}.jpg", quality=95)
    print(f"Wrote annotation previews to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "review", "build"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v0.yaml")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exclude", type=Path, action="append",
                        help="Existing annotation manifest whose scenes must not be sampled; repeatable")
    parser.add_argument("--contact-sheets", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = Path(config.get("data_root", ROOT / "data"))
    if args.annotations is None:
        args.annotations = data_root / "annotations/annotation_template.jsonl"
    if args.contact_sheets is None:
        args.contact_sheets = data_root / "annotations/contact_sheets"
    if args.output is None:
        args.output = data_root / ("annotations/review" if args.command == "review" else "splits")
    if args.command == "sample":
        run_sample(args, config)
    elif args.command == "review":
        run_review(args)
    else:
        run_build(args, config)


if __name__ == "__main__":
    main()
