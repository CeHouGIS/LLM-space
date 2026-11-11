"""Render all pilot detector boxes in numbered pages for visual quality review."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .build_dataset import read_jsonl


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0/data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DATA_ROOT / "annotations/pilot200_annotations.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "annotations/pilot200_review_pages")
    args = parser.parse_args()
    records = read_jsonl(args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 400, 430
    for start in range(0, len(records), 25):
        page = Image.new("RGB", (tile_w * 5, tile_h * 5), "white")
        draw = ImageDraw.Draw(page)
        for local_index, row in enumerate(records[start:start + 25]):
            col, line = local_index % 5, local_index // 5
            x, y = col * tile_w, line * tile_h
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            marks = ImageDraw.Draw(image)
            for obj in row["objects"]:
                marks.rectangle(obj["bbox"], outline="red", width=2)
                marks.text((obj["bbox"][0], max(0, obj["bbox"][1] - 12)),
                           f"{obj['object_category']} {obj.get('detection_score', 0):.2f}", fill="red")
            page.paste(image, (x, y + 20))
            label = f"{start + local_index:03d} {row['city_id']} {row['annotation_status']}"
            draw.text((x + 4, y + 2), label, fill="black")
        filename = args.output / f"page_{start // 25 + 1:02d}.jpg"
        page.save(filename, quality=90)
        print(filename, flush=True)


if __name__ == "__main__":
    main()
