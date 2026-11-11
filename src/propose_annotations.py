"""Generate reviewable object-box proposals from a frozen Grounding DINO model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from .build_dataset import read_jsonl, write_jsonl


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0")
MODEL_PATH = DATA_ROOT / "models/grounding-dino-tiny"
PROMPT = "car. tree. traffic sign. lamp post."
CATEGORIES = {
    "car": "Car", "tree": "Tree", "traffic sign": "Traffic Sign",
    "street sign": "Traffic Sign", "stop sign": "Traffic Sign",
    "lamp post": "Lamp Post", "street light": "Lamp Post",
}


def category(label):
    label = label.lower().strip(" .")
    for phrase, canonical in sorted(CATEGORIES.items(), key=lambda item: -len(item[0])):
        if phrase in label:
            return canonical
    return None


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0


def clean_detections(result, size, max_objects):
    width, height = size
    proposed = []
    for score, label, tensor in zip(result["scores"], result["text_labels"], result["boxes"]):
        label = category(label)
        box = [round(float(v), 2) for v in tensor.tolist()]
        if label is None or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            continue
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            continue
        proposed.append({"object_category": label, "bbox": box, "detection_score": round(float(score), 4)})
    selected = []
    for obj in sorted(proposed, key=lambda item: item["detection_score"], reverse=True):
        if any(iou(obj["bbox"], old["bbox"]) > 0.5 for old in selected):
            continue
        selected.append(obj)
        if len(selected) == max_objects:
            break
    counts = Counter()
    for obj in selected:
        counts[obj["object_category"]] += 1
        obj["object_id"] = f"{obj['object_category'].lower().replace(' ', '_')}_{counts[obj['object_category']]}"
        obj["annotation_source"] = "Grounding DINO tiny proposal; visual review required"
    return selected


def draw_preview(record, output):
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in record["objects"]:
        box = obj["bbox"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 12)),
                  f"{obj['object_category']} {obj['detection_score']:.2f}", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_ROOT / "data/annotations/smoke10.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.23)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_proposals.jsonl")
    previews = args.preview_dir or output.with_name(output.stem + "_previews")
    if output.exists():
        raise FileExistsError(output)
    records = read_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    counts = Counter()
    for idx, record in enumerate(records, start=1):
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"], threshold=args.threshold,
                text_threshold=args.text_threshold, target_sizes=[(image.height, image.width)],
            )[0]
        record["objects"] = clean_detections(result, record["image_size"], args.max_objects)
        record["annotation_status"] = "unreviewed_proposal"
        counts[len(record["objects"])] += 1
        draw_preview(record, previews / f"{idx:03d}_{record['image_id']}.jpg")
        print(f"{idx}/{len(records)} {record['image_id']}: {len(record['objects'])} proposals", flush=True)
    write_jsonl(output, records)
    print(json.dumps({"output": str(output), "proposals_per_image": dict(counts),
                      "model": str(args.model), "threshold": args.threshold}, indent=2))


if __name__ == "__main__":
    main()
