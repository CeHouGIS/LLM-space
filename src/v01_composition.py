"""Extract real two-step image transforms and evaluate composition on held-out scenes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .analyze import DATA_ROOT, SEEDS, map_prediction
from .build_dataset import map_bbox, read_jsonl
from .v01 import arrays, design, fit, load_cache, scene_sample, write_csv


def make_compositions(manifest, output, limit):
    views = read_jsonl(manifest)
    by_key = {(v["image_id"], v["transformation"]): v for v in views if v["split"] == "test"}
    scenes = sorted({v["scene_id"] for v in views if v["split"] == "test"})
    output.mkdir(parents=True, exist_ok=True)
    composed = []
    for scene in scenes:
        candidates = [v for v in views if v["split"] == "test" and v["scene_id"] == scene
                      and v["transformation"] == "original"]
        if not candidates:
            continue
        original = candidates[0]
        for first_name, second_name, crop, mirror in (
            ("mirror", "mirror", [0, 0, 400, 300], True),
            ("shift_right", "shift_up", [0, 15, 400, 300], False),
        ):
            first = by_key.get((original["image_id"], first_name))
            if first is None or any(not (crop[0] <= o["bbox"][0] < o["bbox"][2] <= crop[2] and
                                         crop[1] <= o["bbox"][1] < o["bbox"][3] <= crop[3]) for o in first["objects"]):
                continue
            objects = []
            for obj in first["objects"]:
                mapped = map_bbox(obj["bbox"], crop, [400, 300], mirror)
                objects.append({**obj, "bbox": mapped, "normalized_center": [(mapped[0]+mapped[2])/800,
                                                                      (mapped[1]+mapped[3])/600]})
            name = first_name + "_then_" + second_name
            destination = output / f"{first['image_id']}__{name}.jpg"
            with Image.open(first["image_path"]) as source:
                image = source.convert("RGB").crop(tuple(crop))
                if mirror:
                    image = ImageOps.mirror(image)
                image.resize((400, 300), Image.Resampling.BICUBIC).save(destination, quality=95)
            composed.append({"image_id": first["image_id"], "scene_id": scene, "city_id": first["city_id"],
                             "split": "test", "annotation_status": first["annotation_status"],
                             "transformation": name, "first_transformation": first_name,
                             "image_path": str(destination), "crop_xyxy": crop, "mirror": mirror,
                             "source_size": [400, 300], "output_size": [400, 300], "objects": objects,
                             "original_crop": original["crop_xyxy"], "first_crop": first["crop_xyxy"],
                             "first_mirror": first["mirror"]})
        if len({v["scene_id"] for v in composed}) >= limit:
            break
    return composed


def extract(composed, model_path, destination):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from .extract_features import extract_one

    destination.mkdir(parents=True, exist_ok=True)
    pending = [view for view in composed if not (destination / f"{view['image_id']}__{view['transformation']}.npz").exists()]
    if not pending:
        return
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", low_cpu_mem_usage=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index, view in enumerate(pending, start=1):
        arrays, metadata = extract_one(model, processor, view, "cuda:0", [16, 31], [14])
        path = destination / f"{view['image_id']}__{view['transformation']}.npz"
        np.savez(path, **arrays, metadata_json=json.dumps(metadata))
        print(f"Composition feature {index}/{len(pending)}: {path.name}", flush=True)


def evaluate(composed, original_features, composition_features, output):
    data, _ = load_cache(original_features, stages=("encoder_16", "encoder_31", "llm_14"))
    rows = []
    for stage, original_rows in data.items():
        train = [r for r in original_rows if r["split"] == "train" and r["transformation"] == "original"]
        val = [r for r in original_rows if r["split"] == "val" and r["transformation"] == "original"]
        test_original = {(r["image_id"], r["object_id"]): r for r in original_rows
                         if r["split"] == "test" and r["transformation"] == "original"}
        with_features = []
        for view in composed:
            path = composition_features / f"{view['image_id']}__{view['transformation']}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                for index, obj in enumerate(view["objects"]):
                    with_features.append((view, obj, payload[stage][index].astype(np.float32)))
        for seed in SEEDS:
            a, b, c = design(train, val, [dict(x=feature) for _, _, feature in with_features], "x", 128)
            model, _ = fit(a, arrays(train, "y"), b, arrays(val, "y"), seed, scene_sample(train, seed))
            original_design = design(train, val, list(test_original.values()), "x", 128)[2]
            original_keys = list(test_original)
            original_prediction = dict(zip(original_keys, model.predict(original_design)))
            predictions = model.predict(c)
            for (view, obj, _), prediction in zip(with_features, predictions):
                key = view["image_id"], obj["object_id"]
                base = original_prediction[key]
                first = map_prediction(base, view["original_crop"], view["first_crop"], view["first_mirror"])
                composed_expected = map_prediction(first, [0, 0, 400, 300], view["crop_xyxy"], view["mirror"])
                truth = np.asarray(obj["normalized_center"])
                rows.append({"stage": stage, "seed": seed, "scene_id": view["scene_id"],
                             "image_id": view["image_id"], "object_id": obj["object_id"],
                             "transformation": view["transformation"], "annotation_status": view["annotation_status"],
                             "true_x": truth[0], "true_y": truth[1], "pred_x": prediction[0], "pred_y": prediction[1],
                             "expected_x": composed_expected[0], "expected_y": composed_expected[1],
                             "coordinate_mae": np.abs(prediction - truth).mean(),
                             "composition_mae": np.abs(prediction - composed_expected).mean()})
    write_csv(output / "metrics" / "e2_transformation_composition.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/pilot200_final/transformations.jsonl")
    parser.add_argument("--features", type=Path, default=DATA_ROOT / "outputs/v0/pilot200_features")
    parser.add_argument("--model", type=Path, default=DATA_ROOT / "models/qwen2.5-vl-7b")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/v0_1")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    composed = make_compositions(args.manifest, args.output / "composition_images", args.limit)
    if not composed:
        raise ValueError("No valid two-step transforms for held-out objects")
    (args.output / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifests" / "compositions.jsonl"
    manifest.write_text("".join(json.dumps(v) + "\n" for v in composed), encoding="utf-8")
    extract(composed, args.model, args.output / "composition_features")
    rows = evaluate(composed, args.features, args.output / "composition_features", args.output)
    print(json.dumps({"views": len(composed), "objects": len(rows) // 9,
                      "mean_composition_mae": float(np.mean([r["composition_mae"] for r in rows]))}))


if __name__ == "__main__":
    main()
