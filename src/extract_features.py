"""Extract frozen Qwen2.5-VL object features after explicit token-alignment checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, __version__ as transformers_version

from .build_dataset import read_jsonl
from .token_alignment import (overlap_mask, restore_window_order, token_boxes,
                              verify_image_tokens, verify_processor_mapping)


DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0")
MODEL_PATH = DATA_ROOT / "models/qwen2.5-vl-7b"
NEUTRAL_PROMPT = "Describe the image."


def encode_image(processor, image):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": NEUTRAL_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def pool(hidden, boxes, objects):
    pooled = []
    for obj in objects:
        mask = overlap_mask(boxes, obj["bbox"])
        if not mask.any():
            raise ValueError(f"No visual token overlaps object {obj['object_id']}")
        pooled.append(hidden[mask].astype(np.float32).mean(axis=0))
    result = np.stack(pooled)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite object feature after pooling")
    return result.astype(np.float16) if np.max(np.abs(result)) < 65000 else result


def extract_one(model, processor, view, device, visual_layers, language_layers):
    with Image.open(view["image_path"]) as source:
        image = source.convert("RGB")
    inputs = encode_image(processor, image)
    grid = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
    t, h, w = grid
    if t != 1:
        raise ValueError(f"Expected one static image grid, got {grid}")
    visual = model.model.visual
    merge_size = visual.spatial_merge_size
    window_index, _ = visual.get_window_index(inputs["image_grid_thw"])
    window_index = window_index.cpu().numpy()
    if len(window_index) != h * w // merge_size**2 or not np.array_equal(np.sort(window_index), np.arange(len(window_index))):
        raise ValueError("Visual window index is not a complete permutation")
    image_positions = verify_image_tokens(inputs["input_ids"][0].numpy(), model.config.image_token_id, grid, merge_size)
    captures = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            captures[name] = output.detach().cpu().float().numpy()
        return hook

    handles.append(visual.patch_embed.register_forward_hook(capture("encoder_patch")))
    for layer in visual_layers:
        handles.append(visual.blocks[layer].register_forward_hook(capture(f"encoder_{layer}")))
    handles.append(visual.merger.register_forward_hook(capture("merger_window")))
    handles.append(visual.register_forward_hook(capture("visual_output")))
    inputs = inputs.to(device)
    try:
        with torch.inference_mode():
            output = model.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    merger = restore_window_order(captures.pop("merger_window"), window_index, merge_size, merged=True)
    if not np.array_equal(merger, captures.pop("visual_output")):
        raise ValueError("Merger window inverse disagrees with model visual output")
    llm_input = output.hidden_states[0][0, image_positions].detach().cpu().float().numpy()
    max_embedding_error = float(np.max(np.abs(llm_input.astype(np.float32) - merger.astype(np.float32))))
    if max_embedding_error > 0.01:
        raise ValueError(f"LLM image-token embedding mismatch: {max_embedding_error}")
    patch_rects = token_boxes(h, w, view["output_size"], merge_size)
    merged_rects = token_boxes(h, w, view["output_size"], merge_size, merged=True)
    arrays = {"encoder_patch": pool(captures["encoder_patch"], patch_rects, view["objects"])}
    for layer in visual_layers:
        name = f"encoder_{layer}"
        unwindowed = restore_window_order(captures[name], window_index, merge_size)
        arrays[name] = pool(unwindowed, patch_rects, view["objects"])
    arrays["merger"] = pool(merger, merged_rects, view["objects"])
    for layer in language_layers:
        state = output.hidden_states[layer + 1][0, image_positions].detach().cpu().float().numpy()
        arrays[f"llm_{layer}"] = pool(state, merged_rects, view["objects"])
    metadata = {
        "image_id": view["image_id"], "scene_id": view["scene_id"], "split": view["split"],
        "city_id": view["city_id"], "transformation": view["transformation"], "objects": view["objects"],
        "annotation_status": view.get("annotation_status", "unspecified"),
        "crop_xyxy": view["crop_xyxy"], "source_size": view["source_size"],
        "output_size": view["output_size"], "mirror": view["mirror"],
        "image_grid_thw": grid, "image_token_count": len(image_positions),
        "image_token_positions": [int(image_positions[0]), int(image_positions[-1])],
        "embedding_alignment_max_abs_error": max_embedding_error,
        "input_resolution": view["output_size"], "processor_grid_resolution": [w * visual.patch_size, h * visual.patch_size],
        "cache_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "data/splits/smoke10/transformations.jsonl")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "outputs/v0/features")
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--alignment-only", action="store_true")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    processor_check = verify_processor_mapping(processor)
    print(f"Processor alignment: {processor_check}", flush=True)
    if args.alignment_only:
        return
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation="eager",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_layers = sorted({0, len(model.model.visual.blocks) // 2, len(model.model.visual.blocks) - 1})
    language_layers = sorted({0, len(model.model.language_model.layers) // 2,
                              len(model.model.language_model.layers) - 1})
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "model_path": str(args.model), "model_manifest": json.loads((args.model / "snapshot_manifest.json").read_text()),
        "transformers_version": transformers_version, "torch_version": torch.__version__,
        "device": args.device, "dtype": "bfloat16", "attention": "eager",
        "neutral_prompt": NEUTRAL_PROMPT, "visual_layers": visual_layers,
        "language_layers": language_layers, "processor_alignment": processor_check,
        "seed": 42, "processor_use_fast": False,
        "processor_config": processor.image_processor.to_dict(),
    }
    (args.output / "run_config.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    views = read_jsonl(args.manifest)
    if args.limit:
        views = views[:args.limit]
    for idx, view in enumerate(views, start=1):
        stem = f"{view['image_id']}__{view['transformation']}"
        destination = args.output / f"{stem}.npz"
        if destination.exists():
            print(f"{idx}/{len(views)} cached {stem}", flush=True)
            continue
        arrays, metadata = extract_one(model, processor, view, args.device, visual_layers, language_layers)
        np.savez(destination, **arrays, metadata_json=json.dumps(metadata))
        print(f"{idx}/{len(views)} {stem}: {metadata['image_token_count']} image tokens", flush=True)


if __name__ == "__main__":
    main()
