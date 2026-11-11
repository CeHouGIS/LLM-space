"""Verified Qwen2.5-VL visual token geometry for one static image."""

import numpy as np


def patch_xy(index, grid_h, grid_w, merge_size=2):
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Visual grid must be divisible by spatial_merge_size")
    block_size = merge_size**2
    block, offset = divmod(index, block_size)
    block_y, block_x = divmod(block, grid_w // merge_size)
    within_y, within_x = divmod(offset, merge_size)
    return block_x * merge_size + within_x, block_y * merge_size + within_y


def token_boxes(grid_h, grid_w, image_size, merge_size=2, merged=False):
    width, height = image_size
    if merged:
        rows, cols = grid_h // merge_size, grid_w // merge_size
        return np.array([
            [x * width / cols, y * height / rows, (x + 1) * width / cols, (y + 1) * height / rows]
            for y in range(rows) for x in range(cols)
        ], dtype=np.float32)
    return np.array([
        [x * width / grid_w, y * height / grid_h,
         (x + 1) * width / grid_w, (y + 1) * height / grid_h]
        for index in range(grid_h * grid_w)
        for x, y in [patch_xy(index, grid_h, grid_w, merge_size)]
    ], dtype=np.float32)


def overlap_mask(token_rectangles, bbox):
    boxes = np.asarray(token_rectangles)
    left = np.maximum(boxes[:, 0], bbox[0])
    top = np.maximum(boxes[:, 1], bbox[1])
    right = np.minimum(boxes[:, 2], bbox[2])
    bottom = np.minimum(boxes[:, 3], bbox[3])
    return (right > left) & (bottom > top)


def restore_window_order(hidden_states, window_index, merge_size=2, merged=False):
    """Undo the actual window permutation in Qwen2.5-VL visual.forward."""
    order = np.argsort(np.asarray(window_index))
    if merged:
        return hidden_states[order]
    shape = hidden_states.shape
    if shape[0] != len(order) * merge_size**2:
        raise ValueError("Visual sequence length does not match window permutation")
    return hidden_states.reshape(len(order), merge_size**2, *shape[1:])[order].reshape(shape)


def verify_processor_mapping(processor):
    """Perturb a known pixel square and confirm only its predicted patch IDs change."""
    from PIL import Image, ImageDraw

    size = (392, 280)
    blank = Image.new("RGB", size, "gray")
    altered = blank.copy()
    ImageDraw.Draw(altered).rectangle((140, 112, 167, 139), fill="red")
    a = processor.image_processor(images=blank, return_tensors="np")
    b = processor.image_processor(images=altered, return_tensors="np")
    grid = tuple(int(v) for v in a["image_grid_thw"][0])
    if grid != (1, 20, 28) or tuple(b["image_grid_thw"][0]) != grid:
        raise ValueError(f"Unexpected processor grid {grid}")
    changed = set(np.flatnonzero(np.any(np.abs(a["pixel_values"] - b["pixel_values"]) > 1e-6, axis=1)))
    expected = {idx for idx in range(20 * 28) if 10 <= patch_xy(idx, 20, 28)[0] <= 11
                and 8 <= patch_xy(idx, 20, 28)[1] <= 9}
    if changed != expected:
        raise ValueError(f"Processor patch mapping failed: changed={sorted(changed)} expected={sorted(expected)}")
    return {"grid_thw": list(grid), "changed_patch_ids": sorted(int(index) for index in changed),
            "status": "verified"}


def verify_image_tokens(input_ids, image_token_id, grid_thw, merge_size):
    ids = np.asarray(input_ids)
    if ids.ndim != 1:
        raise ValueError("Expected one unbatched input sequence")
    positions = np.flatnonzero(ids == image_token_id)
    t, h, w = [int(v) for v in grid_thw]
    expected = t * h * w // merge_size**2
    if len(positions) != expected or not np.array_equal(positions, np.arange(positions[0], positions[0] + expected)):
        raise ValueError(f"Image placeholder alignment failed: {len(positions)} versus {expected}")
    return positions
