import unittest

from src.build_dataset import TRANSFORMS, crop_for, make_view, relative_relations, scene_id, split_scenes


CONFIG = {
    "base_crop_ratio": 0.8, "shift_ratio": 0.1, "zoom_in_ratio": 0.85,
    "zoom_out_ratio": 1.15, "output_size": [400, 300],
}


def fixture():
    return {
        "image_id": "Austin_pano_90", "scene_id": "Austin_pano", "city_id": "Austin",
        "image_path": "unused.jpg", "image_size": [400, 300],
        "objects": [
            {"object_id": "a", "object_category": "Car", "bbox": [100, 100, 130, 130]},
            {"object_id": "b", "object_category": "Tree", "bbox": [240, 160, 270, 200]},
        ],
    }


class GeometryTest(unittest.TestCase):
    def test_all_views_keep_boxes_inside_and_relations(self):
        record = fixture()
        for name in TRANSFORMS:
            view, reason = make_view(record, name, CONFIG)
            self.assertIsNone(reason, (name, reason))
            self.assertEqual(len(view["objects"]), 2)
            for obj in view["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                self.assertTrue(0 <= x1 < x2 <= 400)
                self.assertTrue(0 <= y1 < y2 <= 300)
                self.assertTrue(all(0 <= p <= 1 for p in obj["normalized_center"]))
            relation = relative_relations(view["objects"])[0]
            self.assertEqual(relation["horizontal"], "right" if name == "mirror" else "left")
            self.assertEqual(relation["vertical"], "above")

    def test_exact_affine_coordinate_mapping(self):
        record = fixture()
        original, _ = make_view(record, "original", CONFIG)
        mirror, _ = make_view(record, "mirror", CONFIG)
        left, _ = make_view(record, "shift_left", CONFIG)
        for a, b in zip(original["objects"], mirror["objects"]):
            self.assertAlmostEqual(b["normalized_center"][0], 1 - a["normalized_center"][0])
            self.assertAlmostEqual(b["normalized_center"][1], a["normalized_center"][1])
        for a, b in zip(original["objects"], left["objects"]):
            self.assertAlmostEqual(b["normalized_center"][0] - a["normalized_center"][0], 0.1)
            self.assertAlmostEqual(b["normalized_center"][1], a["normalized_center"][1])

    def test_clipped_object_skips_view(self):
        record = fixture()
        record["objects"][0]["bbox"] = [5, 100, 30, 130]
        view, reason = make_view(record, "original", CONFIG)
        self.assertIsNone(view)
        self.assertEqual(reason, "object_clipped")

    def test_square_source_uses_four_by_three_crop(self):
        crop = crop_for("original", [400, 400], CONFIG)
        self.assertEqual(crop, [40, 80, 360, 320])
        self.assertEqual(crop_for("zoom_out", [400, 400], CONFIG), [16, 62, 384, 338])

    def test_scene_level_split(self):
        records = [
            {"image_id": "Austin_pano_0", "scene_id": "Austin_pano"},
            {"image_id": "Austin_pano_90", "scene_id": "Austin_pano"},
        ] + [{"image_id": f"Austin_{i}_0", "scene_id": f"Austin_{i}"} for i in range(9)]
        split = split_scenes(records, 42, [0.6, 0.2, 0.2])
        self.assertEqual(len(split), 10)
        self.assertEqual(scene_id(records[0]["image_id"]), scene_id(records[1]["image_id"]))
        self.assertEqual(set(split.values()), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
