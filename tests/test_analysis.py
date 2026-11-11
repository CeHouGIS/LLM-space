import unittest

import numpy as np

from src.analyze import filter_quality, map_prediction, patch_position_readout
from src.finalize_pilot import corrected_annotations


class AnalysisTest(unittest.TestCase):
    def test_affine_prediction_mapping(self):
        original = [40, 30, 360, 270]
        shift_left = [8, 30, 328, 270]
        np.testing.assert_allclose(map_prediction([0.25, 0.7], original, shift_left, False), [0.35, 0.7])
        np.testing.assert_allclose(map_prediction([0.25, 0.7], original, original, True), [0.75, 0.7])

    def test_unreviewed_pole_is_broadly_labeled(self):
        records = [
            {"annotation_status": "machine_proposal_unreviewed",
             "objects": [{"object_id": "lamp_post_1", "object_category": "Lamp Post", "bbox": [1, 2, 3, 4]}]},
            {"annotation_status": "visually_reviewed",
             "objects": [{"object_id": "lamp_post_1", "object_category": "Lamp Post", "bbox": [1, 2, 3, 4]}]},
        ]
        corrected = corrected_annotations(records)
        self.assertEqual(corrected[0]["objects"][0]["object_category"], "Pole")
        self.assertEqual(corrected[1]["objects"][0]["object_category"], "Lamp Post")
        self.assertEqual(records[0]["objects"][0]["object_category"], "Lamp Post")

    def test_patch_position_baseline_uses_token_grid(self):
        row = {"grid": [1, 4, 4], "image_size": [56, 56], "bbox": [28, 0, 42, 14]}
        np.testing.assert_allclose(patch_position_readout(row), [0.625, 0.125])

    def test_quality_filter_keeps_scene_and_object_identity(self):
        rows = [
            {"scene_id": "a", "image_id": "a", "object_id": "big1", "bbox": [0, 0, 30, 30], "transformation": "original"},
            {"scene_id": "a", "image_id": "a", "object_id": "big2", "bbox": [50, 50, 80, 80], "transformation": "original"},
            {"scene_id": "a", "image_id": "a", "object_id": "small", "bbox": [1, 1, 10, 10], "transformation": "original"},
            {"scene_id": "a", "image_id": "a", "object_id": "big1", "bbox": [0, 0, 15, 15], "transformation": "zoom_out"},
            {"scene_id": "b", "image_id": "b", "object_id": "only1", "bbox": [0, 0, 30, 30], "transformation": "original"},
        ]
        result = filter_quality({"encoder_0": rows}, 20)["encoder_0"]
        self.assertEqual([(r["scene_id"], r["object_id"]) for r in result],
                         [("a", "big1"), ("a", "big2"), ("a", "big1")])


if __name__ == "__main__":
    unittest.main()
