import unittest

import numpy as np

from src.v01 import position_features
from src.token_alignment import overlap_mask, token_boxes


class PositionBaselineTests(unittest.TestCase):
    def test_same_mask_as_real_pooling(self):
        meta = {"image_grid_thw": [1, 22, 28], "output_size": [400, 300], "image_id": "test"}
        obj = {"bbox": [100, 80, 180, 160], "object_id": "car"}
        for stage in ("encoder_31", "merger", "llm_14"):
            features, indices, count = position_features(meta, obj, stage)
            boxes = token_boxes(22, 28, [400, 300], merged=stage != "encoder_31")
            np.testing.assert_array_equal(indices, np.flatnonzero(overlap_mask(boxes, obj["bbox"])))
            self.assertEqual(count, len(boxes))
            self.assertTrue(np.isfinite(features).all())

    def test_position_features_do_not_embed_exact_box(self):
        meta = {"image_grid_thw": [1, 22, 28], "output_size": [400, 300], "image_id": "test"}
        a = {"bbox": [110, 90, 170, 145], "object_id": "car"}
        b = {"bbox": [111, 91, 169, 144], "object_id": "car"}
        pa, ia, _ = position_features(meta, a, "encoder_31")
        pb, ib, _ = position_features(meta, b, "encoder_31")
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_allclose(pa, pb)
