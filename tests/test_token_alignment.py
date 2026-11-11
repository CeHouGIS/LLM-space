import unittest

import numpy as np

from src.token_alignment import overlap_mask, patch_xy, restore_window_order, token_boxes, verify_image_tokens


class TokenAlignmentTest(unittest.TestCase):
    def test_patch_geometry_and_mask(self):
        boxes = token_boxes(4, 4, (56, 56))
        self.assertEqual([patch_xy(i, 4, 4) for i in range(8)],
                         [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)])
        self.assertEqual(np.flatnonzero(overlap_mask(boxes, (28, 0, 42, 14))).tolist(), [4])
        self.assertEqual(np.flatnonzero(overlap_mask(token_boxes(4, 4, (56, 56), merged=True),
                                                         (28, 0, 42, 14))).tolist(), [1])

    def test_window_inverse(self):
        original = np.arange(16)
        permutation = np.array([2, 0, 3, 1])
        windowed = original.reshape(4, 4)[permutation].reshape(-1)
        np.testing.assert_array_equal(restore_window_order(windowed, permutation), original)
        merged = np.arange(4)[permutation]
        np.testing.assert_array_equal(restore_window_order(merged, permutation, merged=True), np.arange(4))

    def test_image_token_count_and_contiguity(self):
        np.testing.assert_array_equal(verify_image_tokens([7, 9, 9, 9, 9, 8], 9, (1, 4, 4), 2), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            verify_image_tokens([9, 8, 9, 9, 9], 9, (1, 4, 4), 2)


if __name__ == "__main__":
    unittest.main()
