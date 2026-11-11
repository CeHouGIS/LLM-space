import unittest

import numpy as np

from src.analyze import map_prediction


class TrajectoryTransformTests(unittest.TestCase):
    def test_shift_uses_actual_crop(self):
        original = [40, 50, 360, 290]
        shifted = [72, 50, 392, 290]
        np.testing.assert_allclose(map_prediction([.5, .5], original, shifted, False), [.4, .5])

    def test_mirror(self):
        crop = [40, 50, 360, 290]
        np.testing.assert_allclose(map_prediction([.2, .4], crop, crop, True), [.8, .4])
