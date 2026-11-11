import unittest

import numpy as np

from src.analyze import map_prediction


class TransformationCompositionTests(unittest.TestCase):
    def test_two_crop_maps_compose(self):
        a, b, c = [40, 50, 360, 290], [72, 50, 392, 290], [72, 74, 392, 314]
        point = [.4, .5]
        first = map_prediction(point, a, b, False)
        second = map_prediction(first, b, c, False)
        np.testing.assert_allclose(second, map_prediction(point, a, c, False))
