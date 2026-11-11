import unittest

import numpy as np

from src.v01 import design


class SubspaceSeparationTests(unittest.TestCase):
    def test_test_outlier_does_not_change_training_scaler(self):
        train = [{"x": np.array([0., 1.])}, {"x": np.array([1., 0.])}, {"x": np.array([2., 2.])}]
        val = [{"x": np.array([1., 1.])}]
        a, _, _ = design(train, val, [{"x": np.array([1000., 1000.])}], "x")
        b, _, _ = design(train, val, [{"x": np.array([-1000., -1000.])}], "x")
        np.testing.assert_allclose(a, b)
