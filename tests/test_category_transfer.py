import unittest

import numpy as np

from src.v01 import scene_sample


class CategoryTransferTests(unittest.TestCase):
    def test_bootstrap_samples_whole_scenes(self):
        rows = [{"scene_id": "a"}, {"scene_id": "a"}, {"scene_id": "b"}]
        sampled = scene_sample(rows, 42)
        self.assertEqual(int(np.sum(sampled == 0)), int(np.sum(sampled == 1)))
