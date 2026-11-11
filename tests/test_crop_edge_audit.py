import unittest

from src.v01_crop_audit import edge_margin


class CropEdgeAuditTests(unittest.TestCase):
    def test_margin_is_minimum_distance_from_box_to_image_edge(self):
        self.assertAlmostEqual(edge_margin([3.825, 61, 169, 184], [400, 300]), 3.825)
        self.assertEqual(edge_margin([30, 40, 300, 250], [400, 300]), 30)
