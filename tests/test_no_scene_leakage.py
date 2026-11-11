import unittest

from src.v01 import stable_seed


class SceneLeakageTests(unittest.TestCase):
    def test_stable_seed_is_reproducible(self):
        self.assertEqual(stable_seed("scene-a"), stable_seed("scene-a"))
        self.assertNotEqual(stable_seed("scene-a"), stable_seed("scene-b"))
