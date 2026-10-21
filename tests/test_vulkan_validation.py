"""Numerical screening must reject plausible-looking but changed distributions."""
from array import array
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_vulkan import compare_logits


class VulkanNumericalTests(unittest.TestCase):
    def test_identical_logits_pass(self):
        result = compare_logits(array("f", [1, 2, -3]), array("f", [1, 2, -3]))
        self.assertTrue(result["logits_close"] and result["kl_passed"])
        self.assertEqual(result["softmax_kl"], 0)

    def test_changed_distribution_rejected(self):
        result = compare_logits(array("f", [1, 2]), array("f", [2, 1]))
        self.assertFalse(result["logits_close"] or result["kl_passed"])

    def test_common_offset_still_fails_logit_gate(self):
        result = compare_logits(array("f", [1, 2]), array("f", [2, 3]))
        self.assertFalse(result["logits_close"])
        self.assertTrue(result["kl_passed"])

    def test_nonfinite_or_wrong_vocabulary_rejected(self):
        for a, b in [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                compare_logits(array("f", a), array("f", b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
