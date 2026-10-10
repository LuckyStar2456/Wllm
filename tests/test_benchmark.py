"""Guard against misleading performance summaries."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class SummaryTests(unittest.TestCase):
    def test_diagnostics_do_not_enter_performance_aggregates(self):
        result = benchmark.summarize([
            {"case": "a", "profile": "b", "ok": True, "diagnostic": True,
             "metrics": {"elapsed_s": 0.1, "decode_average_tps": 1000}},
        ])["a/b"]
        self.assertEqual(result, {"samples": 0, "failures": 1, "metrics": {}})

    def test_failed_runs_are_not_fast_successes(self):
        result = benchmark.summarize([
            {"case": "a", "profile": "b", "ok": True, "metrics": {"elapsed_s": 10, "decode_average_tps": None}},
            {"case": "a", "profile": "b", "ok": False, "metrics": {"elapsed_s": 0.1, "decode_average_tps": 1000}},
            {"case": "a", "profile": "b", "ok": True, "metrics": {"elapsed_s": 12, "decode_average_tps": 5}},
        ])["a/b"]
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["metrics"]["elapsed_s"], {"n": 2, "median": 11, "min": 10, "max": 12})
        self.assertEqual(result["metrics"]["decode_average_tps"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
