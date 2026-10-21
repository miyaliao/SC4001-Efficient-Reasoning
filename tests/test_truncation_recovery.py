import tempfile
import unittest
from pathlib import Path

from sc4001.pilot import atomic_json
from sc4001.mixed_protocol import case_path
from scripts.run_truncation_recovery import cohort


class RecoveryTests(unittest.TestCase):
    def test_cohort_only_original_truncated_slow(self):
        rows = [
            {"id":"gsmplus-000001","benchmark":"gsm_plus"},
            {"id":"math500-0001","benchmark":"math_500"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for row,truncated in zip(rows,[True,False]):
                atomic_json(case_path(root,row["id"],"slow"),{
                    "id":row["id"],"mode":"slow","fingerprint":"fixed",
                    "status":"ok","response":"x","output_tokens":4096,
                    "truncated":truncated
                })
            selected=cohort(rows,root,{"fingerprint":"fixed"})
            self.assertEqual([r["id"] for r in selected],["gsmplus-000001"])

    def test_missing_baseline_case_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"incomplete"):
                cohort([{"id":"gsmplus-000001"}],Path(tmp),{"fingerprint":"fixed"})
