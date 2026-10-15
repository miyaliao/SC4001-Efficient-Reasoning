"""Regression tests for Unicode line separators inside JSONL strings."""
import json
import tempfile
import unittest
from pathlib import Path


class JSONLUnicodeTests(unittest.TestCase):
    def test_raw_unicode_separator_does_not_split_a_record(self):
        # GSM8K text can contain Unicode separators. str.splitlines() is unsafe.
        sample = {"id": "sample-1", "question": "before\u2028after", "answer": "#### 42"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            with path.open("r", encoding="utf-8", newline="") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(records, [sample])

    def test_ascii_escaped_jsonl_is_one_physical_line(self):
        sample = {"question": "before\u2028after", "answer": "#### 42"}
        encoded = json.dumps(sample, ensure_ascii=True)
        self.assertEqual(len(encoded.splitlines()), 1)
        self.assertEqual(json.loads(encoded), sample)


if __name__ == "__main__":
    unittest.main()
