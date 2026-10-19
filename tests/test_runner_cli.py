"""Subprocess-level dry-run test, without torch or GPU."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunnerCLITests(unittest.TestCase):
    def test_dry_run_200_pairs_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model = tmp / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "tokenizer_config.json").write_text("{}")
            data = tmp / "pilot.jsonl"
            with data.open("w", encoding="utf-8") as out:
                for i in range(200):
                    out.write(json.dumps({
                        "id": f"gsm8k-train-{i:06d}", "question": "A\u2028B",
                        "answer": "#### 1"
                    }, ensure_ascii=False) + "\n")
            cfg = tmp / "pilot.yaml"
            cfg.write_text(
                f"model_path: {model}\ndataset: {data}\nseed: 42\nmodes: [fast, slow]\n"
                "decode:\n  do_sample: true\n  max_new_tokens: 4096\n"
                "  sampling:\n"
                "    fast: {temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0}\n"
                "    slow: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}\n"
            )
            dest = tmp / "outputs"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_pilot.py"), "--config", str(cfg),
                "--output-dir", str(dest), "--dry-run",
            ], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"pending_tasks": 400', result.stdout)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
