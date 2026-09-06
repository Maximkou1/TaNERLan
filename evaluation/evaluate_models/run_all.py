"""Sequential launcher; never invoked automatically."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("model_config.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all shortlisted models sequentially.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    models = json.loads(CONFIG.read_text(encoding="utf-8"))["models"]
    failures = []
    for model in models:
        output = ROOT / "artifacts" / "models" / model["name"] / "dev_predictions.jsonl"
        command = [sys.executable, "-m", "evaluation.evaluate_models.run_model", "--model", model["name"], "--output", str(output), "--device", args.device]
        if args.max_records is not None:
            command.extend(["--max-records", str(args.max_records)])
        print("+", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures.append((model["name"], error.returncode))
            print(f"FAILED: {model['name']} (exit code {error.returncode}); continuing", flush=True)
    if failures:
        print("Failures:", ", ".join(f"{name} ({code})" for name, code in failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
