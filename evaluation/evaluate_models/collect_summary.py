"""Collect completed model metrics and metadata into artifacts/summary.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "artifacts" / "models"
    rows = []
    for metrics_path in sorted(source.glob("*/dev_metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata_path = metrics_path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        row = {"model": metrics_path.parent.name, "status": "success", "metrics": metrics, "metadata": metadata}
        rows.append(row)
    output = ROOT / "artifacts" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {output} ({len(rows)} completed models)")


if __name__ == "__main__":
    main()
