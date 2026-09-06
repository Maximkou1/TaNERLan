"""Score a predictions JSONL against dev/train gold, with error breakdown.

Unlike evaluation.evaluate_model (the official scorer) this tolerates a
predictions file that only covers a subset of gold hashes (useful while
iterating on a small sample) and prints scorer.py's boundary/type/spurious/
missed breakdown for prompt debugging. Run the official scorer for the
number that counts; use this one while tuning.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from .scorer import score


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--show-errors", type=int, default=0, help="print up to N examples per error type")
    args = p.parse_args()

    preds = load_jsonl(args.predictions)
    pred_hashes = {r["hash"] for r in preds}
    gold_all = load_jsonl(args.gold)
    gold = [r for r in gold_all if r["hash"] in pred_hashes]
    if len(gold) != len(preds):
        missing = pred_hashes - {r["hash"] for r in gold}
        raise SystemExit(f"{len(missing)} predicted hashes not found in gold: {sorted(missing)[:5]}")

    gold_recs = [
        {
            "id": r["hash"],
            "text": r["text"],
            "entities": [{"start": e["start"], "end": e["end"], "type": e["label"]} for e in r["entities"]],
        }
        for r in gold
    ]
    pred_by_id = {
        r["hash"]: [{"start": e["start"], "end": e["end"], "type": e["label"]} for e in r["entities"]]
        for r in preds
    }

    result = score(gold_recs, pred_by_id)
    print(f"records: {len(gold_recs)}")
    print(f"micro: P={result['micro']['P']:.4f} R={result['micro']['R']:.4f} F={result['micro']['F']:.4f} "
          f"(TP={result['TP']} FP={result['FP']} FN={result['FN']})")
    print("by_class:")
    for cls, m in result["by_class"].items():
        print(f"  {cls:6s} P={m['P']:.4f} R={m['R']:.4f} F={m['F']:.4f}")
    print("by_script:")
    for scr, m in result["by_script"].items():
        print(f"  {scr:10s} P={m['P']:.4f} R={m['R']:.4f} F={m['F']:.4f}")
    print("error_counts:", result["error_counts"])

    if args.show_errors:
        for kind, items in result["errors"].items():
            if not items:
                continue
            print(f"\n-- {kind} (showing up to {args.show_errors}/{len(items)}) --")
            for ex in items[: args.show_errors]:
                print(" ", ex)


if __name__ == "__main__":
    main()
