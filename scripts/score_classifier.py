# scripts/score_classifier.py
"""
Score ClassifierAgent accuracy against hand labels.

Workflow:
  1. Run experiments — each classified kernel appends a row to
     results/classifier_log.csv with `manual_type` and `match` left blank.
  2. Open that CSV and fill the `manual_type` column by hand (one of:
     elementwise, reduction, matmul, attention, convolution, other) for each
     kernel you want to score. Leave rows you haven't labelled blank; they are
     skipped.
  3. Run:  python scripts/score_classifier.py
     Prints overall accuracy and a per-type breakdown (precision-style: of the
     kernels whose TRUE type is T, how many did the classifier get right).

The `match` column is (re)written in place as 1/0 for each labelled row so the
CSV doubles as an at-a-glance audit.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

LOG_CSV = "results/classifier_log.csv"
FIELDS = ["timestamp", "kernel", "predicted_type", "predicted_bottleneck",
          "confidence", "manual_type", "match"]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    path = Path(LOG_CSV)
    if not path.exists():
        print(f"No classifier log at {LOG_CSV}. Run some experiments first.")
        sys.exit(1)

    rows = load_rows(path)
    labelled = [r for r in rows if (r.get("manual_type") or "").strip()]

    if not labelled:
        print(f"{len(rows)} rows in {LOG_CSV}, but none have `manual_type` filled in.")
        print("Fill the manual_type column by hand, then re-run this script.")
        return

    total = 0
    correct = 0
    per_type_total = defaultdict(int)   # keyed by TRUE (manual) type
    per_type_correct = defaultdict(int)

    for r in rows:
        manual = (r.get("manual_type") or "").strip().lower()
        if not manual:
            r["match"] = ""
            continue
        pred = (r.get("predicted_type") or "").strip().lower()
        hit = (pred == manual)
        r["match"] = "1" if hit else "0"

        total += 1
        correct += int(hit)
        per_type_total[manual] += 1
        per_type_correct[manual] += int(hit)

    # persist the recomputed match column
    write_rows(path, rows)

    print(f"\nClassifier accuracy over {total} hand-labelled kernels")
    print("=" * 48)
    overall = 100.0 * correct / total if total else 0.0
    print(f"  OVERALL: {correct}/{total} = {overall:.1f}%\n")

    print(f"  {'true type':<14}{'correct':>9}{'total':>7}{'acc':>8}")
    print(f"  {'-'*13:<14}{'-'*8:>9}{'-'*6:>7}{'-'*7:>8}")
    for t in sorted(per_type_total):
        tot = per_type_total[t]
        cor = per_type_correct[t]
        acc = 100.0 * cor / tot if tot else 0.0
        print(f"  {t:<14}{cor:>9}{tot:>7}{acc:>7.1f}%")
    print()


if __name__ == "__main__":
    main()
