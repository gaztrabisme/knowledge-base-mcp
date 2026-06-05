"""Phase B verdict — agreement between Claude (blind) and deepseek relevance grades.

Joins the master validation sample (deepseek_grade) with the Claude grades from the validation
workflow on `idx`, and reports exact + within-±1 agreement plus deepseek precision/recall on the
"relevant" decision (grade>=1) using Claude as reference. Gate: proceed with the deepseek labels
only if within-±1 agreement >= 0.80.

Usage: python3 scripts/judge_agreement.py data/golden/audit/claude_grades.json
       (claude_grades.json = the workflow result's claude_grades list: [{idx, grade}, ...])
"""

import json
import sys
from pathlib import Path

OUT = Path("data/golden/audit")


def main():
    claude_path = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "claude_grades.json")
    master = {r["idx"]: r for r in (json.loads(ln) for ln in open(OUT / "judge_validation_sample.jsonl"))}
    cg = json.load(open(claude_path))
    if isinstance(cg, dict):
        cg = cg.get("claude_grades", [])
    claude = {c["idx"]: int(c["grade"]) for c in cg}

    pairs = [(master[i]["deepseek_grade"], claude[i]) for i in master if i in claude]
    n = len(pairs)
    if not n:
        raise SystemExit("no aligned pairs — check idx join")
    exact = sum(1 for d, c in pairs if d == c) / n
    within1 = sum(1 for d, c in pairs if abs(d - c) <= 1) / n
    # deepseek "relevant" (>=1) vs Claude reference
    tp = sum(1 for d, c in pairs if d >= 1 and c >= 1)
    fp = sum(1 for d, c in pairs if d >= 1 and c == 0)
    fn = sum(1 for d, c in pairs if d == 0 and c >= 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    # confusion
    conf = {f"d{d}_c{c}": sum(1 for dd, cc in pairs if dd == d and cc == c) for d in (0, 1, 2) for c in (0, 1, 2)}
    verdict = "PASS" if within1 >= 0.80 else "FAIL — tighten judge prompt and re-run Phase A"
    out = {"n": n, "exact_agreement": round(exact, 3), "within1_agreement": round(within1, 3),
           "deepseek_precision_rel": round(prec, 3), "deepseek_recall_rel": round(rec, 3),
           "confusion": conf, "gate": verdict}
    (OUT / "judge_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
