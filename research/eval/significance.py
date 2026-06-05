"""Paired significance harness for the retrieval gate.

The eval JSONs only stored aggregate metrics, so every prior gate decision was made
on raw aggregate gaps with no notion of whether the gap exceeds sampling noise. This
script puts a confidence interval and a p-value on a delta between two runs.

The retrieval pipeline is DETERMINISTIC given a fixed index, so there is no run-to-run
noise — the only uncertainty is sampling over the n=200 query set. Hence:

  single file  : per-metric mean + bootstrap 95% CI = the gate's *discrimination floor*
                 (a delta whose magnitude is below this CI half-width is indistinguishable
                 from having drawn a different 200-query sample).
  two files    : paired delta B-A, paired bootstrap 95% CI, exact sign test, Wilcoxon
                 signed-rank (normal approx) -> verdict at alpha=0.05.

Consumes the --per-query JSONL dumps from eval_retrieval.py (aligned on `query`).

Usage:
  python3 scripts/significance.py pq-baseline-hard.jsonl                 # discrimination floor
  python3 scripts/significance.py pq-jinav5-hard.jsonl pq-qwen3-hard.jsonl
"""

import argparse
import json
import math
import random

METRICS = ["ndcg", "recall10", "mrr", "ceiling_hit"]
N_BOOT = 10000
SEED = 42


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["query"]] = r
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def pearson(xs, ys):
    """Per-query correlation between two systems — drives paired power."""
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


def bootstrap_ci_mean(xs, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for the mean of one sample."""
    rng = random.Random(SEED)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        s = sum(xs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def bootstrap_ci_paired(deltas, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for the mean paired delta."""
    rng = random.Random(SEED)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def sign_test(deltas):
    """Exact two-sided binomial sign test on non-zero deltas (p=0.5)."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    # two-sided: 2 * P(X <= k) under Binomial(n, 0.5), clipped at 1
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf), pos, neg


def wilcoxon_signed_rank(deltas):
    """Wilcoxon signed-rank, normal approximation with tie + continuity correction.
    Returns two-sided p. Adequate for n>=~20."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 1.0
    z = (w - mu + 0.5) / sigma  # continuity correction
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return min(1.0, p)


def single_file(path):
    rows = load(path)
    xs_by = {m: [r[m] for r in rows.values()] for m in METRICS}
    print(f"\n=== discrimination floor: {path}  (n={len(rows)}) ===")
    print("  metric        mean     95% CI                half-width  (= min. detectable gap)")
    for m in METRICS:
        xs = xs_by[m]
        mu = mean(xs)
        lo, hi = bootstrap_ci_mean(xs)
        half = (hi - lo) / 2
        print(f"  {m:12s} {mu:7.4f}  [{lo:7.4f}, {hi:7.4f}]   ±{half:.4f}")
    print("\n  Interpretation: any A-vs-B aggregate gap smaller than ~the half-width is\n"
          "  within the sampling noise of a 200-query draw — not a reliable verdict.")


def two_file(path_a, path_b):
    A, B = load(path_a), load(path_b)
    common = [q for q in A if q in B]
    only_a, only_b = len(A) - len(common), len(B) - len(common)
    print("\n=== paired significance: B - A ===")
    print(f"  A = {path_a}")
    print(f"  B = {path_b}")
    print(f"  paired on {len(common)} queries"
          + (f"  (dropped {only_a} A-only, {only_b} B-only)" if only_a or only_b else ""))
    print("\n  metric        mean_A   mean_B    delta    95% CI delta          sign(+/-)  sign_p   wilcox_p  verdict")
    for m in METRICS:
        a = [A[q][m] for q in common]
        b = [B[q][m] for q in common]
        deltas = [bi - ai for ai, bi in zip(a, b)]
        d = mean(deltas)
        lo, hi = bootstrap_ci_paired(deltas)
        sp, pos, neg = sign_test(deltas)
        wp = wilcoxon_signed_rank(deltas)
        sig = lo > 0 or hi < 0  # CI excludes zero
        verdict = "SIGNIFICANT" if (sig and min(sp, wp) < 0.05) else "noise (n.s.)"
        print(f"  {m:12s} {mean(a):7.4f}  {mean(b):7.4f}  {d:+7.4f}  "
              f"[{lo:+7.4f},{hi:+7.4f}]  {pos:3d}/{neg:<3d}    {sp:6.4f}   {wp:6.4f}   {verdict}")
    # power readout on the gate metric (ndcg): measured rho → the smallest true gap
    # this paired test can resolve at 80% power (z_.975 + z_.8 = 2.802).
    a = [A[q]["ndcg"] for q in common]
    b = [B[q]["ndcg"] for q in common]
    deltas = [bi - ai for ai, bi in zip(a, b)]
    rho = pearson(a, b)
    sd = stdev(deltas)
    n = len(common)
    mde = 2.802 * sd / math.sqrt(n) if n else float("nan")
    print(f"\n  power (ndcg): rho(A,B)={rho:+.3f}  sigma_delta={sd:.4f}  n={n}  "
          f"MDE@80%={mde:.4f}")
    print("\n  SIGNIFICANT = paired bootstrap CI excludes 0 AND min(sign,wilcoxon) p < 0.05.\n"
          "  Otherwise the two systems are statistically indistinguishable on this gate.\n"
          "  If |delta| < MDE@80%, an n.s. verdict is 'underpowered', not 'no effect'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", help="per-query JSONL (system A / baseline)")
    ap.add_argument("file_b", nargs="?", default="", help="per-query JSONL (system B). Omit for discrimination-floor mode.")
    args = ap.parse_args()
    if args.file_b:
        two_file(args.file_a, args.file_b)
    else:
        single_file(args.file_a)


if __name__ == "__main__":
    main()
