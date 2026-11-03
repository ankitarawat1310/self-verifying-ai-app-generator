#!/usr/bin/env python3
"""Paired comparison: tuned vs plain-completion base, per benchmark (exact two-sided sign test on the discordant problems).
Run from /workspace/kit:   python paired_test.py            (reads outputs/func_eval/*.jsonl)
Or:  python paired_test.py some_results_dir [cond_A] [cond_B]     e.g.  python paired_test.py /workspace/results_30b tuned base
"""
import json, math, sys

d = sys.argv[1] if len(sys.argv) > 1 else "outputs/func_eval"
A = sys.argv[2] if len(sys.argv) > 2 else "tuned"       # condition being tested
B = sys.argv[3] if len(sys.argv) > 3 else "base_plain"   # baseline condition


def load(f):
    return {r["id"]: r["passed"] for r in map(json.loads, open(f, encoding="utf-8"))}


for b in ("humaneval", "mbpp", "humanevalfix"):
    try:
        t = load(f"{d}/{b}_{A}.jsonl")
        p = load(f"{d}/{b}_{B}.jsonl")
    except FileNotFoundError as e:
        if b != "humanevalfix":
            print(b, "skipped (missing file):", e)
        continue
    only_t = sum(t[i] and not p[i] for i in t)
    only_p = sum(p[i] and not t[i] for i in t)
    both = sum(t[i] and p[i] for i in t)
    n, k = only_t + only_p, min(only_t, only_p)
    pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"{b}: n={len(t)} {A}={sum(t.values())} {B}={sum(p.values())} both={both} "
          f"only_{A}={only_t} only_{B}={only_p} exact_p={pv:.4f}")
