"""Public benchmark results (pass@1, greedy) for all fine-tuned models: HumanEval+, MBPP+ and HumanEvalFix (Python).
Reads the result packages from the benchmark pods:
  training_kits/func_eval_ft*.zip   (StarCoder2, HumanEval+/MBPP+)
  all_results_pod*.zip              (Qwen, DeepSeek, Granite: HumanEval+/MBPP+ and HumanEvalFix; StarCoder2 under kit/)
  results_30b*.zip                  (Qwen3-Coder 30B, HumanEval+/MBPP+)
Conditions: base = base model with the SVAGA prompt format, base_plain = base model with its own plain prompt (StarCoder2 only),
tuned = fine-tuned. Paired exact McNemar (sign) test per benchmark: fine-tuned vs the better base condition, same problems.
Usage: python analysis/public_benchmarks.py -> reports/models/public_benchmarks.csv, .md, charts/5_public_benchmarks.png"""
import csv, json, re, zipfile
from collections import defaultdict
from math import comb
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "models"
NAMES = {"ftA": "Qwen2.5-Coder 1.5B", "ftB": "DeepSeek-Coder 1.3B", "ftC": "Granite-3.3 2B", "ftD": "StarCoder2 3B", "30B": "Qwen3-Coder 30B-A3B"}
ORDER = list(NAMES)
BENCH = {"humaneval": "HumanEval+", "mbpp": "MBPP+", "humanevalfix": "HumanEvalFix"}


def mcnemar(a, b):
    ks = a.keys() & b.keys(); x = sum(1 for k in ks if a[k] and not b[k]); y = sum(1 for k in ks if b[k] and not a[k]); n = x + y
    return x, y, (min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0)


def run_of(folder, zname):
    m = re.search(r"(ft[A-D])", folder) or re.search(r"(ft[A-D])", zname)
    if m: return m.group(1)
    if "30b" in folder.lower() or "30b" in zname.lower(): return "30B"
    if folder.startswith("kit/"): return "ftD"          # the StarCoder2 kit inside the pod A bundle
    return None


def packages():
    zips = sorted((ROOT / "training_kits").glob("func_eval_ft*.zip")) + sorted(ROOT.glob("all_results_pod*.zip")) + sorted(ROOT.glob("results_30b*.zip"))
    for zp in zips:
        z = zipfile.ZipFile(zp); byfolder = defaultdict(list)
        for n in z.namelist():
            if not n.endswith("/"): byfolder[str(PurePosixPath(n).parent)].append(n)
        for folder, names in byfolder.items():
            sums = [n for n in names if re.search(r"summary[^/]*\.json$", n)]
            run = run_of(folder + "/", zp.name)
            if not sums or not run: continue
            summary = {}
            for s in sums: summary.update(json.loads(z.read(s)))
            files = {PurePosixPath(n).name: n for n in names if n.endswith(".jsonl")}
            yield run, zp.name, z, summary, files


rows, seen = [], set()
for run, src, z, summary, files in packages():
    for bm in sorted({k.split("/")[0] for k in summary}):
        if (run, bm) in seen or f"{bm}/tuned" not in summary: continue
        res = {}
        for cond in ("base", "base_plain", "tuned"):
            f = files.get(f"{bm}_{cond}.jsonl")
            if f: res[cond] = {json.loads(l)["id"]: bool(json.loads(l)["passed"]) for l in z.read(f).decode().splitlines() if l.strip()}
        if "tuned" not in res: continue
        bases = [c for c in ("base", "base_plain") if c in res]
        best = max(bases, key=lambda c: summary[f"{bm}/{c}"]["pass_at_1_pct"])
        x, y, p = mcnemar(res["tuned"], res[best])
        t, b = summary[f"{bm}/tuned"], summary[f"{bm}/{best}"]
        seen.add((run, bm))
        rows.append({"model": NAMES[run], "run": run, "benchmark": BENCH.get(bm, bm), "n": t["n"],
                     "base_svaga_prompt_pct": summary.get(f"{bm}/base", {}).get("pass_at_1_pct"),
                     "base_plain_prompt_pct": summary.get(f"{bm}/base_plain", {}).get("pass_at_1_pct"),
                     "best_base": best, "best_base_pct": b["pass_at_1_pct"], "fine_tuned_pct": t["pass_at_1_pct"],
                     "change_pts": round(t["pass_at_1_pct"] - b["pass_at_1_pct"], 1),
                     "solved_only_by_fine_tuned": x, "solved_only_by_best_base": y, "mcnemar_p": round(p, 6), "significant": p < 0.05,
                     "base_diff_valid_pct": b.get("diff_valid_pct"), "tuned_diff_valid_pct": t.get("diff_valid_pct"),
                     "base_applies_pct": b.get("applies_pct"), "tuned_applies_pct": t.get("applies_pct"),
                     "base_pass_given_applied_pct": b.get("pass_given_applied_pct"), "tuned_pass_given_applied_pct": t.get("pass_given_applied_pct"),
                     "source": src})
rows.sort(key=lambda r: (list(BENCH.values()).index(r["benchmark"]) if r["benchmark"] in BENCH.values() else 9, ORDER.index(r["run"])))
OUT.mkdir(parents=True, exist_ok=True); (OUT / "charts").mkdir(exist_ok=True)
if rows:
    with (OUT / "public_benchmarks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fmt = lambda v: "-" if v is None else f"{v}%"
    md = ["# Public benchmarks (pass@1, greedy decoding)\n",
          "| Benchmark | Model | Problems | Base | Fine-tuned | Change (pts) | Only fine-tuned solves | Only base solves | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = fmt(r["best_base_pct"]) + (" (plain prompt)" if r["best_base"] == "base_plain" else "")
        md.append(f"| {r['benchmark']} | {r['model']} | {r['n']} | {base} | {fmt(r['fine_tuned_pct'])} | {r['change_pts']:+} | "
                  f"{r['solved_only_by_fine_tuned']} | {r['solved_only_by_best_base']} | {'<0.0001' if r['mcnemar_p'] < 0.0001 else round(r['mcnemar_p'], 4)}{' *' if r['significant'] else ''} |")
    hef = [r for r in rows if r["benchmark"] == "HumanEvalFix"]
    if hef:
        md += ["\n## HumanEvalFix detail (base → fine-tuned)\n", "| Model | Usable diff | Diff applies | Fixed, given it applied |", "|---|---|---|---|"]
        for r in hef:
            md.append(f"| {r['model']} | {r['base_diff_valid_pct']}% → {r['tuned_diff_valid_pct']}% | {r['base_applies_pct']}% → {r['tuned_applies_pct']}% | "
                      f"{r['base_pass_given_applied_pct']}% → {r['tuned_pass_given_applied_pct']}% |")
    md += ["\nBase = the better of the base model's prompts (StarCoder2 is not instruction-tuned, so its plain completion prompt is the fair baseline).",
           "p = exact paired McNemar test on the problems where the two disagree; * = p < 0.05. HumanEvalFix was not run for the 30B.",
           "Runner: the team's own scripts (not the official evalplus harness), greedy decoding, one sample per problem."]
    (OUT / "public_benchmarks.md").write_text("\n".join(md), encoding="utf-8")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    benches = [b for b in BENCH.values() if any(r["benchmark"] == b for r in rows)]
    fig, axes = plt.subplots(1, len(benches), figsize=(5.2 * len(benches), 4.4), sharey=True)
    for ax, bname in zip(axes if len(benches) > 1 else [axes], benches):
        rs = [r for r in rows if r["benchmark"] == bname]; xs = range(len(rs))
        for i, (key, lab, col) in enumerate((("best_base_pct", "base", "#9aa3b2"), ("fine_tuned_pct", "fine-tuned", "#e08a1e"))):
            vals = [r[key] for r in rs]
            ax.bar([x + (i - 0.5) * 0.38 for x in xs], vals, 0.38, color=col, label=lab)
            for x, v in zip(xs, vals): ax.text(x + (i - 0.5) * 0.38, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        for x, r in zip(xs, rs):
            if r["significant"]: ax.text(x, max(r["best_base_pct"], r["fine_tuned_pct"]) + 6, "p<0.05", ha="center", fontsize=8, color="#b5591a")
        ax.set_xticks(list(xs)); ax.set_xticklabels([r["model"].replace(" ", "\n", 1) for r in rs], fontsize=8)
        ax.set_title(bname); ax.spines[["top", "right"]].set_visible(False)
    (axes[0] if len(benches) > 1 else axes).set_ylabel("pass@1 (%)"); (axes[0] if len(benches) > 1 else axes).legend(frameon=False, loc="upper left")
    fig.suptitle("Public benchmarks: base vs fine-tuned (executed tests, greedy)"); fig.tight_layout()
    fig.savefig(OUT / "charts" / "5_public_benchmarks.png", dpi=150); plt.close(fig)
    print("\n".join(md))
