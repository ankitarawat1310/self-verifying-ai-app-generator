"""Compare all fine-tuned SVAGA models from their RETURN_<run>.zip packages.
Re-run any time a new zip arrives:  python analysis/compare_models.py
Writes reports/models/: comparison.md, comparison.csv, per_task_loss.csv, generation_metrics.csv, examples.md, charts/*.png"""
from __future__ import annotations
import csv, io, json, math, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITS = ROOT / "training_kits"
OUT = ROOT / "reports" / "models"
CH = OUT / "charts"
NAMES = {"ftA_qwen25coder_1p5b": "Qwen2.5-Coder 1.5B", "ftB_deepseekcoder_1p3b": "DeepSeek-Coder 1.3B",
         "ftC_granite33_2b": "Granite-3.3 2B", "ftD_starcoder2_3b": "StarCoder2 3B"}
COLORS = {"ftA_qwen25coder_1p5b": "#7b5ea7", "ftB_deepseekcoder_1p3b": "#2a78d6", "ftC_granite33_2b": "#2e9e6a", "ftD_starcoder2_3b": "#e08a1e"}
TASKS = ("generate", "edit", "repair")


def load_run(zp: Path) -> dict | None:
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    def find(suffix):
        hits = [n for n in names if n.replace("\\", "/").endswith(suffix)]
        return hits[0] if hits else None
    need = {"summary": "metrics/train_summary.json", "loss": "metrics/eval_loss.json", "gen": "metrics/eval_generation.json",
            "log": "metrics/train_log.jsonl", "samples": "samples/test_generations.jsonl", "preflight": "outputs/preflight.json"}
    paths = {k: find(v) for k, v in need.items()}
    missing = [k for k, v in paths.items() if v is None and k != "preflight"]
    if missing:
        print(f"skip {zp.name}: missing {missing}")
        return None
    rd = lambda k: z.read(paths[k]).decode("utf-8")
    run = json.loads(rd("summary"))
    return {"run": zp.stem.replace("RETURN_", ""), "summary": run, "loss": json.loads(rd("loss")), "gen": json.loads(rd("gen")),
            "log": [json.loads(l) for l in rd("log").splitlines() if l.strip()],
            "samples": [json.loads(l) for l in rd("samples").splitlines() if l.strip()],
            "preflight": json.loads(rd("preflight")) if paths["preflight"] else {}}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CH.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (load_run(p) for p in sorted(KITS.glob("RETURN_*.zip"))) if r]
    runs.sort(key=lambda r: r["run"])
    label = lambda r: NAMES.get(r["run"], r["run"])
    color = lambda r: COLORS.get(r["run"], "#555555")

    # ---------- tables
    rows = []
    for r in runs:
        s, L, G = r["summary"], r["loss"], r["gen"]
        bt, ft = L["base/test"]["all"]["loss"], L["fine_tuned/test"]["all"]["loss"]
        rows.append({
            "model": label(r), "run": r["run"], "base_model": s["model_id"], "gpu": s.get("gpu"),
            "chat_template": s.get("use_chat_template"), "train_rows": s["train_rows"], "dropped_over_4096": s["train_rows_dropped_over_max_len"],
            "trainable_params_M": round(s["trainable_params"] / 1e6, 1), "total_params_M": round(s["total_params"] / 1e6, 1),
            "updates": s["global_steps"], "train_minutes": round(s["train_seconds"] / 60, 1),
            "test_loss_base": round(bt, 4), "test_loss_finetuned": round(ft, 4), "test_loss_reduction_pct": round(100 * (bt - ft) / bt, 1),
            "test_perplexity_base": round(math.exp(bt), 3), "test_perplexity_finetuned": round(math.exp(ft), 3),
            **{f"{t}_{m}_{c}": round(G.get(f"{c}/{t}/{m}", float("nan")), 3) for c in ("base", "fine_tuned")
               for t, m in (("generate", "python_parses"), ("edit", "valid_diff"), ("repair", "valid_diff"),
                            ("edit", "changed_line_f1"), ("repair", "changed_line_f1"), ("repair", "exact_match"),
                            ("generate", "seconds"), ("edit", "seconds"), ("repair", "seconds"))},
            "transformers": r["preflight"].get("transformers"), "revision": s.get("revision") or "not recorded"})
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT / "per_task_loss.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "split", "task", "base_loss", "finetuned_loss", "reduction_pct", "target_tokens"])
        for r in runs:
            for split in ("val", "test"):
                for t in TASKS + ("all",):
                    b, ftl = r["loss"][f"base/{split}"][t], r["loss"][f"fine_tuned/{split}"][t]
                    w.writerow([label(r), split, t, round(b["loss"], 4), round(ftl["loss"], 4), round(100 * (b["loss"] - ftl["loss"]) / b["loss"], 1), b["target_tokens"]])
    with (OUT / "generation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "condition", "task", "metric", "value", "n"])
        for r in runs:
            n = {t: sum(1 for s in r["samples"] if s["task_type"] == t) for t in TASKS}
            for k, v in sorted(r["gen"].items()):
                c, t, m = k.split("/")
                w.writerow([label(r), c, t, m, round(v, 4), n[t]])

    # ---------- charts
    # 1 training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for r in runs:
        tr = [(e["step"], e["loss"]) for e in r["log"] if "loss" in e and "eval_loss" not in e]
        ev = [(e["step"], e["eval_loss"]) for e in r["log"] if "eval_loss" in e]
        if tr: axes[0].plot(*zip(*tr), color=color(r), lw=1.2, label=label(r))
        if ev: axes[1].plot(*zip(*ev), color=color(r), lw=2, marker="o", ms=4, label=label(r))
    axes[0].set_title("Training loss (answer tokens only)"); axes[1].set_title("Validation loss during training (150 rows)")
    for ax in axes:
        ax.set_xlabel("optimizer update"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(CH / "1_training_curves.png", dpi=150); plt.close(fig)
    # 2 test loss before/after per task
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, t in zip(axes, TASKS + ("all",)):
        xs = range(len(runs))
        b = [r["loss"]["base/test"][t]["loss"] for r in runs]; f_ = [r["loss"]["fine_tuned/test"][t]["loss"] for r in runs]
        ax.bar([x - 0.2 for x in xs], b, 0.4, color="#c3c9d4", label="base")
        ax.bar([x + 0.2 for x in xs], f_, 0.4, color=[color(r) for r in runs], label="fine-tuned")
        for x, bv, fv in zip(xs, b, f_):
            ax.text(x + 0.2, fv, f"-{100*(bv-fv)/bv:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs)); ax.set_xticklabels([label(r).split()[0] for r in runs], fontsize=9)
        ax.set_title(f"{t}"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test loss (lower is better)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Untouched test split: base vs fine-tuned loss (each model on its own tokenizer)")
    fig.tight_layout(); fig.savefig(CH / "2_test_loss_by_task.png", dpi=150); plt.close(fig)
    # 3 generation quality
    metrics = [("generate", "python_parses", "Generate: Python parses"), ("edit", "valid_diff", "Edit: valid diff"),
               ("repair", "valid_diff", "Repair: valid diff"), ("repair", "changed_line_f1", "Repair: changed-line F1"),
               ("edit", "changed_line_f1", "Edit: changed-line F1"), ("repair", "exact_match", "Repair: exact match")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (t, m, title) in zip(axes.flat, metrics):
        xs = range(len(runs))
        b = [r["gen"].get(f"base/{t}/{m}", 0) for r in runs]; f_ = [r["gen"].get(f"fine_tuned/{t}/{m}", 0) for r in runs]
        ax.bar([x - 0.2 for x in xs], b, 0.4, color="#c3c9d4", label="base")
        ax.bar([x + 0.2 for x in xs], f_, 0.4, color=[color(r) for r in runs], label="fine-tuned")
        for x, v in zip(xs, f_): ax.text(x + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_title(title); ax.set_xticks(list(xs)); ax.set_xticklabels([label(r).split()[0] for r in runs], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Generated outputs on 30 test examples per task (base vs fine-tuned)")
    fig.tight_layout(); fig.savefig(CH / "3_generation_quality.png", dpi=150); plt.close(fig)
    # 4 cost: training time and latency
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = range(len(runs))
    axes[0].bar(xs, [r["summary"]["train_seconds"] / 60 for r in runs], color=[color(r) for r in runs])
    for x, r in zip(xs, runs): axes[0].text(x, r["summary"]["train_seconds"] / 60, f"{r['summary']['train_seconds']/60:.0f} min\n{r['summary'].get('gpu','').replace('NVIDIA ','')}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Training time, one epoch (different GPUs)"); axes[0].set_xticks(list(xs)); axes[0].set_xticklabels([label(r).split()[0] for r in runs])
    for i, t in enumerate(TASKS):
        axes[1].bar([x + (i - 1) * 0.25 for x in xs], [r["gen"].get(f"fine_tuned/{t}/seconds", 0) for r in runs], 0.25, label=t)
    axes[1].set_title("Fine-tuned latency per answer, s (different GPUs; greedy, max 1,024 tokens)"); axes[1].set_xticks(list(xs)); axes[1].set_xticklabels([label(r).split()[0] for r in runs]); axes[1].legend(frameon=False)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(CH / "4_cost_and_latency.png", dpi=150); plt.close(fig)

    # ---------- examples (original input, reference, base output, fine-tuned output)
    ex = ["# Example outputs (untouched test split)\n", "Same input to every model; each shows the base model and its fine-tuned adapter.\n"]
    pick = {t: next(s["sample_id"] for s in runs[0]["samples"] if s["task_type"] == t) for t in TASKS}
    for t, sid in pick.items():
        ex.append(f"\n## {t}: {sid}\n")
        first = next(s for s in runs[0]["samples"] if s["sample_id"] == sid)
        ex.append("**Reference answer (from the dataset):**\n```\n" + first["reference"][:1200] + "\n```\n")
        for r in runs:
            s = next((x for x in r["samples"] if x["sample_id"] == sid), None)
            if not s: continue
            for c in ("base", "fine_tuned"):
                sc = {k: v for k, v in s[c].items() if k not in ("output",)}
                ex.append(f"**{label(r)}, {c.replace('_', '-')}** `{json.dumps(sc)}`\n```\n" + s[c]["output"][:900] + "\n```\n")
    (OUT / "examples.md").write_text("\n".join(ex), encoding="utf-8")

    # ---------- summary markdown
    md = ["# Fine-tuned model comparison\n", f"Models with complete results: {len(runs)} of 4. Re-run `python analysis/compare_models.py` when a new RETURN zip arrives.\n",
          "Same dataset (10,155 train rows; generate, edit, repair balanced), same QLoRA settings (rank 16, alpha 32, 4-bit, lr 2e-4, 1 epoch, 16 rows per update).",
          "Loss values are only comparable within a model (each model uses its own tokenizer); compare the reduction, not the raw loss.\n",
          "| Model | GPU | Train rows | Minutes | Test loss base → fine-tuned | Reduction | Repair F1 | Repair exact | Edit valid diff | Generate parses |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['model']} | {str(x['gpu']).replace('NVIDIA ','')} | {x['train_rows']:,} | {x['train_minutes']:.0f} | {x['test_loss_base']:.3f} → {x['test_loss_finetuned']:.3f} | "
                  f"{x['test_loss_reduction_pct']:.0f}% | {x['repair_changed_line_f1_base']:.2f} → {x['repair_changed_line_f1_fine_tuned']:.2f} | "
                  f"{x['repair_exact_match_base']:.2f} → {x['repair_exact_match_fine_tuned']:.2f} | {x['edit_valid_diff_base']:.2f} → {x['edit_valid_diff_fine_tuned']:.2f} | "
                  f"{x['generate_python_parses_base']:.2f} → {x['generate_python_parses_fine_tuned']:.2f} |")
    md += ["\nGeneration metrics are on 30 test examples per task: small samples, so differences of one or two examples (0.03 to 0.07) are noise.",
           "\nCharts: `charts/1_training_curves.png`, `2_test_loss_by_task.png`, `3_generation_quality.png`, `4_cost_and_latency.png`. Examples: `examples.md`."]
    (OUT / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
