"""Learned release gate (SVAGA model 5): predict whether a generated app is actually correct (passes every hidden check)
from signals available at decision time only. Compared with the rule gate (ACCEPT only if all six verifiers pass).

Usage: python analysis/gate_classifier.py --grid "<SVAGA repo>/results/grid/demo1"
Writes reports/gate/: gate_runs.csv (features + label), metrics.json, gate_report.md, charts/*.png

Leakage rule: no hidden-check result, safety rate, seeded-benchmark-bug recall or judge output is used as a feature.
Evaluation: leave-one-task-out cross-validation (the model never sees any run of the task it is scored on)."""
from __future__ import annotations
import argparse, ast, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "gate"
VERIFIERS = ["pytest_sandbox", "contract_checks", "spec_invariants", "policy_static", "api_fuzz", "benchmark_properties"]


def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _len_literal(x):
    if isinstance(x, list):
        return len(x)
    try:
        return len(ast.literal_eval(x))
    except Exception:
        return 0


def build_rows(grid: Path):
    rows = []
    for rec_path in sorted(grid.glob("m*/*/r*/record.json")):
        d = rec_path.parent
        rec = json.loads(rec_path.read_text())
        ver = json.loads((d / "verification.json").read_text()) if (d / "verification.json").exists() else {"verifiers": [], "counterexamples": []}
        vmap = {v["name"]: v for v in ver.get("verifiers", [])}
        app = (d / "app.py").read_text(encoding="utf-8", errors="ignore") if (d / "app.py").exists() else ""
        tests = (d / "test_app.py").read_text(encoding="utf-8", errors="ignore") if (d / "test_app.py").exists() else ""
        py = vmap.get("pytest_sandbox", {}).get("details", {})
        total, passed = _num(py.get("total_tests")), _num(py.get("passed_tests"))
        om = rec.get("own_mutation_detail") or {}
        f = {
            "pipeline": rec["model"], "task": rec["task"], "category": rec["category"], "repeat": rec["repeat"],
            "rule_gate_accept": int(rec["decision"] == "ACCEPT"),
            **{f"v_{v}": int(bool(vmap.get(v, {}).get("passed"))) for v in VERIFIERS},
            "n_failed_verifiers": len(rec.get("failed_verifiers") or []),
            "own_tests_total": total, "own_tests_pass_ratio": passed / total if total else 0.0,
            "contract_missing_paths": _len_literal(vmap.get("contract_checks", {}).get("details", {}).get("missing_paths", "[]")),
            "fuzz_failures": _len_literal(vmap.get("api_fuzz", {}).get("details", {}).get("failures", "[]")),
            "spec_violations": _len_literal(vmap.get("spec_invariants", {}).get("details", {}).get("violations", "[]")),
            "policy_violations": _len_literal(vmap.get("policy_static", {}).get("details", {}).get("violations", "[]")),
            "counterexamples": len(ver.get("counterexamples") or []),
            "own_mutation_score": _num(rec.get("own_mutation_score")), "own_mutants": _num(om.get("mutants")),
            "app_lines": app.count("\n"), "test_functions": tests.count("def test_"),
            "app_checks_identity_header": int("x-actor" in app.lower()), "app_raises_403": int("403" in app), "app_raises_409": int("409" in app),
            "app_forbids_extra_fields": int("extra" in app and "forbid" in app),
            "tokens": _num((rec.get("budget") or {}).get("tokens_used")), "llm_calls": _num((rec.get("budget") or {}).get("llm_calls")),
            "generation_seconds": _num(rec.get("generation_seconds")),
            "excess_imports": _num((rec.get("permission_excess") or {}).get("count")),
            "label_correct": int(bool(rec.get("hidden_pass"))),
        }
        rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", required=True); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, roc_curve
    from sklearn.inspection import permutation_importance
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    rows = build_rows(Path(a.grid))
    with (OUT / "gate_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    meta = ["pipeline", "task", "category", "repeat", "label_correct", "rule_gate_accept"]
    num_cols = [k for k in rows[0] if k not in meta]
    cats = {"pipeline": sorted({r["pipeline"] for r in rows}), "category": sorted({r["category"] for r in rows})}
    feat_names = num_cols + [f"{k}={v}" for k, vs in cats.items() for v in vs]
    X = np.array([[float(r[c]) for c in num_cols] + [float(r[k] == v) for k, vs in cats.items() for v in vs] for r in rows])
    y = np.array([r["label_correct"] for r in rows]); groups = np.array([r["task"] for r in rows])
    rule = np.array([r["rule_gate_accept"] for r in rows])

    models = {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=2026),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=2026),
    }
    logo = LeaveOneGroupOut()
    oof = {}
    for name, mk in models.items():
        p = np.zeros(len(y))
        for tr, te in logo.split(X, y, groups):
            m = mk().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        oof[name] = p

    def scores(pred, prob=None):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        acc_n = int(pred.sum())
        out = {"accepted": acc_n, "tp_correct_shipped": int(tp), "fp_broken_shipped": int(fp), "fn_correct_rejected": int(fn), "tn_broken_rejected": int(tn),
               "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "accuracy": round((tp + tn) / len(y), 3),
               "false_assurance_of_accepted": round(fp / acc_n, 3) if acc_n else None,
               "broken_shipped_of_all_runs": round(fp / len(y), 3)}
        if prob is not None:
            out["roc_auc"] = round(roc_auc_score(y, prob), 3); out["pr_auc"] = round(average_precision_score(y, prob), 3)
        return out

    res = {"n_runs": len(y), "n_correct": int(y.sum()), "base_rate_correct": round(y.mean(), 3), "n_tasks": len(set(groups)),
           "features": feat_names, "rule_gate": scores(rule)}
    k = int(rule.sum())   # same number of ACCEPTs as the rule gate, so false assurance is compared at equal shipping volume
    for name, p in oof.items():
        thr_same = np.sort(p)[::-1][k - 1]
        res[name] = {"threshold_0.5": scores((p >= 0.5).astype(int), p),
                     "same_accept_count_as_rule": {**scores((p >= thr_same).astype(int), p), "threshold": round(float(thr_same), 3)},
                     "rule_and_model_0.5": scores(((p >= 0.5) & (rule == 1)).astype(int), p)}
    # bootstrap over tasks for AUC of the best model
    best = max(oof, key=lambda n: res[n]["threshold_0.5"]["roc_auc"])
    rng = np.random.default_rng(2026); tasks = np.array(sorted(set(groups))); aucs = []
    for _ in range(2000):
        pick = rng.choice(tasks, len(tasks), replace=True)
        idx = np.concatenate([np.where(groups == t)[0] for t in pick])
        if len(set(y[idx])) == 2: aucs.append(roc_auc_score(y[idx], oof[best][idx]))
    res["best_model"] = best; res["best_auc_95ci_task_bootstrap"] = [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]
    # importance on a model fit to all data (descriptive only)
    full = models[best]().fit(X, y)
    pi = permutation_importance(full, X, y, n_repeats=30, random_state=2026, scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1][:12]
    res["top_features"] = [[feat_names[i], round(float(pi.importances_mean[i]), 4)] for i in order]
    json.dump(res, open(OUT / "metrics.json", "w"), indent=2)

    # charts
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ')} (AUC {res[name]['threshold_0.5']['roc_auc']:.2f})")
    r = res["rule_gate"]; ax.scatter([r["fp_broken_shipped"] / (len(y) - y.sum())], [r["recall"]], color="black", zorder=5, label="rule gate (all 6 checks pass)")
    ax.plot([0, 1], [0, 1], ls="--", color="#aaa"); ax.set_xlabel("broken apps shipped / all broken (false positive rate)"); ax.set_ylabel("correct apps shipped / all correct (recall)")
    ax.set_title("Learned gate vs rule gate, leave-one-task-out"); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "charts" / "1_roc.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, s) in zip(axes, [("Rule gate", res["rule_gate"]), (f"{best.replace('_', ' ')}, same # shipped", res[best]["same_accept_count_as_rule"]),
                                      (f"{best.replace('_', ' ')}, threshold 0.5", res[best]["threshold_0.5"])]):
        m = np.array([[s["tn_broken_rejected"], s["fp_broken_shipped"]], [s["fn_correct_rejected"], s["tp_correct_shipped"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, m[i, j], ha="center", va="center", fontsize=14, color="black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["REJECT", "ACCEPT"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["actually broken", "actually correct"])
        fa = s["false_assurance_of_accepted"]
        ax.set_title(f"{title}\nfalse assurance {fa:.0%}" if fa is not None else title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "charts" / "2_confusion_matrices.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [n for n, _ in res["top_features"]][::-1]; vals = [v for _, v in res["top_features"]][::-1]
    ax.barh(names, vals, color="#2a78d6"); ax.set_xlabel("drop in ROC-AUC when shuffled"); ax.set_title(f"What the {best.replace('_', ' ')} gate relies on")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "charts" / "3_feature_importance.png", dpi=150); plt.close(fig)

    rg = res["rule_gate"]
    md = [f"# Learned release gate\n", f"{len(y)} Demo 1 runs (M0, M1, M2 × 15 test tasks × 3), {int(y.sum())} actually correct ({y.mean():.0%}). "
          "Leave-one-task-out cross-validation; features are decision-time signals only.\n",
          "| Gate | Shipped | Correct shipped | Broken shipped | False assurance | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|---|---|---|",
          f"| Rule (all 6 checks pass) | {rg['accepted']} | {rg['tp_correct_shipped']} | {rg['fp_broken_shipped']} | {rg['false_assurance_of_accepted']:.0%} | {rg['precision']:.2f} | {rg['recall']:.2f} | {rg['f1']:.2f} | n/a |"]
    for name in models:
        for mode, lab in (("same_accept_count_as_rule", "same # shipped as rule"), ("threshold_0.5", "threshold 0.5"), ("rule_and_model_0.5", "rule AND model")):
            s = res[name][mode]; fa = s["false_assurance_of_accepted"]
            md.append(f"| {name.replace('_', ' ')}, {lab} | {s['accepted']} | {s['tp_correct_shipped']} | {s['fp_broken_shipped']} | {'' if fa is None else f'{fa:.0%}'} | "
                      f"{s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['roc_auc']:.2f} |")
    md += [f"\nBest model: {best}, ROC-AUC 95% interval over tasks {res['best_auc_95ci_task_bootstrap']}.",
           "Top features: " + ", ".join(f"{n} ({v})" for n, v in res["top_features"][:6]) + ".",
           "\nSmall data (135 runs, one LLM): treat as a first estimate. More runs from the fine-tuned models will be added to the same table."]
    (OUT / "gate_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
