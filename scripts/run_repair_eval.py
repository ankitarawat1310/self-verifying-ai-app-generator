"""M4 repair experiment: can a fine-tuned repair model fix apps that SVAGA rejected?

Takes rejected runs from a saved grid (default: the frozen Demo 1 grid), gives each to a repair model served by
svaga_plugin/serve_svaga_model.py (base or fine-tuned), applies its diff, re-runs SVAGA's verification + release gate, and scores the
result with the hidden judge. The repair model never sees hidden checks.

  $env:SVAGA_MODEL_ENDPOINTS = "ftC_granite33_2b=http://127.0.0.1:11500,ftC_granite33_2b-base=http://127.0.0.1:11501"
  python scripts\\run_repair_eval.py --llm-model ftC_granite33_2b-base --name repair_granite_base
  python scripts\\run_repair_eval.py --llm-model ftC_granite33_2b      --name repair_granite_tuned
Resumable: re-running the same command skips finished cases. Results: results/repair/<name>/ (summary.md, cases.csv, one folder per case).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.benchmarks.task_package import find_benchmark  # noqa: E402
from shared.budget.tracker import BudgetProfile, BudgetTracker  # noqa: E402
from shared.llm.provider import model_endpoints  # noqa: E402
from shared.policy.checker import PolicyDocument  # noqa: E402
from svaga_platform.app.artifacts import PipelineArtifacts  # noqa: E402
from svaga_platform.app.pipelines.common import finalize_pipeline_result  # noqa: E402
from svaga_platform.app.repair.finetuned import (apply_unified_diff, build_issue, build_repair_prompt,  # noqa: E402
                                                 call_repair_model, extract_diff, strip_platform_blocks)


def repair_case(run_dir: Path, out: Path, *, endpoint: str, model: str, rounds: int, call=call_repair_model) -> dict:
    from svaga_platform.app.judge.runner import judge_candidate
    rec_before = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    task = rec_before["task"]
    bench = find_benchmark(task)
    spec = json.loads((run_dir / "workflow_spec.json").read_text(encoding="utf-8"))
    policy = PolicyDocument(**json.loads((run_dir / "policy.json").read_text(encoding="utf-8")))
    app = (run_dir / "app.py").read_text(encoding="utf-8")
    tests = (run_dir / "test_app.py").read_text(encoding="utf-8")
    verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    history, decision, changed = [], rec_before["decision"], False
    for rnd in range(1, rounds + 1):
        issue = build_issue(bench.natural_language_requirement, verification)
        prompt = build_repair_prompt(issue, {"app.py": strip_platform_blocks(app), "test_app.py": tests})
        (out / f"round{rnd}_prompt.txt").write_text(prompt, encoding="utf-8")
        reply = call(endpoint, model, prompt)
        (out / f"round{rnd}_answer.txt").write_text(reply["text"], encoding="utf-8")
        diff = extract_diff(reply["text"])
        files, rep = apply_unified_diff({"app.py": strip_platform_blocks(app), "test_app.py": tests}, diff)
        step = {"round": rnd, "valid_diff": bool(diff) and rep.hunks > 0, "apply": rep.to_dict(), "tokens": reply["tokens"], "seconds": reply["seconds"]}
        if rep.applied == 0:
            step["decision"] = decision
            history.append(step)
            break
        changed = True
        app, tests = files["app.py"], files["test_app.py"]
        arts = PipelineArtifacts(app_code=app, test_code=tests, workflow_spec=spec, policy=policy)
        res = finalize_pipeline_result("m4", bench, arts, BudgetTracker(profile=BudgetProfile()), frozen_spec_hash=arts.spec_hash)
        app, verification, decision = arts.app_code, res.verification, res.decision
        step["decision"] = decision
        step["failed_verifiers"] = [v["name"] for v in verification.get("verifiers", []) if not v.get("passed")]
        history.append(step)
        if decision == "ACCEPT":
            break
    (out / "app.py").write_text(app, encoding="utf-8"); (out / "test_app.py").write_text(tests, encoding="utf-8")
    (out / "verification.json").write_text(json.dumps(verification, indent=2, default=str), encoding="utf-8")
    if changed:
        judged = judge_candidate(task, app)
        (out / "judge.json").write_text(json.dumps(judged.to_dict(), indent=2, default=str), encoding="utf-8")
        after_passed, after_total, after_pass = judged.passed_count, len(judged.checks), bool(judged.passed)
    else:
        after_passed, after_total, after_pass = rec_before["hidden_passed"], rec_before["hidden_total"], bool(rec_before["hidden_pass"])
    rec = {"pipeline": rec_before["model"], "task": task, "category": rec_before["category"], "repeat": rec_before["repeat"],
           "repair_model": model, "rounds_used": len(history),
           "valid_diff": any(h["valid_diff"] for h in history), "diff_applied_fully": any(h["apply"]["ok"] for h in history),
           "code_changed": changed, "decision_before": rec_before["decision"], "decision_after": decision,
           "hidden_before": f'{rec_before["hidden_passed"]}/{rec_before["hidden_total"]}', "hidden_after": f"{after_passed}/{after_total}",
           "hidden_rate_before": rec_before["hidden_passed"] / rec_before["hidden_total"] if rec_before["hidden_total"] else 0.0,
           "hidden_rate_after": after_passed / after_total if after_total else 0.0,
           "correct_before": bool(rec_before["hidden_pass"]), "correct_after": after_pass,
           "tokens": sum(h["tokens"] for h in history), "model_seconds": round(sum(h["seconds"] for h in history), 1),
           "wall_seconds": round(time.monotonic() - started, 1), "history": history}
    (out / "record.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def summarize(out_root: Path) -> str:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_root.glob("*/*/r*/record.json"))]
    if not recs:
        return "no cases yet"
    with (out_root / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [k for k in recs[0] if k != "history"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(recs)
    n = len(recs)
    pct = lambda k: sum(1 for r in recs if r[k]) / n
    broken = [r for r in recs if not r["correct_before"]]
    correct = [r for r in recs if r["correct_before"]]
    accepted = [r for r in recs if r["decision_after"] == "ACCEPT"]
    s = {"cases": n, "repair_model": recs[0]["repair_model"],
         "valid_diff": pct("valid_diff"), "diff_applied_fully": pct("diff_applied_fully"), "code_changed": pct("code_changed"),
         "accepted_after_repair": len(accepted) / n,
         "correct_before": sum(r["correct_before"] for r in recs) / n, "correct_after": sum(r["correct_after"] for r in recs) / n,
         "broken_apps_fixed": sum(r["correct_after"] for r in broken), "broken_apps": len(broken),
         "correct_apps_broken_by_repair": sum(not r["correct_after"] for r in correct), "correct_apps": len(correct),
         "hidden_rate_before": sum(r["hidden_rate_before"] for r in recs) / n, "hidden_rate_after": sum(r["hidden_rate_after"] for r in recs) / n,
         "false_assurance_after": (sum(not r["correct_after"] for r in accepted) / len(accepted)) if accepted else None,
         "mean_model_seconds": sum(r["model_seconds"] for r in recs) / n}
    (out_root / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    p = lambda v: "-" if v is None else f"{v:.0%}"
    md = [f"# M4 repair: {s['repair_model']} on {n} rejected apps\n", "| Measure | Value |", "|---|---|",
          f"| Valid diff returned | {p(s['valid_diff'])} |", f"| Diff applied fully | {p(s['diff_applied_fully'])} |",
          f"| Code changed | {p(s['code_changed'])} |", f"| Gate ACCEPT after repair | {p(s['accepted_after_repair'])} |",
          f"| Fully correct (all hidden checks): before → after | {p(s['correct_before'])} → {p(s['correct_after'])} |",
          f"| Broken apps fixed | {s['broken_apps_fixed']} of {s['broken_apps']} |",
          f"| Correct apps broken by repair | {s['correct_apps_broken_by_repair']} of {s['correct_apps']} |",
          f"| Hidden checks passed (mean): before → after | {p(s['hidden_rate_before'])} → {p(s['hidden_rate_after'])} |",
          f"| False assurance among repaired-and-accepted | {p(s['false_assurance_after'])} |",
          f"| Model time per case (s) | {s['mean_model_seconds']:.0f} |"]
    (out_root / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/grid/demo1", help="saved grid whose rejected runs are repaired")
    ap.add_argument("--pipelines", nargs="+", default=["m1"])
    ap.add_argument("--llm-model", required=True, help="a name from SVAGA_MODEL_ENDPOINTS")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (0 = all)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    eps = model_endpoints()
    if a.llm_model not in eps:
        print(f"unknown model {a.llm_model}; SVAGA_MODEL_ENDPOINTS has {sorted(eps)}"); return 2
    src, out_root = ROOT / a.source, ROOT / "results" / "repair" / a.name
    cases = [p.parent for pl in a.pipelines for p in sorted((src / pl).glob("*/r*/record.json"))
             if json.loads(p.read_text(encoding="utf-8")).get("decision") == "REJECT" and (p.parent / "app.py").exists()]
    cases = cases[: a.limit] if a.limit else cases
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(vars(a) | {"endpoint": eps[a.llm_model], "cases": len(cases)}, indent=2), encoding="utf-8")
    print(f"{len(cases)} rejected apps from {a.source} {a.pipelines}; repair model {a.llm_model} at {eps[a.llm_model]}")
    t0 = time.monotonic()
    for i, d in enumerate(cases, 1):
        out = out_root / d.parent.parent.name / d.parent.name / d.name
        if (out / "record.json").exists():
            continue
        try:
            r = repair_case(d, out, endpoint=eps[a.llm_model], model=a.llm_model, rounds=a.rounds)
            print(f"[{i}/{len(cases)}] {r['pipeline']} {r['task']} r{r['repeat']}: diff {'ok' if r['diff_applied_fully'] else ('partial' if r['code_changed'] else 'none')}, "
                  f"{r['decision_before']} -> {r['decision_after']}, hidden {r['hidden_before']} -> {r['hidden_after']}  "
                  f"({r['wall_seconds']}s, elapsed {round((time.monotonic() - t0) / 60, 1)} min)", flush=True)
        except Exception as e:  # keep going; record the failure
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            print(f"[{i}/{len(cases)}] {d.parent.name} r: ERROR {type(e).__name__}: {e}", flush=True)
    print(summarize(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
