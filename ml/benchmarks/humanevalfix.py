#!/usr/bin/env python3
"""HumanEvalFix (bug repair, public benchmark): base vs fine-tuned, prompted in the SAME format the SVAGA repair data uses
(issue text + FILE/LINES + buggy code, answer = unified diff). The model's diff is applied to the buggy file and the
original HumanEval tests are run on the result.

Metrics per condition:  diff_valid_pct (answer contains a usable unified diff), applies_pct (diff applies and changes the
file), pass_at_1_pct (applied AND tests pass), pass_given_applied_pct.

Needs func_eval.py in the same folder (and svaga_format.py via the kit, like func_eval.py).
  python humanevalfix.py --selftest                       # CPU only: checks dataset + diff applier + test runner
  python humanevalfix.py --limit 10 [path options]        # quick GPU check
  python humanevalfix.py [path options]                   # full run, conds base,tuned
Path options are the same as func_eval.py:  --config --data_dir --adapter --summary --out_dir
"""
import argparse, ast, difflib, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import func_eval as fe  # noqa: E402

USER_HEAD = ("Repair the issue using the supplied buggy Python source context. Return a unified diff only. "
             "Affected file locations are supplied; repository search is outside this task.\n\nISSUE:\n")


# ----------------------------------------------------------------------------- dataset
def build_prob(r):
    entry = r["entry_point"]
    imp = (r.get("import") or "").rstrip()
    decl = r["declaration"]
    head = (imp + "\n" if imp and imp not in decl else "") + decl
    buggy = head + r["buggy_solution"]
    good = head + r["canonical_solution"]
    symptom = (r.get("failure_symptom") or "").strip()
    doc = (r.get("docstring") or "").strip()
    issue = (f"`{entry}` in solution.py has a bug"
             + (f" ({symptom})" if symptom else "") + " and fails its tests. Fix the bug.\n\n"
             + (f"The function must behave as described here:\n{doc}" if doc else ""))
    n = len(buggy.splitlines())
    user = f"{USER_HEAD}{issue.rstrip()}\n\nFILE: solution.py\nLINES: 1-{n}\n{buggy}"
    return {"id": str(r["task_id"]), "entry": entry, "buggy": buggy, "good": good, "user": user,
            "tests": "\n" + r["test"] + f"\n\ncheck({entry})\n"}


def load_fix(limit=None):
    rows, err = None, None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("bigcode/humanevalpack", "data/python/data/humanevalpack.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:  # noqa: BLE001
        err = e
    if rows is None:
        try:
            from datasets import load_dataset
            rows = list(load_dataset("bigcode/humanevalpack", "python", split="test"))
        except Exception as e2:  # noqa: BLE001
            raise SystemExit(f"Could not load bigcode/humanevalpack (python).\n first error: {err!r}\n second error: {e2!r}")
    print(f"[humanevalfix] bigcode/humanevalpack python: {len(rows)} rows, columns = {sorted(rows[0].keys())}")
    probs = [build_prob(r) for r in rows]
    return probs[:limit] if limit else probs


# ----------------------------------------------------------------------------- diff handling
def parse_hunks(text):
    """Return a list of hunks {'start': int, 'old': [...], 'new': [...]} from the model's answer."""
    text = text.replace("\r\n", "\n")
    hunks, cur = [], None
    for ln in text.split("\n"):
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", ln)
        if m:
            cur = {"start": int(m.group(1)), "old": [], "new": []}
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith("```") or ln.startswith("diff --git"):
            cur = None
            continue
        if ln.startswith("\\"):
            continue
        if ln.startswith("+") and not ln.startswith("+++ "):
            cur["new"].append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("--- "):
            cur["old"].append(ln[1:])
        elif ln.startswith(" "):
            cur["old"].append(ln[1:]); cur["new"].append(ln[1:])
        elif ln == "":
            cur["old"].append(""); cur["new"].append("")
        elif ln.startswith("--- ") or ln.startswith("+++ "):
            continue
        else:
            cur = None
    return [h for h in hunks if any(l != "" for l in h["old"] + h["new"])]


def _find(lines, old, start, loose):
    n = len(old)
    norm = (lambda x: x.rstrip()) if loose else (lambda x: x)
    o = [norm(x) for x in old]
    for i in list(range(start, len(lines) - n + 1)) + list(range(0, min(start, len(lines) - n + 1))):
        if [norm(x) for x in lines[i:i + n]] == o:
            return i
    return -1


def apply_diff(src, answer):
    """Apply the unified diff in `answer` to `src` by matching hunk content (line numbers are not trusted).
    Returns (new_src or None, diff_valid)."""
    hunks = parse_hunks(answer)
    valid = any(any(l != "" for l in h["old"] + h["new"]) and h["old"] != h["new"] for h in hunks)
    if not valid:
        return None, False
    lines = src.split("\n")
    pos = 0
    for h in hunks:
        old, new = h["old"], h["new"]
        while old and old[-1] == "" and new and new[-1] == "" and len(old) > 1:
            old, new = old[:-1], new[:-1]
        if not old:
            i = max(0, min(len(lines), h["start"] - 1))
            lines[i:i] = new
            pos = i + len(new)
            continue
        i = _find(lines, old, pos, False)
        if i < 0:
            i = _find(lines, old, pos, True)
        if i < 0:
            return None, True
        lines[i:i + len(old)] = new
        pos = i + len(new)
    out = "\n".join(lines)
    return (out if out != src else None), True


# ----------------------------------------------------------------------------- selftest
def selftest():
    probs = load_fix()
    ok_apply = 0
    progs_good, progs_bug, progs_patch = [], [], []
    for p in probs:
        d = "\n".join(difflib.unified_diff(p["buggy"].split("\n"), p["good"].split("\n"), "a/solution.py", "b/solution.py", lineterm="", n=3))
        new, valid = apply_diff(p["buggy"], d)
        ok_apply += (new == p["good"])
        progs_good.append(p["good"]); progs_bug.append(p["buggy"]); progs_patch.append(new if new is not None else p["buggy"])
    g = sum(r[0] for r in fe.check_all(probs, progs_good))
    b = sum(r[0] for r in fe.check_all(probs, progs_bug))
    pa = sum(r[0] for r in fe.check_all(probs, progs_patch))
    n = len(probs)
    print(f"correct solutions pass tests : {g}/{n}   (must be >= 95%)")
    print(f"buggy versions pass tests    : {b}/{n}   (should be small; the bugs must be real)")
    print(f"reference diff applies & equals the fix: {ok_apply}/{n}   patched program passes: {pa}/{n}")
    print("SELFTEST", "OK" if g >= 0.95 * n and b <= 0.2 * n and pa >= 0.95 * n else "FAILED - do not trust results from this script until fixed")
    print("\nExample prompt the model will see:\n" + probs[0]["user"][:1200])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conds", default="base,tuned")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import torch, yaml
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    kit = Path(a.config).resolve().parent
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here)); sys.path.insert(0, str(kit))
    data_dir = Path(a.data_dir) if a.data_dir else (kit / "data" if (kit / "data" / "train.jsonl").exists() else here / "data")
    adapter_dir = Path(a.adapter) if a.adapter else kit / "outputs" / "final_adapter"
    summary_path = Path(a.summary) if a.summary else kit / "outputs" / "metrics" / "train_summary.json"
    from svaga_format import render_prompt, load_tokenizer
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out_dir) if a.out_dir else kit / "outputs" / "humanevalfix"
    out.mkdir(parents=True, exist_ok=True)
    use_ct = json.load(open(summary_path))["use_chat_template"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = load_tokenizer(cfg["model_id"], cfg.get("revision"))
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg.get("revision"), dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    sysmsg = fe.system_prompt(data_dir, task="repair")
    print(f"adapter: {adapter_dir}\nchat template: {use_ct}\nSYSTEM PROMPT (repair): {sysmsg[:200]}\n")

    probs = load_fix(a.limit)
    prompts = [render_prompt(tok, [{"role": "system", "content": sysmsg}, {"role": "user", "content": p["user"]}], use_ct)
               for p in probs]
    summary = {}
    for cond in a.conds.split(","):
        print(f"== humanevalfix / {cond}", flush=True)
        gens = fe.generate(model, tok, prompts, cond == "tuned", a.batch_size, a.max_new_tokens, add_bos=not use_ct)
        patched, flags = [], []
        for p, (t, _) in zip(probs, gens):
            new, valid = apply_diff(p["buggy"], t)
            patched.append(new if new is not None else p["buggy"])
            flags.append((valid, new is not None))
        res = fe.check_all(probs, patched)
        rows = []
        for p, (t, nt), (valid, applied), (ok, err) in zip(probs, gens, flags, res):
            passed = bool(applied and ok)
            rows.append({"id": p["id"], "passed": passed, "diff_valid": valid, "applied": applied, "new_tokens": nt,
                         "error": ("" if passed else (err.strip().splitlines()[-1] if applied and err.strip() else "diff did not apply" if valid else "no usable diff")),
                         "raw_output": t})
        with open(out / f"humanevalfix_{cond}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(rows)
        na = sum(r["applied"] for r in rows)
        summary[f"humanevalfix/{cond}"] = {
            "n": n, "pass_at_1_pct": round(100 * sum(r["passed"] for r in rows) / n, 1),
            "diff_valid_pct": round(100 * sum(r["diff_valid"] for r in rows) / n, 1),
            "applies_pct": round(100 * na / n, 1),
            "pass_given_applied_pct": round(100 * sum(r["passed"] for r in rows) / na, 1) if na else None,
            "mean_new_tokens": round(sum(r["new_tokens"] for r in rows) / n, 1)}
        print(summary[f"humanevalfix/{cond}"], flush=True)
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\nHUMANEVALFIX DONE:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
