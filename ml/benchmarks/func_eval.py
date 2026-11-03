#!/usr/bin/env python3
"""Functional benchmark: base StarCoder2-3B vs the fine-tuned adapter, same prompts, same greedy decoding.

Benchmarks: HumanEval+ (164) and MBPP+ (378), pass@1 on the plus tests (base tests also reported).
Needs svaga_format.py next to this script, a config.yaml, a data/train.jsonl (only to read the system prompt) and the adapter.
For a RETURN_* folder that lacks pieces, point at them:
  python func_eval.py --config RETURN_x/config.yaml --data_dir kit/data --adapter RETURN_x/final_adapter --summary RETURN_x/outputs/metrics/train_summary.json --out_dir results_x

  python func_eval.py --selftest 20          # NO model, CPU only: checks the data and the test runner (canonical solutions must pass)
  python func_eval.py --limit 10             # quick end-to-end check on the GPU (10 problems per benchmark)
  python func_eval.py                        # full run: base, fine-tuned and base_plain (plain-completion baseline, only for models without a chat template)

Model-written code is executed in a subprocess with a time limit and a memory limit. Run it on the pod, not on a personal computer.
Outputs: outputs/func_eval/*.jsonl (every raw output), summary.json, side_by_side.md
"""
import argparse, ast, json, os, re, resource, subprocess, sys, tempfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

STOPS = ["### User:", "### System:"]
HE_USER = ("Complete the following Python function. Reply with the complete function in a single ```python code block.\n\n"
           "```python\n{prompt}```")
MBPP_USER = ("Write a Python function that solves the task below. Reply with the complete code in a single ```python code block. "
             "Do not include test code.\n\n{task}\n\nYour code must pass this test:\n{test}")
# base_plain = plain code completion (how base models are normally evaluated): stop when the model starts something new
PLAIN_STOPS = {"humaneval": ["\ndef ", "\nclass ", "\nif ", "\nprint(", "\n@", "\n#", '\n"""', "### User:"],
               "mbpp": ["\nassert", "\nprint(", "\nif __name__", '\n"""', "\n\n\n", "### User:"]}


# ----------------------------------------------------------------------------- problems
def _row_get(row, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def load_problems(bench, limit=None):
    from datasets import load_dataset
    repo = {"humaneval": "evalplus/humanevalplus", "mbpp": "evalplus/mbppplus"}[bench]
    ds = load_dataset(repo, split="test")
    print(f"[{bench}] {repo}: {len(ds)} rows, columns = {ds.column_names}")
    probs = []
    for row in ds:
        if bench == "humaneval":
            prompt, entry, tests = row["prompt"], row["entry_point"], row["test"]
            probs.append({"id": str(row["task_id"]), "bench": bench, "prompt": prompt, "entry": entry,
                          "user": HE_USER.format(prompt=prompt.rstrip() + "\n"), "plain": prompt.rstrip(),
                          "tests": tests + f"\n\ncheck({entry})\n",
                          "canonical": prompt + row["canonical_solution"]})
        else:
            task = str(row["prompt"]).strip().strip('"').strip()
            imports = "\n".join(_row_get(row, "test_imports", default=[]) or [])
            tl = _row_get(row, "test_list", default=[]) or []
            first_test = str(tl[0]).strip() if len(tl) else ""
            probs.append({"id": str(row["task_id"]), "bench": bench, "prompt": task, "entry": None,
                          "user": MBPP_USER.format(task=task, test=first_test),
                          "plain": f'"""\n{task}\n{first_test}\n"""\n',
                          "tests": "\n" + str(row["test"]) + "\n", "imports": imports,
                          "canonical": str(_row_get(row, "code", "canonical_solution", default=""))})
    if limit:
        probs = probs[:limit]
    return probs


# ----------------------------------------------------------------------------- output -> program
def extract_code(text):
    for s in STOPS:
        text = text.split(s)[0]
    blocks = [b for b in re.findall(r"```(?:python|py|Python)?[ \t]*\r?\n(.*?)(?:```|\Z)", text, flags=re.S) if b.strip()]
    if blocks:
        withdef = [b for b in blocks if re.search(r"^\s*def ", b, re.M)]
        return (withdef or blocks)[0], True
    return text, False


def cut_at(text, stops):
    for s_ in stops:
        i = text.find(s_)
        if i >= 0:
            text = text[:i]
    return text


def build_program(p, code):
    """Full program text (without tests) from the model's code."""
    if p["bench"] == "humaneval":
        if re.search(rf"^\s*def\s+{re.escape(p['entry'])}\b", code, re.M):
            return p["prompt"].rstrip() + "\n\n" + code          # prompt keeps its imports and helper functions; model def overrides
        body = code if code[:1].isspace() else "\n".join("    " + l for l in code.splitlines())
        return p["prompt"] + body                                # model gave only a function body
    return p.get("imports", "") + "\n" + code


# ----------------------------------------------------------------------------- sandboxed runner
def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
    os.setsid()


def run_program(src, timeout=30):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
            return r.returncode == 0, r.stderr.decode("utf-8", "replace")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def check_all(probs, programs, workers=8):
    srcs = [programs[i] + "\n\n" + probs[i]["tests"] for i in range(len(probs))]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run_program, srcs))


# ----------------------------------------------------------------------------- model
def system_prompt(data_dir, task="generate"):
    c = Counter()
    with open(Path(data_dir) / "train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("task_type") == task:
                c["\n".join(m["content"] for m in r["messages"] if m["role"] == "system")] += 1
                if sum(c.values()) >= 300:
                    break
    return c.most_common(1)[0][0]


def generate(model, tok, prompts, adapter_on, bs, max_new, add_bos, stops=None):
    stops = stops or STOPS
    import torch
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = {}
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        pre = [tok.bos_token_id] if (add_bos and tok.bos_token_id is not None) else []   # same rule as training: BOS only without a chat template
        enc = [pre + tok(prompts[i], add_special_tokens=False)["input_ids"] for i in idx]
        batch = tok.pad({"input_ids": enc}, padding=True, return_tensors="pt").to(model.device)
        ctx = nullcontext() if adapter_on else model.disable_adapter()
        with torch.no_grad(), ctx:
            o = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                               stop_strings=stops, tokenizer=tok)
        L = batch["input_ids"].shape[1]
        for j, i in enumerate(idx):
            gen = o[j][L:]
            outs[i] = (tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False), int((gen != tok.pad_token_id).sum()))   # cleanup=False: the default deletes spaces before . , ! ? and breaks code
        print(f"   generated {min(s + bs, len(order))}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    return [outs[i] for i in range(len(prompts))]


# ----------------------------------------------------------------------------- main
def selftest(n):
    ok_all = True
    for bench in ("humaneval", "mbpp"):
        probs = load_problems(bench, n)
        res = check_all(probs, [build_program(p, p["canonical"]) if bench == "mbpp" else p["canonical"] for p in probs])
        passed = sum(r[0] for r in res)
        print(f"[{bench}] canonical solutions passing: {passed}/{len(probs)}")
        for p, (ok, err) in zip(probs, res):
            if not ok:
                print("   FAIL", p["id"], err.strip().splitlines()[-1:] or "")
        ok_all &= passed >= 0.95 * len(probs)
    print("SELFTEST", "OK" if ok_all else "FAILED - do not trust results from this script until fixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", type=int, default=None)
    ap.add_argument("--benchmarks", default="humaneval,mbpp")
    ap.add_argument("--conds", default="base,tuned,base_plain")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    # Paths: by default everything is looked up next to config.yaml. Override when a kit folder lacks a piece
    # (for example the RETURN_* folders do not contain data/train.jsonl).
    ap.add_argument("--data_dir", default=None, help="folder with train.jsonl (used only to read the system prompt)")
    ap.add_argument("--adapter", default=None, help="folder with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--summary", default=None, help="train_summary.json (use_chat_template)")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.selftest)

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
    out = Path(a.out_dir) if a.out_dir else kit / "outputs" / "func_eval"
    out.mkdir(parents=True, exist_ok=True)
    use_ct = json.load(open(summary_path))["use_chat_template"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = load_tokenizer(cfg["model_id"], cfg.get("revision"))
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg.get("revision"), dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    sysmsg = system_prompt(data_dir)
    print(f"adapter: {adapter_dir}\ndata (system prompt): {data_dir}\nchat template: {use_ct}")
    print("SYSTEM PROMPT USED (from the training data, generate task):\n", sysmsg[:400], "\n")

    summary, results = {}, {}
    for bench in a.benchmarks.split(","):
        probs = load_problems(bench, a.limit)
        prompts = [render_prompt(tok, [{"role": "system", "content": sysmsg}, {"role": "user", "content": p["user"]}], use_ct)
                   for p in probs]
        for cond in a.conds.split(","):
            plain = cond == "base_plain"
            if plain and use_ct:
                print(f"== {bench} / base_plain: skipped (this model has a chat template; its 'base' condition is already the right one)")
                continue
            print(f"== {bench} / {cond}")
            gens = generate(model, tok, [p["plain"] for p in probs] if plain else prompts, cond == "tuned", a.batch_size,
                            a.max_new_tokens, add_bos=not use_ct, stops=PLAIN_STOPS[bench] if plain else None)
            codes = [(cut_at(t, PLAIN_STOPS[bench]), False) for t, _ in gens] if plain else [extract_code(t) for t, _ in gens]
            programs = [build_program(p, c) for p, (c, _) in zip(probs, codes)]
            res = check_all(probs, programs)
            rows = []
            for p, (t, nt), (c, fenced), (ok, err), prog in zip(probs, gens, codes, res, programs):
                try:
                    ast.parse(prog); parses = True
                except SyntaxError:
                    parses = False
                rows.append({"id": p["id"], "passed": ok, "fenced": fenced, "parses": parses, "new_tokens": nt,
                             "error": err.strip().splitlines()[-1] if err.strip() else "", "raw_output": t})
            results[(bench, cond)] = rows
            with open(out / f"{bench}_{cond}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n = len(rows)
            summary[f"{bench}/{cond}"] = {"n": n, "pass_at_1_pct": round(100 * sum(r["passed"] for r in rows) / n, 1),
                                          "parses_pct": round(100 * sum(r["parses"] for r in rows) / n, 1),
                                          "code_block_pct": round(100 * sum(r["fenced"] for r in rows) / n, 1),
                                          "mean_new_tokens": round(sum(r["new_tokens"] for r in rows) / n, 1)}
            print(summary[f"{bench}/{cond}"], flush=True)
            json.dump(summary, open(out / "summary.json", "w"), indent=2)

    # side-by-side examples for the slides
    md = ["# Base vs fine-tuned: examples\n"]
    for bench in a.benchmarks.split(","):
        if (bench, "base") in results and (bench, "tuned") in results:
            b, t = results[(bench, "base")], results[(bench, "tuned")]
            groups = {"fine-tuned passes, base fails": [i for i in range(len(b)) if t[i]["passed"] and not b[i]["passed"]],
                      "base passes, fine-tuned fails": [i for i in range(len(b)) if b[i]["passed"] and not t[i]["passed"]],
                      "both fail": [i for i in range(len(b)) if not b[i]["passed"] and not t[i]["passed"]]}
            md.append(f"\n## {bench}: {', '.join(f'{k} = {len(v)}' for k, v in groups.items())}\n")
            for k, v in groups.items():
                for i in v[:3]:
                    md.append(f"\n### {bench} {b[i]['id']} ({k})\n**Base:**\n```\n{b[i]['raw_output'][:900]}\n```\n"
                              f"**Fine-tuned:**\n```\n{t[i]['raw_output'][:900]}\n```\n")
    (out / "side_by_side.md").write_text("\n".join(md), encoding="utf-8")
    print("\nFUNC EVAL DONE:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
