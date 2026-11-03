"""Base model vs fine-tuned adapter, same loaded weights (adapter switched off/on), on the val and untouched test splits.
  1. loss and perplexity per task type   (lower is better)
  2. generations for N test rows per task type, scored automatically:
       generate: Python parses; edit/repair: valid unified diff, exact match to reference diff, changed-line F1 vs reference
Usage: python evaluate.py --config config.yaml [--gen_per_task 30]
Outputs: outputs/metrics/eval_loss.json, outputs/metrics/eval_generation.json, outputs/samples/test_generations.jsonl"""
import argparse, ast, json, math, re, sys, time
from pathlib import Path
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--gen_per_task", type=int, default=30)
ap.add_argument("--max_new_tokens", type=int, default=1024); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); kit = Path(a.config).resolve().parent; sys.path.insert(0, str(kit))
from svaga_format import encode, render_prompt, target_of, load_tokenizer
out = kit / "outputs"; (out / "metrics").mkdir(parents=True, exist_ok=True); (out / "samples").mkdir(exist_ok=True)
summary = json.load(open(out / "metrics" / "train_summary.json"))
use_ct = summary["use_chat_template"]
bf16 = torch.cuda.is_bf16_supported(); dtype = torch.bfloat16 if bf16 else torch.float16
tok = load_tokenizer(cfg["model_id"], cfg.get("revision"))
base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg.get("revision"), torch_dtype=dtype, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
model = PeftModel.from_pretrained(base, str(out / "final_adapter")); model.eval()
rows = {s: [json.loads(l) for l in open(kit / "data" / f"{s}.jsonl", encoding="utf-8")] for s in ("val", "test")}


@torch.no_grad()
def loss_of(row):
    e = encode(tok, row, use_ct, cfg["max_len"])
    if e is None: return None
    ids = torch.tensor([e["input_ids"]], device=model.device); lab = torch.tensor([e["labels"]], device=model.device)
    n = int((lab != -100).sum())
    return float(model(input_ids=ids, labels=lab).loss) * n, n


res = {}
for cond in ("base", "fine_tuned"):
    for split, rs in rows.items():
        agg = {}
        for r in rs:
            if cond == "base":
                with model.disable_adapter(): v = loss_of(r)
            else:
                v = loss_of(r)
            if v is None: continue
            s = agg.setdefault(r["task_type"], [0.0, 0]); s[0] += v[0]; s[1] += v[1]
        tot = [sum(x[0] for x in agg.values()), sum(x[1] for x in agg.values())]
        res[f"{cond}/{split}"] = {**{k: {"loss": x[0] / x[1], "perplexity": math.exp(x[0] / x[1]), "target_tokens": x[1]} for k, x in agg.items()},
                                  "all": {"loss": tot[0] / tot[1], "perplexity": math.exp(tot[0] / tot[1]), "target_tokens": tot[1]}}
        print(cond, split, round(res[f"{cond}/{split}"]["all"]["loss"], 4))
json.dump(res, open(out / "metrics" / "eval_loss.json", "w"), indent=2)


def changed(diff):
    return {l for l in diff.splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))}


def score(task, text, ref):
    if task == "generate":
        blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, flags=re.S) or [text]
        try:
            for b in blocks: ast.parse(b)
            return {"python_parses": 1}
        except SyntaxError:
            return {"python_parses": 0}
    ok = int(("---" in text and "+++" in text or text.lstrip().startswith("diff --git")) and "@@" in text)
    p, g = changed(text), changed(ref)
    tp = len(p & g); f1 = 0.0 if not p or not g else 2 * tp / (len(p) + len(g))
    return {"valid_diff": ok, "exact_match": int(p == g and bool(g)), "changed_line_f1": f1}


picked = []
for task in ("generate", "edit", "repair"):
    picked += [r for r in rows["test"] if r["task_type"] == task][: a.gen_per_task]
gens, agg = [], {}
for i, r in enumerate(picked):
    prompt = render_prompt(tok, r["messages"], use_ct)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=not use_ct).to(model.device)
    rec = {"sample_id": r["sample_id"], "task_type": r["task_type"], "reference": target_of(r["messages"])}
    for cond in ("base", "fine_tuned"):
        t0 = time.time()
        with torch.no_grad():
            if cond == "base":
                with model.disable_adapter():
                    o = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            else:
                o = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        sc = score(r["task_type"], text, rec["reference"])
        rec[cond] = {"output": text, "seconds": round(time.time() - t0, 2), "new_tokens": int(o.shape[1] - ids["input_ids"].shape[1]), **sc}
        for k, v in sc.items():
            s = agg.setdefault(f"{cond}/{r['task_type']}/{k}", [0, 0]); s[0] += v; s[1] += 1
        s = agg.setdefault(f"{cond}/{r['task_type']}/seconds", [0, 0]); s[0] += rec[cond]["seconds"]; s[1] += 1
    gens.append(rec); print(f"[{i+1}/{len(picked)}] {r['task_type']}")
with open(out / "samples" / "test_generations.jsonl", "w", encoding="utf-8") as f:
    for g in gens: f.write(json.dumps(g, ensure_ascii=False) + "\n")
json.dump({k: v[0] / v[1] for k, v in sorted(agg.items())}, open(out / "metrics" / "eval_generation.json", "w"), indent=2)
print(json.dumps({k: round(v[0] / v[1], 3) for k, v in sorted(agg.items())}, indent=2))
print("EVALUATION DONE. Next: python package_results.py --config config.yaml")
