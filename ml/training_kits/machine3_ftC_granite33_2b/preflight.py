"""Run first on each machine (about 2 minutes). Checks GPU, downloads the tokenizer, verifies data hashes and exact token lengths.
Usage: python preflight.py --config config.yaml"""
import argparse, hashlib, json, sys
from pathlib import Path
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
kit = Path(a.config).resolve().parent
report = {"model_id": cfg["model_id"]}
try:
    import torch
    report["cuda"] = torch.cuda.is_available()
    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else 0
    report["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
except Exception as e:
    report["cuda_error"] = repr(e)
man = json.load(open(kit / "data" / "manifest.json"))
for name, info in man["files"].items():
    got = hashlib.sha256((kit / "data" / name).read_bytes()).hexdigest()
    report[f"hash_ok_{name}"] = got == info["sha256"]
sys.path.insert(0, str(kit))
from svaga_format import encode, has_chat_template, load_tokenizer, target_of
import transformers
report["transformers"] = transformers.__version__
sample = [target_of(json.loads(l)["messages"]) for _, l in zip(range(200), open(kit / "data" / "train.jsonl", encoding="utf-8"))]
tok = load_tokenizer(cfg["model_id"], cfg.get("revision"), sample)
report["tokenizer_class"] = type(tok).__name__
report["tokenizer_roundtrip_ok"] = True
use_ct = has_chat_template(tok) if cfg.get("use_chat_template", "auto") == "auto" else bool(cfg["use_chat_template"])
report["use_chat_template"] = use_ct
for split in ("train", "val", "test"):
    rows = [json.loads(l) for l in open(kit / "data" / f"{split}.jsonl", encoding="utf-8")]
    lens, over = [], 0
    for r in rows:
        e = encode(tok, r, use_ct, cfg["max_len"])
        if e is None: over += 1
        else: lens.append(len(e["input_ids"]))
    lens.sort()
    report[split] = {"rows": len(rows), "over_max_len_dropped": over, "median_tokens": lens[len(lens)//2], "max_tokens": lens[-1],
                     "target_tokens": None}
print(json.dumps(report, indent=2))
(kit / "outputs").mkdir(exist_ok=True)
json.dump(report, open(kit / "outputs" / "preflight.json", "w"), indent=2)
bad = [k for k, v in report.items() if k.startswith("hash_ok") and not v]
if bad or not report.get("cuda"):
    sys.exit(f"PREFLIGHT FAILED: {bad or 'no CUDA GPU'}")
print("PREFLIGHT OK")
