"""Check that Ollama really applies the fine-tuned adapter, and measure speed, before the long M4 run.
Sends one held-out REPAIR example from gold/test.jsonl (training format) to svaga-qwen30b-base and svaga-qwen30b-ft.
Usage (from this folder): python check_adapter.py"""
import json, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:11434/api/chat"
test = Path(__file__).resolve().parent.parent / "gold" / "test.jsonl"
rows = [json.loads(l) for l in test.open(encoding="utf-8")]
rows = [r for r in rows if "Task: REPAIR" in r["messages"][0]["content"] and len(r["messages"][1]["content"]) < 2500]
row = min(rows, key=lambda r: abs(len(r["messages"][1]["content"]) - 1500))


def ask(model):
    body = {"model": model, "stream": False, "messages": row["messages"][:2],
            "options": {"temperature": 0, "num_predict": 400, "repeat_penalty": 1.0}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    secs = time.time() - t0
    tps = d.get("eval_count", 0) / max(d.get("eval_duration", 1) / 1e9, 1e-9)
    return d["message"]["content"], secs, tps, d.get("prompt_eval_count")


out = {}
for m in ("svaga-qwen30b-base", "svaga-qwen30b-ft"):
    print(f"asking {m} (the first call loads the model, can take a few minutes)...", flush=True)
    try:
        out[m] = ask(m)
    except Exception as e:
        sys.exit(f"FAILED calling {m}: {e}\nIs Ollama running? Base model: ollama create svaga-qwen30b-base -f Modelfile.base. Fine-tune: python install_ft_model.py")
    text, secs, tps, n_in = out[m]
    print(f"--- {m}: {secs:.0f}s total, {tps:.1f} tokens/s, prompt {n_in} tokens\n{text[:700]}\n")

base, ft = out["svaga-qwen30b-base"][0], out["svaga-qwen30b-ft"][0]
print("Reference answer (first lines):\n" + "\n".join(row["messages"][2]["content"].splitlines()[:8]) + "\n")
if base.strip() == ft.strip():
    print("ADAPTER CHECK FAILED: base and fine-tuned answers are identical, so Ollama is not applying the adapter. Don't run M4; send this output.")
    sys.exit(1)
print("ADAPTER CHECK OK: the answers differ, so the adapter is active.")
print("Fine-tuned answer contains a diff hunk:", "@@" in ft)
tps = out["svaga-qwen30b-ft"][2]
print(f"Rough time for the full run (56 answers per model, about 600 tokens each, plus verification): "
      f"{2 * 56 * 600 / max(tps, 0.1) / 3600 + 0.5:.1f} hours for both models")
