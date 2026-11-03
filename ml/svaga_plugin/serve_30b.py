"""Serve Qwen3-Coder-30B-A3B (base and fine-tuned) for SVAGA's M4 repair loop, from ONE copy of the model.
Speaks the same Ollama-style /api/chat as svaga_plugin/serve_svaga_model.py:
  model "qwen30b"      -> base + the fine-tuned LoRA adapter
  model "qwen30b-base" -> same weights with the adapter switched off
4-bit nf4 loading (about 18 GB of GPU memory), prompts rendered with the adapter's own chat template (as in training),
greedy when temperature is 0, decode without whitespace clean-up (clean_up_tokenization_spaces=False).
Usage: python serve_30b.py --adapter final_adapter --port 11600"""
import argparse, asyncio, dataclasses, json, shutil, tempfile, time
from pathlib import Path

import torch, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="folder with adapter_config.json, adapter_model.safetensors and the tokenizer")
ap.add_argument("--base_model", default=None, help="default: base_model_name_or_path from adapter_config.json")
ap.add_argument("--name", default="qwen30b")
ap.add_argument("--port", type=int, default=11600)
ap.add_argument("--max_new_tokens", type=int, default=1500)
ap.add_argument("--log", default="serve_log_30b.jsonl")
ap.add_argument("--no_quant", action="store_true", help="load in bf16 instead of 4-bit (needs an 80 GB GPU; faster)")
a = ap.parse_args()

src = Path(a.adapter)
cfg = json.loads((src / "adapter_config.json").read_text())
base_id = a.base_model or cfg["base_model_name_or_path"]
# The adapter was saved by a newer PEFT through Unsloth: keep only the fields this PEFT knows, and name the 4 attention
# projections it was trained on (q/k/v/o) instead of Unsloth's regex. Weights are untouched.
known = {f.name for f in dataclasses.fields(LoraConfig)}
clean = {k: v for k, v in cfg.items() if k in known}
clean["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
clean.pop("auto_mapping", None)
adir = Path(tempfile.mkdtemp(prefix="adapter30b_"))
shutil.copy(src / "adapter_model.safetensors", adir / "adapter_model.safetensors")
(adir / "adapter_config.json").write_text(json.dumps(clean, indent=1))

tok = AutoTokenizer.from_pretrained(str(src))
probe = "def f(x):\n    return ''.join(x)  # spaces\n"
assert tok.decode(tok(probe, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False) == probe, "TOKENIZER CHECK FAILED"
print(f"loading {base_id} {'in bf16' if a.no_quant else 'in 4-bit'} (first run downloads about 61 GB)...")
quant = None if a.no_quant else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                   bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map={"": 0} if torch.cuda.is_available() else "cpu",
                                             torch_dtype=torch.bfloat16, quantization_config=quant)
model = PeftModel.from_pretrained(model, str(adir))
model.eval()
n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
print(f"adapter attached: {n_lora} LoRA tensors (expected 384)")
assert n_lora == 384, "adapter did not attach to the expected modules"
if torch.cuda.is_available(): print(f"GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

app = FastAPI(title="SVAGA 30B model server")
LOCK = asyncio.Lock()
LOG = Path(a.log)


def generate(msgs, use_adapter, opts):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    temp = float(opts.get("temperature", 0.2))
    limit = min(int(opts.get("num_predict", a.max_new_tokens)), a.max_new_tokens)
    kw = dict(max_new_tokens=limit, do_sample=temp > 0, pad_token_id=tok.pad_token_id or tok.eos_token_id,
              repetition_penalty=float(opts.get("repeat_penalty", 1.0)))
    if temp > 0: kw["temperature"] = temp
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**ids, **kw)
        else:
            with model.disable_adapter():
                out = model.generate(**ids, **kw)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False), int(ids["input_ids"].shape[1]), int(new.shape[0])


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    name = body.get("model") or a.name
    use_adapter = not name.endswith("-base")
    t0 = time.time()
    async with LOCK:   # one generation at a time on the single GPU copy
        text, n_in, n_out = await asyncio.to_thread(generate, body.get("messages", []), use_adapter, body.get("options") or {})
    secs = time.time() - t0
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": name, "adapter": use_adapter,
                                "prompt_tokens": n_in, "new_tokens": n_out, "seconds": round(secs, 1), "raw": text}) + "\n")
    except Exception as e:
        print("log write failed:", e)
    return JSONResponse({"model": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "done": True, "done_reason": "stop",
                         "message": {"role": "assistant", "content": text}, "prompt_eval_count": n_in, "eval_count": n_out,
                         "total_duration": int(secs * 1e9), "load_duration": 0})


@app.get("/svaga/health")
async def health():
    return {"model": a.name, "base": base_id, "adapter_tensors": n_lora, "names": [a.name, a.name + "-base"]}


if __name__ == "__main__":
    print(f"serving {a.name} and {a.name}-base on http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)
