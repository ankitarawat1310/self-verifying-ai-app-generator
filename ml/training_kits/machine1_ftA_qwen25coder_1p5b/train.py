"""QLoRA fine-tuning, identical settings for all four SVAGA models. Resumable.
Usage: python train.py --config config.yaml            (fresh run, or resumes automatically from the last checkpoint)
Outputs (outputs/): checkpoints/, final_adapter/, tokenizer/, metrics/train_log.json, metrics/train_summary.json, environment.txt"""
import argparse, json, math, os, platform, subprocess, sys, time
from pathlib import Path

import torch, yaml
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--max_steps", type=int, default=-1,
                                                help="debug: stop after N steps"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
kit = Path(a.config).resolve().parent
sys.path.insert(0, str(kit))
from svaga_format import encode, has_chat_template, load_tokenizer, target_of

out = kit / "outputs"; (out / "metrics").mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg["seed"])
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else torch.float16

sample = [target_of(json.loads(l)["messages"]) for _, l in zip(range(200), open(kit / "data" / "train.jsonl", encoding="utf-8"))]
tok = load_tokenizer(cfg["model_id"], cfg.get("revision"), sample)   # stops here if spaces/newlines would be lost
use_ct = has_chat_template(tok) if cfg.get("use_chat_template", "auto") == "auto" else bool(cfg["use_chat_template"])


def load(split, limit=None):
    rows = [json.loads(l) for l in open(kit / "data" / f"{split}.jsonl", encoding="utf-8")]
    enc = [e for e in (encode(tok, r, use_ct, cfg["max_len"]) for r in rows[:limit]) if e is not None]
    return enc, len(rows[:limit]) - len(enc)


train_ds, train_drop = load("train")
eval_ds, eval_drop = load("val", cfg.get("eval_subset"))


class Collate:
    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}


model = AutoModelForCausalLM.from_pretrained(
    cfg["model_id"], revision=cfg.get("revision"), torch_dtype=dtype, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=dtype))
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = cfg["lora"]
model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                                         target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            with open(out / "metrics" / "train_log.jsonl", "a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, "time": time.time(), **logs}) + "\n")


t = cfg["train"]
args = TrainingArguments(
    output_dir=str(out / "checkpoints"), per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=1,
    gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"], num_train_epochs=t["epochs"], max_steps=a.max_steps,
    lr_scheduler_type="cosine", warmup_ratio=t["warmup_ratio"], weight_decay=0.0, max_grad_norm=1.0,
    logging_steps=t["log_every"], eval_strategy="steps", eval_steps=t["eval_every"], save_strategy="steps", save_steps=t["save_every"],
    save_total_limit=t["keep_checkpoints"], bf16=bf16, fp16=not bf16, gradient_checkpointing=True, optim="paged_adamw_8bit",
    report_to="none", seed=cfg["seed"], data_seed=cfg["seed"], remove_unused_columns=False, dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=Collate(), callbacks=[LogCallback()])
last = get_last_checkpoint(str(out / "checkpoints")) if (out / "checkpoints").exists() else None
t0 = time.time()
result = trainer.train(resume_from_checkpoint=last)
final_eval = trainer.evaluate()
model.save_pretrained(str(out / "final_adapter"))
tok.save_pretrained(str(out / "tokenizer"))
summary = {"model_id": cfg["model_id"], "revision": cfg.get("revision"), "use_chat_template": use_ct, "bf16": bf16,
           "train_rows": len(train_ds), "train_rows_dropped_over_max_len": train_drop, "eval_rows": len(eval_ds),
           "trainable_params": trainable, "total_params": total, "resumed_from": last, "train_seconds": round(time.time() - t0),
           "train_loss": result.training_loss, "final_eval_loss": final_eval.get("eval_loss"),
           "final_eval_perplexity": math.exp(final_eval["eval_loss"]) if final_eval.get("eval_loss") else None,
           "global_steps": trainer.state.global_step, "gpu": torch.cuda.get_device_name(0), "config": cfg}
json.dump(summary, open(out / "metrics" / "train_summary.json", "w"), indent=2)
(out / "environment.txt").write_text(platform.platform() + "\n" + subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                                               capture_output=True, text=True).stdout)
print(json.dumps({k: summary[k] for k in ("train_loss", "final_eval_loss", "global_steps", "train_seconds")}, indent=2))
print("TRAINING DONE. Next: python evaluate.py --config config.yaml")
