"""One place that turns a dataset row into (prompt text, target text). Training, evaluation and the SVAGA server all import it,
so the model always sees the same format at train time and at serving time."""
from __future__ import annotations

FALLBACK = "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"   # for base models without a chat template


def has_chat_template(tok) -> bool:
    return bool(getattr(tok, "chat_template", None))


def render_prompt(tok, messages, use_chat_template: bool) -> str:
    """Everything up to (not including) the assistant answer."""
    head = [m for m in messages if m["role"] != "assistant"]
    if use_chat_template:
        return tok.apply_chat_template(head, tokenize=False, add_generation_prompt=True)
    system = "\n".join(m["content"] for m in head if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in head if m["role"] == "user")
    return FALLBACK.format(system=system, user=user)


def target_of(messages) -> str:
    return "".join(m["content"] for m in messages if m["role"] == "assistant")


def encode(tok, row, use_chat_template: bool, max_len: int):
    """Token ids with prompt tokens masked (-100) so loss is computed only on the answer. Returns None if too long (never truncated)."""
    prompt = render_prompt(tok, row["messages"], use_chat_template)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if not use_chat_template and tok.bos_token_id is not None:
        p_ids = [tok.bos_token_id] + p_ids
    t_ids = tok(target_of(row["messages"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + t_ids}


# ---------------------------------------------------------------- tokenizer safety
# transformers 5.x can silently re-map some byte-level BPE tokenizers (seen with deepseek-coder) to a Llama/Metaspace
# tokenizer that drops spaces and newlines. Training on that corrupts the model, so every script loads the tokenizer
# here and refuses to continue unless text survives an encode -> decode round trip.

def roundtrip_failures(tok, texts):
    bad = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        back = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if back.strip() != t.strip():
            bad.append((t[:120], back[:120]))
    return bad


def load_tokenizer(model_id, revision=None, sample_texts=None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    probe = list(sample_texts or []) + ["def add(a, b):\n    return a + b\n", "Here's the fix:\n\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if roundtrip_failures(tok, probe):
        # load tokenizer.json exactly as published, without the class conversion
        raw = PreTrainedTokenizerFast.from_pretrained(model_id, revision=revision)
        for attr in ("bos_token", "eos_token", "pad_token", "unk_token", "chat_template"):
            if getattr(raw, attr, None) is None and getattr(tok, attr, None) is not None:
                setattr(raw, attr, getattr(tok, attr))
        tok = raw
    bad = roundtrip_failures(tok, probe)
    if bad:
        raise SystemExit("TOKENIZER CHECK FAILED: text does not survive encode->decode (spaces or newlines lost).\n"
                         f"example: {bad[0]}\nInstall transformers<5 (pip install \"transformers>=4.46,<5\") and run again.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
