"""M4 counterexample-guided repair with a fine-tuned repair model.

Loop: SVAGA's own verifiers fail -> their failures become an ISSUE -> the repair model (served by svaga_plugin/serve_svaga_model.py,
in the exact prompt format it was fine-tuned on) answers with a unified diff -> the diff is applied to app.py / test_app.py ->
full verification and the release gate run again. Up to N rounds.

Leakage rule: the repair model only ever sees the public task text, the app, its own tests and the verifiers' messages.
Hidden checks are used only afterwards, by the judge, to score the result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

REPAIR_SYSTEM = "You are SVAGA's code model. Task: REPAIR. Fix the bug in the given code and reply with a unified diff only."
REPAIR_HEADER = ("Repair the issue using the supplied buggy Python source context. Return a unified diff only. "
                 "Affected file locations are supplied; repository search is outside this task.")


# ---------------------------------------------------------------- inputs
def strip_platform_blocks(app_code: str) -> str:
    """Remove routes SVAGA injects into every generated app (/, /health, the web UI) so the model sees only its own code."""
    from shared.generation.simple_ui import _strip_platform_roots
    code = _strip_platform_roots(app_code)
    code = re.sub(r'\n*@app\.get\("/health"\)\ndef _svaga_health\(\):\n    return \{[^\n]*\}\n', "\n", code)
    marker = "from fastapi.responses import HTMLResponse as _SvagaHTMLResponse"
    if marker in code:
        code = code[: code.index(marker)].rstrip() + "\n"
    return code


def _short(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 20] + "\n... (truncated)"


def build_issue(task_prompt: str, verification: dict[str, Any], *, max_chars: int = 2500) -> str:
    """Turn failed verifiers into an issue report, like the bug reports the model was trained on."""
    lines = ["The application does not meet its specification. Automated checks failed.", "",
             "Specification:", _short(task_prompt.split("Public interface")[0], 900), ""]
    for v in verification.get("verifiers", []):
        if v.get("passed"):
            continue
        d = v.get("details") or {}
        name = v.get("name")
        if name == "pytest_sandbox":
            out = str(d.get("stdout", ""))
            failed = re.findall(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", out, flags=re.M)
            lines.append(f"Unit tests: {d.get('failed_tests', '?')} of {d.get('total_tests', '?')} failed.")
            for test, msg in failed[:8]:
                lines.append(f"- {test.split('::')[-1]}: {_short(msg, 160)}")
            if not failed:
                lines.append(_short(out[-600:], 600))
        elif name == "contract_checks":
            missing = d.get("missing_paths") or []
            if missing:
                lines.append("Required routes are missing: " + ", ".join(map(str, missing)))
        elif name == "api_fuzz":
            for f in (d.get("failures") or [])[:5]:
                lines.append(f"- Unexpected request crashed the server: {_short(f, 200)}")
        elif name == "policy_static":
            lines.append("Policy violations: " + "; ".join(map(str, d.get("violations") or []))[:400])
        elif name == "spec_invariants":
            lines.append("Rule violations: " + "; ".join(map(str, d.get("violations") or []))[:400])
        elif name == "generation_output":
            lines.append("Generation problem: " + _short(d.get("error", ""), 300))
        else:
            lines.append(f"{name} failed: {_short(json.dumps(d, default=str), 300)}")
    return _short("\n".join(lines), max_chars)


def build_repair_prompt(issue: str, files: dict[str, str]) -> str:
    parts = [REPAIR_HEADER, "", "ISSUE:", issue, ""]
    for path, code in files.items():
        n = code.count("\n") + (0 if code.endswith("\n") else 1)
        parts += [f"FILE: {path}", f"LINES: 1-{n}", code.rstrip("\n"), "", ""]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------- diff handling
def extract_diff(text: str) -> str:
    fenced = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.S)
    body = "\n".join(fenced) if fenced else text
    start = min([i for i in (body.find("diff --git"), body.find("--- ")) if i >= 0], default=-1)
    return body[start:] if start >= 0 else ""


@dataclass
class ApplyReport:
    files_touched: list[str] = field(default_factory=list)
    hunks: int = 0
    applied: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.hunks > 0 and self.applied == self.hunks

    def to_dict(self) -> dict[str, Any]:
        return {"files_touched": self.files_touched, "hunks": self.hunks, "applied": self.applied, "failed": self.failed, "ok": self.ok}


def _norm(line: str) -> str:
    return line.rstrip()


def _find_block(haystack: list[str], needle: list[str], hint: int) -> int:
    """Index where `needle` occurs in `haystack`, nearest the hinted line; exact first, then ignoring trailing whitespace,
    then ignoring all indentation differences. -1 if absent."""
    if not needle:
        return min(max(hint, 0), len(haystack))
    for key in (lambda s: s, _norm, lambda s: s.strip()):
        h = [key(x) for x in haystack]; n = [key(x) for x in needle]
        hits = [i for i in range(len(h) - len(n) + 1) if h[i:i + len(n)] == n]
        if hits:
            return min(hits, key=lambda i: abs(i - hint))
    return -1


def apply_unified_diff(files: dict[str, str], diff: str) -> tuple[dict[str, str], ApplyReport]:
    """Apply a unified diff to in-memory files. Hunks are located by their content (line numbers are only a hint),
    since model-written line numbers are often off. Unknown files are ignored and counted as failed hunks."""
    rep = ApplyReport()
    out = {k: v for k, v in files.items()}
    current = None
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            path = line[4:].strip().split("\t")[0]
            path = re.sub(r"^[ab]/", "", path)
            current = next((f for f in out if f == path or f.endswith("/" + path) or path.endswith("/" + f) or path.endswith(f)), None)
            if current and current not in rep.files_touched:
                rep.files_touched.append(current)
            i += 1; continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            hint = int(m.group(1)) - 1
            old, new = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "+++ ", "diff --git")):
                h = lines[i]
                if h.startswith("+"):
                    new.append(h[1:])
                elif h.startswith("-"):
                    old.append(h[1:])
                elif h.startswith("\\"):
                    pass
                else:
                    ctx = h[1:] if h.startswith(" ") else h
                    old.append(ctx); new.append(ctx)
                i += 1
            while old and new and old[-1] == "" and new[-1] == "":
                old.pop(); new.pop()
            rep.hunks += 1
            if current is None:
                rep.failed.append(f"hunk {rep.hunks}: file not found"); continue
            src = out[current].split("\n")
            at = _find_block(src, old, hint)
            if at < 0:
                rep.failed.append(f"hunk {rep.hunks} ({current}): context not found"); continue
            out[current] = "\n".join(src[:at] + new + src[at + len(old):])
            rep.applied += 1
            continue
        i += 1
    return out, rep


# ---------------------------------------------------------------- model call
def call_repair_model(endpoint: str, model_name: str, prompt: str, *, max_new_tokens: int = 1500, timeout: float = 900) -> dict[str, Any]:
    """Plain-text chat call to the SVAGA model server (Ollama-compatible /api/chat, no JSON format: the answer is a diff).
    Greedy decoding and no repetition penalty, matching how the models were evaluated after fine-tuning."""
    import httpx
    payload = {"model": model_name, "stream": False,
               "messages": [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_predict": max_new_tokens, "repeat_penalty": 1.0}}
    r = httpx.post(endpoint.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return {"text": (data.get("message") or {}).get("content", ""), "tokens": int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0),
            "seconds": round((data.get("total_duration") or 0) / 1e9, 1)}
