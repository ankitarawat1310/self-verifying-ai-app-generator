# Fine-tuned models inside SVAGA (M1, frozen test tasks)

Granite-3.3 2B, base and fine-tuned, served through `serve_svaga_model.py`, run with M1 on the frozen test split. No configuration
produced a working app, so no hidden-check comparison is possible yet. The failure mode is the finding.

| Attempt | Setting | Base Granite | Fine-tuned Granite |
|---|---|---|---|
| 1. 15 test tasks | Free-form output, 3,000-token cap | 13/15 invalid JSON (raw line breaks in code strings); 2 apps built, both failed verification | 10/15 invalid JSON (cut off or a diff instead of JSON), 3 timeouts; 2 apps built, both failed |
| 2. 1 task | JSON forced while generating, 3,000-token cap | (not run) | JSON valid in structure but cut off at the cap while writing tests |
| 3. 1 task | JSON forced, 6,000 tokens (Demo 1 limit) | (not run) | Same endpoint lines repeated until the cap; app never finished |
| 4. 1 task | JSON forced, 6,000 tokens, repetition penalty 1.1 (Ollama default, as Demo 1) | Complete answer; app had a syntax error at line 50, so 0 of 13 checks | Invented a variable name that grew without end until the cap |

What it shows:
- SVAGA asks for an entire app and its tests in one answer. The 30B model in Demo 1 could do this; 1.3B to 3B models cannot, even with
  forced JSON and the same decoding settings.
- Fine-tuning on snippet, edit and repair data made long whole-app answers worse (loops), while it made short code and diffs better
  (see the comparison and public benchmarks).
- Speed: about 8 tokens per second for a 4-bit 2B model on the laptop RTX 5080 with forced JSON, so a 6,000-token answer takes about
  12 minutes; the full 15-task run was not feasible before the deadline.

Next steps this points to: use the small fine-tuned models where their output is short (SVAGA's repair step, which takes a diff), and
train round 2 on whole SVAGA app answers that SVAGA itself has verified. Raw answers are logged in `returned/<run>/serve_log_<port>.jsonl`.
