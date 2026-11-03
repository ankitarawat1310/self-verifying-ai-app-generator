# M4: fine-tuned model as SVAGA's repairer

**Setup.** The 28 apps that M1 rejected in the frozen Demo 1 grid (20 truly broken, 8 correct but rejected). The repair model gets the verifier's failure messages plus `app.py` and `test_app.py` in the training prompt format and answers with a unified diff. SVAGA applies it, re-runs verification and the release gate, and the hidden judge scores the result. Up to 2 rounds, greedy decoding, identical prompts for every model. The model never sees hidden checks.

Two model sizes, each base vs fine-tuned on the same 10K dataset:
- **Granite-3.3 2B**, run on Colab (transformers, 4-bit).
- **Qwen3-Coder 30B-A3B**, run locally on Ollama's `qwen3-coder:30b` (4-bit). The fine-tuned version is the same Ollama model plus the team's LoRA adapter. The adapter was confirmed active: all 28 fine-tuned answers use the training diff format, and none of the 28 first answers match the base model's.

| Measure | Granite 2B base | Granite 2B fine-tuned | Qwen 30B base | Qwen 30B fine-tuned |
|---|---|---|---|---|
| Answer is a valid diff | 18% | 100% | 71% | 100% |
| Diff applies fully | 0% | 7% | 43% | 14% |
| Code changed | 4% | 100% | 50% | 86% |
| Changes per first answer (median diff hunks) | 0.5 | 6 | 1 | 10 |
| Gate accepts the repaired app | 0% | 0% | 0% | 0% |
| Broken apps fixed | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| Hidden checks passed by the repair candidate, before → after | 78% → 75% | 78% → 37% | 78% → 71% | 78% → 44% |
| Correct apps whose candidate got worse | 0 of 8 | 6 of 8 | 2 of 8 | 5 of 8 |
| Damaged apps shipped (false assurance) | 0 | 0 | 0 | 0 |
| Model time, all 28 apps | 40 min | 104 min | 12 min | 20 min |

**Result: negative at both sizes.** No model fixed a broken app. The closest was the base 30B on one app (`connector_shipping_quote` r3), which went from 0/11 to 10/11 hidden checks but still failed SVAGA's own tests and was rejected. The gate rejected every repaired candidate, so no damaged app was shipped by any model.

**Fine-tuning made app-scale repair worse, at both sizes.** It reliably taught the output format (valid diffs rose to 100% for both models). But the fine-tuned models edit many places at once (median 6 and 10 hunks per answer, against 1 for the base 30B and 1 in the training answers). Those edits are mostly unrelated to the reported failure, e.g. extra field limits or type hints when the failing test was a `KeyError`. Many of their diffs have context lines that don't match the file, so only parts apply, and the partial edits break working code. Fine-tuned candidates lost 34 to 41 points of hidden checks, against 3 to 7 for the bases.

**Likely causes (from the saved answers):**
- **Input size mismatch.** SVAGA's repair prompts (median 10,327 characters: the whole app plus tests) are 2.4 times longer than the training repair rows (median 4,343 characters, an excerpt around one bug).
- **No explanation step.** The training answers are a diff only. The base 30B writes an explanation first ("Looking at the...") and then a short diff, and that version was more conservative and applied more often. Training on diff-only answers may have removed the step where the model works out the bug. This is a hypothesis, not tested here.

**How this fits the other results.** On HumanEvalFix (short, single-function bugs), fine-tuning gave significant gains for Granite (0% → 11.6%) and StarCoder2 (1.2% → 17.7%). The skill learned on short excerpts did not carry over to repairing a 109 to 286 line app.

**Next steps (future work):**
- Send only the failing function and traceback instead of the whole file.
- Reject a patch unless every hunk applies, as `git apply` does.
- Fine-tune on SVAGA's own rejected-then-fixed apps, with an explanation before the diff.

Raw cases: `results/repair/repair_granite_{base,tuned}/` and `results/repair/repair_qwen30b_local_{base,tuned}/` (prompt, answer, apply report, verification and judge for every case).
