# Public benchmarks (pass@1, greedy decoding)

| Benchmark | Model | Problems | Base | Fine-tuned | Change (pts) | Only fine-tuned solves | Only base solves | p |
|---|---|---|---|---|---|---|---|---|
| HumanEval+ | Qwen2.5-Coder 1.5B | 164 | 59.1% | 62.8% | +3.7 | 15 | 9 | 0.3075 |
| HumanEval+ | DeepSeek-Coder 1.3B | 164 | 61.6% | 56.1% | -5.5 | 8 | 17 | 0.1078 |
| HumanEval+ | Granite-3.3 2B | 164 | 51.2% | 47.6% | -3.6 | 14 | 20 | 0.3915 |
| HumanEval+ | StarCoder2 3B | 164 | 23.8% (plain prompt) | 32.9% | +9.1 | 25 | 10 | 0.0167 * |
| HumanEval+ | Qwen3-Coder 30B-A3B | 164 | 80.5% | 82.9% | +2.4 | 16 | 12 | 0.5716 |
| MBPP+ | Qwen2.5-Coder 1.5B | 378 | 56.1% | 57.4% | +1.3 | 34 | 29 | 0.6147 |
| MBPP+ | DeepSeek-Coder 1.3B | 378 | 53.2% | 54.5% | +1.3 | 20 | 15 | 0.4996 |
| MBPP+ | Granite-3.3 2B | 378 | 49.5% | 50.5% | +1.0 | 27 | 23 | 0.6718 |
| MBPP+ | StarCoder2 3B | 378 | 49.2% (plain prompt) | 44.7% | -4.5 | 35 | 52 | 0.0857 |
| MBPP+ | Qwen3-Coder 30B-A3B | 378 | 72.0% | 73.0% | +1.0 | 36 | 32 | 0.7163 |
| HumanEvalFix | Qwen2.5-Coder 1.5B | 164 | 6.1% | 4.9% | -1.2 | 8 | 10 | 0.8145 |
| HumanEvalFix | DeepSeek-Coder 1.3B | 164 | 4.9% | 6.1% | +1.2 | 9 | 7 | 0.8036 |
| HumanEvalFix | Granite-3.3 2B | 164 | 0.0% | 11.6% | +11.6 | 19 | 0 | <0.0001 * |
| HumanEvalFix | StarCoder2 3B | 164 | 1.2% | 17.7% | +16.5 | 28 | 1 | <0.0001 * |

## HumanEvalFix detail (base → fine-tuned)

| Model | Usable diff | Diff applies | Fixed, given it applied |
|---|---|---|---|
| Qwen2.5-Coder 1.5B | 65.2% → 85.4% | 30.5% → 44.5% | 20.0% → 11.0% |
| DeepSeek-Coder 1.3B | 29.9% → 72.6% | 12.2% → 32.3% | 40.0% → 18.9% |
| Granite-3.3 2B | 6.7% → 88.4% | 2.4% → 53.0% | 0.0% → 21.8% |
| StarCoder2 3B | 11.0% → 86.6% | 6.1% → 61.6% | 20.0% → 28.7% |

Base = the better of the base model's prompts (StarCoder2 is not instruction-tuned, so its plain completion prompt is the fair baseline).
p = exact paired McNemar test on the problems where the two disagree; * = p < 0.05. HumanEvalFix was not run for the 30B.
Runner: the team's own scripts (not the official evalplus harness), greedy decoding, one sample per problem.