# Fine-tuned model comparison

Models with complete results: 4 of 4. Re-run `python analysis/compare_models.py` when a new RETURN zip arrives.

Same dataset (10,155 train rows; generate, edit, repair balanced), same QLoRA settings (rank 16, alpha 32, 4-bit, lr 2e-4, 1 epoch, 16 rows per update).
Loss values are only comparable within a model (each model uses its own tokenizer); compare the reduction, not the raw loss.

| Model | GPU | Train rows | Minutes | Test loss base → fine-tuned | Reduction | Repair F1 | Repair exact | Edit valid diff | Generate parses |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder 1.5B | Tesla T4 | 10,150 | 926 | 0.573 → 0.446 | 22% | 0.07 → 0.21 | 0.00 → 0.03 | 1.00 → 1.00 | 0.87 → 0.97 |
| DeepSeek-Coder 1.3B | GeForce RTX 5080 Laptop GPU | 9,827 | 90 | 0.552 → 0.360 | 35% | 0.00 → 0.24 | 0.00 → 0.03 | 0.20 → 1.00 | 0.67 → 0.77 |
| Granite-3.3 2B | A100-SXM4-40GB | 10,034 | 126 | 0.661 → 0.405 | 39% | 0.00 → 0.29 | 0.00 → 0.07 | 0.40 → 1.00 | 0.90 → 0.87 |
| StarCoder2 3B | GeForce RTX 4090 | 10,034 | 69 | 0.541 → 0.419 | 22% | 0.00 → 0.16 | 0.00 → 0.00 | 0.93 → 1.00 | 0.30 → 1.00 |

Generation metrics are on 30 test examples per task: small samples, so differences of one or two examples (0.03 to 0.07) are noise.

Charts: `charts/1_training_curves.png`, `2_test_loss_by_task.png`, `3_generation_quality.png`, `4_cost_and_latency.png`. Examples: `examples.md`.