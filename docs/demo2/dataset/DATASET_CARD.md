# SVAGA fine-tuning dataset v1 (dataset card)

Built 2026-10-29 by the SVAGA team (DATA 298B, Group 4). Seed 2026. Pipeline: `pipeline/` (Airflow DAG `svaga_dataset_v1`), warehouse `warehouse/svaga.duckdb`.

## Purpose
Teach small open code models the three jobs SVAGA needs: **generate** Python from a request, **edit** existing code as asked, and **repair** a bug.
It is a skills dataset. It does not yet contain full SVAGA app bundles (app + tests + policy); that is round 2 (verified SVAGA runs).

## Contents
| Split | generate | edit | repair | Total |
|---|---|---|---|---|
| train | 3,385 | 3,385 | 3,385 | 10,155 |
| val | 189 | 190 | 190 | 569 |
| test (untouched) | 190 | 190 | 190 | 570 |

Training rows are balanced by count. By estimated tokens the training split is generate 30%, edit 17%, repair 53%, because repair examples are longer (median 1,460 vs 904 and 523 tokens).

## Sources
| Task | Source | Raw rows | Upstream package | Notes |
|---|---|---|---|---|
| generate | m-a-p/CodeFeedback-Filtered-Instruction | 156,526 | 3,889 (KS) | web/API filter level 2, about 75% on-topic in a manual sample |
| edit | bigcode/commitpackft (Python) | 56,025 | 8,060 (KS) | commit message + old file -> diff; per-row licenses kept |
| repair | SWE-bench/SWE-smith-py rev 77cab905 | 50,908 | 11,000 (AR) | issue + buggy functions -> diff; repository-separated split |

## Processing
1. bronze: rows loaded unchanged with file hashes; upstream funnel counts from the KS and AR build reports.
2. silver: schema check; Python must parse (Python-labelled code blocks only); edit/repair targets must be unified diffs;
   license required (dataset-level licence for CodeFeedback and SWE-smith rows, marked `dataset_card_to_verify`);
   length <= 4,096 tokens (exact Qwen count for SWE-smith, chars/3.2 estimate otherwise); decontamination: any 13-word overlap with
   HumanEval, HumanEvalFix, MBPP+ or the 20 SVAGA benchmark prompts is removed; exact duplicates removed. Every drop is in `ops.filter_log`.
3. gold: balanced sample per task, repository round-robin for repair, seed 2026. Unselected rows stay in the warehouse (`gold.unused`).
   Splits: generate test carved from the training pool; edit val/test from the KS validation file; repair val and test use
   disjoint repositories that never appear in training.
4. Quality gates (all passed): no row or content in two splits, repair repositories disjoint across splits, no unknown licence,
   no empty target, training split balanced.
5. ml_ready: chat `messages` (task-specific system message, user = input, assistant = target), shuffled so every batch mixes tasks.

## Removed in cleaning
- codefeedback: validity: python_does_not_parse (122)
- codefeedback: length: over_4096_estimated_tokens (3)
- commitpackft: license: license_unknown (3)

## Known limits
- Public data is mostly snippets and edits, not prompt-to-app examples.
- SWE-smith rows are `independently_verified: false`, `locally_replayed: false` (carried from AR's package); affected locations are supplied.
- CodeFeedback rows have no per-row licence; the dataset-level licence must be confirmed on the dataset card.
- About 25% of generate rows are off-topic (general Python rather than web/API).

## Files
`gold/train.jsonl`, `gold/val.jsonl`, `gold/test.jsonl` (hashes in `gold/manifest.json`), `gold/gold_metadata.csv` (provenance per row),
`reports/charts/*.png`, `reports/dataset_stats.json`, `reports/pipeline_run.json`.
