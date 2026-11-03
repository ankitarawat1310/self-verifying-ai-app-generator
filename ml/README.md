# ml/: SVAGA 3.0 training data, fine-tuning and evaluation (Demo 2)

Code and reports only. The data files (gold JSONL, warehouse, raw sources) and model files (adapters, RETURN zips, GGUF)
are on the team Drive; see Workbook 2, Appendix B. `gold/manifest.json` holds the row counts and SHA-256 of the gold files,
so anyone can check that their copy is the one the models were trained on.

| Folder | What it is | Workbook 2 |
|---|---|---|
| `pipeline/` | Dataset pipeline: bronze, silver, gold (DuckDB), Airflow DAG | 4.4.1, 5.2 |
| `DATASET_CARD.md`, `gold/manifest.json`, `reports/charts/` | What is in the data, file hashes, funnel and balance charts | 4.4.1, 6.4 |
| `training_kits/` | One QLoRA kit per model (`_common/` scripts + per-machine `config.yaml`), `TRAINING_INSTRUCTIONS.md` | 4.1, 4.2 |
| `analysis/` | `compare_models.py`, `gate_classifier.py`, `public_benchmarks.py`, `publish_reports.py` | 4.5 |
| `reports/models/`, `reports/gate/` | Loss, output quality, benchmarks, M4 and learned-gate results with charts | 4.5, 6.1 |
| `benchmarks/` | Public benchmark runner (HumanEval+, MBPP+, HumanEvalFix) and `HANDOFF.md`; result bundles in `all_results_podA.zip`, `results_30b.zip` | 4.5.3 |
| `svaga_plugin/` | Model servers with an Ollama-compatible `/api/chat` (`serve_svaga_model.py`, `serve_30b.py`) | 5.2, 5.3 |
| `m4_colab/`, `m4_30b_local/` | How the M4 repair runs were executed (Colab Granite, local Ollama 30B) | 4.5.5 |

Re-run: `python pipeline/run_pipeline.py` (needs the bronze files from the Drive), then the analysis scripts.
Publish the reports to the console: `python analysis/publish_reports.py --repo ..`
