# Run a fine-tuned model inside SVAGA (Demo 2 / Workbook 2 section 6)

Goal: the same 15 frozen test tasks and the same hidden checks as Demo 1, run with M1 twice: once with a base model, once with
its fine-tuned adapter. This measures whether fine-tuning changes real app outcomes (fully correct apps, false assurance, JSON failures).

Needs: a CUDA GPU (16 GB is enough for two 2B models in 4-bit), Ollama running with `nomic-embed-text` (retrieval embeddings), internet for the
first model download. Use the updated `svaga_repo` (model picker + `--llm-model`) and `torchenv` with transformers 4.x.

## 1. Unzip the results package
```powershell
cd D:\298B\svaga_data
Expand-Archive training_kits\RETURN_ftC_granite33_2b.zip -DestinationPath returned
pip install fastapi uvicorn httpx "transformers>=4.46,<5" peft bitsandbytes accelerate pyyaml lm-format-enforcer
```

## 2. Start two model servers (two PowerShell windows, leave them running)
```powershell
python svaga_plugin\serve_svaga_model.py --run returned\ftC_granite33_2b --port 11500
python svaga_plugin\serve_svaga_model.py --run returned\ftC_granite33_2b --port 11501 --base_only
```
Check: open http://127.0.0.1:11500/svaga/health and http://127.0.0.1:11501/svaga/health (adapter true / false).

## 3. Tell SVAGA about them (same PowerShell window you start the API or run the grid from)
```powershell
cd D:\298B\svaga_repo
$env:PYTHONPATH = "."
$env:SVAGA_LLM_PROVIDER = "ollama"; $env:SVAGA_SCRIPTED_LLM = "0"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"          # real Ollama, for embeddings
$env:SVAGA_MODEL_ENDPOINTS = "ftC_granite33_2b=http://127.0.0.1:11500,ftC_granite33_2b-base=http://127.0.0.1:11501"
```

## 4a. Batch evaluation on the frozen test split (about 1 to 2 hours for both)
```powershell
python scripts\run_grid.py --models m1 --split test --repeats 1 --no-recall --llm-model ftC_granite33_2b-base --name ft_granite_base
python scripts\run_grid.py --models m1 --split test --repeats 1 --no-recall --llm-model ftC_granite33_2b      --name ft_granite_tuned
```
Results appear on the console's Results page (grid dropdown) next to Demo 1. Runs where the small model returned no usable JSON show as
ERROR or REJECT: count them, they are a real finding about small models.

## 4b. Live demo in the console
Start the API in the same window (`python -m uvicorn svaga_platform.app.main:app --host 127.0.0.1 --port 8003`) and the console as usual.
New run now has a **Model** dropdown listing both servers; the run page shows which model produced the app. The **Models** page shows
the fine-tuning comparison, the learned gate and the dataset charts.

Repeat with another model by unzipping its RETURN package and changing the run folder, ports and names.

## M4 repair experiment (the fine-tuned models doing the job they were trained for)

Takes the 28 apps that SVAGA's gate rejected in the frozen Demo 1 M1 run, gives each to the repair model (issue built from SVAGA's own
verifier failures, same prompt format as training), applies its diff, re-verifies (up to 2 rounds) and scores with the hidden judge.
The repair model never sees hidden checks. Same two model servers as above; answers are short diffs, so a case takes about 1 to 3 minutes.
```powershell
python scripts\run_repair_eval.py --llm-model ftC_granite33_2b --name repair_smoke --limit 2
python scripts\run_repair_eval.py --llm-model ftC_granite33_2b-base --name repair_granite_base
python scripts\run_repair_eval.py --llm-model ftC_granite33_2b      --name repair_granite_tuned
```
Resumable; results in `results\repair\<name>\summary.md` and `cases.csv`. Live M4 in the console: set
`$env:SVAGA_REPAIR_MODEL = "ftC_granite33_2b"` before starting the API, then choose pipeline M4 (generation needs the default 30B model).
