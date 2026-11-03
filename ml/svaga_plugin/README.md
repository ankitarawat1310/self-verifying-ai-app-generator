# Plug a fine-tuned model back into SVAGA

1. Unzip a returned `RETURN_<run>.zip` (for example into `svaga_data/returned/`).
2. Start the model server (needs a CUDA GPU; on a laptop without one, run it on Colab and expose the port, or use the team's GPU machine):
   ```
   pip install fastapi uvicorn httpx "transformers>=4.46,<5" peft bitsandbytes accelerate pyyaml lm-format-enforcer
   python serve_svaga_model.py --run ..\returned\ftA_qwen25coder_1p5b --port 11500
   ```
3. Start SVAGA pointed at it. Nothing in SVAGA changes; only these variables:
   ```powershell
   $env:SVAGA_LLM_PROVIDER = "ollama"; $env:SVAGA_SCRIPTED_LLM = "0"
   $env:OLLAMA_BASE_URL = "http://127.0.0.1:11500"   # the SVAGA model server
   $env:OLLAMA_MODEL = "ftA_qwen25coder_1p5b"
   python -m uvicorn svaga_platform.app.main:app --host 127.0.0.1 --port 8003
   ```
   Retrieval embeddings still reach your real Ollama: the server forwards every non-chat call to http://127.0.0.1:11434.
4. Compare inside SVAGA (same 20-task benchmark and metrics as Demo 1):
   ```
   python scripts\run_grid.py --models m1 --split test --repeats 1 --name ftA_tuned
   ```
   Then restart the server with `--base_only` and run again with `--name ftA_base`. The Results page shows both grids.

Notes
- Names start with `ft` (ftA..ftD) so they are never confused with SVAGA's pipeline names M0 to M4.
- Small models may not always return valid JSON for SVAGA's app bundle; the server extracts the first JSON object it finds.
  Count JSON failures as errors in the results, do not hide them.
