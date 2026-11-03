# M4 repair with the fine-tuned Qwen3-Coder 30B, on your own machine (Ollama)

Same experiment as the Granite and Colab runs: the 28 apps M1 rejected in Demo 1, the model answers with a diff, SVAGA re-verifies, the hidden judge scores. It uses the `qwen3-coder:30b` you already have in Ollama, so there's no download and no bitsandbytes.

## What's in this folder
- `svaga-qwen30b-lora.gguf`: the team's fine-tuned 30B adapter (attention layers only), converted to Ollama's format.
- `Modelfile.base`: the base Ollama model on top of `qwen3-coder:30b`, with an 8,192-token context. Repair prompts are longer than Ollama's default context and would otherwise be cut off silently.
- `install_ft_model.py`: registers `svaga-qwen30b-ft` as that same model plus the adapter. Current Ollama rejects a Modelfile `ADAPTER` line, so this writes the adapter into the model store directly. The server still applies it.
- `check_adapter.py`: proves the adapter is active and measures speed.
- `run_m4_30b_local.ps1`: runs the experiment. `run_m4_30b_local.cmd` launches it when PowerShell's execution policy blocks `.ps1` files.

## Steps (PowerShell, in this folder)
```powershell
cd D:\298B\svaga_data\m4_30b_local
ollama list                                   # qwen3-coder:30b must be listed
ollama create svaga-qwen30b-base -f Modelfile.base
python install_ft_model.py                    # registers svaga-qwen30b-ft
python check_adapter.py                       # must end with ADAPTER CHECK OK
.\run_m4_30b_local.cmd -Smoke                 # 2 apps
.\run_m4_30b_local.cmd                        # all 28 apps, fine-tuned then base
```
`run_m4_30b_local.cmd` starts the `.ps1` with execution policy bypassed for that process. Use the `.ps1` directly only after `Set-ExecutionPolicy -Scope Process Bypass`.
The run uses `D:\298B\svaga_repo\.venv` if it exists (otherwise `python`). Pass another with `-Python <path>`.

Results: `D:\298B\svaga_repo\results\repair\repair_qwen30b_local_tuned\` and `..._base\`. Re-running skips finished cases.

## If something fails
- `ollama create` on an old Modelfile errors with "LoRA adapters are no longer supported": that is expected. Register the fine-tune with `python install_ft_model.py` instead. If `check_adapter.py` then says the answers are identical, or the fine-tuned call fails while loading, this Ollama build is not applying the adapter. Use the Colab kit (`svaga_m4_30b_colab.zip`) instead.
- Calls time out (900 seconds each): the model is spilling too much into CPU memory. Close other GPU programs and try again.

## Note for the write-up
Ollama runs the base at its own 4-bit quantization (Q4_K_M), while the adapter was trained on a bitsandbytes 4-bit base. Small numeric differences are expected. Base and fine-tuned use the same quantized base, so the comparison between them is fair.
