# Machine 3: fine-tune `ibm-granite/granite-3.3-2b-instruct`

Run name: `ftC_granite33_2b`. Model note: instruct; Apache-2.0.
Data: SVAGA final dataset (10,155 train / 569 val / 570 test rows; generate, edit and repair balanced). Do not edit `data/`;
`preflight.py` checks its SHA-256 hashes against `data/manifest.json`.

## What you need
- One NVIDIA GPU with 16 GB or more (Colab T4 is enough; A100/L4 is faster). Python 3.10+.
- About 15 GB free disk (model download plus checkpoints) and a Hugging Face login if the model asks for one (`huggingface-cli login`).

## Steps (Colab: upload this folder to Google Drive, open a GPU notebook, `%cd` into the folder, then run each line with `!`)
1. `pip install -r requirements.txt`
2. `python preflight.py --config config.yaml` (about 2 min). It must end with `PREFLIGHT OK`.
   Then open the model page on Hugging Face, copy the current commit hash into `revision:` in `config.yaml`, so the run is reproducible.
3. Quick smoke test: `python train.py --config config.yaml --max_steps 5` . If it finishes, delete `outputs/checkpoints` and continue.
4. `python train.py --config config.yaml`  (one epoch; rough time 2 to 5 h on a T4, under 1 h on an A100).
   If the session dies, run the same command again: it resumes from the latest checkpoint in `outputs/checkpoints/`.
   On Colab keep this folder on Google Drive so checkpoints survive a disconnect.
5. `python evaluate.py --config config.yaml`  (base vs fine-tuned loss on val and test, plus 30 test generations per task type).
6. `python package_results.py --config config.yaml`  creates `../RETURN_ftC_granite33_2b.zip`. Send that zip back.
   Keep `outputs/checkpoints/` until the team confirms the zip works.

`run_all.sh` runs steps 1, 2, 4, 5 and 6 in order.

## What comes back (inside RETURN_ftC_granite33_2b.zip)
- `outputs/final_adapter/`  LoRA weights (`adapter_model.safetensors`, `adapter_config.json`): this is the fine-tuned model, used with the same base model
- `outputs/tokenizer/`
- `outputs/metrics/`  `train_log.jsonl` (loss curve), `train_summary.json`, `eval_loss.json`, `eval_generation.json`
- `outputs/samples/test_generations.jsonl`  base and fine-tuned outputs side by side
- `outputs/preflight.json`, `outputs/environment.txt` (exact package versions), `config.yaml`, data manifest

## Rules
- Do not change any setting except `revision`. All four machines must use the same settings, or the comparison is not fair.
- Do not train on `val.jsonl` or `test.jsonl`.
- Record GPU type and wall time in your message back (they are also in `train_summary.json`).
