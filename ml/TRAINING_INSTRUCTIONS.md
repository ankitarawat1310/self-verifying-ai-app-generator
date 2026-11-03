# SVAGA fine-tuning: instructions for the four trainers

Each of you trains **one** model on the same SVAGA dataset with the same settings. Then you send back one zip. Nothing else is needed from you.

| Machine | Kit zip | Base model |
|---|---|---|
| 1 | `machine1_ftA_qwen25coder_1p5b.zip` | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| 2 | `machine2_ftB_deepseekcoder_1p3b.zip` | deepseek-ai/deepseek-coder-1.3b-instruct |
| 3 | `machine3_ftC_granite33_2b.zip` | ibm-granite/granite-3.3-2b-instruct |
| 4 | `machine4_ftD_starcoder2_3b.zip` | bigcode/starcoder2-3b |

Each zip holds the data (10,155 train / 569 val / 570 test rows), the scripts and a `config.yaml`. It's about 12 MB.

## Ground rules

1. **Do not change any setting in `config.yaml`** except `revision` (step 2 below). Same settings on all four machines is what makes the comparison fair.
2. **Do not edit anything in `data/`.** Preflight checks the file hashes and stops if they differ.
3. **Keep the kit folder on persistent storage:** Google Drive on Colab, a normal disk locally. Checkpoints are saved there, so a crash only loses the work since the last checkpoint.
4. If a step fails, **don't work around it by changing settings.** Send the error text and `outputs/preflight.json` to the group.

## You need

- An NVIDIA GPU with **16 GB or more** of memory. A Colab T4 is enough; L4 or A100 is faster.
- About **15 GB of free disk** for the model download and checkpoints.
- Time: roughly **2–5 hours** of training on a T4, or under 1 hour on an A100 (an estimate; it hasn't been timed yet). Evaluation adds 30–60 minutes.

---

## Option A: Google Colab

### One-time setup
1. Upload your kit zip to Google Drive, in a folder `MyDrive/SVAGA/`.
2. Open a new notebook at https://colab.research.google.com.
3. **Runtime → Change runtime type → GPU** (T4, or better if you have Colab Pro). Save.

### Cells to run, in order

**Cell 1: connect Drive and unzip** (change the kit name to yours)
```python
from google.colab import drive
drive.mount('/content/drive')
KIT = "machine1_ftA_qwen25coder_1p5b"          # <- your kit name
%cd /content/drive/MyDrive/SVAGA
!unzip -n -q {KIT}.zip
%cd /content/drive/MyDrive/SVAGA/{KIT}
!nvidia-smi
```
`nvidia-smi` must show a GPU. If it says "command not found", the runtime isn't a GPU runtime; go back to step 3.

**Cell 2: install**
```python
!pip install -q -r requirements.txt
```

**Cell 3: preflight** (about 2 minutes)
```python
!python preflight.py --config config.yaml
```
It must end with `PREFLIGHT OK`. Then open your model's page on huggingface.co, click **Files and versions**, copy the latest commit hash, and paste it after `revision:` in `config.yaml`. You can edit the file in Colab's file panel on the left (double-click it).

**Cell 4: 5-step smoke test** (a few minutes)
```python
!python train.py --config config.yaml --max_steps 5
!rm -rf outputs/checkpoints outputs/final_adapter outputs/metrics
```
This proves the whole training loop works before you commit hours to it. The second line removes the smoke-test output.

**Cell 5: full training**
```python
!python train.py --config config.yaml
```
It prints the loss every 10 updates and saves a checkpoint every 100. It ends with `TRAINING DONE`.

**If Colab disconnects:** reconnect and run Cells 1 and 2 again, then Cell 5 again. **Do not run Cell 4.** Training resumes from the latest checkpoint in `outputs/checkpoints/` automatically.

**Cell 6: evaluation**
```python
!python evaluate.py --config config.yaml
```

**Cell 7: package the results**
```python
!python package_results.py --config config.yaml
```
This writes `MyDrive/SVAGA/RETURN_<run_name>.zip`.

Colab tips:
- Keep the browser tab open. Free Colab drops idle sessions, and resume covers you when it does.
- Free T4 sessions can end after a few hours. Expect to resume once, which is normal.

---

## Option B: your own GPU machine

### Linux, or Windows with WSL2 (recommended on Windows)
```bash
nvidia-smi                                   # must show your GPU and a CUDA version
python3 -m venv ~/svaga-ft && source ~/svaga-ft/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build of PyTorch
unzip machine1_ftA_qwen25coder_1p5b.zip && cd machine1_ftA_qwen25coder_1p5b
pip install -r requirements.txt
python preflight.py --config config.yaml     # must end with PREFLIGHT OK; then set revision in config.yaml
python train.py --config config.yaml --max_steps 5 && rm -rf outputs/checkpoints outputs/final_adapter outputs/metrics
python train.py --config config.yaml         # full run; re-run the same command to resume after any interruption
python evaluate.py --config config.yaml
python package_results.py --config config.yaml
```
Use the PyTorch build that matches your driver. Check the selector at https://pytorch.org/get-started/locally/ if `cu121` doesn't fit.

To keep training running after you close the terminal, start it inside `tmux` (`tmux new -s svaga`, run the command, press Ctrl+B then D to leave it running; `tmux attach -t svaga` to come back).

### Windows without WSL (PowerShell)
```powershell
nvidia-smi
py -3.11 -m venv $HOME\svaga-ft; & $HOME\svaga-ft\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
Expand-Archive machine1_ftA_qwen25coder_1p5b.zip -DestinationPath .; cd machine1_ftA_qwen25coder_1p5b
pip install -r requirements.txt
python preflight.py --config config.yaml
python train.py --config config.yaml --max_steps 5; Remove-Item -Recurse -Force outputs\checkpoints, outputs\final_adapter, outputs\metrics -ErrorAction SilentlyContinue
python train.py --config config.yaml
python evaluate.py --config config.yaml
python package_results.py --config config.yaml
```
`bitsandbytes` (4-bit loading) is less reliable on native Windows. If the smoke test fails with a bitsandbytes or CUDA error, switch to WSL2 or Colab rather than changing settings. (`run_all.sh` is for Linux and Colab only.)

---

## What to send back

1. **`RETURN_<run_name>.zip`** (about 50–150 MB). Upload it to the team's shared Drive folder.
2. A short message with:
   - your GPU type and total training time;
   - the last lines printed by `train.py` and `evaluate.py`;
   - anything unusual, such as a resume or a warning.

**Keep `outputs/checkpoints/`** until the team confirms the zip opens and the model loads.

The zip contains the LoRA adapter (`final_adapter/`), the tokenizer, the loss curve (`metrics/train_log.jsonl`), training and evaluation metrics, base-vs-fine-tuned test outputs (`samples/test_generations.jsonl`), the exact package versions and your config. That's everything needed to plug the model into SVAGA and report results.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `PREFLIGHT FAILED: no CUDA GPU` | Colab: change the runtime to GPU. Local: install a CUDA build of PyTorch. |
| `hash_ok_... false` | The data files changed or were damaged. Unzip the kit again. |
| Model download fails, or 401/404 | Check internet access. Run `huggingface-cli login` with a free Hugging Face token. If the page doesn't exist, tell the team; don't swap models yourself. |
| `CUDA out of memory` | Close other GPU programs and restart the runtime, then run again (it resumes). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training. If it still fails, report it with your GPU name. |
| `eval_strategy` is not a valid argument | Your `transformers` is too old: `pip install -U "transformers>=4.46"`. |
| Training stops partway | Run the same `train.py` command again. It continues from the latest checkpoint. |
| `package_results.py` says files are missing | Training didn't finish. Run `train.py` again to finish it. |
| Loss shows `nan` or doesn't go down after ~200 updates | Stop, then send `outputs/metrics/train_log.jsonl` to the team. |
