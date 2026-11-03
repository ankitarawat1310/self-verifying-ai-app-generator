# M4 repair experiment with the fine-tuned Qwen3-Coder 30B (Colab)

Same experiment as the Granite run: the 28 apps M1 rejected in the Demo 1 grid, a diff from the model, SVAGA re-verifies, the hidden judge scores. Here the repairer is Qwen3-Coder-30B-A3B, base and with the team's fine-tuned adapter (trained on the same 10K dataset).

This kit holds SVAGA's private benchmarks and hidden checks. Keep it in your own Drive. Don't share the zip.

## You need
- An **A100** runtime (40 GB is enough: the model loads in 4-bit, about 18 GB). L4 may fit but is slow. T4 won't work.
- About 70 GB of free runtime disk for the one-time model download (an A100 runtime has plenty).

## Cells

**Cell 1: unzip to the fast local disk**
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/SVAGA/m4_results_30b
!unzip -n -q /content/drive/MyDrive/SVAGA/svaga_m4_30b_colab.zip -d /content
%cd /content/svaga_m4_30b_colab
!nvidia-smi
```

**Cell 2: install (about 2 minutes)**
```python
!pip install -q -r requirements_m4.txt
```

**Cell 3: smoke test, 2 apps.** The first start downloads the model (about 61 GB, often 10 to 20 minutes), then loads it (a few minutes).
```python
!SVAGA_M4_RESULTS=/content/drive/MyDrive/SVAGA/m4_results_30b bash run_m4_30b.sh smoke
```
Check that `server_30b.log` shows `adapter attached: 384 LoRA tensors`. If the server stops, send the last lines it printed.

**Cell 4: full run (fine-tuned, then base)**
```python
!SVAGA_M4_RESULTS=/content/drive/MyDrive/SVAGA/m4_results_30b bash run_m4_30b.sh full
```
The server stays loaded between cells, so this starts right away. Results go to Drive case by case. If Colab disconnects, run Cells 1 and 2 again, then Cell 4. It skips finished cases (the model download happens again).
Progress from another cell: `!tail -n 3 /content/drive/MyDrive/SVAGA/m4_results_30b/eval_*.log`

If time runs short, the fine-tuned results finish first, then you can stop. Base Granite and the Demo 1 numbers give a partial comparison, but the base 30B run is the fair one.

On an 80 GB GPU (A100 80 GB or H100), bf16 is faster: run `SVAGA_30B_ARGS=--no_quant` before the bash command in Cells 3 and 4.

**Cell 5: zip the results**
```python
!cd /content/drive/MyDrive/SVAGA && zip -q -r m4_results_30b.zip m4_results_30b && ls -lh m4_results_30b.zip
```
Put `m4_results_30b.zip` in `D:\298B\svaga_data\`.
