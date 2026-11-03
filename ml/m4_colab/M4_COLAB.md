# M4 repair experiment on Colab

What it does: takes the 28 apps that SVAGA's M1 pipeline rejected in the Demo 1 grid, asks Granite (base, then fine-tuned) to fix each one with a diff, re-runs SVAGA's verification and release gate, and scores the result with the hidden judge. The model only sees the verifier's failure messages and the code. It never sees hidden checks.

This kit holds SVAGA's private benchmarks and hidden checks. Keep it in your own Drive. Don't share the zip.

## Setup
1. Upload `svaga_m4_colab.zip` to `MyDrive/SVAGA/`.
2. New Colab notebook, then **Runtime > Change runtime type > GPU**. A100 or L4 is best; T4 works but is slower.

## Cells

**Cell 1: unzip to the fast local disk**
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/SVAGA/m4_results
!unzip -n -q /content/drive/MyDrive/SVAGA/svaga_m4_colab.zip -d /content
%cd /content/svaga_m4_colab
!nvidia-smi
```

**Cell 2: install (about 2 minutes)**
```python
!pip install -q -r requirements_m4.txt
```

**Cell 3: smoke test, 2 apps (about 10 minutes, mostly the model download)**
```python
!SVAGA_M4_RESULTS=/content/drive/MyDrive/SVAGA/m4_results bash run_m4.sh smoke
```
It should print a summary table. If it says the server did not start, send the lines it printed.

**Cell 4: full run, base and fine-tuned in parallel (roughly 1 to 2 hours on an A100)**
```python
!SVAGA_M4_RESULTS=/content/drive/MyDrive/SVAGA/m4_results bash run_m4.sh full
```
Results are written to Drive as they finish. If Colab disconnects, run Cells 1 and 2 again, then Cell 4. It skips finished cases.
To check progress from another cell: `!tail -n 5 /content/drive/MyDrive/SVAGA/m4_results/eval_*.log`

**Cell 5: zip the results**
```python
!cd /content/drive/MyDrive/SVAGA && zip -q -r m4_results.zip m4_results && ls -lh m4_results.zip
```
Download `m4_results.zip` and put it in `D:\298B\svaga_data\`.
