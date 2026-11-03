"""Copy the reports the console's Models page reads into the SVAGA repo (docs/demo2). Small files only.
Usage: python analysis/publish_reports.py --repo D:\\298B\\svaga_repo"""
import argparse, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); a = ap.parse_args()
dst = Path(a.repo) / "docs" / "demo2"
plan = {
    "models": [ROOT / "reports/models/comparison.csv", ROOT / "reports/models/comparison.md", ROOT / "reports/models/per_task_loss.csv",
               ROOT / "reports/models/generation_metrics.csv", ROOT / "reports/models/examples.md",
               ROOT / "reports/models/public_benchmarks.csv", ROOT / "reports/models/public_benchmarks.md", ROOT / "reports/models/in_svaga_results.md", ROOT / "reports/models/m4_repair.csv", ROOT / "reports/models/m4_repair.md", *sorted((ROOT / "reports/models/charts").glob("*.png"))],
    "gate": [ROOT / "reports/gate/metrics.json", ROOT / "reports/gate/gate_report.md", ROOT / "reports/gate/gate_runs.csv", *sorted((ROOT / "reports/gate/charts").glob("*.png"))],
    "dataset": [ROOT / "reports/dataset_stats.json", ROOT / "gold/manifest.json", ROOT / "DATASET_CARD.md", *sorted((ROOT / "reports/charts").glob("*.png"))],
}
for section, files in plan.items():
    for f in files:
        if not f.exists():
            print("missing", f); continue
        sub = "charts" if f.suffix == ".png" else ""
        out = dst / section / sub / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
        print(f"{section}/{sub + '/' if sub else ''}{f.name}")
