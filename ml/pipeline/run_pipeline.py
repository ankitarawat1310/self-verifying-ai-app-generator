"""Run the whole pipeline without Airflow (same functions the DAG calls). Usage: python pipeline/run_pipeline.py"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import svaga_pipeline as p

log = {}
for step in (p.ingest, p.clean, p.build_gold, p.quality_checks, p.export, p.report):
    t = time.time()
    log[step.__name__] = {"result": step(ROOT), "seconds": round(time.time() - t, 1)}
    print(f"{step.__name__:15} {log[step.__name__]['seconds']:6.1f}s  {json.dumps(log[step.__name__]['result'], default=str)[:300]}")
(ROOT / "reports" / "pipeline_run.json").write_text(json.dumps(log, indent=2, default=str))
