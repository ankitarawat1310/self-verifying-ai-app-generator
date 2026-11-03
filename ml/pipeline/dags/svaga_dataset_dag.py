"""Airflow DAG for the SVAGA fine-tuning dataset. Each task calls one function in pipeline/svaga_pipeline.py.

Local run without a scheduler:   airflow dags test svaga_dataset_v1 2026-10-29
Setup (once):  pip install "apache-airflow==2.10.*" duckdb matplotlib
               set AIRFLOW__CORE__DAGS_FOLDER to <svaga_data>/pipeline/dags  and  SVAGA_DATA_ROOT to <svaga_data>
On Windows run Airflow inside WSL or Docker (Airflow does not run natively on Windows).
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(os.getenv("SVAGA_DATA_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "pipeline"))
import svaga_pipeline as p  # noqa: E402


def step(fn):
    def run(**_):
        result = fn(ROOT)
        print(result)
        return result
    return run


with DAG(
    dag_id="svaga_dataset_v1",
    description="bronze -> silver -> gold -> ml_ready for SVAGA fine-tuning (generate, edit, repair)",
    start_date=datetime(2026, 10, 29),
    schedule=None,
    catchup=False,
    tags=["svaga", "data-engineering"],
) as dag:
    ingest = PythonOperator(task_id="bronze_ingest_sources", python_callable=step(p.ingest))
    clean = PythonOperator(task_id="silver_clean_validate_decontaminate_dedupe", python_callable=step(p.clean))
    gold = PythonOperator(task_id="gold_balance_and_split", python_callable=step(p.build_gold))
    checks = PythonOperator(task_id="data_quality_gates", python_callable=step(p.quality_checks))
    export = PythonOperator(task_id="export_ml_ready_jsonl", python_callable=step(p.export))
    report = PythonOperator(task_id="statistics_and_charts", python_callable=step(p.report))
    ingest >> clean >> gold >> checks >> export >> report
