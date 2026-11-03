"""Read-only API for the Models page: selectable models, the fine-tuned model comparison, the learned gate and the dataset.

Data comes from docs/demo2/ (copied from svaga_data/reports by svaga_data/analysis/publish_reports.py), so the page works
from a plain clone without any GPU or model files."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from shared.llm.provider import describe_llm_provider, model_endpoints
from shared.paths import SVAGA_ROOT

router = APIRouter(prefix="/api/v1/models", tags=["models"])
DEMO2 = SVAGA_ROOT / "docs" / "demo2"
SECTIONS = {"models", "gate", "dataset"}
SAFE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@router.get("")
def list_models():
    """Models the New run page can choose from. The default is whatever SVAGA_LLM_PROVIDER configures."""
    try:
        default = describe_llm_provider()
    except Exception as exc:  # provider misconfigured; still list endpoints
        default = {"provider": "unavailable", "model": str(exc)}
    return {"default": default, "selectable": [{"name": n, "url": u} for n, u in sorted(model_endpoints().items())]}


@router.get("/report")
def report():
    return {
        "comparison": _csv(DEMO2 / "models" / "comparison.csv"),
        "per_task_loss": _csv(DEMO2 / "models" / "per_task_loss.csv"),
        "generation": _csv(DEMO2 / "models" / "generation_metrics.csv"),
        "public_benchmarks": _csv(DEMO2 / "models" / "public_benchmarks.csv"),
        "m4_repair": _csv(DEMO2 / "models" / "m4_repair.csv"),
        "gate": _json(DEMO2 / "gate" / "metrics.json"),
        "dataset": _json(DEMO2 / "dataset" / "dataset_stats.json"),
        "dataset_manifest": _json(DEMO2 / "dataset" / "manifest.json"),
        "charts": {s: sorted(p.name for p in (DEMO2 / s / "charts").glob("*.png")) for s in SECTIONS if (DEMO2 / s / "charts").exists()},
    }


@router.get("/charts/{section}/{file}")
def chart(section: str, file: str):
    if section not in SECTIONS or not SAFE.match(file):
        raise HTTPException(400, "bad chart name")
    path = DEMO2 / section / "charts" / file
    if not path.exists():
        raise HTTPException(404, "chart not found")
    return FileResponse(path, media_type="image/png")
