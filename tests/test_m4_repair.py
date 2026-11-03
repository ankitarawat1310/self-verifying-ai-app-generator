"""M4 repair with a fine-tuned model: diff application, issue building, and one full case on a saved Demo 1 app."""
import json
import shutil
from pathlib import Path

import pytest

from svaga_platform.app.repair.finetuned import (apply_unified_diff, build_issue, build_repair_prompt, extract_diff,
                                                 strip_platform_blocks)

APP = "from fastapi import FastAPI\n\napp = FastAPI()\n\n\ndef total(a, b):\n    return a - b\n\n\ndef unused():\n    pass\n"


def test_diff_applies_by_content_even_when_line_numbers_are_wrong():
    diff = "--- a/app.py\n+++ b/app.py\n@@ -40,3 +40,3 @@\n def total(a, b):\n-    return a - b\n+    return a + b\n"
    files, rep = apply_unified_diff({"app.py": APP, "test_app.py": ""}, diff)
    assert rep.ok and rep.files_touched == ["app.py"]
    assert "return a + b" in files["app.py"] and "return a - b" not in files["app.py"]


def test_diff_tolerates_indentation_drift_and_reports_missing_context():
    drift = "--- a/app.py\n+++ b/app.py\n@@ -6,2 +6,2 @@\n def total(a, b):\n-  return a - b\n+    return a + b\n"
    files, rep = apply_unified_diff({"app.py": APP}, drift)
    assert rep.ok and "return a + b" in files["app.py"]
    bad = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-this line does not exist\n+x\n"
    files2, rep2 = apply_unified_diff({"app.py": APP}, bad)
    assert not rep2.ok and rep2.failed and files2["app.py"] == APP


def test_diff_for_unknown_file_is_not_applied():
    files, rep = apply_unified_diff({"app.py": APP}, "--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert rep.applied == 0 and files["app.py"] == APP


def test_extract_diff_from_fenced_answer():
    text = "Here is the fix:\n```diff\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n```\nDone."
    assert extract_diff(text).startswith("--- a/app.py") and "Done" not in extract_diff(text)
    assert extract_diff("no diff here") == ""


def test_issue_uses_only_verifier_messages_and_prompt_matches_training_format():
    verification = {"verifiers": [
        {"name": "pytest_sandbox", "passed": False, "details": {"failed_tests": 1, "total_tests": 5,
                                                               "stdout": "FAILED test_app.py::test_self_approval - assert 200 == 403\n"}},
        {"name": "contract_checks", "passed": False, "details": {"missing_paths": ["POST /leave-requests/{}/reject"]}},
        {"name": "api_fuzz", "passed": True, "details": {}}]}
    issue = build_issue("Create a leave-request API.\n\nPublic interface\nSECRET-SHOULD-NOT-APPEAR", verification)
    assert "test_self_approval" in issue and "assert 200 == 403" in issue and "POST /leave-requests/{}/reject" in issue
    assert "Public interface" not in issue and "SECRET-SHOULD-NOT-APPEAR" not in issue
    prompt = build_repair_prompt(issue, {"app.py": APP})
    assert prompt.startswith("Repair the issue using the supplied buggy Python source context.")
    assert "ISSUE:\n" in prompt and "FILE: app.py\nLINES: 1-11\n" in prompt


def test_platform_routes_are_stripped_before_the_model_sees_the_app():
    from shared.generation.app_bootstrap import ensure_fastapi_welcome_routes
    injected = ensure_fastapi_welcome_routes(APP)
    assert "_svaga_root" in injected
    clean = strip_platform_blocks(injected)
    assert "_svaga_root" not in clean and "_svaga_health" not in clean and "def total" in clean


DEMO1 = Path(__file__).resolve().parents[1] / "results" / "grid" / "demo1" / "m1" / "crud_inventory_tracker" / "r2"


@pytest.mark.skipif(not DEMO1.exists(), reason="saved Demo 1 grid not present in this checkout")
def test_full_repair_case_reverifies_and_judges(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_repair_eval import repair_case, summarize
    seen = {}

    def fake_model(endpoint, model, prompt):
        seen["prompt"] = prompt
        return {"text": "```diff\n--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,2 @@\n+# repaired by test\n" +
                        " " + strip_platform_blocks((DEMO1 / "app.py").read_text(encoding="utf-8")).split("\n")[0] + "\n```",
                "tokens": 10, "seconds": 0.1}

    out = tmp_path / "m1" / "crud_inventory_tracker" / "r2"
    rec = repair_case(DEMO1, out, endpoint="http://x", model="fake", rounds=1, call=fake_model)
    judge_words = [c["id"] for c in json.loads((DEMO1 / "judge.json").read_text())["checks"]]
    assert not any(w in seen["prompt"] for w in judge_words)          # no hidden check names leak into the prompt
    assert rec["code_changed"] and rec["diff_applied_fully"] and rec["decision_after"] in ("ACCEPT", "REJECT")
    assert "# repaired by test" in (out / "app.py").read_text() and (out / "judge.json").exists()
    assert rec["hidden_after"].split("/")[1] == "11"
    assert "Broken apps fixed" in summarize(tmp_path)


def test_m4_pipeline_repairs_until_accept_and_keeps_spec_frozen(monkeypatch):
    """Scripted M1 output is rejected; the stand-in repair model returns a diff; M4 re-verifies and records each round."""
    from shared.benchmarks.task_package import find_benchmark
    from shared.budget.tracker import BudgetProfile, BudgetTracker
    from svaga_platform.app.pipelines import m4_cegr
    from svaga_platform.app.pipelines.base import PipelineResult
    from svaga_platform.app.repair import finetuned

    monkeypatch.setenv("SVAGA_SCRIPTED_LLM", "1")
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "fixer=http://127.0.0.1:9")
    monkeypatch.setenv("SVAGA_REPAIR_MODEL", "fixer")
    bench = find_benchmark("crud_contact_directory")
    real_m1 = m4_cegr.M1RagPipeline.run

    def rejected_m1(self, benchmark, *, budget, model=None):
        res = real_m1(self, benchmark, budget=budget, model=model)
        res.decision = "REJECT"
        return res

    calls = []

    def fake_call(endpoint, model, prompt, **kw):
        calls.append(prompt)
        first = prompt.split("FILE: app.py\nLINES: ")[1].split("\n", 1)[1].split("\n")[0]
        return {"text": f"--- a/app.py\n+++ b/app.py\n@@ -1 +1,2 @@\n+# touched by repair\n {first}\n", "tokens": 5, "seconds": 0.1}

    monkeypatch.setattr(m4_cegr.M1RagPipeline, "run", rejected_m1)
    monkeypatch.setattr(finetuned, "call_repair_model", fake_call)
    res: PipelineResult = m4_cegr.M4CegrPipeline().run(bench, budget=BudgetTracker(profile=BudgetProfile()))
    assert res.pipeline == "m4" and res.metadata["repair_model"] == "fixer"
    assert 1 <= res.metadata["repair_rounds"] <= 2 and len(calls) == res.metadata["repair_rounds"]
    assert "# touched by repair" in res.artifacts.app_code
    assert res.metadata["repair_history"][0]["apply"]["ok"]


def test_m4_without_a_repair_model_returns_m1_result(monkeypatch):
    from shared.benchmarks.task_package import find_benchmark
    from shared.budget.tracker import BudgetProfile, BudgetTracker
    from svaga_platform.app.pipelines.m4_cegr import M4CegrPipeline
    monkeypatch.setenv("SVAGA_SCRIPTED_LLM", "1")
    monkeypatch.delenv("SVAGA_REPAIR_MODEL", raising=False)
    res = M4CegrPipeline().run(find_benchmark("crud_contact_directory"), budget=BudgetTracker(profile=BudgetProfile()))
    assert res.pipeline == "m4" and res.metadata["repair_model"] is None and res.metadata["repair_rounds"] == 0
