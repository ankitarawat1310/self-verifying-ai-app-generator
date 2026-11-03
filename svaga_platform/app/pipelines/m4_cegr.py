"""M4: counterexample-guided repair.

Generate with M1 (retrieval + verification). If the release gate rejects, a repair model gets the verifier failures as an issue,
answers with a unified diff, the diff is applied and the app is verified again, for up to SVAGA_REPAIR_ROUNDS rounds (default 2).
The spec and policy are frozen: a repair can only change app.py and test_app.py, and the gate still requires every check to pass.

The repair model is SVAGA_REPAIR_MODEL, a name from SVAGA_MODEL_ENDPOINTS (a fine-tuned or base model served by
svaga_data/svaga_plugin/serve_svaga_model.py). Without it, M4 returns M1's result unchanged and says so in its metadata.
"""
from __future__ import annotations

import os

from shared.benchmarks.loader import BenchmarkWorkflow
from shared.budget.tracker import BudgetTracker
from shared.llm.provider import get_llm_provider, model_endpoints  # noqa: F401  get_llm_provider: patched by the privacy canary test

from svaga_platform.app.artifacts import PipelineArtifacts
from svaga_platform.app.pipelines.base import Pipeline, PipelineResult
from svaga_platform.app.pipelines.common import finalize_pipeline_result
from svaga_platform.app.pipelines.m1_rag import M1RagPipeline
from svaga_platform.app.repair import finetuned


class M4CegrPipeline(Pipeline):
    name = "m4"

    def run(self, benchmark: BenchmarkWorkflow, *, budget: BudgetTracker, model: str | None = None) -> PipelineResult:
        result = M1RagPipeline().run(benchmark, budget=budget, model=model)
        repair_model = os.getenv("SVAGA_REPAIR_MODEL") or None
        endpoint = model_endpoints().get(repair_model) if repair_model else None
        meta = {"generator": "m1", "repair_model": repair_model, "repair_rounds": 0, "repair_history": []}
        if result.decision == "ACCEPT" or not endpoint or not result.artifacts.app_code.strip():
            if repair_model and not endpoint:
                meta["repair_note"] = f"SVAGA_REPAIR_MODEL '{repair_model}' is not in SVAGA_MODEL_ENDPOINTS"
            result.pipeline = self.name
            result.metadata = {**result.metadata, **meta}
            return result

        arts = result.artifacts
        frozen_spec, frozen_policy = arts.spec_hash, arts.policy.model_dump_json()
        max_rounds = int(os.getenv("SVAGA_REPAIR_ROUNDS", "2"))
        for rnd in range(1, max_rounds + 1):
            issue = finetuned.build_issue(benchmark.natural_language_requirement, result.verification)
            files = {"app.py": finetuned.strip_platform_blocks(arts.app_code), "test_app.py": arts.test_code}
            reply = finetuned.call_repair_model(endpoint, repair_model, finetuned.build_repair_prompt(issue, files))
            budget.record_llm(tokens=reply["tokens"])
            new_files, rep = finetuned.apply_unified_diff(files, finetuned.extract_diff(reply["text"]))
            step = {"round": rnd, "apply": rep.to_dict(), "seconds": reply["seconds"]}
            meta["repair_rounds"] = rnd
            if rep.applied == 0:
                meta["repair_history"].append({**step, "decision": result.decision})
                break
            arts = PipelineArtifacts(app_code=new_files["app.py"], test_code=new_files["test_app.py"],
                                     workflow_spec=arts.workflow_spec, policy=arts.policy, metadata=arts.metadata)
            result = finalize_pipeline_result(self.name, benchmark, arts, budget, frozen_spec_hash=frozen_spec)
            if arts.policy.model_dump_json() != frozen_policy:   # cannot happen through a diff, kept as a guard
                result.decision = "REJECT"
                result.counterexamples.append({"property_id": "policy_freeze", "message": "policy changed during repair"})
            meta["repair_history"].append({**step, "decision": result.decision})
            if result.decision == "ACCEPT":
                break
        result.pipeline = self.name
        result.metadata = {**result.metadata, **meta}
        return result
