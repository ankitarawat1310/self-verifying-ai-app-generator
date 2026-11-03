"""Per-run model selection and the Models page API."""
import pytest
from fastapi.testclient import TestClient

from shared.llm import provider as prov


def test_model_endpoints_parse_both_formats(monkeypatch):
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", '{"ftC_granite33_2b": "http://127.0.0.1:11500"}')
    assert prov.model_endpoints() == {"ftC_granite33_2b": "http://127.0.0.1:11500"}
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "a=http://h:1, b-base=http://h:2")
    assert prov.model_endpoints() == {"a": "http://h:1", "b-base": "http://h:2"}


def test_selected_model_routes_to_its_endpoint_and_resets(monkeypatch):
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "ftB_deepseekcoder_1p3b=http://127.0.0.1:11501")
    monkeypatch.setenv("SVAGA_SCRIPTED_LLM", "1")
    with prov.selected_model("ftB_deepseekcoder_1p3b"):
        llm = prov.get_llm_provider()
        assert isinstance(llm, prov.OllamaLLMProvider)
        assert llm.model == "ftB_deepseekcoder_1p3b" and llm.base_url == "http://127.0.0.1:11501"
    assert isinstance(prov.get_llm_provider(), prov.ScriptedLLMProvider)   # default restored
    with prov.selected_model(None):
        assert isinstance(prov.get_llm_provider(), prov.ScriptedLLMProvider)


def test_unknown_model_is_rejected(monkeypatch):
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "")
    with pytest.raises(KeyError):
        with prov.selected_model("nope"):
            pass


def test_models_api_lists_endpoints_and_serves_report(monkeypatch):
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "ftC_granite33_2b=http://127.0.0.1:11500")
    monkeypatch.setenv("SVAGA_SCRIPTED_LLM", "1")
    from svaga_platform.app.main import app
    c = TestClient(app)
    m = c.get("/api/v1/models").json()
    assert m["selectable"] == [{"name": "ftC_granite33_2b", "url": "http://127.0.0.1:11500"}]
    r = c.get("/api/v1/models/report")
    assert r.status_code == 200 and {"comparison", "gate", "dataset", "charts"} <= set(r.json())
    assert c.get("/api/v1/models/charts/models/..%2F..%2Fsecret.png").status_code in (400, 404)
    assert c.get("/api/v1/models/charts/other/x.png").status_code == 400


def test_run_records_the_selected_model(monkeypatch):
    monkeypatch.setenv("SVAGA_SCRIPTED_LLM", "1")
    monkeypatch.setenv("SVAGA_MODEL_ENDPOINTS", "")
    from svaga_platform.app.main import app
    c = TestClient(app)
    bad = c.post("/api/v1/workflows/run", json={"pipeline": "m0", "natural_language": "x", "workflow_id": "crud_contact_directory", "llm_model": "missing"})
    assert bad.status_code == 404
