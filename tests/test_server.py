"""Server behaviour with a fake backend (no model needed)."""

from fastapi.testclient import TestClient

from jeff.core import Engine, PromptOptions, ScoredText
from jeff.server.app import create_app
from jeff.server.config import Settings
from tests.fixtures import CHOICE_REQ, SCORE_REQ


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.batches = []

    def score(self, texts, groups):
        self.batches.append(len(texts))
        out = []
        for gs in groups:
            scores = {}
            for g in gs:
                n = len(g.labels)
                scores[g.key] = [(i + 1) / n for i in range(n)]  # last label wins
            out.append(ScoredText(scores=scores, input_tokens=42))
        return out


def make_client(**overrides):
    s = Settings(**overrides)
    backend = FakeBackend()
    app = create_app(s, Engine(backend, s.model_name))
    return TestClient(app), backend


def test_roundtrip_and_headers():
    with make_client()[0] as c:
        r = c.post("/v1/systemone", json=SCORE_REQ)
        assert r.status_code == 200, r.text
        assert "x-typesafe-request-id" in r.headers
        body = r.json()
        assert body["model"] == "gliformer-large-v1"
        a = body["answers"]["bug_severity"]
        assert a["type"] == "score" and a["legend"]["2"].startswith("Blocking")
        assert body["usage"]["input_tokens"] == 42


def test_models_endpoint():
    with make_client()[0] as c:
        r = c.get("/v1/models")
        names = [m["name"] for m in r.json()["models"]]
        assert "gliformer-large-v1" in names and "jev-latest" in names


def test_auth():
    with make_client(api_keys=["k1"])[0] as c:
        assert c.post("/v1/systemone", json=SCORE_REQ).status_code == 401
        r = c.post("/v1/systemone", json=SCORE_REQ, headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert c.post("/v1/systemone", json=SCORE_REQ, headers={"Authorization": "Bearer k1"}).status_code == 200


def test_422_shapes():
    with make_client()[0] as c:
        r = c.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": {"q": {"type": "banana"}}})
        assert r.status_code == 422
        d = r.json()["detail"]
        assert isinstance(d, list) and d[0]["loc"][0] == "body"
        r = c.post("/v1/systemone", json={**SCORE_REQ, "selectedModels": ["nope-9000"]})
        assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "model"]


def test_limits():
    with make_client(max_questions=1, max_labels_per_question=2, max_state_chars=10)[0] as c:
        r = c.post("/v1/systemone", json=CHOICE_REQ)
        assert r.status_code == 422
        big = {**SCORE_REQ, "state": "short"}
        r = c.post("/v1/systemone", json=big)
        assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "criteria"


def test_state_limit_uses_configured_format():
    # kv-rendering of this state is 23 chars, json-rendering is 29; the cap must be
    # checked against the format actually sent to the model (state_format), not kv.
    state = {"a": "x" * 20}
    s = Settings(max_state_chars=25)
    app_json = create_app(s, Engine(FakeBackend(), s.model_name, PromptOptions(state_format="json")))
    with TestClient(app_json) as c:
        r = c.post("/v1/systemone", json={**SCORE_REQ, "state": state})
        assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "state"
    app_kv = create_app(s, Engine(FakeBackend(), s.model_name, PromptOptions(state_format="kv")))
    with TestClient(app_kv) as c:
        r = c.post("/v1/systemone", json={**SCORE_REQ, "state": state})
        assert r.status_code == 200, r.text


def test_rate_limit():
    with make_client(rate_limit_rps=1, rate_limit_burst=2)[0] as c:
        codes = [c.post("/v1/systemone", json=SCORE_REQ).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.post("/v1/systemone", json=SCORE_REQ)
        assert "retry-after-ms" in r.headers


def test_dynamic_batching_coalesces():
    client, backend = make_client(max_wait_ms=50, max_batch=8)
    with client as c:
        import threading

        results = []

        def go():
            results.append(c.post("/v1/systemone", json=SCORE_REQ).status_code)

        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert results == [200] * 6
        assert max(backend.batches) > 1, backend.batches
        assert c.get("/stats").json()["requests"] == 6
