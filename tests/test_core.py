import math

import pytest
from pydantic import ValidationError

from jeff.core import (
    Engine,
    Group,
    PromptOptions,
    ScoredText,
    SystemOneRequest,
    build_groups,
)
from jeff.core.answers import confidence, decode, normalize
from jeff.core.schemas import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion
from jeff.core.state import serialize_state
from tests.fixtures import CHOICE_REQ, SCORE_REQ


class FakeBackend:
    name = "fake"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, texts, groups):
        self.calls.append((texts, groups))
        return [ScoredText(scores={g.key: self.scores[g.key] for g in gs}, input_tokens=100) for gs in groups]


def test_state_serialization():
    assert serialize_state("x") == "x"
    assert serialize_state({"a": 1, "b": "two", "c": {"d": [1]}}) == 'a: 1\nb: two\nc: {"d":[1]}'
    assert serialize_state(["a", "b"]) == "a\nb"


def test_request_validation():
    r = SystemOneRequest.model_validate(SCORE_REQ)
    assert r.model == "jev-latest"
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}}
        )
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "?", "criteria": []}}}
        )
    # instructions are optional on the wire; the question id stands in
    r = SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {"spam": {"type": "noul"}}})
    (g,) = build_groups(r.questions)
    assert g.name == "spam" and g.labels == ("yes", "no")
    (g,) = build_groups(r.questions, PromptOptions(noul_mode="single"))
    assert g.labels == ("spam",)
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "x", "model": "m", "questions": {"q": {"type": "banana", "instructions": "?"}}}
        )


def test_build_groups_default_prompt():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions)
    by_key = {g.key: g for g in gs}
    assert by_key["team"].name == "Which team should handle this?"
    assert by_key["team"].labels == (
        "billing: Payment or subscription issues",
        "technical: Bugs or integration problems",
        "sales: Pricing or account questions",
    )
    # default noul rendering is yes/no under the question (see PromptOptions)
    assert by_key["refund"].labels == ("yes", "no")
    assert by_key["refund"].name == "Does the customer request a refund?"
    assert by_key["urgency"].labels == ("yes: time-sensitive", "no: no urgency")
    assert PromptOptions().isolate == "nouls"


def test_build_groups_single_noul():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions, PromptOptions(noul_mode="single"))
    by_key = {g.key: g for g in gs}
    assert by_key["refund"].labels == ("Does the customer request a refund?",)
    assert by_key["refund"].name is None
    assert by_key["urgency"].labels == ("Does this convey urgency?: time-sensitive",)


def test_build_groups_no_fold():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = build_groups(r.questions, PromptOptions(fold_descriptions=False, instruction_as_name=False))
    team = next(g for g in gs if g.key == "team")
    assert team.name == "team"
    assert team.labels == ("billing", "technical", "sales")


def test_duplicate_levels_stay_aligned():
    r = SystemOneRequest.model_validate(
        {
            "state": "x",
            "model": "m",
            "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["low", "low", "high"]}},
        }
    )
    (g,) = build_groups(r.questions)
    assert g.labels == ("low", "low #2", "high")


def test_dedupe_avoids_colliding_with_existing_label():
    # "a" repeats and would naively dedupe to "a #2", but "a #2" is already
    # present in the input, so the naive suffix must not collide with it.
    r = SystemOneRequest.model_validate(
        {
            "state": "x",
            "model": "m",
            "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["a #2", "a", "a"]}},
        }
    )
    (g,) = build_groups(r.questions)
    assert len(set(g.labels)) == len(g.labels)


def test_group_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        Group(key="k", labels=("a", "a"))


def test_normalize_and_confidence():
    assert normalize([0.0, 0.0]) == [0.5, 0.5]
    p = normalize([0.2, 0.6, 0.2])
    assert math.isclose(sum(p), 1.0)
    assert confidence([1 / 3] * 3) == 0.0
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.0, 0.7, 0.3]) == 0.55


def test_score_answer_matches_docs_formula():
    r = SystemOneRequest.model_validate(SCORE_REQ)
    eng = Engine(FakeBackend({"bug_severity": [0.0, 0.7, 0.3]}), "jev-latest", temperature=1.0)  # raw formula
    resp = eng.run(r).model_dump()
    a = resp["answers"]["bug_severity"]
    assert a["type"] == "score"
    assert a["score"] == 1.3
    assert a["probabilities"] == {"0": 0.0, "1": 0.7, "2": 0.3}
    assert a["legend"] == {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists",
    }
    assert 0 <= a["confidence"] <= 1
    assert resp["usage"] == {"input_tokens": 100, "output_tokens": 6}
    assert resp["model"] == "jev-latest"


def test_choice_and_noul_answers():
    r = SystemOneRequest.model_validate(CHOICE_REQ)
    eng = Engine(
        FakeBackend({"team": [0.9, 0.05, 0.05], "refund": [0.8], "urgency": [0.1]}), "jev-latest", temperature=1.0
    )
    a = eng.run(r).model_dump()["answers"]
    assert a["team"]["choice"] == "billing"
    assert a["team"]["probabilities"]["billing"] == 0.9
    assert set(a["team"]) == {"type", "choice", "confidence", "probabilities"}
    assert a["refund"] == {"type": "noul", "noul": 0.8}
    assert a["urgency"]["noul"] == 0.1


def test_score_levels_with_examples_legend():
    r = SystemOneRequest.model_validate(
        {
            "state": "x",
            "model": "m",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "?",
                    "criteria": [{"what": "calm", "examples": ["ok thanks"]}, {"what": "angry"}],
                }
            },
        }
    )
    eng = Engine(FakeBackend({"q": [0.5, 0.5]}), "m")
    a = eng.run(r).model_dump()["answers"]["q"]
    assert a["legend"] == {"0": {"what": "calm", "examples": ["ok thanks"]}, "1": {"what": "angry"}}
    assert a["score"] == 0.5


def test_state_formats():
    obj = {"subject": "Hi", "message": "Refund me", "n": 2}
    assert serialize_state(obj, "kv") == "subject: Hi\nmessage: Refund me\nn: 2"
    assert serialize_state(obj, "values") == "Hi\nRefund me\n2"
    assert serialize_state(obj, "json") == '{"subject": "Hi", "message": "Refund me", "n": 2}'
    assert serialize_state("plain", "json") == "plain"
    with pytest.raises(ValueError):
        serialize_state(obj, "yaml")


def test_build_groups_single_named_noul():
    req = SystemOneRequest.model_validate(CHOICE_REQ)
    gs = {g.key: g for g in build_groups(req.questions, PromptOptions(noul_mode="single_named"))}
    assert gs["refund"].name == "Does the customer request a refund?"
    assert gs["refund"].labels == ("yes",)
    assert gs["urgency"].labels == ("yes: time-sensitive",)
    with pytest.raises(ValueError):
        PromptOptions(noul_mode="maybe")
    with pytest.raises(ValueError):
        PromptOptions(isolate="some")


def test_engine_isolation_splits_prompts():
    class Recorder:
        name = "rec"

        def __init__(self):
            self.calls = []

        def score(self, texts, groups):
            self.calls.append([[g.key for g in gs] for gs in groups])
            return [ScoredText(scores={g.key: [0.25] * len(g.labels) for g in gs}, input_tokens=10) for gs in groups]

    req = SystemOneRequest.model_validate(CHOICE_REQ)
    for isolate, units, tokens in (
        ("none", [["team", "refund", "urgency"]], 10),
        ("nouls", [["refund"], ["urgency"], ["team"]], 30),
        ("all", [["team"], ["refund"], ["urgency"]], 30),
    ):
        b = Recorder()
        resp = Engine(b, "m", PromptOptions(isolate=isolate)).run(req)
        assert b.calls == [units], isolate
        assert resp.usage.input_tokens == tokens
        assert list(resp.answers) == ["team", "refund", "urgency"]
    # batches of several requests map each unit back to the right request
    b = Recorder()
    out = Engine(b, "m", PromptOptions(isolate="nouls")).run_batch([req, SystemOneRequest.model_validate(SCORE_REQ)])
    assert len(b.calls[0]) == 4 and out[1].usage.input_tokens == 10 and list(out[1].answers) == ["bug_severity"]


def test_temperature_flattens_but_keeps_argmax():
    raw = [0.9, 0.3, 0.1]
    p1, p3 = normalize(raw), normalize(raw, 3.0)
    assert abs(sum(p3) - 1) < 1e-9
    assert max(p3) < max(p1) and p3.index(max(p3)) == 0
    assert normalize(raw, 1.0) == p1

    q = ChoiceQuestion(type="choice", criteria={"a": None, "b": None, "c": None})
    hot, flat = decode(q, raw, temperature=1.0), decode(q, raw, temperature=3.0)
    assert isinstance(hot, ChoiceAnswer) and isinstance(flat, ChoiceAnswer)
    assert hot.choice == flat.choice == "a" and flat.confidence < hot.confidence

    s = ScoreQuestion(type="score", criteria=["lo", "mid", "hi"])
    hot, flat = decode(s, [0.05, 0.1, 0.9], temperature=1.0), decode(s, [0.05, 0.1, 0.9], temperature=3.0)
    assert isinstance(hot, ScoreAnswer) and isinstance(flat, ScoreAnswer)
    assert flat.score == hot.score  # score always comes from the raw distribution
    assert flat.probabilities["2"] < hot.probabilities["2"] and flat.confidence < hot.confidence

    n = NoulQuestion(type="noul", instructions="?")

    def noul(scores, t):
        a = decode(n, scores, temperature=t)
        assert isinstance(a, NoulAnswer)
        return a.noul

    assert 0.5 < noul([0.9, 0.1], 3.0) < noul([0.9, 0.1], 1.0)
    assert 0.5 < noul([0.9], 3.0) < 0.9  # single-label mode
