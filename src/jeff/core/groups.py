"""Turn jev questions into GLiFormer classification groups."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Group
from .schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

NOUL_YES = "yes"
NOUL_NO = "no"
NOUL_MODES = ("single", "single_named", "yes_no")
ISOLATE_MODES = ("none", "nouls", "all")


@dataclass(frozen=True)
class PromptOptions:
    """Prompt rendering and isolation defaults; measurements in bench/RESULTS.md."""

    instruction_as_name: bool = True
    # Append option/level descriptions to the label text ("key: description").
    fold_descriptions: bool = True
    # For score levels given as {what, examples}, append examples to the label.
    fold_examples: bool = False
    sep: str = ": "
    # single: question label; single_named: yes under question; yes_no: yes/no under question.
    # Use single with the base checkpoint.
    noul_mode: str = "yes_no"
    # Separate encoder passes prevent cross-question effects: none, nouls, or all.
    isolate: str = "nouls"
    # Non-string state: key/value lines, JSON, or value-only lines.
    state_format: str = "kv"

    def __post_init__(self):
        if self.noul_mode not in NOUL_MODES:
            raise ValueError(f"noul_mode must be one of {NOUL_MODES}, got {self.noul_mode!r}")
        if self.isolate not in ISOLATE_MODES:
            raise ValueError(f"isolate must be one of {ISOLATE_MODES}, got {self.isolate!r}")

    def isolated(self, q: Question) -> bool:
        return self.isolate == "all" or (self.isolate == "nouls" and isinstance(q, NoulQuestion))


def build_groups(questions: dict[str, Question], opts: PromptOptions | None = None) -> list[Group]:
    opts = opts or PromptOptions()
    return [_group_for(qid, q, opts) for qid, q in questions.items()]


def _group_for(qid: str, q: Question, opts: PromptOptions) -> Group:
    instr = q.instructions_text()
    name = (instr or qid) if opts.instruction_as_name else qid
    if isinstance(q, NoulQuestion):
        if opts.noul_mode == "single":
            label = _fold(instr or qid, q.criterion("true"), opts)
            return Group(key=qid, labels=(label,), name=None)
        if opts.noul_mode == "single_named":
            label = _fold(NOUL_YES, q.criterion("true"), opts)
            return Group(key=qid, labels=(label,), name=name)
        labels = (
            _fold(NOUL_YES, q.criterion("true"), opts),
            _fold(NOUL_NO, q.criterion("false"), opts),
        )
        return Group(key=qid, labels=labels, name=name)
    if isinstance(q, ChoiceQuestion):
        labels = tuple(_fold(k, d, opts) for k, d in q.options())
        return Group(key=qid, labels=_dedupe(labels), name=name)
    if isinstance(q, ScoreQuestion):
        labels = []
        for what, examples in q.levels():
            text = what
            if opts.fold_examples and examples:
                text = f"{what} (e.g. {'; '.join(examples)})"
            labels.append(text)
        return Group(key=qid, labels=_dedupe(tuple(labels)), name=name)
    raise TypeError(type(q))


def _fold(key: str, desc: str | None, opts: PromptOptions) -> str:
    if opts.fold_descriptions and desc:
        return f"{key}{opts.sep}{desc}"
    return key


def _dedupe(labels: tuple[str, ...]) -> tuple[str, ...]:
    """Suffix duplicate labels so GLiFormer does not collapse their scores."""
    if len(set(labels)) == len(labels):
        return labels
    used: set[str] = set()
    out = []
    for label in labels:
        candidate = label
        n = 2
        while candidate in used:
            candidate = f"{label} #{n}"
            n += 1
        used.add(candidate)
        out.append(candidate)
    return tuple(out)
