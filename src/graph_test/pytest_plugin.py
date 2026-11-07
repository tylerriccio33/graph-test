"""pytest plugin: narrow collection to graph-test's selection, down to parametrized cases.

Inert unless ``GRAPH_TEST_SELECTION`` points at a selection file written by
``graph-test run`` / ``graph-test rank --format cmd``. Registered via the ``pytest11``
entry point, so it's active wherever graph-test is installed.

Selection file::

    {"root": "/abs/repo", "full": ["tests/a.py::test_x", "tests/b.py"],
     "cases": {"tests/c.py::test_corpus": ["tests/corpus/x.sas"]},
     "budget": 200, "order": ["tests/c.py::test_corpus", "tests/a.py::test_x", "tests/b.py"]}

``full`` entries keep every item; ``cases`` entries keep only the parametrized items whose
parameters (or ids) refer to one of the listed changed files. If none match, all of that
test's items are kept, so ambiguity never skips anything.

``budget`` and ``order`` come from ``-n`` and are absent otherwise. After narrowing and any
other deselection (``-k``, ``-m``), at most ``budget`` items stay: one at a time per entry of
``order`` (the rank order), round-robin, so a heavily parametrized test can't crowd out the rest.
Within an entry the kept items are spread evenly over the sorted node ids. The result depends
only on the selection file and the set of collected node ids, so every xdist worker keeps the
same items whatever their order. Items dropped this way are reported in the terminal summary.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_test.config import SELECTION_ENV as ENV

_REPORT_KEY = "graph_test_budget_report"


def _key(item: pytest.Item, root: Path) -> str:
    try:
        file = Path(item.path).resolve().relative_to(root).as_posix()
    except ValueError:
        file = item.nodeid.split("::", 1)[0]
    rest = item.nodeid.split("::", 1)[1] if "::" in item.nodeid else ""
    rest = rest.split("[", 1)[0]
    return f"{file}::{rest}" if rest else file


def _flatten(value: Any, depth: int = 0) -> Iterator[Any]:
    if depth > 3:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _flatten(v, depth + 1)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # A case object (e.g. Case(path=...)): its fields name the files it uses.
        for f in dataclasses.fields(value):
            yield from _flatten(getattr(value, f.name, None), depth + 1)
    else:
        yield value


def _matches(item: pytest.Item, changed: list[Path], root: Path) -> int:
    """2 if a parameter resolves to a changed path, 1 if only a bare name/id matches, else 0."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return 0
    names = {c.name for c in changed} | {c.stem for c in changed}
    cid = str(callspec.id)
    weak = cid in names or any(part in names for part in cid.split("-"))
    for value in _flatten(callspec.params):
        if not isinstance(value, (str, os.PathLike)):
            continue
        text = os.fspath(value)
        if text in names:
            weak = True
            continue
        p = Path(text)
        bases = [Path()] if p.is_absolute() else [root, Path.cwd(), Path(item.path).parent]
        for base in bases:
            try:
                cand = (base / p).resolve()
            except (OSError, ValueError):
                continue
            if cand == root:
                continue
            if any(c == cand or c.is_relative_to(cand) for c in changed):
                return 2
    return 1 if weak else 0


def _load_selection() -> dict[str, Any] | None:
    path = os.environ.get(ENV)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    sel = _load_selection()
    if sel is None:
        return
    root = Path(sel["root"]).resolve()
    full = set(sel.get("full", ()))
    cases: dict[str, list[str]] = sel.get("cases", {})

    keep: set[int] = set()
    groups: dict[str, list[pytest.Item]] = {}
    for item in items:
        key = _key(item, root)
        if key in full or key.split("::", 1)[0] in full:
            keep.add(id(item))
        elif key in cases:
            groups.setdefault(key, []).append(item)
    for key, group in groups.items():
        changed = [(root / f).resolve() for f in cases[key]]
        # A bare name like "expected" can appear in unrelated ids, so a real path match wins.
        scores = {id(it): _matches(it, changed, root) for it in group}
        best = max(scores.values())
        keep.update(id(it) for it in group if not best or scores[id(it)] == best)

    dropped = [it for it in items if id(it) not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [it for it in items if id(it) in keep]


def _quotas(sizes: list[int], budget: int) -> list[int]:
    """Deal ``budget`` out one at a time, in order, skipping groups that are already full."""
    quotas = [0] * len(sizes)
    remaining = min(budget, sum(sizes))
    while remaining > 0:
        for i, size in enumerate(sizes):
            if remaining > 0 and quotas[i] < size:
                quotas[i] += 1
                remaining -= 1
    return quotas


def _spread(group: list[pytest.Item], k: int) -> list[pytest.Item]:
    """``k`` evenly spaced items by node id: neighbours in a sorted corpus tend to be alike.

    Sorted here, not in collection order, because pytest-randomly may already have shuffled.
    """
    ordered = sorted(group, key=lambda it: it.nodeid)
    return [ordered[(2 * i + 1) * len(ordered) // (2 * k)] for i in range(k)]


class _Budget:
    """Applies ``-n``. Registered only when the selection has a budget, and runs last so the
    budget is spent on items that would really run, not on ones ``-k``/``-m`` then remove."""

    def __init__(self, sel: dict[str, Any]) -> None:
        self.root = Path(sel["root"]).resolve()
        self.budget: int = sel["budget"]
        self.order: list[str] = sel["order"]
        self.report: str | None = None

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        if len(items) <= self.budget:
            return
        rank = {nodeid: i for i, nodeid in enumerate(self.order)}
        groups: list[list[pytest.Item]] = [[] for _ in range(len(self.order) + 1)]
        for item in items:
            key = _key(item, self.root)
            slots = [rank[k] for k in (key, key.split("::", 1)[0]) if k in rank]
            groups[min(slots, default=len(self.order))].append(item)

        quotas = _quotas([len(g) for g in groups], self.budget)
        keep = {id(it) for g, k in zip(groups, quotas, strict=True) if k for it in _spread(g, k)}
        tests = sum(1 for g in groups if g)
        kept_tests = sum(1 for k in quotas if k)
        self.report = (
            f"graph-test: -n budget of {self.budget} kept {len(keep)} of {len(items)} cases "
            f"from {kept_tests} of {tests} tests; {len(items) - len(keep)} did not run"
        )
        output = getattr(config, "workeroutput", None)  # set on an xdist worker
        if output is not None:
            output[_REPORT_KEY] = self.report
        config.hook.pytest_deselected(items=[it for it in items if id(it) not in keep])
        items[:] = [it for it in items if id(it) in keep]

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any) -> None:
        # Under xdist only workers collect; the report reaches the terminal through here.
        self.report = self.report or getattr(node, "workeroutput", {}).get(_REPORT_KEY)

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.report:
            terminalreporter.write_line(self.report)


def pytest_configure(config: pytest.Config) -> None:
    sel = _load_selection()
    if sel is not None and sel.get("budget") is not None:
        config.pluginmanager.register(_Budget(sel), "graph_test_budget")
