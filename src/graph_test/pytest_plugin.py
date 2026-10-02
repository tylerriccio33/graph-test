"""pytest plugin: narrow collection to graph-test's selection, down to parametrized cases.

Inert unless ``GRAPH_TEST_SELECTION`` points at a selection file written by
``graph-test run`` / ``graph-test rank --format cmd``. Registered via the ``pytest11``
entry point, so it's active wherever graph-test is installed.

Selection file::

    {"root": "/abs/repo", "full": ["tests/a.py::test_x", "tests/b.py"],
     "cases": {"tests/c.py::test_corpus": ["tests/corpus/x.sas"]}}

``full`` entries keep every item; ``cases`` entries keep only the parametrized items whose
parameters (or ids) refer to one of the listed changed files. If none match, all of that
test's items are kept, so ambiguity never skips anything.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_test.config import SELECTION_ENV as ENV


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
    else:
        yield value


def _matches(item: pytest.Item, changed: list[Path], root: Path) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    names = {c.name for c in changed} | {c.stem for c in changed}
    cid = str(callspec.id)
    if cid in names or any(part in names for part in cid.split("-")):
        return True
    for value in _flatten(callspec.params):
        if not isinstance(value, (str, os.PathLike)):
            continue
        text = os.fspath(value)
        if text in names:
            return True
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
                return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    try:
        sel = json.loads(Path(path).read_text())
    except (OSError, ValueError):
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
        matched = [it for it in group if _matches(it, changed, root)]
        keep.update(id(it) for it in (matched or group))

    dropped = [it for it in items if id(it) not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [it for it in items if id(it) in keep]
