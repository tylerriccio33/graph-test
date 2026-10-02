"""Test selection and ranking: graph reachability + git co-change history."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from graph_test import gitutil, pyscan
from graph_test.config import Config
from graph_test.gitutil import Changes
from graph_test.graph import STATE_DIR, Graph, TestTarget


@dataclass
class Ranked:
    target: TestTarget
    score: float
    distance: int | None = None
    path: list[str] = field(default_factory=list)  # test -> ... -> changed node
    cochange: float = 0.0
    cochange_with: str | None = None
    global_hit: bool = False
    # Non-empty when the test is reached only through data files: just the parametrized
    # cases using these files need rerunning.
    cases: list[str] = field(default_factory=list)

    @property
    def affected(self) -> bool:
        """Reached through the dependency graph (as opposed to history or a global file)."""
        return self.distance is not None

    def explain(self) -> str:
        bits = []
        if self.path:
            bits.append("graph: " + " -> ".join(self.path))
        if self.cases:
            bits.append("cases using: " + ", ".join(self.cases))
        if self.cochange_with:
            bits.append(f"co-changed with {self.cochange_with} ({self.cochange:.2f})")
        if self.global_hit:
            bits.append("global file changed")
        return "; ".join(bits)


@dataclass
class Result:
    ranked: list[Ranked]
    changed: list[str]
    unmapped: list[str]  # changed files that select nothing and aren't ignored
    global_changes: list[str]
    ignored: list[str]


def starts_for(
    cfg: Config, g: Graph, nodes: set[str], path: str, hunks: list[gitutil.Hunk] | None
) -> tuple[list[str], list[str]]:
    """Graph nodes a changed file starts from, split into (code, data) starts."""
    code: list[str] = []
    data: list[str] = []
    info = g.py_info.get(path)
    if info is not None:
        if hunks is None:
            code.append(path)
            code += [f"{path}::{q}" for q in info["symbols"]]
        else:
            try:
                lines = (cfg.root / path).read_text(errors="replace").splitlines()
            except OSError:
                lines = None
            for qual in pyscan.symbols_at(info, hunks, lines):
                code.append(path if qual is None else f"{path}::{qual}")
    elif path in nodes and path not in g.data_nodes:
        code.append(path)
        if f"{path}#self" in nodes:
            code.append(f"{path}#self")
    elif path.endswith((".py", ".pyi")) and not (cfg.root / path).exists():
        # Deleted module: start from whatever imported it by name.
        node = f"missing:{pyscan.module_name(Path(path), cfg.root)}"
        if node in nodes:
            code.append(node)

    if path in g.data_nodes:
        data.append(path)
    for parent in Path(path).parents:
        node = f"data:{parent}"
        if node in g.data_nodes:
            data.append(node)
    for pattern, targets in cfg.data.items():
        if fnmatch(path, pattern):
            data += [n for n in nodes if any(fnmatch(n, t) for t in targets)]
    return code, data


def cochange_scores(
    cfg: Config, tests: set[str], changed: set[str]
) -> dict[str, tuple[float, str]]:
    """test file -> (best score, changed file responsible).

    Score is P(test touched | file touched), shrunk toward 0 for rarely-seen files.
    """
    file_count: Counter[str] = Counter()
    pair: Counter[tuple[str, str]] = Counter()
    for files in gitutil.commit_file_sets(cfg.root, cfg.history_commits):
        if len(files) > cfg.max_commit_files:
            continue
        touched_tests = [f for f in files if f in tests]
        for f in files:
            if f in changed:
                file_count[f] += 1
                for t in touched_tests:
                    if t != f:
                        pair[(f, t)] += 1
    out: dict[str, tuple[float, str]] = {}
    for (f, t), n in pair.items():
        if n < cfg.min_cochange_support:
            continue
        score = n / (file_count[f] + 2)
        if score > out.get(t, (0.0, ""))[0]:
            out[t] = (score, f)
    return out


def rank(cfg: Config, g: Graph, changes: Changes, use_history: bool = True) -> Result:
    nodes = g.nodes()
    global_set = set(cfg.global_files)
    changed = sorted(f for f in changes if not f.startswith(f"{STATE_DIR}/"))
    ignored = [f for f in changed if any(fnmatch(f, p) for p in cfg.ignore)]
    global_changes = [f for f in changed if f in global_set]

    code_starts: list[str] = []
    data_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    unmapped: list[str] = []
    for f in changed:
        if f in ignored or f in global_set:
            continue
        code, data = starts_for(cfg, g, nodes, f, changes[f])
        code_starts += code
        if data:
            data_groups[tuple(sorted(set(data)))].append(f)
        # A Python edit touching only comments/blank lines/docstrings maps to nothing by design.
        quiet = f in g.py_info and changes[f] is not None
        if not code and not data and not quiet:
            unmapped.append(f)

    reach = g.dependents(code_starts)
    data_reach = [(g.dependents(list(starts)), files) for starts, files in data_groups.items()]
    test_files = {t.file for t in g.tests.values()}
    co = cochange_scores(cfg, test_files, set(changed)) if use_history else {}

    ranked: list[Ranked] = []
    for node, target in g.tests.items():
        r = Ranked(target=target, score=0.0)
        if node in reach:
            r.distance = reach[node][0]
            r.path = _path(node, reach)
        else:
            for dreach, files in data_reach:
                if node in dreach:
                    r.cases += files
                    if r.distance is None or dreach[node][0] < r.distance:
                        r.distance = dreach[node][0]
                        r.path = _path(node, dreach)
        if r.distance is not None:
            r.score += cfg.graph_weight / (1 + r.distance)
        if target.file in co:
            r.cochange, r.cochange_with = co[target.file]
            r.score += cfg.cochange_weight * r.cochange
        if global_changes and r.score < cfg.global_floor:
            r.score = cfg.global_floor
            r.global_hit = True
        if r.score > 0:
            ranked.append(r)
    ranked.sort(
        key=lambda r: (-r.score, r.distance if r.distance is not None else 1 << 30, r.target.node)
    )
    return Result(ranked, changed, unmapped, global_changes, ignored)


def _path(node: str, reach: dict[str, tuple[int, str | None]]) -> list[str]:
    out = [node]
    prev = reach[node][1]
    while prev is not None:
        out.append(prev)
        prev = reach[prev][1]
    return out
