"""Heuristic test ranking: graph distance + git co-change history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from graph_test import gitutil
from graph_test.config import Config
from graph_test.graph import Graph, TestTarget


@dataclass
class Ranked:
    target: TestTarget
    score: float
    distance: int | None = None
    path: list[str] = field(default_factory=list)  # test -> ... -> changed file
    cochange: float = 0.0
    cochange_with: str | None = None
    global_hit: bool = False

    def explain(self) -> str:
        bits = []
        if self.path:
            bits.append("graph: " + " -> ".join(self.path))
        if self.cochange_with:
            bits.append(f"co-changed with {self.cochange_with} ({self.cochange:.2f})")
        if self.global_hit:
            bits.append("global file changed")
        return "; ".join(bits)


@dataclass
class Result:
    ranked: list[Ranked]
    changed: list[str]
    unmapped: list[str]  # changed files that don't appear in the graph
    global_changes: list[str]


def cochange_scores(
    cfg: Config, tests: set[str], changed: set[str]
) -> dict[str, tuple[float, str]]:
    """test -> (best score, changed file responsible).

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


def rank(cfg: Config, g: Graph, changed: list[str], use_history: bool = True) -> Result:
    nodes = g.nodes() | set(g.tests)
    global_set = set(cfg.global_files)
    global_changes = [f for f in changed if f in global_set]
    starts = [f for f in changed if f in nodes]
    unmapped = [f for f in changed if f not in nodes and f not in global_set]

    reach = g.dependents(starts)
    co = cochange_scores(cfg, set(g.tests), set(changed)) if use_history else {}

    ranked: list[Ranked] = []
    for node, target in g.tests.items():
        r = Ranked(target=target, score=0.0)
        if node in reach:
            r.distance = reach[node][0]
            r.path = _path(node, reach)
            r.score += cfg.graph_weight / (1 + r.distance)
        if node in co:
            r.cochange, r.cochange_with = co[node]
            r.score += cfg.cochange_weight * r.cochange
        if global_changes and r.score < cfg.global_floor:
            r.score = cfg.global_floor
            r.global_hit = True
        if r.score > 0:
            ranked.append(r)
    ranked.sort(
        key=lambda r: (-r.score, r.distance if r.distance is not None else 1 << 30, r.target.node)
    )
    return Result(ranked=ranked, changed=changed, unmapped=unmapped, global_changes=global_changes)


def _path(node: str, reach: dict[str, tuple[int, str | None]]) -> list[str]:
    out = [node]
    prev = reach[node][1]
    while prev is not None:
        out.append(prev)
        prev = reach[prev][1]
    return out
