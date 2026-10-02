"""Command-line interface: ``graph-test rank | run | graph``."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from graph_test import gitutil
from graph_test.config import SELECTION_ENV, Config
from graph_test.gitutil import Changes
from graph_test.graph import STATE_DIR, Graph, TestTarget, build, ensure_state_dir
from graph_test.rank import Ranked, Result, rank

# Above this many characters of node ids, pass test files instead (the plugin still filters).
_MAX_ARGV_CHARS = 100_000


def _setup(args: argparse.Namespace) -> tuple[Config, Graph]:
    root = gitutil.repo_root(Path(args.root).resolve())
    cfg = Config.load(root)
    return cfg, build(cfg)


def _changes(cfg: Config, args: argparse.Namespace) -> Changes:
    if args.files:
        return {str(Path(f)): None for f in args.files}
    return gitutil.changed_hunks(cfg.root, args.base or gitutil.default_base(cfg.root))


def select(g: Graph, result: Result, args: argparse.Namespace) -> tuple[list[Ranked], str]:
    """Pick tests to run. Returns (selection, reason it's everything, or "")."""
    if args.top:
        return [r for r in result.ranked if r.score >= args.min_score][: args.top], ""
    if result.global_changes:
        why = f"global file changed: {', '.join(result.global_changes)}"
    elif result.unmapped and args.on_unmapped == "all":
        why = f"unmapped change: {', '.join(result.unmapped[:5])}"
    else:
        picked = [
            r
            for r in result.ranked
            if (r.affected or (args.include_history and r.cochange_with))
            and r.score >= args.min_score
        ]
        return picked, ""
    by_node = {r.target.node: r for r in result.ranked}
    everything = [by_node.get(n) or Ranked(t, 0.0) for n, t in g.tests.items()]
    for r in everything:
        r.cases = []  # running everything: no case narrowing
    return everything, why


def write_selection(root: Path, selected: list[Ranked]) -> Path | None:
    """Selection file for the pytest plugin; None if nothing needs narrowing."""
    py = [r for r in selected if r.target.kind == "pytest"]
    if not py:
        return None
    payload = {
        "root": str(root),
        "full": sorted({r.target.nodeid for r in py if not r.cases}),
        "cases": {r.target.nodeid: sorted(r.cases) for r in py if r.cases},
    }
    path = root / STATE_DIR / "selection.json"
    ensure_state_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))
    return path


def batch_commands(targets: list[TestTarget]) -> list[list[str]]:
    """Group targets into as few invocations as possible, preserving rank order of first use."""
    pytest_ids: dict[str, None] = {}
    cargo: dict[tuple[str, tuple[str, str] | None], None] = {}
    for t in targets:
        if t.kind == "pytest":
            pytest_ids[t.nodeid or t.node] = None
        else:
            assert t.crate is not None
            cargo[(t.crate, t.selector)] = None
    cmds: list[list[str]] = []
    if pytest_ids:
        ids = list(pytest_ids)
        if sum(len(i) + 1 for i in ids) > _MAX_ARGV_CHARS:
            ids = list(dict.fromkeys(i.split("::", 1)[0] for i in ids))
        cmds.append(["pytest", *ids])
    for crate, selector in cargo:
        cmds.append(TestTarget("", "cargo", crate=crate, selector=selector).command())
    return cmds


def _print_summary(g: Graph, result: Result, selected: list[Ranked], why: str) -> None:
    n_cases = sum(1 for r in selected if r.cases)
    print(
        f"{len(result.changed)} changed file(s); selected {len(selected)} of {len(g.tests)} tests"
        + (f" ({n_cases} narrowed to specific cases)" if n_cases else "")
    )
    if why:
        print(f"  running everything - {why}")
    if result.unmapped:
        more = " ..." if len(result.unmapped) > 10 else ""
        print(f"  not in graph: {', '.join(result.unmapped[:10])}{more}")


def cmd_rank(args: argparse.Namespace) -> int:
    cfg, g = _setup(args)
    result = rank(cfg, g, _changes(cfg, args), use_history=not args.no_history)
    if args.affected or args.top:
        shown, why = select(g, result, args)
    else:
        shown, why = [r for r in result.ranked if r.score >= args.min_score], ""

    if args.format == "json":
        payload = {
            "changed": result.changed,
            "unmapped": result.unmapped,
            "ignored": result.ignored,
            "global_changes": result.global_changes,
            "run_all_reason": why or None,
            "tests": [
                {
                    "test": r.target.nodeid or r.target.node,
                    "score": round(r.score, 4),
                    "distance": r.distance,
                    "cases": r.cases,
                    "command": r.target.command(),
                    "why": r.explain(),
                }
                for r in shown
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.format == "cmd":
        sel = write_selection(cfg.root, shown) if any(r.cases for r in shown) else None
        for c in batch_commands([r.target for r in shown]):
            prefix = f"{SELECTION_ENV}={shlex.quote(str(sel))} " if sel and c[0] == "pytest" else ""
            print(prefix + shlex.join(c))
        return 0

    _print_summary(g, result, shown, why)
    for i, r in enumerate(shown, 1):
        cases = f"  [{len(r.cases)} data file(s)]" if r.cases else ""
        print(f"{i:4d}. {r.score:.3f}  {r.target.nodeid or r.target.node}{cases}")
        if args.explain:
            print(f"        {r.explain()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg, g = _setup(args)
    result = rank(cfg, g, _changes(cfg, args), use_history=not args.no_history)
    selected, why = select(g, result, args)
    _print_summary(g, result, selected, why)
    if not selected:
        return 0
    sel = write_selection(cfg.root, selected)
    env = {**os.environ, SELECTION_ENV: str(sel)} if sel else dict(os.environ)
    rc = 0
    for c in batch_commands([r.target for r in selected]):
        if c[0] == "pytest":
            c = [*shlex.split(args.pytest), *c[1:]]
        print(f"$ {shlex.join(c)}", flush=True)
        if args.dry_run:
            continue
        code = subprocess.call(c, cwd=cfg.root, env=env)
        if code == 5 and c[0] != "cargo":  # pytest: nothing collected after narrowing
            code = 0
        rc = rc or code
        if code and args.fail_fast:
            break
    return rc


def cmd_graph(args: argparse.Namespace) -> int:
    _, g = _setup(args)
    if args.deps:
        for d in sorted(g.deps.get(args.deps, ())):
            print(d)
    elif args.rdeps:
        reach = g.dependents([args.rdeps])
        for node, (dist, _) in sorted(reach.items(), key=lambda kv: (kv[1][0], kv[0])):
            mark = " [test]" if node in g.tests else ""
            print(f"{dist}  {node}{mark}")
    else:
        edges = sum(len(v) for v in g.deps.values())
        kinds: dict[str, int] = {}
        for t in g.tests.values():
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        print(
            f"nodes: {len(g.nodes())}  edges: {edges}  crates: {len(g.crates)}  "
            f"data nodes: {len(g.data_nodes)}"
        )
        print("tests: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graph-test", description=__doc__)
    p.add_argument("-C", "--root", default=".", help="path inside the repo (default: .)")
    sub = p.add_subparsers(dest="command", required=True)

    def selection(sp: argparse.ArgumentParser, gate: bool) -> None:
        sp.add_argument("--base", help="git ref to diff against (default: origin/main, main, ...)")
        sp.add_argument("--files", nargs="+", help="explicit changed files instead of git diff")
        sp.add_argument("-n", "--top", type=int, help="only the top N tests by score")
        sp.add_argument("--min-score", type=float, default=0.0)
        sp.add_argument("--no-history", action="store_true", help="skip git co-change mining")
        sp.add_argument(
            "--include-history",
            action="store_true",
            help="in affected mode, also select tests linked only by git co-change",
        )
        sp.add_argument(
            "--on-unmapped",
            choices=("all", "warn"),
            default="all" if gate else "warn",
            help="a changed file the graph can't place: run everything, or just report it",
        )

    r = sub.add_parser("rank", help="print tests ranked by likelihood of being affected")
    selection(r, gate=False)
    r.add_argument("--affected", action="store_true", help="only tests the graph reaches")
    r.add_argument("--explain", action="store_true", help="show why each test was picked")
    r.add_argument("--format", choices=("text", "json", "cmd"), default="text")
    r.set_defaults(func=cmd_rank)

    run = sub.add_parser(
        "run", help="run every affected test (pass/fail gate), or the top N with -n"
    )
    selection(run, gate=True)
    run.add_argument("--pytest", default="pytest", help="pytest invocation, e.g. 'uv run pytest'")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("-x", "--fail-fast", action="store_true")
    run.set_defaults(func=cmd_run)

    gr = sub.add_parser("graph", help="inspect the dependency graph")
    gr.add_argument("--deps", metavar="NODE", help="direct dependencies of NODE")
    gr.add_argument("--rdeps", metavar="NODE", help="everything that transitively depends on NODE")
    gr.set_defaults(func=cmd_graph)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except gitutil.GitError as e:
        print(f"graph-test: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
