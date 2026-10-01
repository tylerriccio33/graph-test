"""Command-line interface: ``graph-test rank | run | graph``."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from graph_test import gitutil
from graph_test.config import Config
from graph_test.graph import Graph, TestTarget, build
from graph_test.rank import Result, rank


def _setup(args: argparse.Namespace) -> tuple[Config, Graph]:
    root = gitutil.repo_root(Path(args.root).resolve())
    cfg = Config.load(root)
    return cfg, build(cfg)


def _changed(cfg: Config, args: argparse.Namespace) -> list[str]:
    if args.files:
        return sorted({str(Path(f)) for f in args.files})
    return gitutil.changed_files(cfg.root, args.base or gitutil.default_base(cfg.root))


def _select(result: Result, top: int | None, min_score: float) -> list[TestTarget]:
    picked = [r for r in result.ranked if r.score >= min_score]
    return [r.target for r in (picked[:top] if top else picked)]


def batch_commands(targets: list[TestTarget]) -> list[list[str]]:
    """Group targets into as few invocations as possible, preserving rank order of first use."""
    pytest_files: list[str] = []
    cargo: dict[tuple[str, tuple[str, str] | None], None] = {}
    for t in targets:
        if t.kind == "pytest":
            pytest_files.append(t.node)
        else:
            assert t.crate is not None
            cargo[(t.crate, t.selector)] = None
    cmds: list[list[str]] = []
    if pytest_files:
        cmds.append(["pytest", *pytest_files])
    for crate, selector in cargo:
        cmds.append(TestTarget("", "cargo", crate, selector).command())
    return cmds


def cmd_rank(args: argparse.Namespace) -> int:
    cfg, g = _setup(args)
    changed = _changed(cfg, args)
    result = rank(cfg, g, changed, use_history=not args.no_history)
    shown = [r for r in result.ranked if r.score >= args.min_score]
    if args.top:
        shown = shown[: args.top]

    if args.format == "json":
        payload = {
            "changed": result.changed,
            "unmapped": result.unmapped,
            "global_changes": result.global_changes,
            "tests": [
                {
                    "test": r.target.node,
                    "score": round(r.score, 4),
                    "distance": r.distance,
                    "command": r.target.command(),
                    "why": r.explain(),
                }
                for r in shown
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.format == "cmd":
        for c in batch_commands([r.target for r in shown]):
            print(shlex.join(c))
        return 0

    print(f"{len(changed)} changed file(s), {len(result.ranked)} affected of {len(g.tests)} tests")
    if result.global_changes:
        print(f"  global changes: {', '.join(result.global_changes)}")
    if result.unmapped:
        print(
            f"  not in graph: {', '.join(result.unmapped[:10])}"
            + (" ..." if len(result.unmapped) > 10 else "")
        )
    for i, r in enumerate(shown, 1):
        print(f"{i:4d}. {r.score:.3f}  {r.target.node}")
        if args.explain:
            print(f"        {r.explain()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg, g = _setup(args)
    result = rank(cfg, g, _changed(cfg, args), use_history=not args.no_history)
    targets = _select(result, args.top, args.min_score)
    if not targets:
        print("no affected tests")
        return 0
    rc = 0
    for c in batch_commands(targets):
        if c[0] == "pytest":
            c = [*shlex.split(args.pytest), *c[1:]]
        print(f"$ {shlex.join(c)}", flush=True)
        if args.dry_run:
            continue
        code = subprocess.call(c, cwd=cfg.root)
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
        print(f"nodes: {len(g.nodes())}  edges: {edges}  crates: {len(g.crates)}")
        print("tests: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graph-test", description=__doc__)
    p.add_argument("-C", "--root", default=".", help="path inside the repo (default: .)")
    sub = p.add_subparsers(dest="command", required=True)

    def selection(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--base", help="git ref to diff against (default: origin/main, main, ...)")
        sp.add_argument("--files", nargs="+", help="explicit changed files instead of git diff")
        sp.add_argument("-n", "--top", type=int, help="only the top N tests")
        sp.add_argument("--min-score", type=float, default=0.0)
        sp.add_argument("--no-history", action="store_true", help="skip git co-change mining")

    r = sub.add_parser("rank", help="print tests ranked by likelihood of being affected")
    selection(r)
    r.add_argument("--explain", action="store_true", help="show why each test was picked")
    r.add_argument("--format", choices=("text", "json", "cmd"), default="text")
    r.set_defaults(func=cmd_rank)

    run = sub.add_parser("run", help="run the selected tests")
    selection(run)
    run.add_argument("--pytest", default="pytest", help="pytest invocation, e.g. 'uv run pytest'")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("-x", "--fail-fast", action="store_true")
    run.set_defaults(func=cmd_run)

    gr = sub.add_parser("graph", help="inspect the dependency graph")
    gr.add_argument("--deps", metavar="FILE", help="direct dependencies of FILE")
    gr.add_argument("--rdeps", metavar="FILE", help="everything that transitively depends on FILE")
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
