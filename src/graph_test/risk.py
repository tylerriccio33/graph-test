"""How much a Python edit can change behavior, judged by comparing old and new ASTs.

``None`` from :func:`symbol_risk` means the edit is a no-op (formatting, or with
``checked`` only type annotations), so it selects nothing. Otherwise it returns a weight
in (0, 1] and a short reason, used to scale the test's graph score.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter
from typing import Any

from graph_test import pyscan

_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
_BRANCH = (
    ast.If,
    ast.IfExp,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.Try,
    ast.Match,
    ast.BoolOp,
    ast.comprehension,
)

# Weights by what the edit changed, highest first.
BRANCHES = 1.0
RETURNS = 0.8
CALLS = 0.6
NAMES = 0.5  # new names/attributes: might not resolve (moot when type-checked)
OTHER = 0.4
VALUES = 0.2  # only literals (or, when type-checked, which names are passed around)
FULL = (1.0, "")


def parse(source: str | None) -> ast.Module | None:
    if source is None:
        return None
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (*_FUNC, ast.ClassDef, ast.Module))
            and isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


def _is_type_checking(test: ast.expr) -> bool:
    return pyscan.dotted(test).endswith("TYPE_CHECKING")


class _StripTypes(ast.NodeTransformer):
    """Drop annotations that don't run: function signatures and locals, TYPE_CHECKING
    blocks. Class-level annotations stay, since dataclasses and the like act on them."""

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def _func(self, node: Any) -> Any:
        node.returns = None
        node.body = [self._local(s) for s in node.body]
        node.body = [s for s in node.body if s is not None] or [ast.Pass()]
        self.generic_visit(node)
        return node

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def _local(self, stmt: ast.stmt) -> ast.stmt | None:
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return None
            return ast.Assign(targets=[stmt.target], value=stmt.value, lineno=stmt.lineno)
        return stmt

    def visit_If(self, node: ast.If) -> ast.AST | None:
        if _is_type_checking(node.test):
            return None if not node.orelse else node.orelse  # type: ignore[return-value]
        self.generic_visit(node)
        return node


def normalize(node: ast.AST, checked: bool) -> str:
    """A dump that ignores positions, docstrings and, if ``checked``, type-only code."""
    node = copy.deepcopy(node)
    _strip_docstrings(node)
    if checked:
        node = _StripTypes().visit(node)
    return ast.dump(node, include_attributes=False)


def find(tree: ast.Module, qual: str) -> list[ast.stmt]:
    """Top-level statements defining ``qual`` (``name``, ``name@line`` or ``Cls.meth``)."""
    name = qual.partition("@")[0]
    cls, _, meth = name.rpartition(".")
    out: list[ast.stmt] = []
    for stmt in tree.body:
        if cls:
            if isinstance(stmt, ast.ClassDef) and stmt.name == cls:
                out += [m for m in stmt.body if isinstance(m, _FUNC) and m.name == meth]
        elif isinstance(stmt, (*_FUNC, ast.ClassDef)):
            if stmt.name == name:
                out.append(stmt)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t)):
                out.append(stmt)
    return out


def at_lines(tree: ast.Module, spans: list[list[int]], qual: str) -> list[ast.stmt]:
    """The definitions of ``qual`` occupying ``spans`` (as recorded by pyscan)."""
    starts = {s for s, _ in spans}
    out = []
    for stmt in find(tree, qual):
        decos = getattr(stmt, "decorator_list", [])
        if min([stmt.lineno, *(d.lineno for d in decos)]) in starts:
            out.append(stmt)
    return out


def _counter(nodes: list[ast.stmt], kinds: tuple[type, ...], key: Any) -> Counter[str]:
    return Counter(key(n) for s in nodes for n in ast.walk(s) if isinstance(n, kinds))


def _dump(n: ast.AST) -> str:
    return ast.dump(n, include_attributes=False)


def _names(nodes: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for s in nodes:
        for n in ast.walk(s):
            if isinstance(n, ast.Attribute):
                d = pyscan.dotted(n)
                if d:
                    out.add(d)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
    return out


class _Blank(ast.NodeTransformer):
    def __init__(self, names: bool) -> None:
        self.names = names

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.Constant(value=None)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        return ast.Constant(value=None)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.names and isinstance(node.ctx, ast.Load):
            return ast.Constant(value=None)
        return node


def _blanked(nodes: list[ast.stmt], names: bool) -> list[str]:
    return [
        ast.dump(_Blank(names).visit(copy.deepcopy(s)), include_attributes=False) for s in nodes
    ]


def compare(old: list[ast.stmt], new: list[ast.stmt], checked: bool) -> tuple[float, str] | None:
    """Risk of replacing ``old`` with ``new``; ``None`` if they are equivalent."""
    if sorted(normalize(s, checked) for s in old) == sorted(normalize(s, checked) for s in new):
        return None

    def branch_key(n: ast.AST) -> str:
        test = getattr(n, "test", None) or getattr(n, "iter", None) or getattr(n, "subject", None)
        ifs = getattr(n, "ifs", None)
        if isinstance(n, ast.BoolOp):
            return _dump(n)
        if ifs is not None:
            return f"{_dump(n.iter)}|{[_dump(i) for i in ifs]}"  # type: ignore[attr-defined]
        if isinstance(n, ast.Try):
            return f"try:{[_dump(h.type) if h.type else '' for h in n.handlers]}"
        return f"{type(n).__name__}:{_dump(test) if test else ''}"

    b_old, b_new = _counter(old, _BRANCH, branch_key), _counter(new, _BRANCH, branch_key)
    if b_old != b_new:
        delta = sum(b_new.values()) - sum(b_old.values())
        why = f"+{delta} branches" if delta > 0 else "branch logic changed"
        return BRANCHES, why

    if _counter(old, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom), _dump) != _counter(
        new, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom), _dump
    ):
        return RETURNS, "return/raise changed"

    def call_key(n: ast.AST) -> str:
        return pyscan.dotted(n.func) or _dump(n.func)  # type: ignore[attr-defined]

    if _counter(old, (ast.Call,), call_key) != _counter(new, (ast.Call,), call_key):
        return CALLS, "calls changed"

    if not checked:
        added = sorted(_names(new) - _names(old))
        if added:
            return NAMES, "new names: " + ", ".join(added[:3])

    if sorted(_blanked(old, checked)) == sorted(_blanked(new, checked)):
        return VALUES, "values only"
    return OTHER, "other"


def symbol_risk(
    old_tree: ast.Module | None,
    new_tree: ast.Module | None,
    qual: str | None,
    info: pyscan.ModuleInfo,
    checked: bool,
) -> tuple[float, str] | None:
    """Risk of the edit to one symbol (``None`` qual = module-level code)."""
    if old_tree is None or new_tree is None:
        return FULL
    if qual is None:
        # Module-level code: only "nothing changed" is worth detecting here.
        if normalize(old_tree, checked) == normalize(new_tree, checked):
            return None
        return FULL
    sym = info["symbols"].get(qual)
    if sym is None:
        return FULL
    new = at_lines(new_tree, sym["spans"], qual)
    old = find(old_tree, qual)
    if not new or not old:
        return FULL  # added or can't line up: assume the worst
    if "@" not in qual and len(old) != len(new):
        return FULL
    if "@" in qual:
        # Repeated name: match the old definition that looks most alike.
        results = [compare([o], new, checked) for o in old]
        if None in results:
            return None
        return min(r for r in results if r is not None)
    return compare(old, new, checked)
