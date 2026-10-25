"""Python symbol-level analysis.

Each module is split into *symbols* (top-level functions, classes, assigned names, and
test methods inside test classes) plus a *module* bucket for remaining top-level code.
For each we record the names it references, path-like string literals, and (for
functions) argument names so pytest fixtures can be resolved.

The result is a plain JSON-serializable dict so it can be cached.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ModuleInfo = dict[str, Any]

_PATHISH = re.compile(r"^[\w.\-/ *?\[\]{}%<>]+$")
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def module_name(rel: Path, root: Path) -> str:
    """Dotted module name for a repo-relative .py/.pyi path.

    Walks up while the parent directory is a package (has ``__init__.py``), so
    ``src/pkg/sub/mod.py`` becomes ``pkg.sub.mod`` regardless of source layout.
    """
    parts = [rel.stem] if rel.stem != "__init__" else []
    parent = rel.parent
    while parent != Path(".") and (root / parent / "__init__.py").exists():
        parts.insert(0, parent.name)
        parent = parent.parent
    if not parts:  # a top-level __init__.py with no package parent
        parts = [rel.parent.name]
    return ".".join(parts)


def is_test_file(rel: Path) -> bool:
    name = rel.name
    return rel.suffix == ".py" and (
        name.startswith("test_") or name.endswith("_test.py") or name == "tests.py"
    )


def dotted(node: ast.AST) -> str:
    """``a.b.c`` for an attribute chain rooted at a Name, else ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _collect(node: ast.AST, exclude: set[int]) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    lits: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in exclude:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            chain = dotted(n)
            if chain:
                # The whole chain only: its root Name alone would read as a bare use.
                refs.add(chain)
                continue
        elif (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and 0 < len(n.value) <= 300
            and _PATHISH.match(n.value)
        ):
            lits.add(n.value)
        stack.extend(ast.iter_child_nodes(n))
    return refs, lits


def _span(node: ast.stmt) -> list[int]:
    decos = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(d.lineno for d in decos)])
    return [start, node.end_lineno or node.lineno]


def _fixture_flags(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[bool, bool, str | None]:
    fixture = autouse = False
    name = None
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if dotted(target).endswith("fixture"):
            fixture = True
            if isinstance(d, ast.Call):
                autouse = any(
                    k.arg == "autouse" and isinstance(k.value, ast.Constant) and k.value.value
                    for k in d.keywords
                )
                for k in d.keywords:
                    if k.arg == "name" and isinstance(k.value, ast.Constant):
                        name = str(k.value.value)
    return fixture, autouse, name


_STR_ARGS = {"usefixtures": "uses", "getfixturevalue": "uses"}
_DYN_IMPORTS = ("import_module", "__import__")


def _dynamic(node: ast.AST, exclude: set[int]) -> dict[str, list[str]]:
    """String-named dependencies: fixtures requested by name (``usefixtures("x")``,
    ``request.getfixturevalue("x")``) and modules imported by literal name
    (``importlib.import_module("a.b")``)."""
    out: dict[str, set[str]] = {}
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in exclude:
            continue
        if isinstance(n, ast.Call):
            tail = dotted(n.func).rpartition(".")[2]
            strs = [
                a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            if tail in _STR_ARGS:
                out.setdefault("uses", set()).update(strs)
            elif tail in _DYN_IMPORTS and strs:
                out.setdefault("dyn", set()).add(strs[0])
        stack.extend(ast.iter_child_nodes(n))
    return {k: sorted(v) for k, v in out.items()}


def _symbol(node: ast.stmt, kind: str, exclude: set[int] | None = None) -> dict[str, Any]:
    refs, lits = _collect(node, exclude or set())
    sym: dict[str, Any] = {"kind": kind, "spans": [_span(node)], "refs": sorted(refs)}
    if lits:
        sym["lits"] = sorted(lits)
    sym.update(_dynamic(node, exclude or set()))
    if isinstance(node, _FUNC):
        a = node.args
        sym["args"] = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
        fixture, autouse, fname = _fixture_flags(node)
        if fixture:
            sym["fixture"] = True
            if fname:
                sym["fname"] = fname
        if autouse:
            sym["autouse"] = True
        registers = _registrations(node)
        if registers:
            sym["registers"] = registers
    return sym


def _registrations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Dispatchers this function registers with: ``@f.register`` / ``@f.register(T)`` -> ``f``.

    Covers functools.singledispatch / singledispatchmethod and similar registries, whose
    implementations are reached at runtime through the dispatcher, never by name.
    """
    out = []
    for d in node.decorator_list:
        chain = dotted(d.func if isinstance(d, ast.Call) else d)
        if chain.endswith(".register"):
            out.append(chain.removesuffix(".register"))
    return out


def _add_def(symbols: dict[str, dict[str, Any]], name: str, sym: dict[str, Any]) -> None:
    """Add a def/class; a redefinition takes the plain name (it's what references see) and
    the earlier one is kept as ``name@line``, so e.g. many ``def _`` all stay mappable."""
    old = symbols.pop(name, None)
    if old is not None:
        symbols[f"{name}@{old['spans'][0][0]}"] = old
    symbols[name] = sym


def _merge(into: dict[str, Any], other: dict[str, Any]) -> None:
    into["spans"] += other["spans"]
    into["refs"] = sorted(set(into["refs"]) | set(other["refs"]))
    for key in ("uses", "dyn"):
        if key in other:
            into[key] = sorted(set(into.get(key, [])) | set(other[key]))
    if "lits" in other:
        into["lits"] = sorted(set(into.get("lits", [])) | set(other["lits"]))


def _toplevel(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Module-level statements, looking inside ``if``/``try``/``with`` blocks so a def under
    ``if sys.platform...`` or ``try: ... except ImportError:`` is still a symbol. The
    enclosing statement is yielded too, as module-level code."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            yield stmt
            yield from _toplevel(stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield stmt
            inner = stmt.body + [x for h in stmt.handlers for x in h.body]
            yield from _toplevel(inner + stmt.orelse + stmt.finalbody)
        elif isinstance(stmt, ast.With):
            yield stmt
            yield from _toplevel(stmt.body)
        else:
            yield stmt


def analyze(source: str, module: str, is_package: bool) -> ModuleInfo:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {"imports": {}, "stars": [], "modules": [], "symbols": {}, "module": {"refs": []}}
    package = module if is_package else module.rpartition(".")[0]

    imports: dict[str, list[str]] = {}  # local alias -> [module, name or ""]
    stars: list[str] = []
    modules: set[str] = set()
    # An import inside a function body runs when the function is called, not when the
    # module loads: it names a dependency of that function (via `imports`), but is no
    # import side effect of the module, so it stays out of `modules`.
    deferred = {
        id(n)
        for fn in ast.walk(tree)
        if isinstance(fn, (*_FUNC, ast.Lambda))
        for n in ast.walk(fn)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if id(node) not in deferred:
                    modules.add(alias.name)
                if alias.asname:
                    imports[alias.asname] = [alias.name, ""]
                else:
                    top = alias.name.split(".")[0]
                    imports.setdefault(top, [top, ""])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                if node.level > 1:
                    base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join([*base_parts, node.module] if node.module else base_parts)
            else:
                base = node.module or ""
            if not base:
                continue
            if id(node) not in deferred:
                modules.add(base)
            for alias in node.names:
                if alias.name == "*":
                    stars.append(base)
                else:
                    imports[alias.asname or alias.name] = [base, alias.name]
                    if id(node) not in deferred:
                        modules.add(f"{base}.{alias.name}")

    symbols: dict[str, dict[str, Any]] = {}
    rest: list[ast.stmt] = []
    inert: list[list[int]] = []  # docstrings / bare constants: changing them affects nothing
    for stmt in _toplevel(tree.body):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            inert.append(_span(stmt))
        elif isinstance(stmt, _FUNC):
            _add_def(symbols, stmt.name, _symbol(stmt, "func"))
        elif isinstance(stmt, ast.ClassDef):
            is_test_cls = stmt.name.startswith("Test") or any(
                dotted(b).endswith("TestCase") for b in stmt.bases
            )
            methods = (
                [m for m in stmt.body if isinstance(m, _FUNC) and m.name.startswith("test")]
                if is_test_cls
                else []
            )
            cls = _symbol(stmt, "class", {id(m) for m in methods})
            if is_test_cls:
                cls["test_class"] = True
            _add_def(symbols, stmt.name, cls)
            for m in methods:
                sym = _symbol(m, "method")
                sym["parent"] = stmt.name
                symbols[f"{stmt.name}.{m.name}"] = sym
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if not names:
                rest.append(stmt)
            for name in names:
                sym = _symbol(stmt, "assign")
                if name in symbols:
                    _merge(symbols[name], sym)
                else:
                    symbols[name] = sym
        elif not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            rest.append(stmt)

    plugins: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in targets):
                plugins += [
                    n.value
                    for n in ast.walk(stmt.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]

    mod_refs: set[str] = set()
    mod_lits: set[str] = set()
    for stmt in rest:
        r, lt = _collect(stmt, set())
        mod_refs |= r
        mod_lits |= lt
    mod: dict[str, Any] = {"refs": sorted(mod_refs)}
    if mod_lits:
        mod["lits"] = sorted(mod_lits)
    for stmt in rest:
        for key, vals in _dynamic(stmt, set()).items():
            mod[key] = sorted(set(mod.get(key, [])) | set(vals))
    return {
        "imports": imports,
        "stars": stars,
        "modules": sorted(modules),
        "symbols": symbols,
        "module": mod,
        "inert": inert,
        "plugins": plugins,
    }


def symbols_at(
    info: ModuleInfo, hunks: list[tuple[int, int, bool]], lines: list[str] | None = None
) -> set[str | None]:
    """Innermost symbols touched by diff hunks; ``None`` means module-level code.

    Each hunk is ``(start, end, pure_deletion)`` in new-file line numbers. ``lines`` (the
    current file contents) lets blank and comment-only lines be skipped. A pure deletion
    that lands only on skipped lines removed something whole, so it counts as module-level.
    """
    inert = info.get("inert", [])
    spans: list[tuple[int, int, int, str]] = []
    for qual, sym in info["symbols"].items():
        depth = 1 if sym["kind"] == "method" else 0
        for s, e in sym["spans"]:
            spans.append((s, e, depth, qual))
    out: set[str | None] = set()
    for start, end, deletion in hunks:
        hit = False
        for line in range(start, end + 1):
            if lines is not None:
                text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
                if not text or text.startswith("#"):
                    continue
            if any(s <= line <= e for s, e in inert):
                continue
            best: tuple[int, str] | None = None
            for s, e, depth, qual in spans:
                if s <= line <= e and (best is None or depth > best[0]):
                    best = (depth, qual)
            out.add(best[1] if best else None)
            hit = True
        if deletion and not hit:
            out.add(None)
    return out
