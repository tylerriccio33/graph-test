"""Python import extraction and module naming."""

from __future__ import annotations

import ast
from pathlib import Path


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


def imports(source: str, module: str, is_package: bool) -> set[str]:
    """Candidate dotted names imported by ``source``.

    For ``from a import b`` both ``a.b`` and ``a`` are returned; the resolver keeps
    whichever exists. Relative imports are made absolute using ``module``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    package = module if is_package else module.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
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
            out.add(base)
            for alias in node.names:
                if alias.name != "*":
                    out.add(f"{base}.{alias.name}")
    return out


def is_test_file(rel: Path) -> bool:
    name = rel.name
    return rel.suffix == ".py" and (
        name.startswith("test_") or name.endswith("_test.py") or name == "tests.py"
    )
