"""Rust crate discovery and lightweight (regex-based) source scanning.

This deliberately avoids a full parser: we only need ``mod x;`` declarations,
``use`` paths and whether a file contains tests.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.M)
_USE_RE = re.compile(r"\buse\s+([^;]+);")
_TEST_RE = re.compile(r"#\[(?:cfg\(test\)|test|tokio::test|rstest)")
_PATH_SEG_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ATTRS = r"(?:#\[[^\]]*\]\s*)*"
_VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
_PYFN_RE = re.compile(
    r"#\[pyfunction[^\]]*\]\s*"
    + _ATTRS
    + _VIS
    + r"(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"
)
_PYCLASS_RE = re.compile(r"#\[pyclass[^\]]*\]\s*" + _ATTRS + _VIS + r"(?:struct|enum)\s+(\w+)")
_PYMETHODS_RE = re.compile(r"#\[pymethods\]\s*" + _ATTRS + r"impl(?:<[^>]*>)?\s+(\w+)")
_PYNAME_RE = re.compile(r'(?:pyo3|pyfunction|pyclass)\((?:[^()"]*,\s*)?name\s*=\s*"([^"]+)"')


@dataclass
class Crate:
    name: str  # package name as in Cargo.toml
    dir: Path  # repo-relative crate directory
    lib_name: str  # name used in `use` paths (dashes -> underscores)
    roots: list[Path] = field(default_factory=list)  # lib.rs / main.rs, repo-relative
    deps: set[str] = field(default_factory=set)  # dependency package names
    pyo3_modules: set[str] = field(default_factory=set)  # python module names it provides

    @property
    def node(self) -> str:
        return f"crate:{self.name}"


def load_crate(root: Path, manifest: Path) -> Crate | None:
    try:
        data = tomllib.loads((root / manifest).read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return None
    pkg = data.get("package")
    if not isinstance(pkg, dict) or "name" not in pkg:
        return None  # virtual workspace manifest
    name = str(pkg["name"])
    crate_dir = manifest.parent
    lib = data.get("lib", {})
    lib_name = str(lib.get("name", name)).replace("-", "_")
    crate = Crate(name=name, dir=crate_dir, lib_name=lib_name)

    lib_path = crate_dir / str(lib.get("path", "src/lib.rs"))
    for candidate in (lib_path, crate_dir / "src/main.rs"):
        if (root / candidate).exists():
            crate.roots.append(candidate)
    for b in data.get("bin", []) or []:
        if isinstance(b, dict) and "path" in b and (root / crate_dir / b["path"]).exists():
            crate.roots.append(crate_dir / b["path"])

    dep_tables = [
        data.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")
    ]
    for table in dep_tables:
        for dep_name, spec in table.items():
            real = spec.get("package", dep_name) if isinstance(spec, dict) else dep_name
            crate.deps.add(str(real))

    if "pyo3" in crate.deps:
        crate.pyo3_modules.add(lib_name)
        pyproject = root / crate_dir / "pyproject.toml"
        if pyproject.exists():
            try:
                pdata = tomllib.loads(pyproject.read_text())
                mod = pdata.get("tool", {}).get("maturin", {}).get("module-name")
                if mod:
                    crate.pyo3_modules.add(str(mod))
            except (tomllib.TOMLDecodeError, OSError):
                pass
    return crate


@dataclass
class RustFile:
    mods: list[str]
    uses: list[list[str]]  # each a path split on ::
    has_tests: bool
    # PyO3: python-visible functions, [rust_struct, python_name] classes, and
    # rust structs with a #[pymethods] impl in this file.
    pyfunctions: list[str] = field(default_factory=list)
    pyclasses: list[list[str]] = field(default_factory=list)
    pymethods: list[str] = field(default_factory=list)


def _py_name(match: re.Match[str], default: str) -> str:
    named = _PYNAME_RE.search(match.group(0))
    return named.group(1) if named else default


def scan(source: str) -> RustFile:
    code = _COMMENT_RE.sub("", source)
    mods = _MOD_RE.findall(code)
    uses: list[list[str]] = []
    for m in _USE_RE.finditer(code):
        uses.extend(_expand_use(m.group(1)))
    return RustFile(
        mods=mods,
        uses=uses,
        has_tests=bool(_TEST_RE.search(code)),
        pyfunctions=[_py_name(m, m.group(1)) for m in _PYFN_RE.finditer(code)],
        pyclasses=[[m.group(1), _py_name(m, m.group(1))] for m in _PYCLASS_RE.finditer(code)],
        pymethods=_PYMETHODS_RE.findall(code),
    )


def _expand_use(text: str) -> list[list[str]]:
    """Expand ``a::b::{c, d::e}`` into ``[[a,b,c],[a,b,d,e]]`` (one brace level deep is enough)."""
    text = " ".join(text.split())
    if "{" not in text:
        return [_segments(text)]
    prefix, _, rest = text.partition("{")
    inner = rest.rsplit("}", 1)[0]
    base = _segments(prefix)
    out: list[list[str]] = []
    depth = 0
    item = ""
    for ch in inner + ",":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            if item.strip():
                out.extend([base + p for p in _expand_use(item.strip())])
            item = ""
        else:
            item += ch
    return out or [base]


def _segments(path: str) -> list[str]:
    path = path.split(" as ")[0]
    return [s for s in (p.strip() for p in path.split("::")) if s and _PATH_SEG_RE.fullmatch(s)]


def child_module_file(root: Path, parent_file: Path, mod: str) -> Path | None:
    """Resolve ``mod foo;`` declared in ``parent_file`` to its file."""
    if parent_file.name in ("lib.rs", "main.rs", "mod.rs"):
        base = parent_file.parent
    else:
        base = parent_file.parent / parent_file.stem
    for candidate in (base / f"{mod}.rs", base / mod / "mod.rs"):
        if (root / candidate).exists():
            return candidate
    return None


def module_path(crate_src: Path, file: Path) -> list[str]:
    """``src/a/b.rs`` -> ``[a, b]``; ``src/a/mod.rs`` -> ``[a]``; ``src/lib.rs`` -> ``[]``."""
    try:
        rel = file.relative_to(crate_src)
    except ValueError:
        return []
    parts = list(rel.parts)
    last = parts.pop()
    if last not in ("lib.rs", "main.rs", "mod.rs"):
        parts.append(last.removesuffix(".rs"))
    return parts
