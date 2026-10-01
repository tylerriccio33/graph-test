"""Build a file-level dependency graph across Python and Rust sources.

Nodes are repo-relative file paths plus synthetic ``crate:<name>`` nodes. An edge
``a -> b`` means "a depends on b", so a change to ``b`` may affect ``a``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from graph_test import pyscan, rsscan
from graph_test.config import Config

CACHE_VERSION = 1


@dataclass
class TestTarget:
    node: str
    kind: str  # "pytest" | "cargo"
    crate: str | None = None
    # For cargo: ("--test", "name") for integration tests, ("--lib", "a::b") for unit tests.
    selector: tuple[str, str] | None = None

    def command(self) -> list[str]:
        if self.kind == "pytest":
            return ["pytest", self.node]
        assert self.crate is not None
        cmd = ["cargo", "test", "-p", self.crate]
        if self.selector:
            flag, value = self.selector
            if flag == "--test":
                cmd += ["--test", value]
            else:
                cmd += [flag, *(["--", f"{value}::"] if value else [])]
        return cmd


@dataclass
class Graph:
    deps: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    tests: dict[str, TestTarget] = field(default_factory=dict)
    crates: dict[str, rsscan.Crate] = field(default_factory=dict)
    _rdeps: dict[str, set[str]] | None = None

    def add_edge(self, src: str, dst: str) -> None:
        if src != dst:
            self.deps[src].add(dst)
            self._rdeps = None

    @property
    def rdeps(self) -> dict[str, set[str]]:
        if self._rdeps is None:
            r: dict[str, set[str]] = defaultdict(set)
            for src, dsts in self.deps.items():
                for dst in dsts:
                    r[dst].add(src)
            self._rdeps = r
        return self._rdeps

    def nodes(self) -> set[str]:
        out = set(self.deps)
        for dsts in self.deps.values():
            out |= dsts
        return out

    def dependents(self, starts: list[str]) -> dict[str, tuple[int, str | None]]:
        """BFS over reverse edges: node -> (distance, predecessor toward a start)."""
        seen: dict[str, tuple[int, str | None]] = {s: (0, None) for s in starts}
        queue = deque(starts)
        rdeps = self.rdeps
        while queue:
            cur = queue.popleft()
            dist = seen[cur][0]
            for nxt in rdeps.get(cur, ()):
                if nxt not in seen:
                    seen[nxt] = (dist + 1, cur)
                    queue.append(nxt)
        return seen


class _ParseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, list] = {}
        self.dirty = False
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") == CACHE_VERSION:
                self.data = raw["files"]
        except (OSError, ValueError, KeyError):
            pass

    def get(self, rel: str, st: os.stat_result) -> object | None:
        entry = self.data.get(rel)
        if entry and entry[0] == st.st_mtime_ns and entry[1] == st.st_size:
            return entry[2]
        return None

    def put(self, rel: str, st: os.stat_result, payload: object) -> None:
        self.data[rel] = [st.st_mtime_ns, st.st_size, payload]
        self.dirty = True

    def save(self, live: set[str]) -> None:
        if not self.dirty and live >= set(self.data):
            return
        self.data = {k: v for k, v in self.data.items() if k in live}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"version": CACHE_VERSION, "files": self.data}))


def _walk(cfg: Config) -> tuple[list[Path], list[Path], list[Path]]:
    py, rs, manifests = [], [], []
    excluded = set(cfg.exclude)
    for dirpath, dirnames, filenames in os.walk(cfg.root):
        dirnames[:] = [d for d in dirnames if d not in excluded and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(cfg.root)
        for fn in filenames:
            rel = rel_dir / fn
            if fn.endswith((".py", ".pyi")):
                py.append(rel)
            elif fn.endswith(".rs"):
                rs.append(rel)
            elif fn == "Cargo.toml":
                manifests.append(rel)
    return py, rs, manifests


def build(cfg: Config) -> Graph:
    root = cfg.root
    g = Graph()
    cache = _ParseCache(root / ".graph-test" / "parse-cache.json")
    py_files, rs_files, manifests = _walk(cfg)
    live: set[str] = set()

    for m in manifests:
        crate = rsscan.load_crate(root, m)
        if crate:
            g.crates[crate.name] = crate
    by_lib = {c.lib_name: c for c in g.crates.values()}
    for crate in g.crates.values():
        for dep in crate.deps:
            if dep in g.crates:
                g.add_edge(crate.node, g.crates[dep].node)
        for r in crate.roots:
            g.add_edge(crate.node, str(r))
        # A crate's own manifest is part of it.
        g.add_edge(crate.node, str(crate.dir / "Cargo.toml"))

    _build_rust(cfg, g, rs_files, by_lib, cache, live)
    _build_python(cfg, g, py_files, cache, live)
    cache.save(live)
    return g


def _owning_crate(rel: Path, crates: list[rsscan.Crate]) -> rsscan.Crate | None:
    best: rsscan.Crate | None = None
    for c in crates:
        if (c.dir == Path(".") or rel.is_relative_to(c.dir)) and (
            best is None or len(c.dir.parts) > len(best.dir.parts)
        ):
            best = c
    return best


def _scan_rs(root: Path, rel: Path, cache: _ParseCache, live: set[str]) -> rsscan.RustFile:
    key = str(rel)
    live.add(key)
    st = (root / rel).stat()
    hit = cache.get(key, st)
    if isinstance(hit, dict):
        return rsscan.RustFile(mods=hit["mods"], uses=hit["uses"], has_tests=hit["has_tests"])
    try:
        src = (root / rel).read_text(errors="replace")
    except OSError:
        src = ""
    rf = rsscan.scan(src)
    cache.put(key, st, {"mods": rf.mods, "uses": rf.uses, "has_tests": rf.has_tests})
    return rf


def _build_rust(
    cfg: Config,
    g: Graph,
    rs_files: list[Path],
    by_lib: dict[str, rsscan.Crate],
    cache: _ParseCache,
    live: set[str],
) -> None:
    root = cfg.root
    crates = list(g.crates.values())
    scanned = {f: _scan_rs(root, f, cache, live) for f in rs_files}

    # Per crate: module path -> file, discovered by walking `mod` declarations from roots.
    modmap: dict[str, dict[tuple[str, ...], Path]] = {}
    file_mod: dict[Path, tuple[str, ...]] = {}
    lib_files: set[Path] = set()
    for crate in crates:
        mm: dict[tuple[str, ...], Path] = {}
        for crate_root in crate.roots:
            is_lib = crate_root.name != "main.rs"
            queue: deque[tuple[Path, tuple[str, ...]]] = deque([(crate_root, ())])
            while queue:
                f, mp = queue.popleft()
                if is_lib:
                    mm.setdefault(mp, f)
                    lib_files.add(f)
                file_mod.setdefault(f, mp)
                for mod in scanned.get(f, rsscan.RustFile([], [], False)).mods:
                    child = rsscan.child_module_file(root, f, mod)
                    if child and child not in file_mod:
                        g.add_edge(str(f), str(child))
                        queue.append((child, (*mp, mod)))
        modmap[crate.name] = mm

    for f, rf in scanned.items():
        crate = _owning_crate(f, crates)
        if crate is None:
            continue
        node = str(f)
        rel_in_crate = f.relative_to(crate.dir) if crate.dir != Path(".") else f
        top = rel_in_crate.parts[0] if rel_in_crate.parts else ""
        is_aux_root = top in ("tests", "benches", "examples") and len(rel_in_crate.parts) == 2
        if f.name == "build.rs" and len(rel_in_crate.parts) == 1:
            g.add_edge(crate.node, node)
            continue

        if is_aux_root:
            # Integration tests/benches/examples are crate roots linking the library.
            g.add_edge(node, crate.node)
            for mod in rf.mods:
                child = rsscan.child_module_file(root, f.parent / "lib.rs", mod)
                if child:
                    g.add_edge(node, str(child))
            if top == "tests":
                g.tests[node] = TestTarget(node, "cargo", crate.name, ("--test", f.stem))

        cur_mod = file_mod.get(f, ())
        for path in rf.uses:
            target = _resolve_use(path, crate, cur_mod, by_lib, modmap)
            if target:
                g.add_edge(node, target)

        if rf.has_tests and not is_aux_root and f in file_mod:
            flag = "--lib" if f in lib_files else "--bins"
            g.tests[node] = TestTarget(node, "cargo", crate.name, (flag, "::".join(cur_mod)))
            # Unit tests also depend on how their crate is built.
            g.add_edge(node, str(crate.dir / "Cargo.toml"))
            if (root / crate.dir / "build.rs").exists():
                g.add_edge(node, str(crate.dir / "build.rs"))


def _resolve_use(
    path: list[str],
    crate: rsscan.Crate,
    cur_mod: tuple[str, ...],
    by_lib: dict[str, rsscan.Crate],
    modmap: dict[str, dict[tuple[str, ...], Path]],
) -> str | None:
    if not path:
        return None
    head, rest = path[0], path[1:]
    if head == "crate":
        target, base = crate, ()
    elif head in ("self", "super"):
        target, base = crate, cur_mod if head == "self" else cur_mod[:-1]
        while rest and rest[0] == "super":
            base, rest = base[:-1], rest[1:]
    elif head in by_lib:
        target, base = by_lib[head], ()
    else:
        return None  # external crate or std
    mm = modmap.get(target.name, {})
    full = (*base, *rest)
    for i in range(len(full), 0, -1):
        hit = mm.get(tuple(full[:i]))
        if hit is not None:
            return str(hit)
    if target is crate:
        return None
    return target.node


def _build_python(
    cfg: Config, g: Graph, py_files: list[Path], cache: _ParseCache, live: set[str]
) -> None:
    root = cfg.root
    modules: dict[str, str] = {}
    names: dict[Path, str] = {}
    for f in py_files:
        name = pyscan.module_name(f, root)
        names[f] = name
        # Prefer .py over .pyi stubs for import resolution.
        if f.suffix == ".py" or name not in modules:
            modules[name] = str(f)

    pyo3: dict[str, str] = {}
    for crate in g.crates.values():
        for m in crate.pyo3_modules:
            pyo3[m] = crate.node

    conftests: dict[Path, str] = {f.parent: str(f) for f in py_files if f.name == "conftest.py"}

    for f in py_files:
        node = str(f)
        name = names[f]
        live.add(node)
        st = (root / f).stat()
        hit = cache.get(node, st)
        if isinstance(hit, list):
            imported: set[str] = set(hit)
        else:
            try:
                src = (root / f).read_text(errors="replace")
            except OSError:
                src = ""
            imported = pyscan.imports(src, name, f.name == "__init__.py")
            cache.put(node, st, sorted(imported))

        for imp in imported:
            target = _resolve_py(imp, modules, pyo3)
            if target:
                g.add_edge(node, target)

        # Importing pkg.mod executes pkg/__init__.py.
        parts = name.split(".")
        for i in range(1, len(parts)):
            init = modules.get(".".join(parts[:i]))
            if init:
                g.add_edge(node, init)

        # Compiled extension stubs/shims stand in for the Rust crate.
        if name in pyo3 or parts[-1] in pyo3:
            g.add_edge(node, pyo3.get(name) or pyo3[parts[-1]])

        if pyscan.is_test_file(f) or any(f.match(glob) for glob in cfg.extra_test_globs):
            g.tests[node] = TestTarget(node, "pytest")
            d = f.parent
            while True:
                if d in conftests:
                    g.add_edge(node, conftests[d])
                if d == Path("."):
                    break
                d = d.parent


def _resolve_py(imp: str, modules: dict[str, str], pyo3: dict[str, str]) -> str | None:
    parts = imp.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in modules:
            return modules[prefix]
        if prefix in pyo3:
            return pyo3[prefix]
    return None
