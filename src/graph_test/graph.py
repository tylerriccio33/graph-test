"""Build a dependency graph across Python and Rust sources.

Node kinds:

- ``path/to/file.rs`` / ``path/to/module.py``: a Rust file, or a Python module's top-level code.
- ``path/to/module.py::qual``: a Python symbol (function, class, assigned name, test method).
- ``path/to/module.py::*``: "anything in this module", used when a reference can't be resolved.
- ``crate:<name>``: a Rust crate.
- ``data:<dir>``: a data directory referenced by Python code; data files are plain path nodes.
- ``path/to/file.rs#self``: just that Rust file's own text, excluding its ``mod`` children.
- ``path/to/conftest.py#self``: a conftest's own top-level code and hooks, not its imports.
- ``missing:<module>``: an internal import that doesn't resolve (e.g. a deleted module).

An edge ``a -> b`` means "a depends on b", so a change to ``b`` may affect ``a``.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph_test import pyscan, rsscan
from graph_test.config import Config

CACHE_VERSION = 4


@dataclass
class TestTarget:
    node: str
    kind: str  # "pytest" | "cargo"
    file: str = ""
    crate: str | None = None
    # For cargo: ("--test", "name") for integration tests, ("--lib", "a::b") for unit tests.
    selector: tuple[str, str] | None = None
    nodeid: str = ""  # pytest node id (file, or file::func, or file::Class::method)

    def command(self) -> list[str]:
        if self.kind == "pytest":
            return ["pytest", self.nodeid or self.node]
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
    py_info: dict[str, pyscan.ModuleInfo] = field(default_factory=dict)
    py_module: dict[str, str] = field(default_factory=dict)  # file -> dotted module name
    data_nodes: set[str] = field(default_factory=set)
    _rdeps: dict[str, set[str]] | None = None

    def add_node(self, node: str) -> None:
        self.deps.setdefault(node, set())

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
        self.data: dict[str, list[Any]] = {}
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
        ensure_state_dir(self.path.parent)
        self.path.write_text(json.dumps({"version": CACHE_VERSION, "files": self.data}))


STATE_DIR = ".graph-test"


def ensure_state_dir(path: Path) -> None:
    """Create graph-test's state dir, git-ignored so it never shows up as a change."""
    path.mkdir(parents=True, exist_ok=True)
    ignore = path / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


@dataclass
class _Walk:
    py: list[Path] = field(default_factory=list)
    rs: list[Path] = field(default_factory=list)
    manifests: list[Path] = field(default_factory=list)
    files: set[Path] = field(default_factory=set)
    dirs: set[Path] = field(default_factory=set)


def _walk(cfg: Config) -> _Walk:
    w = _Walk()
    excluded = set(cfg.exclude)
    for dirpath, dirnames, filenames in os.walk(cfg.root):
        dirnames[:] = [d for d in dirnames if d not in excluded and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(cfg.root)
        w.dirs.add(rel_dir)
        for fn in filenames:
            rel = rel_dir / fn
            w.files.add(rel)
            if fn.endswith((".py", ".pyi")):
                w.py.append(rel)
            elif fn.endswith(".rs"):
                w.rs.append(rel)
            elif fn == "Cargo.toml":
                w.manifests.append(rel)
    return w


def build(cfg: Config) -> Graph:
    root = cfg.root
    g = Graph()
    cache = _ParseCache(root / STATE_DIR / "parse-cache.json")
    w = _walk(cfg)
    live: set[str] = set()

    for m in w.manifests:
        crate = rsscan.load_crate(root, m)
        if crate:
            g.crates[crate.name] = crate
    _maturin_root_config(root, g)
    by_lib = {c.lib_name: c for c in g.crates.values()}
    for crate in g.crates.values():
        for dep in crate.deps:
            if dep in g.crates:
                g.add_edge(crate.node, g.crates[dep].node)
        for r in crate.roots:
            g.add_edge(crate.node, str(r))
        # A crate's own manifest is part of it.
        g.add_edge(crate.node, str(crate.dir / "Cargo.toml"))

    exports = _build_rust(cfg, g, w.rs, by_lib, cache, live)
    _PythonBuilder(cfg, g, w, cache, live, exports).build()
    cache.save(live)
    return g


def _maturin_root_config(root: Path, g: Graph) -> None:
    """A root pyproject's ``[tool.maturin]`` can name the module for a nested crate."""
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return
    maturin = data.get("tool", {}).get("maturin", {})
    module = maturin.get("module-name")
    if not module:
        return
    manifest = Path(maturin.get("manifest-path", "Cargo.toml"))
    for crate in g.crates.values():
        if crate.dir / "Cargo.toml" == manifest:
            crate.pyo3_modules.add(str(module))


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
        return rsscan.RustFile(**hit)
    try:
        src = (root / rel).read_text(errors="replace")
    except OSError:
        src = ""
    rf = rsscan.scan(src)
    cache.put(key, st, dict(rf.__dict__))
    return rf


# crate node -> python name -> rust files implementing it
Exports = dict[str, dict[str, set[str]]]


def _build_rust(
    cfg: Config,
    g: Graph,
    rs_files: list[Path],
    by_lib: dict[str, rsscan.Crate],
    cache: _ParseCache,
    live: set[str],
) -> Exports:
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

    exports: Exports = defaultdict(lambda: defaultdict(set))
    pyclass_names: dict[tuple[str, str], str] = {}  # (crate, rust struct) -> python name
    pymethod_files: list[tuple[str, str, str]] = []  # (crate, rust struct, file)

    for f, rf in scanned.items():
        crate = _owning_crate(f, crates)
        if crate is None:
            continue
        node = str(f)
        g.add_node(node)
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
                g.tests[node] = TestTarget(node, "cargo", node, crate.name, ("--test", f.stem))

        cur_mod = file_mod.get(f, ())
        for path in rf.uses:
            target = _resolve_use(path, crate, cur_mod, by_lib, modmap)
            if target:
                g.add_edge(node, target)

        if rf.has_tests and not is_aux_root and f in file_mod:
            flag = "--lib" if f in lib_files else "--bins"
            g.tests[node] = TestTarget(node, "cargo", node, crate.name, (flag, "::".join(cur_mod)))
            # Unit tests also depend on how their crate is built.
            g.add_edge(node, str(crate.dir / "Cargo.toml"))
            if (root / crate.dir / "build.rs").exists():
                g.add_edge(node, str(crate.dir / "build.rs"))

        for name in rf.pyfunctions:
            exports[crate.node][name].add(node)
        for rust_name, py_name in rf.pyclasses:
            exports[crate.node][py_name].add(node)
            pyclass_names[(crate.name, rust_name)] = py_name
        for rust_name in rf.pymethods:
            pymethod_files.append((crate.name, rust_name, node))

    for crate_name, rust_name, node in pymethod_files:
        py_name = pyclass_names.get((crate_name, rust_name), rust_name)
        exports[f"crate:{crate_name}"][py_name].add(node)
    return exports


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


_SKIP_FIXTURE_ARGS = frozenset({"self", "cls", "request"})
_GLOB_CHARS = "*?[{<%"


class _PythonBuilder:
    def __init__(
        self,
        cfg: Config,
        g: Graph,
        w: _Walk,
        cache: _ParseCache,
        live: set[str],
        exports: Exports,
    ) -> None:
        self.cfg, self.g, self.w, self.cache, self.live = cfg, g, w, cache, live
        self.exports = exports
        self.modules: dict[str, str] = {}  # dotted -> file
        self.infos: dict[str, pyscan.ModuleInfo] = {}  # file -> info
        self.pyo3: dict[str, str] = {}  # python module -> crate node
        for crate in g.crates.values():
            for m in crate.pyo3_modules:
                self.pyo3[m] = crate.node
        self.internal_tops: set[str] = set()
        self.conftests: dict[Path, str] = {}
        self.conftest_all = cfg.conftest_imports == "all"
        self.code_dirs: set[Path] = {Path(".")}

    # ------------------------------------------------------------------ setup

    def build(self) -> None:
        root = self.cfg.root
        for f in self.w.py:
            name = pyscan.module_name(f, root)
            self.g.py_module[str(f)] = name
            # Prefer .py over .pyi stubs for import resolution.
            if f.suffix == ".py" or name not in self.modules:
                self.modules[name] = str(f)
            self.internal_tops.add(name.split(".")[0])
            if f.name == "__init__.py":
                self.code_dirs.add(f.parent)
            if f.name == "conftest.py":
                self.conftests[f.parent] = str(f)
                self.g.add_node(f"{f}#self")
        for crate in self.g.crates.values():
            self.code_dirs.add(crate.dir)
            self.code_dirs.add(crate.dir / "src")

        for f in self.w.py:
            self.infos[str(f)] = self._analyze(f)
        self.g.py_info = self.infos
        for f in self.w.py:
            self._wire(f)

    def _analyze(self, f: Path) -> pyscan.ModuleInfo:
        node = str(f)
        self.live.add(node)
        st = (self.cfg.root / f).stat()
        hit = self.cache.get(node, st)
        if isinstance(hit, dict):
            return hit
        try:
            src = (self.cfg.root / f).read_text(errors="replace")
        except OSError:
            src = ""
        info = pyscan.analyze(src, self.g.py_module[node], f.name == "__init__.py")
        self.cache.put(node, st, info)
        return info

    # ------------------------------------------------------------------ resolution

    def _module_target(self, module: str) -> str | None:
        if module in self.modules:
            return self.modules[module]
        if module in self.pyo3:
            return self.pyo3[module]
        return None

    def _whole(self, file: str) -> str:
        """Node meaning "anything in this module"."""
        node = f"{file}::*"
        if node not in self.g.deps:
            self.g.add_edge(node, file)
            for qual in self.infos.get(file, {}).get("symbols", {}):
                self.g.add_edge(node, f"{file}::{qual}")
        return node

    def _resolve_name(self, module: str, name: str, depth: int = 0) -> str | None:
        """Node for ``module.name``: a symbol, submodule, re-export, or None if unknown."""
        sub = self._module_target(f"{module}.{name}")
        if sub:
            return sub
        file = self.modules.get(module)
        if file is None:
            return self.pyo3.get(module)
        info = self.infos[file]
        if name in info["symbols"]:
            return f"{file}::{name}"
        if depth < 10:
            imp = info["imports"].get(name)
            if imp:
                target_mod, target_name = imp
                if target_name:
                    return self._resolve_name(target_mod, target_name, depth + 1)
                return self._module_target(target_mod)
            for star in info["stars"]:
                hit = self._resolve_name(star, name, depth + 1)
                if hit:
                    return hit
        return None

    def _resolve_attr(self, module: str, rest: list[str]) -> str | None:
        while rest and f"{module}.{rest[0]}" in self.modules:
            module, rest = f"{module}.{rest[0]}", rest[1:]
        if not rest:
            return self._module_target(module)
        hit = self._resolve_name(module, rest[0])
        if hit:
            return hit
        file = self.modules.get(module)
        return self._whole(file) if file else self.pyo3.get(module)

    def _resolve_ref(self, file: str, info: pyscan.ModuleInfo, chain: str) -> str | None:
        head, *rest = chain.split(".")
        if head in info["symbols"]:
            return f"{file}::{head}"
        imp = info["imports"].get(head)
        if imp:
            module, name = imp
            if not name:
                return self._resolve_attr(module, rest)
            if self._module_target(f"{module}.{name}"):
                return self._resolve_attr(f"{module}.{name}", rest)
            hit = self._resolve_name(module, name)
            if hit:
                return hit
            mfile = self.modules.get(module)
            return self._whole(mfile) if mfile else self.pyo3.get(module)
        for star in info["stars"]:
            hit = self._resolve_name(star, head)
            if hit:
                return hit
        return None

    def _resolve_fixture(self, file: Path, name: str) -> str | None:
        sym = self.infos[str(file)]["symbols"].get(name)
        if sym and sym.get("fixture"):
            return f"{file}::{name}"
        d = file.parent
        while True:
            conf = self.conftests.get(d)
            if conf and conf != str(file):
                csym = self.infos[conf]["symbols"].get(name)
                if csym and csym.get("fixture"):
                    return f"{conf}::{name}"
            if d == Path("."):
                return None
            d = d.parent

    def _resolve_literal(self, file: Path, lit: str) -> str | None:
        """Map a path-like string literal to a data file or directory node."""
        text = lit.strip()
        if not text or text.startswith(("/", "http:", "https:", "~")):
            return None
        cut = min((text.index(c) for c in _GLOB_CHARS if c in text), default=len(text))
        text = text[:cut].removeprefix("./").rstrip("/")
        if not text or text in (".", ".."):
            return None
        rel = Path(text)
        # Bare names ("corpus") only resolve next to the file; paths also search upward.
        bases = [file.parent]
        if "/" in text or rel.suffix:
            bases += list(file.parent.parents)
        for base in bases:
            p = Path(os.path.normpath(base / rel))
            if p.parts and p.parts[0] == "..":
                continue
            if p in self.w.files:
                if p.suffix in (".py", ".pyi", ".rs"):
                    return None
                self.g.data_nodes.add(str(p))
                return str(p)
            if p in self.w.dirs:
                if p in self.code_dirs or file.is_relative_to(p):
                    return None
                node = f"data:{p}"
                self.g.data_nodes.add(node)
                return node
        return None

    # ------------------------------------------------------------------ wiring

    def _wire_refs(
        self, src: str, file: Path, info: pyscan.ModuleInfo, entry: dict[str, Any]
    ) -> None:
        for chain in entry.get("refs", ()):
            target = self._resolve_ref(str(file), info, chain)
            if target and target != src:
                self.g.add_edge(src, target)
        for lit in entry.get("lits", ()):
            target = self._resolve_literal(file, lit)
            if target:
                self.g.add_edge(src, target)

    def _wire(self, f: Path) -> None:
        g = self.g
        file = str(f)
        info = self.infos[file]
        module = g.py_module[file]
        g.add_node(file)
        is_test_file = pyscan.is_test_file(f) or any(
            f.match(glob) for glob in self.cfg.extra_test_globs
        )

        # Module-level code: its own references plus import side effects.
        self._wire_refs(file, f, info, info["module"])
        for m in info["modules"]:
            target = self._module_target(m)
            if target:
                g.add_edge(file, target)
            elif m.split(".")[0] in self.internal_tops:
                # Lets a deleted module still find its importers.
                g.add_edge(file, f"missing:{m}")
        parts = module.split(".")
        for i in range(1, len(parts)):
            init = self.modules.get(".".join(parts[:i]))
            if init:
                g.add_edge(file, init)

        stub_crate = self.pyo3.get(module) or self.pyo3.get(parts[-1])
        if stub_crate:
            # Importing the extension runs its #[pymodule] init: depend on the crate root's
            # own text ("#self"), not on every module it declares.
            for r in g.crates[stub_crate.removeprefix("crate:")].roots:
                g.add_edge(file, f"{r}#self")

        tests: list[tuple[str, str]] = []
        for qual, sym in info["symbols"].items():
            node = f"{file}::{qual}"
            g.add_node(node)
            self._wire_refs(node, f, info, sym)
            for arg in sym.get("args", ()):
                if arg not in _SKIP_FIXTURE_ARGS:
                    fx = self._resolve_fixture(f, arg)
                    if fx and fx != node:
                        g.add_edge(node, fx)
            if sym.get("parent"):
                g.add_edge(node, f"{file}::{sym['parent']}")
            for dispatcher in sym.get("registers", ()):
                # The dispatcher calls every registered implementation at runtime.
                target = self._resolve_ref(file, info, dispatcher)
                if target and target != node:
                    g.add_edge(target, node)
            if sym.get("autouse"):
                # Autouse fixtures run for every test that sees this module.
                g.add_edge(file, node)
                if f.name == "conftest.py":
                    g.add_edge(f"{file}#self", node)
            elif f.name == "conftest.py" and qual.startswith("pytest_") and self.conftest_all:
                g.add_edge(file, node)
            if stub_crate:
                # Stub symbol -> the Rust files defining it (#[pyfunction]/#[pyclass]/#[pymethods]).
                impls = self.exports.get(stub_crate, {}).get(qual.split(".")[0])
                for impl in impls or (stub_crate,):
                    g.add_edge(node, impl)

            if is_test_file and not sym.get("fixture"):
                name = qual.rpartition(".")[2]
                if (sym["kind"] == "func" and name.startswith("test")) or sym["kind"] == "method":
                    tests.append((qual, node))
                    g.add_edge(node, file)

        if not is_test_file:
            return
        if not tests:
            # Nothing recognisable: the whole file is one test, depending on all it imports.
            tests = [("", file)]
            for alias in info["imports"]:
                target = self._resolve_ref(file, info, alias)
                if target in self.infos:  # a whole module was imported
                    target = self._whole(target)
                if target:
                    g.add_edge(file, target)
        for qual, node in tests:
            nodeid = f"{file}::{qual.replace('.', '::')}" if qual else file
            g.tests[node] = TestTarget(node, "pytest", file, nodeid=nodeid)
        # Link to each enclosing conftest. By default only to its own text ("#self": top-level
        # code, hooks, autouse fixtures); requested fixtures are linked per test above. Going
        # through the conftest's module node would pull in everything it imports, so one
        # heavy import in a root conftest would select every test.
        d = f.parent
        while True:
            if d in self.conftests:
                conf = self.conftests[d]
                g.add_edge(file, conf if self.conftest_all else f"{conf}#self")
            if d == Path("."):
                break
            d = d.parent
