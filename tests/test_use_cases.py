"""Numbered use cases: each asserts a concrete situation is (or isn't) caught for rerun.

Hops are counted in the symbol graph: test function -> function it calls -> ...

Python, call chains
UC 1   Changing a test file reruns its tests (0 hops).
UC 2   1 hop  - test calls the changed function.
UC 3   2 hops - test -> f -> g, g changes.
UC 4   3 hops - test -> f -> g -> h, h changes.
UC 5   Closer tests rank above farther ones.
UC 6   Relative imports (`from . import x`, `from ..c import y`) are followed.
UC 7   `from pkg import submodule` + `submodule.func()` resolves to the function.
UC 8   Package `__init__.py` change reruns tests importing that package (import-time).
UC 9   `conftest.py` change reruns tests at or below its directory only.
UC 10  Unrelated tests are NOT rerun.

Rust
UC 11  Changing a file with `#[cfg(test)]` reruns its own unit tests.
UC 12  1 hop - integration test `use`s the changed module directly.
UC 13  2 hops - `mod` chain (lib.rs -> text/mod.rs -> upper.rs).
UC 14  Cross-crate - integration test in crate B reaches a file in dependency crate A.
UC 15  `use crate::` inside a crate is followed.
UC 16  A crate's own Cargo.toml reruns that crate's tests only.
UC 17  Rust -> Python through PyO3, per symbol: only Python tests calling the affected
       #[pyfunction] rerun, not every test importing the extension.

Global / history / CLI
UC 18  Global file (Cargo.lock / pyproject.toml) gives every test a floor score.
UC 19  Git co-change: a test repeatedly committed with a file is ranked with no import edge.
UC 20  Git co-change: a single shared commit is NOT enough.
UC 21  Git co-change: huge commits (reformat/merge) are ignored.
UC 22  CLI picks up uncommitted and untracked changes via git.
UC 23  Rust tests map to the right `cargo test` invocations.

Precision (symbol + line level)
UC 24  Editing one function's body reruns tests reaching that function, not the module's
       other users.
UC 25  Editing one test function reruns only that test, not its siblings in the file.
UC 26  Re-exports through an `__init__.py` hub resolve to the defining module, so a
       change elsewhere in the package doesn't fan out to the hub's users.
UC 27  Comment/blank/docstring-only edits select nothing and aren't "unmapped".
UC 28  Changing module-level code (e.g. imports) reruns importers (import-time effects).
UC 29  Pytest fixtures resolve through conftest: tests using the fixture rerun, others don't.
UC 30  Deleting a module reruns tests that imported it.

Data files
UC 31  A data directory referenced as `Path(__file__).parent / "corpus"` is tracked; only
       that test reruns, narrowed to the changed case files.
UC 32  A repo-relative data file literal ("tests/fixtures/input.txt") is tracked.
UC 33  `[tool.graph-test.data]` config maps file patterns to tests explicitly.

Gate mode
UC 34  `run` executes every affected test and returns its pass/fail exit code.
UC 35  Unaffected failing tests don't fail the gate.
UC 36  An unmapped changed file makes `run` run everything (never silently skip);
       `rank` only warns.
UC 37  Ignored files (README.md, docs/) select nothing and aren't unmapped.
UC 38  The pytest plugin runs only the parametrized cases whose data file changed.

Regressions (GitHub issue #1)
UC 39  A diff containing non-UTF-8 bytes (latin-1 corpus) doesn't crash.
UC 40  A root conftest importing heavy code for a hook doesn't link every test to that code.
UC 41  ...but editing that conftest's hook or top-level code still reruns everything under it.
UC 42  `conftest-imports = "all"` restores the conservative behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import git, init_repo, write_files

from graph_test import gitutil
from graph_test.cli import main
from graph_test.config import Config
from graph_test.graph import build
from graph_test.rank import Ranked, rank

PY = {
    "pkg/__init__.py": "from pkg.c import X\nfrom pkg.b import g\n",
    "pkg/a.py": "from pkg.b import g\n\n\ndef f():\n    return g()\n",
    "pkg/b.py": "from pkg.c import h\n\n\ndef g():\n    return h()\n",
    "pkg/c.py": '"""Leaf module."""\n\n\ndef h():\n    return 1\n\n\nX = 1\n',
    "pkg/sub/__init__.py": "",
    "pkg/sub/rel.py": (
        "from . import leaf\nfrom ..c import X\n\n\ndef r():\n    return leaf.Y + X\n"
    ),
    "pkg/sub/leaf.py": "Y = 2\n",
    "other/__init__.py": "",
    "other/thing.py": "Z = 3\n",
    "tests/conftest.py": "",
    "tests/test_a.py": "from pkg.a import f\n\n\ndef test_f():\n    assert f()\n",
    "tests/test_c.py": (
        "from pkg.c import X, h\n\n\ndef test_x():\n    assert X\n\n\n"
        "def test_h():\n    assert h()\n"
    ),
    "tests/test_rel.py": "from pkg.sub import rel\n\n\ndef test_rel():\n    assert rel.r()\n",
    "tests/test_other.py": "import other.thing\n\n\ndef test_o():\n    assert other.thing.Z\n",
    "tests/test_hub.py": "from pkg import X\n\n\ndef test_hub():\n    assert X\n",
    "tests/deep/conftest.py": "",
    "tests/deep/test_deep.py": "def test_deep():\n    assert True\n",
    "tests/fx/conftest.py": (
        "import pytest\nfrom pkg.c import h\n\n\n@pytest.fixture\ndef hval():\n    return h()\n"
    ),
    "tests/fx/test_fx.py": (
        "def test_uses_fixture(hval):\n    assert hval\n\n\ndef test_plain():\n    assert True\n"
    ),
}

RS = {
    "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
    "Cargo.lock": "",
    "crates/core/Cargo.toml": '[package]\nname = "my-core"\nversion = "0.1.0"\n',
    "crates/core/src/lib.rs": "pub mod math;\npub mod text;\npub mod util;\n",
    "crates/core/src/math.rs": (
        "use crate::util::helper;\n"
        "pub fn add(a: i32, b: i32) -> i32 { helper(a) + b }\n"
        "#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn t() {}\n}\n"
    ),
    "crates/core/src/util.rs": "pub fn helper(x: i32) -> i32 { x }\n",
    "crates/core/src/text/mod.rs": "pub mod upper;\n",
    "crates/core/src/text/upper.rs": "pub fn up(s: &str) -> String { s.to_uppercase() }\n",
    "crates/core/tests/text_it.rs": "use my_core::text::upper::up;\n#[test]\nfn it() {}\n",
    "crates/app/Cargo.toml": (
        '[package]\nname = "app"\nversion = "0.1.0"\n'
        '[dependencies]\nmy-core = { path = "../core" }\n'
    ),
    "crates/app/src/lib.rs": "use my_core::math::add;\npub fn run() -> i32 { add(1, 2) }\n",
    "crates/app/tests/app_it.rs": "use app::run;\n#[test]\nfn it() {}\n",
    "crates/lonely/Cargo.toml": '[package]\nname = "lonely"\nversion = "0.1.0"\n',
    "crates/lonely/src/lib.rs": "#[cfg(test)]\nmod tests { #[test] fn t() {} }\n",
    "crates/pyext/Cargo.toml": (
        '[package]\nname = "pyext"\nversion = "0.1.0"\n[lib]\nname = "_native"\n'
        '[dependencies]\npyo3 = "0.22"\nmy-core = { path = "../core" }\n'
    ),
    "crates/pyext/src/lib.rs": (
        "use pyo3::prelude::*;\nmod funcs;\nmod text;\n\n"
        "#[pymodule]\nfn _native(m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }\n"
    ),
    "crates/pyext/src/funcs.rs": (
        "use my_core::math::add as core_add;\n"
        "#[pyfunction]\npub fn add(a: i32, b: i32) -> i32 { core_add(a, b) }\n"
    ),
    "crates/pyext/src/text.rs": (
        "use my_core::text::upper::up;\n"
        '#[pyfunction]\n#[pyo3(name = "shout")]\npub fn shout_impl(s: &str) -> String { up(s) }\n'
    ),
    "python/pyx/__init__.py": "",
    "python/pyx/_native.pyi": (
        "def add(a: int, b: int) -> int: ...\ndef shout(s: str) -> str: ...\n"
    ),
    "python/pyx/slow.py": "X = 1\n",
    "python/tests/test_fast.py": (
        "from pyx import _native\n\n\ndef test_add():\n    assert _native.add(1, 2)\n\n\n"
        "def test_shout():\n    assert _native.shout('a')\n"
    ),
    "python/tests/test_slow.py": "from pyx.slow import X\n\n\ndef test_slow():\n    assert X\n",
    "pyproject.toml": "[project]\nname = 'x'\n",
}

CORPUS_TEST = """\
from pathlib import Path

import pytest
from pkg.parse import parse

HERE = Path(__file__).parent
CASES = sorted((HERE / "corpus").glob("*.sas"))


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.stem)
def test_corpus(case):
    expected = HERE / "expected" / f"{case.stem}.csv"
    assert parse(case.read_text()) == expected.read_text()


def test_unrelated():
    assert parse("x") == "x"
"""

DATA = {
    "pytest.ini": "[pytest]\npythonpath = .\naddopts = -p no:cacheprovider\n",
    "pyproject.toml": (
        "[tool.graph-test]\nglobal-files = []\n"
        '[tool.graph-test.data]\n"golden/*" = ["tests/test_input.py::*"]\n'
    ),
    "pkg/__init__.py": "",
    "pkg/parse.py": "def parse(text):\n    return text\n",
    "pkg/other.py": "def o():\n    return 1\n",
    "golden/z.txt": "z",
    "tests/corpus/a.sas": "a",
    "tests/corpus/b.sas": "b",
    "tests/expected/a.csv": "a",
    "tests/expected/b.csv": "b",
    "tests/fixtures/input.txt": "hello",
    "tests/test_corpus.py": CORPUS_TEST,
    "tests/test_input.py": (
        'def test_input():\n    assert open("tests/fixtures/input.txt").read() == "hello"\n'
    ),
    "tests/test_broken.py": (
        "from pkg.other import o\n\n\ndef test_broken():\n    assert o() == 1\n"
    ),
    "tools/script.sh": "echo hi\n",
    "README.md": "readme\n",
}

PYTEST = f"{sys.executable} -m pytest -q"


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, PY)


@pytest.fixture
def rs_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, RS)


@pytest.fixture
def data_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, DATA)


def selected(repo: Path, *changed: str, history: bool = True) -> dict[str, Ranked]:
    """Rank with whole-file changes (as with ``--files``)."""
    cfg = Config.load(repo)
    return {r.target.node: r for r in rank(cfg, build(cfg), dict.fromkeys(changed), history).ranked}


def selected_from_git(repo: Path) -> dict[str, Ranked]:
    """Rank using the real line-level diff of the working tree."""
    cfg = Config.load(repo)
    changes = gitutil.changed_hunks(repo, "main")
    return {r.target.node: r for r in rank(cfg, build(cfg), changes, use_history=False).ranked}


def edit(repo: Path, rel: str, old: str, new: str) -> None:
    p = repo / rel
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new))


# ---------------------------------------------------------------- Python call chains


def test_uc01_changed_test_file_reruns_itself(py_repo: Path) -> None:
    got = selected(py_repo, "tests/test_other.py")
    assert got["tests/test_other.py::test_o"].distance == 0


def test_uc02_python_one_hop(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/a.py")
    assert got["tests/test_a.py::test_f"].distance == 1


def test_uc03_python_two_hops(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/b.py")
    assert got["tests/test_a.py::test_f"].distance == 2
    assert got["tests/test_a.py::test_f"].path == [
        "tests/test_a.py::test_f",
        "pkg/a.py::f",
        "pkg/b.py::g",
    ]


def test_uc04_python_three_hops(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/c.py")
    assert got["tests/test_a.py::test_f"].distance == 3


def test_uc05_closer_tests_rank_higher(py_repo: Path) -> None:
    ranking = list(selected(py_repo, "pkg/c.py"))
    assert ranking.index("tests/test_c.py::test_h") < ranking.index("tests/test_a.py::test_f")


def test_uc06_relative_imports(py_repo: Path) -> None:
    assert "tests/test_rel.py::test_rel" in selected(py_repo, "pkg/sub/leaf.py")
    assert "tests/test_rel.py::test_rel" in selected(py_repo, "pkg/c.py")


def test_uc07_from_package_import_submodule(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/sub/rel.py")
    assert got["tests/test_rel.py::test_rel"].distance == 1


def test_uc08_package_init_change(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/__init__.py")
    assert {"tests/test_a.py::test_f", "tests/test_hub.py::test_hub"} <= set(got)
    assert "tests/test_other.py::test_o" not in got
    assert "tests/deep/test_deep.py::test_deep" not in got


def test_uc09_conftest_scope(py_repo: Path) -> None:
    top = selected(py_repo, "tests/conftest.py")
    assert {"tests/test_a.py::test_f", "tests/deep/test_deep.py::test_deep"} <= set(top)
    deep = selected(py_repo, "tests/deep/conftest.py")
    assert set(deep) == {"tests/deep/test_deep.py::test_deep"}


def test_uc10_unrelated_tests_not_rerun(py_repo: Path) -> None:
    assert set(selected(py_repo, "other/thing.py")) == {"tests/test_other.py::test_o"}


# ---------------------------------------------------------------- Rust


def test_uc11_rust_unit_tests_in_changed_file(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/math.rs")
    assert got["crates/core/src/math.rs"].distance == 0


def test_uc12_rust_one_hop_use(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/text/upper.rs")
    assert got["crates/core/tests/text_it.rs"].distance == 1


def test_uc13_rust_two_hop_mod_chain(rs_repo: Path) -> None:
    reach = build(Config.load(rs_repo)).dependents(["crates/core/src/text/upper.rs"])
    assert reach["crates/core/src/text/mod.rs"][0] == 1
    assert reach["crates/core/src/lib.rs"][0] == 2
    assert "crates/app/tests/app_it.rs" in selected(rs_repo, "crates/core/src/text/upper.rs")


def test_uc14_rust_cross_crate(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/math.rs")
    assert "crates/app/tests/app_it.rs" in got
    assert "crates/lonely/src/lib.rs" not in got


def test_uc15_rust_crate_paths(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/util.rs")
    assert got["crates/core/src/math.rs"].distance == 1


def test_uc16_crate_manifest_scoped(rs_repo: Path) -> None:
    assert set(selected(rs_repo, "crates/lonely/Cargo.toml")) == {"crates/lonely/src/lib.rs"}


def test_uc17_rust_to_python_per_pyfunction(rs_repo: Path) -> None:
    math = selected(rs_repo, "crates/core/src/math.rs")
    assert "python/tests/test_fast.py::test_add" in math
    assert "python/tests/test_fast.py::test_shout" not in math
    assert "python/tests/test_slow.py::test_slow" not in math
    upper = selected(rs_repo, "crates/core/src/text/upper.rs")
    assert "python/tests/test_fast.py::test_shout" in upper  # via #[pyo3(name = "shout")]
    assert "python/tests/test_fast.py::test_add" not in upper
    # Editing the #[pymodule] file itself affects everything importing the extension.
    root = selected(rs_repo, "crates/pyext/src/lib.rs")
    assert {"python/tests/test_fast.py::test_add", "python/tests/test_fast.py::test_shout"} <= set(
        root
    )


# ---------------------------------------------------------------- global / history / CLI


@pytest.mark.parametrize("f", ["Cargo.lock", "pyproject.toml"])
def test_uc18_global_file_floor(rs_repo: Path, f: str) -> None:
    got = selected(rs_repo, f)
    assert len(got) == len(build(Config.load(rs_repo)).tests)
    assert all(r.global_hit for r in got.values())


def _commit(repo: Path, files: dict[str, str], msg: str) -> None:
    write_files(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def test_uc19_cochange_without_import(py_repo: Path) -> None:
    for i in range(3):
        _commit(
            py_repo,
            {"other/thing.py": f"Z = {i}\n", "tests/test_c.py": PY["tests/test_c.py"] + f"# {i}\n"},
            f"c{i}",
        )
    got = selected(py_repo, "other/thing.py")
    assert got["tests/test_c.py::test_x"].cochange_with == "other/thing.py"
    assert got["tests/test_c.py::test_x"].distance is None


def test_uc20_single_cochange_ignored(py_repo: Path) -> None:
    _commit(
        py_repo,
        {"other/thing.py": "Z = 9\n", "tests/test_c.py": PY["tests/test_c.py"] + "#\n"},
        "x",
    )
    assert "tests/test_c.py::test_x" not in selected(py_repo, "other/thing.py")


def test_uc21_huge_commits_ignored(py_repo: Path) -> None:
    for i in range(3):
        noise = {f"noise/f{j}.txt": str(i) for j in range(60)}
        test_c = PY["tests/test_c.py"] + f"# {i}\n"
        _commit(py_repo, {**noise, "other/thing.py": f"Z={i}\n", "tests/test_c.py": test_c}, "big")
    assert "tests/test_c.py::test_x" not in selected(py_repo, "other/thing.py")


def test_uc22_cli_sees_uncommitted_and_untracked(py_repo: Path, capsys) -> None:
    edit(py_repo, "pkg/c.py", "return 1", "return 2")
    write_files(py_repo, {"tests/test_new.py": "def test_new():\n    assert True\n"})
    assert main(["-C", str(py_repo), "rank", "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert '"pkg/c.py"' in out
    assert '"tests/test_new.py::test_new"' in out


def test_uc23_cargo_commands(rs_repo: Path) -> None:
    g = build(Config.load(rs_repo))
    assert g.tests["crates/core/tests/text_it.rs"].command()[-2:] == ["--test", "text_it"]
    assert g.tests["crates/core/src/math.rs"].command()[-3:] == ["--lib", "--", "math::"]
    assert g.tests["crates/lonely/src/lib.rs"].command() == [
        "cargo",
        "test",
        "-p",
        "lonely",
        "--lib",
    ]


# ---------------------------------------------------------------- precision


def test_uc24_function_body_edit_is_precise(py_repo: Path) -> None:
    edit(py_repo, "pkg/c.py", "return 1", "return 2")  # inside h()
    got = selected_from_git(py_repo)
    assert {"tests/test_c.py::test_h", "tests/test_a.py::test_f"} <= set(got)
    assert "tests/fx/test_fx.py::test_uses_fixture" in got  # fixture calls h()
    assert "tests/test_c.py::test_x" not in got  # same module, uses X only
    assert "tests/test_hub.py::test_hub" not in got


def test_uc25_single_test_edit(py_repo: Path) -> None:
    edit(py_repo, "tests/test_c.py", "assert X", "assert X == 1")
    assert set(selected_from_git(py_repo)) == {"tests/test_c.py::test_x"}


def test_uc26_reexport_hub_does_not_fan_out(py_repo: Path) -> None:
    g = build(Config.load(py_repo))
    # `from pkg import X` resolves through pkg/__init__.py to the defining module.
    assert "pkg/c.py::X" in g.deps["tests/test_hub.py::test_hub"]
    edit(py_repo, "pkg/b.py", "return h()", "return h() + 0")
    got = selected_from_git(py_repo)
    assert "tests/test_a.py::test_f" in got
    assert "tests/test_hub.py::test_hub" not in got


def test_uc27_comment_only_edit(py_repo: Path) -> None:
    edit(py_repo, "pkg/c.py", "def h():\n", "# a note\ndef h():\n")
    edit(py_repo, "pkg/c.py", '"""Leaf module."""', '"""Leaf module, documented."""')
    cfg = Config.load(py_repo)
    result = rank(cfg, build(cfg), gitutil.changed_hunks(py_repo, "main"), use_history=False)
    assert result.ranked == []
    assert result.unmapped == []


def test_uc28_module_level_change_reaches_importers(py_repo: Path) -> None:
    edit(py_repo, "pkg/c.py", "X = 1\n", "X = 1\nprint('import side effect')\n")
    got = selected_from_git(py_repo)
    assert {"tests/test_a.py::test_f", "tests/test_hub.py::test_hub"} <= set(got)
    assert "tests/test_other.py::test_o" not in got


def test_uc29_fixture_through_conftest(py_repo: Path) -> None:
    edit(py_repo, "pkg/c.py", "return 1", "return 3")
    got = selected_from_git(py_repo)
    assert "tests/fx/test_fx.py::test_uses_fixture" in got
    assert "tests/fx/test_fx.py::test_plain" not in got


def test_uc30_deleted_module(py_repo: Path) -> None:
    git(py_repo, "rm", "-q", "pkg/sub/leaf.py")
    got = selected_from_git(py_repo)
    assert "tests/test_rel.py::test_rel" in got
    assert "tests/test_other.py::test_o" not in got


# ---------------------------------------------------------------- data files


def test_uc31_data_dir_narrowed_to_cases(data_repo: Path) -> None:
    write_files(data_repo, {"tests/corpus/a.sas": "a2"})
    got = selected_from_git(data_repo)
    assert set(got) == {"tests/test_corpus.py::test_corpus"}
    assert got["tests/test_corpus.py::test_corpus"].cases == ["tests/corpus/a.sas"]
    write_files(data_repo, {"tests/expected/b.csv": "b2"})
    assert (
        "tests/expected/b.csv"
        in selected_from_git(data_repo)["tests/test_corpus.py::test_corpus"].cases
    )


def test_uc32_repo_relative_data_file(data_repo: Path) -> None:
    write_files(data_repo, {"tests/fixtures/input.txt": "bye"})
    assert set(selected_from_git(data_repo)) == {"tests/test_input.py::test_input"}


def test_uc33_configured_data_mapping(data_repo: Path) -> None:
    write_files(data_repo, {"golden/z.txt": "zz"})
    assert set(selected_from_git(data_repo)) == {"tests/test_input.py::test_input"}


# ---------------------------------------------------------------- gate mode


def test_uc34_gate_runs_affected_and_fails(data_repo: Path, capfd) -> None:
    edit(data_repo, "pkg/other.py", "return 1", "return 2")
    rc = main(["-C", str(data_repo), "run", "--no-history", "--pytest", PYTEST])
    out = capfd.readouterr().out
    assert rc != 0
    assert "test_broken" in out
    assert "test_corpus" not in out


def test_uc35_unaffected_failures_ignored(data_repo: Path, capfd) -> None:
    edit(data_repo, "pkg/other.py", "return 1", "return 2")
    git(data_repo, "commit", "-qam", "break other")  # test_broken now fails, but is committed
    edit(data_repo, "pkg/parse.py", "return text", "return str(text)")
    rc = main(["-C", str(data_repo), "run", "--base", "HEAD", "--no-history", "--pytest", PYTEST])
    out = capfd.readouterr().out
    assert rc == 0, out
    assert "test_broken" not in out


def test_uc36_unmapped_runs_everything_in_gate(data_repo: Path, capfd) -> None:
    write_files(data_repo, {"tools/script.sh": "echo bye\n"})
    assert main(["-C", str(data_repo), "run", "--dry-run", "--no-history"]) == 0
    out = capfd.readouterr().out
    assert "running everything" in out
    assert "tools/script.sh" in out
    assert main(["-C", str(data_repo), "rank", "--affected", "--no-history"]) == 0
    out = capfd.readouterr().out
    assert "running everything" not in out
    assert "not in graph: tools/script.sh" in out


def test_uc37_ignored_files(data_repo: Path) -> None:
    write_files(data_repo, {"README.md": "changed\n", "docs/x.md": "new\n"})
    cfg = Config.load(data_repo)
    result = rank(cfg, build(cfg), gitutil.changed_hunks(data_repo, "main"), use_history=False)
    assert result.ranked == []
    assert result.unmapped == []
    assert set(result.ignored) == {"README.md", "docs/x.md"}


def test_uc38_plugin_runs_only_changed_cases(data_repo: Path, capfd) -> None:
    write_files(data_repo, {"tests/corpus/a.sas": "a2", "tests/expected/a.csv": "a2"})
    rc = main(
        ["-C", str(data_repo), "run", "--no-history", "--pytest", f"{sys.executable} -m pytest -v"]
    )
    out = capfd.readouterr().out
    assert rc == 0, out
    assert "test_corpus[a] PASSED" in out
    assert "test_corpus[b]" not in out
    assert "1 deselected" in out


# ---------------------------------------------------------------- regressions (issue #1)


def test_uc39_non_utf8_diff(data_repo: Path) -> None:
    (data_repo / "tests/corpus/a.sas").write_bytes("caf\xe9 \xba\n".encode("latin-1"))
    got = selected_from_git(data_repo)
    assert got["tests/test_corpus.py::test_corpus"].cases == ["tests/corpus/a.sas"]


HEAVY_CONFTEST = {
    "tests/conftest.py": (
        "import heavy.suites\n\n\ndef pytest_sessionstart(session):\n    heavy.suites.load()\n"
    ),
    "heavy/__init__.py": "",
    "heavy/suites.py": "from pkg.c import h\n\nREGISTRY = h()\n\n\ndef load():\n    return h()\n",
}


@pytest.fixture
def heavy_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, {**PY, **HEAVY_CONFTEST})


def test_uc40_conftest_imports_do_not_fan_out(heavy_repo: Path) -> None:
    edit(heavy_repo, "pkg/c.py", "return 1", "return 2")
    got = selected_from_git(heavy_repo)
    assert "tests/test_c.py::test_h" in got
    assert "tests/test_other.py::test_o" not in got
    assert "tests/deep/test_deep.py::test_deep" not in got


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("    heavy.suites.load()\n", "    heavy.suites.load()\n    print('hook')\n"),  # hook body
        ("import heavy.suites\n", "import heavy.suites\nimport os\n"),  # top-level code
    ],
)
def test_uc41_conftest_own_edits_still_select_all(heavy_repo: Path, old: str, new: str) -> None:
    edit(heavy_repo, "tests/conftest.py", old, new)
    got = selected_from_git(heavy_repo)
    assert {"tests/test_other.py::test_o", "tests/deep/test_deep.py::test_deep"} <= set(got)


def test_uc42_conftest_imports_all_mode(heavy_repo: Path) -> None:
    write_files(heavy_repo, {"pyproject.toml": '[tool.graph-test]\nconftest-imports = "all"\n'})
    git(heavy_repo, "add", "-A")
    git(heavy_repo, "commit", "-qm", "config")
    edit(heavy_repo, "pkg/c.py", "return 1", "return 2")
    assert "tests/test_other.py::test_o" in selected_from_git(heavy_repo)
