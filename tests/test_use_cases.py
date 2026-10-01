"""Numbered use cases: each asserts a concrete situation is (or isn't) caught for rerun.

UC 1   Changing a test file itself reruns that test (0 hops).
UC 2   Python: 1 hop - test imports the changed module.
UC 3   Python: 2 hops - test -> a -> b, b changes.
UC 4   Python: 3 hops - test -> a -> b -> c, c changes.
UC 5   Closer tests rank above farther ones.
UC 6   Python relative imports (`from . import x`, `from ..pkg import y`) are followed.
UC 7   `from pkg import submodule` resolves to the submodule file.
UC 8   Changing a package `__init__.py` reruns tests importing its submodules.
UC 9   `conftest.py` change reruns every test at or below its directory, not siblings above.
UC 10  Unrelated tests are NOT rerun.
UC 11  Rust: changing a file with `#[cfg(test)]` reruns its own unit tests.
UC 12  Rust: 1 hop - integration test `use`s the changed module directly.
UC 13  Rust: 2 hops - `mod` chain (lib.rs -> text/mod.rs -> upper.rs).
UC 14  Rust: cross-crate - integration test in crate B reaches a file in dependency crate A.
UC 15  Rust: `use super::` / `use crate::` inside a crate are followed.
UC 16  Rust: a crate's own Cargo.toml reruns that crate's tests only.
UC 17  Rust -> Python: change in a PyO3 crate's dependency reruns Python tests importing it.
UC 18  Global file (Cargo.lock / pyproject.toml) gives every test a floor score.
UC 19  Git co-change: a test repeatedly committed with a file is rerun with no import edge.
UC 20  Git co-change: a single shared commit is NOT enough.
UC 21  Git co-change: huge commits (reformat/merge) are ignored.
UC 22  CLI picks up uncommitted and untracked changes via git.
UC 23  Commands: Rust tests map to the right `cargo test` invocations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import git, init_repo, write_files

from graph_test.cli import main
from graph_test.config import Config
from graph_test.graph import build
from graph_test.rank import Ranked, rank

PY = {
    "pkg/__init__.py": "",
    "pkg/a.py": "from pkg import b\n",
    "pkg/b.py": "import pkg.c\n",
    "pkg/c.py": "X = 1\n",
    "pkg/sub/__init__.py": "",
    "pkg/sub/rel.py": "from . import leaf\nfrom ..c import X\n",
    "pkg/sub/leaf.py": "Y = 2\n",
    "other/__init__.py": "",
    "other/thing.py": "Z = 3\n",
    "tests/conftest.py": "",
    "tests/test_a.py": "from pkg.a import b\n",
    "tests/test_c.py": "from pkg.c import X\n",
    "tests/test_rel.py": "from pkg.sub import rel\n",
    "tests/test_other.py": "import other.thing\n",
    "tests/deep/conftest.py": "",
    "tests/deep/test_deep.py": "import os\n",
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
    "crates/pyext/src/lib.rs": "use my_core::math::add;\n",
    "python/pyx/__init__.py": "",
    "python/pyx/_native.pyi": "def add(a: int, b: int) -> int: ...\n",
    "python/pyx/fast.py": "from pyx import _native\n",
    "python/pyx/slow.py": "X = 1\n",
    "python/tests/test_fast.py": "from pyx.fast import _native\n",
    "python/tests/test_slow.py": "from pyx.slow import X\n",
    "pyproject.toml": "[project]\nname = 'x'\n",
}


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, PY)


@pytest.fixture
def rs_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, RS)


def selected(repo: Path, *changed: str, history: bool = True) -> dict[str, Ranked]:
    cfg = Config.load(repo)
    return {r.target.node: r for r in rank(cfg, build(cfg), list(changed), history).ranked}


def order(repo: Path, *changed: str) -> list[str]:
    return list(selected(repo, *changed))


# ---------------------------------------------------------------- Python


def test_uc01_changed_test_file_reruns_itself(py_repo: Path) -> None:
    got = selected(py_repo, "tests/test_other.py")
    assert got["tests/test_other.py"].distance == 0


def test_uc02_python_one_hop(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/a.py")
    assert got["tests/test_a.py"].distance == 1


def test_uc03_python_two_hops(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/b.py")
    assert got["tests/test_a.py"].distance == 2
    assert got["tests/test_a.py"].path == ["tests/test_a.py", "pkg/a.py", "pkg/b.py"]


def test_uc04_python_three_hops(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/c.py")
    assert got["tests/test_a.py"].distance == 3


def test_uc05_closer_tests_rank_higher(py_repo: Path) -> None:
    ranking = order(py_repo, "pkg/c.py")
    assert ranking.index("tests/test_c.py") < ranking.index("tests/test_a.py")


def test_uc06_relative_imports(py_repo: Path) -> None:
    assert "tests/test_rel.py" in selected(py_repo, "pkg/sub/leaf.py")  # from . import leaf
    assert "tests/test_rel.py" in selected(py_repo, "pkg/c.py")  # from ..c import X


def test_uc07_from_package_import_submodule(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/sub/rel.py")
    assert got["tests/test_rel.py"].distance == 1


def test_uc08_package_init_change(py_repo: Path) -> None:
    got = selected(py_repo, "pkg/__init__.py")
    assert {"tests/test_a.py", "tests/test_c.py", "tests/test_rel.py"} <= set(got)
    assert "tests/test_other.py" not in got


def test_uc09_conftest_scope(py_repo: Path) -> None:
    top = selected(py_repo, "tests/conftest.py")
    assert {"tests/test_a.py", "tests/test_other.py", "tests/deep/test_deep.py"} <= set(top)
    deep = selected(py_repo, "tests/deep/conftest.py")
    assert set(deep) == {"tests/deep/test_deep.py"}


def test_uc10_unrelated_tests_not_rerun(py_repo: Path) -> None:
    assert set(selected(py_repo, "other/thing.py")) == {"tests/test_other.py"}


# ---------------------------------------------------------------- Rust


def test_uc11_rust_unit_tests_in_changed_file(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/math.rs")
    assert got["crates/core/src/math.rs"].distance == 0


def test_uc12_rust_one_hop_use(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/text/upper.rs")
    assert got["crates/core/tests/text_it.rs"].distance == 1


def test_uc13_rust_two_hop_mod_chain(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/text/upper.rs", history=False)
    # lib.rs -> text/mod.rs -> upper.rs, so anything depending on the crate root is >= 2 hops
    g = build(Config.load(rs_repo))
    reach = g.dependents(["crates/core/src/text/upper.rs"])
    assert reach["crates/core/src/text/mod.rs"][0] == 1
    assert reach["crates/core/src/lib.rs"][0] == 2
    assert "crates/app/tests/app_it.rs" in got


def test_uc14_rust_cross_crate(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/math.rs")
    assert "crates/app/tests/app_it.rs" in got
    assert "crates/lonely/src/lib.rs" not in got


def test_uc15_rust_crate_and_super_paths(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/util.rs")
    # math.rs has `use crate::util::helper` -> its unit tests are 1 hop away
    assert got["crates/core/src/math.rs"].distance == 1


def test_uc16_crate_manifest_scoped(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/lonely/Cargo.toml")
    assert set(got) == {"crates/lonely/src/lib.rs"}


def test_uc17_rust_change_reaches_python_via_pyo3(rs_repo: Path) -> None:
    got = selected(rs_repo, "crates/core/src/math.rs")
    assert "python/tests/test_fast.py" in got
    assert "python/tests/test_slow.py" not in got
    assert "crate:pyext" in got["python/tests/test_fast.py"].path


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
        _commit(py_repo, {"other/thing.py": f"Z = {i}\n", "tests/test_c.py": f"# {i}\n"}, f"c{i}")
    got = selected(py_repo, "other/thing.py")
    assert got["tests/test_c.py"].cochange_with == "other/thing.py"
    assert got["tests/test_c.py"].distance is None


def test_uc20_single_cochange_ignored(py_repo: Path) -> None:
    _commit(py_repo, {"other/thing.py": "Z = 9\n", "tests/test_c.py": "# x\n"}, "once")
    assert "tests/test_c.py" not in selected(py_repo, "other/thing.py")


def test_uc21_huge_commits_ignored(py_repo: Path) -> None:
    for i in range(3):
        noise = {f"noise/f{j}.txt": str(i) for j in range(60)}
        _commit(
            py_repo, {**noise, "other/thing.py": f"Z={i}\n", "tests/test_c.py": f"#{i}\n"}, "big"
        )
    assert "tests/test_c.py" not in selected(py_repo, "other/thing.py")


def test_uc22_cli_sees_uncommitted_and_untracked(py_repo: Path, capsys) -> None:
    write_files(py_repo, {"pkg/c.py": "X = 2\n", "tests/test_new.py": "import pkg.c\n"})
    assert main(["-C", str(py_repo), "rank", "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert '"pkg/c.py"' in out
    assert '"tests/test_new.py"' in out


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
