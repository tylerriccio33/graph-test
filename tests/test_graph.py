from __future__ import annotations

from pathlib import Path

from conftest import git

from graph_test import rsscan
from graph_test.cli import batch_commands, main
from graph_test.config import Config
from graph_test.graph import build
from graph_test.rank import rank


def ranked_nodes(repo: Path, changed: list[str]) -> list[str]:
    cfg = Config.load(repo)
    return [r.target.node for r in rank(cfg, build(cfg), dict.fromkeys(changed)).ranked]


def test_discovers_tests(repo: Path) -> None:
    g = build(Config.load(repo))
    assert set(g.tests) == {
        "crates/core/src/math.rs",
        "crates/core/tests/text_it.rs",
        "python/tests/test_util.py",
        "python/tests/test_fast.py",
        "python/tests/test_alone.py",
    }
    assert g.tests["crates/core/tests/text_it.rs"].command() == [
        "cargo",
        "test",
        "-p",
        "my-core",
        "--test",
        "text_it",
    ]
    assert g.tests["crates/core/src/math.rs"].command() == [
        "cargo",
        "test",
        "-p",
        "my-core",
        "--lib",
        "--",
        "math::",
    ]


def test_python_change_selects_only_importers(repo: Path) -> None:
    assert ranked_nodes(repo, ["python/pkg/util.py"]) == ["python/tests/test_util.py"]


def test_rust_change_crosses_pyo3_boundary(repo: Path) -> None:
    got = ranked_nodes(repo, ["crates/core/src/math.rs"])
    assert got[0] == "crates/core/src/math.rs"
    assert "python/tests/test_fast.py" in got
    assert "python/tests/test_util.py" not in got


def test_rust_use_resolves_to_specific_file(repo: Path) -> None:
    got = ranked_nodes(repo, ["crates/core/src/text/upper.rs"])
    assert got[0] == "crates/core/tests/text_it.rs"


def test_conftest_affects_sibling_tests(repo: Path) -> None:
    got = set(ranked_nodes(repo, ["python/tests/conftest.py"]))
    assert got == {
        "python/tests/test_util.py",
        "python/tests/test_fast.py",
        "python/tests/test_alone.py",
    }


def test_global_file_floors_everything(repo: Path) -> None:
    assert len(ranked_nodes(repo, ["Cargo.lock"])) == 5


def test_cochange_history(repo: Path) -> None:
    for i in range(3):
        (repo / "python/pkg/util.py").write_text(f"X = {i}\n")
        (repo / "python/tests/test_alone.py").write_text(f"import os  # {i}\n")
        git(repo, "commit", "-qam", f"c{i}")
    assert "python/tests/test_alone.py" in ranked_nodes(repo, ["python/pkg/util.py"])


def test_expand_use() -> None:
    assert rsscan._expand_use("crate::{a, b::{c, d}}") == [
        ["crate", "a"],
        ["crate", "b", "c"],
        ["crate", "b", "d"],
    ]


def test_batch_commands_groups_pytest(repo: Path) -> None:
    g = build(Config.load(repo))
    cmds = batch_commands(list(g.tests.values()))
    assert sum(c[0] == "pytest" for c in cmds) == 1


def test_cli_uses_git_diff(repo: Path, capsys) -> None:
    (repo / "python/pkg/util.py").write_text("def f():\n    return 2\n")
    assert main(["-C", str(repo), "rank", "--format", "cmd"]) == 0
    assert capsys.readouterr().out.strip() == "pytest python/tests/test_util.py"
