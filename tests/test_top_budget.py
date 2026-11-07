"""`-n N` bounds the pytest items that run, parametrized cases included, and says so."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import init_repo

from graph_test.cli import main, pytest_budget
from graph_test.graph import TestTarget as Target
from graph_test.rank import Ranked

PYTEST = f"{sys.executable} -m pytest -q"
FILES = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def f():\n    return 1\n",
    "tests/test_heavy.py": (
        "import pytest\nfrom pkg.core import f\n\n\n"
        '@pytest.mark.parametrize("n", range(50))\ndef test_heavy(n):\n    assert f()\n'
    ),
    "tests/test_light.py": (
        "from pkg.core import f\n\n\n"
        "def test_one():\n    assert f()\n\n\ndef test_two():\n    assert f()\n"
    ),
    "tests/conftest.py": (
        "import sys\nfrom pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
    ),
}
CHANGED = ["--no-history", "--files", "pkg/core.py"]


@pytest.fixture
def budget_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, FILES)


def run(repo: Path, *args: str) -> int:
    return main(["-C", str(repo), "run", *args, *CHANGED])


@pytest.mark.parametrize(
    ("top", "heavy", "light"),
    [(3, 1, 2), (10, 8, 2)],
    ids=["one-case-per-function", "spare-budget-goes-to-cases"],
)
def test_run_top_bounds_items(budget_repo: Path, capfd, top: int, heavy: int, light: int) -> None:
    assert run(budget_repo, "-n", str(top), "--pytest", f"{sys.executable} -m pytest -v") == 0
    out = capfd.readouterr().out
    assert out.count("test_heavy[") == heavy
    assert out.count("PASSED") == top
    assert out.count("test_one PASSED") + out.count("test_two PASSED") == light
    assert f"kept {top} of 52 cases from 3 of 3 tests; {52 - top} did not run" in out


def test_budget_big_enough_reports_nothing(budget_repo: Path, capfd) -> None:
    assert run(budget_repo, "-n", "100", "--pytest", PYTEST) == 0
    out = capfd.readouterr().out
    assert "52 passed" in out
    assert "did not run" not in out


def test_no_top_is_unchanged(budget_repo: Path, capfd) -> None:
    assert run(budget_repo, "--pytest", PYTEST) == 0
    out = capfd.readouterr().out
    assert "52 passed" in out
    assert "-p graph_test" not in out
    assert "budget" not in out
    assert "budget" not in (budget_repo / ".graph-test/selection.json").read_text()


def test_budget_names_itself_on_dry_run(budget_repo: Path, capfd) -> None:
    assert run(budget_repo, "-n", "3", "--dry-run") == 0
    out = capfd.readouterr().out
    assert "at most 3 pytest item(s)" in out
    assert "pytest -p graph_test tests/" in out
    sel = json.loads((budget_repo / ".graph-test/selection.json").read_text())
    assert sel["budget"] == 3
    assert sel["order"] == [
        "tests/test_heavy.py::test_heavy",
        "tests/test_light.py::test_one",
        "tests/test_light.py::test_two",
    ]


def test_rank_cmd_carries_the_budget(budget_repo: Path, capfd) -> None:
    # No data files are involved, so this used to print a bare pytest with no selection at all.
    argv = ["-C", str(budget_repo), "rank", "-n", "3", "--format", "cmd", *CHANGED]
    assert main(argv) == 0
    line = capfd.readouterr().out.strip()
    var, _pytest, *rest = shlex.split(line)
    name, _, value = var.partition("=")
    assert name == "GRAPH_TEST_SELECTION"
    assert rest[:2] == ["-p", "graph_test"]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *rest],
        cwd=budget_repo,
        env={**os.environ, name: value},
        capture_output=True,
        text=True,
    )
    assert "3 passed" in proc.stdout, proc.stdout + proc.stderr
    assert "kept 3 of 52 cases" in proc.stdout


def test_rank_cmd_without_top_is_unchanged(budget_repo: Path, capfd) -> None:
    assert main(["-C", str(budget_repo), "rank", "--format", "cmd", *CHANGED]) == 0
    out = capfd.readouterr().out
    assert "GRAPH_TEST_SELECTION" not in out
    assert "-p graph_test" not in out


def test_missing_plugin_hint(budget_repo: Path, capfd) -> None:
    # What pytest exits with when `-p graph_test` can't be imported.
    fake = f"{sys.executable} -c 'import sys; sys.exit(4)'"
    assert run(budget_repo, "-n", "3", "--pytest", fake) == 4
    assert "install graph-test where the tests run" in capfd.readouterr().err


def _budget_run(root: Path, sel: dict, *args: str) -> list[str]:
    (root / "sel.json").write_text(json.dumps({"root": str(root), **sel}))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--co", "-q", "-p", "no:randomly", *args],
        cwd=root,
        env={**os.environ, "GRAPH_TEST_SELECTION": str(root / "sel.json")},
        capture_output=True,
        text=True,
    )
    return [ln for ln in proc.stdout.splitlines() if "::" in ln]


TEST = """
import pytest

@pytest.mark.parametrize("n", [f"c{i:02}" for i in range(100)])
def test_a(n):
    pass

def test_b():
    pass

@pytest.mark.slow
def test_c():
    pass
"""
A, B, C = (f"t_test.py::test_{x}" for x in "abc")


@pytest.fixture
def sel_repo(tmp_path: Path) -> Path:
    (tmp_path / "t_test.py").write_text(TEST)
    return tmp_path


def test_budget_follows_rank_not_collection_order(sel_repo: Path) -> None:
    sel = {"full": [A, B, C], "budget": 2, "order": [C, B, A]}
    assert _budget_run(sel_repo, sel) == [B, C]


def test_cases_are_spread_not_first_k(sel_repo: Path) -> None:
    sel = {"full": [A], "budget": 4, "order": [A]}
    assert _budget_run(sel_repo, sel) == [f"{A}[c{i}]" for i in (12, 37, 62, 87)]


def test_file_entry_is_one_group(sel_repo: Path) -> None:
    sel = {"full": ["t_test.py", B], "budget": 3, "order": [B, "t_test.py"]}
    got = _budget_run(sel_repo, sel)
    assert len(got) == 3
    assert B in got


def test_budget_spent_after_other_deselection(sel_repo: Path) -> None:
    # -k runs before the budget, so what it removes doesn't eat into the N items.
    sel = {"full": [A, B, C], "budget": 2, "order": [C, B, A]}
    assert _budget_run(sel_repo, sel, "-k", "not test_c") == [f"{A}[c50]", B]


def test_cargo_targets_use_budget() -> None:
    py = Ranked(Target("a", "pytest", "a.py", nodeid="a.py::t"), 1.0)
    rs = Ranked(Target("b", "cargo", crate="c"), 1.0)
    assert pytest_budget(5, [py, rs]) == 4
    assert pytest_budget(5, [rs]) is None
    assert pytest_budget(None, [py]) is None
