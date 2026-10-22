"""Flakiness guards: the plugin narrows identically on every xdist worker and hash seed,
and fixture repos don't depend on the caller's git environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FILES, init_repo

TEST = """
import pytest

CASES = [f"corpus/c{i}.txt" for i in range(12)]

@pytest.mark.parametrize("path", CASES)
def test_corpus(path):
    pass

@pytest.mark.parametrize("n", range(6))
def test_full(n):
    pass

def test_dropped():
    pass
"""


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text(TEST)
    (tmp_path / "corpus").mkdir()
    for i in range(12):
        (tmp_path / f"corpus/c{i}.txt").write_text("")
    (tmp_path / "sel.json").write_text(
        json.dumps(
            {
                "root": str(tmp_path),
                "full": ["tests/test_x.py::test_full"],
                "cases": {"tests/test_x.py::test_corpus": ["corpus/c3.txt", "corpus/c7.txt"]},
            }
        )
    )
    return tmp_path


def _run(root: Path, *args: str, seed: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GRAPH_TEST_SELECTION": str(root / "sel.json"), "PYTHONHASHSEED": seed}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-p", "no:cacheprovider", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_selection_stable_across_hash_seeds(proj: Path) -> None:
    runs = [
        [ln for ln in _run(proj, "--co", "-q", seed=s).stdout.splitlines() if "::" in ln]
        for s in ("0", "1", "123")
    ]
    assert runs[0] == runs[1] == runs[2]
    assert runs[0] == [
        "tests/test_x.py::test_corpus[corpus/c3.txt]",
        "tests/test_x.py::test_corpus[corpus/c7.txt]",
        *[f"tests/test_x.py::test_full[{n}]" for n in range(6)],
    ]


@pytest.mark.parametrize("dist", ["load", "loadfile"])
def test_selection_identical_under_xdist(proj: Path, dist: str) -> None:
    pytest.importorskip("xdist")
    proc = _run(proj, "-p", "xdist", "-n", "3", "--dist", dist, "-q", seed="random")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "Different tests were collected" not in out
    assert "8 passed" in proc.stdout, out


def test_fixture_repo_ignores_caller_git_env(tmp_path: Path) -> None:
    # Under a git hook GIT_DIR/GIT_INDEX_FILE are set; the autouse fixture must scrub them
    # and isolate global/system config so fixture commits can't be redirected or rejected.
    leaked = {v for v in os.environ if v.startswith("GIT_")}
    assert leaked == {"GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"}
    repo = init_repo(tmp_path / "r", FILES)
    assert (repo / ".git").is_dir()


def test_path_match_beats_bare_name_match(tmp_path: Path) -> None:
    """An unrelated case whose id merely contains "expected" mustn't crowd out the real one."""
    from types import SimpleNamespace

    from graph_test import pytest_plugin

    (tmp_path / "a").mkdir()
    (tmp_path / "a/expected.py").write_text("")
    (tmp_path / "t.py").write_text("")

    def item(cid: str, path: str) -> SimpleNamespace:
        cs = SimpleNamespace(id=cid, params={"path": path})
        return SimpleNamespace(path=tmp_path / "t.py", nodeid=f"t.py::test_x[{cid}]", callspec=cs)

    items = [item("a", str(tmp_path / "a")), item("b-expected", "b"), item("c", "c")]
    sel = tmp_path / "sel.json"
    sel.write_text(
        json.dumps({"root": str(tmp_path), "cases": {"t.py::test_x": ["a/expected.py"]}})
    )
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
    mp = pytest.MonkeyPatch()
    mp.setenv("GRAPH_TEST_SELECTION", str(sel))
    try:
        pytest_plugin.pytest_collection_modifyitems(config, items)  # type: ignore[arg-type]
    finally:
        mp.undo()
    assert [i.callspec.id for i in items] == ["a"]
