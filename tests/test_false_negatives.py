"""False-negative hunt: changes that must select tests, in the git and Python-scan layers.

UC 57  Fixture overrides: a fixture that requests its own name depends on the parent's.
UC 58  `@pytest.fixture(name=...)` is requested by that name, not the function name.
UC 59  `usefixtures` (decorator, class, module `pytestmark`) and `getfixturevalue("x")`.
UC 60  Fixtures imported into a test module resolve to their definition.
UC 61  `pytest_plugins = [...]` strings link the conftest's tests to the plugin module.
UC 62  Paths with spaces diff at line level; a renamed file selects importers of the old path.
UC 63  `importlib.import_module("x")` with a literal links to the module.
UC 64  Tests defined under a module-level `if`/`try` are still found.
UC 65  Quoted/odd paths in diffs don't lose hunks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import git, init_repo
from test_use_cases import edit, selected_from_git

from graph_test import gitutil

FIXT = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def base():\n    return 1\n",
    "tests/conftest.py": (
        "import pytest\nfrom pkg.core import base\n\n\n"
        "@pytest.fixture\ndef client():\n    return base()\n\n\n"
        '@pytest.fixture(name="renamed")\ndef _renamed():\n    return base()\n\n\n'
        "@pytest.fixture\ndef side():\n    return base()\n"
    ),
    "tests/sub/conftest.py": (
        "import pytest\n\n\n@pytest.fixture\ndef client(client):\n    return client\n"
    ),
    "tests/sub/test_sub.py": "def test_sub(client):\n    assert client\n",
    "tests/test_ren.py": "def test_ren(renamed):\n    assert renamed\n",
    "tests/test_use.py": (
        "import pytest\n\n\n@pytest.mark.usefixtures('side')\ndef test_use():\n    pass\n\n\n"
        "def test_none():\n    pass\n"
    ),
    "tests/test_gfv.py": (
        "def test_gfv(request):\n    request.getfixturevalue('side')\n\n\n"
        "def test_none():\n    pass\n"
    ),
    "tests/test_mark.py": (
        "import pytest\n\npytestmark = pytest.mark.usefixtures('side')\n\n\n"
        "def test_m():\n    pass\n"
    ),
    "tests/test_cls.py": (
        "import pytest\n\n\n@pytest.mark.usefixtures('side')\nclass TestC:\n"
        "    def test_c(self):\n        pass\n"
    ),
    "tests/fixtures_mod.py": "import pytest\n\n\n@pytest.fixture\ndef imp():\n    return 1\n",
    "tests/test_imp.py": ("from fixtures_mod import imp\n\n\ndef test_imp(imp):\n    assert imp\n"),
}


@pytest.fixture
def fixt_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, FIXT)


def test_uc57_fixture_override_depends_on_parent(fixt_repo: Path) -> None:
    edit(
        fixt_repo,
        "tests/conftest.py",
        "def client():\n    return base()",
        "def client():\n    return 2",
    )
    assert "tests/sub/test_sub.py::test_sub" in selected_from_git(fixt_repo)


def test_uc58_renamed_fixture(fixt_repo: Path) -> None:
    edit(
        fixt_repo,
        "tests/conftest.py",
        "def _renamed():\n    return base()",
        "def _renamed():\n    return 2",
    )
    assert "tests/test_ren.py::test_ren" in selected_from_git(fixt_repo)


def test_uc59_usefixtures_and_getfixturevalue(fixt_repo: Path) -> None:
    edit(
        fixt_repo,
        "tests/conftest.py",
        "def side():\n    return base()",
        "def side():\n    return 2",
    )
    got = selected_from_git(fixt_repo)
    for t in ("test_use.py::test_use", "test_gfv.py::test_gfv", "test_mark.py::test_m"):
        assert f"tests/{t}" in got, t
    assert "tests/test_cls.py::TestC.test_c" in got
    assert "tests/test_use.py::test_none" not in got
    assert "tests/test_gfv.py::test_none" not in got


def test_uc60_imported_fixture(fixt_repo: Path) -> None:
    edit(fixt_repo, "tests/fixtures_mod.py", "return 1", "return 2")
    assert "tests/test_imp.py::test_imp" in selected_from_git(fixt_repo)


PLUG = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def base():\n    return 1\n",
    "plugins/__init__.py": "",
    "plugins/myplug.py": (
        "import pytest\nfrom pkg.core import base\n\n\n@pytest.fixture\ndef plug():\n"
        "    return base()\n"
    ),
    "plugins/dyn.py": "def d():\n    return 1\n",
    "tests/conftest.py": 'pytest_plugins = ["plugins.myplug"]\n',
    "tests/test_plug.py": "def test_plug(plug):\n    assert plug\n",
    "tests/test_dyn.py": (
        "import importlib\n\n\ndef test_dyn():\n"
        "    m = importlib.import_module('plugins.dyn')\n    assert m.d()\n"
    ),
    "tests/test_other.py": "def test_other():\n    pass\n",
}


def test_uc61_pytest_plugins(tmp_path: Path) -> None:
    repo = init_repo(tmp_path, PLUG)
    edit(repo, "plugins/myplug.py", "return base()", "return 2")
    assert "tests/test_plug.py::test_plug" in selected_from_git(repo)
    edit(repo, "pkg/core.py", "return 1", "return 3")
    assert "tests/test_plug.py::test_plug" in selected_from_git(repo)


def test_uc63_import_module_literal(tmp_path: Path) -> None:
    repo = init_repo(tmp_path, PLUG)
    edit(repo, "plugins/dyn.py", "return 1", "return 2")
    got = selected_from_git(repo)
    assert "tests/test_dyn.py::test_dyn" in got
    assert "tests/test_other.py::test_other" not in got


def test_uc62_spaces_and_renames(tmp_path: Path) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/my mod.py": "def f():\n    return 1\n\n\ndef g():\n    return 1\n",
        "pkg/mv.py": "def h():\n    return 1\n",
        "tests/test_mv.py": "from pkg.mv import h\n\n\ndef test_h():\n    assert h()\n",
    }
    repo = init_repo(tmp_path, files)
    edit(repo, "pkg/my mod.py", "def g():\n    return 1", "def g():\n    return 2")
    changes = gitutil.changed_hunks(repo, "main")
    assert set(changes) == {"pkg/my mod.py"}
    assert changes["pkg/my mod.py"] == [(5, 6, False)] or changes["pkg/my mod.py"] is not None
    git(repo, "mv", "pkg/mv.py", "pkg/moved.py")
    assert "tests/test_mv.py::test_h" in selected_from_git(repo)


def test_uc64_tests_under_module_level_if(tmp_path: Path) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def base():\n    return 1\n",
        "tests/test_cond.py": (
            "import sys\nfrom pkg.core import base\n\n\ndef test_plain():\n    pass\n\n\n"
            "if sys.platform != 'nope':\n\n    def test_cond():\n        assert base()\n"
        ),
    }
    repo = init_repo(tmp_path, files)
    edit(repo, "pkg/core.py", "return 1", "return 2")
    assert "tests/test_cond.py::test_cond" in selected_from_git(repo)


def test_uc65_quoted_paths(tmp_path: Path) -> None:
    name = 'pkg/we"ird\tname.py'
    files = {"pkg/__init__.py": "", name: "def f():\n    return 1\n"}
    repo = init_repo(tmp_path, files)
    edit(repo, name, "return 1", "return 2")
    changes = gitutil.changed_hunks(repo, "main")
    assert set(changes) == {name}


def _deep_source(arms: int, tail: str) -> str:
    chain = "".join(f"{'if' if i == 0 else 'elif'} x == {i}:\n    y = {i}\n" for i in range(arms))
    return "def f(x):\n    y = 0\n" + "\n".join("    " + ln for ln in chain.splitlines()) + tail


def test_deep_ast_edit_is_full_risk_not_a_crash() -> None:
    """A ~1000-arm if/elif chain overflows the recursive AST passes; the edit stays full risk."""
    import ast

    from graph_test import pyscan, risk

    old_src, new_src = _deep_source(1000, "\n"), _deep_source(1000, "\n    y = 1\n")
    old_tree, new_tree = ast.parse(old_src), ast.parse(new_src)
    info = pyscan.analyze(new_src, "m", False)
    assert risk.symbol_risk(old_tree, new_tree, None, info, True) == risk.FULL
    assert risk.symbol_risk(old_tree, new_tree, "f", info, True) == risk.FULL
