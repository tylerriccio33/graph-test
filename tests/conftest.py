from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

FILES = {
    "Cargo.toml": '[workspace]\nmembers = ["crates/core", "crates/pyext"]\n',
    "crates/core/Cargo.toml": '[package]\nname = "my-core"\nversion = "0.1.0"\n',
    "crates/core/src/lib.rs": "pub mod math;\npub mod text;\n",
    "crates/core/src/math.rs": (
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        "#[cfg(test)]\nmod tests {\n    use super::*;\n"
        "    #[test]\n    fn t() { assert_eq!(add(1, 2), 3); }\n}\n"
    ),
    "crates/core/src/text/mod.rs": "pub mod upper;\n",
    "crates/core/src/text/upper.rs": "pub fn up(s: &str) -> String { s.to_uppercase() }\n",
    "crates/core/tests/text_it.rs": (
        'use my_core::text::upper::up;\n#[test]\nfn it() { assert_eq!(up("a"), "A"); }\n'
    ),
    "crates/pyext/Cargo.toml": (
        '[package]\nname = "pyext"\nversion = "0.1.0"\n[lib]\nname = "_native"\n'
        '[dependencies]\npyo3 = "0.22"\nmy-core = { path = "../core" }\n'
    ),
    "crates/pyext/src/lib.rs": "use my_core::math::add;\n",
    "python/pkg/__init__.py": "",
    "python/pkg/util.py": "def f():\n    return 1\n",
    "python/pkg/fast.py": "from pkg import _native\n",
    "python/pkg/_native.pyi": "def add(a: int, b: int) -> int: ...\n",
    "python/tests/conftest.py": "",
    "python/tests/test_util.py": "from pkg.util import f\n",
    "python/tests/test_fast.py": "from pkg.fast import _native\n",
    "python/tests/test_alone.py": "import os\n",
}


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def init_repo(root: Path, files: dict[str, str]) -> Path:
    write_files(root, files)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path, FILES)
