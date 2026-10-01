"""Configuration, read from ``[tool.graph-test]`` in the repo's pyproject.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "target",
    "node_modules",
    "__pycache__",
    ".graph-test",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
)

# Files whose change can plausibly break anything.
DEFAULT_GLOBAL_FILES = (
    "Cargo.lock",
    "Cargo.toml",
    "pyproject.toml",
    "uv.lock",
    "setup.py",
    "setup.cfg",
    "rust-toolchain.toml",
    "rust-toolchain",
)


@dataclass
class Config:
    root: Path
    exclude: tuple[str, ...] = DEFAULT_EXCLUDES
    global_files: tuple[str, ...] = DEFAULT_GLOBAL_FILES
    graph_weight: float = 0.7
    cochange_weight: float = 0.3
    # Score floor given to every test when a global file changes.
    global_floor: float = 0.2
    history_commits: int = 1000
    # Commits touching more files than this are ignored for co-change (merges, reformats).
    max_commit_files: int = 50
    # A test must co-change with a file at least this many times to count.
    min_cochange_support: int = 2
    extra_test_globs: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, root: Path) -> Config:
        cfg = cls(root=root)
        pyproject = root / "pyproject.toml"
        if not pyproject.exists():
            return cfg
        try:
            data = tomllib.loads(pyproject.read_text())
        except (tomllib.TOMLDecodeError, OSError):
            return cfg
        section = data.get("tool", {}).get("graph-test", {})
        if "exclude" in section:
            cfg.exclude = DEFAULT_EXCLUDES + tuple(section["exclude"])
        if "global-files" in section:
            cfg.global_files = tuple(section["global-files"])
        for key in (
            "graph_weight",
            "cochange_weight",
            "global_floor",
            "history_commits",
            "max_commit_files",
            "min_cochange_support",
        ):
            toml_key = key.replace("_", "-")
            if toml_key in section:
                setattr(cfg, key, type(getattr(cfg, key))(section[toml_key]))
        if "test-globs" in section:
            cfg.extra_test_globs = tuple(section["test-globs"])
        return cfg
