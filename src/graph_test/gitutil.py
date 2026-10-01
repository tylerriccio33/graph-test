"""Thin wrappers over the git CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def repo_root(start: Path) -> Path:
    return Path(_git(start, "rev-parse", "--show-toplevel").strip())


def default_base(root: Path) -> str:
    for ref in ("origin/main", "origin/master", "main", "master"):
        try:
            _git(root, "rev-parse", "--verify", "--quiet", ref)
            return ref
        except GitError:
            continue
    return "HEAD"


def changed_files(root: Path, base: str) -> list[str]:
    """Files changed vs the merge-base with ``base``, including uncommitted and untracked."""
    try:
        merge_base = _git(root, "merge-base", base, "HEAD").strip()
    except GitError:
        merge_base = base
    files: set[str] = set()
    files.update(_git(root, "diff", "--name-only", merge_base).splitlines())
    files.update(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    return sorted(f for f in files if f)


def commit_file_sets(root: Path, limit: int) -> Iterator[list[str]]:
    """Yield the files touched by each of the last ``limit`` non-merge, non-root commits.

    Root commits are skipped: an initial import says nothing about what changes together.
    """
    try:
        out = _git(root, "log", f"-n{limit}", "--no-merges", "--name-only", "--format=%x00%P")
    except GitError:
        return
    for chunk in out.split("\x00")[1:]:
        parents, _, body = chunk.partition("\n")
        files = [line for line in body.splitlines() if line.strip()]
        if parents.strip() and files:
            yield files
