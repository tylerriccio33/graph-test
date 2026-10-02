"""Thin wrappers over the git CLI."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

_HUNK_RE = re.compile(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@")


class GitError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    # Diffs routinely contain non-UTF-8 content (e.g. latin-1 corpora). surrogateescape
    # never fails and round-trips any non-UTF-8 bytes in paths back to the filesystem.
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
    )
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


# (start, end, pure_deletion) in new-file line numbers.
Hunk = tuple[int, int, bool]
# path -> hunks, or None when the whole file should count as changed (new/untracked/binary).
Changes = dict[str, list[Hunk] | None]


def changed_hunks(root: Path, base: str) -> Changes:
    """Changes vs the merge-base with ``base``, including uncommitted and untracked files."""
    try:
        merge_base = _git(root, "merge-base", base, "HEAD").strip()
    except GitError:
        merge_base = base
    q = ("-c", "core.quotePath=false")
    out: Changes = {}
    for f in _git(root, *q, "diff", "--no-renames", "--name-only", merge_base).splitlines():
        if f:
            out[f] = None
    diff = _git(root, *q, "diff", "--no-renames", "--no-color", "--no-ext-diff", "-U0", merge_base)
    new = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            new = line[4:].removeprefix("b/")
        elif line.startswith("@@"):
            if new == "/dev/null":
                continue  # deleted file: stays None
            path = new
            m = _HUNK_RE.match(line)
            if not m:
                continue
            start, count = int(m.group(1)), int(m.group(2) or "1")
            hunks = out.get(path) or []
            if count == 0:
                hunks.append((max(start, 1), start + 1, True))
            else:
                hunks.append((start, start + count - 1, False))
            out[path] = hunks
    for f in _git(root, *q, "ls-files", "--others", "--exclude-standard").splitlines():
        if f:
            out[f] = None
    return out


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
