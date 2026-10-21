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


def merge_base(root: Path, base: str) -> str:
    try:
        return _git(root, "merge-base", base, "HEAD").strip()
    except GitError:
        return base


def show(root: Path, rev: str, path: str) -> str | None:
    """Contents of ``path`` at ``rev``, or None if it didn't exist there."""
    try:
        return _git(root, "show", f"{rev}:{path}")
    except GitError:
        return None


_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _diff_path(raw: str) -> str:
    """Path from a ``+++`` header: git appends a tab after names containing spaces and
    C-quotes names with quotes, backslashes or control characters."""
    if not raw.startswith('"'):
        return raw.rstrip("\t") if raw.endswith("\t") and "\t" not in raw[:-1] else raw
    body = raw[1 : raw.rindex('"')]
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            n = body[i + 1]
            if n in _ESCAPES:
                out.append(_ESCAPES[n])
                i += 2
                continue
            m = re.match(r"[0-7]{1,3}", body[i + 1 :])
            if m:
                out.append(int(m.group(), 8) & 0xFF)
                i += 1 + len(m.group())
                continue
        out += c.encode("utf-8", "surrogateescape")
        i += 1
    return out.decode("utf-8", "surrogateescape")


def changed_hunks(root: Path, base: str) -> Changes:
    """Changes vs the merge-base with ``base``, including uncommitted and untracked files."""
    mb = merge_base(root, base)
    q = ("-c", "core.quotePath=false")
    out: Changes = {}
    for f in _git(root, *q, "diff", "--no-renames", "--name-only", "-z", mb).split("\0"):
        if f:
            out[f] = None
    diff = _git(
        root, *q, "diff", "--no-renames", "--no-color", "--no-ext-diff", "--no-prefix", "-U0", mb
    )
    new = ""
    header = False  # only a header's "+++ " names a file; an added line "++ x" looks the same
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            header = True
        elif header and line.startswith("+++ "):
            new = _diff_path(line[4:])
        elif line.startswith("@@"):
            header = False
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
    for f in _git(root, *q, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
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
