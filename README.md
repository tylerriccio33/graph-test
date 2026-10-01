# graph-test

Ranks the tests most likely to be affected by your diff in mixed Rust + Python repos, so you can run a useful slice of CI locally.

```sh
graph-test rank --explain          # ranked tests vs merge-base with origin/main
graph-test rank -n 20 --format cmd # pytest/cargo commands for the top 20
graph-test run -n 20 --pytest "uv run pytest"
graph-test graph --rdeps crates/core/src/math.rs
```

## How it scores

- **Static graph:** Python `import`s (via `ast`, including relative imports, package `__init__`, and `conftest.py`), Rust `mod`/`use` and workspace crate dependencies, and PyO3 crates linked to the Python modules they expose (`[lib] name` / `tool.maturin.module-name`, including `.pyi` stubs). Each test scores `graph_weight / (1 + distance)` to the changed file.
- **Co-change:** a test that changed together with a file in git history (at least 2 commits; commits touching more than 50 files are ignored) adds `cochange_weight * P(test | file)`.
- **Global files** (`Cargo.lock`, `pyproject.toml`, ...) give every test a minimum score.

Rust test targets: `tests/*.rs` → `cargo test -p <crate> --test <name>`; files with `#[cfg(test)]` → `cargo test -p <crate> --lib -- <mod::path>::`.

## Config (`pyproject.toml`)

```toml
[tool.graph-test]
exclude = ["vendor"]
global-files = ["Cargo.lock", "pyproject.toml"]
graph-weight = 0.7
cochange-weight = 0.3
global-floor = 0.2
history-commits = 1000
max-commit-files = 50
min-cochange-support = 2
test-globs = ["integration/*_check.py"]
```

Parse results are cached in `.graph-test/` (add it to `.gitignore`).

## Development

`make install`, `make fmt`, `make check` (ruff + pyrefly + pytest).
