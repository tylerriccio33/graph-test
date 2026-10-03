# graph-test

Selects and ranks the tests affected by your diff in mixed Rust + Python repos. It works as a pass/fail gate (`run`) or as a ranked list for quick local runs (`rank -n N`).

```sh
graph-test run --pytest "uv run pytest"   # gate: every affected test, exit code = pass/fail
graph-test run -n 40                      # just the top 40 by score
graph-test rank --explain                 # ranked list and why each test was picked
graph-test rank --affected --format cmd   # pytest/cargo commands to paste
graph-test graph --rdeps pkg/core.py::parse
```

The diff is taken against the merge-base with `origin/main` (or `--base`), plus uncommitted and untracked files.

## What it tracks

**Python, per symbol.** Each test function or method is its own node. Diff hunks map to the functions, classes, or assigned names they touch, so editing one function selects only the tests that reach that function. Comment, blank-line, and docstring edits select nothing. The graph also follows:

- imports, including relative imports and re-exports through `__init__.py` hubs, so a hub doesn't fan out to everything
- `module.attr` chains
- `functools.singledispatch`-style registrations: a function decorated with `@f.register` or `@f.register(T)` is treated as a dependency of `f`, in the same module or another one. Repeated names such as many `def _` stay separate symbols, so an edit maps to the right one
- pytest fixtures defined in the test file or a `conftest.py`, including autouse fixtures and `pytest_*` hooks. A conftest's own imports don't link to every test, so a heavy import used by a session hook won't select the whole suite (see `conftest-imports`)
- module-level code (import-time side effects)
- deleted modules, so their importers are selected

**Rust.** `mod` declarations, `use` paths (`crate::`, `super::`, other workspace crates), and crate dependencies.

- Integration tests run with `cargo test -p <crate> --test <name>`.
- Unit tests run with `cargo test -p <crate> --lib -- <module>::`.

**PyO3.** Stub symbols in `.pyi` files link to the Rust file defining them: `#[pyfunction]`, `#[pyclass]`, `#[pymethods]`, and `#[pyo3(name = "...")]`. A Rust change therefore reaches only the Python tests calling the affected functions. Editing the `#[pymodule]` file itself affects everything that imports the extension.

**Data files.** Path-like string literals in Python link to the files and directories they name, for example `HERE / "corpus"`, `"expected/*.csv"`, or `"tests/fixtures/input.sas"`. A test reached only through data is narrowed to individual cases: the bundled pytest plugin keeps only the parametrized cases whose parameters or ids refer to a changed file, and keeps every case when none match. Anything the tool can't infer can be mapped explicitly in config.

Case narrowing needs graph-test installed in the environment where the tests run (for example as a dev dependency), because the plugin loads through pytest's `pytest11` entry point. Without it, the selected test functions still run, just with every case.

**Safety in gate mode.** A changed file the graph can't place makes `run` run everything (`--on-unmapped all`, the default). A changed global file does the same. Ignored paths (`*.md`, `docs/*`, ...) never select anything.

**Ranking.** A test scores `graph_weight / (1 + hops)`, plus a git co-change score (at least 2 shared commits; root commits and commits touching more than 50 files are skipped). In gate mode, tests linked only by history are added with `--include-history`.

## Config (`pyproject.toml`)

```toml
[tool.graph-test]
exclude = ["vendor"]
ignore = ["*.md", "docs/*"]
global-files = ["Cargo.lock", "pyproject.toml", "uv.lock"]
test-globs = ["integration/*_check.py"]
# "fixtures-only" (default): tests depend on a conftest's own code, hooks, autouse fixtures
# and the fixtures they request. "all": also on everything the conftest imports.
conftest-imports = "fixtures-only"

[tool.graph-test.data]
"tests/integration/code/*" = ["tests/test_integration.py::test_code*"]
"golden/*" = ["tests/test_golden.py::*"]
```

Weights and other tuning keys: `graph-weight`, `cochange-weight`, `global-floor`, `history-commits`, `max-commit-files`, `min-cochange-support`.

State lives in `.graph-test/`, which ignores itself in git.

## Getting the most out of it

graph-test reads your code statically. The more directly a test's dependencies appear in the source, the fewer tests it selects and the less it misses. Most of these tips are also ordinary good structure.

### Test suites and conftest

- **Request fixtures by argument and keep them focused.** A test links to exactly the fixtures it names. A narrow `parser` fixture beats a do-everything `env` fixture, because every test using `env` reruns whenever anything `env` touches changes.
- **Be sparing with autouse fixtures and hooks in a root conftest.** An autouse fixture applies to every test under its directory, so everything it calls becomes a dependency of all those tests. Editing a hook such as `pytest_sessionstart` reruns everything under that conftest. Put session-wide setup in the narrowest directory that needs it.
- **Define fixtures in `conftest.py`, not in `pytest_plugins` modules.** Fixtures loaded through `pytest_plugins = ["myproj.fixtures"]` aren't resolved, so tests using them miss those dependencies.
- **Keep test modules light at import time.** A test file's top-level code (imports, module-level calls) is followed in full and links every test in the file. Build expensive objects in fixtures or inside the test, not at module level.
- **Prefer smaller, focused tests.** Selection is per test function. One huge test that exercises half the package reruns for any change to that half; parametrized or split tests rerun only the relevant parts.
- **Import what you use.** `from pkg.parse import parse` and `pkg.parse.parse(...)` both resolve to the function. `getattr(module, name)`, `importlib.import_module(...)` and string-based lookups are invisible.

### Library code

- **Avoid import-time side effects.** Module-level statements, such as registering things, calling setup functions, or building tables with calls, count as the module's own code, so editing them reruns every importer. Plain constants (`X = 1`) are separate symbols and stay precise.
- **Prefer functions and small classes to large classes.** Calls through an instance (`obj.method()`) can't be resolved statically, so a test using a class depends on the whole class. A change to any method reruns every test that uses the class.
- **Use `.register` for registries.** `@f.register` and `@f.register(T)` link implementations to the dispatcher `f`. Other registry decorators, such as `@app.route` or `@REGISTRY.add`, aren't recognized, so code reached only through them can be missed. Either name the decorator `register` or make sure tests reference the implementations directly.
- **Re-export freely.** `from .core import parse` in `__init__.py` is followed to the defining module, so public-API hubs don't widen selection. `from x import *` works too, but explicit names resolve faster and more reliably.

### Test data

- **Name data paths with literals in the test file:** `Path(__file__).parent / "corpus"`, `"expected/*.csv"` or `"tests/fixtures/input.sas"`. Paths assembled entirely at runtime (from env vars, config files or computed names) can't be seen, so map them in `[tool.graph-test.data]`.
- **Pass the data file as the parametrize argument** (`@pytest.mark.parametrize("case", sorted(DIR.glob("*.sas")))`), ideally with `ids=lambda p: p.stem`. The plugin then reruns only the cases whose file changed. When cases are keyed by something else, every case reruns.

### Rust and PyO3

- **Keep `.pyi` stubs complete and in sync.** Python tests reach Rust through stub symbols. A function missing from the stub falls back to depending on the whole crate.
- **Keep the `#[pymodule]` file thin.** Editing it reruns every test that imports the extension. Put `#[pyfunction]`s and `#[pyclass]`es in separate files grouped by area, so a change there reaches only the tests calling those functions.
- **Smaller crates and modules give sharper selection.** Rust is tracked per file, and unit tests are selected per module.

### Running it

- **Install graph-test where the tests run** (for example as a dev dependency), so the pytest plugin can narrow to individual parametrized cases.
- **Use the gate before pushing:** `graph-test run --pytest "uv run pytest"`. For a fast inner loop, use `graph-test rank -n 20 --format cmd` or `graph-test run -n 20`.
- **Don't let "unmapped" files force full runs.** If `run` reports `running everything - unmapped change: X`, add `X` to `ignore` if it can't affect tests, or map it in `[tool.graph-test.data]` if it can.
- **Keep `global-files` honest.** Every global file makes any change to it run everything. Drop entries that don't affect your tests, such as `uv.lock` if you never pin test behavior through it.
- **Debug selections** with `graph-test rank --explain` (why a test was picked) and `graph-test graph --rdeps path/to/file.py::func` (everything that depends on a symbol). If something important is missing, the cause is usually one of the patterns above.
- **Keep a full run as the backstop.** Static selection can miss dynamic behavior, so run the whole suite on `main` or nightly, and use graph-test for local and pre-merge runs.

## Limitations

- Dynamic dispatch, such as `getattr` or methods called on instances, resolves at class level or not at all. Unresolvable `module.attr` references fall back to "anything in that module".
- Data links come from string literals, so paths built entirely at runtime need a `[tool.graph-test.data]` entry.
- Rust tests are selected per file or per module, not per `#[test]` function.
- A test file's own module-level imports are still followed in full, so a test file that imports a module with heavy import-time side effects can be selected broadly. Conftests avoid this by default (`conftest-imports`).

## Development

`make install`, `make fmt`, `make check` (ruff + pyrefly + pytest). The numbered use cases are in `tests/test_use_cases.py`.
