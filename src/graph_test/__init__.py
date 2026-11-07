"""graph-test: rank tests by likelihood of being affected by a diff."""

__version__ = "0.1.0"

# `pytest -p graph_test` (what `-n` passes) must load the hooks even without the entry point.
pytest_plugins = ["graph_test.pytest_plugin"]
