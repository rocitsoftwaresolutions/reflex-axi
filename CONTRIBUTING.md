# Development

Use Python 3.11+ and uv. The runtime uses Pydantic for strict contracts, JSON Schema
for StateBuilders, and standard-library SQLite, threads, HTTP, hashing and CLI parsing.

```sh
uv sync --frozen --extra dev
uv run python scripts/check.py
```

The check runs Ruff lint and formatting, mypy, generated-skill freshness, all unit and
subprocess/mocked-HTTP/crash integration tests, wheel and sdist builds, an isolated installed-wheel
smoke test, and the relative version/cache benchmark. Tests forbid network sockets and never call a live provider.
Tests and tools use ignored `.scratch/` state and fixtures. CI covers macOS/Linux and
Python 3.11/3.13. Runtime support is POSIX; Windows is not currently supported.

Useful focused commands:

```sh
uv run pytest -q tests/test_cli.py
uv run pytest -q tests/test_storage_jobs.py tests/test_optimization.py
uv run ruff check src tests scripts examples
uv run ruff format src tests scripts examples
uv run mypy src
uv run python scripts/generate_skill.py
```

Reproduce bugs through the console entry point or provider boundary before fixing them.
Keep errors structured and sanitized. Preserve immutable component versions, provider
identity, explicit evidence authorization, fenced writes, and the no-action boundary.
Never include real evidence, learned state, datasets, provider keys or inference caches
in fixtures or commits. Keep public protocols small; add adapters instead of core domain logic.

Release by changing the package version and leaf `__version__`, updating version assertions,
running the complete check, then publishing built artifacts through the repository's release
authority. No automatic publishing, changelog editing, merge, or production promotion occurs.
