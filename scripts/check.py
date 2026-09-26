"""The repository's complete direct-PR verification suite."""

import os
import subprocess
import sys

python = sys.executable
commands = [
    [python, "-m", "ruff", "check", "src", "tests", "scripts", "examples"],
    [python, "-m", "ruff", "format", "--check", "src", "tests", "scripts", "examples"],
    [python, "-m", "mypy", "src"],
    [python, "scripts/generate_skill.py", "--check"],
    [python, "-m", "pytest", "-q"],
    ["uv", "build", "--cache-dir", ".scratch/uv-cache"],
    [python, "scripts/installed_smoke.py"],
    [python, "scripts/benchmark.py", ".scratch/performance.json"],
]
env = {**os.environ, "UV_CACHE_DIR": os.path.abspath(".scratch/uv-cache")}
for command in commands:
    print("check:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
print("All validation passed.")
