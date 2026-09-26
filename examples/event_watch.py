"""Optional external event integration.

Reflex never calls a supervisor and never needs one. Meaningful lifecycle transitions are
appended to a durable, monotonically numbered event log; an external process - FirstMate or
anything else - reads it with a cursor it stores itself. No fork, plugin or modification of
the supervisor is required, and routine evidence collection emits nothing.

    REFLEX_STATE_ROOT=$(mktemp -d) python examples/event_watch.py --once

The watcher shells out to the installed CLI, so it works against any Reflex installation
without importing the package. Use the library form (Store(...).events(after, limit)) when
the supervisor already runs in the same process.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Silence is the default. Only these transitions deserve a supervisor's attention.
NOTIFY = {
    "optimization_ready",
    "optimizing",
    "optimization_failed",
    "no_update_warranted",
    "candidate_created",
    "candidate_rejected",
    "shadow_started",
    "promotion_ready",
    "promoted",
    "rolled_back",
    "distillation_candidate",
}


def read_events(binary: str, after: int, limit: int) -> dict:
    output = subprocess.check_output(
        [binary, "events", "--after", str(after), "--limit", str(limit), "--full", "--json"],
        text=True,
    )
    return json.loads(output)


def notify(event: dict) -> None:
    # Replace with a webhook, a status file, or a supervisor inbox write.
    print(json.dumps({k: event[k] for k in ["sequence", "kind", "subject", "detail"]}))


def watch(binary: str, cursor_path: Path, interval: float, once: bool) -> int:
    cursor = int(cursor_path.read_text()) if cursor_path.exists() else 0
    while True:
        page = read_events(binary, cursor, 500)
        for event in page["events"]:
            if event["kind"] in NOTIFY:
                notify(event)
        if page["events"]:
            cursor = page["cursor"]
            # Persist only after handling, so a crash replays instead of skipping.
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(str(cursor))
        if once:
            return cursor
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=os.environ.get("REFLEX_BIN", "reflex-axi"))
    parser.add_argument("--cursor", default=".scratch/reflex-events.cursor")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="drain once and exit")
    args = parser.parse_args()
    cursor = watch(args.binary, Path(args.cursor), args.interval, args.once)
    if args.once:
        print(f"cursor: {cursor}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
