"""Private, crash-safe local records with transactions, immutable assets and leased work."""

from __future__ import annotations

import builtins
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import ReflexError
from .identity import canonical, digest


class Store:
    def __init__(self, root: Path | str | None = None) -> None:
        base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        self.root = (
            Path(root or os.environ.get("REFLEX_STATE_ROOT", str(base / "reflex-axi")))
            .expanduser()
            .absolute()
        )
        if self.root.is_symlink():
            raise ReflexError("unsafe_state", "state root must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.path = self.root / "reflex.sqlite3"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                  bucket TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL,
                  revision INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(bucket,key));
                CREATE TABLE IF NOT EXISTS leases (
                  key TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL,
                  kind TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rates (key TEXT PRIMARY KEY, next REAL NOT NULL);
            """)
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ReflexError(
                    "storage_version", "unsupported state schema; use the matching Reflex release"
                )
            db.execute("PRAGMA user_version=1")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def get(self, bucket: str, key: str, db: sqlite3.Connection | None = None) -> Any:
        if db is None:
            with self.connect() as conn:
                return self.get(bucket, key, conn)
        row = db.execute(
            "SELECT body FROM records WHERE bucket=? AND key=?", (bucket, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def require(self, bucket: str, key: str, db: sqlite3.Connection | None = None) -> Any:
        value = self.get(bucket, key, db)
        if value is None:
            raise ReflexError("not_found", f"{bucket} record not found: {key}")
        return value

    def put(
        self,
        bucket: str,
        key: str,
        body: Any,
        db: sqlite3.Connection | None = None,
        *,
        immutable: bool = False,
    ) -> None:
        if db is None:
            with self.transaction() as conn:
                self.put(bucket, key, body, conn, immutable=immutable)
            return
        encoded = canonical(body)
        if immutable:
            old = self.get(bucket, key, db)
            if old is not None and canonical(old) != encoded:
                raise ReflexError(
                    "version_conflict",
                    f"{bucket}/{key} already has different content; increment its version",
                )
        db.execute(
            "INSERT INTO records(bucket,key,body) VALUES(?,?,?) ON CONFLICT(bucket,key) DO UPDATE SET body=excluded.body,revision=records.revision+1",
            (bucket, key, encoded),
        )

    def list(self, bucket: str, db: sqlite3.Connection | None = None) -> list[Any]:
        if db is None:
            with self.connect() as conn:
                return self.list(bucket, conn)
        return [
            json.loads(r[0])
            for r in db.execute("SELECT body FROM records WHERE bucket=? ORDER BY key", (bucket,))
        ]

    def event(self, kind: str, subject: str, body: Any, db: sqlite3.Connection) -> None:
        db.execute(
            "INSERT INTO events(time,kind,subject,body) VALUES(?,?,?,?)",
            (time.time(), kind, subject, canonical(body)),
        )

    def events(self, after: int = 0, limit: int = 100) -> builtins.list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(sequence=r[0], time=r[1], kind=r[2], subject=r[3], detail=json.loads(r[4]))
                for r in db.execute(
                    "SELECT sequence,time,kind,subject,body FROM events WHERE sequence>? ORDER BY sequence LIMIT ?",
                    (after, limit),
                )
            ]

    def claim(self, key: str, ttl: float = 360) -> str | None:
        owner = uuid.uuid4().hex
        with self.transaction() as db:
            row = db.execute("SELECT expires FROM leases WHERE key=?", (key,)).fetchone()
            if row and row[0] > time.time():
                return None
            db.execute(
                "INSERT INTO leases VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET owner=excluded.owner,expires=excluded.expires",
                (key, owner, time.time() + ttl),
            )
        return owner

    def owns(self, key: str, owner: str, db: sqlite3.Connection) -> bool:
        row = db.execute("SELECT owner,expires FROM leases WHERE key=?", (key,)).fetchone()
        return bool(row and row[0] == owner and row[1] > time.time())

    def release(self, key: str, owner: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND owner=?", (key, owner))

    def reserve_rate(self, provider: str, rate: float) -> float:
        with self.transaction() as db:
            row = db.execute("SELECT next FROM rates WHERE key=?", (provider,)).fetchone()
            now = time.time()
            start = max(now, row[0] if row else now)
            db.execute(
                "INSERT INTO rates VALUES(?,?) ON CONFLICT(key) DO UPDATE SET next=excluded.next",
                (provider, start + 1 / rate),
            )
            return start - now

    def register(self, bucket: str, name: str, version: str, body: Any) -> str:
        key = f"{name}@{version}"
        self.put(bucket, key, body, immutable=True)
        return key

    def project_key(self) -> str:
        return digest(str(Path.cwd().resolve()))
