"""Atomic, private, contention-safe export and integration writes."""

import fcntl
import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, "w") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    finally:
        os.close(lock_fd)
