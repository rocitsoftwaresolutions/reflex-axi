import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from reflex_axi.engine import Engine
from reflex_axi.errors import ReflexError
from reflex_axi.jobs import BatchRunner
from reflex_axi.providers import MockProvider
from reflex_axi.store import Store


def test_store_permissions_immutable_and_transaction_rollback(store):
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600
    store.register("packs", "x", "1", {"x": 1})
    store.register("packs", "x", "1", {"x": 1})
    with pytest.raises(ReflexError):
        store.register("packs", "x", "1", {"x": 2})
    with pytest.raises(RuntimeError), store.transaction() as db:
        store.put("test", "uncommitted", True, db)
        raise RuntimeError("simulated failure")
    assert store.get("test", "uncommitted") is None


def test_concurrent_writers_no_lost_updates(store):
    def increment(_):
        with store.transaction() as db:
            count = store.get("test", "count", db) or 0
            store.put("test", "count", count + 1, db)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(increment, range(100)))
    assert store.get("test", "count") == 100


def test_atomic_process_crash_recovery(store):
    script = """
import os,sys
from reflex_axi.store import Store
s=Store(sys.argv[1])
with s.transaction() as db:
 s.put('test','uncommitted',{'bad':True},db)
 s.event('must-not-appear','x',{},db)
 os._exit(73)
"""
    result = subprocess.run([sys.executable, "-c", script, str(store.root)])
    assert result.returncode == 73
    recovered = Store(store.root)
    assert recovered.get("test", "uncommitted") is None
    assert recovered.events() == []
    with recovered.connect() as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_batch_resume_after_actual_process_exit(store, tmp_path, pack, binding):
    definition = tmp_path / "input.json"
    definition.write_text(
        json.dumps({"pack": pack.model_dump(by_alias=True), "binding": binding.model_dump()})
    )
    script = """
import json,os,sys
from reflex_axi.store import Store
from reflex_axi.models import Pack,Binding
from reflex_axi.jobs import BatchRunner
from reflex_axi.engine import Engine
data=json.load(open(sys.argv[2]));store=Store(sys.argv[1])
class Crash(Engine):
 calls=0
 def evaluate_many(self,*args,**kwargs):
  self.calls+=1
  if self.calls==2: os._exit(74)
  return super().evaluate_many(*args,**kwargs)
BatchRunner(store,Crash(store)).run([{'text':str(i)} for i in range(5)],[Pack.model_validate(data['pack'])],Binding.model_validate(data['binding']),job_id='crash',chunk_size=2)
"""
    result = subprocess.run([sys.executable, "-c", script, str(store.root), str(definition)])
    assert result.returncode == 74
    before = BatchRunner(store).results("crash")
    assert len(before) == 2
    # Simulate elapsed lease TTL, rather than sleeping for 15 minutes.
    with store.transaction() as db:
        db.execute("UPDATE leases SET expires=0")
    completed = BatchRunner(Store(store.root)).resume("crash", chunk_size=2)
    assert completed["completed"] == 5
    assert BatchRunner(store).results("crash")[:2] == before


def test_job_identity_conflict_and_completed_noop(store, pack, binding):
    provider = MockProvider(binding.provider)
    runner = BatchRunner(store, Engine(store, provider))
    runner.run([{"text": "a"}], [pack], binding, job_id="id")
    assert runner.resume("id")["completed"] == 1
    assert provider.calls == 1
    with pytest.raises(ReflexError, match="different inputs"):
        runner.run([{"text": "b"}], [pack], binding, job_id="id")


def test_fenced_leases_and_cross_process_contention(store):
    owner = store.claim("work")
    assert store.claim("work") is None
    with store.transaction() as db:
        db.execute("UPDATE leases SET expires=0 WHERE key='work'")
    replacement = store.claim("work")
    assert replacement != owner
    store.release("work", owner)
    with store.connect() as db:
        assert store.owns("work", replacement, db)


def test_symlink_state_refused(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination)
    with pytest.raises(ReflexError, match="symlink"):
        Store(link)
