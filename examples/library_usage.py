"""Use Reflex as a library, without the CLI.

Run it against a throwaway state root:

    REFLEX_STATE_ROOT=$(mktemp -d) python examples/library_usage.py

Every provider call here goes to the explicit mock fixture, so the output shows framework
behaviour (identity, caching, bundles, batching, cascade), not semantic quality.
"""

import os
import tempfile

from reflex_axi.catalog import builtin_packs, default_binding
from reflex_axi.engine import Engine
from reflex_axi.jobs import BatchRunner
from reflex_axi.models import Bundle, ExecutionPolicy, Policy
from reflex_axi.store import Store

STATE = {
    "task": "Summarize the release changes",
    "context": "Release notes include a database migration and a new search filter.",
}


def main(root: str) -> None:
    store = Store(root)
    engine = Engine(store)
    binding = default_binding()
    packs = {p.id: p for p in builtin_packs()}

    # 1. Judgment only. No policy, no evidence, no lifecycle writes.
    result = engine.evaluate(packs["filter-context"], STATE, binding)
    print("evaluate:", result.selected, round(result.confidence, 3), result.distribution)
    print("  identity:", result.inference_hash[:16], "state:", result.state_hash[:16])

    # 2. policy:none is first-class - decide returns judgment with no recommendation.
    quiet = packs["route-model"].model_copy(update={"policy": Policy()})
    print("decide policy:none:", engine.decide(quiet, STATE, binding).policy_result)

    # 3. A pack that does define a policy recommends, but never executes.
    decided = engine.decide(packs["route-model"], STATE, binding)
    assert decided.policy_result is not None
    print("decide recommendation:", decided.policy_result)

    # 4. A bundle evaluates several versioned packs independently against one state.
    bundle = Bundle(
        id="result-triage",
        version="1",
        packs=["result-sufficiency@1", "retry-worthiness@1", "escalation-need@1"],
    )
    members = [packs[reference.split("@")[0]] for reference in bundle.packs]
    for value in engine.evaluate_many([(p, STATE) for p in members], binding):
        assert not isinstance(value, Exception)
        print(f"  {value.pack}: {value.selected} ({value.confidence:.2f})")

    # 5. Identity is deterministic, so the second run is a cache hit and costs nothing.
    again = engine.evaluate(packs["filter-context"], STATE, binding)
    print(
        "cache reuse:", again.execution["cache_hit"], again.inference_hash == result.inference_hash
    )

    # 6. Many states x one pack, checkpointed per cell and resumable by job ID.
    states = [{"task": "Assess relevance", "context": f"Example context {i}"} for i in range(8)]
    runner = BatchRunner(store)
    job = runner.run(states, [packs["filter-context"]], binding, job_id="example-job")
    print("batch:", job["status"], job["completed"], "of", job["total"])
    print("resume:", runner.resume("example-job")["completed"])

    # 7. Cascades are explicit and ordered; nothing falls back on its own.
    cascaded = engine.cascade(packs["filter-context"], STATE, [binding], threshold=0.95)
    print("cascade attempts:", cascaded.execution["attempts"])
    print("system2 required:", cascaded.execution["system2_required"])

    # 8. Research/offline mode: judgment only, caching on, nothing recorded or mutated.
    research = ExecutionPolicy(cache=True, workload="research", scope="offline")
    feature = engine.evaluate(packs["result-sufficiency"], STATE, binding, execution=research)
    print("research expected_score:", feature.expected_score)


if __name__ == "__main__":
    configured = os.environ.get("REFLEX_STATE_ROOT")
    if configured:
        main(configured)
    else:
        with tempfile.TemporaryDirectory(prefix="reflex-example-") as directory:
            main(directory)
