import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn

from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

from . import __version__
from .catalog import DESCRIPTION, GUIDANCE, builtin_packs, load_binding, load_file, load_pack
from .distill import Distillation, FixtureTrainer
from .engine import Engine
from .errors import ReflexError
from .evidence import Evidence
from .identity import digest, read_json
from .io import atomic_write
from .jobs import BatchRunner
from .lifecycle import Lifecycle, SuppliedProposal
from .models import Bundle, Deployment, ExecutionPolicy, Outcome, Pack, Policy, ProviderSpec, Scope
from .optimization import StructuredOptimizer, SweepConfig
from .store import Store
from .toon import encode


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        flags = sorted({option for action in self._actions for option in action.option_strings})
        raise ReflexError(
            "usage", message, f"{self.prog} valid flags: {' '.join(flags)}; {self.prog} --help"
        )


def parser() -> Parser:
    common = Parser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
    common.add_argument(
        "--state-root", help="private runtime directory (default REFLEX_STATE_ROOT or XDG state)"
    )
    common.add_argument("--json", action="store_true", help="JSON output (default compact TOON)")
    common.add_argument("--full", action="store_true", help="include complete records")
    common.add_argument("--fields", help="comma-separated top-level fields for result records")
    common.add_argument("--output", help="atomic private output file; stdout gives a receipt")
    root = Parser(
        prog="reflex-axi",
        description=DESCRIPTION,
        allow_abbrev=False,
        parents=[common],
        epilog="Examples:\n  reflex-axi\n  reflex-axi evaluate --pack filter-context --state state.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument("-v", "-V", "--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", parser_class=Parser)

    def command(name: str, description: str, examples: list[str]) -> Parser:
        p = commands.add_parser(
            name,
            description=description,
            help=description,
            parents=[common],
            allow_abbrev=False,
            epilog="Examples:\n  " + "\n  ".join("reflex-axi " + x for x in examples),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        return p

    def selection(p: Parser) -> None:
        group = p.add_mutually_exclusive_group(required=not p.prog.endswith("batch"))
        group.add_argument("--pack", help="pack ID, id@version or JSON file")
        group.add_argument("--bundle", help="versioned Pack Bundle JSON file")
        p.add_argument(
            "--provider",
            help="binding/provider JSON file or id@version (default active binding or explicit mock fixture)",
        )
        p.add_argument("--local-only", action="store_true", help="reject remote providers")
        p.add_argument("--no-cache", action="store_true", help="disable inference cache")
        p.add_argument(
            "--research", action="store_true", help="judgment only; no evidence or lifecycle writes"
        )
        p.add_argument(
            "--record",
            action="store_true",
            help="retain decisions for later evidence (evaluate default false)",
        )
        p.add_argument(
            "--workload", default="default", help="calibration workload (default default)"
        )
        p.add_argument(
            "--scope", default="default", help="calibration evaluation scope (default default)"
        )

    for name in ["evaluate", "decide"]:
        p = command(
            name,
            "Return bounded judgment"
            if name == "evaluate"
            else "Return judgment and optional policy recommendation; never execute actions",
            [
                f"{name} --pack filter-context --state state.json",
                f"{name} --bundle bundle.json --state state.json --full",
            ],
        )
        selection(p)
        p.add_argument("--state", required=True, help="state JSON file or - for stdin")
        if name == "decide":
            p.add_argument(
                "--policy",
                choices=["pack", "none"],
                default="pack",
                help="policy selection (default pack)",
            )
    p = command(
        "batch",
        "Run or resume many states x packs with per-cell checkpoints",
        [
            "batch --pack filter-context --states states.jsonl --job example",
            "batch --resume example --retry-failed --results --full",
        ],
    )
    selection(p)
    p.add_argument("--states", help="JSONL input file, one object per nonblank line")
    p.add_argument("--job", help="explicit stable job ID (default input hash)")
    p.add_argument("--resume", help="resume existing job using its exact saved inputs")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry failed cells; successful cells remain untouched",
    )
    p.add_argument("--results", action="store_true", help="include cell results")
    p.add_argument("--decide", action="store_true", help="attach pack policy recommendations")
    p.add_argument(
        "--chunk-size", type=int, default=128, help="checkpoint chunk size (default 128; 1-10000)"
    )
    p = command(
        "feedback",
        "Silently record idempotent outcome evidence",
        ["feedback --file outcome.json", "feedback --file outcome.json --json"],
    )
    p.add_argument("--file", required=True, help="Outcome JSON file")
    p = command(
        "scopes",
        "Inspect or explicitly configure evidence permissions",
        ["scopes", "scopes --import scope.json"],
    )
    p.add_argument(
        "--import",
        dest="import_file",
        help="Scope JSON file; grants readable/evaluatable/optimizable/allowed_to_mutate independently",
    )
    for name in ["packs", "providers"]:
        p = command(
            name,
            f"List or register immutable versioned {name}",
            [name, f"{name} --import {name[:-1]}.json --full"],
        )
        p.add_argument("--import", dest="import_file", help="strict JSON definition file")
    p = command(
        "activate",
        "Explicitly initialize a pack's active deployment; upgrades use candidates",
        ["activate --file deployment.json", "activate --file deployment.json --full"],
    )
    p.add_argument("--file", required=True, help="Deployment JSON file")
    p = command(
        "status",
        "Current directory's active packs, jobs and candidate counts",
        ["status", "status --full"],
    )
    p = command(
        "events",
        "Read meaningful lifecycle events using a durable cursor",
        ["events --after 0", "events --after 12 --limit 100 --json"],
    )
    p.add_argument(
        "--after", type=int, default=0, help="exclusive event sequence cursor (default 0)"
    )
    p.add_argument(
        "--limit", type=int, default=100, help="maximum events (default 100, maximum 10000)"
    )
    p = command(
        "improve",
        "Check readiness or run an approved evidence-authorized optimization sweep",
        [
            "improve --pack filter-context",
            "improve --pack filter-context --scope training --proposal candidate.json --approve",
        ],
    )
    p.add_argument("--pack", required=True, help="active pack ID")
    p.add_argument(
        "--scope",
        action="append",
        default=[],
        help="explicit optimization evidence scope; repeatable",
    )
    p.add_argument(
        "--minimum", type=int, default=3, help="minimum distinct evidence decisions (default 3)"
    )
    p.add_argument(
        "--proposal", help="proposed Deployment from a strong sweep; JSON null means no update"
    )
    p.add_argument(
        "--approve", action="store_true", help="authorize optimization; never implies promotion"
    )
    p.add_argument("--reason", default="evidence-backed proposal", help="proposal rationale")
    p.add_argument(
        "--optimizer", help="strong-model SweepConfig JSON file; one approved bounded call"
    )
    p = command(
        "candidates",
        "List, inspect or reject candidate deployments",
        ["candidates", "candidates --candidate <id> --reject 'regression' "],
    )
    p.add_argument("--candidate", help="candidate ID")
    p.add_argument("--reject", help="rejection reason; requires --candidate")
    p = command(
        "eval",
        "Replay candidate against historical outcomes and protected regressions",
        [
            "eval --candidate <id> --scope evaluation",
            "eval --candidate <id> --scope evaluation --full",
        ],
    )
    p.add_argument("--candidate", required=True, help="candidate ID")
    p.add_argument(
        "--scope", action="append", required=True, help="authorized evaluation scope; repeatable"
    )
    p = command(
        "shadow",
        "Run an explicit shadow pair or assess real downstream outcomes",
        [
            "shadow --candidate <id> --state state.json",
            "shadow --candidate <id> --assess --scope evaluation",
        ],
    )
    p.add_argument("--candidate", required=True, help="candidate ID")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--state", help="new state JSON file")
    group.add_argument("--assess", action="store_true", help="assess paired real outcomes")
    p.add_argument(
        "--scope", action="append", default=[], help="evaluation scope for --assess; repeatable"
    )
    p = command(
        "promote",
        "Promote a ready candidate with explicit production approval",
        ["promote --candidate <id> --approve", "promote --candidate <id> --approve --full"],
    )
    p.add_argument("--candidate", required=True, help="candidate ID")
    p.add_argument("--approve", action="store_true", help="explicit production authorization")
    p = command(
        "rollback",
        "Atomically restore a retained deployment, including provider and calibration",
        [
            "rollback --pack filter-context",
            "rollback --pack filter-context --target <deployment-hash>",
        ],
    )
    p.add_argument("--pack", required=True, help="active pack ID")
    p.add_argument("--target", help="retained deployment hash (default previous)")
    p = command(
        "distill",
        "Build an authorized teacher dataset and train a local candidate",
        [
            "distill --pack filter-context --scope training --trainer fixture --approve",
            "distill --pack filter-context --scope training --trainer fixture --approve --full",
        ],
    )
    p.add_argument("--pack", required=True, help="active mature pack ID")
    p.add_argument(
        "--scope", action="append", required=True, help="authorized training scope; repeatable"
    )
    p.add_argument(
        "--trainer",
        choices=["fixture"],
        required=True,
        help="explicit trainer; fixture is a lifecycle test student",
    )
    p.add_argument(
        "--approve", action="store_true", help="authorize teacher collection and training"
    )
    p = command(
        "compare",
        "Explicitly compare pinned providers; ordinary inference never does this",
        [
            "compare --pack filter-context --state state.json --provider a.json --against b.json",
            "compare --pack filter-context --state state.json --provider a.json --against b.json --full",
        ],
    )
    selection(p)
    p.add_argument("--state", required=True, help="state JSON file")
    p.add_argument("--against", required=True, help="second binding/provider JSON file")
    p = command(
        "setup",
        "Write an opt-in session integration snippet for review and installation",
        [
            "setup --app codex --output codex-hooks.json",
            "setup --app opencode --output reflex-plugin.js",
        ],
    )
    p.add_argument(
        "--app",
        choices=["codex", "claude", "opencode"],
        required=True,
        help="target harness; existing harness config is never replaced",
    )
    return root


def state_file(path: str) -> dict[str, Any]:
    value = read_json(sys.stdin.read()) if path == "-" else load_file(path)
    if not isinstance(value, dict):
        raise ReflexError("usage", "state must be a JSON object")
    return value


def execution(args: argparse.Namespace) -> ExecutionPolicy:
    return ExecutionPolicy(
        local_only=args.local_only,
        cache=not args.no_cache,
        workload=args.workload,
        scope=args.scope,
    )


def selected_packs(args: argparse.Namespace, store: Store) -> list[Pack]:
    if args.bundle:
        bundle = Bundle.model_validate(load_file(args.bundle))
        packs = [load_pack(p, store) for p in bundle.packs]
        if len({digest(p.state.model_dump(by_alias=True)) for p in packs}) != 1:
            raise ReflexError(
                "bundle_state", "bundle packs must share one versioned StateBuilder contract"
            )
        return packs
    if not args.pack:
        raise ReflexError("usage", "--pack or --bundle is required")
    return [load_pack(args.pack, store)]


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: result[k] for k in ["id", "pack", "selected", "confidence"]}
    if result.get("policy_result"):
        out["policy"] = result["policy_result"]["recommendation"]
    return out


def dispatch(args: argparse.Namespace, store: Store) -> Any:
    command = args.command
    lifecycle = Lifecycle(store)
    if command in {None, "status"}:
        active = [r for r in store.list("deployments") if r.get("project") == store.project_key()]
        result: dict[str, Any] = {
            "bin": str(Path(sys.argv[0]).resolve()).replace(str(Path.home()), "~", 1),
            "description": DESCRIPTION,
            "active": [
                {
                    "pack": r["active"]["pack"]["id"],
                    "version": r["active"]["pack"]["version"],
                    "status": r["status"],
                }
                for r in active
            ],
            "count": len(active),
        }
        if not active:
            result["status"] = (
                "0 active deployments in this directory; built-in packs use the mock fixture until configured"
            )
        if args.full:
            result["deployments"] = active
            result["jobs"] = [
                {k: v for k, v in j.items() if k != "definition"}
                for j in store.list("jobs")
                if j["definition"]["project"] == store.project_key()
            ]
        result["help"] = GUIDANCE
        return result
    if command in {"evaluate", "decide", "compare"}:
        packs = selected_packs(args, store)
        binding = load_binding(args.provider, store, packs[0])
        state = state_file(args.state)
        if args.research and args.record:
            raise ReflexError("usage", "--research cannot be combined with --record")
        if command == "compare":
            other = load_binding(args.against, store, packs[0])
            engine = Engine(store)
            pairs = []
            for pack in packs:
                a = engine.evaluate(pack, state, binding, execution=execution(args))
                b = engine.evaluate(pack, state, other, execution=execution(args))
                pairs.append(
                    {
                        "pack": pack.id,
                        "agreement": a.selected == b.selected,
                        "distance": sum(
                            abs(a.distribution[k] - b.distribution[k]) for k in a.distribution
                        )
                        / 2,
                        "a": a.model_dump() if args.full else compact_result(a.model_dump()),
                        "b": b.model_dump() if args.full else compact_result(b.model_dump()),
                    }
                )
            return {"comparisons": pairs}
        is_decide = command == "decide" and not args.research
        if is_decide and args.policy == "none":
            packs = [p.model_copy(update={"policy": Policy()}) for p in packs]
        values = Engine(store).evaluate_many(
            [(p, state) for p in packs],
            binding,
            execution=execution(args),
            record=(args.record or is_decide) and not args.research,
            decide=is_decide,
        )
        output = []
        for value in values:
            if isinstance(value, ReflexError):
                raise value
            full = value.model_dump()
            output.append(full if args.full or args.fields else compact_result(full))
        return output[0] if len(output) == 1 else {"count": len(output), "results": output}
    if command == "batch":
        runner = BatchRunner(store)
        if args.resume:
            if (
                args.states
                or args.pack
                or args.bundle
                or args.provider
                or args.job
                or args.decide
                or args.record
                or args.research
                or args.no_cache
                or args.local_only
                or args.workload != "default"
                or args.scope != "default"
            ):
                raise ReflexError(
                    "usage",
                    "--resume uses saved inputs; do not combine it with inference configuration",
                )
            result = runner.resume(
                args.resume, retry_failed=args.retry_failed, chunk_size=args.chunk_size
            )
        else:
            if not args.states:
                raise ReflexError("usage", "--states is required unless resuming")
            if args.research and (args.record or args.decide):
                raise ReflexError(
                    "usage", "research batch is judgment-only and cannot record evidence"
                )
            packs = selected_packs(args, store)
            states = [
                read_json(line)
                for line in Path(args.states).read_text().splitlines()
                if line.strip()
            ]
            if any(not isinstance(s, dict) for s in states):
                raise ReflexError("usage", "every JSONL state must be an object")
            result = runner.run(
                states,
                packs,
                load_binding(args.provider, store, packs[0]),
                job_id=args.job,
                execution=execution(args),
                decide=args.decide,
                record=args.record,
                retry_failed=args.retry_failed,
                chunk_size=args.chunk_size,
            )
        if args.results:
            rows = runner.results(result["id"])
            if not args.full:
                rows = [
                    {
                        **{k: v for k, v in r.items() if k != "result"},
                        **({"result": compact_result(r["result"])} if "result" in r else {}),
                    }
                    for r in rows
                ]
            result["results"] = rows
        return result
    if command == "feedback":
        return Evidence(store).feedback(Outcome.model_validate(load_file(args.file)))
    if command == "scopes":
        if args.import_file:
            Evidence(store).scope(Scope.model_validate(load_file(args.import_file)))
        rows = store.list("scopes")
        return {
            "count": len(rows),
            "scopes": rows,
            "help": ["reflex-axi scopes --import scope.json"],
        }
    if command in {"packs", "providers"}:
        if args.import_file:
            obj = (Pack if command == "packs" else ProviderSpec).model_validate(
                load_file(args.import_file)
            )
            store.register(command, obj.id, obj.version, obj.model_dump(by_alias=True))
        rows = store.list(command)
        if command == "packs":
            known = {(p["id"], p["version"]) for p in rows}
            rows += [
                p.model_dump(by_alias=True)
                for p in builtin_packs()
                if (p.id, p.version) not in known
            ]
        if not args.full:
            rows = [
                {
                    "id": r["id"],
                    "version": r["version"],
                    "primitive" if command == "packs" else "model": r["judgment"]["primitive"]
                    if command == "packs"
                    else r["model"],
                }
                for r in rows
            ]
        return {
            "count": len(rows),
            command: rows,
            "help": [f"reflex-axi {command} --import <file>"],
        }
    if command == "activate":
        return lifecycle.activate(Deployment.model_validate(load_file(args.file)))
    if command == "events":
        if args.after < 0 or not 1 <= args.limit <= 10000:
            raise ReflexError("usage", "event cursor must be nonnegative and limit must be 1-10000")
        rows = store.events(args.after, args.limit)
        return {
            "count": len(rows),
            "cursor": rows[-1]["sequence"] if rows else args.after,
            "events": rows
            if args.full
            else [{k: r[k] for k in ["sequence", "kind", "subject"]} for r in rows],
        }
    if command == "improve":
        if not args.approve and not args.proposal and not args.optimizer:
            return lifecycle.readiness(args.pack, args.minimum)
        if args.optimizer and args.proposal:
            raise ReflexError("usage", "use --optimizer or --proposal, not both")
        data = load_file(args.proposal) if args.proposal else None
        proposal = Deployment.model_validate(data) if data is not None else None
        return lifecycle.improve(
            args.pack,
            StructuredOptimizer(SweepConfig.model_validate(load_file(args.optimizer)))
            if args.optimizer
            else SuppliedProposal(proposal),
            scopes=args.scope,
            approve=args.approve,
            minimum=args.minimum,
            reason=args.reason,
        )
    if command == "candidates":
        if args.reject:
            if not args.candidate:
                raise ReflexError("usage", "--reject requires --candidate")
            return lifecycle.reject(args.candidate, args.reject)
        if args.candidate:
            row = lifecycle.candidate(args.candidate)
            if args.full:
                return row
            return {
                "id": row["id"],
                "pack": row["pack"],
                "status": row["status"],
                "reason": row["reason"][:500],
                **(
                    {
                        "help": [
                            f"reflex-axi candidates --candidate {row['id']} --full ({len(row['reason'])} characters)"
                        ]
                    }
                    if len(row["reason"]) > 500
                    else {}
                ),
            }
        rows = [c for c in store.list("candidates") if c["key"] == lifecycle.key(c["pack"])]
        return {
            "count": len(rows),
            "candidates": rows
            if args.full
            else [{k: r[k] for k in ["id", "pack", "status"]} for r in rows],
            "help": ["reflex-axi candidates --candidate <id> --full"],
        }
    if command == "eval":
        return lifecycle.replay(args.candidate, scopes=args.scope)
    if command == "shadow":
        if args.assess:
            if not args.scope:
                raise ReflexError("usage", "shadow --assess requires --scope")
            return lifecycle.assess_shadow(args.candidate, scopes=args.scope)
        return lifecycle.shadow(args.candidate, state_file(args.state))
    if command == "promote":
        return lifecycle.promote(args.candidate, approve=args.approve)
    if command == "rollback":
        return lifecycle.rollback(args.pack, target=args.target)
    if command == "distill":
        return Distillation(store).run(
            args.pack, FixtureTrainer(), scopes=args.scope, approve=args.approve
        )
    raise ReflexError("usage", "unknown command")


def setup(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output:
        raise ReflexError(
            "usage",
            "setup requires --output; merge the generated snippet into your harness configuration",
        )
    import shlex
    import shutil

    executable = Path(sys.argv[0]).resolve()
    on_path = shutil.which("reflex-axi")
    binary = "reflex-axi" if on_path and Path(on_path).resolve() == executable else str(executable)
    command = shlex.quote(binary)
    if args.app == "opencode":
        # Harness-owned execution, opt-in only. No transcript or secret collection.
        content = (
            'export const ReflexPlugin = async ({ $ }) => ({\n  "experimental.chat.system.transform": async (_, output) => {\n    const state = await $`'
            + binary.replace("`", "\\`").replace("${", "\\${")
            + " status`.text();\n    output.system.push(state);\n  }\n});\n"
        )
    else:
        content = (
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": command + " status"}]}
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )
    atomic_write(Path(args.output), content)
    return {
        "file": args.output,
        "status": "snippet written",
        "help": [
            f"Merge this snippet into {args.app}'s project configuration; enable hooks in Codex config.toml when required."
        ],
    }


def run(argv: list[str]) -> int:
    args = None
    try:
        args = parser().parse_args(
            argv,
            namespace=argparse.Namespace(
                state_root=None, json=False, full=False, fields=None, output=None
            ),
        )
        if args.command == "setup":
            result = setup(args)
            args.output = None
        else:
            result = dispatch(args, Store(args.state_root))
        if args.fields:
            names = args.fields.split(",")
            if not isinstance(result, dict) or not set(names) <= set(result):
                raise ReflexError("usage", "--fields must name existing top-level output fields")
            result = {k: result[k] for k in names}
        rendered = (
            json.dumps(result, ensure_ascii=False, allow_nan=False) if args.json else encode(result)
        )
        if args.output:
            atomic_write(Path(args.output), rendered + "\n")
            result = {"file": args.output, "status": "written"}
            rendered = json.dumps(result) if args.json else encode(result)
        print(rendered)
        return 1 if isinstance(result, dict) and result.get("status") == "PARTIAL" else 0
    except SystemExit as error:
        return int(error.code or 0)
    except (ReflexError, ValidationError, SchemaError, OSError, ValueError, sqlite3.Error) as error:
        if isinstance(error, ReflexError):
            code, message, hint = error.code, str(error), error.help
        elif isinstance(error, ValidationError):
            details = error.errors(include_url=False, include_input=False)
            code, message, hint = (
                "validation",
                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in details[:5]),
                "Check the input contract in docs/packs.md and docs/providers.md",
            )
        elif isinstance(error, OSError | sqlite3.Error):
            code, message, hint = (
                "storage",
                f"local I/O failed ({type(error).__name__})",
                "Check file paths, permissions and --state-root",
            )
        else:
            code, message, hint = (
                "validation",
                "invalid JSON, schema or numeric input",
                "Use strict JSON and the documented input schema",
            )
        result = {"error": message, "code": code, "help": hint}
        print(json.dumps(result) if "--json" in argv else encode(result))
        return 2 if code in {"usage", "validation", "state_invalid"} else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
