"""Base ``btran`` run command; correction commands intentionally do not exist yet."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from btran.artifacts import ArtifactStore, ArtifactError, RevisionStore
from btran.corrections import CorrectionError, CorrectionStore, correction_transition, parse_correction_json
from btran.config import WorkspaceResolutionError, load_config, resolve_workspace
from btran.orchestrator import orchestrator_run
from btran.orchestrator_contract import OrchestratorCallable, RunResult
from btran.schema import Finding


def _read_pointer(workspace: Path, filename: str, field: str) -> str | None:
    """Resolve one explicit/default immutable selector without selecting by age."""
    path = workspace / filename
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {filename} pointer") from exc
    if set(value) != {field} or not isinstance(value[field], str) or not value[field]:
        raise ValueError(f"invalid {filename} pointer")
    return value[field]


def resolve_base_run_pointers(workspace: Path, base_revision: str | None, correction_set: str | None) -> tuple[str | None, str | None]:
    """Use CLI selectors when supplied, otherwise workspace active pointers."""
    return (
        base_revision if base_revision is not None else _read_pointer(workspace, "active-revision.json", "revision_id"),
        correction_set if correction_set is not None else _read_pointer(workspace, "active-correction-set.json", "set_id"),
    )


def _workspace_fallback_finding(requested: Path, selected: Path) -> Finding:
    return Finding(
        kind="workspace_fallback", severity="warning", stage="configuration",
        subject_refs=(), evidence={"requested_workspace": str(requested), "workspace": str(selected)},
        message="Requested workspace was unavailable; using output-adjacent workspace.",
    )


def _result_field(result: RunResult, name: str, default: Any = None) -> Any:
    report = getattr(result, "report", None)
    if report is not None:
        if isinstance(report, dict):
            return report.get(name, getattr(result, name, default))
        return getattr(report, name, getattr(result, name, default))
    return getattr(result, name, default)


def _result_status(result: RunResult) -> str:
    status = _result_field(result, "status")
    if status is not None:
        return str(status)
    # Gate-1 runner predates RunReport.  Its errors are recoverable page
    # failures under the new contract, never a CLI-level invocation failure.
    return "completed_degraded" if result.errors else "completed"


def _print_invocation_failure(result: RunResult) -> None:
    failure = _result_field(result, "invocation_failure")
    if failure is None:
        return
    if isinstance(failure, dict):
        code = failure.get("code", "unknown")
        path = failure.get("path", "unknown")
        exception_type = failure.get("exception_type", "unknown")
        message = failure.get("message", "")
    else:
        code = getattr(failure, "code", "unknown")
        path = getattr(failure, "path", "unknown")
        exception_type = getattr(failure, "exception_type", "unknown")
        message = getattr(failure, "message", "")
    print(
        f"btran invocation_failed code={code} path={path} "
        f"error={exception_type}: {message}",
        file=sys.stderr,
    )


def _print_summary(result: RunResult, status: str) -> None:
    report = _result_field(result, "run_id", _result_field(result, "report_path", "none"))
    candidate = _result_field(result, "candidate_revision_id", "none")
    active = _result_field(result, "active_revision_id", "none")
    cache_events = _result_field(result, "cache_events", ())
    findings = _result_field(result, "finding_ids", None)
    if findings is None:
        findings = _result_field(result, "findings", ())
    print(
        "btran "
        f"status={status} report={report or 'none'} candidate={candidate or 'none'} "
        f"active={active or 'none'} cache_events={len(cache_events or ())} "
        f"findings={len(findings or ())}"
    )


def _correction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="btran immutable correction commands")
    top = parser.add_subparsers(dest="area", required=True)
    correction = top.add_parser("correction")
    correction_commands = correction.add_subparsers(dest="command", required=True)
    apply = correction_commands.add_parser("apply")
    apply.add_argument("workspace", metavar="WORKSPACE")
    apply.add_argument("--payload", required=True, metavar="CORRECTION.json")
    revert = correction_commands.add_parser("revert")
    revert.add_argument("workspace", metavar="WORKSPACE")
    revert.add_argument("--correction-id", required=True, metavar="CORRECTION_ID")
    revert.add_argument("--revision", required=True, metavar="REVISION_ID")
    supersede = correction_commands.add_parser("supersede")
    supersede.add_argument("workspace", metavar="WORKSPACE")
    supersede.add_argument("--supersedes", required=True, metavar="CORRECTION_ID")
    supersede.add_argument("--payload", required=True, metavar="CORRECTION.json")
    revision = top.add_parser("revision")
    revision_commands = revision.add_subparsers(dest="command", required=True)
    activate = revision_commands.add_parser("activate")
    activate.add_argument("workspace", metavar="WORKSPACE")
    activate.add_argument("revision_id", metavar="REVISION_ID")
    return parser


def _payload_bytes(path: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise CorrectionError("cannot read correction payload") from exc


def _correction_main(argv: list[str]) -> None:
    parser = _correction_parser()
    namespace = parser.parse_args(argv)
    workspace = Path(namespace.workspace)
    try:
        if namespace.area == "revision":
            # RevisionStore.activate verifies the sealed bundle then atomically
            # changes only active-revision.json.  Do not construct/publish any
            # correction state on this path.
            RevisionStore(workspace).activate(namespace.revision_id)
            print(f"btran revision_activated revision={namespace.revision_id}")
            return
        store = CorrectionStore(workspace)
        revisions = RevisionStore(workspace)
        if namespace.command == "apply":
            successor, impact = correction_transition(
                store, revisions, event_kind="apply", payload=parse_correction_json(_payload_bytes(namespace.payload)),
            )
        elif namespace.command == "revert":
            successor, impact = correction_transition(
                store, revisions, event_kind="revert", correction_id=namespace.correction_id,
                revision_id=namespace.revision,
            )
        else:
            successor, impact = correction_transition(
                store, revisions, event_kind="supersede", supersedes_id=namespace.supersedes,
                payload=parse_correction_json(_payload_bytes(namespace.payload)),
            )
    except (CorrectionError, ArtifactError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(
        f"btran correction_{namespace.command} set={successor.set_id} "
        f"plan={impact.projection_plan_id} affected={len(impact.affected)} "
        f"unaffected={len(impact.unaffected)} ambiguous={len(impact.ambiguous)} "
        f"protected={len(impact.protected)} regenerated={len(impact.regenerated)}"
    )


def main() -> None:
    """Dispatch immutable correction commands or base-run invocation."""
    argv = sys.argv[1:]
    if argv and argv[0] in {"correction", "revision"}:
        _correction_main(argv)
        return
    try:
        config = load_config()
        resolution = resolve_workspace(config)
        config.workspace = resolution.workspace
        # Legacy executor still receives its work root through this migration
        # field until Task 13 replaces it with immutable stage contracts.
        config.intermediate_dir = resolution.workspace
        config.base_revision, config.correction_set = resolve_base_run_pointers(
            resolution.workspace, config.base_revision, config.correction_set,
        )
        if resolution.used_fallback:
            ArtifactStore(resolution.workspace).put_finding(
                _workspace_fallback_finding(resolution.fallback_from, resolution.workspace)  # type: ignore[arg-type]
            )
    except SystemExit:
        raise
    except (ValueError, WorkspaceResolutionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    # Do not check input paths or pi here.  Native runs have no model process;
    # translated model spawn and every input/output access belongs to the
    # invocation boundary and is reported as a typed failure there.
    print(
        f"btran mode={config.mode} base_revision={config.base_revision or 'none'} "
        f"correction_set={config.correction_set or 'none'}"
    )

    def on_page_error(page_number: int, message: str) -> None:
        print(f"[btran] page {page_number} failed: {message}", file=sys.stderr)

    runner: OrchestratorCallable = orchestrator_run
    try:
        result: RunResult = asyncio.run(runner(config, on_page_error=on_page_error))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(1)

    status = _result_status(result)
    if status == "invocation_failed":
        _print_invocation_failure(result)
    _print_summary(result, status)
    if status == "invocation_failed":
        raise SystemExit(1)
    # Completed/completed_degraded are success paths.  No content-quality or
    # per-page recoverable finding changes this exit contract.


if __name__ == "__main__":  # pragma: no cover
    main()
