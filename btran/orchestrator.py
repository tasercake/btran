"""Core immutable executor through effective-target materialization.

Task 13 intentionally stops before reconciliation, validation, rendering, sealing,
and final report publication.  ``build_epub`` is imported only to keep the Task
12 renderer contract visible at this boundary; it is never called here.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from btran.artifacts import ArtifactEnvelope, ArtifactError, ArtifactStore, DependencyGraph, RevisionStore
from btran.config import Config, resolve_workspace
from btran.corrections import (
    CorrectionError,
    CorrectionStore,
    OverlayResolution,
    resolve_selected_overlays,
)
from btran.epub_builder import (
    EpubInvocationError,
    RenderPlacement,
    build_epub,
    seal_effective_content,
)
from btran.manifest import BookDiscovery, InvocationFailure, discover_book
from btran.orchestrator_contract import (
    CacheEvent,
    OrchestratorCallable,
    PageErrorCallback,
    RunResult,
    SegmentProvenance,
    SelectedRunInputs,
    StageContract,
    StageInputs,
    StageOutputs,
    initialized_report,
)
from btran.schema import (
    CorrectionImpact,
    EffectivePage,
    EffectiveSegment,
    Finding,
    RevisionSnapshot,
    RunReport,
    StageRecord,
    canonical_json,
    canonical_json_bytes,
    stage_summary_finding,
    tagged_sha256,
)
from btran.source_extractor import (
    RawPageInput,
    empty_input_diagnostic_placement,
    empty_input_diagnostic_raw_run,
    extract_raw_pages,
    materialize_effective_source,
)
from btran.terminology import build_terminology_evidence, make_pi_consolidation_call
from btran.translator import materialize_effective_target, refresh_model_leaves
from btran.reconciliation import reconcile_effective
from btran.validators import validate_effective


# Kept for migration callers/tests.  Task 13 uses ArtifactStore atomic writes.
def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def _with_retries(action: Callable[[], Awaitable[Any]], retries: int) -> Any:
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return await action()
        except Exception as exc:
            last = exc
    assert last is not None
    raise last


def _stage_key(record: StageRecord) -> str:
    return tagged_sha256("stage-record-v1", canonical_json(record.to_dict()).encode("utf-8"))


def _store_finding(store: ArtifactStore, finding: Finding) -> str:
    return store.put_finding(finding)


def _selection_finding(stage: str, kind: str, message: str, *, subject_refs: tuple[str, ...] = ()) -> Finding:
    return Finding(kind=kind, severity="warning", stage=stage, subject_refs=tuple(sorted(set(subject_refs))),
                   evidence={}, message=message)


class _CoreExecutor:
    """Owns ordering and durable stage boundaries, never model-stage internals."""

    def __init__(self, config: Config, workspace: Path, selected: SelectedRunInputs,
                 store: ArtifactStore, graph: DependencyGraph) -> None:
        self.config = config
        self.workspace = workspace
        self.selected = selected
        self.store = store
        self.graph = graph
        self.records: list[StageRecord] = []
        self.cache_events: list[CacheEvent] = []
        self.report = initialized_report(run_id=uuid.uuid4().hex, mode=config.mode, selected=selected)

    def _record(self, inputs: StageInputs, outputs: StageOutputs, *, duration_ms: float) -> StageRecord:
        """Publish StageRecord only after every named artifact/finding/edge exists."""
        for artifact_id in (*inputs.input_artifact_ids, *outputs.output_artifact_ids):
            self.store.get(artifact_id)
        for finding_id in (*inputs.input_finding_ids, *outputs.finding_ids):
            self.store.get_finding(finding_id)
        for edge_id in outputs.graph_edge_ids:
            self.graph.get(edge_id)
        summaries = [self.store.get_finding(item) for item in outputs.finding_ids]
        summary = next((item for item in summaries if item.kind == "stage_summary" and item.stage == inputs.stage
                        and item.evidence.get("status") == outputs.status), None)
        if summary is None:
            raise RuntimeError(f"{inputs.stage} stage did not persist a stage_summary finding")
        record = StageRecord(stage=inputs.stage, status=outputs.status,
                             input_artifact_ids=inputs.input_artifact_ids,
                             output_artifact_ids=outputs.output_artifact_ids,
                             finding_ids=outputs.finding_ids,
                             stage_summary_finding_id=summary.finding_id,
                             duration_ms=duration_ms)
        # Stage records are immutable artifacts too.  Their dependency closure
        # makes a later candidate able to select exact prior stage evidence.
        self.store.put("StageRecord", record.to_dict(),
                       dependency_ids=tuple(sorted(set((*inputs.input_artifact_ids, *outputs.output_artifact_ids)))),
                       finding_ids=record.finding_ids, semantic_key=_stage_key(record))
        self.records.append(record)
        self.cache_events.extend(outputs.cache_events)
        self.report = replace(
            self.report,
            stage_records=tuple(self.records),
            total_stage_duration_ms=round(sum(item.duration_ms for item in self.records), 3),
            cache_events=tuple(event.to_dict() for event in self.cache_events),
        )
        return record

    async def stage(self, name: str, inputs: StageInputs,
                    runner: Callable[[StageInputs], StageOutputs | Awaitable[StageOutputs]]) -> StageOutputs:
        started_ns = time.perf_counter_ns()
        outputs = await StageContract(name, runner).execute(inputs)
        duration_ms = round((time.perf_counter_ns() - started_ns) / 1_000_000, 3)
        self._record(inputs, outputs, duration_ms=duration_ms)
        return outputs


def _book_artifact(discovery: BookDiscovery, store: ArtifactStore) -> tuple[str, tuple[str, ...]]:
    assert discovery.book is not None
    finding_ids = tuple(sorted(_store_finding(store, finding) for finding in discovery.findings))
    body = discovery.book.to_dict()
    artifact = store.put("BookRecord", body, finding_ids=finding_ids,
                         semantic_key=tagged_sha256("book-discovery-v1", canonical_json(body).encode("utf-8")))
    return artifact.artifact_id, finding_ids


def _selected_inputs(config: Config, revisions: RevisionStore) -> tuple[SelectedRunInputs, Finding | None]:
    """Resolve active/default base only by explicit pointer or CLI selector."""
    try:
        active = revisions.active_snapshot()
        active_id = None if active is None else active.revision_id
        base_id = config.base_revision or active_id or "unsealed"
        snapshot = None if base_id == "unsealed" else revisions.snapshot(base_id)
        selected_ids = () if snapshot is None else snapshot.selected_artifact_ids
        return SelectedRunInputs(active_id, base_id, config.correction_set, selected_ids), None
    except Exception as exc:
        # Existing corrupted pointer is a non-gating selection diagnostic.  It
        # cannot grant cache reuse, and a new unsealed run remains executable.
        finding = _selection_finding("selection", "base_revision_unavailable",
                                     f"Selected base revision is unavailable: {type(exc).__name__}.")
        return SelectedRunInputs(None, "unsealed", config.correction_set, ()), finding


def _selected_model_leaf_inputs(
    revisions: RevisionStore, selected: SelectedRunInputs,
) -> tuple[RevisionSnapshot | None, dict[str, str], dict[str, str]]:
    """Resolve exact prior leaf IDs from sealed final provenance, never cache history.

    The augmented snapshot names verified closure members so CacheValidator can
    validate transitive translation leaves while preserving one sealed revision
    as selection authority.  Mutable artifact reads remain fail-closed there.
    """
    if selected.base_revision_id == "unsealed":
        return None, {}, {}
    try:
        snapshot = revisions.snapshot(selected.base_revision_id)
        bundle = revisions.revisions_dir / snapshot.revision_id
        sealed = {
            path.stem: ArtifactEnvelope.from_file(path)
            for path in (bundle / "artifacts").glob("*.json")
        }
        # ``snapshot()`` verified every copied closure member.  Still reject a
        # malformed provenance shape rather than guessing from artifact age.
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        rows = provenance.get("segments")
        if not isinstance(rows, list):
            return None, {}, {}
        raw_by_segment: dict[str, str] = {}
        for artifact in sealed.values():
            if artifact.kind == "RawSourceExtraction":
                page_id = artifact.payload.get("page_id")
                segment_ids = artifact.payload.get("segment_artifact_ids")
                if isinstance(page_id, str) and isinstance(segment_ids, list):
                    for segment_id in segment_ids:
                        if isinstance(segment_id, str):
                            raw_by_segment[segment_id] = artifact.artifact_id
            elif artifact.kind == "DiagnosticSourceFallback":
                segment = artifact.payload.get("segment")
                if isinstance(segment, dict) and isinstance(segment.get("segment_id"), str):
                    raw_by_segment[segment["segment_id"]] = artifact.artifact_id
        source_ids: dict[str, str] = {}
        translation_ids: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("segment_id"), str):
                continue
            effective_source_id = row.get("effective_source_artifact_id")
            if isinstance(effective_source_id, str) and effective_source_id in sealed:
                raw_candidates = set(sealed[effective_source_id].dependency_ids) & set(raw_by_segment)
                if len(raw_candidates) == 1:
                    raw_id = raw_by_segment[raw_candidates.pop()]
                    page_id = sealed[raw_id].payload.get("page_id")
                    if isinstance(page_id, str):
                        prior = source_ids.setdefault(page_id, raw_id)
                        if prior != raw_id:
                            source_ids.pop(page_id, None)
            translation_id = row.get("translation_artifact_id")
            if isinstance(translation_id, str) and translation_id in sealed:
                prior = translation_ids.setdefault(row["segment_id"], translation_id)
                if prior != translation_id:
                    translation_ids.pop(row["segment_id"], None)
        closure_snapshot = RevisionSnapshot(
            revision_id=snapshot.revision_id,
            selected_artifact_ids=tuple(sorted(sealed)),
            selected_cache_attestation_ids=snapshot.selected_cache_attestation_ids,
        )
        return closure_snapshot, source_ids, translation_ids
    except (ArtifactError, OSError, ValueError, json.JSONDecodeError):
        return None, {}, {}


def _default_correction_set_id(workspace: Path) -> str | None:
    path = workspace / "active-correction-set.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if set(value) != {"set_id"} or not isinstance(value["set_id"], str) or not value["set_id"]:
            raise ValueError("invalid correction set pointer")
        return value["set_id"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _resolve_overlays(store: ArtifactStore, workspace: Path, selected: SelectedRunInputs,
                      requested_set_id: str | None) -> tuple[OverlayResolution, tuple[str, ...], str | None]:
    """Resolve only a matching sealed base/set; errors become inspectable no-op."""
    set_id = requested_set_id or _default_correction_set_id(workspace)
    if set_id is None:
        return OverlayResolution(selected.base_revision_id, None), (), None
    if selected.base_revision_id == "unsealed":
        finding = _selection_finding("corrections", "correction_set_inapplicable",
                                     "Correction set requires a selected sealed base revision.", (set_id,))
        store.put_finding(finding)
        return OverlayResolution(selected.base_revision_id, set_id, findings=(finding,)), (finding.finding_id,), set_id
    try:
        resolution = resolve_selected_overlays(CorrectionStore(workspace), RevisionStore(workspace),
                                               base_revision_id=selected.base_revision_id, correction_set_id=set_id)
        finding_ids = tuple(sorted(_store_finding(store, finding) for finding in resolution.findings))
        return resolution, finding_ids, set_id
    except (CorrectionError, OSError, ValueError) as exc:
        finding = _selection_finding("corrections", "correction_resolution_failed",
                                     f"Correction resolution fell back to no overlays: {type(exc).__name__}.", (set_id,))
        store.put_finding(finding)
        return OverlayResolution(selected.base_revision_id, set_id, findings=(finding,)), (finding.finding_id,), set_id


def _stage_summary(store: ArtifactStore, stage: str, status: str, counts: dict[str, int],
                   subjects: tuple[str, ...] = ()) -> str:
    finding = stage_summary_finding(stage, status, counts, subject_refs=subjects)
    store.put_finding(finding)
    return finding.finding_id


def _provenance(store: ArtifactStore, target_run: Any) -> tuple[SegmentProvenance, ...]:
    entries: list[SegmentProvenance] = []
    for leaf in target_run.leaves:
        for artifact_id in leaf.segment_artifact_ids:
            artifact = store.get(artifact_id)
            payload = artifact.payload
            # Target artifacts depend on translation/native effective-source
            # artifacts.  This references exact immutable closure, not text.
            source_id = artifact.dependency_ids[0] if artifact.dependency_ids else artifact_id
            effective_source_id = source_id
            translation_id = payload.get("translation_artifact_id")
            projection_ids: tuple[str, ...] = ()
            if translation_id:
                translation = store.get(translation_id)
                source_id = translation.payload.get("source_artifact_id", source_id)
                effective_source_id = source_id
                projection_ids = tuple(sorted(translation.payload.get("projection_ids", ())))
            entries.append(SegmentProvenance(
                segment_id=payload["segment_id"], page_id=leaf.page_id, source_artifact_id=source_id,
                effective_source_artifact_id=effective_source_id,
                effective_target_artifact_id=artifact_id, translation_artifact_id=translation_id,
                projection_artifact_ids=projection_ids,
                correction_ids=tuple(sorted(payload.get("correction_ids", ()))),
                finding_ids=tuple(sorted(payload.get("finding_ids", ()))),
            ))
    return tuple(sorted(entries, key=lambda item: item.segment_id))


async def _run_core(config: Config, on_page_error: PageErrorCallback | None = None) -> RunResult:
    """Execute immutable core stages through target materialization only."""
    errors: list[str] = []
    try:
        workspace = resolve_workspace(config).workspace
    except Exception as exc:
        failure = InvocationFailure.input_access(config.input_dir, exc)
        return RunResult([f"[btran] workspace unavailable: {type(exc).__name__}: {exc}"],
                         status="invocation_failed", invocation_failure=failure)
    store, graph, revisions = ArtifactStore(workspace), DependencyGraph(workspace), RevisionStore(workspace)
    selected, selection_finding = _selected_inputs(config, revisions)
    # Both explicit CLI selector and retained pointer become immutable run
    # inputs before any stage can inspect correction records.
    selected = replace(selected, correction_set_id=(config.correction_set or _default_correction_set_id(workspace)))
    executor = _CoreExecutor(config, workspace, selected, store, graph)
    selected_leaf_snapshot, selected_source_leaf_ids, selected_translation_leaf_ids = _selected_model_leaf_inputs(
        revisions, selected,
    )
    # Refresh explicitly requests fresh reachable model executions; ordinary
    # activated-base reruns use only exact sealed selections above.
    if config.refresh:
        selected_leaf_snapshot, selected_source_leaf_ids, selected_translation_leaf_ids = None, {}, {}

    discovery = discover_book(Path(config.input_dir), workspace)
    if discovery.invocation_failure is not None:
        return RunResult([f"[btran] invocation failed [{discovery.invocation_failure.code}] {discovery.invocation_failure.path}"],
                         status="invocation_failed", invocation_failure=discovery.invocation_failure, report=executor.report)
    assert discovery.book is not None
    empty_input = not discovery.pages

    async def discovery_stage(_: StageInputs) -> StageOutputs:
        book_id, finding_ids = _book_artifact(discovery, store)
        if selection_finding is not None:
            store.put_finding(selection_finding)
            finding_ids = tuple(sorted((*finding_ids, selection_finding.finding_id)))
        if empty_input:
            finding = _selection_finding("discovery", "no_supported_pages",
                                         "No supported pages found in readable input directory.")
            store.put_finding(finding)
            finding_ids = tuple(sorted((*finding_ids, finding.finding_id)))
        status = "degraded" if empty_input else "completed"
        summary = _stage_summary(store, "discovery", status, {"pages": len(discovery.pages), "findings": len(finding_ids)},
                                 tuple(page.page.page_id for page in discovery.pages))
        return StageOutputs(status, (book_id,), tuple(sorted((*finding_ids, summary))), (),
                            (CacheEvent("discovery", discovery.book.book_id, "produced", book_id),), discovery)

    discovered = await executor.stage("discovery", StageInputs("discovery", selected), discovery_stage)
    book_artifact_id = discovered.output_artifact_ids[0]

    # One raw-hash logical page supplies all model/effective work. Retain the
    # full discovery order separately for renderer placement materialization.
    logical_discovery: list[Any] = []
    seen_logical_pages: set[str] = set()
    for discovered_page in discovery.pages:
        if discovered_page.page.page_id not in seen_logical_pages:
            logical_discovery.append(discovered_page)
            seen_logical_pages.add(discovered_page.page.page_id)
    # Discovery placement is relative to input root. It is intentionally not
    # a logical/model input: rename/reorder must only alter output display.
    input_root = Path(config.input_dir).resolve()
    raw_inputs = tuple(RawPageInput(page.page.page_id, input_root / page.placement.relative_path,
                                    page.page.raw_file_sha256, number + 1)
                       for number, page in enumerate(logical_discovery))

    async def extraction_stage(_: StageInputs) -> StageOutputs:
        # Readable empty directories are degraded content, not input access
        # failures. Materialize one deterministic typed source leaf; it avoids
        # invoking source/terminology/translation models while preserving every
        # later effective-content and revision contract.
        result = (empty_input_diagnostic_raw_run(store=store, base_revision_id=selected.base_revision_id)
                  if empty_input else await extract_raw_pages(
                      raw_inputs, store=store, workspace=workspace, model=config.model,
                      pi_bin=config.pi_bin, max_retries=config.max_retries,
                      base_revision_id=selected.base_revision_id, concurrency=config.concurrency,
                      selected_snapshot=selected_leaf_snapshot,
                      selected_page_artifact_ids=selected_source_leaf_ids))
        # Assessment artifacts feed deterministic effective-source confidence.
        # Select/seal them with their raw leaves; later reruns must not recover
        # them from mutable global index history.
        roots = tuple(sorted({*(leaf.page_artifact_id for leaf in result.leaves),
                              *(assessment_id for leaf in result.leaves
                                for assessment_id in leaf.assessment_artifact_ids)}))
        finding_ids = tuple(sorted(set((result.stage_summary_finding_id,
                                        *(finding_id for leaf in result.leaves for finding_id in leaf.finding_ids)))))
        for index, leaf in enumerate(result.leaves):
            if not empty_input and leaf.degraded and on_page_error is not None:
                on_page_error(raw_inputs[index].page_number, "source extraction degraded; diagnostic content retained")
        return StageOutputs(result.status, roots, finding_ids, (), result.cache_events, result)

    raw = await executor.stage("source_extraction", StageInputs("source_extraction", selected,
                               (book_artifact_id,), discovered.finding_ids), extraction_stage)
    raw_run = raw.value

    async def corrections_stage(_: StageInputs) -> StageOutputs:
        resolution, resolution_findings, set_id = _resolve_overlays(store, workspace, selected, selected.correction_set_id)
        payload = {"base_revision_id": selected.base_revision_id, "correction_set_id": set_id,
                   "applicable_correction_ids": list(resolution.applicable_correction_ids)}
        artifact = store.put("CorrectionOverlaySelection", payload, finding_ids=resolution_findings,
                             semantic_key=tagged_sha256("correction-selection-v1", canonical_json(payload).encode("utf-8")))
        status = "degraded" if resolution_findings else "completed"
        summary = _stage_summary(store, "corrections", status,
                                 {"applicable": len(resolution.applicable_correction_ids), "findings": len(resolution_findings)})
        return StageOutputs(status, (artifact.artifact_id,), tuple(sorted((*resolution_findings, summary))), (),
                            (CacheEvent("corrections", selected.base_revision_id, "produced", artifact.artifact_id),), resolution)

    corrections = await executor.stage("corrections", StageInputs("corrections", selected,
                                      selected.base_snapshot_artifact_ids, ()), corrections_stage)
    overlays = corrections.value

    async def effective_source_stage(_: StageInputs) -> StageOutputs:
        fallback_finding: str | None = None
        try:
            result = materialize_effective_source(raw_run, store=store, graph=graph, source_overlays=overlays,
                                                  base_revision_id=selected.base_revision_id)
            status = result.status
        except Exception as exc:
            # A malformed correction resolution must not erase raw diagnostic
            # leaves. Re-materialize their typed source representation without
            # overlays; correction history remains persisted and inspectable.
            finding = _selection_finding("effective_source", "effective_source_exception",
                                         f"Effective-source stage fell back to raw content: {type(exc).__name__}.")
            fallback_finding = _store_finding(store, finding)
            result = materialize_effective_source(raw_run, store=store, graph=graph, source_overlays=(),
                                                  base_revision_id=selected.base_revision_id)
            status = "degraded"
        roots = tuple(sorted(leaf.page_artifact_id for leaf in result.leaves))
        finding_ids = set((result.stage_summary_finding_id,
                           *(finding_id for leaf in result.leaves for finding_id in leaf.finding_ids)))
        if fallback_finding is not None:
            finding_ids.add(fallback_finding)
            finding_ids.add(_stage_summary(store, "effective_source", "degraded",
                                            {"pages": len(result.leaves), "exception_fallbacks": 1}))
        return StageOutputs(status, roots, tuple(sorted(finding_ids)), result.graph_edge_ids,
                            tuple(CacheEvent("effective_source", leaf.page_id, "produced", leaf.page_artifact_id) for leaf in result.leaves), result)

    effective_source = await executor.stage("effective_source", StageInputs("effective_source", selected,
                                              tuple(sorted((*raw.output_artifact_ids, *corrections.output_artifact_ids))),
                                              tuple(sorted((*raw.finding_ids, *corrections.finding_ids)))), effective_source_stage)
    effective_source_run = effective_source.value

    async def terminology_stage(_: StageInputs) -> StageOutputs:
        fallback_finding: str | None = None
        try:
            pi_call = None if config.mode == "native" or empty_input else make_pi_consolidation_call(pi_bin=config.pi_bin, model=config.model, timeout=config.timeout)
            result = build_terminology_evidence(effective_source_run, store=store, graph=graph, mode=config.mode,
                                                target_lang=config.target_lang, terminology_overlays=overlays,
                                                pi_call=pi_call, base_revision_id=selected.base_revision_id,
                                                selected_evidence_shard_ids=selected.base_snapshot_artifact_ids,
                                                selected_membership_artifact_ids=selected.base_snapshot_artifact_ids,
                                                selected_projection_artifact_ids=selected.base_snapshot_artifact_ids,
                                                selected_snapshot=selected_leaf_snapshot,
                                                model_executable_identity=f"pi-bin:{config.pi_bin}", model_id=config.model,
                                                token_budget=config.glossary_budget)
            status = result.status
        except Exception as exc:
            # Local evidence/projections are a typed source-form fallback and
            # make target materialization executable without terminology model.
            finding = _selection_finding("terminology", "terminology_exception",
                                         f"Terminology stage used local fallback: {type(exc).__name__}.")
            fallback_finding = _store_finding(store, finding)
            result = build_terminology_evidence(effective_source_run, store=store, graph=graph, mode="native",
                                                terminology_overlays=(), base_revision_id=selected.base_revision_id,
                                                selected_evidence_shard_ids=selected.base_snapshot_artifact_ids,
                                                selected_membership_artifact_ids=selected.base_snapshot_artifact_ids,
                                                selected_projection_artifact_ids=selected.base_snapshot_artifact_ids,
                                                selected_snapshot=selected_leaf_snapshot,
                                                model_executable_identity=f"pi-bin:{config.pi_bin}", model_id=config.model,
                                                token_budget=config.glossary_budget)
            status = "degraded"
        roots = result.selected_artifact_ids
        finding_ids = set(result.finding_ids)
        if fallback_finding is not None:
            finding_ids.add(fallback_finding)
            finding_ids.add(_stage_summary(store, "terminology", "degraded",
                                            {"evidence_shards": len(result.evidence_leaves), "exception_fallbacks": 1}))
        return StageOutputs(status, roots, tuple(sorted(finding_ids)), result.graph_edge_ids,
                            tuple(CacheEvent("terminology", leaf.segment_id, "produced", leaf.evidence_shard_artifact_id)
                                  for leaf in result.evidence_leaves), result)

    terminology = await executor.stage("terminology", StageInputs("terminology", selected,
                                    effective_source.output_artifact_ids, effective_source.finding_ids), terminology_stage)
    terminology_run = terminology.value

    async def target_stage(_: StageInputs) -> StageOutputs:
        fallback_finding: str | None = None
        try:
            result = await materialize_effective_target(effective_source_run, terminology_run, store=store, graph=graph,
                                                        mode=config.mode, target_lang=config.target_lang,
                                                        target_overlays=overlays, model=config.model, pi_bin=config.pi_bin,
                                                        max_retries=config.max_retries,
                                                        base_revision_id=selected.base_revision_id,
                                                        selected_snapshot=selected_leaf_snapshot,
                                                        selected_translation_artifact_ids=selected_translation_leaf_ids)
            status = result.status
        except Exception as exc:
            # Reuse Task-10's per-segment fallback path. It emits target-page
            # and target-segment records even if all translation leaves fail.
            finding = _selection_finding("target_materialization", "target_materialization_exception",
                                         f"Target stage used diagnostic translation fallback: {type(exc).__name__}.")
            fallback_finding = _store_finding(store, finding)
            async def fail_translation(**_: Any) -> str:
                raise RuntimeError("outer target stage fallback")
            result = await materialize_effective_target(effective_source_run, (), store=store, graph=graph,
                                                        mode=config.mode, target_lang=config.target_lang,
                                                        target_overlays=(), model=config.model, pi_bin=config.pi_bin,
                                                        max_retries=config.max_retries,
                                                        base_revision_id=selected.base_revision_id,
                                                        segment_translator=fail_translation)
            status = "degraded"
        roots = tuple(sorted(leaf.page_artifact_id for leaf in result.leaves))
        finding_ids = {result.stage_summary_finding_id}
        for leaf in result.leaves:
            finding_ids.update(leaf.finding_ids)
        if fallback_finding is not None:
            finding_ids.add(fallback_finding)
            finding_ids.add(_stage_summary(store, "target_materialization", "degraded",
                                            {"pages": len(result.leaves), "exception_fallbacks": 1}))
        return StageOutputs(status, roots, tuple(sorted(finding_ids)), result.graph_edge_ids,
                            result.cache_events, result)

    target = await executor.stage("target_materialization", StageInputs("target_materialization", selected,
                                tuple(sorted((*effective_source.output_artifact_ids, *terminology.output_artifact_ids))),
                                tuple(sorted((*effective_source.finding_ids, *terminology.finding_ids)))), target_stage)
    target_run = target.value
    refresh_attempt_ids: tuple[str, ...] = ()
    if config.refresh:
        async def refresh_stage(_: StageInputs) -> StageOutputs:
            result = await _refresh_reachable_model_leaves(store=store, selected=selected, raw_run=raw_run,
                                                            terminology_run=terminology_run, target_run=target_run,
                                                            mode=config.mode, finding_ids=target.finding_ids)
            summary = _stage_summary(store, "refresh", "completed", {
                "reachable_model_leaves": len(result.attempt.reachable_artifact_ids),
                "refresh_attempts": 1,
            })
            return StageOutputs("completed", tuple(sorted({result.candidate_artifact_id, result.attempt_artifact_id})),
                                tuple(sorted((*target.finding_ids, summary))), (),
                                (CacheEvent("refresh", result.attempt.refresh_attempt_id, "produced",
                                            result.attempt_artifact_id),), result)

        refreshed = await executor.stage("refresh", StageInputs("refresh", selected,
                                         target.output_artifact_ids, target.finding_ids), refresh_stage)
        refresh_attempt_ids = (refreshed.value.attempt.refresh_attempt_id,)
    provenance = _provenance(store, target_run)
    # Task 14 owns reconciliation/validation/rendering/sealing and final report.
    if empty_input:
        errors.append("no supported pages found; diagnostic content rendered")
    return RunResult(errors, status="core_completed", report=executor.report, target_run=target_run,
                     terminology_run=terminology_run, provenance=provenance,
                     placements=((empty_input_diagnostic_placement(),) if empty_input else tuple(page.placement for page in discovery.pages)),
                     cache_events=tuple(executor.cache_events), refresh_attempt_ids=refresh_attempt_ids)


async def _refresh_reachable_model_leaves(*, store: ArtifactStore, selected: SelectedRunInputs,
                                            raw_run: Any, terminology_run: Any, target_run: Any,
                                            mode: str, finding_ids: tuple[str, ...]) -> Any:
    """Refresh returned model leaves without changing any revision pointer.

    Core stages already reinvoked current leaves. Pair their immutable returned
    IDs with base closure leaves by stable page/segment identity, then delegate
    durable candidate/attempt construction to Task 10. Source extraction is a
    model leaf even in native mode; translated runs also refresh terminology
    projections and translation leaves. An unsealed first run pairs returned
    leaves with themselves.
    """
    refreshable_kinds = {
        "RawSourceExtraction", "DiagnosticSourceFallback",
        "TranslationArtifact", "DiagnosticTranslationFallback", "ConceptProjection",
    }

    def subject(artifact: Any) -> tuple[Any, ...] | None:
        if artifact.kind not in refreshable_kinds:
            return None
        if isinstance(artifact.payload.get("page_id"), str):
            return ("page", artifact.payload["page_id"])
        if isinstance(artifact.payload.get("segment_id"), str):
            return ("segment", artifact.payload["segment_id"])
        if artifact.kind == "ConceptProjection" and isinstance(artifact.payload.get("concept_id"), str):
            selector = artifact.payload.get("selector_occurrence_ids")
            if isinstance(selector, list) and all(isinstance(item, str) for item in selector):
                return ("projection", artifact.payload["concept_id"], tuple(selector))
        return None

    returned: dict[tuple[Any, ...], str] = {}
    returned_ids = [leaf.page_artifact_id for leaf in raw_run.leaves]
    returned_ids.extend(artifact_id for leaf in target_run.leaves for artifact_id in leaf.translation_artifact_ids)
    # Native projections are deterministic local fallback, not model leaves.
    if mode == "translated":
        returned_ids.extend(getattr(terminology_run, "projection_artifact_ids", ()))
    for artifact_id in returned_ids:
        artifact = store.get(artifact_id)
        if (key := subject(artifact)) is not None:
            returned[key] = artifact.artifact_id

    if selected.base_revision_id == "unsealed":
        reachable = tuple(sorted(returned.values()))
        replacements = {artifact_id: artifact_id for artifact_id in reachable}
    else:
        base_artifacts, _ = store.closure(selected.base_snapshot_artifact_ids)
        reachable_pairs = sorted(
            (artifact.artifact_id, returned[key])
            for artifact in base_artifacts
            if (key := subject(artifact)) is not None and key in returned
        )
        reachable = tuple(old_id for old_id, _ in reachable_pairs)
        replacements = dict(reachable_pairs)

    async def returned_leaf(artifact_id: str) -> str:
        return replacements[artifact_id]

    return await refresh_model_leaves(store=store, base_revision_id=selected.base_revision_id,
                                      reachable_artifact_ids=reachable, refresh_leaf=returned_leaf,
                                      selected_finding_ids=finding_ids,
                                      correction_set_id=selected.correction_set_id)


def _target_projection_ids(target_run: Any, store: ArtifactStore) -> tuple[str, ...]:
    """Read exact Task-10 selected projection closure from translation artifacts."""
    values: set[str] = set()
    for leaf in target_run.leaves:
        for segment_id in leaf.segment_artifact_ids:
            segment = store.get(segment_id)
            translation_id = segment.payload.get("translation_artifact_id")
            if translation_id:
                values.update(store.get(translation_id).payload.get("projection_ids", ()))
    return tuple(sorted(values))


def _sealed_content(target_run: Any, store: ArtifactStore, placements: tuple[Any, ...] = ()) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    """Close logical Task-10 content, then apply ordered physical placements."""
    pages: list[EffectivePage] = []
    segments: list[EffectiveSegment] = []
    page_ids: list[str] = []
    segment_ids: list[str] = []
    pages_by_logical_id: dict[str, EffectivePage] = {}
    for leaf in target_run.leaves:
        page = store.get(leaf.page_artifact_id)
        if page.kind != "EffectiveTargetPage":
            raise ValueError("target materialization did not produce an effective target page")
        effective_page = EffectivePage.from_dict(page.payload)
        if effective_page.page_id in pages_by_logical_id:
            raise ValueError("target materialization duplicated a logical page")
        pages_by_logical_id[effective_page.page_id] = effective_page
        pages.append(effective_page); page_ids.append(page.artifact_id)
        for child_id in page.dependency_ids:
            child = store.get(child_id)
            if child.kind not in {"EffectiveTargetSegment", "DiagnosticEffectiveTargetSegment"}:
                continue
            segments.append(EffectiveSegment.from_dict(child.payload)); segment_ids.append(child.artifact_id)
    render_placements = tuple(
        RenderPlacement(placement.placement_id, placement.page_id,
                        pages_by_logical_id[placement.page_id].effective_page_id, placement.relative_path)
        for placement in placements
    )
    return (seal_effective_content(tuple(pages), tuple(segments), render_placements),
            tuple(sorted(set(page_ids))), tuple(sorted(set(segment_ids))))


def _all_finding_ids(records: tuple[StageRecord, ...]) -> tuple[str, ...]:
    return tuple(sorted({finding_id for record in records for finding_id in record.finding_ids}))


def _report_groups(store: ArtifactStore, finding_ids: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    content: list[str] = []; uncertainty: list[str] = []; review: list[str] = []; failures: list[str] = []
    for finding_id in finding_ids:
        finding = store.get_finding(finding_id)
        if finding.kind == "review_request": review.append(finding_id)
        elif finding.kind == "uncertainty": uncertainty.append(finding_id)
        elif finding.severity == "error": failures.append(finding_id)
        else: content.append(finding_id)
    return tuple(sorted(content)), tuple(sorted(uncertainty)), tuple(sorted(review)), tuple(sorted(failures))


def _revision_id(selected_artifacts: tuple[str, ...], selected_findings: tuple[str, ...],
                 selected_cache_attestation_ids: tuple[str, ...], correction_set_id: str | None) -> str:
    return tagged_sha256("candidate-revision-v1", canonical_json({
        "selected_artifact_ids": list(selected_artifacts), "selected_finding_ids": list(selected_findings),
        "selected_cache_attestation_ids": list(selected_cache_attestation_ids),
        "correction_set_id": correction_set_id,
    }).encode("utf-8"))


def _embed_provenance(epub_path: Path, provenance: Mapping[str, Any]) -> None:
    """Add canonical provenance before sealing; renderer itself remains Task-12-only."""
    try:
        payload = canonical_json_bytes(dict(provenance))
        temporary = epub_path.with_suffix(epub_path.suffix + ".provenance.tmp")
        with zipfile.ZipFile(epub_path, "r") as source, zipfile.ZipFile(temporary, "w") as destination:
            for source_info in source.infolist():
                if source_info.filename != "META-INF/btran-provenance.json":
                    # Renderer ZIP timestamps are transport metadata. Normalize
                    # every copied member before hashing/sealing so a repeated
                    # identical candidate has identical EPUB bytes.
                    info = zipfile.ZipInfo(source_info.filename, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = source_info.compress_type
                    info.external_attr = source_info.external_attr
                    info.create_system = source_info.create_system
                    member = source.read(source_info.filename)
                    if source_info.filename in {"EPUB/content.opf", "OEBPS/content.opf"}:
                        # ebooklib emits wall-clock dcterms:modified metadata.
                        # It is not selected artifact state and must not make an
                        # otherwise identical revision non-reproducible.
                        member = re.sub(
                            rb'(<meta property=["\']dcterms:modified["\']>)[^<]*(</meta>)',
                            rb'\g<1>1980-01-01T00:00:00Z\g<2>', member,
                        )
                    destination.writestr(info, member)
            info = zipfile.ZipInfo("META-INF/btran-provenance.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            destination.writestr(info, payload)
        temporary.replace(epub_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise EpubInvocationError(f"unable to embed EPUB provenance: {exc}") from exc


def _write_report(workspace: Path, report: RunReport) -> Path:
    path = workspace / "reports" / f"{report.run_id}.json"
    _atomic_write(path, report.to_json())
    return path


def _invocation_failure(code: str, path: Path | str, error: BaseException) -> dict[str, str]:
    return {"code": code, "path": str(path), "exception_type": type(error).__name__, "message": str(error)}


def _execution_impact_finding(correction_id: str, error: BaseException) -> Finding:
    return Finding(
        kind="execution_impact_unavailable", severity="warning", stage="corrections",
        subject_refs=(correction_id,),
        evidence={"correction_id": correction_id, "exception_type": type(error).__name__, "reason": str(error)},
        message="execution impact unavailable",
    )


def _execution_impacts(
    workspace: Path, selected: SelectedRunInputs, produced: set[str],
) -> tuple[tuple[str, ...], tuple[Finding, ...]]:
    """Persist every usable execution observation against its original plan."""
    if not selected.correction_set_id or selected.base_revision_id == "unsealed":
        return (), ()
    try:
        corrections = CorrectionStore(workspace)
        correction_set = corrections.get_set(selected.correction_set_id)
    except (ArtifactError, CorrectionError, OSError, ValueError) as exc:
        finding = _execution_impact_finding(selected.correction_set_id, exc)
        CorrectionStore(workspace).persist_findings((finding,))
        return (), (finding,)

    result: list[str] = []
    findings: list[Finding] = []
    for correction_id in correction_set.active_correction_ids:
        try:
            # Load plan recorded when this correction was applied. Do not
            # readdress it under current set, which may include later events.
            plan = corrections.correction_time_impact(correction_id)
            categories = {name: tuple(getattr(plan, name)) for name in ("affected", "unaffected", "ambiguous", "protected")}
            regenerated = tuple(item for item in categories["affected"] if item["base_artifact_id"] not in produced)
            reused = tuple(item for name in ("unaffected", "protected") for item in categories[name]
                           if item["base_artifact_id"] in produced or item["base_artifact_id"] in selected.base_snapshot_artifact_ids)
            impact = CorrectionImpact(
                phase="execution", base_revision_id=plan.base_revision_id,
                projection_plan_id=plan.projection_plan_id, correction_id=plan.correction_id,
                correction_set_id=plan.correction_set_id, projected_universe=plan.projected_universe,
                affected=categories["affected"], unaffected=categories["unaffected"], ambiguous=categories["ambiguous"],
                protected=categories["protected"], reused=tuple(sorted(reused, key=lambda x: (x["stage"], x["subject_id"], x["base_artifact_id"]))),
                regenerated=tuple(sorted(regenerated, key=lambda x: (x["stage"], x["subject_id"], x["base_artifact_id"]))),
            )
            corrections.put_impact(impact)
            result.append(impact.projection_plan_id)
        except (ArtifactError, CorrectionError, OSError, ValueError) as exc:
            finding = _execution_impact_finding(correction_id, exc)
            corrections.persist_findings((finding,))
            findings.append(finding)
    return tuple(sorted(result)), tuple(findings)


async def _finalize(config: Config, core: RunResult) -> RunResult:
    """Task-14 fixed tail: reconciliation -> validation -> render -> seal -> report."""
    if core.status == "invocation_failed" or core.target_run is None or core.report is None:
        return core
    workspace = resolve_workspace(config).workspace
    store, graph, revisions = ArtifactStore(workspace), DependencyGraph(workspace), RevisionStore(workspace)
    selected, _ = _selected_inputs(config, revisions)
    selected = replace(selected, correction_set_id=config.correction_set or _default_correction_set_id(workspace))
    executor = _CoreExecutor(config, workspace, selected, store, graph)
    executor.records = list(core.report.stage_records)
    executor.cache_events = list(core.cache_events)
    executor.report = core.report
    target_run = core.target_run

    async def reconciliation_stage(_: StageInputs) -> StageOutputs:
        projections = getattr(core.terminology_run, "projection_artifact_ids", _target_projection_ids(target_run, store))
        result = reconcile_effective(effective_pages=target_run, projections=projections,
                                     store=store, base_revision_id=selected.base_revision_id)
        edges = []
        for parent in (*result.effective_page_artifact_ids, *result.projection_artifact_ids):
            edges.append(graph.put(graph.edge(stable_subject_id=parent, parent_artifact_id=parent,
                                               child_artifact_id=result.artifact_id, stage="reconciliation", edge_kind="selected_input")))
        return StageOutputs(result.status, (result.artifact_id,), result.finding_ids, tuple(sorted(edges)),
                            (CacheEvent("reconciliation", "selected-content", "produced", result.artifact_id),), result)

    reconciliation_inputs = tuple(sorted(
        tuple(getattr(target_run, "selected_artifact_ids", ())) + tuple(leaf.page_artifact_id for leaf in target_run.leaves)
    ))
    reconciliation = await executor.stage("reconciliation", StageInputs("reconciliation", selected,
        reconciliation_inputs, _all_finding_ids(tuple(executor.records))), reconciliation_stage)

    async def validation_stage(_: StageInputs) -> StageOutputs:
        result = validate_effective(effective_pages=target_run, reconciliation=reconciliation.value, store=store,
                                    base_revision_id=selected.base_revision_id, mode=config.mode)
        edges = [graph.put(graph.edge(stable_subject_id="validation", parent_artifact_id=reconciliation.value.artifact_id,
                                      child_artifact_id=result.artifact_id, stage="validation", edge_kind="reconciliation_input"))]
        return StageOutputs(result.status, (result.artifact_id,), result.finding_ids, tuple(edges),
                            (CacheEvent("validation", "selected-content", "produced", result.artifact_id),), result)

    validation = await executor.stage("validation", StageInputs("validation", selected,
        tuple(sorted((*reconciliation.output_artifact_ids, *(leaf.page_artifact_id for leaf in target_run.leaves)))), reconciliation.finding_ids), validation_stage)

    content, page_ids, segment_ids = _sealed_content(target_run, store, core.placements)
    render_input = store.put("SealedRenderInput", {"pages": [page.to_dict() for page in content.pages],
        "segments": [segment.to_dict() for segment in content.segments],
        "placements": [placement.to_dict() for placement in content.placements]}, dependency_ids=page_ids,
        semantic_key=tagged_sha256("sealed-render-input-v1", canonical_json({
            "pages": list(page_ids), "segments": list(segment_ids),
            "placements": [placement.to_dict() for placement in content.placements],
        }).encode("utf-8")))

    async def render_stage(_: StageInputs) -> StageOutputs:
        page_artifacts = {store.get(page_id).payload["page_id"]: page_id for page_id in page_ids}
        input_edges = tuple(sorted(graph.put(graph.edge(stable_subject_id=placement.placement_id,
            parent_artifact_id=page_artifacts[placement.page_id], child_artifact_id=render_input.artifact_id,
            stage="rendering", edge_kind="render_placement_input")) for placement in content.placements))
        result = build_epub(content, Path(config.output_epub), title=config.title, author=config.author,
                            epub_check=config.epub_check, epub_check_path=config.epub_check_path,
                            timeout_seconds=config.timeout, artifact_store=store)
        status = "degraded" if result.status != "completed" else "completed"
        summary = _stage_summary(store, "rendering", status, {"pages": len(page_ids), "segments": len(segment_ids), "findings": len(result.finding_ids)})
        rendered = store.put("RenderedEpub", {"output_filename": Path(config.output_epub).name, "status": result.status,
            "render_input_artifact_id": render_input.artifact_id}, dependency_ids=(render_input.artifact_id,),
            finding_ids=tuple(sorted((*result.finding_ids, summary))), semantic_key=tagged_sha256("rendered-epub-v1", canonical_json({"render_input": render_input.artifact_id, "status": result.status}).encode("utf-8")))
        rendered_edge = graph.put(graph.edge(stable_subject_id="epub", parent_artifact_id=render_input.artifact_id,
            child_artifact_id=rendered.artifact_id, stage="rendering", edge_kind="rendered_epub"))
        return StageOutputs(status, tuple(sorted((render_input.artifact_id, rendered.artifact_id))), tuple(sorted((*result.finding_ids, summary))),
                            tuple(sorted((*input_edges, rendered_edge))), (CacheEvent("rendering", "epub", "produced", rendered.artifact_id),), result)

    rendering_inputs = tuple(sorted(tuple(validation.output_artifact_ids) + tuple(page_ids)))
    rendering = await executor.stage("rendering", StageInputs("rendering", selected,
        rendering_inputs, validation.finding_ids), render_stage)

    selected_artifacts = tuple(sorted({artifact_id for record in executor.records for artifact_id in record.output_artifact_ids}))
    execution_impact_ids, execution_impact_findings = _execution_impacts(workspace, selected, set(selected_artifacts))
    selected_findings = tuple(sorted((*_all_finding_ids(tuple(executor.records)), *(finding.finding_id for finding in execution_impact_findings))))
    # Translation/raw leaves are often transitive beneath effective-page stage
    # roots. Seal their exact key attestations too, not only direct outputs.
    closure_artifacts, _ = store.closure(selected_artifacts, finding_ids=selected_findings)
    closure_ids = tuple(artifact.artifact_id for artifact in closure_artifacts)
    selected_cache_attestation_ids = store.attestation_ids_for(closure_ids)
    revision_id = _revision_id(selected_artifacts, selected_findings,
                               selected_cache_attestation_ids, selected.correction_set_id)
    snapshot = RevisionSnapshot(
        revision_id=revision_id, selected_artifact_ids=selected_artifacts,
        selected_finding_ids=selected_findings,
        selected_cache_attestation_ids=selected_cache_attestation_ids,
        correction_set_id=selected.correction_set_id,
    )
    execution_impact_records = [f"corrections/impacts/{plan_id}.execution.json" for plan_id in execution_impact_ids]
    # Run IDs are execution metadata, not revision provenance.  Keeping one in
    # sealed EPUB bytes would make identical unactivated reruns collide with
    # their immutable candidate despite identical selected inputs/outputs.
    target_leaf_by_page = {leaf.page_id: leaf for leaf in target_run.leaves}
    provenance = {"mode": config.mode, "segments": [item.to_dict() for item in core.provenance],
                  "placements": [{**placement.to_dict(),
                                  "effective_page_artifact_id": target_leaf_by_page[placement.page_id].page_artifact_id,
                                  "effective_segment_artifact_ids": list(target_leaf_by_page[placement.page_id].segment_artifact_ids)}
                                 for placement in content.placements],
                  "render_input_artifact_id": render_input.artifact_id, "reconciliation_artifact_id": reconciliation.value.artifact_id,
                  "validation_artifact_id": validation.value.artifact_id,
                  "correction_execution_projection_plan_ids": list(execution_impact_ids),
                  "correction_execution_impact_records": execution_impact_records}
    _embed_provenance(Path(config.output_epub), provenance)

    # Older Task-13 records intentionally do not retain edge IDs. Select only
    # persisted graph edges whose endpoints are in this exact candidate closure;
    # history from another run can therefore never leak into this bundle.
    closure_ids = set(closure_ids)
    candidate_edge_ids = tuple(sorted(
        edge_id for path in graph.edges_dir.glob("*.json")
        for edge_id in (path.stem,)
        if (edge := graph.get(edge_id)).parent_artifact_id in closure_ids and edge.child_artifact_id in closure_ids
    ))

    async def sealing_stage(_: StageInputs) -> StageOutputs:
        revisions.seal_bundle(snapshot, provenance, Path(config.output_epub), render_input_artifact_id=render_input.artifact_id,
                              edge_ids=candidate_edge_ids)
        candidate = store.put("CandidateRevision", snapshot.to_dict(), dependency_ids=selected_artifacts,
                              finding_ids=selected_findings, semantic_key=tagged_sha256("candidate-revision-v1", snapshot.to_json().encode("utf-8")))
        summary = _stage_summary(store, "candidate_seal", "completed", {"artifacts": len(selected_artifacts), "findings": len(selected_findings)})
        return StageOutputs("completed", (candidate.artifact_id,), (summary,), (),
                            (CacheEvent("candidate_seal", revision_id, "produced", candidate.artifact_id),), candidate)

    sealing = await executor.stage("candidate_seal", StageInputs("candidate_seal", selected, selected_artifacts, selected_findings), sealing_stage)
    final_findings = selected_findings
    content_findings, uncertainty, review, failures = _report_groups(store, final_findings)
    # Refresh leaves are immutable artifacts. Report only attempts in this
    # selected closure; unrelated historic attempts are not current-run state.
    refresh_attempt_ids = tuple(sorted(
        artifact.payload["refresh_attempt_id"] for artifact in closure_artifacts
        if artifact.kind == "RefreshAttempt" and isinstance(artifact.payload.get("refresh_attempt_id"), str)
    ))
    # Explicit target documents with retained source diagnostics or translation
    # fallbacks are usable output, but must report degraded completion rather
    # than appear equivalent to an all-translated run.
    completed_degraded = (rendering.value.status != "completed" or bool(core.errors)
                          or (config.mode == "translated" and target_run.status == "degraded"))
    report = RunReport(run_id=executor.report.run_id, mode=config.mode, content_finding_ids=content_findings,
        uncertainty_finding_ids=uncertainty, review_finding_ids=review, recoverable_failure_finding_ids=failures,
        cache_events=tuple(event.to_dict() for event in executor.cache_events),
        placement_provenance=tuple(provenance["placements"]),
        correction_execution_projection_plan_ids=execution_impact_ids, refresh_attempt_ids=refresh_attempt_ids,
        selected_base_revision_id=None if selected.base_revision_id == "unsealed" else selected.base_revision_id,
        candidate_revision_id=revision_id, active_revision_id=selected.active_revision_id,
        final_epub_status="completed_degraded" if completed_degraded else rendering.value.status,
        stage_records=tuple(executor.records), total_stage_duration_ms=executor.report.total_stage_duration_ms)
    report_path = _write_report(workspace, report)
    store.put("RunReport", report.to_dict(), dependency_ids=(sealing.output_artifact_ids[0],), semantic_key=tagged_sha256("run-report-v1", report.to_json().encode("utf-8")))
    return RunResult(core.errors, status="completed_degraded" if completed_degraded else "completed",
                     report=report, target_run=target_run, terminology_run=core.terminology_run, provenance=core.provenance,
                     placements=core.placements, cache_events=tuple(executor.cache_events), candidate_revision_id=revision_id,
                     report_path=str(report_path), refresh_attempt_ids=refresh_attempt_ids)


async def run(config: Config, on_page_error: PageErrorCallback | None = None) -> RunResult:
    """Invocation boundary: file failures end promptly; quality failures render."""
    try:
        core = await _run_core(config, on_page_error=on_page_error)
        if core.status == "invocation_failed":
            failure = core.invocation_failure
            diagnostic = failure.to_dict() if failure is not None else _invocation_failure("input_access", config.input_dir, RuntimeError("unknown input failure"))
            if core.report is not None:
                report = replace(core.report, invocation_failures=(diagnostic,), final_epub_status="invocation_failed")
                report_path = _write_report(resolve_workspace(config).workspace, report)
                return replace(core, report=report, report_path=str(report_path))
            print(canonical_json(diagnostic), file=sys.stderr)
            return core
        return await _finalize(config, core)
    except EpubInvocationError as exc:
        try:
            workspace = resolve_workspace(config).workspace
            selected, _ = _selected_inputs(config, RevisionStore(workspace))
            report = initialized_report(run_id=uuid.uuid4().hex, mode=config.mode, selected=selected)
            report = replace(report, invocation_failures=(_invocation_failure("output_access", config.output_epub, exc),), final_epub_status="invocation_failed")
            report_path = _write_report(workspace, report)
            return RunResult([f"[btran] invocation failed [output_access] {config.output_epub}"], status="invocation_failed",
                             invocation_failure=_invocation_failure("output_access", config.output_epub, exc), report=report,
                             report_path=str(report_path))
        except Exception:
            print(canonical_json(_invocation_failure("output_access", config.output_epub, exc)), file=sys.stderr)
            return RunResult([f"[btran] invocation failed [output_access] {config.output_epub}"], status="invocation_failed",
                             invocation_failure=_invocation_failure("output_access", config.output_epub, exc))


async def orchestrator_run(config: Config, on_page_error: PageErrorCallback | None = None) -> RunResult:
    """Public full executor; finalization is deliberately never activation."""
    return await run(config, on_page_error=on_page_error)


__all__ = ["RunResult", "OrchestratorCallable", "build_epub", "orchestrator_run", "run"]
