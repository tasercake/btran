"""Immutable contracts used by the pre-finalization pipeline executor.

This module deliberately contains no stage algorithms.  It makes stage inputs,
outputs, cache observations, provenance, and selected revision state explicit so
Task 14 can add finalization without reopening earlier stages.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from btran.config import Config
from btran.identity import PagePlacement
from btran.manifest import InvocationFailure
from btran.schema import RunReport, StageRecord


PageErrorCallback: TypeAlias = Callable[[int, str], None]
StageRunner: TypeAlias = Callable[["StageInputs"], "StageOutputs | Awaitable[StageOutputs]"]


@dataclass(frozen=True)
class CacheEvent:
    """One inspectable cache decision; no event selects an artifact by itself."""

    stage: str
    subject_id: str
    outcome: str
    artifact_id: str | None = None
    semantic_key: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.stage or not self.subject_id:
            raise ValueError("cache event needs stage and subject_id")
        if self.outcome not in {"hit", "miss", "produced", "rejected"}:
            raise ValueError("cache event outcome is invalid")
        if self.outcome == "hit" and (not self.artifact_id or not self.semantic_key):
            raise ValueError("cache hit needs explicit artifact_id and semantic_key")

    def to_dict(self) -> dict[str, str | None]:
        return {"stage": self.stage, "subject_id": self.subject_id, "outcome": self.outcome,
                "artifact_id": self.artifact_id, "semantic_key": self.semantic_key, "detail": self.detail}


@dataclass(frozen=True)
class SegmentProvenance:
    """Closure references required to explain one eventual rendered segment."""

    segment_id: str
    page_id: str
    source_artifact_id: str
    effective_source_artifact_id: str | None = None
    projection_artifact_ids: tuple[str, ...] = ()
    translation_artifact_id: str | None = None
    effective_target_artifact_id: str | None = None
    correction_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id or not self.page_id or not self.source_artifact_id:
            raise ValueError("segment provenance needs segment, page, and source artifact IDs")
        for name in ("projection_artifact_ids", "correction_ids", "finding_ids"):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id, "page_id": self.page_id,
            "source_artifact_id": self.source_artifact_id,
            "effective_source_artifact_id": self.effective_source_artifact_id,
            "projection_artifact_ids": list(self.projection_artifact_ids),
            "translation_artifact_id": self.translation_artifact_id,
            "effective_target_artifact_id": self.effective_target_artifact_id,
            "correction_ids": list(self.correction_ids), "finding_ids": list(self.finding_ids),
        }


@dataclass(frozen=True)
class SelectedRunInputs:
    """Explicit snapshot/correction selectors.  ``unsealed`` means first run."""

    active_revision_id: str | None
    base_revision_id: str
    correction_set_id: str | None
    base_snapshot_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.base_revision_id:
            raise ValueError("base_revision_id must be explicit")
        ids = tuple(self.base_snapshot_artifact_ids)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("base snapshot artifact IDs must be sorted and unique")
        object.__setattr__(self, "base_snapshot_artifact_ids", ids)


@dataclass(frozen=True)
class StageInputs:
    """Named immutable closure passed to one executable stage."""

    stage: str
    selected: SelectedRunInputs
    input_artifact_ids: tuple[str, ...] = ()
    input_finding_ids: tuple[str, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage input needs stage")
        for name in ("input_artifact_ids", "input_finding_ids"):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class StageOutputs:
    """Stage output IDs are persisted before executor creates ``StageRecord``."""

    status: str
    output_artifact_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    graph_edge_ids: tuple[str, ...] = ()
    cache_events: tuple[CacheEvent, ...] = ()
    value: Any = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "degraded"}:
            raise ValueError("stage output status must be completed or degraded")
        for name in ("output_artifact_ids", "finding_ids", "graph_edge_ids"):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "cache_events", tuple(self.cache_events))
        if not all(isinstance(item, CacheEvent) for item in self.cache_events):
            raise ValueError("cache_events must contain CacheEvent values")


@dataclass(frozen=True)
class StageContract:
    """Executable, immutable stage declaration."""

    stage: str
    runner: StageRunner

    def __post_init__(self) -> None:
        if not self.stage or not callable(self.runner):
            raise ValueError("stage contract needs name and callable runner")

    async def execute(self, inputs: StageInputs) -> StageOutputs:
        if inputs.stage != self.stage:
            raise ValueError("stage contract/input stage mismatch")
        result = self.runner(inputs)
        result = await result if inspect.isawaitable(result) else result
        if not isinstance(result, StageOutputs):
            raise TypeError("stage runner must return StageOutputs")
        return result


def initialized_report(*, run_id: str, mode: str, selected: SelectedRunInputs) -> RunReport:
    """Return non-final report state. Task 14 owns final report publication."""
    return RunReport(
        run_id=run_id, mode=mode, selected_base_revision_id=(None if selected.base_revision_id == "unsealed"
                                                              else selected.base_revision_id),
        active_revision_id=selected.active_revision_id, final_epub_status="not_finalized", stage_records=(),
    )


@dataclass
class RunResult:
    """Outcome at caller boundary; no EPUB/final report exists at Task 13."""

    errors: list[str]
    status: str | None = None
    invocation_failure: InvocationFailure | None = None
    report: RunReport | None = None
    target_run: Any = None
    terminology_run: Any = None
    provenance: tuple[SegmentProvenance, ...] = ()
    # Ordered physical output positions. They deliberately do not participate
    # in logical page/segment/model identities.
    placements: tuple[PagePlacement, ...] = ()
    cache_events: tuple[CacheEvent, ...] = ()
    candidate_revision_id: str | None = None
    report_path: str | None = None
    refresh_attempt_ids: tuple[str, ...] = ()


class OrchestratorCallable(Protocol):
    def __call__(self, config: Config, on_page_error: PageErrorCallback | None = None) -> Awaitable[RunResult]: ...
