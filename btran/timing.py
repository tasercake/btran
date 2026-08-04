"""Run-scoped monotonic timing contracts.

Timing is execution metadata.  It is never part of an artifact or semantic
identity.  Intervals are retained as nanoseconds until the final report is
serialized, which keeps nested model/persistence work deterministic.
"""
from __future__ import annotations

import math
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator


_CATEGORIES = frozenset({"model", "model_execution", "persistence", "durable_persistence"})


def _category(value: str) -> str:
    if value not in _CATEGORIES:
        raise ValueError(f"unknown timing category: {value}")
    return "model_execution" if value == "model" else "durable_persistence" if value == "persistence" else value


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += max(0, end - start)
            start, end = next_start, next_end
    return total + max(0, end - start)


class PausedStopwatch:
    """A direct monotonic stopwatch which can be paused without counting pause."""

    def __init__(self, *, clock=time.perf_counter_ns) -> None:
        self._clock = clock
        self._started = clock()
        self._last = self._started
        self._elapsed = 0
        self._running = True

    def pause(self) -> None:
        if self._running:
            now = self._clock()
            self._elapsed += max(0, now - self._last)
            self._last = now
            self._running = False

    def resume(self) -> None:
        if not self._running:
            self._last = self._clock()
            self._running = True

    @property
    def elapsed_ns(self) -> int:
        if self._running:
            return self._elapsed + max(0, self._clock() - self._last)
        return self._elapsed

    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    def stop(self) -> int:
        self.pause()
        return self.elapsed_ns

    def __enter__(self) -> "PausedStopwatch":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


DirectStopwatch = PausedStopwatch


@dataclass(frozen=True)
class TimingReport:
    clock: str
    run_origin: str
    report_snapshot_boundary: str
    completion_boundary: str
    model_execution_ms: float
    deterministic_computation_ms: float
    durable_persistence_ms: float
    deterministic_wall_ms: float
    whole_process_wall_ms: float

    def __post_init__(self) -> None:
        for name in (
            "model_execution_ms", "deterministic_computation_ms",
            "durable_persistence_ms", "deterministic_wall_ms", "whole_process_wall_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

    def to_dict(self) -> dict[str, object]:
        return {
            "clock": self.clock,
            "run_origin": self.run_origin,
            "report_snapshot_boundary": self.report_snapshot_boundary,
            "completion_boundary": self.completion_boundary,
            "model_execution_ms": self.model_execution_ms,
            "deterministic_computation_ms": self.deterministic_computation_ms,
            "durable_persistence_ms": self.durable_persistence_ms,
            "deterministic_wall_ms": self.deterministic_wall_ms,
            "whole_process_wall_ms": self.whole_process_wall_ms,
        }


@dataclass(frozen=True)
class CompletionTiming(TimingReport):
    """Final timing snapshot.  No write-back into RunReport is permitted."""


class _TimingContext:
    def __init__(self, ledger: "TimingLedger", category: str) -> None:
        self.ledger, self.category = ledger, _category(category)
        self.start: int | None = None

    def __enter__(self) -> "_TimingContext":
        self.start = self.ledger._enter(self.category)
        return self

    def __exit__(self, *_: object) -> None:
        self.ledger._exit(self.category, self.start)


class TimingLedger:
    """One invocation's timing ledger with outermost-union event accounting."""

    def __init__(self, run_origin: str = "run", *, clock=time.perf_counter_ns) -> None:
        if not isinstance(run_origin, str) or not run_origin:
            raise ValueError("run_origin must be non-empty")
        self.run_origin = run_origin
        self._clock = clock
        self.origin_ns = clock()
        self._intervals: dict[str, list[tuple[int, int]]] = {
            "model_execution": [], "durable_persistence": []
        }
        self._depth = {"model_execution": 0, "durable_persistence": 0}
        self._starts: dict[str, int | None] = {"model_execution": None, "durable_persistence": None}
        self._snapshot_ns: int | None = None
        self._completion_ns: int | None = None
        self._snapshot_taken = False
        self._completion_taken = False

    def _enter(self, category: str) -> int:
        if self._completion_taken:
            raise RuntimeError("timing ledger is finalized")
        now = self._clock()
        if self._depth[category] == 0:
            self._starts[category] = now
        self._depth[category] += 1
        return now

    def _exit(self, category: str, _start: int | None) -> None:
        if self._depth[category] <= 0:
            raise RuntimeError("timing context exited without entering")
        self._depth[category] -= 1
        if self._depth[category] == 0:
            start = self._starts[category]
            end = self._clock()
            if start is not None:
                self._intervals[category].append((start, max(start, end)))
            self._starts[category] = None

    def measure(self, category: str) -> _TimingContext:
        return _TimingContext(self, category)

    def measure_async(self, category: str):
        ledger = self
        normalized = _category(category)

        @asynccontextmanager
        async def context() -> AsyncIterator[None]:
            start = ledger._enter(normalized)
            try:
                yield
            finally:
                ledger._exit(normalized, start)

        return context()

    # Explicit names make call sites clear and keep the sync/async contracts
    # discoverable to integration tests.
    def model_execution(self) -> _TimingContext:
        return self.measure("model_execution")

    def durable_persistence(self) -> _TimingContext:
        return self.measure("durable_persistence")

    def model_execution_async(self):
        return self.measure_async("model_execution")

    def durable_persistence_async(self):
        return self.measure_async("durable_persistence")

    def _boundary_ms(self, end_ns: int) -> dict[str, object]:
        process = max(0, end_ns - self.origin_ns)
        model = _union_ns(self._intervals["model_execution"])
        persistence = _union_ns(self._intervals["durable_persistence"])
        # Exact interval arithmetic happens before conversion/rounding.
        deterministic_wall = max(0, process - model)
        deterministic = max(0, process - _union_ns(
            self._intervals["model_execution"] + self._intervals["durable_persistence"]
        ))
        return {
            "clock": "perf_counter_ns",
            "run_origin": self.run_origin,
            "report_snapshot_boundary": "before_run_report_persistence",
            "completion_boundary": "before_completion_timing_serialization",
            "model_execution_ms": round(model / 1_000_000, 3),
            "deterministic_computation_ms": round(deterministic / 1_000_000, 3),
            "durable_persistence_ms": round(persistence / 1_000_000, 3),
            "deterministic_wall_ms": round(deterministic_wall / 1_000_000, 3),
            "whole_process_wall_ms": round(process / 1_000_000, 3),
        }

    def snapshot_before_report_persist(self) -> dict[str, object]:
        if self._snapshot_taken:
            raise RuntimeError("report timing snapshot already taken")
        self._snapshot_taken = True
        self._snapshot_ns = self._clock()
        return self._boundary_ms(self._snapshot_ns)

    def complete_before_timing_serialization(self) -> CompletionTiming:
        if not self._snapshot_taken:
            raise RuntimeError("report snapshot must precede completion")
        if self._completion_taken:
            raise RuntimeError("completion timing already finalized")
        if any(self._depth.values()):
            raise RuntimeError("cannot finalize timing with active intervals")
        self._completion_taken = True
        self._completion_ns = self._clock()
        data = self._boundary_ms(self._completion_ns)
        return CompletionTiming(**data)

    @property
    def snapshot_taken(self) -> bool:
        return self._snapshot_taken

    @property
    def completion_taken(self) -> bool:
        return self._completion_taken

    def completion_timing(self) -> CompletionTiming:
        return self.complete_before_timing_serialization()


class NoOpTimingLedger(TimingLedger):
    """No-op implementation for callers which do not request instrumentation."""

    def __init__(self, run_origin: str = "noop", **_: object) -> None:
        self.run_origin = run_origin
        self.origin_ns = 0
        self._snapshot_taken = False
        self._completion_taken = False

    def measure(self, category: str) -> _TimingContext:
        _category(category)
        if self._completion_taken:
            raise RuntimeError("timing ledger is finalized")
        return _NoOpContext()

    def measure_async(self, category: str):
        _category(category)
        ledger = self

        @asynccontextmanager
        async def context() -> AsyncIterator[None]:
            if ledger._completion_taken:
                raise RuntimeError("timing ledger is finalized")
            yield

        return context()

    def model_execution(self) -> _TimingContext:
        return self.measure("model_execution")

    def durable_persistence(self) -> _TimingContext:
        return self.measure("durable_persistence")

    def model_execution_async(self):
        return self.measure_async("model_execution")

    def durable_persistence_async(self):
        return self.measure_async("durable_persistence")

    def snapshot_before_report_persist(self) -> dict[str, object]:
        if self._snapshot_taken:
            raise RuntimeError("report timing snapshot already taken")
        self._snapshot_taken = True
        return TimingReport("perf_counter_ns", self.run_origin, "before_run_report_persistence", "before_completion_timing_serialization", 0.0, 0.0, 0.0, 0.0, 0.0).to_dict()

    def complete_before_timing_serialization(self) -> CompletionTiming:
        if not self._snapshot_taken:
            raise RuntimeError("report snapshot must precede completion")
        if self._completion_taken:
            raise RuntimeError("completion timing already finalized")
        self._completion_taken = True
        return CompletionTiming("perf_counter_ns", self.run_origin, "before_run_report_persistence", "before_completion_timing_serialization", 0.0, 0.0, 0.0, 0.0, 0.0)


class _NoOpContext:
    def __enter__(self) -> "_NoOpContext": return self
    def __exit__(self, *_: object) -> None: return None


def noop_timing_ledger(run_origin: str = "noop") -> NoOpTimingLedger:
    return NoOpTimingLedger(run_origin)


def sync_timing(ledger: TimingLedger, category: str):
    return ledger.measure(category)


def async_timing(ledger: TimingLedger, category: str):
    return ledger.measure_async(category)


# Friendly aliases used by integration callers.
NoOpLedger = NoOpTimingLedger
Ledger = TimingLedger
no_op_ledger = noop_timing_ledger
outermost_sync = sync_timing
outermost_async = async_timing
paused_stopwatch = PausedStopwatch

def model_execution(ledger: TimingLedger):
    return ledger.measure("model_execution")


def durable_persistence(ledger: TimingLedger):
    return ledger.measure("durable_persistence")


@asynccontextmanager
async def model_execution_async(ledger: TimingLedger):
    async with ledger.measure_async("model_execution"):
        yield


@asynccontextmanager
async def durable_persistence_async(ledger: TimingLedger):
    async with ledger.measure_async("durable_persistence"):
        yield
