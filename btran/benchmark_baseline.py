"""Frozen FC8 legacy benchmark corpus, callbacks, and baseline harness."""
from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from btran.config import Config, resolve_pi_session_dir
from btran.identity import occurrence_id_for
from btran.schema import PageExtraction, SourceBlock, TermMention, canonical_json_bytes

FIXTURE_VERSION = "deterministic-pipeline-optimization-fc8-v1"
FIXTURE_ROOT = Path(__file__).parent / "tests" / "fixtures" / "deterministic_pipeline_optimization"
CORPUS_DIR = FIXTURE_ROOT / "corpus"
BASELINE_PATH = FIXTURE_ROOT / "legacy-baseline.json"
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL7oAAAAABJRU5ErkJggg=="
PNG_BYTES = base64.b64decode(PNG_BASE64)
WAIT_MS = (137, 139, 149, 151, 157, 163)
P05_SOURCE = "美咲は量子センサーを校正する。"
P05_TERM_START = P05_SOURCE.index("量子センサー")
P05_TERM_END = P05_TERM_START + len("量子センサー")

# This table is intentionally literal. It is the frozen FC8 input, not a value
# inferred from a directory listing or a model response.
_ROWS: tuple[dict[str, Any], ...] = (
    {"page_id":"p01", "source_lang":"en", "source": ["Ada Lovelace", "Ada Lovelace documented the Analytical Engine.", "The Analytical Engine computes tables."], "mentions":[("Ada Lovelace", "proper_name"), ("Analytical Engine", "technical_term")], "target":["Ada Lovelace", "Ada Lovelace documented the Analytical Engine.", "The Analytical Engine computes tables."]},
    {"page_id":"p02", "source_lang":"ar", "source":["ليلى وجهاز التحليل", "تشرح ليلى جهاز التحليل.", "جهاز التحليل يحسب الجداول."], "mentions":[("ليلى", "proper_name"), ("جهاز التحليل", "technical_term")], "target":["Layla and the Analytical Engine", "Layla explains the Analytical Engine.", "The Analytical Engine calculates tables."]},
    {"page_id":"p03", "source_lang":"hi", "source":["रवि और क्वांटम सेंसर", "रवि क्वांटम सेंसर का परीक्षण करता है।", "क्वांटम सेंसर सटीक माप देता है।"], "mentions":[("रवि", "proper_name"), ("क्वांटम सेंसर", "technical_term")], "target":["Ravi and the quantum sensor", "Ravi tests the quantum sensor.", "The quantum sensor gives precise measurements."]},
    {"page_id":"p04", "source_lang":"te", "source":["అనిత మరియు సముద్ర పటం", "అనిత సముద్ర పటాన్ని పరిశీలిస్తుంది.", "సముద్ర పటం నౌకలకు దారి చూపుతుంది."], "mentions":[("అనిత", "proper_name"), ("సముద్ర పటం", "technical_term")], "target":["Anita and the sea map", "Anita examines the sea map.", "The sea map guides ships."]},
    {"page_id":"p05", "source_lang":"ja", "source":["美咲と量子センサー", P05_SOURCE, "量子センサーは正確な測定を行う。"], "mentions":[("美咲", "proper_name"), ("量子センサー", "technical_term")], "target":["Misaki and the quantum sensor", "Misaki calibrates the quantum sensor.", "The quantum sensor performs precise measurements."]},
    {"page_id":"p06", "source_lang":"en", "source":["Aurora Protocol", "The Aurora Protocol protects archive keys.", "The Aurora Protocol rotates archive keys."], "mentions":[("Aurora Protocol", "technical_term")], "target":["Aurora Protocol", "The Aurora Protocol protects archive keys.", "The Aurora Protocol rotates archive keys."]},
)
_TERMINOLOGY: dict[str, str] = {
    "Ada Lovelace":"Ada Lovelace", "Analytical Engine":"Analytical Engine", "ليلى":"Layla",
    "جهاز التحليل":"Analytical Engine", "रवि":"Ravi", "क्वांटम सेंसर":"quantum sensor",
    "అనిత":"Anita", "సముద్ర పటం":"sea map", "美咲":"Misaki", "量子センサー":"quantum sensor",
    "Aurora Protocol":"Aurora Protocol", "archive keys":"archive keys",
}
_TRANSLATIONS = {source: target for row in _ROWS for source, target in zip(row["source"], row["target"])}
_TRANSLATION_SEEN: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("fc8_translation_seen", default=None)
_DIRECT_TRANSLATION_SEEN: set[str] = set()


@dataclass
class _FC8TermMention(TermMention):
    category: str = "other"


@dataclass(frozen=True)
class BenchmarkCorpus:
    pages: tuple[dict[str, Any], ...]
    version: str = FIXTURE_VERSION

    @property
    def page_ids(self) -> tuple[str, ...]:
        return tuple(row["page_id"] for row in self.pages)

    def row(self, page_id: str) -> dict[str, Any]:
        return next(row for row in self.pages if row["page_id"] == page_id)


def load_corpus() -> BenchmarkCorpus:
    value = json.loads((FIXTURE_ROOT / "corpus.json").read_text(encoding="utf-8"))
    if value.get("version") != FIXTURE_VERSION or value.get("pages") != json.loads(json.dumps(_ROWS, ensure_ascii=False)):
        raise ValueError("FC8 corpus fixture was modified")
    for number in range(1, 7):
        if (CORPUS_DIR / f"page-{number:02d}.png").read_bytes() != PNG_BYTES:
            raise ValueError(f"FC8 image fixture was modified: page-{number:02d}.png")
    return BenchmarkCorpus(_ROWS)


def copy_corpus(destination: Path) -> Path:
    load_corpus()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for number in range(1, 7):
        shutil.copyfile(CORPUS_DIR / f"page-{number:02d}.png", destination / f"page-{number:02d}.png")
    return destination


def _illustration(page_number: int) -> dict[str, str]:
    return {"block_id": f"p{page_number:02d}:4", "path": f"page-{page_number:02d}.png", "alt": f"benchmark illustration {page_number:02d}"}


def page_extraction(page_number: int, image_path: Path, *, model: str = "fc8-vision") -> PageExtraction:
    """Return the exact FC8 callback artifact, including declared evidence.

    The legacy ``PageExtraction`` reader is deliberately used as a transport
    object here.  FC8 fields (mention categories and illustration descriptors)
    are retained on that object for the frozen callback contract; Wave 1 schema
    readers consume them as validated extraction semantics.
    """
    row = load_corpus().pages[page_number - 1]
    blocks = [SourceBlock(f"{row['page_id']}:{index}", kind, text, index - 1)
              for index, (text, kind) in enumerate(zip(row["source"], ("heading", "paragraph", "paragraph")), 1)]
    mentions: list[TermMention] = []
    for term, category in row["mentions"]:
        block = next(block for block in blocks if term in block.text)
        # The subclass makes category survive legacy ``asdict``/JSON transport;
        # it is not a dynamic attribute that could be silently discarded.
        mentions.append(_FC8TermMention(term, block.id, category))
    extraction = PageExtraction(page_number, str(image_path), hashlib.sha256(image_path.read_bytes()).hexdigest(), "0" * 16,
                                row["source_lang"], model, "2026-01-01T00:00:00Z", blocks, mentions, [_illustration(page_number)])
    return extraction


def extraction_callback(*args: Any, **kwargs: Any) -> PageExtraction:
    image_path = Path(args[0] if args else kwargs["image_path"])
    page_number = int(args[4] if len(args) > 4 else kwargs["page_number"])
    return page_extraction(page_number, image_path, model=str(args[1] if len(args) > 1 else kwargs.get("model", "fc8-vision")))


async def async_extraction_callback(*args: Any, **kwargs: Any) -> PageExtraction:
    # Six independent calls are gathered by the pipeline. The union is exactly
    # 10,244 ms, rather than six serial sleeps.
    await asyncio.sleep(10.244)
    return extraction_callback(*args, **kwargs)


def _target_for_text(text: str) -> str:
    return _TRANSLATIONS.get(text, text)


async def translation_callback(*, segment: Any, **_: Any) -> str:
    """Sleep once for each page (not once per segment or illustration)."""
    page = next((row for row in _ROWS if segment.source_text in row["source"]), None)
    if page is None:
        return _target_for_text(segment.source_text)
    seen = _TRANSLATION_SEEN.get()
    if seen is None:
        seen = _DIRECT_TRANSLATION_SEEN
    if page["page_id"] not in seen:
        seen.add(page["page_id"])
        await asyncio.sleep(WAIT_MS[_ROWS.index(page)] / 1000)
    return _target_for_text(segment.source_text)


def native_consolidation_callback(*_: Any, **__: Any) -> str:
    raise AssertionError("native benchmark must not invoke terminology consolidation")


async def native_translation_callback(*_: Any, **__: Any) -> str:
    raise AssertionError("native benchmark must not invoke translation")


def consolidation_callback(prompt: str) -> str:
    """Frozen response: protected declared terms plus one useful repetition."""
    time.sleep(0.131)
    # Validate that the callback receives the expected structured request, but
    # never let ordinary response terms (notably ``the``) become concepts.
    payload = json.loads(prompt.rsplit("\n", 1)[-1])
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("FC8 consolidation prompt is malformed")
    entries = [{"concept_id": term, "source_terms": [term], "target_term": target,
                "provenance": "declared" if term != "archive keys" else "automatic", "confidence": 1.0, "notes": ""}
               for term, target in _TERMINOLOGY.items()]
    return json.dumps({"entries": entries}, ensure_ascii=False, separators=(",", ":"))


class _Timing:
    def __init__(self) -> None:
        self.started_ns = time.perf_counter_ns()
        self.model: list[tuple[int, int]] = []
        self.persistence: list[tuple[int, int]] = []

    def add(self, bucket: list[tuple[int, int]], start: int, end: int) -> None:
        bucket.append((start, end))

    @staticmethod
    def union(intervals: list[tuple[int, int]]) -> int:
        total = 0
        current: tuple[int, int] | None = None
        for start, end in sorted(intervals):
            if current is None:
                current = (start, end)
            elif start <= current[1]:
                current = (current[0], max(current[1], end))
            else:
                total += current[1] - current[0]
                current = (start, end)
        if current is not None:
            total += current[1] - current[0]
        return total

    def report(self, completed_ns: int) -> dict[str, Any]:
        model = self.union(self.model)
        persistence = self.union(self.persistence)
        whole = max(0, completed_ns - self.started_ns)

        def complement(intervals: list[tuple[int, int]]) -> int:
            cursor = self.started_ns
            outside = 0
            for start, end in sorted(intervals):
                if start > cursor:
                    outside += start - cursor
                cursor = max(cursor, end)
            return outside + max(0, completed_ns - cursor)
        # Calculate complements from raw nanosecond intervals. No rounded-ms
        # subtraction is used, and persistence remains part of deterministic wall.
        events = sorted((start, end, "model") for start, end in self.model) + sorted((start, end, "persistence") for start, end in self.persistence)
        deterministic = 0
        cursor = self.started_ns
        for start, end, _ in sorted(events):
            if start > cursor:
                deterministic += start - cursor
            cursor = max(cursor, end)
        deterministic += max(0, completed_ns - cursor)
        deterministic_wall = complement(self.model)
        ms = lambda value: round(value / 1_000_000, 3)
        return {"clock": "perf_counter_ns", "run_origin": "run_benchmark_case",
                "report_snapshot_boundary": "before_run_report_persistence",
                "completion_boundary": "before_completion_timing_serialization",
                "model_execution_ms": ms(model), "deterministic_computation_ms": ms(deterministic),
                "durable_persistence_ms": ms(persistence), "deterministic_wall_ms": ms(deterministic_wall),
                "whole_process_wall_ms": ms(whole)}


@contextmanager
def patched_legacy_callbacks(mode: str = "translated", *, timing: _Timing | None = None,
                             stats: dict[str, int] | None = None) -> Iterator[None]:
    if mode not in {"native", "translated"}:
        raise ValueError("mode must be native or translated")
    from unittest.mock import patch
    timing = timing or _Timing()
    stats = stats if stats is not None else {}
    stats.setdefault("extraction", 0); stats.setdefault("translation", 0); stats.setdefault("consolidation", 0)
    token = _TRANSLATION_SEEN.set(set())

    async def timed_extraction(*args: Any, **kwargs: Any) -> PageExtraction:
        stats["extraction"] += 1
        start = time.perf_counter_ns()
        try:
            return await async_extraction_callback(*args, **kwargs)
        finally:
            timing.add(timing.model, start, time.perf_counter_ns())

    async def timed_translation(*, segment: Any, **kwargs: Any) -> str:
        stats["translation"] += 1
        start = time.perf_counter_ns()
        try:
            return await translation_callback(segment=segment, **kwargs)
        finally:
            timing.add(timing.model, start, time.perf_counter_ns())

    def timed_consolidation(prompt: str) -> str:
        stats["consolidation"] += 1
        start = time.perf_counter_ns()
        try:
            return consolidation_callback(prompt)
        finally:
            timing.add(timing.model, start, time.perf_counter_ns())

    def timed(method: Any, bucket: list[tuple[int, int]]) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter_ns()
            try:
                return method(*args, **kwargs)
            finally:
                timing.add(bucket, start, time.perf_counter_ns())
        return wrapper

    try:
        consolidation = native_consolidation_callback if mode == "native" else timed_consolidation
        translation = native_translation_callback if mode == "native" else timed_translation
        artifacts = __import__("btran.artifacts", fromlist=["_atomic_bytes", "_atomic_json"])
        orchestrator = __import__("btran.orchestrator", fromlist=["_atomic_write"])
        with patch("btran.source_extractor.extract_page", new=timed_extraction), \
             patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidation), \
             patch("btran.translator.translate_segment", new=translation), \
             patch("btran.artifacts._atomic_bytes", new=timed(artifacts._atomic_bytes, timing.persistence)), \
             patch("btran.artifacts._atomic_json", new=timed(artifacts._atomic_json, timing.persistence)), \
             patch("btran.orchestrator._atomic_write", new=timed(orchestrator._atomic_write, timing.persistence)):
            yield
    finally:
        _TRANSLATION_SEEN.reset(token)


def quality_oracle(mode: str) -> dict[str, Any]:
    corpus = load_corpus()
    if mode not in {"native", "translated"}:
        raise ValueError("mode must be native or translated")
    pages = []
    for row in corpus.pages:
        number = int(row["page_id"][1:])
        illustration = _illustration(number)
        declared = [[f"{row['page_id']}:1", "heading", f"{row['page_id']}:1", row["source"][0], None],
                    [f"{row['page_id']}:2", "paragraph", f"{row['page_id']}:2", row["source"][1], None],
                    [f"{row['page_id']}:3", "paragraph", f"{row['page_id']}:3", row["source"][2], None],
                    [f"{row['page_id']}:4", "illustration", None, illustration["alt"], illustration]]
        page = {"page_id": row["page_id"], "declared": declared, "source_language": row["source_lang"]}
        if mode == "translated":
            page["target_text"] = row["target"]; page["target_language"] = "en"
        pages.append(page)
    result: dict[str, Any] = {"mode": mode, "pages": pages,
                              "spine_text": [text for row in corpus.pages for text in (row["target"] if mode == "translated" else row["source"])]}
    if mode == "translated":
        result["terminology"] = [{"source_forms": [source], "target": target} for source, target in _TERMINOLOGY.items()]
        result["occurrences"] = [{"id": occurrence_id_for("p05:2", P05_TERM_START, P05_TERM_END, "量子センサー"), "concept_forms": ["量子センサー"], "selected_target": "quantum sensor"}]
        result["occurrence_correction"] = {"page_id": "p05", "block_id": "p05:2", "source": "量子センサー", "target": "quantum sensor", "start": P05_TERM_START, "end": P05_TERM_END}
    return result


def canonical_quality_bytes(mode: str) -> bytes:
    return canonical_json_bytes(quality_oracle(mode))


def expected_dirty_set(segment_id: str = "p05:2", start: int = P05_TERM_START, end: int = P05_TERM_END) -> dict[str, Any]:
    if (start, end) != (P05_TERM_START, P05_TERM_END):
        raise ValueError("FC8 correction must use the actual p05:2 occurrence span")
    return {"correction_id": "fc8-p05-quantum-sensor", "occurrence_id": occurrence_id_for(segment_id, start, end, "量子センサー"),
            "dirty_segment_ids": [segment_id], "dirty_page_ids": ["p05"]}


def state_files(workspace: Path) -> tuple[tuple[str, int], ...]:
    root = Path(workspace)
    return tuple(sorted((path.relative_to(root).as_posix(), path.stat().st_size)
                        for path in root.rglob("*") if path.is_file() and not path.is_symlink()))


def state_measure(workspace: Path) -> dict[str, Any]:
    files = state_files(workspace)
    return {"files": [path for path, _ in files], "file_count": len(files), "bytes": sum(size for _, size in files)}


def _state_snapshot(workspace: Path) -> dict[str, tuple[str, int]]:
    root = Path(workspace)
    return {path.relative_to(root).as_posix(): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


def _corruption_probe(workspace: Path, parent: Path) -> dict[str, Any]:
    original = _state_snapshot(workspace)
    disposable = parent / "corrupt-disposable-workspace"
    shutil.copytree(workspace, disposable)
    victim = next((path for path in disposable.rglob("*") if path.is_file() and not path.is_symlink()), None)
    if victim is None:
        return {"disposable_copy_only": True, "corruption_observable": False, "source_unchanged": True}
    before = victim.read_bytes()
    victim.write_bytes(before + b"\0fc8-corruption")
    copy_changed = hashlib.sha256(victim.read_bytes()).hexdigest() != original[victim.relative_to(disposable).as_posix()][0]
    unchanged = _state_snapshot(workspace) == original
    return {"disposable_copy_only": disposable.parent == parent, "corruption_observable": copy_changed,
            "source_unchanged": unchanged, "original_hashes_mtimes_unchanged": unchanged,
            "disposable_path": disposable.name}


def fixture_manifest() -> dict[str, Any]:
    paths = sorted(CORPUS_DIR.glob("page-*.png")) + [FIXTURE_ROOT / name for name in ("callbacks.json", "corpus.json", "oracles.json", "waits.json")]
    files = [{"path": path.relative_to(FIXTURE_ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in paths]
    return {"version": FIXTURE_VERSION, "files": files,
            "quality_sha256": {mode: hashlib.sha256(canonical_quality_bytes(mode)).hexdigest() for mode in ("native", "translated")}}


def _run_once(config: Config, mode: str, timing: _Timing, stats: dict[str, int]) -> Any:
    with patched_legacy_callbacks(mode, timing=timing, stats=stats):
        from btran.orchestrator import run
        return asyncio.run(run(config))


def run_benchmark_case(mode: str, root: Path | None = None) -> dict[str, Any]:
    """Run cold and warm legacy cases and write a sibling JSON measurement."""
    started = time.perf_counter_ns()
    if mode not in {"native", "translated"}:
        raise ValueError("mode must be native or translated")
    parent = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="btran-fc8-"))
    parent.mkdir(parents=True, exist_ok=True)
    corpus_dir, workspace, output = parent / "corpus", parent / "workspace", parent / "output.epub"
    if not corpus_dir.exists(): copy_corpus(corpus_dir)
    workspace.mkdir(exist_ok=True)
    resolve_pi_session_dir(workspace)
    config = Config(input_dir=corpus_dir, workspace=workspace, output_epub=output, target_lang="en" if mode == "translated" else None,
                    max_retries=0, concurrency=6, timeout=120)
    cold_timing, cold_stats = _Timing(), {}
    cold_timing.started_ns = started
    result = _run_once(config, mode, cold_timing, cold_stats)
    cold_completed = time.perf_counter_ns()
    warm_timing, warm_stats = _Timing(), {}
    # Pin the second invocation to the exact sealed result of the cold run;
    # this is the selected-authority cache exercise, not a fresh unsealed run.
    warm_config = replace(config, base_revision=result.candidate_revision_id) if result.candidate_revision_id else config
    warm_result = _run_once(warm_config, mode, warm_timing, warm_stats)
    completed = time.perf_counter_ns()
    report = result.report
    measurement = {"version": FIXTURE_VERSION, "mode": mode, "status": result.status, "workspace": str(workspace),
                   "output": str(output), "output_json": str(parent / f"{mode}-baseline.json"),
                   "timing": cold_timing.report(cold_completed), "warm_timing": warm_timing.report(completed),
                   "state": state_measure(workspace), "quality": quality_oracle(mode),
                   "quality_sha256": hashlib.sha256(canonical_quality_bytes(mode)).hexdigest(),
                   "report_non_actionable_finding_count": getattr(report, "non_actionable_finding_count", None) if report else None,
                   "terminology_oracle": _TERMINOLOGY if mode == "translated" else {},
                   "correction_oracle": expected_dirty_set() if mode == "translated" else None,
                   "cache_oracle": {"cold_model_calls": cold_stats, "warm_model_calls": warm_stats,
                                    "cold_model_call_count": sum(cold_stats.values()), "warm_model_call_count": sum(warm_stats.values()),
                                    "warm_cache_reuse": sum(warm_stats.values()) < sum(cold_stats.values())},
                   "corruption_oracle": _corruption_probe(workspace, parent)}
    output_json = parent / f"{mode}-baseline.json"
    output_json.write_bytes(canonical_json_bytes(measurement))
    return measurement


__all__ = ["BASELINE_PATH", "BenchmarkCorpus", "CORPUS_DIR", "FIXTURE_ROOT", "FIXTURE_VERSION", "PNG_BASE64", "async_extraction_callback", "canonical_quality_bytes", "consolidation_callback", "copy_corpus", "expected_dirty_set", "fixture_manifest", "load_corpus", "native_consolidation_callback", "native_translation_callback", "page_extraction", "patched_legacy_callbacks", "quality_oracle", "run_benchmark_case", "state_files", "state_measure", "translation_callback"]
