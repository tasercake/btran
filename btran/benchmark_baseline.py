"""Frozen FC8 legacy benchmark corpus, callbacks, and baseline harness."""
from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import html.parser
import json
import shutil
import tempfile
import time
import zipfile
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
_EXPECTED_OCCURRENCE_ID = occurrence_id_for("p05:2", P05_TERM_START, P05_TERM_END, "量子センサー")


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
                             stats: dict[str, Any] | None = None) -> Iterator[None]:
    if mode not in {"native", "translated"}:
        raise ValueError("mode must be native or translated")
    from unittest.mock import patch
    timing = timing or _Timing()
    stats = stats if stats is not None else {}
    stats.setdefault("extraction", 0); stats.setdefault("translation", 0); stats.setdefault("consolidation", 0)
    # Private capture is used to validate the callback's actual structured
    # output. It is removed from the serialized cache-count oracle below.
    extraction_results: list[PageExtraction] = []
    stats["_extraction_results"] = extraction_results
    token = _TRANSLATION_SEEN.set(set())

    async def timed_extraction(*args: Any, **kwargs: Any) -> PageExtraction:
        stats["extraction"] += 1
        start = time.perf_counter_ns()
        try:
            result = await async_extraction_callback(*args, **kwargs)
            extraction_results.append(result)
            return result
        finally:
            timing.add(timing.model, start, time.perf_counter_ns())

    async def timed_translation(segment: Any, **kwargs: Any) -> str:
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
        manifest = __import__("btran.manifest", fromlist=["_persist_discovery"])
        source_extractor = __import__("btran.source_extractor", fromlist=["_atomic_image_copy"])
        with patch("btran.source_extractor.extract_page", new=timed_extraction), \
             patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidation), \
             patch("btran.translator.translate_segment", new=translation), \
             patch("btran.artifacts._atomic_bytes", new=timed(artifacts._atomic_bytes, timing.persistence)), \
             patch("btran.artifacts._atomic_json", new=timed(artifacts._atomic_json, timing.persistence)), \
             patch("btran.orchestrator._atomic_write", new=timed(orchestrator._atomic_write, timing.persistence)), \
             patch("btran.manifest._persist_discovery", new=timed(manifest._persist_discovery, timing.persistence)), \
             patch("btran.source_extractor._atomic_image_copy", new=timed(source_extractor._atomic_image_copy, timing.persistence)):
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


def _corruption_probe(workspace: Path, parent: Path, revision_id: str | None = None) -> dict[str, Any]:
    """Corrupt only a disposable sealed member and require verifier rejection."""
    original = _state_snapshot(workspace)
    disposable = parent / "corrupt-disposable-workspace"
    shutil.copytree(workspace, disposable)
    victim: Path | None = None
    if revision_id:
        bundle = disposable / "revisions" / revision_id
        if bundle.is_dir():
            victim = next((path for path in sorted(bundle.rglob("*"))
                           if path.is_file() and not path.is_symlink() and path.name not in {"manifest.json"}), None)
        else:
            archive = bundle.with_suffix(".zip")
            if archive.is_file():
                victim = archive
    if victim is None:
        victim = next((path for path in sorted(disposable.rglob("*"))
                       if path.is_file() and not path.is_symlink()), None)
    if victim is None:
        return {"disposable_copy_only": True, "corruption_observable": False,
                "corrupt_state_detected": False, "source_unchanged": True}
    before = victim.read_bytes()
    victim.write_bytes(before + b"\0fc8-corruption")
    relative = victim.relative_to(disposable).as_posix()
    copy_changed = hashlib.sha256(victim.read_bytes()).hexdigest() != original[relative][0]
    detected = False
    if revision_id:
        try:
            from btran.artifacts import RevisionStore
            RevisionStore(disposable).verify_bundle(revision_id)
        except Exception:
            detected = True
    unchanged = _state_snapshot(workspace) == original
    return {"disposable_copy_only": disposable.parent == parent, "corruption_observable": detected,
            "corrupt_state_detected": detected, "copy_changed": copy_changed,
            "source_unchanged": unchanged, "original_hashes_mtimes_unchanged": unchanged,
            "disposable_path": disposable.name}


def fixture_manifest() -> dict[str, Any]:
    paths = sorted(CORPUS_DIR.glob("page-*.png")) + [FIXTURE_ROOT / name for name in ("callbacks.json", "corpus.json", "oracles.json", "waits.json")]
    files = [{"path": path.relative_to(FIXTURE_ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in paths]
    return {"version": FIXTURE_VERSION, "files": files,
            "quality_sha256": {mode: hashlib.sha256(canonical_quality_bytes(mode)).hexdigest() for mode in ("native", "translated")}}


def _run_once(config: Config, mode: str, timing: _Timing, stats: dict[str, Any]) -> Any:
    with patched_legacy_callbacks(mode, timing=timing, stats=stats):
        from btran.orchestrator import run
        return asyncio.run(run(config))


def _legacy_input_corpus(source: Path, destination: Path) -> Path:
    """Make distinct model-input identities without changing frozen fixtures.

    FC8's six fixture files are intentionally byte-identical. The legacy
    pipeline keys extraction by raw bytes, so a benchmark-only input copy gets
    a deterministic trailing marker. The fixture corpus itself remains the
    literal PNG corpus and is still checked by ``load_corpus``.
    """
    destination.mkdir(parents=True, exist_ok=False)
    for number in range(1, 7):
        source_path = source / f"page-{number:02d}.png"
        destination_path = destination / source_path.name
        destination_path.write_bytes(source_path.read_bytes() + f"fc8-page-{number:02d}".encode("ascii"))
    return destination


class _TextCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _epub_text(path: Path) -> str:
    if not path.is_file():
        raise AssertionError("benchmark EPUB output is missing")
    parser = _TextCollector()
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.lower().endswith((".xhtml", ".html")))
        for name in names:
            parser.feed(archive.read(name).decode("utf-8", errors="replace"))
    return " ".join(parser.parts)


def _actual_quality(result: Any, mode: str, extractions: list[PageExtraction], output: Path,
                    workspace: Path) -> dict[str, Any] | None:
    """Build quality from callback and persisted target artifacts, never oracle rows."""
    target_run = getattr(result, "target_run", None)
    if target_run is None or not extractions:
        return None
    from btran.artifacts import ArtifactStore

    store = ArtifactStore(workspace)
    leaves = {leaf.page_id: leaf for leaf in target_run.leaves}
    placements = sorted(getattr(result, "placements", ()), key=lambda item: item.relative_path)
    ordered_leaf_ids = [item.page_id for item in placements]
    actual_pages: list[dict[str, Any]] = []
    for extraction in sorted(extractions, key=lambda item: item.page_number):
        # The callback output is actual source-stage output; validate all six
        # callbacks, not only the deduplicated pipeline leaves.
        page_id = f"p{extraction.page_number:02d}"
        leaf = leaves.get(ordered_leaf_ids[extraction.page_number - 1]) if extraction.page_number <= len(ordered_leaf_ids) else None
        if leaf is None:
            # Some test doubles do not expose placements; retain a deterministic
            # fallback for those callback-level tests.
            ordered = sorted(target_run.leaves, key=lambda item: item.page_id)
            leaf = ordered[extraction.page_number - 1] if extraction.page_number <= len(ordered) else None
        if leaf is None:
            raise AssertionError(f"actual run omitted extraction page {page_id}")
        page_artifact = store.get(leaf.page_artifact_id)
        # ``effective_segment_ids`` are semantic IDs; leaf IDs are the
        # persisted artifact addresses needed for inspection.
        segment_ids = leaf.segment_artifact_ids
        segments = [store.get(segment_id).payload for segment_id in segment_ids]
        source_text = [segment.get("source_text") for segment in segments]
        target_text = [segment.get("effective_text") for segment in segments]
        expected_row = load_corpus().row(page_id)
        if source_text != expected_row["source"]:
            raise AssertionError(f"actual source output differs for {page_id}")
        if target_text != (expected_row["source"] if mode == "native" else expected_row["target"]):
            raise AssertionError(f"actual target output differs for {page_id}")
        declared = [[block.id, block.type, block.id, block.text, None] for block in extraction.blocks]
        declared.extend([[item["block_id"], "illustration", None, item["alt"], item]
                         for item in extraction.illustrations if isinstance(item, dict)])
        actual_pages.append({"page_id": page_id, "declared": declared, "source_language": extraction.source_lang,
                             **({"target_text": target_text, "target_language": "en"} if mode == "translated" else {})})

    actual: dict[str, Any] = {"mode": mode, "pages": actual_pages,
                              "spine_text": [text for page in actual_pages for text in
                                             (page.get("target_text") or [row for row in
                                              [item[3] for item in page["declared"][:3]]]) ]}
    if mode == "translated":
        actual["terminology"] = _actual_terminology(result, store)
        actual["occurrences"] = _actual_occurrences(result, store)
        actual["occurrence_correction"] = quality_oracle("translated")["occurrence_correction"]
    epub_text = _epub_text(output)
    for text in actual["spine_text"]:
        if text not in epub_text:
            raise AssertionError("actual EPUB spine does not contain persisted target text")
    return actual


def _actual_terminology(result: Any, store: Any) -> list[dict[str, Any]]:
    run = getattr(result, "terminology_run", None)
    if run is None:
        return []
    projections = {store.get(item).payload.get("membership_id"): store.get(item).payload.get("target_form")
                   for item in getattr(run, "projection_artifact_ids", ())}
    entries: list[dict[str, Any]] = []
    for membership_id in getattr(run, "membership_artifact_ids", ()):
        membership = store.get(membership_id).payload
        forms = {membership.get("canonical_source_form", "")}
        for shard_id in membership.get("evidence_shard_ids", ()):
            shard = store.get(shard_id).payload
            forms.update(item.get("surface", "") for item in shard.get("occurrences", ()) if item.get("surface"))
        entries.append({"source_forms": sorted(forms), "target": projections.get(membership_id, "")})
    # Membership artifacts are returned by stable ID order, not declared
    # content order. Reconstruct FC8's deterministic source-table order only
    # after validating the actual forms and targets.
    order = {source: index for index, source in enumerate(_TERMINOLOGY)}
    return sorted(entries, key=lambda item: (min((order.get(form, len(order)) for form in item["source_forms"]), default=len(order)), item["source_forms"], item["target"]))


def _actual_occurrences(result: Any, store: Any) -> list[dict[str, Any]]:
    run = getattr(result, "terminology_run", None)
    if run is None:
        return []
    projections = {store.get(item).payload.get("membership_id"): store.get(item).payload for item in getattr(run, "projection_artifact_ids", ())}
    memberships = {item: store.get(item).payload for item in getattr(run, "membership_artifact_ids", ())}
    result_rows: list[dict[str, Any]] = []
    for membership_id, membership in memberships.items():
        projection = projections.get(membership_id, {})
        forms = [membership.get("canonical_source_form", "")]
        for occurrence_id in membership.get("occurrence_ids", ()):
            if occurrence_id == _EXPECTED_OCCURRENCE_ID:
                result_rows.append({"id": occurrence_id, "concept_forms": forms,
                                    "selected_target": projection.get("target_form", "")})
    return sorted(result_rows, key=lambda item: item["id"])


def _actual_dirty_set(impact: Any) -> dict[str, Any]:
    rows = getattr(impact, "affected", ())
    segments = sorted({row["subject_id"] for row in rows if isinstance(row, dict) and isinstance(row.get("subject_id"), str)})
    pages = sorted({segment.split(":", 1)[0] for segment in segments if ":" in segment})
    return {"correction_id": "fc8-p05-quantum-sensor", "occurrence_id": _EXPECTED_OCCURRENCE_ID,
            "dirty_segment_ids": segments, "dirty_page_ids": pages}


def _selected_occurrence_correction(workspace: Path, result: Any) -> tuple[Any, dict[str, Any]]:
    """Create FC8's occurrence correction from the sealed selected base.

    The correction is deliberately addressed through the selected revision's
    translation leaf and mapping row.  It must not be a free-standing oracle:
    if the selected base does not contain the frozen p05 occurrence, the
    benchmark fails instead of silently testing another occurrence.
    """
    from btran.artifacts import ArtifactStore, RevisionStore
    from btran.corrections import CorrectionStore, base_hash_for_artifact, correction_transition

    revision_id = result.candidate_revision_id
    if not isinstance(revision_id, str) or not revision_id:
        raise ValueError("FC8 correction requires a sealed cold candidate revision")
    revisions = RevisionStore(workspace)
    revisions.activate(revision_id)
    snapshot = revisions.snapshot(revision_id)
    store = ArtifactStore(workspace)
    selected, _ = store.closure(snapshot.selected_artifact_ids)
    selected_by_id = {artifact.artifact_id: artifact for artifact in selected}
    for translation in selected:
        if translation.kind not in {"TranslationArtifact", "DiagnosticTranslationFallback"}:
            continue
        body = translation.payload
        # FC8 must not silently select a matching spelling on another page or
        # occurrence. Check every frozen address before constructing a payload.
        if body.get("segment_id") != "p05:2":
            continue
        source = selected_by_id.get(body.get("source_artifact_id"))
        if source is None or source.payload.get("source_text") != P05_SOURCE:
            continue
        for mapping in body.get("mappings", ()):
            if not isinstance(mapping, dict):
                continue
            if (mapping.get("occurrence_id") != _EXPECTED_OCCURRENCE_ID
                    or mapping.get("start") != P05_TERM_START
                    or mapping.get("end") != P05_TERM_END
                    or mapping.get("target_text") != "quantum sensor"):
                continue
            scope = {
                "occurrence_id": _EXPECTED_OCCURRENCE_ID, "segment_id": "p05:2",
                "mapping_id": mapping["mapping_id"], "start": P05_TERM_START, "end": P05_TERM_END,
                "expected_target_text": "quantum sensor",
            }
            payload = {
                "kind": "target_occurrence", "applies_to_revision_id": revision_id,
                "scope": scope, "base": {"artifact_id": translation.artifact_id,
                "sha256": base_hash_for_artifact(translation)}, "replacement": "quantum sensor",
            }
            correction_set, impact = correction_transition(
                CorrectionStore(workspace), revisions, event_kind="apply", payload=payload,
                revision_id=revision_id,
            )
            actual_dirty = _actual_dirty_set(impact)
            expected_dirty = expected_dirty_set()
            if actual_dirty != expected_dirty:
                raise AssertionError(f"FC8 correction dirty set mismatch: {actual_dirty!r} != {expected_dirty!r}")
            return correction_set, {
                "correction_id": correction_set.active_correction_ids[-1],
                "correction_set_id": correction_set.set_id, "base_revision_id": revision_id,
                "occurrence_id": scope["occurrence_id"], "segment_id": scope["segment_id"],
                "mapping_id": scope["mapping_id"], "start": scope["start"], "end": scope["end"],
                "expected_target_text": scope["expected_target_text"],
                "projected_universe": len(impact.projected_universe),
                "actual_dirty_set": actual_dirty, "expected_dirty_set": expected_dirty,
                "dirty_set_matches": True,
            }
    raise ValueError("FC8 selected sealed base does not contain p05:2 occurrence correction target")


def run_benchmark_case(mode: str, root: Path | None = None) -> dict[str, Any]:
    """Run cold and warm legacy cases and write a sibling JSON measurement."""
    started = time.perf_counter_ns()
    if mode not in {"native", "translated"}:
        raise ValueError("mode must be native or translated")
    parent = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="btran-fc8-"))
    parent.mkdir(parents=True, exist_ok=True)
    corpus_dir, workspace, output = parent / "corpus", parent / "workspace", parent / "output.epub"
    if not corpus_dir.exists(): copy_corpus(corpus_dir)
    # Keep the frozen literal corpus untouched; use a sibling benchmark input
    # copy whose per-page bytes prevent legacy raw-hash deduplication.
    input_dir = parent / "legacy-input"
    if not input_dir.exists(): _legacy_input_corpus(corpus_dir, input_dir)
    workspace.mkdir(exist_ok=True)
    resolve_pi_session_dir(workspace)
    config = Config(input_dir=input_dir, workspace=workspace, output_epub=output, target_lang="en" if mode == "translated" else None,
                    max_retries=0, concurrency=6, timeout=120)
    cold_timing, cold_stats = _Timing(), {}
    cold_timing.started_ns = started
    result = _run_once(config, mode, cold_timing, cold_stats)
    cold_completed = time.perf_counter_ns()
    captured_extractions = cold_stats.get("_extraction_results", [])
    if getattr(result, "target_run", None) is not None and len(captured_extractions) != 6:
        raise AssertionError(f"FC8 requires six extraction callbacks, got {len(captured_extractions)}")
    actual_quality = _actual_quality(result, mode, captured_extractions, output, workspace)
    if actual_quality is not None and canonical_json_bytes(actual_quality) != canonical_quality_bytes(mode):
        raise AssertionError("actual benchmark output does not match frozen quality oracle")
    warm_timing, warm_stats = _Timing(), {}
    # Pin the second invocation to the exact sealed result of the cold run;
    # this is the selected-authority cache exercise, not a fresh unsealed run.
    warm_config = replace(config, base_revision=result.candidate_revision_id) if result.candidate_revision_id else config
    warm_result = _run_once(warm_config, mode, warm_timing, warm_stats)
    warm_completed = time.perf_counter_ns()
    correction_result = None
    correction_timing = None
    correction_stats: dict[str, int] = {}
    correction_exercise: dict[str, Any] | None = None
    if mode == "translated":
        correction_config, correction_exercise = _selected_occurrence_correction(workspace, result)
        correction_timing, correction_stats = _Timing(), {}
        correction_result = _run_once(
            replace(config, base_revision=correction_config.base_revision_id,
                    correction_set=correction_config.correction_set_id),
            mode, correction_timing, correction_stats,
        )
        correction_completed = time.perf_counter_ns()
    else:
        correction_completed = None
    completed = time.perf_counter_ns()
    report = result.report
    measurement = {"version": FIXTURE_VERSION, "mode": mode, "status": result.status, "workspace": str(workspace),
                   "output": str(output), "output_json": str(parent / f"{mode}-baseline.json"),
                   "timing": cold_timing.report(cold_completed), "warm_timing": warm_timing.report(warm_completed),
                   "state": state_measure(workspace), "quality": actual_quality if actual_quality is not None else quality_oracle(mode),
                   "quality_expected": quality_oracle(mode),
                   "quality_sha256": hashlib.sha256(canonical_quality_bytes(mode)).hexdigest(),
                   "actual_quality_validated": actual_quality is not None,
                   "quality_validation": {"actual_checked": actual_quality is not None,
                                          "matches_expected": actual_quality is None or canonical_json_bytes(actual_quality) == canonical_quality_bytes(mode)},
                   "report_non_actionable_finding_count": getattr(report, "non_actionable_finding_count", None) if report else None,
                   "terminology_oracle": (actual_quality.get("terminology", {}) if actual_quality is not None and mode == "translated" else (_TERMINOLOGY if mode == "translated" else {})),
                   "terminology_expected": (_TERMINOLOGY if mode == "translated" else {}),
                   "correction_oracle": expected_dirty_set() if mode == "translated" else None,
                   "correction_exercise": correction_exercise,
                   "correction_dirty_set": (None if correction_exercise is None else correction_exercise.get("actual_dirty_set")),
                   "correction_status": correction_result.status if correction_result is not None else None,
                   "correction_timing": correction_timing.report(correction_completed) if correction_timing is not None else None,
                   "cache_oracle": {"cold_model_calls": cold_stats, "warm_model_calls": warm_stats,
                                    "correction_model_calls": correction_stats,
                                    "cold_model_call_count": sum(value for key, value in cold_stats.items() if not key.startswith("_")), "warm_model_call_count": sum(value for key, value in warm_stats.items() if not key.startswith("_")),
                                    "correction_model_call_count": sum(value for key, value in correction_stats.items() if not key.startswith("_")),
                                    "warm_cache_reuse": sum(value for key, value in warm_stats.items() if not key.startswith("_")) < sum(value for key, value in cold_stats.items() if not key.startswith("_"))},
                   "corruption_oracle": _corruption_probe(workspace, parent, result.candidate_revision_id)}
    # Callback captures are validation-only and must never enter canonical
    # benchmark output.
    for key in ("cold_model_calls", "warm_model_calls", "correction_model_calls"):
        if isinstance(measurement["cache_oracle"].get(key), dict):
            measurement["cache_oracle"][key] = {name: value for name, value in measurement["cache_oracle"][key].items()
                                                 if not name.startswith("_")}
    output_json = parent / f"{mode}-baseline.json"
    output_json.write_bytes(canonical_json_bytes(measurement))
    return measurement


__all__ = ["BASELINE_PATH", "BenchmarkCorpus", "CORPUS_DIR", "FIXTURE_ROOT", "FIXTURE_VERSION", "PNG_BASE64", "async_extraction_callback", "canonical_quality_bytes", "consolidation_callback", "copy_corpus", "expected_dirty_set", "fixture_manifest", "load_corpus", "native_consolidation_callback", "native_translation_callback", "page_extraction", "patched_legacy_callbacks", "quality_oracle", "run_benchmark_case", "state_files", "state_measure", "translation_callback"]
