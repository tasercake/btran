"""Fixed two-pass orchestration over the Gate 1 typed module contracts."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from btran.config import Config
from btran.epub_builder import build_epub
from btran.hasher import compute_phash, compute_sha256
from btran.manifest import ManifestValidationError, load_or_generate_manifest, manifest_page_paths
from btran.orchestrator_contract import PageErrorCallback, RunResult
from btran.preflight import normalize_exif_orientation_copy, preflight_manifest
from btran.reconciliation import ReconciliationResult, reconcile
from btran.review import ReviewItem, corrections, unresolved_items, write_items
from btran.schema import ErrorResult, PageExtraction, PageResult, TerminologyMap, TranslatedBlock
from btran.source_extractor import (
    extract_page,
    extraction_cache_identity,
    legacy_page_text,
    to_file,
    validate_extraction_artifact,
)
from btran.terminology import (
    HARD_TOKEN_CAP,
    MAX_TOKEN_BUDGET,
    consolidate_terminology,
    freeze_terminology,
    make_pi_consolidation_call,
)
from btran.translator import translate_blocks, translation_cache_identity
from btran.validators import check_block_id_correspondence, validate_page


T = TypeVar("T")
_GLOSSARY_REVIEW_THRESHOLD = 0.8


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = file.name
            json.dump(value, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise


# Kept for Issue #5 compatibility and tests that exercise atomic checkpoint writes.
def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _manifest_path(config: Config) -> Path:
    """Resolve a relative manifest against the requested input, never CWD."""
    candidate = Path(config.manifest_path)
    return candidate if candidate.is_absolute() else Path(config.input_dir).resolve() / candidate


def _run_manifest(path: Path, value: dict) -> None:
    _atomic_json(path, value)


def _record_stage(run_state: dict, path: Path, stage: str, status: str, **details: object) -> None:
    run_state["stages"][stage] = {"version": "1", "status": status, **details}
    _run_manifest(path, run_state)


def _error(errors: list[str], page: int, message: str, callback: PageErrorCallback | None) -> None:
    error = f"[btran] page {page} failed: {message}"
    errors.append(error)
    if callback is not None:
        callback(page, message)


def _source_artifact_hash(extraction: PageExtraction) -> str:
    data = json.dumps(extraction.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _translation_source_hash(
    extraction: PageExtraction,
    previous_page: PageExtraction | None,
    next_page: PageExtraction | None,
) -> str:
    """Bind a translation cache entry to its page plus true boundary excerpts."""
    def boundary(page: PageExtraction | None, tail: bool) -> dict | None:
        if page is None or not page.blocks:
            return None
        blocks = sorted(page.blocks, key=lambda block: block.reading_order)
        block = blocks[-1] if tail else blocks[0]
        return {"page": page.page_number, "block_id": block.id, "text": block.text}

    payload = {
        "source_artifact": _source_artifact_hash(extraction),
        "previous_page_tail": boundary(previous_page, True),
        "next_page_head": boundary(next_page, False),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _valid_translation_cache_entry(value: object, source: PageExtraction) -> list[TranslatedBlock] | None:
    """Return an exact cached block bijection, or fail closed to a model miss."""
    if not isinstance(value, list):
        return None
    try:
        translated = [TranslatedBlock.from_dict(item) for item in value]
    except (TypeError, ValueError, KeyError):
        return None
    source_ids = [block.id for block in source.blocks]
    if (
        len(translated) != len(source_ids)
        or [block.block_id for block in translated] != source_ids
        or any(not isinstance(block.translated_text, str) for block in translated)
    ):
        return None
    return translated


def _valid_source_cache_entry(
    artifact: PageExtraction,
    *,
    page_number: int,
    image_path: Path,
    sha256: str,
    phash: str,
    model: str,
) -> bool:
    try:
        validate_extraction_artifact(artifact)
    except Exception:
        return False
    return (
        artifact.page_number == page_number
        and artifact.image_path == str(image_path)
        and artifact.sha256 == sha256
        and artifact.phash == phash
        and artifact.model == model
    )


def _corpus_source_lang(extractions: dict[int, PageExtraction]) -> str:
    """Summarize the source language detected across the extracted corpus."""
    languages = {extraction.source_lang for extraction in extractions.values()}
    return next(iter(languages)) if len(languages) == 1 else "mixed"


def _result_from(extraction: PageExtraction, translated: list[TranslatedBlock], target_lang: str) -> PageResult:
    return PageResult(
        page_number=extraction.page_number, sha256=extraction.sha256, phash=extraction.phash,
        image_path=extraction.image_path, source_lang=extraction.source_lang, target_lang=target_lang,
        page_text=legacy_page_text(extraction),
        translated_text="\n\n".join(block.translated_text for block in translated),
        image_descriptions=list(extraction.illustrations), model=extraction.model,
        blocks=list(extraction.blocks), translated_blocks=list(translated),
        term_mentions=list(extraction.term_mentions), illustrations=list(extraction.illustrations),
    )


def _review_id(kind: str, key: str, page: int | None = None) -> str:
    digest = hashlib.sha256(f"{kind}\0{key}\0{page or 0}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def _next_glossary_version(version: str) -> str:
    if version.isdigit():
        return str(int(version) + 1)
    parts = version.split(".")
    if parts and parts[0].isdigit():
        return ".".join([str(int(parts[0]) + 1), *(["0"] * (len(parts) - 1))])
    return f"{version}-v2"


def _apply_review_corrections(glossary: TerminologyMap, reviewed: dict[str, str]) -> TerminologyMap:
    """Freeze operator-approved terminology changes before any translation call."""
    unknown = set(reviewed) - {entry.concept_id for entry in glossary.entries}
    if unknown:
        raise ValueError(f"review corrections reference unknown concepts: {sorted(unknown)}")
    entries = [replace(entry, target_term=reviewed.get(entry.concept_id, entry.target_term)) for entry in glossary.entries]
    if all(before.target_term == after.target_term for before, after in zip(glossary.entries, entries)):
        return glossary
    return freeze_terminology(
        entries,
        source_lang=glossary.source_lang,
        target_lang=glossary.target_lang,
        version=_next_glossary_version(glossary.version),
    )


def _initial_review_items(glossary: TerminologyMap, paths: dict[int, Path]) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    aliases: dict[str, list[str]] = {}
    for entry in glossary.entries:
        for term in entry.source_terms:
            aliases.setdefault(term.casefold(), []).append(entry.concept_id)
        if entry.confidence < _GLOSSARY_REVIEW_THRESHOLD:
            page = next((number for number, path in paths.items() if str(number) in entry.provenance[0]), None) if entry.provenance else None
            items.append(ReviewItem(
                _review_id("low-confidence", entry.concept_id, page), "low_confidence", True,
                {"concept_id": entry.concept_id, "confidence": entry.confidence, "target_term": entry.target_term,
                 "provenance": entry.provenance}, str(paths.get(page, "")), page,
            ))
    for alias, concepts in aliases.items():
        if len(set(concepts)) > 1:
            items.append(ReviewItem(
                _review_id("glossary-conflict", alias), "glossary_conflict", True,
                {"source_term": alias, "concept_ids": sorted(set(concepts))},
            ))
    return items


def _reconciliation_review_items(result: ReconciliationResult, paths: dict[int, Path]) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for issue in result.issues:
        for page in issue.pages or (None,):
            items.append(ReviewItem(
                _review_id("terminology", f"{issue.kind}:{issue.concept_id}", page), issue.kind, True,
                {"concept_id": issue.concept_id, "expected_target_term": issue.expected_target_term,
                 "observed_target_terms": list(issue.observed_target_terms)},
                str(paths.get(page, "")), page,
            ))
    return items


async def _parallel_pages(
    pages: list[tuple[int, Path]], concurrency: int,
    operation: Callable[[int, Path], Awaitable[T]],
    on_terminal_failure: PageErrorCallback | None = None,
) -> tuple[dict[int, T], dict[int, str]]:
    """Run independent page operations with bounded concurrency and collect failures."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(number: int, path: Path) -> tuple[int, T | None, str | None]:
        try:
            async with semaphore:
                return number, await operation(number, path), None
        except Exception as exc:  # per-page containment is a workflow requirement
            failure = f"{type(exc).__name__}: {exc}"
            if on_terminal_failure is not None:
                on_terminal_failure(number, failure)
            return number, None, failure

    completed = await asyncio.gather(*(one(number, path) for number, path in pages))
    successes = {number: value for number, value, failure in completed if failure is None and value is not None}
    failures = {number: failure for number, value, failure in completed if failure is not None}
    return successes, failures


async def _with_retries(action: Callable[[], Awaitable[T]], retries: int) -> T:
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return await action()
        except Exception as exc:
            last = exc
    assert last is not None
    raise last


async def run(config: Config, on_page_error: PageErrorCallback | None = None) -> RunResult:
    """Run extraction → frozen glossary → translation → one reconciliation → EPUB.

    Model leaves are called only after the preceding complete stage has passed.
    The manifest is the sole source of page identity and ordering.
    """
    errors: list[str] = []
    input_dir = Path(config.input_dir).resolve()
    work = Path(config.intermediate_dir)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / ".run_manifest.json"
    state: dict = {"run_id": uuid.uuid4().hex, "manifest": "", "expected_pages": [], "stages": {}}
    _run_manifest(checkpoint, state)

    # Gate 0: no model invocation before manifest, budget, and all-page preflight.
    if not 1 <= config.glossary_budget <= MAX_TOKEN_BUDGET or config.glossary_budget >= HARD_TOKEN_CAP:
        errors.append(f"[btran] glossary budget must be between 1 and {MAX_TOKEN_BUDGET}")
        _record_stage(state, checkpoint, "preflight", "failed", reason=errors[-1])
        return RunResult(errors)
    try:
        manifest_file = _manifest_path(config)
        manifest = load_or_generate_manifest(input_dir, manifest_file)
        if Path(manifest.input_dir).resolve() != input_dir:
            raise ManifestValidationError("manifest input_dir does not match requested input directory")
        pages = manifest_page_paths(manifest)
    except (OSError, ValueError, ManifestValidationError) as exc:
        errors.append(f"[btran] manifest failed: {exc}")
        _record_stage(state, checkpoint, "manifest", "failed", reason=str(exc))
        return RunResult(errors)
    state["manifest"] = str(manifest_file)
    state["expected_pages"] = [number for number, _ in pages]
    _record_stage(state, checkpoint, "manifest", "succeeded", total_pages=len(pages))
    if not pages:
        _record_stage(state, checkpoint, "epub", "skipped", reason="manifest has no pages")
        return RunResult(errors)

    preflight = preflight_manifest(manifest)  # Never honor no_preflight: it would silently bypass the gate.
    state["preflight"] = {"issues": [issue.__dict__ for issue in preflight.issues]}
    if not preflight.ok:
        errors.extend(f"[btran] preflight page {issue.page_number}: {issue.message}" for issue in preflight.blocking_issues)
        _record_stage(state, checkpoint, "preflight", "failed", blocking=len(preflight.blocking_issues))
        return RunResult(errors)
    _record_stage(state, checkpoint, "preflight", "succeeded", warnings=len(preflight.warnings))

    # Normalize only warned EXIF pages into work-owned copies after the entire preflight succeeds.
    warning_pages = {issue.page_number for issue in preflight.warnings if issue.check == "orientation"}
    processed: list[tuple[int, Path]] = []
    original_paths = dict(pages)
    for number, path in pages:
        if number in warning_pages:
            copy = work / "normalized" / f"page_{number:04d}{path.suffix.lower()}"
            copy.parent.mkdir(parents=True, exist_ok=True)
            normalize_exif_orientation_copy(path, copy)
            processed.append((number, copy))
        else:
            processed.append((number, path))

    hashes = {number: (compute_sha256(path), compute_phash(path)) for number, path in pages}
    _record_stage(state, checkpoint, "source_extraction", "running")

    async def extract(number: int, path: Path) -> PageExtraction:
        sha, phash = hashes[number]
        source_cache = work / "source_cache" / f"{extraction_cache_identity(sha, config.model)}.json"
        if not config.no_resume and source_cache.exists():
            try:
                cached = PageExtraction.from_file(source_cache)
            except Exception:
                cached = None
            if cached is not None and _valid_source_cache_entry(
                cached, page_number=number, image_path=original_paths[number], sha256=sha, phash=phash,
                model=config.model,
            ):
                to_file(cached, work / "source" / f"page_{number:04d}.json")
                return cached
        result = await _with_retries(lambda: extract_page(path, config.model, sha, phash, number, config.pi_bin, config.timeout), config.max_retries)
        # The artifact refers to the source photo, even if a safe normalized copy was sent to Pi.
        result.image_path = str(original_paths[number])
        to_file(result, work / "source" / f"page_{number:04d}.json")
        if _valid_source_cache_entry(
            result, page_number=number, image_path=original_paths[number], sha256=sha, phash=phash,
            model=config.model,
        ):
            to_file(result, source_cache)
        return result

    extracted, failures = await _parallel_pages(processed, config.concurrency, extract, on_page_error)
    for number in sorted(failures):
        _atomic_json(work / "source" / f"page_{number:04d}.error.json", ErrorResult(number, str(original_paths[number]), failures[number], config.max_retries, config.model).to_dict())
        _error(errors, number, failures[number], None)
    if failures or set(extracted) != set(state["expected_pages"]):
        _record_stage(state, checkpoint, "source_extraction", "failed", failed_pages=sorted(failures))
        _record_stage(state, checkpoint, "epub", "skipped", reason="source extraction gate")
        return RunResult(errors)
    # Treat the atomic checkpoint as the stage boundary: a missing or malformed
    # artifact is never silently replaced by an in-memory extraction.
    checkpointed: dict[int, PageExtraction] = {}
    for number in state["expected_pages"]:
        artifact = work / "source" / f"page_{number:04d}.json"
        try:
            checkpointed[number] = PageExtraction.from_file(artifact)
            validate_extraction_artifact(checkpointed[number])
        except Exception as exc:
            _error(errors, number, f"source checkpoint is missing or malformed: {exc}", on_page_error)
    if errors:
        _record_stage(state, checkpoint, "source_extraction", "failed", reason="source checkpoint gate")
        _record_stage(state, checkpoint, "epub", "skipped", reason="source checkpoint gate")
        return RunResult(errors)
    extracted = checkpointed
    _record_stage(state, checkpoint, "source_extraction", "succeeded")

    # Gate 1: complete source corpus before consolidation; only terminology Pi is allowed here.
    mentions = [mention for number in state["expected_pages"] for mention in extracted[number].term_mentions]
    detected_source_lang = _corpus_source_lang(extracted)
    _record_stage(state, checkpoint, "glossary", "running")
    try:
        if mentions:
            pi_call = make_pi_consolidation_call(pi_bin=config.pi_bin, model=config.model, timeout=config.timeout)
            glossary = consolidate_terminology(mentions, source_lang=detected_source_lang, target_lang=config.target_lang, pi_call=pi_call, token_budget=config.glossary_budget)
        else:
            glossary = freeze_terminology([], source_lang=detected_source_lang, target_lang=config.target_lang)
        glossary_file = Path(config.glossary_path) if Path(config.glossary_path).is_absolute() else work / Path(config.glossary_path)
        _atomic_json(glossary_file, glossary.to_dict())
    except Exception as exc:
        errors.append(f"[btran] glossary failed: {type(exc).__name__}: {exc}")
        _record_stage(state, checkpoint, "glossary", "failed", reason=str(exc))
        return RunResult(errors)
    state["glossary"] = {"version": glossary.version, "hash": glossary.hash}
    _record_stage(state, checkpoint, "glossary", "frozen", hash=glossary.hash, version=glossary.version)

    reviews = work / "needs_review"
    write_items(reviews, _initial_review_items(glossary, original_paths))
    if unresolved_items(reviews):
        errors.append("[btran] blocking glossary review items remain unresolved")
        _record_stage(state, checkpoint, "review", "blocked", count=len(unresolved_items(reviews)))
        _record_stage(state, checkpoint, "epub", "skipped", reason="review gate")
        return RunResult(errors)
    try:
        glossary = _apply_review_corrections(glossary, corrections(reviews))
    except Exception as exc:
        errors.append(f"[btran] review failed: {type(exc).__name__}: {exc}")
        _record_stage(state, checkpoint, "review", "failed", reason=str(exc))
        _record_stage(state, checkpoint, "epub", "skipped", reason="review gate")
        return RunResult(errors)
    if glossary.hash != state["glossary"]["hash"]:
        _atomic_json(work / "glossary.v2.json", glossary.to_dict())
        _atomic_json(glossary_file, glossary.to_dict())
        state["glossary"] = {"version": glossary.version, "hash": glossary.hash}
        _record_stage(state, checkpoint, "review", "applied", glossary_hash=glossary.hash)

    expected_numbers = state["expected_pages"]
    positions = {number: index for index, number in enumerate(expected_numbers)}

    async def translate(number: int, _: Path, frozen: TerminologyMap) -> list[TranslatedBlock]:
        source = extracted[number]
        index = positions[number]
        previous_page = extracted[expected_numbers[index - 1]] if index else None
        next_page = extracted[expected_numbers[index + 1]] if index + 1 < len(expected_numbers) else None
        identity = translation_cache_identity(source_artifact_hash=_translation_source_hash(source, previous_page, next_page), glossary_hash=frozen.hash, target_lang=config.target_lang, model=config.model)
        cache_path = work / "translation_cache" / f"{identity}.json"
        if not config.no_resume and cache_path.exists():
            try:
                cached = _valid_translation_cache_entry(json.loads(cache_path.read_text()), source)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                cached = None
            if cached is not None:
                return cached
        value = await _with_retries(lambda: translate_blocks(source, frozen, model=config.model, pi_bin=config.pi_bin, timeout=config.timeout, previous_page=previous_page, next_page=next_page), config.max_retries)
        _atomic_json(cache_path, [item.to_dict() for item in value])
        return value

    _record_stage(state, checkpoint, "translation", "running", glossary_hash=glossary.hash)
    translated, failures = await _parallel_pages(pages, config.concurrency, lambda n, p: translate(n, p, glossary), on_page_error)
    for number in sorted(failures):
        _error(errors, number, failures[number], None)
    if failures or set(translated) != set(state["expected_pages"]):
        _record_stage(state, checkpoint, "translation", "failed", failed_pages=sorted(failures))
        _record_stage(state, checkpoint, "epub", "skipped", reason="translation gate")
        return RunResult(errors)
    # Preserve the exact source/translation block bijection before terminology
    # reconciliation can inspect the corpus.
    malformed_translations: dict[int, list[str]] = {}
    for number in state["expected_pages"]:
        probe = _result_from(extracted[number], translated[number], glossary.target_lang)
        issues = check_block_id_correspondence(extracted[number], probe)
        if issues:
            malformed_translations[number] = issues
            _error(errors, number, "; ".join(issues), on_page_error)
    if malformed_translations:
        _record_stage(state, checkpoint, "translation", "failed", malformed_pages=malformed_translations)
        _record_stage(state, checkpoint, "epub", "skipped", reason="translation artifact gate")
        return RunResult(errors)
    _record_stage(state, checkpoint, "translation", "succeeded", glossary_hash=glossary.hash)

    # One and only one reconciliation inspection. Reviewed forms were frozen before translation.
    try:
        reconciliation = reconcile(glossary=glossary, extractions=[extracted[n] for n in state["expected_pages"]], translations=translated)
    except Exception as exc:
        errors.append(f"[btran] reconciliation failed: {type(exc).__name__}: {exc}")
        _record_stage(state, checkpoint, "reconciliation", "failed", reason=str(exc))
        _record_stage(state, checkpoint, "epub", "skipped", reason="reconciliation gate")
        return RunResult(errors)
    write_items(
        reviews,
        _reconciliation_review_items(reconciliation, original_paths),
        archive_stale=True,
    )
    if reconciliation.glossary_diff:
        glossary = reconciliation.glossary_v2
        state["glossary"] = {"version": glossary.version, "hash": glossary.hash}
        _atomic_json(work / "glossary.v2.json", glossary.to_dict())
        _atomic_json(glossary_file, glossary.to_dict())
        affected = [(number, original_paths[number]) for number in reconciliation.affected_pages]
        retried, failures = await _parallel_pages(affected, config.concurrency, lambda n, p: translate(n, p, glossary), on_page_error)
        translated.update(retried)
        for number in sorted(failures):
            _error(errors, number, failures[number], None)
    state["reconciliation"] = {"issues": [issue.__dict__ for issue in reconciliation.issues], "affected_pages": reconciliation.affected_pages, "glossary_changes": [change.__dict__ for change in reconciliation.glossary_diff]}
    _record_stage(state, checkpoint, "reconciliation", "succeeded", affected_pages=reconciliation.affected_pages)
    if failures or unresolved_items(reviews):
        if unresolved_items(reviews):
            errors.append("[btran] blocking terminology review items remain unresolved")
        _record_stage(state, checkpoint, "review", "blocked", count=len(unresolved_items(reviews)))
        _record_stage(state, checkpoint, "epub", "skipped", reason="reconciliation/review gate")
        return RunResult(errors)

    results: list[PageResult] = []
    validation: dict[str, dict[str, list[str]]] = {}
    for number in state["expected_pages"]:
        page = _result_from(extracted[number], translated[number], glossary.target_lang)
        validation[str(number)] = validate_page(extracted[number], page, glossary)
        _atomic_json(work / f"page_{number:04d}.json", page.to_dict())
        results.append(page)
    state["validation"] = validation
    blocking_validation = {page: failures for page, failures in validation.items() if any(failures.values())}
    if blocking_validation:
        errors.append(f"[btran] validation failed for pages: {', '.join(sorted(blocking_validation))}")
        _record_stage(state, checkpoint, "validation", "failed", pages=blocking_validation)
        _record_stage(state, checkpoint, "epub", "skipped", reason="validation gate")
        return RunResult(errors)
    _record_stage(state, checkpoint, "validation", "succeeded")

    try:
        build_epub(results, Path(config.output_epub), title=config.title, author=config.author,
                   target_lang=config.target_lang, embed_images=config.embed_images,
                   epub_check=config.epub_check, epub_check_path=config.epub_check_path)
    except Exception as exc:
        errors.append(f"[btran] EPUB failed: {type(exc).__name__}: {exc}")
        _record_stage(state, checkpoint, "epub", "failed", reason=str(exc), epubcheck=config.epub_check)
        return RunResult(errors)
    _record_stage(
        state, checkpoint, "epub", "succeeded",
        epubcheck={"enabled": config.epub_check, "result": "passed" if config.epub_check else "not_requested"},
    )
    return RunResult(errors)


async def orchestrator_run(config: Config, on_page_error: PageErrorCallback | None = None) -> RunResult:
    """Gate 1-compatible async entry point used by the CLI."""
    return await run(config, on_page_error=on_page_error)
