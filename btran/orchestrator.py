"""Main pipeline: scan images → check cache → translate uncached → write JSON."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from btran.config import Config
from btran.epub_builder import build_epub
from btran.hasher import ImageCache, compute_sha256, compute_phash
from btran.schema import ErrorResult, PageResult
from btran.translator import TranslationError, translate_image


@dataclass
class RunResult:
    """Result of an orchestrator run."""

    errors: list[str]


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content)
    tmp.rename(path)


async def run(
    config: Config,
    on_page_error: Callable[[int, str], None] | None = None,
) -> RunResult:
    """Main pipeline. Orchestrates the full translation workflow.

    Args:
        config: Pipeline configuration.
        on_page_error: Optional callback invoked immediately when a page
            reaches terminal failure. Receives (page_number, error_message).

    Returns:
        RunResult with collected errors.
    """

    # 1. Scan config.input_dir for image files (sorted by name)
    image_files = sorted(
        p
        for p in config.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        return RunResult(errors=[])

    # 2. Create config.intermediate_dir if it doesn't exist
    config.intermediate_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run manifest: track expected pages for this run ----
    run_id = uuid.uuid4().hex[:12]
    expected_pages = list(range(1, len(image_files) + 1))
    _atomic_write(
        config.intermediate_dir / ".run_manifest.json",
        json.dumps({"run_id": run_id, "expected_pages": expected_pages}),
    )

    # Clean stale page_*.json files from prior runs (outside expected range)
    for existing in sorted(config.intermediate_dir.glob("page_*.json")):
        try:
            pn_str = existing.stem.split("_", 1)[1]
            pn = int(pn_str)
        except (ValueError, IndexError):
            # Malformed filename → remove
            existing.unlink(missing_ok=True)
            continue
        if pn not in expected_pages:
            existing.unlink(missing_ok=True)

    # 3. Open ImageCache
    cache = ImageCache(config.cache_db)

    total = len(image_files)
    errors: list[str] = []

    # 4. For each image: hash it, check cache, build pending list
    pending: list[tuple[int, Path, str, str]] = []  # (page_number, path, sha256, phash)
    completed = 0
    failed = 0
    cache_lock = asyncio.Lock()

    for page_number, image_path in enumerate(image_files, start=1):
        sha256 = compute_sha256(image_path)
        phash = compute_phash(image_path)

        if not config.no_resume:
            cached = cache.lookup(sha256)
            if cached is not None:
                _write_intermediate(cached, config.intermediate_dir, page_number)
                completed += 1
                pct = int(completed / total * 100)
                print(f"✓ page {page_number}/{total} ({pct}%)")
                continue

            cached = cache.lookup_perceptual(phash)
            if cached is not None:
                _write_intermediate(cached, config.intermediate_dir, page_number)
                completed += 1
                pct = int(completed / total * 100)
                print(f"✓ page {page_number}/{total} ({pct}%)")
                continue

        pending.append((page_number, image_path, sha256, phash))

    # 5. Process pending with Semaphore
    sem = asyncio.Semaphore(config.concurrency)
    results_lock = asyncio.Lock()

    async def process_one(pn: int, img_path: Path, sha: str, ph: str) -> None:
        nonlocal completed, failed
        async with sem:
            result: PageResult | ErrorResult | None = None

            for attempt in range(config.max_retries):
                try:
                    result = await translate_image(
                        image_path=img_path,
                        source_lang=config.source_lang,
                        target_lang=config.target_lang,
                        model=config.model,
                        sha256=sha,
                        phash=ph,
                        page_number=pn,
                        pi_bin=config.pi_bin,
                        timeout=config.timeout,
                    )
                    result.retry_count = attempt
                    break
                except TranslationError as exc:
                    if attempt == config.max_retries - 1:
                        result = ErrorResult(
                            page_number=pn,
                            image_path=str(img_path),
                            error=str(exc),
                            retry_count=attempt + 1,
                            model=config.model,
                        )
                    else:
                        backoff = 0.5 * (2 ** attempt)
                        jitter = random.uniform(0, backoff * 0.2)
                        await asyncio.sleep(backoff + jitter)
                except Exception as exc:
                    # Capture unexpected task exceptions
                    result = ErrorResult(
                        page_number=pn,
                        image_path=str(img_path),
                        error=f"{type(exc).__name__}: {exc}",
                        retry_count=attempt + 1,
                        model=config.model,
                    )
                    break

            # Save result atomically
            out_path = config.intermediate_dir / f"page_{pn:04d}.json"
            assert result is not None
            result_json = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
            _atomic_write(out_path, result_json)

            if isinstance(result, PageResult):
                async with cache_lock:
                    cache.store(sha, ph, str(img_path), result)

            async with results_lock:
                if isinstance(result, ErrorResult):
                    failed += 1
                    error_msg = f"[btran] page {pn} failed: {result.error}"
                    errors.append(error_msg)
                    # Stream error immediately via callback
                    if on_page_error is not None:
                        on_page_error(pn, result.error)
                    symbol = "\u2717"
                else:
                    symbol = "\u2713"
                completed += 1
                pct = int(completed / total * 100)
                print(f"{symbol} page {pn}/{total} ({pct}%)")

    # 6. Run all pending tasks
    if pending:
        tasks = [asyncio.create_task(process_one(pn, ip, sha, ph)) for pn, ip, sha, ph in pending]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nInterrupted. Waiting for running tasks to finish...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    cache.close()

    # 8. Compile EPUB from intermediate JSON files (gated on all-pages success)
    _compile_epub(config, expected_pages, errors)

    # 9. Summary
    if failed:
        print(f"Done: {completed - failed}/{total} pages translated, {failed} failed")
    else:
        print(f"Done: {completed}/{total} pages translated")

    return RunResult(errors=errors)


async def orchestrator_run(
    config: Config,
    on_page_error: Callable[[int, str], None] | None = None,
) -> RunResult:
    """Async entry point returning a RunResult for CLI integration."""
    return await run(config, on_page_error=on_page_error)


def _compile_epub(
    config: Config,
    expected_pages: list[int],
    errors: list[str],
) -> None:
    """Load intermediate JSON files and build the EPUB.

    Gated: only builds EPUB if all expected pages produced valid PageResult
    files.  Stale/missing/malformed/error pages prevent EPUB creation.
    """
    pages: list[PageResult] = []
    missing_or_bad: list[int] = []

    for page_num in expected_pages:
        jf = config.intermediate_dir / f"page_{page_num:04d}.json"
        if not jf.exists():
            missing_or_bad.append(page_num)
            msg = f"[btran] page {page_num} missing from intermediate files"
            print(msg, file=sys.stderr)
            errors.append(msg)
            continue

        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError as exc:
            missing_or_bad.append(page_num)
            msg = f"[btran] page {page_num} intermediate file is malformed: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            continue

        if "error" in data:
            # ErrorResult — already tracked during processing
            missing_or_bad.append(page_num)
            continue

        try:
            pages.append(PageResult.from_dict(data))
        except Exception as exc:
            missing_or_bad.append(page_num)
            msg = f"[btran] page {page_num} failed to parse PageResult: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if missing_or_bad:
        print(
            f"[btran] skipping EPUB build: {len(missing_or_bad)} page(s) "
            f"incomplete — {missing_or_bad}",
            file=sys.stderr,
        )
        return

    if not pages:
        print("[btran] all pages failed — no content for EPUB.", file=sys.stderr)
        return

    build_epub(
        page_results=pages,
        output_path=config.output_epub,
        title=config.title,
        author=config.author,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        embed_images=config.embed_images,
    )
    print(f"EPUB written to {config.output_epub}")


def _write_intermediate(
    cached: PageResult, intermediate_dir: Path, page_number: int
) -> None:
    """Write a cached result to an intermediate JSON file (atomic)."""
    out_path = intermediate_dir / f"page_{page_number:04d}.json"
    cached.page_number = page_number
    result_json = json.dumps(cached.to_dict(), indent=2, ensure_ascii=False) + "\n"
    _atomic_write(out_path, result_json)
