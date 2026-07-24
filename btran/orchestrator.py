"""Main pipeline: scan images → check cache → translate uncached → write JSON."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

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


async def run(config: Config) -> None:
    """Main pipeline. Orchestrates the full translation workflow."""

    # 1. Scan config.input_dir for image files (sorted by name)
    image_files = sorted(
        p
        for p in config.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print("No images found in input directory.")
        return

    # 2. Create config.intermediate_dir if it doesn't exist
    config.intermediate_dir.mkdir(parents=True, exist_ok=True)

    # 3. Open ImageCache
    cache = ImageCache(config.cache_db)

    total = len(image_files)

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

            # Save result
            out_path = config.intermediate_dir / f"page_{pn:04d}.json"
            assert result is not None
            result.to_file(out_path)

            if isinstance(result, PageResult):
                async with cache_lock:
                    cache.store(sha, ph, str(img_path), result)

            async with results_lock:
                if isinstance(result, ErrorResult):
                    failed += 1
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

    # 8. Compile EPUB from intermediate JSON files
    _compile_epub(config)

    # 9. Summary
    if failed:
        print(f"Done: {completed - failed}/{total} pages translated, {failed} failed")
    else:
        print(f"Done: {completed}/{total} pages translated")


async def orchestrator_run(config: Config) -> RunResult:
    """Async entry point returning a RunResult for CLI integration."""
    await run(config)
    return RunResult(errors=[])


def _compile_epub(config: Config) -> None:
    """Load intermediate JSON files and build the EPUB."""
    json_files = sorted(config.intermediate_dir.glob("page_*.json"))
    if not json_files:
        print("No intermediate files found — skipping EPUB build.")
        return

    pages: list[PageResult] = []
    for jf in json_files:
        data = json.loads(jf.read_text())
        if "error" in data:
            continue  # skip failed pages
        pages.append(PageResult.from_dict(data))

    if not pages:
        print("All pages failed — no content for EPUB.")
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
    """Write a cached result to an intermediate JSON file."""
    out_path = intermediate_dir / f"page_{page_number:04d}.json"
    cached.page_number = page_number
    cached.to_file(out_path)
