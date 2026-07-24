"""CLI entry point. Parses args, loads config, and invokes Gate 1 orchestration."""

from __future__ import annotations

import asyncio
import shutil
import sys

from btran.config import load_config
from btran.orchestrator import orchestrator_run
from btran.orchestrator_contract import OrchestratorCallable, RunResult


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def main() -> None:
    """Run the production CLI against the frozen orchestrator callable."""
    try:
        config = load_config()
    except SystemExit:
        raise
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if not config.input_dir.exists():
        print(f"Error: input_dir does not exist: {config.input_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not config.preflight_only and not shutil.which(config.pi_bin):
        print(f"Error: pi_bin not found: {config.pi_bin}", file=sys.stderr)
        raise SystemExit(1)
    if config.epub_check and not shutil.which(config.epub_check_path):
        print(f"Error: epubcheck not found: {config.epub_check_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        image_count = sum(
            1
            for path in config.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    except OSError:
        image_count = 0

    print(
        f"btran — translating {image_count} images from {config.source_lang}"
        f" → {config.target_lang} using {config.model}"
    )

    def on_page_error(page_number: int, message: str) -> None:
        print(f"[btran] page {page_number} failed: {message}", file=sys.stderr)

    runner: OrchestratorCallable = orchestrator_run
    try:
        result: RunResult = asyncio.run(runner(config, on_page_error=on_page_error))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(1)

    if result.errors:
        print(
            f"[btran] {len(result.errors)} page(s) failed — no EPUB produced.",
            file=sys.stderr,
        )
        raise SystemExit(1)
