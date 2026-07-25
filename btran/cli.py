"""CLI entry point. Parses args, loads config, and invokes Gate 1 orchestration."""

from __future__ import annotations

import asyncio
import shutil
import sys

from btran.config import load_config
from btran.orchestrator import orchestrator_run
from btran.orchestrator_contract import OrchestratorCallable, RunResult


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
    if not config.input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {config.input_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not shutil.which(config.pi_bin):
        print(f"Error: pi_bin not found: {config.pi_bin}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"btran — auto-detecting source languages"
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
        page_errors = [
            error for error in result.errors
            if error.startswith("[btran] page ") or error.startswith("page ")
        ]
        if len(page_errors) == len(result.errors):
            summary = f"[btran] {len(page_errors)} page(s) failed — no EPUB produced."
        elif page_errors:
            summary = f"[btran] run failed ({len(page_errors)} page(s) failed) — no EPUB produced."
        else:
            summary = "[btran] run failed — no EPUB produced."
        print(summary, file=sys.stderr)
        raise SystemExit(1)
