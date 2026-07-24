"""CLI entry point. Parses args, loads config, runs pipeline."""

import asyncio
import shutil
import sys

from btran.config import load_config
from btran.orchestrator import run as orchestrator_run

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def main() -> None:
    """CLI entry point. Parse args, load config, run pipeline."""
    try:
        config = load_config()
    except SystemExit:
        raise
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate input_dir exists
    if not config.input_dir.exists():
        print(f"Error: input_dir does not exist: {config.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Validate pi_bin is findable
    if not shutil.which(config.pi_bin):
        print(f"Error: pi_bin not found: {config.pi_bin}", file=sys.stderr)
        sys.exit(1)

    # Count images for banner
    try:
        n_images = sum(
            1
            for p in config.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    except Exception:
        n_images = 0

    # Startup banner
    print(
        f"btran — translating {n_images} images from {config.source_lang}"
        f" → {config.target_lang} using {config.model}"
    )

    # Run pipeline
    try:
        asyncio.run(orchestrator_run(config))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
