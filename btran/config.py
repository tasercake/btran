"""Configuration for btran's small production CLI surface."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from dotenv import find_dotenv, load_dotenv


GLOSSARY_BUDGET_DEFAULT = 100_000
GLOSSARY_BUDGET_MAXIMUM = 120_000
_ENV_PREFIX = "BTRAN_"
_UNSUPPORTED_ENV_CONTROLS = {
    "CACHE_DB": "the merged orchestrator uses its work-owned translation cache",
    "NO_PREFLIGHT": "preflight is always enabled",
    "EVAL_DIR": "the evaluation corpus is a developer harness, not a production control",
    "GLOSSARY_PATH": "glossary output paths are managed by the pipeline",
    "RECONCILIATION_ROUNDS": "reconciliation always uses one round",
    "PREFLIGHT_ONLY": "the merged orchestrator always runs preflight as part of a full run",
    "REVIEW": "the merged orchestrator automatically blocks on unresolved review items",
}


@dataclass
class Config:
    """Runtime settings passed unchanged to the orchestrator boundary.

    A relative ``manifest_path`` is resolved by pipeline integration beneath
    ``input_dir``, so the default is always ``INPUT_DIR/manifest.json``.
    """

    model: str = "gemini-2.5-flash"
    target_lang: str = ""
    concurrency: int = 4
    max_retries: int = 3
    timeout: int = 0
    intermediate_dir: Path = Path("./intermediate")
    pi_bin: str = "pi"
    title: str = "Translated Book"
    author: str = "Unknown"
    input_dir: Path = Path(".")
    output_epub: Path = Path("output.epub")
    embed_images: bool = False
    no_resume: bool = False
    epub_check: bool = False
    epub_check_path: str = "epubcheck"
    manifest_path: Path = Path("manifest.json")
    # Internal pipeline artifact location; deliberately not a CLI/env control.
    glossary_path: Path = Path("glossary.json")
    glossary_budget: int = GLOSSARY_BUDGET_DEFAULT


def _flag_env(value: str) -> bool:
    """Parse an explicitly boolean environment value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


_ENV_FIELDS: dict[str, tuple[str, Callable[[str], object]]] = {
    "model": ("MODEL", str),
    "target_lang": ("TARGET_LANG", str),
    "concurrency": ("CONCURRENCY", int),
    "max_retries": ("MAX_RETRIES", int),
    "timeout": ("TIMEOUT", int),
    "intermediate_dir": ("INTERMEDIATE_DIR", Path),
    "pi_bin": ("PI_BIN", str),
    "title": ("TITLE", str),
    "author": ("AUTHOR", str),
    "input_dir": ("INPUT_DIR", Path),
    "output_epub": ("OUTPUT_EPUB", Path),
    "embed_images": ("EMBED_IMAGES", _flag_env),
    "no_resume": ("NO_RESUME", _flag_env),
    "epub_check": ("EPUB_CHECK", _flag_env),
    "epub_check_path": ("EPUB_CHECK_PATH", str),
    "manifest_path": ("MANIFEST_PATH", Path),
    "glossary_budget": ("GLOSSARY_BUDGET", int),
}
_PATH_FIELDS = {
    "input_dir",
    "output_epub",
    "intermediate_dir",
    "manifest_path",
}


def load_config(argv: list[str] | None = None) -> Config:
    """Load ``.env`` values then override them with explicit CLI arguments."""
    load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    parser = _build_parser()
    cli_ns = parser.parse_args(argv)
    _reject_unsupported_environment_controls()

    defaults = Config()
    values: dict[str, object] = {}
    for name, (env_suffix, converter) in _ENV_FIELDS.items():
        raw = os.getenv(_ENV_PREFIX + env_suffix)
        if raw is None:
            values[name] = getattr(defaults, name)
            continue
        try:
            values[name] = converter(raw)
        except ValueError:
            parser.error(f"BTRAN_{env_suffix} has an invalid value: {raw!r}")

    cli_values = vars(cli_ns).copy()
    input_dir = cli_values.pop("INPUT_DIR")
    output_epub = cli_values.pop("OUTPUT_EPUB")
    for name, value in cli_values.items():
        if value is not None:
            values[name] = Path(value) if name in _PATH_FIELDS else value

    values["input_dir"] = Path(input_dir)
    values["output_epub"] = Path(output_epub)
    config = Config(
        **{
            field.name: values.get(field.name, getattr(defaults, field.name))
            for field in fields(Config)
        }
    )
    _validate_config(config, parser, cli_ns)
    return config


def _reject_unsupported_environment_controls() -> None:
    for suffix, reason in _UNSUPPORTED_ENV_CONTROLS.items():
        if os.getenv(_ENV_PREFIX + suffix) is not None:
            raise ValueError(f"BTRAN_{suffix} is not supported: {reason}.")


def _validate_config(
    config: Config, parser: argparse.ArgumentParser, cli_ns: argparse.Namespace
) -> None:
    if not config.target_lang:
        raise ValueError(
            "target_lang is required. Set BTRAN_TARGET_LANG in .env or "
            "pass --target-lang on the command line."
        )
    minimums = {
        "concurrency": 1,
        "max_retries": 0,
        "timeout": 0,
        "glossary_budget": 1,
    }
    for field_name, minimum in minimums.items():
        if getattr(config, field_name) < minimum:
            parser.error(f"{field_name} must be at least {minimum}")
    if config.glossary_budget > GLOSSARY_BUDGET_MAXIMUM:
        parser.error(
            f"glossary_budget must not exceed {GLOSSARY_BUDGET_MAXIMUM}"
        )
    if config.epub_check and not config.epub_check_path.strip():
        parser.error("epub_check_path must not be empty when --epub-check is enabled")

    manifest_path_value = (
        cli_ns.manifest_path
        if cli_ns.manifest_path is not None
        else os.getenv("BTRAN_MANIFEST_PATH")
    )
    if manifest_path_value is not None and not manifest_path_value.strip():
        parser.error("manifest_path must not be empty")
    if not cli_ns.INPUT_DIR.strip():
        parser.error("INPUT_DIR must not be empty")
    if not cli_ns.OUTPUT_EPUB.strip():
        parser.error("OUTPUT_EPUB must not be empty")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="btran — translate book photos to EPUB")
    parser.add_argument("INPUT_DIR", help="Directory containing book page images")
    parser.add_argument("OUTPUT_EPUB", help="Output EPUB file path")
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--intermediate-dir", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--embed-images", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_true", default=None)
    parser.add_argument("--pi-bin", default=None)
    parser.add_argument("--epub-check", action="store_true", default=None)
    parser.add_argument("--epub-check-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--glossary-budget", type=int, default=None)
    return parser
