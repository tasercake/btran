"""Configuration and bounded utility-process policy for btran runs."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from dotenv import find_dotenv, load_dotenv


GLOSSARY_BUDGET_DEFAULT = 100_000
GLOSSARY_BUDGET_MAXIMUM = 120_000
TIMEOUT_SECONDS_MINIMUM = 1
TIMEOUT_SECONDS_MAXIMUM = 3600
MAX_RETRIES_MINIMUM = 0
MAX_RETRIES_MAXIMUM = 5
PROCESS_TERMINATE_GRACE_SECONDS = 2
PROCESS_KILL_GRACE_SECONDS = 2
_ENV_PREFIX = "BTRAN_"
_UNSUPPORTED_ENV_CONTROLS = {
    "CACHE_DB": "the merged orchestrator uses its work-owned translation cache",
    "EVAL_DIR": "the evaluation corpus is a developer harness, not a production control",
    "GLOSSARY_PATH": "glossary output paths are managed by the pipeline",
    "RECONCILIATION_ROUNDS": "reconciliation always uses one round",
    "REVIEW": "review findings are informational and never block a run",
}


@dataclass
class Config:
    """Run settings.  ``target_lang=None`` selects native mode."""

    model: str = "gemini-2.5-flash"
    target_lang: str | None = None
    concurrency: int = 4
    max_retries: int = 3
    # Deadline for terminology consolidation and EPUBCheck only. Model calls have none.
    timeout: int = 120
    # ``workspace`` is new authority.  ``intermediate_dir`` remains a narrow
    # migration alias for callers of the old CLI surface.
    workspace: Path | None = None
    intermediate_dir: Path | None = None
    base_revision: str | None = None
    correction_set: str | None = None
    refresh: bool = False
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

    @property
    def mode(self) -> str:
        return "translated" if self.target_lang is not None else "native"

    @property
    def retry_backoffs(self) -> tuple[int, ...]:
        return tuple(min(2 ** attempt, 16) for attempt in range(self.max_retries))


@dataclass(frozen=True)
class WorkspaceResolution:
    workspace: Path
    fallback_from: Path | None = None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_from is not None


class WorkspaceResolutionError(ValueError):
    """Neither requested nor output-adjacent workspace can be made writable."""


def validate_timeout_seconds(value: object) -> int:
    """Enforce deadline used by bounded non-model utility subprocesses."""
    if (isinstance(value, bool) or not isinstance(value, int)
            or not TIMEOUT_SECONDS_MINIMUM <= value <= TIMEOUT_SECONDS_MAXIMUM):
        raise ValueError(
            f"timeout must be an integer between {TIMEOUT_SECONDS_MINIMUM} and {TIMEOUT_SECONDS_MAXIMUM}"
        )
    return value


def validate_max_retries(value: object) -> int:
    """Enforce Config's bounded integer retry contract."""
    if (isinstance(value, bool) or not isinstance(value, int)
            or not MAX_RETRIES_MINIMUM <= value <= MAX_RETRIES_MAXIMUM):
        raise ValueError(
            f"max_retries must be an integer between {MAX_RETRIES_MINIMUM} and {MAX_RETRIES_MAXIMUM}"
        )
    return value


def _flag_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


_ENV_FIELDS: dict[str, tuple[str, Callable[[str], object]]] = {
    "model": ("MODEL", str),
    "concurrency": ("CONCURRENCY", int),
    "max_retries": ("MAX_RETRIES", int),
    "timeout": ("TIMEOUT", int),
    "workspace": ("WORKSPACE", Path),
    "intermediate_dir": ("INTERMEDIATE_DIR", Path),
    "base_revision": ("BASE_REVISION", str),
    "correction_set": ("CORRECTION_SET", str),
    "refresh": ("REFRESH", _flag_env),
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
_PATH_FIELDS = {"input_dir", "output_epub", "workspace", "intermediate_dir", "manifest_path"}


def default_workspace(output_epub: Path | str) -> Path:
    """Return output-adjacent state root; this is always fallback authority."""
    return Path(output_epub).parent / ".btran"


def _ensure_writable_directory(path: Path) -> Path:
    """Create and prove a directory is writable without retaining test files."""
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise OSError(f"workspace is not a directory: {path}")
    with tempfile.NamedTemporaryFile(dir=path, prefix=".btran-write-check-", delete=True):
        pass
    return path


def resolve_workspace(config: Config) -> WorkspaceResolution:
    """Prefer explicit workspace/intermediate path; safely fall back beside EPUB."""
    explicit = config.workspace if config.workspace is not None else config.intermediate_dir
    fallback = default_workspace(config.output_epub)
    if explicit is None:
        try:
            return WorkspaceResolution(_ensure_writable_directory(fallback))
        except OSError as exc:
            raise WorkspaceResolutionError("output-adjacent workspace is not writable") from exc
    try:
        return WorkspaceResolution(_ensure_writable_directory(Path(explicit)))
    except OSError:
        try:
            return WorkspaceResolution(_ensure_writable_directory(fallback), Path(explicit))
        except OSError as exc:
            raise WorkspaceResolutionError("requested and output-adjacent workspaces are not writable") from exc


def load_config(argv: list[str] | None = None) -> Config:
    """Load dotenv/environment then apply explicit CLI values."""
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
        if name in {"workspace", "intermediate_dir", "base_revision", "correction_set", "manifest_path"} and not raw.strip():
            parser.error(f"BTRAN_{env_suffix} must not be blank")
        try:
            values[name] = converter(raw)
        except ValueError:
            parser.error(f"BTRAN_{env_suffix} has an invalid value: {raw!r}")

    # Target selection has deliberately narrower semantics than normal config:
    # only CLI or environment may select it, and either present blank selector
    # is an error rather than a native-mode spelling.
    env_target = os.getenv("BTRAN_TARGET_LANG")
    cli_target = cli_ns.target_lang
    if env_target is not None and not env_target.strip():
        parser.error("BTRAN_TARGET_LANG must not be blank")
    if cli_target is not None and not cli_target.strip():
        parser.error("target_lang must not be blank")
    values["target_lang"] = cli_target.strip() if cli_target is not None else (
        env_target.strip() if env_target is not None else None
    )

    cli_values = vars(cli_ns).copy()
    for name in ("workspace", "intermediate_dir", "base_revision", "correction_set", "manifest_path"):
        raw = cli_values.get(name)
        if raw is not None and not raw.strip():
            parser.error(f"{name} must not be empty")
    input_dir = cli_values.pop("INPUT_DIR")
    output_epub = cli_values.pop("OUTPUT_EPUB")
    cli_values.pop("target_lang")
    for name, value in cli_values.items():
        if value is not None:
            values[name] = Path(value) if name in _PATH_FIELDS else value

    values["input_dir"] = Path(input_dir)
    values["output_epub"] = Path(output_epub)
    config = Config(**{
        field.name: values.get(field.name, getattr(defaults, field.name))
        for field in fields(Config)
    })
    _validate_config(config, parser, cli_ns)
    return config


def _reject_unsupported_environment_controls() -> None:
    for suffix, reason in _UNSUPPORTED_ENV_CONTROLS.items():
        if os.getenv(_ENV_PREFIX + suffix) is not None:
            raise ValueError(f"BTRAN_{suffix} is not supported: {reason}.")


def _blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _validate_config(config: Config, parser: argparse.ArgumentParser, cli_ns: argparse.Namespace) -> None:
    if not 1 <= config.concurrency <= 32:
        parser.error("concurrency must be between 1 and 32")
    try:
        validate_max_retries(config.max_retries)
        validate_timeout_seconds(config.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= config.glossary_budget <= GLOSSARY_BUDGET_MAXIMUM:
        parser.error(f"glossary_budget must be between 1 and {GLOSSARY_BUDGET_MAXIMUM}")
    if config.epub_check and not config.epub_check_path.strip():
        parser.error("epub_check_path must not be empty when --epub-check is enabled")

    for name in ("workspace", "intermediate_dir", "base_revision", "correction_set"):
        value = getattr(config, name)
        if _blank(value):
            parser.error(f"{name} must not be empty")
    manifest_path_value = cli_ns.manifest_path if cli_ns.manifest_path is not None else os.getenv("BTRAN_MANIFEST_PATH")
    if manifest_path_value is not None and not manifest_path_value.strip():
        parser.error("manifest_path must not be empty")
    if not cli_ns.INPUT_DIR.strip():
        parser.error("INPUT_DIR must not be empty")
    if not cli_ns.OUTPUT_EPUB.strip():
        parser.error("OUTPUT_EPUB must not be empty")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="btran — book photos to EPUB")
    parser.add_argument("INPUT_DIR", help="Directory containing book page images")
    parser.add_argument("OUTPUT_EPUB", help="Output EPUB file path")
    parser.add_argument("--target-lang", default=None, metavar="LANG")
    parser.add_argument("--workspace", default=None, metavar="DIR")
    parser.add_argument("--base-revision", default=None, metavar="REVISION_ID")
    parser.add_argument("--correction-set", default=None, metavar="SET_ID")
    parser.add_argument("--refresh", action="store_true", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--timeout", type=int, default=None, metavar="SECONDS",
        help="Deadline for terminology consolidation and EPUBCheck; Pi extraction and translation have no deadline.",
    )
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
