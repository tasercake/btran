"""Configuration for btran — loads from .env and CLI args."""

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


@dataclass
class Config:
    model: str = "gemini-2.5-flash"
    source_lang: str = "en"
    target_lang: str = ""  # REQUIRED — must be set or error
    concurrency: int = 4
    max_retries: int = 3
    timeout: int = 120
    intermediate_dir: Path = Path("./intermediate")
    cache_db: Path = Path("./cache.sqlite")
    pi_bin: str = "pi"
    title: str = "Translated Book"
    author: str = "Unknown"
    input_dir: Path = Path(".")
    output_epub: Path = Path("output.epub")
    embed_images: bool = False
    no_resume: bool = False


_ENV_PREFIX = "BTRAN_"

# (field_name, env_suffix, converter)
_FIELD_MAP: dict[str, tuple[str, type]] = {}


def _flag_env(val: str) -> bool:
    """Parse a boolean env var — truthy strings map to True."""
    return val.strip().lower() in ("1", "true", "yes")


def _init_field_map() -> None:
    """Populate _FIELD_MAP; called once on import."""
    _FIELD_MAP.update({
        "model": ("MODEL", str),
        "source_lang": ("SOURCE_LANG", str),
        "target_lang": ("TARGET_LANG", str),
        "concurrency": ("CONCURRENCY", int),
        "max_retries": ("MAX_RETRIES", int),
        "timeout": ("TIMEOUT", int),
        "intermediate_dir": ("INTERMEDIATE_DIR", Path),
        "cache_db": ("CACHE_DB", Path),
        "pi_bin": ("PI_BIN", str),
        "title": ("TITLE", str),
        "author": ("AUTHOR", str),
        "input_dir": ("INPUT_DIR", Path),
        "output_epub": ("OUTPUT_EPUB", Path),
        "embed_images": ("EMBED_IMAGES", _flag_env),
        "no_resume": ("NO_RESUME", _flag_env),
    })


def load_config(argv: list[str] | None = None) -> Config:
    """Load configuration from .env file then CLI args.

    Env vars follow the BTRAN_ prefix (e.g. BTRAN_MODEL, BTRAN_TARGET_LANG).
    CLI args override env vars.  target_lang must be provided somewhere.
    argv=None uses sys.argv[1:].
    """
    load_dotenv(dotenv_path=find_dotenv(usecwd=True))

    # Seed kwargs from env vars (absent env → dataclass default).
    default_cfg = Config()
    env_kwargs: dict[str, object] = {}
    for field in fields(Config):
        env_suffix, converter = _FIELD_MAP[field.name]
        raw = os.getenv(_ENV_PREFIX + env_suffix)
        if raw is not None:
            env_kwargs[field.name] = converter(raw)
        else:
            env_kwargs[field.name] = getattr(default_cfg, field.name)

    # Parse CLI args on top.
    parser = _build_parser()
    cli_ns = parser.parse_args(argv)
    cli_kwargs = vars(cli_ns).copy()

    # Pull out positional args.
    input_dir = cli_kwargs.pop("INPUT_DIR")
    output_epub = cli_kwargs.pop("OUTPUT_EPUB")

    # Merge: env first, then CLI (only non-None values override).
    path_fields = {"input_dir", "output_epub", "intermediate_dir", "cache_db"}
    for k, v in cli_kwargs.items():
        if v is not None:
            env_kwargs[k] = Path(v) if k in path_fields else v

    env_kwargs["input_dir"] = Path(input_dir)
    env_kwargs["output_epub"] = Path(output_epub)

    # Filter to only valid Config fields, then construct.
    valid_keys = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in env_kwargs.items() if k in valid_keys})

    if not cfg.target_lang:
        raise ValueError(
            "target_lang is required. Set BTRAN_TARGET_LANG in .env or "
            "pass --target-lang on the command line."
        )

    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="btran — translate book photos to EPUB")
    p.add_argument("INPUT_DIR", help="Directory containing book page images")
    p.add_argument("OUTPUT_EPUB", help="Output EPUB file path")
    p.add_argument("--source-lang", default=None)
    p.add_argument("--target-lang", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--intermediate-dir", default=None)
    p.add_argument("--cache-db", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--author", default=None)
    p.add_argument("--embed-images", action="store_true", default=None)
    p.add_argument("--no-resume", action="store_true", default=None)
    p.add_argument("--pi-bin", default=None)
    return p


# Fill the field map at import time.
_init_field_map()
