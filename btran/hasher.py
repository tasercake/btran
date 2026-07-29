"""Image hashing (SHA256 + perceptual phash) and SQLite translation cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import imagehash
from PIL import Image

from btran.schema import PageResult, canonical_json


def compute_sha256(image_path: Path) -> str:
    """SHA256 hex digest of file bytes."""
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def compute_phash(image_path: Path) -> str:
    """Perceptual hash (phash) of image as hex string. Uses imagehash library."""
    img = Image.open(image_path)
    return str(imagehash.phash(img))


def compute_prompt_fingerprint(prompt: str) -> str:
    """Deterministic fingerprint of a prompt string (SHA256, first 12 hex chars).

    This identifies a specific prompt/schema version so the cache can
    distinguish translations produced by different prompts.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    if len(h1) != len(h2):
        raise ValueError(
            f"Hash strings must be same length, got {len(h1)} and {len(h2)}"
        )
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS translations (
    sha256 TEXT PRIMARY KEY,
    phash TEXT NOT NULL,
    image_path TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    source_lang TEXT NOT NULL DEFAULT '',
    target_lang TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT ''
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_phash ON translations(phash);"
)

# Columns added after the initial release for semantic cache scoping.
_SEMANTIC_COLUMNS = ["source_lang", "target_lang", "model", "prompt_version"]


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    """Add semantic columns to an existing table if they are missing.

    Legacy databases created before semantic scoping will have rows with
    empty-string defaults for these columns, which naturally causes them
    to miss against any real semantic-context lookup (fail closed).
    """
    cursor = conn.execute("PRAGMA table_info(translations)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    for col in _SEMANTIC_COLUMNS:
        if col not in existing_columns:
            conn.execute(
                f"ALTER TABLE translations ADD COLUMN {col} "
                "TEXT NOT NULL DEFAULT ''"
            )
    conn.commit()


class ImageCache:
    """SQLite-backed cache for translation results.

    Cache keying includes both image identity (SHA256 + phash) and
    semantic context (source_lang, target_lang, model, prompt_version).
    Legacy rows without semantic columns fail closed (cache miss).
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_TABLE_SQL)
        # Migrate older DBs that lack semantic columns.
        _migrate_add_columns(self._conn)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()

    # --- lookup -----------------------------------------------------------------

    def lookup(
        self,
        sha256: str,
        *,
        source_lang: str,
        target_lang: str,
        model: str,
        prompt_version: str,
    ) -> PageResult | None:
        """Exact SHA256 match scoped by semantic context.

        All keyword-only semantic params must match the stored row.
        """
        row = self._conn.execute(
            "SELECT result_json FROM translations"
            " WHERE sha256 = ?"
            " AND source_lang = ?"
            " AND target_lang = ?"
            " AND model = ?"
            " AND prompt_version = ?",
            (sha256, source_lang, target_lang, model, prompt_version),
        ).fetchone()
        if row is None:
            return None
        return PageResult.from_dict(json.loads(row[0]))

    # --- lookup_perceptual ------------------------------------------------------

    def lookup_perceptual(
        self,
        phash: str,
        threshold: int = 5,
        *,
        source_lang: str,
        target_lang: str,
        model: str,
        prompt_version: str,
    ) -> PageResult | None:
        """Find near-match by phash, scoped by semantic context.

        Returns PageResult if Hamming distance ≤ threshold **and** all
        semantic fields match the stored row.
        """
        rows = self._conn.execute(
            "SELECT phash, result_json, source_lang, target_lang,"
            "       model, prompt_version"
            " FROM translations"
        ).fetchall()
        for row_phash, result_json, sl, tl, md, pv in rows:
            if (sl, tl, md, pv) != (source_lang, target_lang, model, prompt_version):
                continue
            if hamming_distance(phash, row_phash) <= threshold:
                return PageResult.from_dict(json.loads(result_json))
        return None

    # --- store ------------------------------------------------------------------

    def store(
        self,
        sha256: str,
        phash: str,
        image_path: str,
        result: PageResult,
        *,
        source_lang: str,
        target_lang: str,
        model: str,
        prompt_version: str,
    ) -> None:
        """Store a new translation result with full semantic context."""
        result_json = canonical_json(asdict(result))
        self._conn.execute(
            "INSERT OR REPLACE INTO translations"
            " (sha256, phash, image_path, page_number, result_json,"
            "  source_lang, target_lang, model, prompt_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                phash,
                image_path,
                result.page_number,
                result_json,
                source_lang,
                target_lang,
                model,
                prompt_version,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close DB connection."""
        self._conn.close()
