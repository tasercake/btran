"""Image hashing (SHA256 + perceptual phash) and SQLite translation cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import imagehash
from PIL import Image

from btran.schema import PageResult


def compute_sha256(image_path: Path) -> str:
    """SHA256 hex digest of file bytes."""
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def compute_phash(image_path: Path) -> str:
    """Perceptual hash (phash) of image as hex string. Uses imagehash library."""
    img = Image.open(image_path)
    return str(imagehash.phash(img))


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
    result_json TEXT NOT NULL
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_phash ON translations(phash);"
)


class ImageCache:
    """SQLite-backed cache for translation results, keyed by image hash."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()

    def lookup(self, sha256: str) -> PageResult | None:
        """Exact SHA256 match. Returns cached PageResult or None."""
        row = self._conn.execute(
            "SELECT result_json FROM translations WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row is None:
            return None
        return PageResult.from_dict(json.loads(row[0]))

    def lookup_perceptual(
        self, phash: str, threshold: int = 5
    ) -> PageResult | None:
        """Find near-match by phash. Returns PageResult if Hamming
        distance ≤ threshold."""
        rows = self._conn.execute(
            "SELECT phash, result_json FROM translations"
        ).fetchall()
        for row_phash, result_json in rows:
            if hamming_distance(phash, row_phash) <= threshold:
                return PageResult.from_dict(json.loads(result_json))
        return None

    def store(
        self,
        sha256: str,
        phash: str,
        image_path: str,
        result: PageResult,
    ) -> None:
        """Store a new translation result."""
        result_json = json.dumps(result.to_dict())
        self._conn.execute(
            "INSERT OR REPLACE INTO translations "
            "(sha256, phash, image_path, page_number, result_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (sha256, phash, image_path, result.page_number, result_json),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close DB connection."""
        self._conn.close()
