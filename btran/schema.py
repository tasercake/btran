"""Intermediate JSON schema types for btran."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PageResult:
    """A successfully translated page."""

    page_number: int
    sha256: str
    phash: str
    image_path: str = ""
    source_lang: str = ""
    target_lang: str = ""
    page_text: str = ""
    translated_text: str = ""
    image_descriptions: list[str] = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    retry_count: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PageResult:
        expected: dict[str, object] = {
            "image_descriptions": [],
            "model": "",
            "timestamp": "",
            "retry_count": 0,
        }
        for k, default in expected.items():
            d.setdefault(k, default)
        return cls(**d)

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def from_file(cls, path: Path) -> PageResult:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class ErrorResult:
    """A page that could not be translated after all retries."""

    page_number: int
    image_path: str = ""
    error: str = ""
    retry_count: int = 0
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ErrorResult:
        expected: dict[str, object] = {
            "image_path": "",
            "retry_count": 0,
            "model": "",
        }
        for k, default in expected.items():
            d.setdefault(k, default)
        return cls(**d)

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def from_file(cls, path: Path) -> ErrorResult:
        return cls.from_dict(json.loads(path.read_text()))
