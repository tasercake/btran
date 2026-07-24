"""Intermediate JSON schema types for btran."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@dataclass
class SourceBlock:
    """A typed structural block extracted from a page image."""

    id: str
    type: str
    text: str
    reading_order: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SourceBlock:
        return cls(**d)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> SourceBlock:
        return cls.from_dict(_read_json(path))


@dataclass
class TermMention:
    """A term mention found during source extraction."""

    term: str
    block_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TermMention:
        return cls(**d)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> TermMention:
        return cls.from_dict(_read_json(path))


@dataclass
class PageExtraction:
    """Full extraction result for a single page (source-only, no translation)."""

    page_number: int
    image_path: str
    sha256: str
    phash: str
    source_lang: str
    model: str
    timestamp: str = ""
    blocks: list[SourceBlock] = field(default_factory=list)
    term_mentions: list[TermMention] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PageExtraction:
        values = d.copy()
        values["blocks"] = [SourceBlock.from_dict(block) for block in values.get("blocks", [])]
        values["term_mentions"] = [
            TermMention.from_dict(mention) for mention in values.get("term_mentions", [])
        ]
        values.setdefault("illustrations", [])
        values.setdefault("timestamp", "")
        return cls(**values)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> PageExtraction:
        return cls.from_dict(_read_json(path))


@dataclass
class TranslatedBlock:
    """A translated block with exact correspondence to a SourceBlock."""

    block_id: str
    translated_text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TranslatedBlock:
        return cls(**d)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> TranslatedBlock:
        return cls.from_dict(_read_json(path))


@dataclass
class TerminologyEntry:
    """One terminology entry after consolidation."""

    concept_id: str
    source_terms: list[str]
    target_term: str
    provenance: list[str]
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TerminologyEntry:
        return cls(**d)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> TerminologyEntry:
        return cls.from_dict(_read_json(path))


@dataclass
class TerminologyMap:
    """Frozen, versioned glossary."""

    version: str
    hash: str
    source_lang: str
    target_lang: str
    entries: list[TerminologyEntry]
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TerminologyMap:
        values = d.copy()
        values["entries"] = [TerminologyEntry.from_dict(entry) for entry in values.get("entries", [])]
        values.setdefault("created_at", "")
        return cls(**values)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> TerminologyMap:
        return cls.from_dict(_read_json(path))


@dataclass
class Manifest:
    """Input manifest describing the set of pages to translate."""

    input_dir: str
    pages: list[dict]
    total_pages: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        return cls(**d)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> Manifest:
        return cls.from_dict(_read_json(path))


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
    blocks: list[SourceBlock] = field(default_factory=list)
    translated_blocks: list[TranslatedBlock] = field(default_factory=list)
    term_mentions: list[TermMention] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)

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
            "blocks": [],
            "translated_blocks": [],
            "term_mentions": [],
            "illustrations": [],
        }
        values = d.copy()
        for k, default in expected.items():
            values.setdefault(k, default)
        values["blocks"] = [SourceBlock.from_dict(block) for block in values["blocks"]]
        values["translated_blocks"] = [
            TranslatedBlock.from_dict(block) for block in values["translated_blocks"]
        ]
        values["term_mentions"] = [
            TermMention.from_dict(mention) for mention in values["term_mentions"]
        ]
        return cls(**values)

    def to_file(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> PageResult:
        return cls.from_dict(_read_json(path))


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
        _write_json(path, self.to_dict())

    @classmethod
    def from_file(cls, path: Path) -> ErrorResult:
        return cls.from_dict(_read_json(path))
