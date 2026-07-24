"""Flat, auditable review artifacts for orchestration gates.

Artifacts are deliberately ordinary JSON files: operators can inspect and resolve
one issue without a database, service, or agent loop.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_VALID_ACTIONS = frozenset({"accept", "correct", "retry"})


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    kind: str
    blocking: bool
    evidence: dict[str, Any]
    image_path: str = ""
    page_number: int | None = None
    status: str = "pending"
    resolution: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewItem":
        return cls(**value)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            name = file.name
            json.dump(value, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    except Exception:
        if name:
            Path(name).unlink(missing_ok=True)
        raise


def write_items(directory: Path, items: list[ReviewItem]) -> list[Path]:
    """Persist one stable, flat JSON object for each review item.

    Existing resolved artifacts are retained if the same issue recurs, allowing a
    user decision to survive a resume run.  A retry resolution deliberately
    becomes pending again, since it requests fresh model work rather than approval.
    """
    paths: list[Path] = []
    for item in items:
        path = Path(directory) / f"{item.item_id}.json"
        value = item.to_dict()
        if path.exists():
            try:
                existing = ReviewItem.from_dict(json.loads(path.read_text()))
                if existing.status == "resolved" and existing.resolution and existing.resolution.get("action") != "retry":
                    value["status"] = existing.status
                    value["resolution"] = existing.resolution
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        _atomic_json(path, value)
        paths.append(path)
    return paths


def resolve_item(path: Path, action: str, correction: str | None = None) -> ReviewItem:
    """Resolve an item as accept/correct/retry; pages are never discarded."""
    if action not in _VALID_ACTIONS:
        raise ValueError(f"unknown review resolution: {action}")
    if action == "correct" and (not isinstance(correction, str) or not correction.strip()):
        raise ValueError("correct resolution requires non-empty correction")
    artifact = Path(path)
    item = ReviewItem.from_dict(json.loads(artifact.read_text()))
    resolution = {"action": action}
    if correction is not None:
        resolution["correction"] = correction
    status = "pending" if action == "retry" else "resolved"
    resolved = ReviewItem(**{**item.to_dict(), "status": status, "resolution": resolution})
    _atomic_json(artifact, resolved.to_dict())
    return resolved


def unresolved_items(directory: Path) -> list[ReviewItem]:
    """Return blocking pending review items in deterministic file order."""
    items: list[ReviewItem] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            item = ReviewItem.from_dict(json.loads(path.read_text()))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed operator artifact is itself unsafe; leave it pending.
            items.append(ReviewItem(path.stem, "malformed_review_artifact", True, {"path": str(path)}))
            continue
        if item.blocking and item.status != "resolved":
            items.append(item)
    return items


def corrections(directory: Path) -> dict[str, str]:
    """Return accepted glossary corrections keyed by concept ID."""
    result: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            item = ReviewItem.from_dict(json.loads(path.read_text()))
            resolution = item.resolution or {}
            concept_id = str(item.evidence.get("concept_id", ""))
            if item.status == "resolved" and resolution.get("action") == "correct" and concept_id:
                result[concept_id] = resolution["correction"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return result
