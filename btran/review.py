"""Flat, auditable review artifacts for orchestration gates.

Artifacts are deliberately ordinary JSON files: operators can inspect and resolve
one issue without a database, service, or agent loop.
"""
from __future__ import annotations

import json
import os
import tempfile
import shutil
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


def _read_item(path: Path) -> ReviewItem:
    value = json.loads(path.read_text())
    expected = {"item_id", "kind", "blocking", "evidence", "image_path", "page_number", "status", "resolution"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid review artifact shape")
    item = ReviewItem.from_dict(value)
    if (
        item.item_id != path.stem or not isinstance(item.kind, str) or not item.kind
        or not isinstance(item.blocking, bool) or not isinstance(item.evidence, dict)
        or not isinstance(item.image_path, str)
        or (item.page_number is not None and (isinstance(item.page_number, bool) or not isinstance(item.page_number, int)))
        or item.status not in {"pending", "resolved"}
        or (item.resolution is not None and not isinstance(item.resolution, dict))
    ):
        raise ValueError("invalid review artifact values")
    if item.status == "resolved":
        if not item.resolution or item.resolution.get("action") not in {"accept", "correct"}:
            raise ValueError("invalid resolved review artifact")
        if item.resolution["action"] == "accept" and set(item.resolution) != {"action"}:
            raise ValueError("invalid accept review artifact")
        if item.resolution["action"] == "correct" and (
            set(item.resolution) != {"action", "correction"}
            or not isinstance(item.resolution.get("correction"), str)
            or not item.resolution["correction"].strip()
        ):
            raise ValueError("invalid correction review artifact")
    elif item.resolution is not None and item.resolution != {"action": "retry"}:
        raise ValueError("invalid pending review artifact")
    return item


def _archive_stale(directory: Path, active_ids: set[str]) -> None:
    archive = directory / "archive"
    for path in sorted(directory.glob("*.json")):
        if path.stem in active_ids:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / path.name
        suffix = 1
        while destination.exists():
            destination = archive / f"{path.stem}.{suffix}.json"
            suffix += 1
        shutil.move(str(path), str(destination))


def write_items(directory: Path, items: list[ReviewItem], *, archive_stale: bool = False) -> list[Path]:
    """Persist the explicit current review set, preserving stable resolutions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    active_ids = {item.item_id for item in items}
    if len(active_ids) != len(items):
        raise ValueError("review item IDs must be unique")
    if archive_stale:
        _archive_stale(directory, active_ids)
    paths: list[Path] = []
    for item in items:
        path = directory / f"{item.item_id}.json"
        value = item.to_dict()
        if path.exists():
            try:
                existing = _read_item(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A malformed current artifact is a blocking operator problem.
                paths.append(path)
                continue
            if existing.status == "resolved" and existing.resolution:
                value["status"] = existing.status
                value["resolution"] = existing.resolution
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
    item = _read_item(artifact)
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
            item = _read_item(path)
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
            item = _read_item(path)
            resolution = item.resolution or {}
            concept_id = str(item.evidence.get("concept_id", ""))
            if item.status == "resolved" and resolution.get("action") == "correct" and concept_id:
                result[concept_id] = resolution["correction"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return result
