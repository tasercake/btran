from __future__ import annotations

import json
from pathlib import Path

from btran.review import ReviewItem, corrections, unresolved_items, write_items, resolve_item


def test_flat_review_artifact_keeps_evidence_image_reference_and_resolution(tmp_path: Path):
    directory = tmp_path / "needs_review"
    item = ReviewItem(
        item_id="low-confidence-island", kind="low_confidence", blocking=True,
        evidence={"concept_id": "island", "confidence": 0.2}, image_path="/book/a.png",
    )
    path = write_items(directory, [item])[0]
    data = json.loads(path.read_text())
    assert path.parent == directory
    assert data["evidence"]["concept_id"] == "island"
    assert data["image_path"] == "/book/a.png"
    assert unresolved_items(directory)[0].item_id == item.item_id

    resolve_item(path, "correct", correction="isle")
    assert unresolved_items(directory) == []
    assert json.loads(path.read_text())["resolution"] == {"action": "correct", "correction": "isle"}


def test_current_review_set_archives_stale_artifacts_without_applying_them(tmp_path: Path):
    """Only stable IDs in this run participate; resolved recurring items survive."""
    directory = tmp_path / "needs_review"
    stale = ReviewItem("old-concept", "low_confidence", True, {"concept_id": "old"})
    current = ReviewItem("current-concept", "low_confidence", True, {"concept_id": "current"})
    write_items(directory, [stale])
    resolve_item(directory / "old-concept.json", "correct", correction="obsolete")
    write_items(directory, [current], archive_stale=True)

    assert not (directory / "old-concept.json").exists()
    assert (directory / "archive" / "old-concept.json").is_file()
    assert unresolved_items(directory) == [current]

    resolve_item(directory / "current-concept.json", "accept")
    write_items(directory, [current], archive_stale=True)
    assert unresolved_items(directory) == []
    assert json.loads((directory / "current-concept.json").read_text())["status"] == "resolved"


def test_resolved_item_reappearing_after_an_absent_run_recovers_archived_decision(tmp_path: Path):
    """Nondeterministic glossary omission must not erase a stable review decision."""
    directory = tmp_path / "needs_review"
    recurring = ReviewItem(
        "recurring", "low_confidence", True,
        {"concept_id": "stable-concept", "target_term": "first wording"},
    )
    write_items(directory, [recurring], archive_stale=True)
    resolve_item(directory / "recurring.json", "correct", correction="reviewed wording")

    write_items(directory, [], archive_stale=True)
    assert not (directory / "recurring.json").exists()
    assert (directory / "archive" / "recurring.json").is_file()

    regenerated = ReviewItem(
        "recurring", "low_confidence", True,
        {"concept_id": "stable-concept", "target_term": "different model wording"},
    )
    write_items(directory, [regenerated], archive_stale=True)

    assert unresolved_items(directory) == []
    assert corrections(directory) == {"stable-concept": "reviewed wording"}


def test_malformed_current_review_artifact_blocks_instead_of_being_replaced(tmp_path: Path):
    """A current malformed decision cannot be silently accepted or overwritten."""
    directory = tmp_path / "needs_review"
    directory.mkdir()
    (directory / "current.json").write_text("not json")
    current = ReviewItem("current", "low_confidence", True, {"concept_id": "current"})

    write_items(directory, [current], archive_stale=True)

    pending = unresolved_items(directory)
    assert len(pending) == 1
    assert pending[0].kind == "malformed_review_artifact"


def test_reused_review_id_cannot_apply_a_correction_to_a_different_concept(tmp_path: Path):
    """A forged/stale stable ID may not transplant its correction to this run's concept."""
    directory = tmp_path / "needs_review"
    current = ReviewItem("shared-id", "low_confidence", True, {"concept_id": "current"})
    unrelated = ReviewItem(
        "shared-id", "low_confidence", True, {"concept_id": "unrelated"},
        status="resolved", resolution={"action": "correct", "correction": "wrong term"},
    )
    write_items(directory, [unrelated])

    write_items(directory, [current], archive_stale=True)

    assert unresolved_items(directory) == [current]
    assert corrections(directory) == {}


def test_invalid_current_resolution_shape_blocks_review(tmp_path: Path):
    """A syntactically JSON but invalid operator decision is also malformed."""
    directory = tmp_path / "needs_review"
    item = ReviewItem(
        "current", "low_confidence", True, {"concept_id": "current"},
        status="resolved", resolution={"action": "correct", "correction": " "},
    )
    write_items(directory, [item])

    assert unresolved_items(directory)[0].kind == "malformed_review_artifact"
