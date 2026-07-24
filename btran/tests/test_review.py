from __future__ import annotations

import json
from pathlib import Path

from btran.review import ReviewItem, unresolved_items, write_items, resolve_item


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
