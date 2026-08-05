"""Immutable correction records and deterministic overlay selection.

This module intentionally stops at storage and selection.  It neither publishes
correction commands nor materializes effective source/target content.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from btran.artifacts import ArtifactError, ArtifactStore, RevisionStore, correction_semantic_key
from btran.schema import (
    SCHEMA_VERSION,
    ArtifactEnvelope,
    CorrectionEvent,
    CorrectionImpact,
    CorrectionRecord,
    Finding,
    SchemaError,
    canonical_json_bytes,
    tagged_sha256,
)


class CorrectionError(ValueError):
    """A correction payload, immutable record, or selected set is invalid."""


_CORRECTION_KINDS = frozenset({"source_text", "target_occurrence", "target_segment", "terminology"})
_HEX64 = frozenset("0123456789abcdef")


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != unicodedata.normalize("NFC", value):
        raise CorrectionError(f"{name} must be a non-empty NFC string")
    return value


def _sha256(value: Any, name: str = "sha256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX64 for char in value):
        raise CorrectionError(f"{name} must be lower-case SHA-256 hex")
    return value


def _exact_object(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        raise CorrectionError(f"{name} fields mismatch: expected={sorted(fields)}, actual={actual}")
    return value


def _base_ref(value: Any, name: str = "base") -> dict[str, str]:
    value = _exact_object(value, {"artifact_id", "sha256"}, name)
    return {"artifact_id": _id(value["artifact_id"], f"{name}.artifact_id"), "sha256": _sha256(value["sha256"], f"{name}.sha256")}


def _subjects(kind: str, scope: Mapping[str, Any]) -> tuple[str, ...]:
    if kind == "source_text" or kind == "target_segment":
        return (scope["segment_id"],)
    if kind == "target_occurrence":
        return tuple(sorted((scope["occurrence_id"], scope["segment_id"], scope["mapping_id"])))
    selector = scope["selector"]
    ids = (scope["concept_id"],) if selector["kind"] == "all_concept_occurrences" else (scope["concept_id"], *selector["ids"])
    return tuple(sorted(ids))


def _base_hashes(kind: str, base: Mapping[str, Any]) -> dict[str, str]:
    if kind == "terminology":
        return {"membership": base["membership"]["sha256"], "projection": base["projection"]["sha256"]}
    return {"source" if kind == "source_text" else "translation": base["sha256"]}


def validate_correction_payload(payload: Mapping[str, Any], *, supersedes_id: str | None = None) -> dict[str, Any]:
    """Reject everything except Task 5's exact correction JSON grammar.

    Returned mapping is canonicalizable and deliberately excludes transport-only
    correction IDs.  Supersession is record metadata, not an unvalidated JSON
    extension to payload grammar.
    """
    payload = _exact_object(payload, {"kind", "applies_to_revision_id", "scope", "base", "replacement"}, "correction payload")
    kind = payload["kind"]
    if kind not in _CORRECTION_KINDS:
        raise CorrectionError("correction kind is invalid")
    revision = _id(payload["applies_to_revision_id"], "applies_to_revision_id")
    if not isinstance(payload["replacement"], str):
        raise CorrectionError("replacement must be a string")
    replacement = unicodedata.normalize("NFC", payload["replacement"])
    if supersedes_id is not None:
        _id(supersedes_id, "supersedes_id")

    scope_value = payload["scope"]
    if kind == "source_text":
        scope = _exact_object(scope_value, {"segment_id"}, "source_text scope")
        scope = {"segment_id": _id(scope["segment_id"], "scope.segment_id")}
        base: dict[str, Any] = _base_ref(payload["base"], "source base")
    elif kind == "target_occurrence":
        scope = _exact_object(scope_value, {"occurrence_id", "segment_id", "mapping_id", "start", "end", "expected_target_text"}, "target_occurrence scope")
        start, end = scope["start"], scope["end"]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise CorrectionError("target occurrence range must be a non-empty half-open range")
        expected = scope["expected_target_text"]
        if not isinstance(expected, str):
            raise CorrectionError("scope.expected_target_text must be a string")
        scope = {
            "occurrence_id": _id(scope["occurrence_id"], "scope.occurrence_id"),
            "segment_id": _id(scope["segment_id"], "scope.segment_id"),
            "mapping_id": _id(scope["mapping_id"], "scope.mapping_id"),
            "start": start, "end": end,
            "expected_target_text": unicodedata.normalize("NFC", expected),
        }
        base = _base_ref(payload["base"], "translation base")
    elif kind == "target_segment":
        scope = _exact_object(scope_value, {"segment_id", "expected_target_text"}, "target_segment scope")
        expected = scope["expected_target_text"]
        if not isinstance(expected, str):
            raise CorrectionError("scope.expected_target_text must be a string")
        scope = {"segment_id": _id(scope["segment_id"], "scope.segment_id"), "expected_target_text": unicodedata.normalize("NFC", expected)}
        base = _base_ref(payload["base"], "translation base")
    else:
        scope = _exact_object(scope_value, {"concept_id", "selector"}, "terminology scope")
        selector_value = scope["selector"]
        if not isinstance(selector_value, Mapping):
            raise CorrectionError("terminology selector must be an object")
        selector_kind = selector_value.get("kind")
        if selector_kind == "all_concept_occurrences":
            selector = _exact_object(selector_value, {"kind"}, "terminology all selector")
            selector = {"kind": selector_kind}
        elif selector_kind == "occurrence_ids":
            selector = _exact_object(selector_value, {"kind", "ids"}, "terminology subset selector")
            ids = selector["ids"]
            if not isinstance(ids, list) or not ids:
                raise CorrectionError("terminology occurrence IDs must be a non-empty array")
            parsed_ids = tuple(_id(item, "scope.selector.ids") for item in ids)
            if tuple(sorted(set(parsed_ids))) != parsed_ids:
                raise CorrectionError("terminology occurrence IDs must be sorted and unique")
            selector = {"kind": selector_kind, "ids": list(parsed_ids)}
        else:
            raise CorrectionError("terminology selector kind is invalid")
        scope = {"concept_id": _id(scope["concept_id"], "scope.concept_id"), "selector": selector}
        base_value = _exact_object(payload["base"], {"projection", "membership"}, "terminology base")
        base = {"projection": _base_ref(base_value["projection"], "projection base"), "membership": _base_ref(base_value["membership"], "membership base")}

    return {"kind": kind, "applies_to_revision_id": revision, "scope": scope, "base": base, "replacement": replacement}


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CorrectionError("correction JSON must not contain duplicate keys")
        value[key] = item
    return value


def parse_correction_json(raw: bytes | str) -> dict[str, Any]:
    """Read one canonical UTF-8 JSON object; no alternate JSON spellings."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raise CorrectionError("correction JSON must be UTF-8 bytes or text")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorrectionError("correction JSON must be valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise CorrectionError("correction JSON must be one object")
    try:
        canonical = canonical_json_bytes(dict(value))
    except (SchemaError, TypeError, ValueError) as exc:
        raise CorrectionError("correction JSON is not canonical JSON data") from exc
    if raw != canonical:
        raise CorrectionError("correction JSON must use exact canonical UTF-8 grammar")
    return validate_correction_payload(value)


def correction_id_for(payload: Mapping[str, Any], *, supersedes_id: str | None = None) -> str:
    validated = validate_correction_payload(payload, supersedes_id=supersedes_id)
    return correction_semantic_key(
        kind=validated["kind"], revision_id=validated["applies_to_revision_id"],
        subjects=_subjects(validated["kind"], validated["scope"]), base_hashes=_base_hashes(validated["kind"], validated["base"]),
        replacement=validated["replacement"], explicit_scope=validated["scope"], supersedes_id=supersedes_id,
    )


def correction_record_for(payload: Mapping[str, Any], *, supersedes_id: str | None = None) -> CorrectionRecord:
    validated = validate_correction_payload(payload, supersedes_id=supersedes_id)
    return CorrectionRecord(correction_id=correction_id_for(validated, supersedes_id=supersedes_id), supersedes_id=supersedes_id, **validated)


@dataclass(frozen=True)
class CorrectionSet:
    set_id: str
    base_revision_id: str
    active_correction_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CorrectionError("unsupported correction set schema version")
        _id(self.set_id, "set_id")
        _id(self.base_revision_id, "base_revision_id")
        for name in ("active_correction_ids", "event_ids"):
            ids = getattr(self, name)
            if not isinstance(ids, tuple) or tuple(sorted(set(_id(item, name) for item in ids))) != ids:
                raise CorrectionError(f"{name} must be sorted and unique")
        expected = correction_set_id_for(self.base_revision_id, self.active_correction_ids, self.event_ids)
        if self.set_id != expected:
            raise CorrectionError("correction set ID does not match content")

    @classmethod
    def create(cls, base_revision_id: str, active_correction_ids: Sequence[str] = (), event_ids: Sequence[str] = ()) -> "CorrectionSet":
        active = tuple(sorted(set(_id(value, "active_correction_ids") for value in active_correction_ids)))
        events = tuple(sorted(set(_id(value, "event_ids") for value in event_ids)))
        if len(active) != len(active_correction_ids) or len(events) != len(event_ids):
            raise CorrectionError("correction set IDs must be unique")
        return cls(correction_set_id_for(base_revision_id, active, events), _id(base_revision_id, "base_revision_id"), active, events)

    def to_dict(self) -> dict[str, Any]:
        return {"active_correction_ids": list(self.active_correction_ids), "base_revision_id": self.base_revision_id, "event_ids": list(self.event_ids), "schema_version": self.schema_version, "set_id": self.set_id}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorrectionSet":
        if not isinstance(value, Mapping) or set(value) != {"set_id", "base_revision_id", "active_correction_ids", "event_ids", "schema_version"}:
            raise CorrectionError("correction set fields mismatch")
        if not isinstance(value["active_correction_ids"], list) or not isinstance(value["event_ids"], list):
            raise CorrectionError("correction set ID lists must be arrays")
        return cls(value["set_id"], value["base_revision_id"], tuple(value["active_correction_ids"]), tuple(value["event_ids"]), value["schema_version"])

    @classmethod
    def from_json(cls, raw: bytes | str) -> "CorrectionSet":
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        try:
            result = cls.from_dict(json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise CorrectionError("invalid correction set JSON") from exc
        if raw != result.to_json().encode("utf-8"):
            raise CorrectionError("correction set JSON is not canonical")
        return result


def correction_set_id_for(base_revision_id: str, active_correction_ids: Sequence[str], event_ids: Sequence[str]) -> str:
    base = _id(base_revision_id, "base_revision_id")
    active = tuple(sorted(set(_id(value, "active_correction_ids") for value in active_correction_ids)))
    events = tuple(sorted(set(_id(value, "event_ids") for value in event_ids)))
    if len(active) != len(active_correction_ids) or len(events) != len(event_ids):
        raise CorrectionError("correction set IDs must be unique")
    return tagged_sha256("correction-set-v1", canonical_json_bytes({"schema_version": SCHEMA_VERSION, "base_revision_id": base, "active_correction_ids": list(active), "event_ids": list(events)}))


def _successor_state(base_revision_id: str, active_correction_ids: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Canonical non-derived fields describing a successor correction set."""
    base = _id(base_revision_id, "successor_base_revision_id")
    active = tuple(sorted(set(_id(value, "successor_active_correction_ids") for value in active_correction_ids)))
    if len(active) != len(active_correction_ids):
        raise CorrectionError("successor_active_correction_ids must be unique")
    return base, active


def _other_event_ids(event_ids: Sequence[str], event_id: str) -> tuple[str, ...]:
    """Return canonical event IDs in a successor set other than ``event_id``."""
    event_id = _id(event_id, "event_id")
    ids = tuple(sorted(set(_id(value, "event_ids") for value in event_ids)))
    if len(ids) != len(event_ids) or event_id not in ids:
        raise CorrectionError("successor set must contain its event ID exactly once")
    return tuple(value for value in ids if value != event_id)


def correction_event_id_for(
    correction_id: str, event_kind: str, successor_base_revision_id: str,
    successor_active_correction_ids: Sequence[str], other_event_ids: Sequence[str] = (),
) -> str:
    """Return non-circular ID bound to exact successor event-ID closure.

    A correction set hashes all its event IDs.  This event instead hashes every
    *other* ID in that set, leaving only its own derived ID out of the input.
    Thus callers first retain prior IDs, derive this ID, then create the set
    from ``other_event_ids + [event_id]``.  On read, linkage validation requires
    that equality exactly; membership alone is insufficient.
    """
    if event_kind not in {"apply", "revert", "supersede"}:
        raise CorrectionError("correction event kind is invalid")
    base, active = _successor_state(successor_base_revision_id, successor_active_correction_ids)
    others = tuple(sorted(set(_id(value, "other_event_ids") for value in other_event_ids)))
    if len(others) != len(other_event_ids):
        raise CorrectionError("other_event_ids must be unique")
    return tagged_sha256(
        "correction-event-v3",
        canonical_json_bytes({
            "schema_version": SCHEMA_VERSION,
            "correction_id": _id(correction_id, "correction_id"),
            "event_kind": event_kind,
            "successor_base_revision_id": base,
            "successor_active_correction_ids": list(active),
            "other_event_ids": list(others),
        }),
    )


def correction_event_for(correction_id: str, event_kind: str, successor_set: CorrectionSet) -> CorrectionEvent:
    """Build event whose ID proves ``successor_set``'s exact event-ID list.

    The caller has already derived one candidate event ID and included it in
    the set.  Find that candidate by recomputing each possible own-ID exclusion;
    this preserves straightforward apply/revert/supersede construction without
    introducing a hash cycle.
    """
    if not isinstance(successor_set, CorrectionSet):
        raise CorrectionError("successor_set must be CorrectionSet")
    candidates = [
        event_id for event_id in successor_set.event_ids
        if correction_event_id_for(
            correction_id, event_kind, successor_set.base_revision_id,
            successor_set.active_correction_ids,
            _other_event_ids(successor_set.event_ids, event_id),
        ) == event_id
    ]
    if len(candidates) != 1:
        raise CorrectionError("successor set must contain exactly one matching event ID")
    event_id = candidates[0]
    return CorrectionEvent(
        event_id=event_id, correction_id=_id(correction_id, "correction_id"), event_kind=event_kind,
        correction_set_id=successor_set.set_id,
        successor_base_revision_id=successor_set.base_revision_id,
        successor_active_correction_ids=successor_set.active_correction_ids,
    )


def base_hash_for_artifact(envelope: ArtifactEnvelope) -> str:
    """Hash exact canonical persisted selected ``ArtifactEnvelope`` bytes."""
    if not isinstance(envelope, ArtifactEnvelope):
        raise CorrectionError("base must name an ArtifactEnvelope")
    return hashlib.sha256(envelope.to_json().encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class CorrectionStore:
    """Content-addressed correction records/events/sets.  No command policy."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.records_dir = self.root / "corrections" / "records"
        self.events_dir = self.root / "corrections" / "events"
        self.sets_dir = self.root / "corrections" / "sets"
        self.impacts_dir = self.root / "corrections" / "impacts"
        self.plan_links_dir = self.root / "corrections" / "plan-links"
        # Reading a legacy workspace must be side-effect free.  Immutable
        # correction writes create their own parent directories in ``_put``.
        self.artifacts = ArtifactStore(self.root)

    @staticmethod
    def _put(path: Path, data: bytes) -> None:
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise CorrectionError("cannot read immutable correction record") from exc
            if existing != data:
                raise CorrectionError("immutable correction ID already has different content")
            return
        _atomic_bytes(path, data)

    @staticmethod
    def _record_payload(record: CorrectionRecord) -> dict[str, Any]:
        return {"kind": record.kind, "applies_to_revision_id": record.applies_to_revision_id, "scope": record.scope, "base": record.base, "replacement": record.replacement}

    def put_record(self, record: CorrectionRecord) -> CorrectionRecord:
        if not isinstance(record, CorrectionRecord) or correction_id_for(self._record_payload(record), supersedes_id=record.supersedes_id) != record.correction_id:
            raise CorrectionError("correction record content hash is invalid")
        self._put(self.records_dir / f"{record.correction_id}.json", record.to_json().encode("utf-8"))
        return record

    def get_record(self, correction_id: str) -> CorrectionRecord:
        try:
            record = CorrectionRecord.from_file(self.records_dir / f"{_id(correction_id, 'correction_id')}.json")
        except (OSError, SchemaError) as exc:
            raise CorrectionError("missing or invalid correction record") from exc
        if correction_id_for(self._record_payload(record), supersedes_id=record.supersedes_id) != record.correction_id:
            raise CorrectionError("correction record content hash is invalid")
        return record

    @staticmethod
    def _validate_event_hash(event: CorrectionEvent, other_event_ids: Sequence[str]) -> None:
        if not isinstance(event, CorrectionEvent) or correction_event_id_for(
            event.correction_id, event.event_kind, event.successor_base_revision_id,
            event.successor_active_correction_ids, other_event_ids,
        ) != event.event_id:
            raise CorrectionError("correction event content hash is invalid")

    def _validate_event_linkage(self, event: CorrectionEvent) -> None:
        """Prove event's record and set pointer describe one exact successor."""
        # An event cannot name an absent/tampered/cross-revision record merely
        # because its opaque ID appears in an otherwise well-formed set.
        record = self.get_record(event.correction_id)
        correction_set = self.get_set(event.correction_set_id)
        if (
            correction_set.base_revision_id != event.successor_base_revision_id
            or correction_set.active_correction_ids != event.successor_active_correction_ids
        ):
            raise CorrectionError("correction event successor-set linkage is invalid")
        other_event_ids = _other_event_ids(correction_set.event_ids, event.event_id)
        self._validate_event_hash(event, other_event_ids)
        if record.applies_to_revision_id != correction_set.base_revision_id:
            raise CorrectionError("correction event record/set revision linkage is invalid")
        for active_id in correction_set.active_correction_ids:
            active_record = self.get_record(active_id)
            if active_record.applies_to_revision_id != correction_set.base_revision_id:
                raise CorrectionError("correction event active-record set linkage is invalid")
        is_active = event.correction_id in correction_set.active_correction_ids
        if event.event_kind in {"apply", "supersede"} and not is_active:
            raise CorrectionError("correction event successor state is invalid")
        if event.event_kind == "revert" and is_active:
            raise CorrectionError("correction event successor state is invalid")
        if event.event_kind == "supersede":
            if record.supersedes_id is None:
                raise CorrectionError("supersede event requires predecessor record")
            self.get_record(record.supersedes_id)

    def put_event(self, event: CorrectionEvent) -> CorrectionEvent:
        # Set/event immutable files may be written in either order.  Full hash
        # and exact set-closure validation therefore belongs to get_event.
        if not isinstance(event, CorrectionEvent):
            raise CorrectionError("correction event is invalid")
        self._put(self.events_dir / f"{event.event_id}.json", event.to_json().encode("utf-8"))
        return event

    def get_event(self, event_id: str) -> CorrectionEvent:
        try:
            event = CorrectionEvent.from_file(self.events_dir / f"{_id(event_id, 'event_id')}.json")
        except (OSError, SchemaError) as exc:
            raise CorrectionError("missing or invalid correction event") from exc
        try:
            self._validate_event_linkage(event)
        except (OSError, SchemaError, CorrectionError) as exc:
            raise CorrectionError("correction event record/set linkage is invalid") from exc
        return event

    def put_set(self, correction_set: CorrectionSet) -> CorrectionSet:
        if not isinstance(correction_set, CorrectionSet):
            raise CorrectionError("correction_set must be CorrectionSet")
        self._put(self.sets_dir / f"{correction_set.set_id}.json", correction_set.to_json().encode("utf-8"))
        return correction_set

    def get_set(self, set_id: str) -> CorrectionSet:
        try:
            return CorrectionSet.from_json((self.sets_dir / f"{_id(set_id, 'set_id')}.json").read_bytes())
        except OSError as exc:
            raise CorrectionError("missing correction set") from exc

    def _impact_path(self, projection_plan_id: str, phase: str) -> Path:
        projection_plan_id = _id(projection_plan_id, "projection_plan_id")
        if phase == "correction":
            return self.impacts_dir / f"{projection_plan_id}.json"
        if phase == "execution":
            return self.impacts_dir / f"{projection_plan_id}.execution.json"
        raise CorrectionError("correction impact phase is invalid")

    def put_impact(self, impact: CorrectionImpact) -> CorrectionImpact:
        """Persist immutable correction projection or later linked execution impact."""
        if not isinstance(impact, CorrectionImpact):
            raise CorrectionError("correction impact must be a CorrectionImpact")
        self._put(self._impact_path(impact.projection_plan_id, impact.phase), impact.to_json().encode("utf-8"))
        if impact.phase == "correction" and impact.correction_id is not None:
            correction_set = self.get_set(impact.correction_set_id)
            # Revert plans describe a removal, never an applied correction.
            # Reapplication gets a distinct immutable link/plan pair.
            if impact.correction_id in correction_set.active_correction_ids:
                link = {
                    "correction_id": impact.correction_id,
                    "correction_set_id": impact.correction_set_id,
                    "projection_plan_id": impact.projection_plan_id,
                }
                self._put(
                    self.plan_links_dir / impact.correction_id / f"{impact.projection_plan_id}.json",
                    canonical_json_bytes(link),
                )
        return impact

    def get_impact(self, projection_plan_id: str, *, phase: str = "correction") -> CorrectionImpact:
        try:
            impact = CorrectionImpact.from_file(self._impact_path(projection_plan_id, phase))
        except (OSError, SchemaError) as exc:
            raise CorrectionError("missing or invalid correction impact") from exc
        if impact.projection_plan_id != projection_plan_id or impact.phase != phase:
            raise CorrectionError("correction impact path and plan ID differ")
        return impact

    def correction_time_impact(self, correction_id: str) -> CorrectionImpact:
        """Load one correction's immutable plan and its original event-set basis.

        Impact filenames are addressed by plan ID, so retain the correction/set
        binding in the plan body and resolve it here.  Execution must never
        derive this address from a later active correction set.
        """
        correction_id = _id(correction_id, "correction_id")
        candidates: list[tuple[int, CorrectionImpact]] = []
        try:
            for path in sorted((self.plan_links_dir / correction_id).glob("*.json")):
                raw_link = path.read_bytes()
                link = json.loads(raw_link)
                if (
                    set(link) != {"correction_id", "correction_set_id", "projection_plan_id"}
                    or raw_link != canonical_json_bytes(link)
                    or link["correction_id"] != correction_id
                ):
                    raise CorrectionError("correction-time impact link is invalid")
                impact = self.get_impact(_id(link["projection_plan_id"], "projection_plan_id"))
                if (
                    impact.phase != "correction"
                    or impact.correction_id != correction_id
                    or impact.correction_set_id != link["correction_set_id"]
                ):
                    raise CorrectionError("correction-time impact has no event-set basis")
                record = self.get_record(correction_id)
                correction_set = self.get_set(impact.correction_set_id)
                events = tuple(self.get_event(event_id) for event_id in correction_set.event_ids)
                if (
                    impact.base_revision_id != record.applies_to_revision_id
                    or correction_set.base_revision_id != record.applies_to_revision_id
                    or correction_id not in correction_set.active_correction_ids
                    or not any(
                        event.correction_id == correction_id
                        and event.correction_set_id == correction_set.set_id
                        and event.event_kind in {"apply", "supersede"}
                        for event in events
                    )
                ):
                    raise CorrectionError("correction-time impact basis is invalid")
                candidates.append((len(correction_set.event_ids), impact))
        except (OSError, TypeError, json.JSONDecodeError, SchemaError, CorrectionError) as exc:
            raise CorrectionError("missing or invalid correction-time impact") from exc
        if not candidates:
            raise CorrectionError("missing or invalid correction-time impact")
        # Each command adds one event.  Latest active application of a reused
        # correction record therefore has largest immutable event-set basis.
        return max(candidates, key=lambda item: (item[0], item[1].projection_plan_id))[1]

    def persist_findings(self, findings: Sequence[Finding]) -> tuple[str, ...]:
        """Persist diagnostics when writable; keep legacy reads in memory."""
        ids = tuple(finding.finding_id for finding in findings)
        try:
            for finding in findings:
                self.artifacts.put_finding(finding)
        except (ArtifactError, OSError):
            # Legacy stores are explicitly read/verify-only.  Resolution still
            # returns typed findings to its caller, but never creates a DB or
            # compatibility sidecar while inspecting them.
            return ids
        return ids


@dataclass(frozen=True)
class OverlayInput:
    correction_id: str
    kind: str
    subject_id: str
    replacement: str
    base_artifact_ids: tuple[str, ...]
    scope: dict[str, Any]


@dataclass(frozen=True)
class OverlayResolution:
    base_revision_id: str
    correction_set_id: str | None
    source: tuple[OverlayInput, ...] = ()
    terminology: tuple[OverlayInput, ...] = ()
    target_occurrence: tuple[OverlayInput, ...] = ()
    target_segment: tuple[OverlayInput, ...] = ()
    findings: tuple[Finding, ...] = ()
    applicable_correction_ids: tuple[str, ...] = ()

    @property
    def source_inputs(self) -> tuple[OverlayInput, ...]:
        return self.source

    @property
    def terminology_inputs(self) -> tuple[OverlayInput, ...]:
        return self.terminology

    @property
    def target_inputs(self) -> tuple[OverlayInput, ...]:
        return tuple(sorted((*self.target_occurrence, *self.target_segment), key=lambda item: (item.kind, item.subject_id, item.correction_id)))


_CORRECTION_AUDIT_CATEGORIES = {
    "correction_conflict": "conflict",
    "correction_mapping_ambiguous": "actionable_ambiguity",
    "correction_ambiguity": "actionable_ambiguity",
    "correction_stale": "validation",
    "correction_inapplicable": "validation",
    "correction_set_inapplicable": "validation",
}


def _finding(kind: str, correction: CorrectionRecord, reason: str, *, severity: str = "warning", related: Sequence[str] = ()) -> Finding:
    refs = tuple(sorted(set((correction.correction_id, *related))))
    category = _CORRECTION_AUDIT_CATEGORIES.get(kind)
    evidence = {
        "correction_id": correction.correction_id,
        "applies_to_revision_id": correction.applies_to_revision_id,
        "reason": reason,
        "trigger": reason,
    }
    return Finding(kind=kind, severity=severity, stage="corrections", subject_refs=refs,
                   evidence=evidence, audit_category=category, message=kind.replace("_", " "))


def _set_finding(correction_set: CorrectionSet, reason: str, subject_ids: Sequence[str]) -> Finding:
    return Finding(kind="correction_set_inapplicable", severity="warning", stage="corrections",
                   subject_refs=tuple(sorted(set(subject_ids))),
                   evidence={"correction_set_id": correction_set.set_id,
                             "base_revision_id": correction_set.base_revision_id,
                             "reason": reason, "trigger": reason},
                   audit_category="validation", message="correction set inapplicable")


def _base_refs(record: CorrectionRecord) -> tuple[dict[str, str], ...]:
    if record.kind == "terminology":
        return (record.base["membership"], record.base["projection"])
    return (record.base,)


def _relation_reason(record: CorrectionRecord, selected_artifacts: Mapping[str, ArtifactEnvelope]) -> str | None:
    """Validate exact kind-specific relation evidence; never retarget."""
    def payload(base: Mapping[str, str]) -> Mapping[str, Any]:
        return selected_artifacts[base["artifact_id"]].payload

    if record.kind == "source_text":
        body = payload(record.base)
        if "segment_id" not in body:
            return "source_relation_missing"
        return None if body["segment_id"] == record.scope["segment_id"] else "source_segment_mismatch"
    if record.kind == "target_segment":
        artifact = selected_artifacts[record.base["artifact_id"]]
        body = artifact.payload
        # Local target overlays are defined against an immutable translation
        # leaf, never an already-materialized effective target.  Otherwise the
        # target stage cannot apply it locally and would incorrectly call a
        # model after accepting the correction.
        if artifact.kind not in {"TranslationArtifact", "DiagnosticTranslationFallback"}:
            return "target_relation_missing"
        if "segment_id" not in body or not isinstance(body.get("translated_text"), str):
            return "target_relation_missing"
        if body["segment_id"] != record.scope["segment_id"]:
            return "translation_segment_mismatch"
        if body["translated_text"] != record.scope["expected_target_text"]:
            return "expected_target_text_mismatch"
        return None
    if record.kind == "target_occurrence":
        artifact = selected_artifacts[record.base["artifact_id"]]
        body = artifact.payload
        if artifact.kind not in {"TranslationArtifact", "DiagnosticTranslationFallback"}:
            return "target_relation_missing"
        if "segment_id" not in body or "mappings" not in body or not isinstance(body.get("translated_text"), str):
            return "target_relation_missing"
        if body["segment_id"] != record.scope["segment_id"]:
            return "translation_segment_mismatch"
        mappings = body["mappings"]
        if not isinstance(mappings, list):
            return "mapping_evidence_invalid"
        candidates = [item for item in mappings if isinstance(item, Mapping) and item.get("mapping_id") == record.scope["mapping_id"]]
        if len(candidates) > 1:
            return "mapping_ambiguous"
        if len(candidates) != 1:
            return "mapping_not_found"
        mapping = candidates[0]
        for name in ("occurrence_id", "segment_id", "start", "end", "target_text"):
            expected = record.scope["expected_target_text"] if name == "target_text" else record.scope[name]
            if mapping.get(name) != expected:
                return "mapping_mismatch"
        start, end = record.scope["start"], record.scope["end"]
        if body["translated_text"][start:end] != record.scope["expected_target_text"]:
            return "mapping_text_mismatch"
        return None
    projection = payload(record.base["projection"])
    membership = payload(record.base["membership"])
    concept = record.scope["concept_id"]
    if "concept_id" not in projection or "concept_id" not in membership:
        return "concept_relation_missing"
    if projection["concept_id"] != concept or membership["concept_id"] != concept:
        return "concept_mismatch"
    if "occurrence_ids" not in membership:
        return "membership_relation_missing"
    known = membership["occurrence_ids"]
    if not isinstance(known, list):
        return "membership_relation_invalid"
    selector = record.scope["selector"]
    if selector["kind"] == "occurrence_ids" and not set(selector["ids"]).issubset(known):
        return "selector_not_verified"
    return None


def _overlay(record: CorrectionRecord) -> OverlayInput:
    if record.kind == "source_text":
        subject = record.scope["segment_id"]
    elif record.kind == "target_occurrence":
        subject = record.scope["occurrence_id"]
    elif record.kind == "target_segment":
        subject = record.scope["segment_id"]
    else:
        subject = record.scope["concept_id"]
    return OverlayInput(record.correction_id, record.kind, subject, record.replacement, tuple(sorted(ref["artifact_id"] for ref in _base_refs(record))), dict(record.scope))


def resolve_correction_set(
    correction_set: CorrectionSet | None, records: Sequence[CorrectionRecord], *, base_revision_id: str,
    selected_artifacts: Mapping[str, ArtifactEnvelope],
    event_activated_correction_ids: frozenset[str] | None = None,
) -> OverlayResolution:
    """Resolve only matching selected base/set closure.  Never retarget records."""
    _id(base_revision_id, "base_revision_id")
    # ``records`` deliberately includes retained historical records.  Active
    # IDs choose overlays; they do not truncate immutable supersession ancestry.
    record_by_id = {record.correction_id: record for record in records}
    findings: list[Finding] = []
    if correction_set is None:
        return OverlayResolution(base_revision_id, None, findings=())
    if correction_set.base_revision_id != base_revision_id:
        for correction_id in correction_set.active_correction_ids:
            record = record_by_id.get(correction_id)
            if record is not None:
                findings.append(_finding("correction_set_inapplicable", record, "base_revision_mismatch"))
        return OverlayResolution(base_revision_id, correction_set.set_id, findings=tuple(sorted(findings, key=lambda item: item.finding_id)))
    missing = tuple(sorted(set(correction_set.active_correction_ids) - set(record_by_id)))
    if missing:
        findings.append(_set_finding(correction_set, "record_closure_incomplete", missing))
        for record in record_by_id.values():
            if record.correction_id in correction_set.active_correction_ids:
                findings.append(_finding("correction_set_inapplicable", record, "record_closure_incomplete"))
        return OverlayResolution(base_revision_id, correction_set.set_id, findings=tuple(sorted(findings, key=lambda item: item.finding_id)))

    active_records = {correction_id: record_by_id[correction_id] for correction_id in correction_set.active_correction_ids}

    def supersession_target(record: CorrectionRecord) -> tuple[str, str, str]:
        if record.kind == "source_text" or record.kind == "target_segment":
            return record.kind, record.scope["segment_id"], ""
        if record.kind == "target_occurrence":
            return record.kind, record.scope["occurrence_id"], record.scope["mapping_id"]
        return record.kind, record.scope["concept_id"], canonical_json_bytes(record.scope["selector"]).decode("utf-8")

    def predecessor_closure_reason(record: CorrectionRecord) -> str | None:
        """Check retained immutable record ancestry; predecessor need not be active."""
        predecessor_id = record.supersedes_id
        depth = 0
        seen = {record.correction_id}
        child = record
        while predecessor_id is not None:
            if predecessor_id in seen:
                return "supersedes_predecessor_cycle"
            predecessor = record_by_id.get(predecessor_id)
            if predecessor is None:
                return "supersedes_predecessor_missing" if depth == 0 else "supersedes_predecessor_closure_incomplete"
            if supersession_target(child) != supersession_target(predecessor):
                return "supersedes_predecessor_scope_mismatch"
            seen.add(predecessor_id)
            child = predecessor
            predecessor_id = predecessor.supersedes_id
            depth += 1
        return None

    # Validate a historical ancestor against the selected base exactly as an
    # active record.  A historical predecessor is valid evidence, but a stale,
    # mismatched, or relation-invalid one invalidates every successor.
    own_valid: dict[str, bool] = {}

    def own_applicable(record: CorrectionRecord) -> bool:
        cached = own_valid.get(record.correction_id)
        if cached is not None:
            return cached
        try:
            valid_id = correction_id_for(self_payload(record), supersedes_id=record.supersedes_id) == record.correction_id
        except (CorrectionError, KeyError, TypeError):
            valid_id = False
        if not valid_id:
            findings.append(_finding("correction_inapplicable", record, "record_content_hash_invalid"))
            own_valid[record.correction_id] = False
            return False
        if event_activated_correction_ids is not None and record.correction_id not in event_activated_correction_ids:
            findings.append(_finding("correction_inapplicable", record, "record_event_ancestry_missing"))
            own_valid[record.correction_id] = False
            return False
        if record.applies_to_revision_id != base_revision_id:
            findings.append(_finding("correction_set_inapplicable", record, "record_revision_mismatch"))
            own_valid[record.correction_id] = False
            return False
        invalid_reason: str | None = None
        for base in _base_refs(record):
            envelope = selected_artifacts.get(base["artifact_id"])
            if envelope is None:
                invalid_reason = "base_artifact_not_selected"
                break
            if base_hash_for_artifact(envelope) != base["sha256"]:
                invalid_reason = "base_hash_mismatch"
                break
        if invalid_reason is not None:
            findings.append(_finding("correction_stale", record, invalid_reason))
            own_valid[record.correction_id] = False
            return False
        relation_reason = _relation_reason(record, selected_artifacts)
        if relation_reason is not None:
            finding_kind = "correction_mapping_ambiguous" if relation_reason == "mapping_ambiguous" else "correction_inapplicable"
            findings.append(_finding(finding_kind, record, relation_reason))
            own_valid[record.correction_id] = False
            return False
        own_valid[record.correction_id] = True
        return True

    def self_payload(record: CorrectionRecord) -> dict[str, Any]:
        return {"kind": record.kind, "applies_to_revision_id": record.applies_to_revision_id,
                "scope": record.scope, "base": record.base, "replacement": record.replacement}

    ancestry_valid: dict[str, bool] = {}
    visiting: set[str] = set()

    def fully_applicable(record: CorrectionRecord) -> bool:
        cached = ancestry_valid.get(record.correction_id)
        if cached is not None:
            return cached
        if record.correction_id in visiting:
            findings.append(_finding("correction_inapplicable", record, "supersedes_predecessor_cycle"))
            ancestry_valid[record.correction_id] = False
            return False
        closure_reason = predecessor_closure_reason(record)
        if closure_reason is not None:
            findings.append(_finding("correction_inapplicable", record, closure_reason))
            ancestry_valid[record.correction_id] = False
            return False
        if not own_applicable(record):
            ancestry_valid[record.correction_id] = False
            return False
        predecessor_id = record.supersedes_id
        if predecessor_id is None:
            ancestry_valid[record.correction_id] = True
            return True
        visiting.add(record.correction_id)
        predecessor = record_by_id[predecessor_id]
        valid = fully_applicable(predecessor)
        visiting.discard(record.correction_id)
        if not valid:
            findings.append(_finding(
                "correction_inapplicable", record, "supersedes_predecessor_inapplicable", related=(predecessor_id,),
            ))
            ancestry_valid[record.correction_id] = False
            return False
        ancestry_valid[record.correction_id] = True
        return True

    viable = [record for record in active_records.values() if fully_applicable(record)]

    # Only explicit supersession can displace a local correction.
    superseded = {record.supersedes_id for record in viable if record.supersedes_id is not None}
    active: list[CorrectionRecord] = []
    for record in viable:
        if record.correction_id in superseded:
            findings.append(_finding("correction_superseded", record, "explicit_supersession"))
        else:
            active.append(record)

    buckets: dict[tuple[str, str, str], list[CorrectionRecord]] = defaultdict(list)
    for record in active:
        if record.kind == "source_text":
            key = (record.kind, record.scope["segment_id"], "")
        elif record.kind == "target_occurrence":
            key = (record.kind, record.scope["occurrence_id"], record.scope["mapping_id"])
        elif record.kind == "target_segment":
            key = (record.kind, record.scope["segment_id"], "")
        else:
            # Equal precedence all/subset selectors conflict only when exact
            # selector/target scope is same; different subsets remain explicit.
            key = (record.kind, record.scope["concept_id"], canonical_json_bytes(record.scope["selector"]).decode("utf-8"))
        buckets[key].append(record)

    selected: list[CorrectionRecord] = []
    for grouped in buckets.values():
        if len(grouped) != 1:
            for record in grouped:
                findings.append(_finding("correction_conflict", record, "equal_precedence", related=[other.correction_id for other in grouped if other != record]))
        else:
            selected.extend(grouped)

    # Broad and subset terminology selectors may overlap.  Competing target
    # forms are equal-precedence conflicts, not implicit last-write-wins.
    terminology_candidates = [record for record in selected if record.kind == "terminology"]
    conflicted_terms: set[str] = set()
    for index, first in enumerate(terminology_candidates):
        first_selector = first.scope["selector"]
        first_ids = None if first_selector["kind"] == "all_concept_occurrences" else set(first_selector["ids"])
        for second in terminology_candidates[index + 1:]:
            if first.scope["concept_id"] != second.scope["concept_id"] or first.replacement == second.replacement:
                continue
            second_selector = second.scope["selector"]
            second_ids = None if second_selector["kind"] == "all_concept_occurrences" else set(second_selector["ids"])
            if first_ids is None or second_ids is None or first_ids & second_ids:
                conflicted_terms.update((first.correction_id, second.correction_id))
                findings.extend((
                    _finding("correction_conflict", first, "overlapping_terminology_selector", related=(second.correction_id,)),
                    _finding("correction_conflict", second, "overlapping_terminology_selector", related=(first.correction_id,)),
                ))
    selected = [record for record in selected if record.correction_id not in conflicted_terms]

    local = [record for record in selected if record.kind in {"target_occurrence", "target_segment"}]
    terminology = [record for record in selected if record.kind == "terminology"]
    if local and terminology:
        for record in local:
            findings.append(_finding("correction_protected", record, "local_target_overlay_protected_from_terminology", related=[item.correction_id for item in terminology]))

    def pick(kind: str) -> tuple[OverlayInput, ...]:
        return tuple(sorted((_overlay(record) for record in selected if record.kind == kind), key=lambda item: (item.subject_id, item.correction_id)))

    return OverlayResolution(
        base_revision_id, correction_set.set_id, source=pick("source_text"), terminology=pick("terminology"),
        target_occurrence=pick("target_occurrence"), target_segment=pick("target_segment"),
        findings=tuple(sorted(findings, key=lambda item: item.finding_id)),
        applicable_correction_ids=tuple(sorted(record.correction_id for record in selected)),
    )


def _closure_artifact_map(selected_closure: Any) -> dict[str, ArtifactEnvelope] | None:
    """Read records from an already verified compact selected closure."""
    if selected_closure is None:
        return None
    candidates: Any = selected_closure if isinstance(selected_closure, Mapping) else None
    if candidates is None:
        for name in ("artifact_map", "artifacts", "record_map", "records", "selected_artifacts",
                     "artifact_by_id", "records_by_id", "artifacts_by_id"):
            value = getattr(selected_closure, name, None)
            if isinstance(value, Mapping):
                candidates = value
                break
    if candidates is None:
        return None
    return {str(key): value for key, value in candidates.items() if isinstance(value, ArtifactEnvelope)}


def _archive_selected_artifacts(revisions: RevisionStore, revision_id: str) -> dict[str, ArtifactEnvelope]:
    """Read selected records from a legacy directory or verified v2 ZIP."""
    bundle = revisions._revision_path(revision_id)
    if bundle.is_dir():
        revisions.snapshot(revision_id)
        manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
        ids = manifest["artifact_ids"]
        if not isinstance(ids, list) or ids != sorted(set(ids)):
            raise CorrectionError("selected revision artifact closure is invalid")
        return {artifact_id: ArtifactEnvelope.from_file(bundle / "artifacts" / f"{artifact_id}.json") for artifact_id in ids}
    storage = getattr(revisions, "storage", None)
    if storage is None:
        raise CorrectionError("selected revision archive is invalid")
    values = storage.verify_revision(revision_id)
    selected: dict[str, ArtifactEnvelope] = {}
    for name, data in values.items():
        if name.startswith("records/"):
            envelope = ArtifactEnvelope.from_json(data.decode("utf-8"))
            selected[envelope.artifact_id] = envelope
    return selected


def _selected_closure_revision_id(selected_closure: Any) -> str:
    """Return the identity proved by a supplied closure.

    A closure is an authority boundary, not just a record map.  Require its
    revision identity before using any records or graph edges so a caller
    cannot pair the requested/base revision with another sealed authority.
    """
    revision_id = (selected_closure.get("revision_id")
                   if isinstance(selected_closure, Mapping)
                   else getattr(selected_closure, "revision_id", None))
    try:
        return _id(revision_id, "selected closure revision_id")
    except CorrectionError as exc:
        raise CorrectionError("selected invocation closure has no valid revision ID") from exc


def _verify_selected_closure_revision(selected_closure: Any, expected_revision_id: str) -> None:
    if selected_closure is None:
        return
    expected = _id(expected_revision_id, "expected selected revision_id")
    actual = _selected_closure_revision_id(selected_closure)
    if actual != expected:
        raise CorrectionError(
            "selected invocation closure revision does not match requested/base revision"
        )


def _sealed_selected_artifacts(
    revisions: RevisionStore, revision_id: str, *, selected_closure: Any = None,
) -> dict[str, ArtifactEnvelope]:
    """Return records from the one closure loaded for this invocation.

    A supplied closure is authoritative.  In particular, do not use the
    revision store as a fallback: that would perform a second selected-state
    traversal and could observe a different immutable revision.  The archive
    path remains only for callers using the pre-closure compatibility API.
    """
    _verify_selected_closure_revision(selected_closure, revision_id)
    selected = _closure_artifact_map(selected_closure)
    if selected_closure is not None:
        if selected is None:
            raise CorrectionError("selected invocation closure has no record map")
        return selected
    try:
        return _archive_selected_artifacts(revisions, revision_id)
    except (ArtifactError, OSError, KeyError, TypeError, SchemaError, json.JSONDecodeError, AttributeError) as exc:
        raise CorrectionError("selected base revision closure is invalid") from exc


def resolve_selected_overlays(
    store: CorrectionStore, revisions: RevisionStore, *, base_revision_id: str, correction_set_id: str | None,
    selected_closure: Any = None,
) -> OverlayResolution:
    """Load verified sealed base closure and resolve one explicitly selected set."""
    requested_revision_id = _id(base_revision_id, "base_revision_id")
    _verify_selected_closure_revision(selected_closure, requested_revision_id)
    if correction_set_id is None:
        return OverlayResolution(requested_revision_id, None)
    correction_set = store.get_set(correction_set_id)
    selected = _sealed_selected_artifacts(revisions, base_revision_id, selected_closure=selected_closure)
    # Load all retained immutable records, not only currently active IDs: a
    # successor legitimately points to historical predecessors.  Filename is
    # merely lookup address; each reader rechecks canonical content ID.
    records: list[CorrectionRecord] = []
    for path in sorted(store.records_dir.glob("*.json")):
        try:
            records.append(store.get_record(path.stem))
        except CorrectionError:
            continue

    invalid_events: list[str] = []
    activated_by_valid_event: set[str] = set()
    all_events: dict[str, CorrectionEvent] = {}
    for path in sorted(store.events_dir.glob("*.json")):
        try:
            event = store.get_event(path.stem)
        except CorrectionError:
            # A corrupt unrelated event is retained for inspection.  It only
            # invalidates resolution when required by selected/event ancestry.
            continue
        all_events[event.event_id] = event
        if event.event_kind in {"apply", "supersede"}:
            activated_by_valid_event.add(event.correction_id)

    # Event IDs may retain historical predecessors whose own successor set is
    # older.  Exactly one current transition must instead point to this set.
    # get_event already proves that transition's ID against its linked set; the
    # equality below is an explicit resolver boundary, never mere membership.
    current_events: list[CorrectionEvent] = []
    for event_id in correction_set.event_ids:
        event = all_events.get(event_id)
        if event is None:
            invalid_events.append(event_id)
            continue
        if event.correction_set_id == correction_set.set_id:
            other_event_ids = _other_event_ids(correction_set.event_ids, event.event_id)
            expected_event_ids = tuple(sorted((*other_event_ids, event.event_id)))
            if (
                expected_event_ids != correction_set.event_ids
                or correction_event_id_for(
                    event.correction_id, event.event_kind, event.successor_base_revision_id,
                    event.successor_active_correction_ids, other_event_ids,
                ) != event.event_id
                or event.successor_base_revision_id != correction_set.base_revision_id
                or event.successor_active_correction_ids != correction_set.active_correction_ids
            ):
                invalid_events.append(event_id)
            else:
                current_events.append(event)
    if not current_events:
        invalid_events.append(correction_set.set_id)
    if invalid_events:
        findings = [_set_finding(correction_set, "event_closure_incomplete", invalid_events)]
        findings.extend(_finding("correction_set_inapplicable", record, "event_closure_incomplete")
                        for record in records if record.correction_id in correction_set.active_correction_ids)
        resolution = OverlayResolution(base_revision_id, correction_set.set_id, findings=tuple(sorted(findings, key=lambda item: item.finding_id)))
    else:
        resolution = resolve_correction_set(
            correction_set, records, base_revision_id=base_revision_id, selected_artifacts=selected,
            event_activated_correction_ids=frozenset(activated_by_valid_event),
        )
    store.persist_findings(resolution.findings)
    return resolution


# Short names make stage-boundary inputs explicit without exposing materialization.
source_overlay_inputs = lambda resolution: resolution.source_inputs
terminology_overlay_inputs = lambda resolution: resolution.terminology_inputs
target_overlay_inputs = lambda resolution: resolution.target_inputs


# Task 9 is deliberately a planner, not an executor.  These constants and
# helpers make that boundary explicit and keep plan identity independent of a
# later candidate snapshot or any mutable cache history.
PROJECTION_ALGORITHM_VERSION = "projection-plan-v1"
_DIRECT_STAGE = "correction_direct_overlay"
_REVERSE_STAGE = "correction_reverse_descendant"
_ARTIFACT_STAGE = "selected_artifact"


def _record_payload(record: CorrectionRecord) -> dict[str, Any]:
    return {
        "kind": record.kind, "applies_to_revision_id": record.applies_to_revision_id,
        "scope": record.scope, "base": record.base, "replacement": record.replacement,
    }


def _all_records(store: CorrectionStore) -> list[CorrectionRecord]:
    records: list[CorrectionRecord] = []
    for path in sorted(store.records_dir.glob("*.json")):
        try:
            records.append(store.get_record(path.stem))
        except CorrectionError:
            # Corrupt unrelated immutable candidates never become command input.
            continue
    return records


def _event_activated_ids(store: CorrectionStore) -> frozenset[str]:
    result: set[str] = set()
    for path in sorted(store.events_dir.glob("*.json")):
        try:
            event = store.get_event(path.stem)
        except CorrectionError:
            continue
        if event.event_kind in {"apply", "supersede"}:
            result.add(event.correction_id)
    return frozenset(result)


def _reject_resolution(resolution: OverlayResolution, expected_active_ids: Sequence[str]) -> None:
    """Commands fail closed; diagnostic-only resolver behavior is for runs."""
    expected = tuple(sorted(expected_active_ids))
    if resolution.applicable_correction_ids != expected:
        reasons = sorted({str(item.evidence.get("reason", item.kind)) for item in resolution.findings})
        detail = ",".join(reasons) if reasons else "inapplicable"
        raise CorrectionError(f"correction successor is {detail}")
    rejected = {
        "correction_stale", "correction_inapplicable", "correction_set_inapplicable",
        "correction_conflict", "correction_mapping_ambiguous", "correction_superseded",
    }
    if any(finding.kind in rejected for finding in resolution.findings):
        raise CorrectionError("correction successor contains stale, inapplicable, conflicting, ambiguous, or superseded record")


def _validate_current_set(
    store: CorrectionStore, revisions: RevisionStore, correction_set: CorrectionSet, *, base_revision_id: str,
    selected_closure: Any = None,
) -> tuple[list[CorrectionRecord], dict[str, ArtifactEnvelope]]:
    """Verify a pointer's event and record closure without persisting findings."""
    if correction_set.base_revision_id != base_revision_id:
        raise CorrectionError("active correction set base revision does not match active revision")
    selected = _sealed_selected_artifacts(revisions, base_revision_id, selected_closure=selected_closure)
    current_events: list[CorrectionEvent] = []
    for event_id in correction_set.event_ids:
        try:
            event = store.get_event(event_id)
        except CorrectionError as exc:
            raise CorrectionError("active correction set event closure is invalid") from exc
        if event.correction_set_id == correction_set.set_id:
            current_events.append(event)
    # Every command transition adds exactly one new event.  Accepting a set
    # without that event would turn bare record membership into authority.
    if len(current_events) != 1:
        raise CorrectionError("active correction set has no unique current transition event")
    records = _all_records(store)
    resolution = resolve_correction_set(
        correction_set, records, base_revision_id=base_revision_id, selected_artifacts=selected,
        event_activated_correction_ids=_event_activated_ids(store),
    )
    _reject_resolution(resolution, correction_set.active_correction_ids)
    return records, selected


def _entry(stage: str, subject_id: str, artifact_id: str) -> dict[str, str]:
    return {"stage": stage, "subject_id": subject_id, "base_artifact_id": artifact_id}


def _entry_key(value: Mapping[str, str]) -> tuple[str, str, str]:
    return value["stage"], value["subject_id"], value["base_artifact_id"]


def _ordered_entries(values: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    unique = {_entry_key(value): _entry(*_entry_key(value)) for value in values}
    return tuple(unique[key] for key in sorted(unique))


def projection_plan_id_for(
    *, base_revision_id: str, active_correction_set: CorrectionSet,
    correction: CorrectionRecord, projected_universe: Sequence[Mapping[str, str]],
) -> str:
    """Hash every and only Task-9 plan input; never a projected snapshot ID."""
    if not isinstance(active_correction_set, CorrectionSet) or not isinstance(correction, CorrectionRecord):
        raise CorrectionError("projection plan requires correction set and correction record")
    universe = _ordered_entries(projected_universe)
    return tagged_sha256(
        PROJECTION_ALGORITHM_VERSION,
        canonical_json_bytes({
            "projection_algorithm_version": PROJECTION_ALGORITHM_VERSION,
            "base_revision_id": _id(base_revision_id, "base_revision_id"),
            # Include both address and canonical content.  A caller cannot make
            # a same-ID/different-body set because CorrectionSet validates it,
            # but retaining both makes the planned authority self-describing.
            "active_correction_event_set_id": active_correction_set.set_id,
            "active_correction_event_set": active_correction_set.to_dict(),
            "correction_payload": _record_payload(correction),
            "ordered_universe": list(universe),
        }),
    )


def _correction_base_artifact_ids(record: CorrectionRecord) -> tuple[str, ...]:
    return tuple(sorted(item["artifact_id"] for item in _base_refs(record)))


class _ClosureGraph:
    """Small traversal view over immutable edge values carried by a closure."""

    def __init__(self, edges: Sequence[Any]):
        self._edges = tuple(edges)

    def edges(self, revision_id: str) -> tuple[Any, ...]:
        return self._edges

    def descendants(self, revision_id: str, artifact_id: str) -> tuple[str, ...]:
        result: set[str] = set()
        queue = [artifact_id]
        while queue:
            current = queue.pop(0)
            for edge in self._edges:
                if getattr(edge, "parent_artifact_id", None) == current:
                    child = edge.child_artifact_id
                    if child not in result:
                        result.add(child)
                        queue.append(child)
        return tuple(sorted(result))


def _closure_graph(selected_closure: Any, revision_id: str) -> Any:
    """Return a graph/traversal view carried by the invocation closure only."""
    if selected_closure is None:
        return None
    owners = [selected_closure]
    if isinstance(selected_closure, Mapping):
        owners.extend(selected_closure.get(name) for name in ("graph", "selected_graph", "dependency_graph"))
    else:
        owners.extend(getattr(selected_closure, name, None)
                      for name in ("graph", "selected_graph", "dependency_graph"))
    for owner in owners:
        if owner is None:
            continue
        if all(callable(getattr(owner, method, None)) for method in ("edges", "descendants")):
            return owner
        edges = owner.get("edges") if isinstance(owner, Mapping) else getattr(owner, "edges", None)
        if isinstance(edges, Mapping):
            return _ClosureGraph(tuple(edges.values()))
    # A record-only closure is valid for overlay resolution, but it is not a
    # valid planning closure.  Never silently broaden it through selected_graph.
    return None


def _graph_entries_and_descendants(
    revisions: RevisionStore, revision_id: str, selected: Mapping[str, ArtifactEnvelope], record: CorrectionRecord,
    *, selected_closure: Any = None,
) -> tuple[tuple[dict[str, str], ...], set[str], set[str], dict[str, tuple[dict[str, str], ...]]]:
    """Build fixed base/virtual universe and exact persisted reverse closure."""
    graph = _closure_graph(selected_closure, revision_id)
    if graph is None:
        if selected_closure is not None:
            raise CorrectionError("selected invocation closure has no dependency graph")
        graph = revisions.selected_graph(revision_id)
    try:
        edges = graph.edges(revision_id)
    except ArtifactError as exc:
        raise CorrectionError("selected base graph is invalid") from exc
    values: list[dict[str, str]] = []
    represented: set[str] = set()
    artifact_entries: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in edges:
        # An edge's stage/subject describes its produced child.  Parent nodes
        # appear as children of their own producing edge or receive a neutral
        # selected-artifact node below.
        item = _entry(edge.stage, edge.stable_subject_id, edge.child_artifact_id)
        values.append(item)
        represented.add(edge.child_artifact_id)
        artifact_entries[edge.child_artifact_id].append(item)
    for artifact_id in sorted(selected):
        if artifact_id not in represented:
            item = _entry(_ARTIFACT_STAGE, artifact_id, artifact_id)
            values.append(item)
            artifact_entries[artifact_id].append(item)

    direct = set(_correction_base_artifact_ids(record))
    descendants: set[str] = set()
    for artifact_id in direct:
        try:
            descendants.update(graph.descendants(revision_id, artifact_id))
        except ArtifactError as exc:
            raise CorrectionError("selected base graph traversal failed") from exc
        direct_node = _entry(_DIRECT_STAGE, record.correction_id, artifact_id)
        values.append(direct_node)
        artifact_entries[artifact_id].append(direct_node)

    # A verified occurrence subset is narrower than the concept projection
    # closure.  Projection edges fan out to every occurrence of the concept;
    # only the selected occurrence's segment and its immediate context
    # consumers may be regenerated.  The graph's stable subject identifies
    # those segment-scoped edges, so retain their child closures and discard
    # unrelated same-concept descendants.
    if record.kind == "terminology" and record.scope["selector"]["kind"] == "occurrence_ids":
        selected_occurrences = set(record.scope["selector"]["ids"])
        selected_segments = {
            occurrence["segment_id"]
            for artifact in selected.values()
            if artifact.kind == "OccurrenceEvidenceShard"
            for occurrence in artifact.payload.get("occurrences", ())
            if isinstance(occurrence, Mapping)
            and occurrence.get("occurrence_id") in selected_occurrences
            and isinstance(occurrence.get("segment_id"), str)
        }
        scoped_roots = {
            edge.child_artifact_id
            for edge in edges
            if edge.stable_subject_id in selected_segments
        }
        scoped_descendants = set(scoped_roots)
        for artifact_id in tuple(sorted(scoped_roots)):
            try:
                scoped_descendants.update(graph.descendants(revision_id, artifact_id))
            except ArtifactError as exc:
                raise CorrectionError("selected base graph traversal failed") from exc
        descendants = scoped_descendants
    for artifact_id in sorted(descendants):
        virtual = _entry(_REVERSE_STAGE, record.correction_id, artifact_id)
        values.append(virtual)
        artifact_entries[artifact_id].append(virtual)
    return _ordered_entries(values), direct, descendants, {
        key: _ordered_entries(value) for key, value in artifact_entries.items()
    }


def _ambiguous_artifacts(edges: Sequence[Any], record: CorrectionRecord, descendants: set[str]) -> set[str]:
    """Only declared graph ambiguity can taint a projection; never infer work."""
    ambiguous: set[str] = set()
    for edge in edges:
        marker = f"{edge.stage}:{edge.edge_kind}".lower()
        if any(word in marker for word in ("ambiguous", "conflict", "identity")):
            ambiguous.add(edge.child_artifact_id)
    # A source edit without any persisted old/new membership relation cannot
    # honestly project terminology membership.  Mark only its known direct
    # closure ambiguous; no synthetic descendants are invented.
    if record.kind == "source_text":
        membership_seen = any(
            "membership" in f"{edge.stage}:{edge.edge_kind}".lower()
            and edge.child_artifact_id in descendants
            for edge in edges
        )
        if not membership_seen:
            ambiguous.update(descendants | set(_correction_base_artifact_ids(record)))
    return ambiguous


def impact_for_correction(*args: Any, **kwargs: Any) -> CorrectionImpact:
    """Return exact sealed-graph impact, with compatibility-friendly arguments.

    Preferred spelling is ``impact_for_correction(store, revisions, record,
    correction_set=successor_set)``.  The named spelling with ``store=``,
    ``revisions=``, and ``correction=`` is equivalent.  This function reads
    only immutable correction/revision state and does not run a stage.
    """
    store = kwargs.pop("store", None)
    revisions = kwargs.pop("revisions", None)
    correction = kwargs.pop("correction", None)
    correction_set = kwargs.pop("correction_set", None)
    selected_closure = kwargs.pop("selected_closure", None)
    if kwargs:
        raise TypeError(f"unexpected impact_for_correction arguments: {sorted(kwargs)}")
    remaining = list(args)
    if remaining and isinstance(remaining[0], CorrectionStore):
        store = remaining.pop(0)
    if remaining and isinstance(remaining[0], RevisionStore):
        revisions = remaining.pop(0)
    if remaining and isinstance(remaining[0], CorrectionRecord):
        correction = remaining.pop(0)
    if remaining:
        if correction_set is None and isinstance(remaining[0], CorrectionSet) and len(remaining) == 1:
            correction_set = remaining.pop(0)
        else:
            raise TypeError("invalid impact_for_correction positional arguments")
    if not isinstance(store, CorrectionStore) or not isinstance(revisions, RevisionStore):
        raise CorrectionError("impact requires CorrectionStore and RevisionStore")
    if isinstance(correction, Mapping):
        correction = correction_record_for(correction)
    if not isinstance(correction, CorrectionRecord):
        raise CorrectionError("impact requires a correction record")
    if correction_set is None:
        correction_set = CorrectionSet.create(correction.applies_to_revision_id, (correction.correction_id,))
    if isinstance(correction_set, str):
        correction_set = store.get_set(correction_set)
    if not isinstance(correction_set, CorrectionSet):
        raise CorrectionError("impact requires a correction set")
    if correction_set.base_revision_id != correction.applies_to_revision_id:
        raise CorrectionError("correction and impact set base revisions differ")

    selected = _sealed_selected_artifacts(
        revisions, correction.applies_to_revision_id, selected_closure=selected_closure,
    )
    for artifact_id in _correction_base_artifact_ids(correction):
        if artifact_id not in selected:
            raise CorrectionError("correction base artifact is not selected in declared revision")
    universe, direct, descendants, by_artifact = _graph_entries_and_descendants(
        revisions, correction.applies_to_revision_id, selected, correction,
        selected_closure=selected_closure,
    )
    graph = _closure_graph(selected_closure, correction.applies_to_revision_id)
    if graph is None:
        if selected_closure is not None:
            raise CorrectionError("selected invocation closure has no dependency graph")
        graph = revisions.selected_graph(correction.applies_to_revision_id)
    edges = graph.edges(correction.applies_to_revision_id)
    ambiguous_artifacts = _ambiguous_artifacts(edges, correction, descendants)

    protected_artifacts: set[str] = set()
    if correction.kind == "terminology":
        # Local target overlays are intentionally stronger than broad concept
        # overlays.  Their exact selected graph closure is protected, not
        # silently included in broad terminology regeneration.
        for active_id in correction_set.active_correction_ids:
            if active_id == correction.correction_id:
                continue
            try:
                active = store.get_record(active_id)
            except CorrectionError:
                continue
            if active.kind not in {"target_occurrence", "target_segment"}:
                continue
            for artifact_id in _correction_base_artifact_ids(active):
                protected_artifacts.add(artifact_id)
                protected_artifacts.update(graph.descendants(
                    correction.applies_to_revision_id, artifact_id,
                ))

    affected_artifacts = direct | descendants
    partition: dict[str, list[dict[str, str]]] = {name: [] for name in ("affected", "unaffected", "ambiguous", "protected")}
    for entry in universe:
        artifact_id = entry["base_artifact_id"]
        if artifact_id in ambiguous_artifacts:
            category = "ambiguous"
        elif correction.kind == "terminology" and artifact_id in protected_artifacts:
            category = "protected"
        elif artifact_id in affected_artifacts:
            category = "affected"
        else:
            category = "unaffected"
        partition[category].append(entry)

    # Reuse is an overlapping reporting dimension, not another partition.  A
    # selected valid artifact can be reused only when its projected node was
    # not marked affected/ambiguous.
    reusable_artifacts = set(selected)
    reused = [
        entry for category in ("unaffected", "protected") for entry in partition[category]
        if entry["base_artifact_id"] in reusable_artifacts
    ]
    plan_id = projection_plan_id_for(
        base_revision_id=correction.applies_to_revision_id, active_correction_set=correction_set,
        correction=correction, projected_universe=universe,
    )
    return CorrectionImpact(
        phase="correction", base_revision_id=correction.applies_to_revision_id, projection_plan_id=plan_id,
        correction_id=correction.correction_id, correction_set_id=correction_set.set_id,
        projected_universe=universe, affected=_ordered_entries(partition["affected"]),
        unaffected=_ordered_entries(partition["unaffected"]), ambiguous=_ordered_entries(partition["ambiguous"]),
        protected=_ordered_entries(partition["protected"]), reused=_ordered_entries(reused), regenerated=(),
    )


def correction_impact_finding(
    impact: CorrectionImpact, correction: CorrectionRecord | None = None,
) -> Finding:
    """Describe the planner's exact selected-state dirty set for audit."""
    if not isinstance(impact, CorrectionImpact):
        raise CorrectionError("correction impact finding requires a CorrectionImpact")
    prior_ids = tuple(sorted({item["base_artifact_id"] for item in impact.projected_universe}))
    invalidated_ids = tuple(sorted({
        item["base_artifact_id"]
        for item in (*impact.affected, *impact.ambiguous)
    }))
    scope_ids: set[str] = set()
    occurrence_ids: tuple[str, ...] = ()
    if correction is not None:
        if correction.correction_id != impact.correction_id:
            raise CorrectionError("correction impact finding correction ID differs from plan")
        scope = correction.scope
        scope_ids.update(
            value for name in ("segment_id", "mapping_id", "concept_id")
            if isinstance((value := scope.get(name)), str)
        )
        direct_occurrence_ids = ((scope["occurrence_id"],)
                                  if isinstance(scope.get("occurrence_id"), str) else ())
        selector = scope.get("selector")
        selector_occurrence_ids = (
            tuple(item for item in selector.get("ids", ()) if isinstance(item, str))
            if isinstance(selector, Mapping) and selector.get("kind") == "occurrence_ids"
            else ()
        )
        occurrence_ids = tuple(sorted(set((*direct_occurrence_ids, *selector_occurrence_ids))))
        scope_ids.update(occurrence_ids)
    subject_refs = tuple(sorted({
        value for value in (impact.correction_id, impact.correction_set_id, *scope_ids, *invalidated_ids)
        if isinstance(value, str) and value
    }))
    return Finding(
        kind="correction_impact", severity="warning", stage="corrections",
        audit_category="correction_impact", subject_refs=subject_refs,
        dependency_ids=invalidated_ids or prior_ids,
        evidence={
            "trigger": "correction_dirty_set_planned",
            "base_revision_id": impact.base_revision_id,
            "projection_plan_id": impact.projection_plan_id,
            "correction_id": impact.correction_id,
            "correction_set_id": impact.correction_set_id,
            "occurrence_ids": list(occurrence_ids),
            "prior_selected_ids": list(prior_ids),
            "invalidated_selected_ids": list(invalidated_ids),
            "dirty_set": [dict(item) for item in (*impact.affected, *impact.ambiguous)],
        },
        message=(f"Correction {impact.correction_id} invalidates "
                 f"{len(invalidated_ids)} selected record(s) in its dirty set."),
    )


def _pointer_set_id(root: Path) -> str | None:
    path = root / "active-correction-set.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectionError("invalid active correction set pointer") from exc
    if set(value) != {"set_id"} or not isinstance(value["set_id"], str) or not value["set_id"]:
        raise CorrectionError("invalid active correction set pointer")
    return value["set_id"]


def _publish_active_set(root: Path, correction_set: CorrectionSet) -> None:
    """Single atomic publication point after all immutable successor state."""
    _atomic_bytes(root / "active-correction-set.json", canonical_json_bytes({"set_id": correction_set.set_id}))


def _active_revision_id(revisions: RevisionStore) -> str:
    snapshot = revisions.active_snapshot()
    if snapshot is None:
        raise CorrectionError("correction command requires an active revision")
    return snapshot.revision_id


def _invocation_revision_id(selected_closure: Any) -> str:
    """Read the already validated closure identity without touching storage."""
    revision_id = (selected_closure.get("revision_id")
                   if isinstance(selected_closure, Mapping)
                   else getattr(selected_closure, "revision_id", None))
    if not isinstance(revision_id, str) or not revision_id:
        raise CorrectionError("selected invocation closure has no revision ID")
    return _id(revision_id, "selected closure revision_id")


def correction_transition(
    store: CorrectionStore, revisions: RevisionStore, *, event_kind: str,
    payload: Mapping[str, Any] | None = None, correction_id: str | None = None,
    supersedes_id: str | None = None, revision_id: str | None = None,
    selected_closure: Any = None,
) -> tuple[CorrectionSet, CorrectionImpact]:
    """Validate then atomically publish one apply/revert/supersede successor.

    No persistent record/event/set/impact is written until every named base,
    selected closure, current event closure, and successor overlay is proven.
    """
    if event_kind not in {"apply", "revert", "supersede"}:
        raise CorrectionError("correction transition kind is invalid")
    active_revision_id = (
        _invocation_revision_id(selected_closure)
        if selected_closure is not None else _active_revision_id(revisions)
    )
    if revision_id is not None and revision_id != active_revision_id:
        raise CorrectionError("requested correction revision does not match active revision")
    pointer_id = _pointer_set_id(store.root)
    if pointer_id is None:
        current = CorrectionSet.create(active_revision_id)
        records: list[CorrectionRecord] = _all_records(store)
        selected = _sealed_selected_artifacts(revisions, active_revision_id, selected_closure=selected_closure)
    else:
        current = store.get_set(pointer_id)
        records, selected = _validate_current_set(
            store, revisions, current, base_revision_id=active_revision_id,
            selected_closure=selected_closure,
        )
    if current.base_revision_id != active_revision_id:
        raise CorrectionError("active correction set base revision does not match active revision")
    active_ids = set(current.active_correction_ids)

    if event_kind == "apply":
        if payload is None or correction_id is not None or supersedes_id is not None:
            raise CorrectionError("apply requires only a correction payload")
        record = correction_record_for(payload)
        if record.applies_to_revision_id != active_revision_id:
            raise CorrectionError("correction payload revision does not match active revision")
        if record.correction_id in active_ids:
            raise CorrectionError("correction is already active")
        successor_active = tuple(sorted((*active_ids, record.correction_id)))
    elif event_kind == "revert":
        if payload is not None or supersedes_id is not None or correction_id is None:
            raise CorrectionError("revert requires one active correction ID")
        record = next((item for item in records if item.correction_id == correction_id), None)
        if record is None or correction_id not in active_ids:
            raise CorrectionError("unknown or inactive correction")
        if revision_id != current.base_revision_id:
            raise CorrectionError("revert revision does not match active correction set")
        successor_active = tuple(sorted(active_ids - {correction_id}))
    else:
        if payload is None or correction_id is not None or supersedes_id is None:
            raise CorrectionError("supersede requires payload and active predecessor ID")
        predecessor = next((item for item in records if item.correction_id == supersedes_id), None)
        if predecessor is None or supersedes_id not in active_ids:
            raise CorrectionError("unknown or inactive superseded correction")
        record = correction_record_for(payload, supersedes_id=supersedes_id)
        if record.applies_to_revision_id != active_revision_id:
            raise CorrectionError("correction payload revision does not match active revision")
        if revision_id is not None and revision_id != current.base_revision_id:
            raise CorrectionError("supersede revision does not match active correction set")
        successor_active = tuple(sorted((active_ids - {supersedes_id}) | {record.correction_id}))

    all_records = [item for item in records if item.correction_id != record.correction_id] + [record]
    event_id = correction_event_id_for(
        record.correction_id, event_kind, active_revision_id, successor_active, current.event_ids,
    )
    successor = CorrectionSet.create(active_revision_id, successor_active, (*current.event_ids, event_id))
    # Validate exact bases, relations, stale/conflict/ambiguity/supersession
    # before any immutable successor file is made observable.
    resolution = resolve_correction_set(
        successor, all_records, base_revision_id=active_revision_id, selected_artifacts=selected,
        event_activated_correction_ids=None,
    )
    _reject_resolution(resolution, successor.active_correction_ids)
    impact = impact_for_correction(
        store, revisions, record, correction_set=successor,
        selected_closure=selected_closure,
    )
    event = correction_event_for(record.correction_id, event_kind, successor)

    # Immutable prerequisites first; pointer swap is sole mutable publication.
    if event_kind in {"apply", "supersede"}:
        store.put_record(record)
    store.put_set(successor)
    store.put_event(event)
    # Keep correction-time dirty-set evidence inspectable and auditable next
    # to the immutable projection plan.  This is not a stage result.
    store.persist_findings((correction_impact_finding(impact, record),))
    store.put_impact(impact)
    _publish_active_set(store.root, successor)
    return successor, impact
