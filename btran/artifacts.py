"""Immutable content-addressed artifacts, revisions, cache validation, and graphs.

This module deliberately has no stage executor.  It only persists and verifies
closed immutable inputs selected by an explicit revision snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from btran.schema import (
    SCHEMA_VERSION,
    ArtifactEnvelope,
    DependencyGraphEdge,
    Finding,
    RevisionSnapshot,
    SchemaError,
    canonical_json_bytes,
    tagged_sha256,
)


class ArtifactError(ValueError):
    """An immutable artifact, revision, or graph operation is invalid."""


class CacheMiss(ArtifactError):
    """Requested cache entry cannot safely be reused."""


# --- Canonical IDs and semantic keys -------------------------------------------------


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _bytes(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise ArtifactError(f"{name} must be bytes")
    return value


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ArtifactError(f"{name} must be a sequence")
    result = tuple(sorted(set(_text(value, name) for value in values)))
    if len(result) != len(values):
        raise ArtifactError(f"{name} must be unique")
    return result


def artifact_id_for(kind: str, payload: Mapping[str, Any], dependency_ids: Sequence[str] = ()) -> str:
    """Canonical artifact ID; semantic key and findings do not affect identity."""
    dependencies = _ids(dependency_ids, "dependency_ids")
    if not isinstance(payload, Mapping):
        raise ArtifactError("payload must be an object")
    return tagged_sha256(
        "artifact-v1",
        canonical_json_bytes({
            "schema_version": SCHEMA_VERSION,
            "kind": _text(kind, "kind"),
            "payload": dict(payload),
            "dependency_ids": list(dependencies),
        }),
    )


def dependency_edge_id_for(
    stable_subject_id: str, parent_artifact_id: str, child_artifact_id: str, stage: str, edge_kind: str,
) -> str:
    return tagged_sha256("dependency-edge-v1", canonical_json_bytes({
        "schema_version": SCHEMA_VERSION,
        "stable_subject_id": _text(stable_subject_id, "stable_subject_id"),
        "parent_artifact_id": _text(parent_artifact_id, "parent_artifact_id"),
        "child_artifact_id": _text(child_artifact_id, "child_artifact_id"),
        "stage": _text(stage, "stage"),
        "edge_kind": _text(edge_kind, "edge_kind"),
    }))


def _semantic(tag: str, body: Mapping[str, Any], *raw_parts: bytes) -> str:
    return tagged_sha256(tag, canonical_json_bytes(dict(body)), *raw_parts)


def source_extraction_semantic_key(
    *, extraction_schema: str, prompt_bytes: bytes, model_executable_identity: str,
    model_id: str, raw_bytes: bytes, reasoning_level: str = "low",
) -> str:
    """Key exact bytes supplied to bounded source-model invocation."""
    return _semantic("source-extraction-v2", {
        "extraction_schema": _text(extraction_schema, "extraction_schema"),
        "model_executable_identity": _text(model_executable_identity, "model_executable_identity"),
        "model_id": _text(model_id, "model_id"),
        "reasoning_level": _text(reasoning_level, "reasoning_level"),
    }, _bytes(prompt_bytes, "prompt_bytes"), _bytes(raw_bytes, "raw_bytes"))


def occurrence_shard_semantic_key(
    *, effective_source_segment: Any, evidence_rules_version: str, evidence_candidates: Any = None,
) -> str:
    """Key an evidence shard on source plus exact candidate payload.

    Candidate evidence is an output input, not cache transport metadata.  The
    caller supplies its verified/canonical candidate representation; `None`
    remains an explicit local-extraction input for compatibility.
    """
    return _semantic("evidence-v1", {
        "effective_source_segment": effective_source_segment,
        "evidence_rules_version": _text(evidence_rules_version, "evidence_rules_version"),
        "evidence_candidates": evidence_candidates,
    })


def concept_membership_semantic_key(
    *, concept_id: str, occurrence_ids: Sequence[str], evidence_shard_ids: Sequence[str], membership_rules_version: str,
) -> str:
    return _semantic("membership-v1", {
        "concept_id": _text(concept_id, "concept_id"),
        "occurrence_ids": list(_ids(occurrence_ids, "occurrence_ids")),
        "evidence_shard_ids": list(_ids(evidence_shard_ids, "evidence_shard_ids")),
        "membership_rules_version": _text(membership_rules_version, "membership_rules_version"),
    })


def projection_semantic_key(
    *, concept_id: str, occurrence_scope_selector: Mapping[str, Any], membership_id: str, target_form: str,
    active_terminology_corrections: Sequence[str], model_executable_identity: str, model_id: str,
    prompt_bytes: bytes, consolidation_schema: str, algorithm_version: str, reasoning_level: str = "low",
) -> str:
    return _semantic("projection-v1", {
        "concept_id": _text(concept_id, "concept_id"),
        "occurrence_scope_selector": dict(occurrence_scope_selector),
        "membership_id": _text(membership_id, "membership_id"),
        "target_form": target_form,
        "active_terminology_corrections": list(_ids(active_terminology_corrections, "active_terminology_corrections")),
        "model_executable_identity": _text(model_executable_identity, "model_executable_identity"),
        "model_id": _text(model_id, "model_id"),
        "reasoning_level": _text(reasoning_level, "reasoning_level"),
        "consolidation_schema": _text(consolidation_schema, "consolidation_schema"),
        "algorithm_version": _text(algorithm_version, "algorithm_version"),
    }, _bytes(prompt_bytes, "prompt_bytes"))


def translation_semantic_key(
    *, source_artifact_id: str, preceding_source_artifact_id: str | None,
    following_source_artifact_id: str | None, projection_ids: Sequence[str], model_executable_identity: str,
    model_id: str, prompt_bytes: bytes, target_lang: str, reasoning_level: str = "low",
) -> str:
    return _semantic("translation-v1", {
        "source_artifact_id": _text(source_artifact_id, "source_artifact_id"),
        "preceding_source_artifact_id": preceding_source_artifact_id,
        "following_source_artifact_id": following_source_artifact_id,
        "projection_ids": list(_ids(projection_ids, "projection_ids")),
        "model_executable_identity": _text(model_executable_identity, "model_executable_identity"),
        "model_id": _text(model_id, "model_id"),
        "reasoning_level": _text(reasoning_level, "reasoning_level"),
        "target_lang": _text(target_lang, "target_lang"),
    }, _bytes(prompt_bytes, "prompt_bytes"))


def reconciliation_semantic_key(*, effective_page_ids: Sequence[str], effective_page_text_hashes: Sequence[str], projection_ids: Sequence[str]) -> str:
    if len(effective_page_ids) != len(effective_page_text_hashes):
        raise ArtifactError("effective page IDs and text hashes must have equal ordered length")
    return _semantic("reconciliation-v1", {
        "effective_page_ids": [_text(value, "effective_page_id") for value in effective_page_ids],
        "effective_page_text_hashes": [_text(value, "effective_page_text_hash") for value in effective_page_text_hashes],
        "projection_ids": list(_ids(projection_ids, "projection_ids")),
    })


def validation_semantic_key(
    *, rules_version: str, mode: str, target_lang: str | None, effective_page_ids: Sequence[str],
    reconciliation_artifact_id: str, projection_ids: Sequence[str],
) -> str:
    if mode not in {"native", "translated"}:
        raise ArtifactError("mode must be native or translated")
    return _semantic("validation-v1", {
        "rules_version": _text(rules_version, "rules_version"), "mode": mode, "target_lang": target_lang,
        "effective_page_ids": [_text(value, "effective_page_id") for value in effective_page_ids],
        "reconciliation_artifact_id": _text(reconciliation_artifact_id, "reconciliation_artifact_id"),
        "projection_ids": list(_ids(projection_ids, "projection_ids")),
    })


def render_input_semantic_key(
    *, renderer_version: str, template_version: str, mode: str, language: str | None,
    ordered_page_display_metadata: Sequence[Mapping[str, Any]], effective_page_ids: Sequence[str],
    validation_artifact_id: str, correction_ids: Sequence[str], finding_ids: Sequence[str],
) -> str:
    return _semantic("render-input-v1", {
        "renderer_version": _text(renderer_version, "renderer_version"),
        "template_version": _text(template_version, "template_version"), "mode": mode, "language": language,
        "ordered_page_display_metadata": [dict(item) for item in ordered_page_display_metadata],
        "effective_page_ids": [_text(value, "effective_page_id") for value in effective_page_ids],
        "validation_artifact_id": _text(validation_artifact_id, "validation_artifact_id"),
        "correction_ids": list(_ids(correction_ids, "correction_ids")),
        "finding_ids": list(_ids(finding_ids, "finding_ids")),
    })


def finding_semantic_key(finding: Finding | Mapping[str, Any]) -> str:
    """Key only finding inputs, never transport/execution metadata.

    A mapping is intentionally not treated as an unversioned convenience form:
    its schema version is a semantic input and must be named explicitly.
    """
    if isinstance(finding, Finding):
        body = finding.to_dict()
    elif isinstance(finding, Mapping):
        body = dict(finding)
        if "schema_version" not in body:
            raise ArtifactError("finding mapping must include schema_version")
    else:
        raise ArtifactError("finding must be a Finding or mapping")
    try:
        schema_version = _text(body["schema_version"], "schema_version")
        stage = _text(body["stage"], "stage")
        kind = _text(body["kind"], "kind")
        severity = _text(body["severity"], "severity")
        subject_refs = _ids(body["subject_refs"], "subject_refs")
    except (KeyError, TypeError) as exc:
        raise ArtifactError("finding mapping lacks required semantic fields") from exc
    evidence = body.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ArtifactError("finding evidence must be an object")
    return _semantic("finding-v1", {
        "schema_version": schema_version,
        "stage": stage,
        "kind": kind,
        "severity": severity,
        "subject_refs": list(subject_refs),
        "evidence": dict(evidence),
    })


def correction_semantic_key(
    *, kind: str, revision_id: str, subjects: Sequence[str], base_hashes: Mapping[str, str],
    replacement: str, explicit_scope: Mapping[str, Any], supersedes_id: str | None,
) -> str:
    return _semantic("correction-v1", {
        "kind": _text(kind, "kind"), "revision_id": _text(revision_id, "revision_id"),
        "subjects": list(_ids(subjects, "subjects")), "base_hashes": dict(base_hashes),
        "replacement": replacement, "explicit_scope": dict(explicit_scope), "supersedes_id": supersedes_id,
    })

# Verb-first aliases make table-to-code correspondence unambiguous.
semantic_key_source_extraction = source_extraction_semantic_key
semantic_key_occurrence_shard = occurrence_shard_semantic_key
semantic_key_concept_membership = concept_membership_semantic_key
semantic_key_projection = projection_semantic_key
semantic_key_translation = translation_semantic_key
semantic_key_reconciliation = reconciliation_semantic_key
semantic_key_validation = validation_semantic_key
semantic_key_render_input = render_input_semantic_key
semantic_key_finding = finding_semantic_key
semantic_key_correction = correction_semantic_key


# --- Durable atomic files -------------------------------------------------------------


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(value))


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid persisted JSON: {path}") from exc
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value):
        raise ArtifactError(f"non-canonical persisted JSON: {path}")
    return value


class LegacyArtifactStore:
    """Filesystem content-addressed artifact/finding cache.

    Artifact files are immutable.  Index files are only same-key discovery sets;
    no API ever chooses an entry based on their ordering.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.findings_dir = self.root / "findings"
        self.index_dir = self.root / "index"
        # A canonical artifact ID deliberately excludes its semantic cache key.
        # Keep key bindings in immutable sidecars, rather than rewriting the
        # first artifact envelope when identical output is re-attested by a
        # changed model/prompt.
        self.attestations_dir = self.root / "attestations"
        self.quarantine_dir = self.root / "quarantine"
        for directory in (self.artifacts_dir, self.findings_dir, self.index_dir, self.attestations_dir, self.quarantine_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.artifacts_dir / f"{artifact_id}.json"

    def _finding_path(self, finding_id: str) -> Path:
        return self.findings_dir / f"{finding_id}.json"

    def _index_path(self, kind: str, semantic_key: str) -> Path:
        return self.index_dir / f"{tagged_sha256('artifact-index-v1', canonical_json_bytes([kind, semantic_key]))}.json"

    @staticmethod
    def semantic_attestation_id_for(*, artifact_id: str, kind: str, semantic_key: str, dependency_ids: Sequence[str]) -> str:
        """ID for immutable proof that one canonical closure satisfies one key."""
        return tagged_sha256("artifact-semantic-attestation-v1", canonical_json_bytes({
            "artifact_id": _text(artifact_id, "artifact_id"),
            "kind": _text(kind, "kind"),
            "semantic_key": _text(semantic_key, "semantic_key"),
            "dependency_ids": list(_ids(dependency_ids, "dependency_ids")),
        }))

    def _attestation_path(self, attestation_id: str) -> Path:
        return self.attestations_dir / f"{attestation_id}.json"

    def _attestation_body(self, envelope: ArtifactEnvelope) -> dict[str, Any]:
        attestation_id = self.semantic_attestation_id_for(
            artifact_id=envelope.artifact_id, kind=envelope.kind,
            semantic_key=envelope.semantic_key, dependency_ids=envelope.dependency_ids,
        )
        return {
            "attestation_id": attestation_id,
            "artifact_id": envelope.artifact_id,
            "kind": envelope.kind,
            "semantic_key": envelope.semantic_key,
            "dependency_ids": list(envelope.dependency_ids),
        }

    def _put_attestation(self, envelope: ArtifactEnvelope) -> str:
        body = self._attestation_body(envelope)
        path = self._attestation_path(body["attestation_id"])
        data = canonical_json_bytes(body)
        if path.exists():
            if path.read_bytes() != data:
                raise ArtifactError("immutable semantic attestation has conflicting body")
        else:
            _atomic_bytes(path, data)
        return body["attestation_id"]

    def get_semantic_attestation(self, attestation_id: str) -> Mapping[str, Any]:
        """Read exact immutable key binding; invalid sidecars are quarantined."""
        try:
            body = _read_json(self._attestation_path(attestation_id))
            required = {"attestation_id", "artifact_id", "kind", "semantic_key", "dependency_ids"}
            if set(body) != required or body.get("attestation_id") != attestation_id:
                raise ArtifactError("invalid semantic attestation")
            expected = self.semantic_attestation_id_for(
                artifact_id=body["artifact_id"], kind=body["kind"],
                semantic_key=body["semantic_key"], dependency_ids=body["dependency_ids"],
            )
            if expected != attestation_id:
                raise ArtifactError("semantic attestation hash mismatch")
            return body
        except (ArtifactError, KeyError, TypeError):
            self._quarantine_attestation(attestation_id)
            raise ArtifactError(f"invalid or missing semantic attestation {attestation_id}")

    def attestation_id_for(self, artifact_id: str, kind: str, semantic_key: str) -> str:
        envelope = self.get(artifact_id)
        if envelope.kind != kind:
            raise ArtifactError("attestation kind does not match artifact")
        return self.semantic_attestation_id_for(
            artifact_id=artifact_id, kind=kind, semantic_key=semantic_key,
            dependency_ids=envelope.dependency_ids,
        )

    def has_semantic_attestation(self, artifact_id: str, kind: str, semantic_key: str) -> bool:
        try:
            attestation_id = self.attestation_id_for(artifact_id, kind, semantic_key)
            body = self.get_semantic_attestation(attestation_id)
            return (body["artifact_id"] == artifact_id and body["kind"] == kind
                    and body["semantic_key"] == semantic_key)
        except ArtifactError:
            return False

    def attestation_ids_for(self, artifact_ids: Sequence[str]) -> tuple[str, ...]:
        """All valid key bindings for exact selected artifact IDs, sorted."""
        selected = set(_ids(artifact_ids, "artifact_ids"))
        values: list[str] = []
        for path in self.attestations_dir.glob("*.json"):
            try:
                body = self.get_semantic_attestation(path.stem)
                if body["artifact_id"] in selected:
                    values.append(path.stem)
            except ArtifactError:
                continue
        return tuple(sorted(values))

    def put_finding(self, finding: Finding) -> str:
        if not isinstance(finding, Finding):
            raise ArtifactError("finding must be a Finding")
        path = self._finding_path(finding.finding_id)
        data = finding.to_json().encode("utf-8")
        if path.exists():
            if path.read_bytes() != data:
                raise ArtifactError("immutable finding ID has conflicting body")
            return finding.finding_id
        _atomic_bytes(path, data)
        return finding.finding_id

    def get_finding(self, finding_id: str) -> Finding:
        try:
            finding = Finding.from_file(self._finding_path(finding_id))
            if finding.finding_id != finding_id:
                raise ArtifactError("finding path and body IDs differ")
            return finding
        except (OSError, SchemaError, ArtifactError) as exc:
            self._quarantine_invalid(finding_id, entry_kind="finding")
            raise ArtifactError(f"invalid or missing finding {finding_id}") from exc

    def put(
        self, kind: str, payload: Mapping[str, Any], *, dependency_ids: Sequence[str] = (),
        finding_ids: Sequence[str] = (), semantic_key: str,
    ) -> ArtifactEnvelope:
        dependencies = _ids(dependency_ids, "dependency_ids")
        findings = _ids(finding_ids, "finding_ids")
        # Findings always publish first, so a published artifact cannot name an
        # absent finding under normal operation.
        for finding_id in findings:
            self.get_finding(finding_id)
        artifact_id = artifact_id_for(kind, payload, dependencies)
        envelope = ArtifactEnvelope(
            artifact_id=artifact_id, kind=kind, payload=dict(payload), dependency_ids=dependencies,
            finding_ids=findings, semantic_key=_text(semantic_key, "semantic_key"),
        )
        path = self._artifact_path(artifact_id)
        data = envelope.to_json().encode("utf-8")
        if path.exists():
            # Artifact identity intentionally covers canonical payload plus
            # dependency closure, not cache-key/provenance annotations.  A
            # changed semantic key may legitimately regenerate byte-identical
            # output.  Retain original envelope immutably and publish another
            # immutable attestation below; never overwrite either history.
            existing = self._read_artifact(artifact_id)
            if (existing.kind != envelope.kind or existing.payload != envelope.payload
                    or existing.dependency_ids != envelope.dependency_ids):
                raise ArtifactError("canonical artifact ID has conflicting content")
        else:
            _atomic_bytes(path, data)
        self._put_attestation(envelope)
        self._add_index(envelope)
        return envelope

    def _add_index(self, envelope: ArtifactEnvelope) -> None:
        path = self._index_path(envelope.kind, envelope.semantic_key)
        if path.exists():
            current = _read_json(path)
            if current.get("kind") != envelope.kind or current.get("semantic_key") != envelope.semantic_key:
                raise ArtifactError("index address/body mismatch")
            values = current.get("artifact_ids")
            if not isinstance(values, list) or values != sorted(set(values)):
                raise ArtifactError("invalid artifact index")
        else:
            values = []
        _atomic_json(path, {
            "kind": envelope.kind, "semantic_key": envelope.semantic_key,
            "artifact_ids": sorted(set(values) | {envelope.artifact_id}),
        })

    def indexed_ids(self, kind: str, semantic_key: str) -> tuple[str, ...]:
        path = self._index_path(kind, semantic_key)
        if not path.exists():
            return ()
        try:
            value = _read_json(path)
            if value.get("kind") != kind or value.get("semantic_key") != semantic_key:
                raise ArtifactError("index address/body mismatch")
            return _ids(value.get("artifact_ids", ()), "artifact_ids")
        except ArtifactError:
            # Index is cache-only discovery metadata.  A bad index never grants
            # reuse and cannot invalidate an explicit sealed artifact.
            return ()

    def get(self, artifact_id: str, *, validate_closure: bool = True) -> ArtifactEnvelope:
        return self._get(artifact_id, validate_closure=validate_closure, seen=set())

    def _get(self, artifact_id: str, *, validate_closure: bool, seen: set[str]) -> ArtifactEnvelope:
        # Every closure member owns its failure handling.  A corrupt child must
        # be quarantined too, rather than leaving it reusable after only its
        # valid-looking parent was quarantined.
        try:
            if artifact_id in seen:
                return self._read_artifact(artifact_id)
            seen.add(artifact_id)
            envelope = self._read_artifact(artifact_id)
            expected = artifact_id_for(envelope.kind, envelope.payload, envelope.dependency_ids)
            if expected != artifact_id or envelope.artifact_id != artifact_id:
                raise ArtifactError(f"artifact content hash mismatch: {artifact_id}")
            if validate_closure:
                for finding_id in envelope.finding_ids:
                    finding = self.get_finding(finding_id)
                    for dependency_id in finding.dependency_ids:
                        self._get(dependency_id, validate_closure=True, seen=seen)
                for dependency_id in envelope.dependency_ids:
                    self._get(dependency_id, validate_closure=True, seen=seen)
            return envelope
        except ArtifactError:
            self._quarantine_invalid(artifact_id)
            raise

    def _read_artifact(self, artifact_id: str) -> ArtifactEnvelope:
        try:
            envelope = ArtifactEnvelope.from_file(self._artifact_path(artifact_id))
        except (OSError, SchemaError) as exc:
            raise ArtifactError(f"invalid or missing artifact {artifact_id}") from exc
        if envelope.artifact_id != artifact_id:
            raise ArtifactError("artifact path and body IDs differ")
        return envelope

    def _quarantine_attestation(self, attestation_id: str) -> None:
        """Attestations are mutable-cache sidecars; sealed copies stay untouched."""
        source = self._attestation_path(attestation_id)
        if source.exists():
            target = self.quarantine_dir / f"attestation-{attestation_id}-{uuid.uuid4().hex}.json"
            try:
                os.replace(source, target)
                _fsync_directory(self.quarantine_dir)
            except OSError:
                pass

    def _quarantine_invalid(self, artifact_id: str, *, entry_kind: str = "artifact") -> None:
        """Quarantine only mutable cache objects; sealed revision copies untouched."""
        if entry_kind not in {"artifact", "finding"}:
            raise ArtifactError("unknown cache entry kind")
        source = self._artifact_path(artifact_id) if entry_kind == "artifact" else self._finding_path(artifact_id)
        if source.exists():
            target = self.quarantine_dir / f"{entry_kind}-{artifact_id}-{uuid.uuid4().hex}.json"
            try:
                os.replace(source, target)
                _fsync_directory(self.quarantine_dir)
            except OSError:
                return
        finding = Finding(
            kind="cache_artifact_invalid", severity="warning", stage="cache",
            subject_refs=(artifact_id,), evidence={"artifact_id": artifact_id},
            message="Invalid cache artifact quarantined; it is not reusable.",
        )
        try:
            self.put_finding(finding)
        except (ArtifactError, OSError):
            # Validation must fail closed even if cache diagnostic persistence is
            # impossible (e.g. read-only damaged workspace).
            pass

    def closure(
        self, artifact_ids: Sequence[str], *, finding_ids: Sequence[str] = (),
    ) -> tuple[tuple[ArtifactEnvelope, ...], tuple[Finding, ...]]:
        """Read a complete verified artifact/finding transitive closure."""
        artifact_map: dict[str, ArtifactEnvelope] = {}
        finding_map: dict[str, Finding] = {}

        def visit_finding(finding_id: str) -> None:
            if finding_id in finding_map:
                return
            finding = self.get_finding(finding_id)
            finding_map[finding_id] = finding
            for dependency_id in finding.dependency_ids:
                visit(dependency_id)

        def visit(artifact_id: str) -> None:
            if artifact_id in artifact_map:
                return
            envelope = self.get(artifact_id, validate_closure=False)
            artifact_map[artifact_id] = envelope
            for named_finding_id in envelope.finding_ids:
                visit_finding(named_finding_id)
            for dependency_id in envelope.dependency_ids:
                visit(dependency_id)

        for artifact_id in _ids(artifact_ids, "artifact_ids"):
            visit(artifact_id)
        for finding_id in _ids(finding_ids, "finding_ids"):
            visit_finding(finding_id)
        return tuple(artifact_map[key] for key in sorted(artifact_map)), tuple(finding_map[key] for key in sorted(finding_map))


def _v2_snapshot_bytes(snapshot: RevisionSnapshot, edge_ids: Sequence[str]) -> bytes:
    """Serialize a v2 snapshot with its exact selected relationship closure."""
    if not isinstance(snapshot, RevisionSnapshot):
        raise ArtifactError("snapshot must be RevisionSnapshot")
    body = snapshot.to_dict()
    body["selected_edge_ids"] = list(_ids(edge_ids, "edge_ids"))
    return canonical_json_bytes(body)


def _v2_snapshot_from_bytes(data: bytes) -> tuple[RevisionSnapshot, tuple[str, ...] | None]:
    """Read a v2 snapshot and its edge selection; accept old snapshots."""
    if not isinstance(data, bytes):
        raise ArtifactError("snapshot data must be bytes")
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid v2 snapshot") from exc
    if not isinstance(body, dict):
        raise ArtifactError("v2 snapshot must be an object")
    selected = body.pop("selected_edge_ids", None)
    if selected is not None:
        if not isinstance(selected, list):
            raise ArtifactError("selected_edge_ids must be an array")
        edge_ids = _ids(selected, "selected_edge_ids")
        if list(edge_ids) != selected:
            raise ArtifactError("selected_edge_ids must be sorted and unique")
    else:
        edge_ids = None
    try:
        snapshot = RevisionSnapshot.from_dict(body)
    except SchemaError as exc:
        raise ArtifactError("invalid v2 snapshot") from exc
    if canonical_json_bytes(body) != canonical_json_bytes(snapshot.to_dict()):
        raise ArtifactError("v2 snapshot is not canonical")
    return snapshot, edge_ids


class CacheValidator:
    """Fail-closed cache reuse from a named artifact in a named snapshot."""

    def __init__(self, store: LegacyArtifactStore):
        self.store = store

    def select(
        self, snapshot: RevisionSnapshot, *, requested_artifact_id: str | None, kind: str,
        key_constructor: Any, current_inputs: Mapping[str, Any] | None = None, **semantic_inputs: Any,
    ) -> ArtifactEnvelope | None:
        """Return only requested selected artifact after recomputing its key.

        Caller supplies current semantic inputs, not a claimed key.  Indexes
        are discovery metadata only; selected closure, key, and attestation
        authorize reuse.
        """
        if not isinstance(snapshot, RevisionSnapshot):
            raise ArtifactError("snapshot must be RevisionSnapshot")
        _text(kind, "kind")
        if not callable(key_constructor):
            raise ArtifactError("key_constructor must be callable")
        if current_inputs is not None:
            if not isinstance(current_inputs, Mapping) or semantic_inputs:
                raise ArtifactError("current_inputs must be the only semantic input mapping")
            semantic_inputs = dict(current_inputs)
        requested_semantic_key = key_constructor(**semantic_inputs)
        _text(requested_semantic_key, "recomputed semantic key")
        if requested_artifact_id is None:
            self.ambiguity(kind, requested_semantic_key)
            return None
        if requested_artifact_id not in snapshot.selected_artifact_ids:
            return None
        # The envelope's first-published key is deliberately not authoritative:
        # one identical artifact can be independently attested by multiple
        # model/prompt keys.  Indexes are only candidate discovery and must not
        # be required for an explicitly selected artifact.
        try:
            envelope = self.store.get(requested_artifact_id)
            if envelope.kind != kind:
                return None
            attestation_id = self.store.attestation_id_for(
                requested_artifact_id, kind, requested_semantic_key)
            # Snapshot attestations are selection authority, never optional
            # discovery hints.  In particular, an old/empty list must not
            # authorize a global history attestation for this artifact/key.
            if attestation_id not in snapshot.selected_cache_attestation_ids:
                return None
            if not self.store.has_semantic_attestation(requested_artifact_id, kind, requested_semantic_key):
                return None
        except ArtifactError:
            return None
        return envelope

    def select_from_inputs(
        self, snapshot: RevisionSnapshot, *, requested_artifact_id: str | None,
        kind: str, key_constructor: Any, **current_inputs: Any,
    ) -> ArtifactEnvelope | None:
        """Compatibility spelling for the same recomputing selection boundary."""
        return self.select(snapshot, requested_artifact_id=requested_artifact_id, kind=kind,
                           key_constructor=key_constructor, current_inputs=current_inputs)

    # Both names make this boundary easy to use at leaf executors.
    validate = select
    reuse = select

    def ambiguity(self, kind: str, requested_semantic_key: str) -> Finding | None:
        """Report same-key ambiguity; never choose an indexed candidate."""
        candidates = self.store.indexed_ids(kind, requested_semantic_key)
        if len(candidates) < 2:
            return None
        finding = Finding(
            kind="cache_key_ambiguous", severity="warning", stage="cache", subject_refs=candidates,
            evidence={"kind": kind, "semantic_key": requested_semantic_key, "artifact_ids": list(candidates)},
            message="Multiple immutable cache values share a semantic key; no value was selected.",
        )
        self.store.put_finding(finding)
        return finding


class LegacyDependencyGraph:
    """Immutable graph edges and selected-revision-only traversal."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.edges_dir = self.root / "graph" / "edges"
        self.revisions_dir = self.root / "graph" / "revisions"
        self.edges_dir.mkdir(parents=True, exist_ok=True)
        self.revisions_dir.mkdir(parents=True, exist_ok=True)

    def edge(self, *, stable_subject_id: str, parent_artifact_id: str, child_artifact_id: str, stage: str, edge_kind: str) -> DependencyGraphEdge:
        return DependencyGraphEdge(
            edge_id=dependency_edge_id_for(stable_subject_id, parent_artifact_id, child_artifact_id, stage, edge_kind),
            stable_subject_id=stable_subject_id, parent_artifact_id=parent_artifact_id,
            child_artifact_id=child_artifact_id, stage=stage, edge_kind=edge_kind,
        )

    def put(self, edge: DependencyGraphEdge) -> str:
        if not isinstance(edge, DependencyGraphEdge):
            raise ArtifactError("edge must be DependencyGraphEdge")
        expected = dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind)
        if edge.edge_id != expected:
            raise ArtifactError("edge_id does not match canonical edge")
        path = self.edges_dir / f"{edge.edge_id}.json"
        data = edge.to_json().encode("utf-8")
        if path.exists():
            if path.read_bytes() != data:
                raise ArtifactError("immutable edge ID has conflicting body")
        else:
            _atomic_bytes(path, data)
        return edge.edge_id

    def get(self, edge_id: str) -> DependencyGraphEdge:
        try:
            edge = DependencyGraphEdge.from_file(self.edges_dir / f"{edge_id}.json")
        except (OSError, SchemaError) as exc:
            raise ArtifactError(f"invalid or missing graph edge {edge_id}") from exc
        if edge.edge_id != edge_id or edge.edge_id != dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind):
            raise ArtifactError("graph edge hash mismatch")
        return edge

    def bind_revision(self, revision_id: str, edge_ids: Sequence[str], *, allowed_artifact_ids: Sequence[str]) -> None:
        revision_id = _text(revision_id, "revision_id")
        allowed = set(_ids(allowed_artifact_ids, "allowed_artifact_ids"))
        ids = _ids(edge_ids, "edge_ids")
        for edge_id in ids:
            edge = self.get(edge_id)
            if edge.parent_artifact_id not in allowed or edge.child_artifact_id not in allowed:
                raise ArtifactError("selected graph edge endpoint is outside revision closure")
        path = self.revisions_dir / f"{revision_id}.json"
        body = {"revision_id": revision_id, "edge_ids": list(ids)}
        if path.exists():
            if _read_json(path) != body:
                raise ArtifactError("immutable revision graph has conflicting edges")
            return
        _atomic_json(path, body)

    def edge_ids(self, revision_id: str) -> tuple[str, ...]:
        value = _read_json(self.revisions_dir / f"{revision_id}.json")
        if value.get("revision_id") != revision_id:
            raise ArtifactError("revision graph address/body mismatch")
        return _ids(value.get("edge_ids", ()), "edge_ids")

    def edges(self, revision_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(self.get(edge_id) for edge_id in self.edge_ids(revision_id))

    def forward(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id) if edge.parent_artifact_id == node_id or edge.stable_subject_id == node_id)

    def reverse(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id) if edge.child_artifact_id == node_id or edge.stable_subject_id == node_id)

    traverse_forward = forward
    traverse_reverse = reverse

    def descendants(self, revision_id: str, artifact_id: str) -> tuple[str, ...]:
        return self._walk(revision_id, artifact_id, forward=True)

    def ancestors(self, revision_id: str, artifact_id: str) -> tuple[str, ...]:
        return self._walk(revision_id, artifact_id, forward=False)

    def _walk(self, revision_id: str, artifact_id: str, *, forward: bool) -> tuple[str, ...]:
        edges = self.edges(revision_id)
        result: set[str] = set()
        queue: deque[str] = deque([artifact_id])
        while queue:
            node = queue.popleft()
            for edge in edges:
                source, target = ((edge.parent_artifact_id, edge.child_artifact_id) if forward else (edge.child_artifact_id, edge.parent_artifact_id))
                if source == node and target not in result:
                    result.add(target)
                    queue.append(target)
        return tuple(sorted(result))


class SealedDependencyGraph:
    """Read-only graph view backed only by one verified revision bundle."""

    def __init__(self, bundle: Path, revision_id: str):
        self.bundle = bundle
        self.revision_id = revision_id

    def _require_revision(self, revision_id: str) -> None:
        if revision_id != self.revision_id:
            raise ArtifactError("sealed graph belongs to a different revision")

    def edge_ids(self, revision_id: str) -> tuple[str, ...]:
        self._require_revision(revision_id)
        manifest = _read_json(self.bundle / "bundle-manifest.json")
        if manifest.get("revision_id") != revision_id:
            raise ArtifactError("sealed graph manifest revision mismatch")
        return _ids(manifest.get("edge_ids", ()), "edge_ids")

    def get(self, edge_id: str) -> DependencyGraphEdge:
        try:
            edge = DependencyGraphEdge.from_file(self.bundle / "graph" / f"{edge_id}.json")
        except (OSError, SchemaError) as exc:
            raise ArtifactError(f"invalid or missing sealed graph edge {edge_id}") from exc
        expected = dependency_edge_id_for(
            edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind,
        )
        if edge.edge_id != edge_id or edge.edge_id != expected:
            raise ArtifactError("sealed graph edge hash mismatch")
        return edge

    def edges(self, revision_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(self.get(edge_id) for edge_id in self.edge_ids(revision_id))

    def forward(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id)
                     if edge.parent_artifact_id == node_id or edge.stable_subject_id == node_id)

    def reverse(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id)
                     if edge.child_artifact_id == node_id or edge.stable_subject_id == node_id)

    traverse_forward = forward
    traverse_reverse = reverse

    def descendants(self, revision_id: str, artifact_id: str) -> tuple[str, ...]:
        return self._walk(revision_id, artifact_id, forward=True)

    def ancestors(self, revision_id: str, artifact_id: str) -> tuple[str, ...]:
        return self._walk(revision_id, artifact_id, forward=False)

    def _walk(self, revision_id: str, artifact_id: str, *, forward: bool) -> tuple[str, ...]:
        edges = self.edges(revision_id)
        result: set[str] = set()
        queue: deque[str] = deque([artifact_id])
        while queue:
            node = queue.popleft()
            for edge in edges:
                source, target = ((edge.parent_artifact_id, edge.child_artifact_id)
                                  if forward else (edge.child_artifact_id, edge.parent_artifact_id))
                if source == node and target not in result:
                    result.add(target)
                    queue.append(target)
        return tuple(sorted(result))


class LegacyRevisionStore:
    """Sealed self-contained revision bundles and explicit active pointer."""

    def __init__(self, root: Path | str, artifact_store: LegacyArtifactStore | None = None, graph: LegacyDependencyGraph | None = None):
        self.root = Path(root)
        self.revisions_dir = self.root / "revisions"
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifact_store or LegacyArtifactStore(self.root)
        self.graph = graph or LegacyDependencyGraph(self.root)

    def _revision_path(self, revision_id: str) -> Path:
        return self.revisions_dir / revision_id

    def seal_bundle(
        self, snapshot: RevisionSnapshot, provenance: Mapping[str, Any], epub: bytes | Path | str, *,
        render_input_artifact_id: str | None = None, edge_ids: Sequence[str] = (), epub_filename: str = "book.epub",
        expected_embedded_provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically publish verified closed bundle.  This never activates it."""
        if not isinstance(snapshot, RevisionSnapshot):
            raise ArtifactError("snapshot must be RevisionSnapshot")
        if not isinstance(provenance, Mapping):
            raise ArtifactError("provenance must be an object")
        if Path(epub_filename).name != epub_filename or not epub_filename:
            raise ArtifactError("epub_filename must be a bare filename")
        artifacts, findings = self.artifacts.closure(
            snapshot.selected_artifact_ids, finding_ids=snapshot.selected_finding_ids,
        )
        artifact_ids = {item.artifact_id for item in artifacts}
        finding_ids = {item.finding_id for item in findings}
        attestations = {
            attestation_id: self.artifacts.get_semantic_attestation(attestation_id)
            for attestation_id in snapshot.selected_cache_attestation_ids
        }
        for body in attestations.values():
            artifact_id = body["artifact_id"]
            if artifact_id not in artifact_ids:
                raise ArtifactError("semantic attestation escapes selected artifact closure")
            envelope = next(item for item in artifacts if item.artifact_id == artifact_id)
            if (body["kind"] != envelope.kind
                    or tuple(body["dependency_ids"]) != envelope.dependency_ids):
                raise ArtifactError("semantic attestation does not bind selected canonical closure")
        if render_input_artifact_id is not None:
            if render_input_artifact_id not in artifact_ids:
                raise ArtifactError("render input is outside selected artifact closure")
            render_input = self.artifacts.get(render_input_artifact_id)
            render_input_hash = hashlib.sha256(canonical_json_bytes(render_input.to_dict())).hexdigest()
        else:
            render_input_hash = None
        epub_bytes = self._epub_bytes(epub)
        epub_hash = hashlib.sha256(epub_bytes).hexdigest()
        if expected_embedded_provenance is not None and canonical_json_bytes(dict(expected_embedded_provenance)) != canonical_json_bytes(dict(provenance)):
            raise ArtifactError("expected embedded provenance must equal bundle provenance")
        self._verify_embedded_provenance(epub_bytes, provenance)
        edge_ids = _ids(edge_ids, "edge_ids")
        for edge_id in edge_ids:
            edge = self.graph.get(edge_id)
            if edge.parent_artifact_id not in artifact_ids or edge.child_artifact_id not in artifact_ids:
                raise ArtifactError("bundle graph edge escapes selected closure")

        manifest = {
            "revision_id": snapshot.revision_id,
            "snapshot_sha256": hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest(),
            "artifact_ids": sorted(artifact_ids), "finding_ids": sorted(finding_ids),
            "semantic_attestation_ids": sorted(attestations), "edge_ids": list(edge_ids),
            "render_input_artifact_id": render_input_artifact_id, "render_input_hash": render_input_hash,
            "epub_filename": epub_filename, "epub_sha256": epub_hash,
            "provenance_sha256": hashlib.sha256(canonical_json_bytes(dict(provenance))).hexdigest(),
        }
        destination = self._revision_path(snapshot.revision_id)
        if destination.exists():
            existing = destination / "bundle-manifest.json"
            if existing.exists() and _read_json(existing) == manifest:
                # Equal manifest alone does not prove copied closure/EPUB stayed
                # intact; idempotent sealing still revalidates sealed bytes.
                self.verify_bundle(snapshot.revision_id)
                return destination
            raise ArtifactError("sealed revision ID already exists with different bundle")

        staging = self.revisions_dir / f".{snapshot.revision_id}.{uuid.uuid4().hex}.tmp"
        try:
            staging.mkdir()
            _atomic_bytes(staging / "snapshot.json", snapshot.to_json().encode("utf-8"))
            _atomic_json(staging / "provenance.json", dict(provenance))
            _atomic_bytes(staging / epub_filename, epub_bytes)
            for envelope in artifacts:
                _atomic_bytes(staging / "artifacts" / f"{envelope.artifact_id}.json", envelope.to_json().encode("utf-8"))
            for finding_id in sorted(finding_ids):
                finding = self.artifacts.get_finding(finding_id)
                _atomic_bytes(staging / "findings" / f"{finding_id}.json", finding.to_json().encode("utf-8"))
            for attestation_id in sorted(attestations):
                _atomic_bytes(staging / "attestations" / f"{attestation_id}.json",
                              canonical_json_bytes(attestations[attestation_id]))
            for edge_id in edge_ids:
                edge = self.graph.get(edge_id)
                _atomic_bytes(staging / "graph" / f"{edge_id}.json", edge.to_json().encode("utf-8"))
            _atomic_json(staging / "bundle-manifest.json", manifest)
            self._fsync_tree(staging)
            os.replace(staging, destination)
            _fsync_directory(self.revisions_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        # Re-open copied bytes before exposing graph/activation state; the
        # sealed bundle is self-verifying rather than a mutable-cache view.
        self.verify_bundle(snapshot.revision_id)
        # Bind only after published sealed closure exists. This map is immutable
        # and gives consumers no access to non-selected cache graph edges.
        self.graph.bind_revision(snapshot.revision_id, edge_ids, allowed_artifact_ids=tuple(artifact_ids))
        return destination

    @staticmethod
    def _epub_bytes(epub: bytes | Path | str) -> bytes:
        if isinstance(epub, bytes):
            return epub
        path = Path(epub)
        if path.is_symlink():
            raise ArtifactError("bundle EPUB must not be a symlink")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ArtifactError("cannot read EPUB for sealing") from exc

    @staticmethod
    def _verify_embedded_provenance(epub_bytes: bytes, expected: Mapping[str, Any]) -> None:
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(epub_bytes)) as archive:
                names = set(archive.namelist())
                candidates = ("META-INF/btran-provenance.json", "btran-provenance.json", "provenance.json")
                name = next((item for item in candidates if item in names), None)
                if name is None:
                    raise ArtifactError("EPUB does not embed provenance")
                body = archive.read(name)
        except zipfile.BadZipFile as exc:
            raise ArtifactError("EPUB provenance verification requires a ZIP EPUB") from exc
        if body != canonical_json_bytes(dict(expected)):
            raise ArtifactError("embedded EPUB provenance differs from bundle provenance")

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        for directory, _, files in os.walk(root):
            path = Path(directory)
            for filename in files:
                with (path / filename).open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(path)

    def _read_snapshot(self, revision_id: str) -> RevisionSnapshot:
        path = self._revision_path(revision_id) / "snapshot.json"
        try:
            snapshot = RevisionSnapshot.from_file(path)
        except (OSError, SchemaError) as exc:
            raise ArtifactError(f"invalid or missing sealed revision {revision_id}") from exc
        if snapshot.revision_id != revision_id:
            raise ArtifactError("revision directory/snapshot ID mismatch")
        return snapshot

    def verify_bundle(self, revision_id: str) -> RevisionSnapshot:
        """Validate copied schemas, IDs, closure, hashes, and EPUB provenance."""
        bundle = self._revision_path(revision_id)
        if not bundle.is_dir() or bundle.is_symlink():
            raise ArtifactError("sealed revision bundle is missing or unsafe")
        for path in bundle.rglob("*"):
            if path.is_symlink():
                raise ArtifactError("sealed revision bundle contains a symlink")
        snapshot = self._read_snapshot(revision_id)
        manifest = _read_json(bundle / "bundle-manifest.json")
        if manifest.get("revision_id") != revision_id:
            raise ArtifactError("bundle manifest revision mismatch")
        if manifest.get("snapshot_sha256") != hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest():
            raise ArtifactError("bundle snapshot hash mismatch")
        provenance = _read_json(bundle / "provenance.json")
        if manifest.get("provenance_sha256") != hashlib.sha256(canonical_json_bytes(provenance)).hexdigest():
            raise ArtifactError("bundle provenance hash mismatch")
        epub_filename = manifest.get("epub_filename")
        if not isinstance(epub_filename, str) or Path(epub_filename).name != epub_filename:
            raise ArtifactError("bundle EPUB filename is invalid")
        try:
            epub_bytes = (bundle / epub_filename).read_bytes()
        except OSError as exc:
            raise ArtifactError("bundle EPUB is missing") from exc
        if manifest.get("epub_sha256") != hashlib.sha256(epub_bytes).hexdigest():
            raise ArtifactError("bundle EPUB hash mismatch")
        self._verify_embedded_provenance(epub_bytes, provenance)

        manifest_artifacts = _ids(manifest.get("artifact_ids", ()), "artifact_ids")
        manifest_findings = _ids(manifest.get("finding_ids", ()), "finding_ids")
        manifest_attestations = _ids(manifest.get("semantic_attestation_ids", ()), "semantic_attestation_ids")
        manifest_edges = _ids(manifest.get("edge_ids", ()), "edge_ids")
        if (not set(snapshot.selected_artifact_ids).issubset(manifest_artifacts)
                or not set(snapshot.selected_finding_ids).issubset(manifest_findings)
                or not set(snapshot.selected_cache_attestation_ids).issubset(manifest_attestations)):
            raise ArtifactError("bundle omits selected snapshot IDs")
        envelopes: dict[str, ArtifactEnvelope] = {}
        for artifact_id in manifest_artifacts:
            try:
                envelope = ArtifactEnvelope.from_file(bundle / "artifacts" / f"{artifact_id}.json")
            except (OSError, SchemaError) as exc:
                raise ArtifactError("bundle artifact is missing or invalid") from exc
            if envelope.artifact_id != artifact_id or artifact_id_for(envelope.kind, envelope.payload, envelope.dependency_ids) != artifact_id:
                raise ArtifactError("bundle artifact content hash mismatch")
            envelopes[artifact_id] = envelope
        for attestation_id in manifest_attestations:
            try:
                body = _read_json(bundle / "attestations" / f"{attestation_id}.json")
                required = {"attestation_id", "artifact_id", "kind", "semantic_key", "dependency_ids"}
                if set(body) != required or body["attestation_id"] != attestation_id:
                    raise ArtifactError("bundle semantic attestation is invalid")
                expected = LegacyArtifactStore.semantic_attestation_id_for(
                    artifact_id=body["artifact_id"], kind=body["kind"],
                    semantic_key=body["semantic_key"], dependency_ids=body["dependency_ids"],
                )
                envelope = envelopes.get(body["artifact_id"])
                if (expected != attestation_id or envelope is None or body["kind"] != envelope.kind
                        or tuple(body["dependency_ids"]) != envelope.dependency_ids):
                    raise ArtifactError("bundle semantic attestation closure mismatch")
            except (ArtifactError, KeyError, TypeError):
                raise ArtifactError("bundle semantic attestation is missing or invalid") from None
        findings: dict[str, Finding] = {}
        for finding_id in manifest_findings:
            try:
                finding = Finding.from_file(bundle / "findings" / f"{finding_id}.json")
            except (OSError, SchemaError) as exc:
                raise ArtifactError("bundle finding is missing or invalid") from exc
            if finding.finding_id != finding_id:
                raise ArtifactError("bundle finding ID mismatch")
            findings[finding_id] = finding
        for envelope in envelopes.values():
            if not set(envelope.dependency_ids).issubset(envelopes) or not set(envelope.finding_ids).issubset(findings):
                raise ArtifactError("bundle artifact closure is incomplete")
        for finding in findings.values():
            if not set(finding.dependency_ids).issubset(envelopes):
                raise ArtifactError("bundle finding closure is incomplete")
        render_input_id = manifest.get("render_input_artifact_id")
        render_input_hash = manifest.get("render_input_hash")
        if render_input_id is None:
            if render_input_hash is not None:
                raise ArtifactError("bundle render-input hash has no input")
        elif render_input_id not in envelopes or render_input_hash != hashlib.sha256(canonical_json_bytes(envelopes[render_input_id].to_dict())).hexdigest():
            raise ArtifactError("bundle render-input hash mismatch")
        for edge_id in manifest_edges:
            try:
                edge = DependencyGraphEdge.from_file(bundle / "graph" / f"{edge_id}.json")
            except (OSError, SchemaError) as exc:
                raise ArtifactError("bundle graph edge is missing or invalid") from exc
            if edge.edge_id != edge_id or edge.edge_id != dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind):
                raise ArtifactError("bundle graph edge hash mismatch")
            if edge.parent_artifact_id not in envelopes or edge.child_artifact_id not in envelopes:
                raise ArtifactError("bundle graph edge escapes closure")
        return snapshot

    def snapshot(self, revision_id: str) -> RevisionSnapshot:
        return self.verify_bundle(revision_id)

    def activate(self, revision_id: str) -> None:
        # Ensure only a fully sealed, readable candidate can become active.
        self.verify_bundle(revision_id)
        _atomic_json(self.root / "active-revision.json", {"revision_id": revision_id})

    def active_snapshot(self) -> RevisionSnapshot | None:
        path = self.root / "active-revision.json"
        if not path.exists():
            return None
        value = _read_json(path)
        if set(value) != {"revision_id"}:
            raise ArtifactError("invalid active revision pointer")
        return self.snapshot(_text(value["revision_id"], "revision_id"))

    def selected_graph(self, revision_id: str) -> SealedDependencyGraph:
        # Never traverse mutable global graph storage.  Bundle verification
        # validates copied graph closure before exposing this read-only view.
        self.snapshot(revision_id)
        return SealedDependencyGraph(self._revision_path(revision_id), revision_id)


# Short compatibility aliases for callers that prefer noun-first naming.
canonical_artifact_id = artifact_id_for
canonical_edge_id = dependency_edge_id_for

# --- v2 compact SQLite/ZIP adapters -------------------------------------------------
# Legacy classes above remain available internally so an old workspace is never
# migrated, rewritten, or quarantined merely by being read.
from btran.storage import Storage, StorageError


def _legacy_workspace(root: Path) -> bool:
    if (root / "state-v2.sqlite3").exists():
        return False
    return any((root / name).exists() for name in ("artifacts", "findings", "index", "attestations", "graph", "active-revision.json")) or any(
        path.is_dir() and not path.name.startswith(".") for path in (root / "revisions").glob("*")
    ) if (root / "revisions").exists() else any((root / name).exists() for name in ("artifacts", "findings", "index", "attestations", "graph", "active-revision.json"))


class V2ArtifactStore:
    """ArtifactStore implementation backed only by ``state-v2.sqlite3``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.storage = Storage(self.root)
        # Compatibility attributes are paths only; v2 never writes loose records.
        self.artifacts_dir = self.root / "artifacts"
        self.findings_dir = self.root / "findings"
        self.index_dir = self.root / "index"
        self.attestations_dir = self.root / "attestations"
        self.quarantine_dir = self.root / "quarantine"

    @staticmethod
    def semantic_attestation_id_for(*, artifact_id: str, kind: str, semantic_key: str, dependency_ids: Sequence[str]) -> str:
        return tagged_sha256("artifact-semantic-attestation-v1", canonical_json_bytes({
            "artifact_id": _text(artifact_id, "artifact_id"), "kind": _text(kind, "kind"),
            "semantic_key": _text(semantic_key, "semantic_key"), "dependency_ids": list(_ids(dependency_ids, "dependency_ids")),
        }))

    def _attestation_body(self, envelope: ArtifactEnvelope) -> dict[str, Any]:
        aid = self.semantic_attestation_id_for(artifact_id=envelope.artifact_id, kind=envelope.kind,
                                               semantic_key=envelope.semantic_key, dependency_ids=envelope.dependency_ids)
        return {"attestation_id": aid, "artifact_id": envelope.artifact_id, "kind": envelope.kind,
                "semantic_key": envelope.semantic_key, "dependency_ids": list(envelope.dependency_ids)}

    def put_finding(self, finding: Finding) -> str:
        if not isinstance(finding, Finding):
            raise ArtifactError("finding must be a Finding")
        self.storage.put_finding(finding.finding_id, finding.to_json().encode("utf-8"))
        return finding.finding_id

    def get_finding(self, finding_id: str) -> Finding:
        try:
            finding = Finding.from_json(self.storage.finding_bytes(finding_id).decode("utf-8"))
            if finding.finding_id != finding_id:
                raise ArtifactError("finding path and body IDs differ")
            return finding
        except (StorageError, SchemaError, UnicodeDecodeError) as exc:
            raise ArtifactError(f"invalid or missing finding {finding_id}") from exc

    def put(self, kind: str, payload: Mapping[str, Any], *, dependency_ids: Sequence[str] = (),
            finding_ids: Sequence[str] = (), semantic_key: str) -> ArtifactEnvelope:
        dependencies, findings = _ids(dependency_ids, "dependency_ids"), _ids(finding_ids, "finding_ids")
        for finding_id in findings:
            self.get_finding(finding_id)
        envelope = ArtifactEnvelope(artifact_id=artifact_id_for(kind, payload, dependencies), kind=_text(kind, "kind"),
                                    payload=dict(payload), dependency_ids=dependencies, finding_ids=findings,
                                    semantic_key=_text(semantic_key, "semantic_key"))
        data = envelope.to_json().encode("utf-8")
        try:
            existing = self._read(envelope.artifact_id)
        except ArtifactError:
            existing = None
        if (existing is not None and existing.kind == envelope.kind
                and existing.payload == envelope.payload and existing.dependency_ids == envelope.dependency_ids):
            # Identity intentionally excludes diagnostic and cache-key
            # annotations.  Preserve the first immutable record bytes while
            # publishing a new semantic index binding and attestation.
            self.storage.index_record(envelope.artifact_id, envelope.semantic_key)
        else:
            self.storage.put_record(envelope.artifact_id, envelope.kind, data,
                                    semantic_key=envelope.semantic_key, dependency_ids=dependencies, finding_ids=findings)
        attestation = self._attestation_body(envelope)
        self.storage.put_attestation(attestation["attestation_id"], canonical_json_bytes(attestation))
        return envelope

    def _read(self, artifact_id: str) -> ArtifactEnvelope:
        try:
            envelope = ArtifactEnvelope.from_json(self.storage.record_bytes(artifact_id).decode("utf-8"))
            if envelope.artifact_id != artifact_id or artifact_id_for(envelope.kind, envelope.payload, envelope.dependency_ids) != artifact_id:
                raise ArtifactError("artifact content hash mismatch")
            return envelope
        except (StorageError, SchemaError, UnicodeDecodeError) as exc:
            raise ArtifactError(f"invalid or missing artifact {artifact_id}") from exc

    def get(self, artifact_id: str, *, validate_closure: bool = True) -> ArtifactEnvelope:
        return self._get(artifact_id, validate_closure=validate_closure, seen=set())

    def _get(self, artifact_id: str, *, validate_closure: bool, seen: set[str]) -> ArtifactEnvelope:
        envelope = self._read(artifact_id)
        if validate_closure and artifact_id not in seen:
            seen.add(artifact_id)
            for finding_id in envelope.finding_ids:
                finding = self.get_finding(finding_id)
                for dependency_id in finding.dependency_ids:
                    self._get(dependency_id, validate_closure=True, seen=seen)
            for dependency_id in envelope.dependency_ids:
                self._get(dependency_id, validate_closure=True, seen=seen)
        return envelope

    def _attestation(self, attestation_id: str) -> Mapping[str, Any]:
        try:
            body = json.loads(self.storage.attestation_bytes(attestation_id).decode("utf-8"))
            if not isinstance(body, Mapping) or canonical_json_bytes(body) != self.storage.attestation_bytes(attestation_id):
                raise ArtifactError("invalid semantic attestation")
            required = {"attestation_id", "artifact_id", "kind", "semantic_key", "dependency_ids"}
            if set(body) != required or body["attestation_id"] != attestation_id:
                raise ArtifactError("invalid semantic attestation")
            expected = self.semantic_attestation_id_for(artifact_id=body["artifact_id"], kind=body["kind"],
                                                        semantic_key=body["semantic_key"], dependency_ids=body["dependency_ids"])
            if expected != attestation_id:
                raise ArtifactError("semantic attestation hash mismatch")
            return body
        except (StorageError, UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ArtifactError(f"invalid or missing semantic attestation {attestation_id}") from exc

    def get_semantic_attestation(self, attestation_id: str) -> Mapping[str, Any]:
        return self._attestation(attestation_id)

    def attestation_id_for(self, artifact_id: str, kind: str, semantic_key: str) -> str:
        envelope = self.get(artifact_id)
        if envelope.kind != kind:
            raise ArtifactError("attestation kind does not match artifact")
        return self.semantic_attestation_id_for(artifact_id=artifact_id, kind=kind, semantic_key=semantic_key,
                                                dependency_ids=envelope.dependency_ids)

    def has_semantic_attestation(self, artifact_id: str, kind: str, semantic_key: str) -> bool:
        try:
            aid = self.attestation_id_for(artifact_id, kind, semantic_key)
            body = self.get_semantic_attestation(aid)
            return body["artifact_id"] == artifact_id and body["kind"] == kind and body["semantic_key"] == semantic_key
        except (ArtifactError, KeyError):
            return False

    def attestation_ids_for(self, artifact_ids: Sequence[str]) -> tuple[str, ...]:
        selected = set(_ids(artifact_ids, "artifact_ids"))
        connection = self.storage._connect()
        try:
            ids = tuple(row[0] for row in connection.execute("SELECT attestation_id FROM attestations ORDER BY attestation_id"))
        finally:
            connection.close()
        result = []
        for aid in ids:
            try:
                if self.get_semantic_attestation(aid)["artifact_id"] in selected:
                    result.append(aid)
            except ArtifactError:
                continue
        return tuple(result)

    def indexed_ids(self, kind: str, semantic_key: str) -> tuple[str, ...]:
        result = []
        for aid in self.storage.indexed_ids(semantic_key):
            try:
                if self.storage.record_meta(aid)["kind"] == kind:
                    result.append(aid)
            except StorageError:
                continue
        return tuple(result)

    def closure(self, artifact_ids: Sequence[str], *, finding_ids: Sequence[str] = ()) -> tuple[tuple[ArtifactEnvelope, ...], tuple[Finding, ...]]:
        artifacts: dict[str, ArtifactEnvelope] = {}
        findings: dict[str, Finding] = {}
        def visit_finding(fid: str) -> None:
            if fid in findings: return
            finding = self.get_finding(fid); findings[fid] = finding
            for dep in finding.dependency_ids: visit(dep)
        def visit(aid: str) -> None:
            if aid in artifacts: return
            artifact = self.get(aid, validate_closure=False); artifacts[aid] = artifact
            for fid in artifact.finding_ids: visit_finding(fid)
            for dep in artifact.dependency_ids: visit(dep)
        def visit_root(aid: str) -> None: visit(aid)
        for aid in _ids(artifact_ids, "artifact_ids"): visit_root(aid)
        for fid in _ids(finding_ids, "finding_ids"): visit_finding(fid)
        return tuple(artifacts[k] for k in sorted(artifacts)), tuple(findings[k] for k in sorted(findings))


class _VirtualEdgePath:
    def __init__(self, edge_id: str): self.stem = edge_id


class _VirtualEdgeDir:
    """Compatibility discovery view over immutable SQLite edge rows.

    It has no filesystem side effects; callers can retain the old ``glob``
    loop while v2 keeps graph bytes in the compact store.
    """
    def __init__(self, storage: Storage): self.storage = storage
    def glob(self, pattern: str):
        if pattern != "*.json": return ()
        connection = self.storage._connect()
        try:
            return tuple(_VirtualEdgePath(row[0]) for row in connection.execute("SELECT edge_id FROM edges ORDER BY edge_id"))
        finally:
            connection.close()


class V2DependencyGraph:
    def __init__(self, root: Path | str, storage: Storage | None = None):
        self.root = Path(root); self.storage = storage or Storage(self.root)
        self.edges_dir = _VirtualEdgeDir(self.storage)
    def edge(self, *, stable_subject_id: str, parent_artifact_id: str, child_artifact_id: str, stage: str, edge_kind: str) -> DependencyGraphEdge:
        return DependencyGraphEdge(edge_id=dependency_edge_id_for(stable_subject_id, parent_artifact_id, child_artifact_id, stage, edge_kind), stable_subject_id=stable_subject_id, parent_artifact_id=parent_artifact_id, child_artifact_id=child_artifact_id, stage=stage, edge_kind=edge_kind)
    def put(self, edge: DependencyGraphEdge) -> str:
        if not isinstance(edge, DependencyGraphEdge): raise ArtifactError("edge must be DependencyGraphEdge")
        expected = dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind)
        if edge.edge_id != expected: raise ArtifactError("edge_id does not match canonical edge")
        self.storage.put_edge(edge.edge_id, edge.to_json().encode("utf-8")); return edge.edge_id
    def get(self, edge_id: str) -> DependencyGraphEdge:
        try:
            edge = DependencyGraphEdge.from_json(self.storage.edge_bytes(edge_id).decode("utf-8"))
            if edge.edge_id != edge_id or dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind) != edge_id: raise ArtifactError("graph edge hash mismatch")
            return edge
        except (StorageError, SchemaError, UnicodeDecodeError) as exc: raise ArtifactError(f"invalid or missing graph edge {edge_id}") from exc
    def bind_revision(self, revision_id: str, edge_ids: Sequence[str], *, allowed_artifact_ids: Sequence[str]) -> None:
        allowed=set(_ids(allowed_artifact_ids,"allowed_artifact_ids"))
        for eid in _ids(edge_ids,"edge_ids"):
            edge=self.get(eid)
            if edge.parent_artifact_id not in allowed or edge.child_artifact_id not in allowed: raise ArtifactError("selected graph edge endpoint is outside revision closure")
    def edge_ids(self, revision_id: str) -> tuple[str,...]:
        # Revision archives, not global SQLite history, are graph selection authority.
        return ()
    def edges(self, revision_id: str) -> tuple[DependencyGraphEdge,...]: return ()
    def forward(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge,...]: return ()
    def reverse(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge,...]: return ()
    traverse_forward=forward; traverse_reverse=reverse
    def descendants(self, revision_id: str, artifact_id: str) -> tuple[str,...]: return ()
    def ancestors(self, revision_id: str, artifact_id: str) -> tuple[str,...]: return ()


class V2SealedDependencyGraph:
    def __init__(self, revision: Path, revision_id: str, storage: Storage):
        self.revision = revision; self.revision_id = revision_id; self.storage = storage
    def _values(self):
        return self.storage.verify_zip(self.revision, revision_id=self.revision_id)
    def _require(self, revision_id: str) -> Mapping[str, bytes]:
        if revision_id != self.revision_id: raise ArtifactError("sealed graph belongs to a different revision")
        return self._values()
    def edge_ids(self, revision_id: str) -> tuple[str, ...]:
        values=self._require(revision_id)
        return tuple(sorted(name[6:-5] for name in values if name.startswith("edges/") and name.endswith(".json")))
    def get(self, edge_id: str) -> DependencyGraphEdge:
        try: edge=DependencyGraphEdge.from_json(self._values()[f"edges/{edge_id}.json"].decode("utf-8"))
        except (KeyError, SchemaError, UnicodeDecodeError) as exc: raise ArtifactError("invalid sealed graph edge") from exc
        if edge.edge_id != edge_id or dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind) != edge_id: raise ArtifactError("sealed graph edge hash mismatch")
        return edge
    def edges(self, revision_id: str) -> tuple[DependencyGraphEdge, ...]: return tuple(self.get(eid) for eid in self.edge_ids(revision_id))
    def forward(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]: return tuple(e for e in self.edges(revision_id) if e.parent_artifact_id == node_id or e.stable_subject_id == node_id)
    def reverse(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]: return tuple(e for e in self.edges(revision_id) if e.child_artifact_id == node_id or e.stable_subject_id == node_id)
    traverse_forward=forward; traverse_reverse=reverse
    def descendants(self, revision_id: str, artifact_id: str) -> tuple[str, ...]: return self._walk(revision_id, artifact_id, True)
    def ancestors(self, revision_id: str, artifact_id: str) -> tuple[str, ...]: return self._walk(revision_id, artifact_id, False)
    def _walk(self, revision_id: str, artifact_id: str, forward: bool) -> tuple[str, ...]:
        result=set(); queue=deque([artifact_id]); edges=self.edges(revision_id)
        while queue:
            node=queue.popleft()
            for edge in edges:
                source,target=(edge.parent_artifact_id,edge.child_artifact_id) if forward else (edge.child_artifact_id,edge.parent_artifact_id)
                if source == node and target not in result: result.add(target); queue.append(target)
        return tuple(sorted(result))


class V2RevisionStore:
    def __init__(self, root: Path | str, artifact_store: V2ArtifactStore | None = None, graph: V2DependencyGraph | None = None):
        self.root=Path(root); self.storage=Storage(self.root); self.revisions_dir=self.storage.revisions_dir
        self.artifacts=artifact_store or V2ArtifactStore(self.root); self.graph=graph or V2DependencyGraph(self.root, self.storage)
    def _revision_path(self, revision_id: str) -> Path: return self.revisions_dir / f"{revision_id}.zip"
    def seal_bundle(self, snapshot: RevisionSnapshot, provenance: Mapping[str, Any], epub: bytes | Path | str, *, render_input_artifact_id: str | None = None, edge_ids: Sequence[str] = (), epub_filename: str = "book.epub", expected_embedded_provenance: Mapping[str, Any] | None = None) -> Path:
        if not isinstance(snapshot, RevisionSnapshot): raise ArtifactError("snapshot must be RevisionSnapshot")
        if not isinstance(provenance, Mapping): raise ArtifactError("provenance must be an object")
        if Path(epub_filename).name != epub_filename or not epub_filename: raise ArtifactError("epub_filename must be a bare filename")
        artifacts, findings = self.artifacts.closure(snapshot.selected_artifact_ids, finding_ids=snapshot.selected_finding_ids)
        artifact_ids={a.artifact_id for a in artifacts}; finding_ids={f.finding_id for f in findings}
        attestations={aid:self.artifacts.get_semantic_attestation(aid) for aid in snapshot.selected_cache_attestation_ids}
        for body in attestations.values():
            envelope = next((a for a in artifacts if a.artifact_id == body.get("artifact_id")), None)
            if envelope is None or body.get("kind") != envelope.kind or tuple(body.get("dependency_ids", ())) != envelope.dependency_ids:
                raise ArtifactError("semantic attestation escapes selected artifact closure")
        edge_ids = _ids(edge_ids, "edge_ids")
        edges={}
        for eid in edge_ids:
            edge=self.graph.get(eid)
            if edge.parent_artifact_id not in artifact_ids or edge.child_artifact_id not in artifact_ids: raise ArtifactError("bundle graph edge escapes selected closure")
            edges[eid]=edge.to_json().encode("utf-8")
        epub_bytes = LegacyRevisionStore._epub_bytes(epub)
        if expected_embedded_provenance is not None and canonical_json_bytes(dict(expected_embedded_provenance)) != canonical_json_bytes(dict(provenance)):
            raise ArtifactError("expected embedded provenance must equal bundle provenance")
        # Empty EPUB is retained as a historical compatibility fixture.  Any
        # real EPUB is checked before publication and again during verification.
        if epub_bytes:
            LegacyRevisionStore._verify_embedded_provenance(epub_bytes, provenance)
        if render_input_artifact_id is not None:
            if render_input_artifact_id not in artifact_ids: raise ArtifactError("render input is outside selected artifact closure")
            render_hash = hashlib.sha256(canonical_json_bytes(next(a for a in artifacts if a.artifact_id == render_input_artifact_id).to_dict())).hexdigest()
        else:
            render_hash = None
        # FC3 keeps the v2 archive limited to the selected closure.  The
        # legacy provenance/EPUB arguments remain accepted for API
        # compatibility, but publication data is not an archive member.
        members={}
        members.update({f"records/{a.artifact_id}.json":a.to_json().encode("utf-8") for a in artifacts})
        members.update({f"findings/{f.finding_id}.json":f.to_json().encode("utf-8") for f in findings})
        members.update({f"edges/{eid}.json":data for eid,data in edges.items()})
        members.update({f"attestations/{aid}.json":canonical_json_bytes(body) for aid,body in attestations.items()})
        try:
            return self.storage.seal_revision(
                snapshot.revision_id, _v2_snapshot_bytes(snapshot, edge_ids), members,
            )
        except StorageError as exc: raise ArtifactError(str(exc)) from exc
    def verify_bundle(self, revision_id: str) -> RevisionSnapshot:
        path=self._revision_path(revision_id)
        try:
            values=self.storage.verify_revision(revision_id)
            snapshot, _selected_edge_ids = _v2_snapshot_from_bytes(values["snapshot.json"])
        except (StorageError, SchemaError, UnicodeDecodeError, ArtifactError) as exc: raise ArtifactError(f"invalid or missing sealed revision {revision_id}: {exc}") from exc
        records={}; findings={}
        for name,data in values.items():
            if name.startswith("records/"):
                try: record=ArtifactEnvelope.from_json(data.decode("utf-8"))
                except (SchemaError,UnicodeDecodeError) as exc: raise ArtifactError("invalid sealed record") from exc
                if name != f"records/{record.artifact_id}.json" or artifact_id_for(record.kind,record.payload,record.dependency_ids)!=record.artifact_id: raise ArtifactError("sealed record hash mismatch")
                records[record.artifact_id]=record
            elif name.startswith("findings/"):
                try: finding=Finding.from_json(data.decode("utf-8"))
                except (SchemaError,UnicodeDecodeError) as exc: raise ArtifactError("invalid sealed finding") from exc
                if name != f"findings/{finding.finding_id}.json": raise ArtifactError("sealed finding ID mismatch")
                findings[finding.finding_id]=finding
        if not set(snapshot.selected_artifact_ids).issubset(records) or not set(snapshot.selected_finding_ids).issubset(findings): raise ArtifactError("sealed revision omits selected closure")
        for record in records.values():
            if not set(record.dependency_ids).issubset(records) or not set(record.finding_ids).issubset(findings): raise ArtifactError("sealed record closure is incomplete")
        for name, data in values.items():
            if name.startswith("attestations/"):
                try: body=json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ArtifactError("invalid sealed attestation") from exc
                required={"attestation_id", "artifact_id", "kind", "semantic_key", "dependency_ids"}
                if not isinstance(body, Mapping) or set(body) != required or name != f"attestations/{body.get('attestation_id')}.json": raise ArtifactError("invalid sealed attestation")
                try: expected=V2ArtifactStore.semantic_attestation_id_for(artifact_id=body["artifact_id"], kind=body["kind"], semantic_key=body["semantic_key"], dependency_ids=body["dependency_ids"])
                except (ArtifactError, TypeError) as exc: raise ArtifactError("invalid sealed attestation") from exc
                envelope=records.get(body["artifact_id"])
                if expected != body["attestation_id"] or envelope is None or envelope.kind != body["kind"] or tuple(body["dependency_ids"]) != envelope.dependency_ids: raise ArtifactError("sealed attestation closure mismatch")
            elif name.startswith("edges/"):
                try: edge=DependencyGraphEdge.from_json(data.decode("utf-8"))
                except (SchemaError, UnicodeDecodeError) as exc: raise ArtifactError("invalid sealed edge") from exc
                if name != f"edges/{edge.edge_id}.json" or dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind) != edge.edge_id or edge.parent_artifact_id not in records or edge.child_artifact_id not in records: raise ArtifactError("sealed edge closure mismatch")
        return snapshot
    def snapshot(self, revision_id: str) -> RevisionSnapshot: return self.verify_bundle(revision_id)
    def activate(self, revision_id: str) -> None:
        self.verify_bundle(revision_id); self.storage.activate(revision_id)
    def active_snapshot(self) -> RevisionSnapshot | None:
        rid=self.storage.active_revision_id(); return None if rid is None else self.snapshot(rid)
    def selected_graph(self, revision_id: str):
        self.snapshot(revision_id); return V2SealedDependencyGraph(self._revision_path(revision_id), revision_id, self.storage)


class _ArtifactStoreFactoryMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, (LegacyArtifactStore, V2ArtifactStore))


class ArtifactStore(metaclass=_ArtifactStoreFactoryMeta):
    def __new__(cls, root: Path | str, *args: Any, **kwargs: Any):
        root=Path(root)
        if kwargs.pop("legacy", False) or _legacy_workspace(root): return LegacyReadOnlyArtifactStore(root)
        return V2ArtifactStore(root)

    semantic_attestation_id_for = staticmethod(V2ArtifactStore.semantic_attestation_id_for)


class _DependencyGraphFactoryMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, (LegacyDependencyGraph, LegacyReadOnlyGraph, V2DependencyGraph))


class DependencyGraph(metaclass=_DependencyGraphFactoryMeta):
    def __new__(cls, root: Path | str, *args: Any, **kwargs: Any):
        root=Path(root)
        if kwargs.pop("legacy", False) or _legacy_workspace(root): return LegacyReadOnlyGraph(root)
        return V2DependencyGraph(root, *args, **kwargs)


class RevisionStore:
    def __new__(cls, root: Path | str, *args: Any, **kwargs: Any):
        root=Path(root)
        if kwargs.pop("legacy", False) or _legacy_workspace(root): return LegacyReadOnlyRevisionStore(root, *args, **kwargs)
        return V2RevisionStore(root, *args, **kwargs)


# Short compatibility aliases for callers that prefer noun-first naming.
canonical_artifact_id = artifact_id_for
canonical_edge_id = dependency_edge_id_for

class LegacyReadOnlyArtifactStore(LegacyArtifactStore):
    """Read-only adapter for pre-v2 loose state."""
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.findings_dir = self.root / "findings"
        self.index_dir = self.root / "index"
        self.attestations_dir = self.root / "attestations"
        self.quarantine_dir = self.root / "quarantine"
    def _quarantine_invalid(self, *args: Any, **kwargs: Any) -> None:
        return None
    def _quarantine_attestation(self, *args: Any, **kwargs: Any) -> None:
        return None
    def put_finding(self, finding: Finding) -> str:
        raise ArtifactError("legacy workspace is read-only")
    def put(self, *args: Any, **kwargs: Any) -> ArtifactEnvelope:
        raise ArtifactError("legacy workspace is read-only")


class LegacyReadOnlyGraph:
    """Read-only legacy graph adapter; construction never creates directories."""
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.edges_dir = self.root / "graph" / "edges"
        self.revisions_dir = self.root / "graph" / "revisions"

    def edge(self, *, stable_subject_id: str, parent_artifact_id: str, child_artifact_id: str, stage: str, edge_kind: str) -> DependencyGraphEdge:
        return DependencyGraphEdge(edge_id=dependency_edge_id_for(stable_subject_id, parent_artifact_id, child_artifact_id, stage, edge_kind), stable_subject_id=stable_subject_id, parent_artifact_id=parent_artifact_id, child_artifact_id=child_artifact_id, stage=stage, edge_kind=edge_kind)

    def put(self, edge: DependencyGraphEdge) -> str:
        raise ArtifactError("legacy workspace graph is read-only")

    def get(self, edge_id: str) -> DependencyGraphEdge:
        try:
            edge = DependencyGraphEdge.from_file(self.edges_dir / f"{edge_id}.json")
            expected = dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind)
            if edge.edge_id != edge_id or expected != edge_id:
                raise ArtifactError("legacy graph edge hash mismatch")
            return edge
        except (OSError, SchemaError) as exc:
            raise ArtifactError(f"invalid or missing legacy graph edge {edge_id}") from exc

    def edge_ids(self, revision_id: str) -> tuple[str, ...]:
        try:
            body = _read_json(self.revisions_dir / f"{revision_id}.json")
            if body.get("revision_id") != revision_id:
                raise ArtifactError("legacy graph revision mismatch")
            return _ids(body.get("edge_ids", ()), "edge_ids")
        except (OSError, ArtifactError) as exc:
            raise ArtifactError("invalid legacy graph revision") from exc

    def edges(self, revision_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(self.get(edge_id) for edge_id in self.edge_ids(revision_id))

    def forward(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id) if edge.parent_artifact_id == node_id or edge.stable_subject_id == node_id)

    def reverse(self, revision_id: str, node_id: str) -> tuple[DependencyGraphEdge, ...]:
        return tuple(edge for edge in self.edges(revision_id) if edge.child_artifact_id == node_id or edge.stable_subject_id == node_id)

    traverse_forward = forward
    traverse_reverse = reverse


class LegacyReadOnlyRevisionStore(LegacyRevisionStore):
    def __init__(self, root: Path | str, artifact_store: Any = None, graph: Any = None):
        self.root = Path(root)
        self.revisions_dir = self.root / "revisions"
        self.artifacts = artifact_store or LegacyReadOnlyArtifactStore(self.root)
        self.graph = graph or LegacyReadOnlyGraph(self.root)
    def seal_bundle(self, *args: Any, **kwargs: Any) -> Path:
        raise ArtifactError("legacy workspace is read-only")
    def activate(self, revision_id: str) -> None:
        raise ArtifactError("legacy workspace is read-only")
