"""Task-15 selective invalidation regressions.

These tests intentionally use sealed selected revisions, rather than mutable cache
history, as correction-planning authority.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore, DependencyGraph, RevisionStore
from btran.config import Config
from btran.corrections import (
    CorrectionStore,
    base_hash_for_artifact,
    correction_transition,
)
from btran.orchestrator import run
from btran.schema import PageExtraction, SourceBlock


def _config(tmp_path: Path) -> Config:
    return Config(
        input_dir=tmp_path / "input", output_epub=tmp_path / "book.epub",
        workspace=tmp_path / "work", target_lang="fr", timeout=1,
    )


def _extract(page_number: int, text: str) -> PageExtraction:
    return PageExtraction(
        page_number=page_number, image_path=f"page-{page_number}.png",
        sha256="a" * 64, phash="b" * 16, source_lang="en", model="test",
        blocks=[SourceBlock(f"block-{page_number}", "paragraph", text, 0)],
    )


def _terminology_response(prompt: str) -> str:
    request = json.loads(prompt.split("\n", 1)[1])
    return json.dumps({"entries": [
        {
            "concept_id": f"concept-{index}",
            "source_terms": item["source_terms"],
            "target_term": "fr-" + item["source_terms"][0],
            "provenance": item["provenance"],
            "confidence": 1.0,
        }
        for index, item in enumerate(request["items"])
    ]})


async def _translate(*args, **kwargs):
    """Deterministic leaf output contains exact selected terminology forms."""
    forms = [projection["target_form"] for projection in kwargs["projections"]]
    return " ".join(forms) or "fr:" + args[0].source_text


async def _baseline(tmp_path: Path, texts: tuple[str, ...]):
    config = _config(tmp_path)
    config.input_dir.mkdir()
    for index in range(len(texts)):
        Image.new("RGB", (32, 32), (index, index, index)).save(config.input_dir / f"{index:02}.png")

    async def fake_extract(*args, **kwargs):
        page_number = kwargs.get("page_number", args[4])
        return _extract(page_number, texts[page_number - 1])

    with patch("btran.source_extractor.extract_page", fake_extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=_terminology_response), \
         patch("btran.translator.translate_segment", _translate):
        result = await run(config)
    assert result.status == "completed"
    assert result.candidate_revision_id
    RevisionStore(config.workspace).activate(result.candidate_revision_id)
    return config, result


async def _execute(config: Config, texts: tuple[str, ...]):
    async def fake_extract(*args, **kwargs):
        page_number = kwargs.get("page_number", args[4])
        return _extract(page_number, texts[page_number - 1])

    with patch("btran.source_extractor.extract_page", fake_extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=_terminology_response), \
         patch("btran.translator.translate_segment", _translate):
        result = await run(config)
    assert result.status == "completed"
    assert result.candidate_revision_id
    return result


@pytest.mark.asyncio
async def test_activated_base_reuses_exact_selected_model_leaves_and_invalidates_context_only(tmp_path: Path):
    """No history lookup: sealed IDs hit; changed page fans out only to neighbors."""
    texts = ("alpha", "beta", "gamma", "delta")
    config, _ = await _baseline(tmp_path, texts)
    extract_calls: list[int] = []
    translated: list[str] = []

    async def extract(*args, **kwargs):
        page_number = kwargs.get("page_number", args[4])
        extract_calls.append(page_number)
        return _extract(page_number, texts[page_number - 1])

    async def translate(*args, **kwargs):
        translated.append(args[0].source_text)
        return await _translate(*args, **kwargs)

    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=_terminology_response), \
         patch("btran.translator.translate_segment", translate):
        reused = await run(config)
    assert reused.status == "completed"
    assert extract_calls == []
    assert translated == []
    assert {event.stage for event in reused.cache_events if event.outcome == "hit"} >= {"source_extraction", "translation"}

    # Same extracted text but different accepted source bytes changes exactly
    # its raw/effective source leaf. Translation sees focal + immediate context.
    Image.new("RGB", (32, 32), (99, 99, 99)).save(config.input_dir / "01.png")
    extract_calls.clear(); translated.clear()
    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=_terminology_response), \
         patch("btran.translator.translate_segment", translate):
        changed = await run(config)
    assert changed.status == "completed"
    assert extract_calls == [2]
    assert translated == ["alpha", "beta", "gamma"]
    assert all(event.subject_id != "delta" for event in changed.cache_events
               if event.stage == "translation" and event.outcome == "miss")


@pytest.mark.asyncio
async def test_changed_model_recomputes_identical_leaves_and_seals_key_attestations(tmp_path: Path):
    """A new semantic key invokes leaves even when their canonical IDs collide."""
    texts = ("alpha",)
    config, _ = await _baseline(tmp_path, texts)
    config.model = "same-output-model-v2"
    extraction_calls: list[int] = []
    consolidation_calls: list[str] = []
    translation_calls: list[str] = []

    async def extract(*args, **kwargs):
        page_number = kwargs.get("page_number", args[4])
        extraction_calls.append(page_number)
        return _extract(page_number, texts[page_number - 1])

    def consolidate(prompt: str) -> str:
        consolidation_calls.append(prompt)
        return _terminology_response(prompt)

    async def translate(*args, **kwargs):
        translation_calls.append(args[0].source_text)
        return await _translate(*args, **kwargs)

    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidate), \
         patch("btran.translator.translate_segment", translate):
        rerun = await run(config)
    assert rerun.status == "completed"
    assert extraction_calls == [1]
    assert consolidation_calls
    assert translation_calls == ["alpha"]
    snapshot = RevisionStore(config.workspace).snapshot(rerun.candidate_revision_id)
    assert snapshot.selected_cache_attestation_ids
    assert all((RevisionStore(config.workspace).revisions_dir / snapshot.revision_id / "attestations" / f"{item}.json").exists()
               for item in snapshot.selected_cache_attestation_ids)

    RevisionStore(config.workspace).activate(rerun.candidate_revision_id)
    extraction_calls.clear(); consolidation_calls.clear(); translation_calls.clear()
    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidate), \
         patch("btran.translator.translate_segment", translate):
        unchanged = await run(config)
    assert unchanged.status == "completed"
    assert extraction_calls == []
    assert consolidation_calls == []
    assert translation_calls == []



@pytest.mark.asyncio
async def test_unactivated_changed_key_attestations_never_authorize_old_active_reuse(tmp_path: Path):
    """Candidate-only attestations stay invisible until that candidate activates."""
    texts = ("alpha",)
    config, _ = await _baseline(tmp_path, texts)
    config.model = "same-output-model-v2"
    extraction_calls: list[int] = []
    consolidation_calls: list[str] = []
    translation_calls: list[str] = []

    async def extract(*args, **kwargs):
        page_number = kwargs.get("page_number", args[4])
        extraction_calls.append(page_number)
        return _extract(page_number, texts[page_number - 1])

    def consolidate(prompt: str) -> str:
        consolidation_calls.append(prompt)
        return _terminology_response(prompt)

    async def translate(*args, **kwargs):
        translation_calls.append(args[0].source_text)
        return await _translate(*args, **kwargs)

    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidate), \
         patch("btran.translator.translate_segment", translate):
        unactivated = await run(config)
    assert extraction_calls == [1]
    assert consolidation_calls and translation_calls == ["alpha"]

    extraction_calls.clear(); consolidation_calls.clear(); translation_calls.clear()
    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidate), \
         patch("btran.translator.translate_segment", translate):
        against_old_active = await run(config)
    assert against_old_active.status == "completed"
    assert extraction_calls == [1]
    assert consolidation_calls and translation_calls == ["alpha"]

    RevisionStore(config.workspace).activate(unactivated.candidate_revision_id)
    extraction_calls.clear(); consolidation_calls.clear(); translation_calls.clear()
    with patch("btran.source_extractor.extract_page", extract), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=consolidate), \
         patch("btran.translator.translate_segment", translate):
        unchanged = await run(config)
    assert unchanged.status == "completed"
    assert extraction_calls == []
    assert consolidation_calls == []
    assert translation_calls == []


def _selected(store: ArtifactStore, revisions: RevisionStore, revision_id: str):
    snapshot = revisions.snapshot(revision_id)
    artifacts, _ = store.closure(snapshot.selected_artifact_ids, finding_ids=snapshot.selected_finding_ids)
    return tuple(artifacts)


def _by_segment(artifacts, kind: str) -> dict[str, object]:
    return {
        artifact.payload["segment_id"]: artifact
        for artifact in artifacts
        if artifact.kind == kind and isinstance(artifact.payload.get("segment_id"), str)
    }


def _by_source_text(artifacts, kind: str) -> dict[str, object]:
    values = [artifact for artifact in artifacts if artifact.kind == kind]
    result = {artifact.payload.get("source_text"): artifact for artifact in values}
    assert len(result) == len(values), f"{kind} source text must be unique in this fixture"
    return result


def _translations_for_sources(artifacts, sources: dict[str, object]) -> dict[str, object]:
    text_by_source_artifact = {artifact.artifact_id: text for text, artifact in sources.items()}
    return {
        text_by_source_artifact[artifact.payload["source_artifact_id"]]: artifact
        for artifact in artifacts if artifact.kind == "TranslationArtifact"
    }


def _targets_for_sources(artifacts, sources: dict[str, object]) -> dict[str, object]:
    translations = _translations_for_sources(artifacts, sources)
    by_translation_id = {artifact.artifact_id: text for text, artifact in translations.items()}
    return {
        by_translation_id[artifact.payload["translation_artifact_id"]]: artifact
        for artifact in artifacts if artifact.kind == "EffectiveTargetSegment"
    }


def _by_form(artifacts, kind: str, field: str) -> dict[str, object]:
    return {
        artifact.payload[field]: artifact
        for artifact in artifacts
        if artifact.kind == kind and isinstance(artifact.payload.get(field), str)
    }


def _current_source_pages(store: ArtifactStore, revisions: RevisionStore, revision_id: str) -> dict[str, object]:
    snapshot = revisions.snapshot(revision_id)
    pages = [store.get(artifact_id) for artifact_id in snapshot.selected_artifact_ids
             if store.get(artifact_id).kind == "EffectiveSourcePage"]
    return {
        store.get(page.dependency_ids[0]).payload["segment_id"]: page
        for page in pages
    }


def _current_target_maps(store: ArtifactStore, revisions: RevisionStore, revision_id: str):
    """Return current target leaves, never retained predecessor closure leaves."""
    snapshot = revisions.snapshot(revision_id)
    pages = [store.get(artifact_id) for artifact_id in snapshot.selected_artifact_ids
             if store.get(artifact_id).kind == "EffectiveTargetPage"]
    targets = {
        segment.payload["segment_id"]: segment
        for page in pages for child_id in page.dependency_ids
        for segment in (store.get(child_id),)
    }
    translations = {
        segment_id: store.get(segment.payload["translation_artifact_id"])
        for segment_id, segment in targets.items()
    }
    return translations, targets


def _entry_keys(impact, category: str) -> set[tuple[str, str, str]]:
    return {(item["stage"], item["subject_id"], item["base_artifact_id"])
            for item in getattr(impact, category)}


@pytest.mark.asyncio
async def test_source_correction_uses_sealed_reverse_closure_and_execution_plan(tmp_path: Path):
    """Source edit changes exact descendants, not unrelated selected leaves."""
    texts = ("alpha", "beta", "gamma", "delta")
    config, baseline = await _baseline(tmp_path, texts)
    workspace = config.workspace
    assert workspace is not None
    store, revisions = ArtifactStore(workspace), RevisionStore(workspace)
    base_id = baseline.candidate_revision_id
    base_artifacts = _selected(store, revisions, base_id)
    raw = _by_source_text(base_artifacts, "RawSourceSegment")
    old_source = _by_source_text(base_artifacts, "EffectiveSourceSegment")
    assert set(raw) == set(texts) == set(old_source)

    # This edge exists in mutable workspace history only. Planner must ignore it.
    graph = DependencyGraph(workspace)
    probe = store.put("UnselectedProbe", {"segment_id": "probe"}, semantic_key="unselected-probe")
    graph.put(graph.edge(stable_subject_id="probe", parent_artifact_id=raw["beta"].artifact_id,
                         child_artifact_id=probe.artifact_id, stage="probe", edge_kind="must_not_select"))
    selected_graph = revisions.selected_graph(base_id)
    sealed_descendants = set(selected_graph.descendants(base_id, raw["beta"].artifact_id))
    assert probe.artifact_id not in sealed_descendants

    corrections = CorrectionStore(workspace)
    payload = {
        "kind": "source_text", "applies_to_revision_id": base_id,
        "scope": {"segment_id": raw["beta"].payload["segment_id"]},
        "base": {"artifact_id": raw["beta"].artifact_id, "sha256": base_hash_for_artifact(raw["beta"])},
        "replacement": "beta-fixed",
    }
    correction_set, correction_impact = correction_transition(
        corrections, revisions, event_kind="apply", payload=payload, revision_id=base_id,
    )
    assert correction_impact.regenerated == ()
    assert probe.artifact_id not in {item["base_artifact_id"] for item in correction_impact.projected_universe}
    reverse_virtual = {
        item["base_artifact_id"] for item in correction_impact.projected_universe
        if item["stage"] == "correction_reverse_descendant"
    }
    assert reverse_virtual == sealed_descendants
    direct_virtual = [item for item in correction_impact.projected_universe
                      if item["stage"] == "correction_direct_overlay"]
    assert direct_virtual == [{"stage": "correction_direct_overlay", "subject_id": correction_set.active_correction_ids[0],
                               "base_artifact_id": raw["beta"].artifact_id}]
    # Four categories alone partition exactly persisted plan universe.
    categories = ("affected", "unaffected", "ambiguous", "protected")
    category_keys = [_entry_keys(correction_impact, category) for category in categories]
    assert not any(left & right for index, left in enumerate(category_keys) for right in category_keys[index + 1:])
    assert set.union(*category_keys) == _entry_keys(correction_impact, "projected_universe")

    config.base_revision = base_id
    config.correction_set = correction_set.set_id
    executed = await _execute(config, texts)
    candidate_artifacts = _selected(store, revisions, executed.candidate_revision_id)

    new_source = _by_source_text(candidate_artifacts, "EffectiveSourceSegment")
    old_source_pages = _current_source_pages(store, revisions, base_id)
    new_source_pages = _current_source_pages(store, revisions, executed.candidate_revision_id)
    old_shards = _by_segment(base_artifacts, "OccurrenceEvidenceShard")
    new_shards = _by_segment(candidate_artifacts, "OccurrenceEvidenceShard")
    assert new_source["beta-fixed"].artifact_id != old_source["beta"].artifact_id
    assert new_shards[new_source["beta-fixed"].payload["segment_id"]].artifact_id != old_shards[old_source["beta"].payload["segment_id"]].artifact_id
    for text in ("alpha", "gamma", "delta"):
        segment_id = old_source[text].payload["segment_id"]
        assert new_source[text].artifact_id == old_source[text].artifact_id
        assert new_source_pages[segment_id].artifact_id == old_source_pages[segment_id].artifact_id
        assert new_shards[segment_id].artifact_id == old_shards[segment_id].artifact_id
    beta_segment_id = raw["beta"].payload["segment_id"]
    assert new_source_pages[beta_segment_id].artifact_id != old_source_pages[beta_segment_id].artifact_id
    assert any(item.kind == "SourceTextOverlay" and item.payload["segment_id"] == beta_segment_id
               for item in candidate_artifacts)

    old_memberships = _by_form(base_artifacts, "ConceptMembership", "canonical_source_form")
    new_memberships = _by_form(candidate_artifacts, "ConceptMembership", "canonical_source_form")
    old_projections = _by_form(base_artifacts, "ConceptProjection", "concept_id")
    new_projections = _by_form(candidate_artifacts, "ConceptProjection", "concept_id")
    for text in ("alpha", "gamma", "delta"):
        assert new_memberships[text].artifact_id == old_memberships[text].artifact_id
        concept_id = old_memberships[text].payload["concept_id"]
        assert new_projections[concept_id].artifact_id == old_projections[concept_id].artifact_id
    assert "beta" not in new_memberships and "beta-fixed" in new_memberships

    old_translations, old_targets = _current_target_maps(store, revisions, base_id)
    new_translations, new_targets = _current_target_maps(store, revisions, executed.candidate_revision_id)
    for text in ("alpha", "beta", "gamma"):
        segment_id = old_source[text].payload["segment_id"]
        assert new_translations[segment_id].artifact_id != old_translations[segment_id].artifact_id
        assert new_targets[segment_id].artifact_id != old_targets[segment_id].artifact_id
    delta_segment_id = old_source["delta"].payload["segment_id"]
    assert new_translations[delta_segment_id].artifact_id == old_translations[delta_segment_id].artifact_id
    assert new_targets[delta_segment_id].artifact_id == old_targets[delta_segment_id].artifact_id
    for kind in ("ReconciliationArtifact", "ValidationArtifact", "SealedRenderInput", "RenderedEpub"):
        assert {item.artifact_id for item in candidate_artifacts if item.kind == kind} != {
            item.artifact_id for item in base_artifacts if item.kind == kind
        }

    execution = corrections.get_impact(correction_impact.projection_plan_id, phase="execution")
    planned = corrections.get_impact(correction_impact.projection_plan_id)
    assert execution.projection_plan_id == correction_impact.projection_plan_id == planned.projection_plan_id
    assert execution.projected_universe == planned.projected_universe
    for category in categories:
        assert getattr(execution, category) == getattr(planned, category)


@pytest.mark.asyncio
async def test_multiple_corrections_keep_original_execution_plan_ids_and_provenance(tmp_path: Path):
    """Later correction set must not readdress earlier correction's plan."""
    texts = ("alpha", "beta", "gamma")
    config, baseline = await _baseline(tmp_path, texts)
    assert config.workspace is not None
    workspace = config.workspace
    store, revisions, corrections = ArtifactStore(workspace), RevisionStore(workspace), CorrectionStore(workspace)
    base_artifacts = _selected(store, revisions, baseline.candidate_revision_id)
    raw = _by_source_text(base_artifacts, "RawSourceSegment")

    def source_payload(text: str, replacement: str):
        artifact = raw[text]
        return {
            "kind": "source_text", "applies_to_revision_id": baseline.candidate_revision_id,
            "scope": {"segment_id": artifact.payload["segment_id"]},
            "base": {"artifact_id": artifact.artifact_id, "sha256": base_hash_for_artifact(artifact)},
            "replacement": replacement,
        }

    first_set, first_plan = correction_transition(
        corrections, revisions, event_kind="apply", payload=source_payload("beta", "beta-fixed"),
    )
    second_set, second_plan = correction_transition(
        corrections, revisions, event_kind="apply", payload=source_payload("gamma", "gamma-fixed"),
    )
    assert first_plan.correction_set_id == first_set.set_id
    assert second_plan.correction_set_id == second_set.set_id
    assert first_plan.projection_plan_id != second_plan.projection_plan_id

    config.base_revision = baseline.candidate_revision_id
    config.correction_set = second_set.set_id
    executed = await _execute(config, texts)
    expected_ids = tuple(sorted((first_plan.projection_plan_id, second_plan.projection_plan_id)))
    assert executed.report.correction_execution_projection_plan_ids == expected_ids
    for plan in (first_plan, second_plan):
        execution = corrections.get_impact(plan.projection_plan_id, phase="execution")
        assert execution.correction_id == plan.correction_id
        assert execution.correction_set_id == plan.correction_set_id
        assert execution.projected_universe == plan.projected_universe

    with zipfile.ZipFile(config.output_epub) as epub:
        provenance = json.loads(epub.read("META-INF/btran-provenance.json"))
    assert provenance["correction_execution_projection_plan_ids"] == list(expected_ids)
    assert provenance["correction_execution_impact_records"] == [
        f"corrections/impacts/{plan_id}.execution.json" for plan_id in expected_ids
    ]
    assert all((workspace / record).is_file() for record in provenance["correction_execution_impact_records"])


@pytest.mark.asyncio
async def test_invalid_one_correction_plan_keeps_other_execution_impact_and_reports_finding(tmp_path: Path):
    """A bad correction plan is local failure, not all-impact failure."""
    texts = ("alpha", "beta", "gamma")
    config, baseline = await _baseline(tmp_path, texts)
    assert config.workspace is not None
    workspace = config.workspace
    store, revisions, corrections = ArtifactStore(workspace), RevisionStore(workspace), CorrectionStore(workspace)
    raw = _by_source_text(_selected(store, revisions, baseline.candidate_revision_id), "RawSourceSegment")

    def apply(text: str, replacement: str):
        artifact = raw[text]
        return correction_transition(corrections, revisions, event_kind="apply", payload={
            "kind": "source_text", "applies_to_revision_id": baseline.candidate_revision_id,
            "scope": {"segment_id": artifact.payload["segment_id"]},
            "base": {"artifact_id": artifact.artifact_id, "sha256": base_hash_for_artifact(artifact)},
            "replacement": replacement,
        })

    _, missing_plan = apply("beta", "beta-fixed")
    active_set, valid_plan = apply("gamma", "gamma-fixed")
    (workspace / "corrections" / "impacts" / f"{missing_plan.projection_plan_id}.json").write_text("{", encoding="utf-8")
    config.base_revision = baseline.candidate_revision_id
    config.correction_set = active_set.set_id
    executed = await _execute(config, texts)

    assert executed.report.correction_execution_projection_plan_ids == (valid_plan.projection_plan_id,)
    assert corrections.get_impact(valid_plan.projection_plan_id, phase="execution").projection_plan_id == valid_plan.projection_plan_id
    findings = [store.get_finding(finding_id) for finding_id in executed.report.content_finding_ids]
    missing = [finding for finding in findings if finding.kind == "execution_impact_unavailable"]
    assert len(missing) == 1
    assert missing[0].subject_refs == (missing_plan.correction_id,)


@pytest.mark.asyncio
async def test_selected_graph_has_separate_preceding_and_following_context_edges(tmp_path: Path):
    """Middle segment reaches two neighbors through independently asserted edges."""
    config, baseline = await _baseline(tmp_path, ("alpha", "beta", "gamma", "delta"))
    assert config.workspace is not None
    store, revisions = ArtifactStore(config.workspace), RevisionStore(config.workspace)
    artifacts = _selected(store, revisions, baseline.candidate_revision_id)
    sources = _by_source_text(artifacts, "EffectiveSourceSegment")
    translations, _ = _current_target_maps(store, revisions, baseline.candidate_revision_id)
    graph = revisions.selected_graph(baseline.candidate_revision_id)
    context_edges = [
        edge for edge in graph.edges(baseline.candidate_revision_id)
        if edge.edge_kind == "translation_context_to_translation"
        and edge.parent_artifact_id == sources["beta"].artifact_id
    ]
    # beta is alpha's following context and gamma's preceding context.
    assert {(edge.stable_subject_id, edge.child_artifact_id) for edge in context_edges} == {
        (sources["alpha"].payload["segment_id"], translations[sources["alpha"].payload["segment_id"]].artifact_id),
        (sources["gamma"].payload["segment_id"], translations[sources["gamma"].payload["segment_id"]].artifact_id),
    }
    descendants = set(graph.descendants(baseline.candidate_revision_id, sources["beta"].artifact_id))
    assert translations[sources["alpha"].payload["segment_id"]].artifact_id in descendants  # following-context consumer
    assert translations[sources["gamma"].payload["segment_id"]].artifact_id in descendants  # preceding-context consumer
    assert translations[sources["delta"].payload["segment_id"]].artifact_id not in descendants


@pytest.mark.asyncio
@pytest.mark.parametrize("selector_kind", ["all_concept_occurrences", "occurrence_ids"])
async def test_terminology_scope_fanout_and_local_protection(tmp_path: Path, selector_kind: str):
    """All scope fans out; verified subset keeps distant same-concept unit unchanged."""
    texts = ("magic", "magic", "magic", "stone")
    config, baseline = await _baseline(tmp_path, texts)
    assert config.workspace is not None
    workspace, store, revisions = config.workspace, ArtifactStore(config.workspace), RevisionStore(config.workspace)
    base_id = baseline.candidate_revision_id
    artifacts = _selected(store, revisions, base_id)
    memberships = _by_form(artifacts, "ConceptMembership", "canonical_source_form")
    magic = memberships["magic"]
    projections = _by_form(artifacts, "ConceptProjection", "concept_id")
    projection = projections[magic.payload["concept_id"]]
    raw = _by_segment(artifacts, "RawSourceSegment")
    sources = _by_segment(artifacts, "EffectiveSourceSegment")
    old_translations, _ = _current_target_maps(store, revisions, base_id)
    magic_segment_ids = {segment_id for segment_id, artifact in raw.items()
                         if artifact.payload["source_text"] == "magic"}
    stone_segment_id = next(segment_id for segment_id, artifact in raw.items()
                            if artifact.payload["source_text"] == "stone")
    occurrence_to_segment = {
        occurrence["occurrence_id"]: occurrence["segment_id"]
        for artifact in artifacts if artifact.kind == "OccurrenceEvidenceShard"
        for occurrence in artifact.payload["occurrences"]
    }
    base_graph = revisions.selected_graph(base_id)
    context_children = {
        segment_id: {
            edge.stable_subject_id for edge in base_graph.edges(base_id)
            if edge.edge_kind == "translation_context_to_translation"
            and edge.parent_artifact_id == sources[segment_id].artifact_id
        }
        for segment_id in magic_segment_ids
    }
    # Select an endpoint occurrence. Its one immediate same-concept context
    # leaves a third occurrence outside subset fanout.
    selected_segment_id = next(segment_id for segment_id, children in context_children.items()
                               if len(children & magic_segment_ids) == 1)
    selected_occurrence_id = next(occurrence_id for occurrence_id in magic.payload["occurrence_ids"]
                                  if occurrence_to_segment[occurrence_id] == selected_segment_id)
    corrections = CorrectionStore(workspace)

    # A local target overlay is stronger than broad terminology fanout.
    local_payload = {
        "kind": "target_segment", "applies_to_revision_id": base_id,
        "scope": {"segment_id": stone_segment_id,
                  "expected_target_text": old_translations[stone_segment_id].payload["translated_text"]},
        "base": {"artifact_id": old_translations[stone_segment_id].artifact_id,
                 "sha256": base_hash_for_artifact(old_translations[stone_segment_id])},
        "replacement": "fixed-stone",
    }
    correction_transition(corrections, revisions, event_kind="apply", payload=local_payload, revision_id=base_id)
    selector = {"kind": selector_kind}
    if selector_kind == "occurrence_ids":
        selector["ids"] = [selected_occurrence_id]
    terminology_payload = {
        "kind": "terminology", "applies_to_revision_id": base_id,
        "scope": {"concept_id": magic.payload["concept_id"], "selector": selector},
        "base": {
            "projection": {"artifact_id": projection.artifact_id, "sha256": base_hash_for_artifact(projection)},
            "membership": {"artifact_id": magic.artifact_id, "sha256": base_hash_for_artifact(magic)},
        },
        "replacement": "arcane",
    }
    correction_set, impact = correction_transition(
        corrections, revisions, event_kind="apply", payload=terminology_payload, revision_id=base_id,
    )
    protected_ids = {item["base_artifact_id"] for item in impact.protected}
    assert old_translations[stone_segment_id].artifact_id in protected_ids

    config.base_revision, config.correction_set = base_id, correction_set.set_id
    executed = await _execute(config, texts)
    candidate = _selected(store, revisions, executed.candidate_revision_id)
    new_translations, _ = _current_target_maps(store, revisions, executed.candidate_revision_id)
    corrected = [item for item in candidate if item.kind == "ConceptProjection"
                 and item.payload.get("correction_id")]
    assert len(corrected) == 1
    assert old_translations[stone_segment_id].artifact_id == new_translations[stone_segment_id].artifact_id
    if selector_kind == "all_concept_occurrences":
        assert corrected[0].payload["selector_occurrence_ids"] == magic.payload["occurrence_ids"]
        assert all(new_translations[segment_id].artifact_id != old_translations[segment_id].artifact_id
                   for segment_id in magic_segment_ids)
    else:
        assert corrected[0].payload["selector_occurrence_ids"] == [selected_occurrence_id]
        context_consumers = {
            edge.stable_subject_id for edge in revisions.selected_graph(base_id).edges(base_id)
            if edge.edge_kind == "translation_context_to_translation"
            and edge.parent_artifact_id == sources[selected_segment_id].artifact_id
        }
        expected_changed = ({selected_segment_id, *context_consumers} & magic_segment_ids)
        distant_magic = magic_segment_ids - expected_changed
        assert distant_magic  # corpus has a same-concept unit beyond one hop.
        assert all(new_translations[segment_id].artifact_id != old_translations[segment_id].artifact_id
                   for segment_id in expected_changed)
        assert all(new_translations[segment_id].artifact_id == old_translations[segment_id].artifact_id
                   for segment_id in distant_magic)
