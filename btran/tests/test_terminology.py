"""Tests for deterministic terminology consolidation and page glossary slices."""

from __future__ import annotations

import json

import pytest

from btran.schema import SourceBlock, TermMention, TerminologyEntry
from btran.terminology import (
    HARD_TOKEN_CAP,
    batch_term_groups,
    cache_identity_with_glossary,
    consolidate_terminology,
    freeze_terminology,
    group_term_mentions,
    shard_terminology_map,
    slice_for_page,
)


def test_grouping_normalizes_identical_forms_and_preserves_provenance():
    groups = group_term_mentions(
        [
            TermMention(term="  Magic   Sword ", block_id="p2-b3"),
            TermMention(term="magic sword", block_id="p1-b1"),
            TermMention(term="MAGIC SWORD", block_id="p1-b1"),
        ]
    )

    assert len(groups) == 1
    assert groups[0].normalized_term == "magic sword"
    assert groups[0].forms == ("MAGIC SWORD", "Magic Sword", "magic sword")
    assert groups[0].provenance == ("p1-b1", "p2-b3")


def test_batching_respects_requested_token_budget_and_rejects_oversized_budget():
    groups = group_term_mentions(
        [TermMention(term=f"term-{number}-with-text", block_id=f"p1-b{number}") for number in range(8)]
    )

    batches = batch_term_groups(groups, token_budget=12)

    assert len(batches) > 1
    assert all(batch.token_count <= 12 for batch in batches)
    with pytest.raises(ValueError, match="120000"):
        batch_term_groups(groups, token_budget=120_001)
    assert HARD_TOKEN_CAP == 200_000


def test_recursive_consolidation_uses_text_only_calls_and_returns_valid_frozen_map():
    mentions = [
        TermMention(term=f"term-{number}", block_id=f"p1-b{number}")
        for number in range(8)
    ]
    prompts: list[str] = []

    def pi_call(prompt: str) -> str:
        prompts.append(prompt)
        request = json.loads(prompt.split("\n", 1)[1])
        source_terms = sorted(
            {
                source_term
                for item in request["items"]
                for source_term in item["source_terms"]
            }
        )
        return json.dumps(
            {
                "entries": [
                    {
                        "concept_id": "concept-" + "-".join(source_terms),
                        "source_terms": source_terms,
                        "target_term": "translated",
                        "provenance": sorted(
                            {
                                block_id
                                for item in request["items"]
                                for block_id in item["provenance"]
                            }
                        ),
                        "confidence": 0.9,
                    }
                ]
            }
        )

    glossary = consolidate_terminology(
        mentions,
        source_lang="en",
        target_lang="fr",
        pi_call=pi_call,
        token_budget=20,
        max_rounds=8,
    )

    assert len(prompts) > 1
    assert all(isinstance(prompt, str) and "@" not in prompt for prompt in prompts)
    assert glossary.version == "1"
    assert glossary.hash
    assert glossary.entries[0].target_term == "translated"
    assert glossary.entries[0].source_terms == [f"term-{number}" for number in range(8)]


def test_stable_sharding_builds_alias_index_for_oversized_map():
    glossary = freeze_terminology(
        [
            TerminologyEntry(
                concept_id=f"c{number}",
                source_terms=[f"Source {number}", f"Alias {number}"],
                target_term=f"Target {number}",
                provenance=[f"p{number}"],
                confidence=1.0,
            )
            for number in range(4)
        ],
        source_lang="en",
        target_lang="fr",
    )

    sharded = shard_terminology_map(glossary, token_budget=25)

    assert len(sharded.shards) > 1
    assert sharded.alias_index["alias 2"] == ("c2",)
    assert [entry.concept_id for shard in sharded.shards for entry in shard.entries] == [
        "c0", "c1", "c2", "c3"
    ]


def test_page_slice_matches_source_terms_aliases_and_adjacent_boundaries_only():
    glossary = freeze_terminology(
        [
            TerminologyEntry(
                concept_id="sword",
                source_terms=["magic sword", "blade"],
                target_term="epee magique",
                provenance=["p1"],
                confidence=1.0,
            ),
            TerminologyEntry(
                concept_id="castle",
                source_terms=["castle"],
                target_term="chateau",
                provenance=["p1"],
                confidence=1.0,
            ),
            TerminologyEntry(
                concept_id="dragon",
                source_terms=["dragon"],
                target_term="dragon",
                provenance=["p1"],
                confidence=1.0,
            ),
        ],
        source_lang="en",
        target_lang="fr",
    )

    selected = slice_for_page(
        glossary,
        [SourceBlock(id="p2-b1", type="paragraph", text="The Blade glowed.", reading_order=1)],
        previous_boundary="They entered the castle.",
        next_boundary="",
    )

    assert [entry.concept_id for entry in selected] == ["castle", "sword"]


def test_freeze_hash_and_cache_identity_are_stable_for_same_input():
    entries = [
        TerminologyEntry(
            concept_id="hero",
            source_terms=["Hero", "champion"],
            target_term="heros",
            provenance=["p2", "p1"],
            confidence=0.8,
        )
    ]

    first = freeze_terminology(entries, source_lang="en", target_lang="fr")
    second = freeze_terminology(list(reversed(entries)), source_lang="en", target_lang="fr")

    assert first.hash == second.hash
    assert cache_identity_with_glossary("image-sha", first) == cache_identity_with_glossary(
        "image-sha", second.hash
    )
    assert cache_identity_with_glossary("image-sha", first) != cache_identity_with_glossary(
        "image-sha", "other-glossary"
    )
