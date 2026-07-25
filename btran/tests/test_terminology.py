"""Tests for deterministic terminology consolidation and page glossary slices."""

from __future__ import annotations

import json

import pytest

from btran.schema import SourceBlock, TermMention, TerminologyEntry
from btran.terminology import (
    HARD_TOKEN_CAP,
    PiConsolidationError,
    batch_term_groups,
    cache_identity_with_glossary,
    consolidate_terminology,
    estimate_tokens,
    freeze_terminology,
    group_term_mentions,
    make_pi_consolidation_call,
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
    assert groups[0].forms == ("  Magic   Sword ", "MAGIC SWORD", "magic sword")
    assert groups[0].provenance == ("p1-b1", "p2-b3")


def test_grouping_preserves_original_source_spelling_while_normalizing_its_key():
    original = "  \ufb00oo  "
    groups = group_term_mentions([TermMention(term=original, block_id="p1-b1")])

    assert groups[0].normalized_term == "ffoo"
    assert groups[0].forms == (original,)
    assert groups[0].provenance == ("p1-b1",)


def test_token_measurement_is_a_conservative_utf8_bound():
    assert estimate_tokens("a b c") == 5
    assert estimate_tokens("猫") == 3


def test_batching_respects_requested_token_budget_and_rejects_oversized_budget():
    groups = group_term_mentions(
        [TermMention(term=f"term-{number}-with-text", block_id=f"p1-b{number}") for number in range(8)]
    )

    batches = batch_term_groups(groups, token_budget=600)

    assert len(batches) > 1
    assert all(batch.token_count <= 600 for batch in batches)
    with pytest.raises(ValueError, match="120000"):
        batch_term_groups(groups, token_budget=120_001)
    assert HARD_TOKEN_CAP == 200_000


def test_batching_rejects_a_single_group_that_cannot_fit_the_configured_input_budget():
    groups = group_term_mentions([TermMention(term="x" * 80, block_id="p1-b1")])

    with pytest.raises(ValueError, match="configured token budget"):
        batch_term_groups(groups, token_budget=10)


def test_consolidation_uses_text_only_calls_and_returns_valid_frozen_map():
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
        max_rounds=8,
    )

    assert len(prompts) == 1
    assert all(isinstance(prompt, str) and "@" not in prompt for prompt in prompts)
    assert glossary.version == "1"
    assert glossary.hash
    assert glossary.entries[0].target_term == "translated"
    assert glossary.entries[0].source_terms == [f"term-{number}" for number in range(8)]


def test_consolidation_rejects_untrusted_entries_that_drop_input_provenance():
    def pi_call(_: str) -> str:
        return json.dumps(
            {
                "entries": [
                    {
                        "concept_id": "sword",
                        "source_terms": ["sword"],
                        "target_term": "epee",
                        "provenance": ["invented-block"],
                        "confidence": 0.9,
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="provenance"):
        consolidate_terminology(
            [TermMention(term="sword", block_id="p1-b1")],
            source_lang="en",
            target_lang="fr",
            pi_call=pi_call,
        )


def test_consolidation_keeps_same_source_context_variants_and_low_confidence_conflicts():
    def pi_call(_: str) -> str:
        return json.dumps(
            {
                "entries": [
                    {
                        "concept_id": "bank-river",
                        "source_terms": ["bank"],
                        "target_term": "rive",
                        "provenance": ["p1-b1"],
                        "confidence": 0.0,
                        "notes": "river context",
                    },
                    {
                        "concept_id": "bank-finance",
                        "source_terms": ["bank"],
                        "target_term": "banque",
                        "provenance": ["p1-b1"],
                        "confidence": 0.2,
                        "notes": "financial context",
                    },
                ]
            }
        )

    glossary = consolidate_terminology(
        [TermMention(term="bank", block_id="p1-b1")],
        source_lang="en",
        target_lang="fr",
        pi_call=pi_call,
    )

    assert [(entry.target_term, entry.confidence) for entry in glossary.entries] == [
        ("banque", 0.2),
        ("rive", 0.0),
    ]


def test_consolidation_fails_when_multiple_batches_do_not_reduce_without_target_term_merging():
    mentions = [
        TermMention(term="first-long-term", block_id="p1-b1"),
        TermMention(term="second-long-term", block_id="p1-b2"),
    ]

    def pi_call(prompt: str) -> str:
        item = json.loads(prompt.split("\n", 1)[1])["items"][0]
        return json.dumps(
            {
                "entries": [
                    {
                        "concept_id": item["source_terms"][0],
                        "source_terms": item["source_terms"],
                        "target_term": "same-translation",
                        "provenance": item["provenance"],
                        "confidence": 0.9,
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="did not reduce"):
        consolidate_terminology(
            mentions,
            source_lang="en",
            target_lang="fr",
            pi_call=pi_call,
            token_budget=500,
        )


@pytest.mark.parametrize("timeout", [-1, float("nan"), True])
def test_pi_consolidation_rejects_invalid_timeout_before_constructing_leaf(timeout):
    with pytest.raises(ValueError, match="timeout must be non-negative and finite"):
        make_pi_consolidation_call(pi_bin="pi", model="test-model", timeout=timeout)


def test_tool_less_ephemeral_pi_call_returns_stdout_and_cleans_up_timeout(tmp_path):
    fake_pi = tmp_path / "fake-pi"
    fake_pi.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "if sys.argv[-1] == 'sleep':\n"
        "    time.sleep(10)\n"
        "else:\n"
        "    print(json.dumps(sys.argv[1:]))\n"
    )
    fake_pi.chmod(0o755)
    no_timeout_call = make_pi_consolidation_call(
        pi_bin=str(fake_pi), model="test-model", timeout=0
    )
    arguments = json.loads(no_timeout_call("prompt"))
    assert "--no-session" in arguments
    assert "--no-tools" in arguments
    assert "--no-extensions" in arguments
    assert "--no-skills" in arguments
    assert "--no-prompt-templates" in arguments
    assert "--no-context-files" in arguments
    assert "--no-approve" in arguments
    assert arguments[-1] == "prompt"

    bounded_call = make_pi_consolidation_call(
        pi_bin=str(fake_pi), model="test-model", timeout=1
    )
    with pytest.raises(PiConsolidationError, match="timed out"):
        bounded_call("sleep")


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

    sharded = shard_terminology_map(glossary, token_budget=50)

    assert len(sharded.shards) > 1
    assert sharded.alias_index["alias 2"] == ("c2",)
    assert [entry.concept_id for shard in sharded.shards for entry in shard.entries] == [
        "c0", "c1", "c2", "c3"
    ]


def test_sharding_rejects_an_entry_that_cannot_fit_the_requested_context_budget():
    glossary = freeze_terminology(
        [
            TerminologyEntry(
                concept_id="long",
                source_terms=["x" * 80],
                target_term="target",
                provenance=["p1-b1"],
                confidence=1.0,
            )
        ],
        source_lang="en",
        target_lang="fr",
    )

    with pytest.raises(ValueError, match="configured token budget"):
        shard_terminology_map(glossary, token_budget=10)


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


def test_freeze_hash_is_stable_when_same_concept_has_context_variants():
    river = TerminologyEntry(
        concept_id="bank",
        source_terms=["bank"],
        target_term="bank",
        provenance=["p1-b1"],
        confidence=0.5,
        notes="river context",
    )
    finance = TerminologyEntry(
        concept_id="bank",
        source_terms=["bank"],
        target_term="bank",
        provenance=["p2-b1"],
        confidence=0.5,
        notes="financial context",
    )

    first = freeze_terminology([river, finance], source_lang="en", target_lang="fr")
    second = freeze_terminology([finance, river], source_lang="en", target_lang="fr")

    assert first.hash == second.hash
    assert [entry.to_dict() for entry in first.entries] == [entry.to_dict() for entry in second.entries]


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
