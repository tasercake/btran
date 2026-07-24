"""Tests for one-round terminology reconciliation."""

from btran.schema import (
    PageExtraction,
    SourceBlock,
    TermMention,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
)


def _glossary(version="1", target_term="cat", digest="v1", include_competitor=False) -> TerminologyMap:
    entries = [
        TerminologyEntry(
            concept_id="cat",
            source_terms=["猫"],
            target_term=target_term,
            provenance=["p1_b1", "p3_b1"],
            confidence=1.0,
        )
    ]
    if include_competitor:
        entries.append(
            TerminologyEntry(
                concept_id="wildcat",
                source_terms=["山猫"],
                target_term="feline",
                provenance=[],
                confidence=1.0,
            )
        )
    return TerminologyMap(
        version=version,
        hash=digest,
        source_lang="ja",
        target_lang="en",
        entries=entries,
        created_at="2026-01-01T00:00:00Z",
    )


def _page(number: int) -> PageExtraction:
    return PageExtraction(
        page_number=number,
        image_path=f"page-{number}.jpg",
        sha256=f"source-{number}",
        phash="unused",
        source_lang="ja",
        model="extractor",
        timestamp="2026-01-01T00:00:00Z",
        blocks=[SourceBlock(id=f"p{number}_b1", type="paragraph", text="猫", reading_order=1)],
        term_mentions=[TermMention(term="猫", block_id=f"p{number}_b1")],
    )


def test_index_terms_to_pages_uses_mentions_to_find_all_affected_pages():
    from btran.reconciliation import index_terms_to_pages

    assert index_terms_to_pages([_page(1), _page(3)], _glossary()) == {"cat": {1, 3}}


def test_glossary_diff_reports_changed_target_form():
    from btran.reconciliation import glossary_diff

    changes = glossary_diff(_glossary(), _glossary(version="2", target_term="feline", digest="v2"))

    assert len(changes) == 1
    assert changes[0].concept_id == "cat"
    assert changes[0].old_target_term == "cat"
    assert changes[0].new_target_term == "feline"


def test_reconcile_flags_missing_terms_and_returns_only_affected_pages_for_retranslation():
    """Only pages whose translation misses the frozen target form are retried."""
    from btran.reconciliation import reconcile

    result = reconcile(
        glossary=_glossary(),
        extractions=[_page(1), _page(3)],
        translations={
            1: [TranslatedBlock(block_id="p1_b1", translated_text="A cat sleeps.")],
            3: [TranslatedBlock(block_id="p3_b1", translated_text="A creature sleeps.")],
        },
    )

    assert [(issue.kind, issue.pages) for issue in result.issues] == [("missing_term", (3,))]
    assert result.affected_pages == [3]


def test_reconcile_reviews_only_ambiguous_context_conflicts_and_applies_one_glossary_v2_change():
    """A conflicting plausible form is reviewed; unambiguous pages never invoke review."""
    from btran.reconciliation import reconcile

    reviewed = []

    def reviewer(issues):
        reviewed.extend(issues)
        return {"cat": "feline"}

    result = reconcile(
        glossary=_glossary(include_competitor=True),
        extractions=[_page(1), _page(3)],
        translations={
            1: [TranslatedBlock(block_id="p1_b1", translated_text="A feline sleeps.")],
            3: [TranslatedBlock(block_id="p3_b1", translated_text="A cat sleeps.")],
        },
        reviewer=reviewer,
    )

    assert len(reviewed) == 1
    assert reviewed[0].kind == "context_conflict"
    assert result.glossary_v2.version == "2"
    assert result.glossary_v2.entries[0].target_term == "feline"
    assert [(change.old_target_term, change.new_target_term) for change in result.glossary_diff] == [
        ("cat", "feline")
    ]
    assert result.affected_pages == [1, 3]
