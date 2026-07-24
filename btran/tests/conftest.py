import pytest

from btran.schema import (
    Manifest,
    PageExtraction,
    SourceBlock,
    TermMention,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
)


@pytest.fixture
def sample_source_block():
    return SourceBlock(id="page_1_block_0", type="heading", text="Chapter 1", reading_order=0)


@pytest.fixture
def sample_page_extraction():
    return PageExtraction(
        page_number=1, image_path="test.jpg", sha256="a" * 64, phash="b" * 16,
        source_lang="en", model="test-model", timestamp="2026-01-01T00:00:00Z",
        blocks=[SourceBlock(id="p1_b0", type="paragraph", text="Hello world", reading_order=0)],
        term_mentions=[TermMention(term="hello", block_id="p1_b0")],
        illustrations=[],
    )


@pytest.fixture
def sample_terminology_entry():
    return TerminologyEntry(
        concept_id="c1", source_terms=["hello", "hi"], target_term="bonjour",
        provenance=["hello", "hi"], confidence=0.95,
    )


@pytest.fixture
def sample_terminology_map(sample_terminology_entry):
    return TerminologyMap(
        version="1.0.0", hash="abc123", source_lang="en", target_lang="fr",
        entries=[sample_terminology_entry], created_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_manifest():
    return Manifest(
        input_dir="/tmp/books",
        pages=[{"filename": "page_001.jpg", "page_number": 1, "sha256": "a" * 64, "status": "pending"}],
        total_pages=1,
    )


@pytest.fixture
def sample_translated_block():
    return TranslatedBlock(block_id="p1_b0", translated_text="Bonjour le monde")
