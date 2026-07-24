"""Tests for deterministic translation-result validators."""

from btran.schema import (
    PageExtraction,
    PageResult,
    SourceBlock,
    TermMention,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
)
from btran.validators import (
    check_block_id_correspondence,
    check_block_schema,
    check_glossary_consistency,
    check_illustration_count,
    check_non_empty_text_fields,
    check_translation_language,
    detect_language,
    validate_page,
)


def source(*, blocks=None, illustrations=None, mentions=None):
    return PageExtraction(
        page_number=1,
        image_path="fixture.png",
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        model="test",
        blocks=blocks or [SourceBlock("b1", "paragraph", "The engine starts.", 0)],
        illustrations=illustrations or [],
        term_mentions=mentions or [],
    )


def translation(*, blocks=None, text="Le moteur démarre.", descriptions=None, target_lang="fr"):
    return PageResult(
        page_number=1,
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang=target_lang,
        page_text="The engine starts.",
        translated_text=text,
        blocks=[SourceBlock("b1", "paragraph", "The engine starts.", 0)],
        translated_blocks=blocks or [TranslatedBlock("b1", text)],
        image_descriptions=descriptions or [],
    )


def glossary():
    return TerminologyMap(
        version="1", hash="hash", source_lang="en", target_lang="fr",
        entries=[TerminologyEntry("engine", ["engine"], "moteur", ["manual"], 1.0)],
    )


def test_valid_page_passes_every_validator():
    extraction = source(illustrations=["engine diagram"], mentions=[TermMention("engine", "b1")])
    result = translation(descriptions=["A diagram of an engine"], text="Le moteur démarre.")

    assert validate_page(extraction, result, glossary()) == {
        "block_schema": [],
        "non_empty_text": [],
        "translation_language": [],
        "illustration_count": [],
        "block_id_correspondence": [],
        "glossary_consistency": [],
    }


def test_block_schema_rejects_unknown_type_duplicate_id_and_negative_order():
    errors = check_block_schema([
        SourceBlock("b1", "unknown", "text", -1),
        SourceBlock("b1", "paragraph", "other", 1),
    ])

    assert any("unsupported type" in error for error in errors)
    assert any("duplicate id" in error for error in errors)
    assert any("reading_order" in error for error in errors)


def test_non_empty_text_rejects_blank_page_and_block_text():
    result = translation(text=" ", blocks=[TranslatedBlock("b1", "")])
    result.page_text = "\t"

    errors = check_non_empty_text_fields(result)

    assert any("page_text" in error for error in errors)
    assert any("translated_text" in error for error in errors)
    assert any("translated block b1" in error for error in errors)


def test_language_detection_uses_character_sets_and_common_words():
    assert detect_language("こんにちは、世界です") == "ja"
    assert detect_language("Bonjour le monde et merci") == "fr"
    assert detect_language("The quick brown fox is here") == "en"


def test_translation_language_rejects_detectable_wrong_language():
    errors = check_translation_language(translation(text="The quick brown fox is here", target_lang="ja"))

    assert errors == ["translated_text appears to be en, expected ja"]


def test_illustration_count_requires_one_description_per_extracted_illustration():
    errors = check_illustration_count(
        source(illustrations=["map", "portrait"]),
        translation(descriptions=["A map"]),
    )

    assert errors == ["expected 2 illustration descriptions, got 1"]


def test_block_id_correspondence_rejects_missing_and_extra_ids():
    errors = check_block_id_correspondence(
        source(blocks=[
            SourceBlock("b1", "paragraph", "one", 0),
            SourceBlock("b2", "paragraph", "two", 1),
        ]),
        translation(blocks=[TranslatedBlock("b1", "un"), TranslatedBlock("b3", "trois")]),
    )

    assert "missing translated block IDs: b2" in errors
    assert "extra translated block IDs: b3" in errors


def test_glossary_consistency_rejects_missing_required_target_term():
    errors = check_glossary_consistency(
        source(mentions=[TermMention("engine", "b1")]),
        translation(text="La machine démarre."),
        glossary(),
    )

    assert errors == ["block b1 translates glossary term 'engine' without required target 'moteur'"]
