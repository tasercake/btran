"""Deterministic, side-effect-free validators for translation artifacts."""

from __future__ import annotations

import re
from collections.abc import Sequence

from btran.schema import (
    PageExtraction,
    PageResult,
    SourceBlock,
    TerminologyMap,
    TranslatedBlock,
)


VALID_BLOCK_TYPES = frozenset({
    "heading", "paragraph", "caption", "footnote", "table", "list_item",
    "pull_quote", "illustration", "quote", "page_number",
})

VALIDATION_STAGES = (
    "block_schema",
    "non_empty_text",
    "translation_language",
    "illustration_count",
    "block_id_correspondence",
    "glossary_consistency",
)

_COMMON_WORDS = {
    "en": frozenset({"the", "and", "is", "are", "this", "that", "with", "for", "of", "to"}),
    "fr": frozenset({"le", "la", "les", "de", "des", "et", "bonjour", "merci", "avec", "pour"}),
    "es": frozenset({"el", "la", "los", "las", "de", "y", "hola", "gracias", "con", "para"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "mit", "für", "ein", "eine", "hallo"}),
}


def check_block_schema(blocks: Sequence[SourceBlock]) -> list[str]:
    """Return schema-contract violations for source blocks, in input order."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_reading_orders: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, SourceBlock):
            errors.append(f"block {index} is not a SourceBlock")
            continue
        if not isinstance(block.id, str) or not block.id.strip():
            errors.append(f"block {index} has an empty id")
        elif block.id in seen_ids:
            errors.append(f"duplicate id: {block.id}")
        else:
            seen_ids.add(block.id)
        if block.type not in VALID_BLOCK_TYPES:
            errors.append(f"block {block.id} has unsupported type: {block.type}")
        if not isinstance(block.reading_order, int) or isinstance(block.reading_order, bool) or block.reading_order < 0:
            errors.append(f"block {block.id} has invalid reading_order")
        elif block.reading_order in seen_reading_orders:
            errors.append(f"duplicate reading_order: {block.reading_order}")
        else:
            seen_reading_orders.add(block.reading_order)
    return errors


def check_translated_block_schema(blocks: Sequence[TranslatedBlock]) -> list[str]:
    """Return structural violations for translated blocks without raising."""
    errors: list[str] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, TranslatedBlock):
            errors.append(f"translated block {index} is not a TranslatedBlock")
        elif not isinstance(block.block_id, str) or not block.block_id.strip():
            errors.append(f"translated block {index} has an empty block_id")
    return errors


def check_non_empty_text_fields(
    result: PageResult,
    source: PageExtraction | None = None,
) -> list[str]:
    """Return errors for required page and source/translated block text fields."""
    errors: list[str] = []
    if not _has_text(result.page_text):
        errors.append("page_text is empty")
    if not _has_text(result.translated_text):
        errors.append("translated_text is empty")
    source_blocks = source.blocks if source is not None else result.blocks
    for index, block in enumerate(source_blocks):
        if isinstance(block, SourceBlock) and not _has_text(block.text):
            errors.append(f"source block {block.id} text is empty")
    for block in result.translated_blocks:
        if isinstance(block, TranslatedBlock) and not _has_text(block.translated_text):
            errors.append(f"translated block {block.block_id} text is empty")
    return errors


def detect_language(text: str) -> str | None:
    """Recognize a small deterministic set of languages from script and common words."""
    if not _has_text(text):
        return None
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"

    words = re.findall(r"[a-zA-ZÀ-ÿ]+", text.casefold())
    scores = {language: sum(word in vocabulary for word in words) for language, vocabulary in _COMMON_WORDS.items()}
    language, score = max(scores.items(), key=lambda item: item[1])
    return language if score else None


def check_translation_language(result: PageResult) -> list[str]:
    """Reject a translation only when its detectable language conflicts with target_lang."""
    detected = detect_language(result.translated_text)
    if detected is not None and detected != result.target_lang.casefold():
        return [f"translated_text appears to be {detected}, expected {result.target_lang}"]
    return []


def check_illustration_count(source: PageExtraction, result: PageResult) -> list[str]:
    """Require one non-empty translated description per extracted illustration."""
    expected = len(source.illustrations)
    actual = len(result.image_descriptions)
    if expected != actual:
        return [f"expected {expected} illustration descriptions, got {actual}"]
    return [
        f"illustration description {index} is empty"
        for index, description in enumerate(result.image_descriptions)
        if not _has_text(description)
    ]


def check_block_id_correspondence(source: PageExtraction, result: PageResult) -> list[str]:
    """Require source and translation block IDs to match exactly."""
    invalid_source = [
        f"source block {index} is not a SourceBlock"
        for index, block in enumerate(source.blocks)
        if not isinstance(block, SourceBlock)
    ]
    invalid_translated = [
        f"translated block {index} is not a TranslatedBlock"
        for index, block in enumerate(result.translated_blocks)
        if not isinstance(block, TranslatedBlock)
    ]
    if invalid_source or invalid_translated:
        return invalid_source + invalid_translated
    source_ids = {block.id for block in source.blocks}
    translated_ids = [block.block_id for block in result.translated_blocks]
    translated_set = set(translated_ids)
    errors: list[str] = []
    missing = sorted(source_ids - translated_set)
    extra = sorted(translated_set - source_ids)
    duplicate = sorted({block_id for block_id in translated_ids if translated_ids.count(block_id) > 1})
    if missing:
        errors.append(f"missing translated block IDs: {', '.join(missing)}")
    if extra:
        errors.append(f"extra translated block IDs: {', '.join(extra)}")
    if duplicate:
        errors.append(f"duplicate translated block IDs: {', '.join(duplicate)}")
    return errors


def check_glossary_consistency(
    source: PageExtraction,
    result: PageResult,
    glossary: TerminologyMap,
) -> list[str]:
    """Ensure mentioned glossary terms use their required target term per block."""
    target_by_source_term: dict[str, set[str]] = {}
    for entry in glossary.entries:
        for source_term in entry.source_terms:
            target_by_source_term.setdefault(source_term.casefold(), set()).add(entry.target_term)
    translated_by_id = {
        block.block_id: block.translated_text
        for block in result.translated_blocks
        if isinstance(block, TranslatedBlock)
    }
    errors: list[str] = []
    for mention in source.term_mentions:
        target_terms = target_by_source_term.get(mention.term.casefold())
        if target_terms is None:
            continue
        translation = translated_by_id.get(mention.block_id, "")
        if not any(target_term.casefold() in translation.casefold() for target_term in target_terms):
            required = "' or '".join(sorted(target_terms, key=str.casefold))
            errors.append(
                f"block {mention.block_id} translates glossary term '{mention.term}' "
                f"without required target '{required}'"
            )
    return errors


def validate_page(
    source: PageExtraction,
    result: PageResult,
    glossary: TerminologyMap,
) -> dict[str, list[str]]:
    """Run every deterministic validation stage and return stage-keyed errors."""
    return {
        "block_schema": (
            check_block_schema(source.blocks)
            + check_translated_block_schema(result.translated_blocks)
        ),
        "non_empty_text": check_non_empty_text_fields(result, source),
        "translation_language": check_translation_language(result),
        "illustration_count": check_illustration_count(source, result),
        "block_id_correspondence": check_block_id_correspondence(source, result),
        "glossary_consistency": check_glossary_consistency(source, result, glossary),
    }


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
