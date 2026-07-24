"""Tests for the JSON corpus evaluation harness."""

import base64
import json

from btran.eval_harness import load_eval_cases, run_corpus


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def case_config(name, *, target_text="Bonjour le monde et merci", expected=None):
    return {
        "name": name,
        "risk_tags": ["test-fixture"],
        "fixture_image": "page.png",
        "source": {
            "page_number": 1, "image_path": "page.png", "sha256": "a" * 64,
            "phash": "b" * 16, "source_lang": "en", "model": "test",
            "blocks": [{"id": "b1", "type": "paragraph", "text": "Hello world", "reading_order": 0}],
            "term_mentions": [], "illustrations": [],
        },
        "translation": {
            "page_number": 1, "sha256": "a" * 64, "phash": "b" * 16,
            "source_lang": "en", "target_lang": "fr", "page_text": "Hello world",
            "translated_text": target_text, "blocks": [{"id": "b1", "type": "paragraph", "text": "Hello world", "reading_order": 0}],
            "translated_blocks": [{"block_id": "b1", "translated_text": target_text}],
            "image_descriptions": [], "term_mentions": [], "illustrations": [],
        },
        "glossary": {"version": "1", "hash": "h", "source_lang": "en", "target_lang": "fr", "entries": []},
        "expected": expected or {
            "block_schema": True, "non_empty_text": True, "translation_language": True,
            "illustration_count": True, "block_id_correspondence": True, "glossary_consistency": True,
        },
    }


def write_case(root, config):
    case_dir = root / config["name"]
    case_dir.mkdir()
    (case_dir / "page.png").write_bytes(PNG_1X1)
    (case_dir / "config.json").write_text(json.dumps(config))


def test_load_eval_cases_needs_only_config_json_and_fixture_image(tmp_path):
    write_case(tmp_path, case_config("case_001"))

    cases = load_eval_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].name == "case_001"
    assert cases[0].fixture_image == tmp_path / "case_001" / "page.png"


def test_load_eval_cases_requires_non_empty_string_risk_tags(tmp_path):
    config = case_config("case_without_risk_tags")
    del config["risk_tags"]
    write_case(tmp_path, config)

    import pytest

    with pytest.raises(ValueError, match="risk_tags"):
        load_eval_cases(tmp_path)


def test_load_eval_cases_requires_an_expectation_for_every_validator_stage(tmp_path):
    config = case_config("case_missing_stage")
    del config["expected"]["glossary_consistency"]
    write_case(tmp_path, config)

    import pytest

    with pytest.raises(ValueError, match="expected stages"):
        load_eval_cases(tmp_path)


def test_run_corpus_asserts_expected_stage_results_and_writes_json_report(tmp_path):
    write_case(tmp_path, case_config("case_pass"))
    write_case(tmp_path, case_config(
        "case_expected_language_failure",
        target_text="The quick brown fox is here",
        expected={
            "block_schema": True, "non_empty_text": True, "translation_language": False,
            "illustration_count": True, "block_id_correspondence": True, "glossary_consistency": True,
        },
    ))
    report_path = tmp_path / "report.json"

    report = run_corpus(tmp_path, report_path)

    assert report["all_passed"] is True
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert json.loads(report_path.read_text()) == report
    assert report["cases"][1]["stages"]["translation_language"]["passed"] is True


def test_repository_corpus_has_reviewed_size_category_coverage_and_runs_cleanly():
    corpus = __import__("pathlib").Path(__file__).parents[2] / "eval_corpus"

    cases = load_eval_cases(corpus)
    report = run_corpus(corpus)
    tags = {tag for case in cases for tag in case.risk_tags}

    assert 20 <= len(cases) <= 50
    assert {
        "headings-paragraphs", "lists", "tables", "footnotes", "captions-illustrations",
        "columns-reading-order", "mixed-language", "blank-near-blank", "low-content-risk",
        "low-resolution-risk", "malformed-blocks", "block-id-mismatch", "wrong-target-language",
        "terminology-consistency", "terminology-aliases", "terminology-context-variants",
    } <= tags
    assert report["all_passed"] is True
