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


def test_repository_corpus_has_at_least_five_cases_and_runs_cleanly():
    corpus = __import__("pathlib").Path(__file__).parents[2] / "eval_corpus"

    report = run_corpus(corpus)

    assert len(report["cases"]) >= 5
    assert report["all_passed"] is True
