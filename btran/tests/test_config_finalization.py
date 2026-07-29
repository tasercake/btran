"""Small acceptance coverage for Task 4 base-run configuration grammar."""

import os

import pytest

from btran.config import Config, load_config


@pytest.fixture(autouse=True)
def _isolated_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("BTRAN_"):
            monkeypatch.delenv(name, raising=False)


def test_base_run_syntax_accepts_native_and_explicit_translated_modes():
    native = load_config(["photos", "book.epub"])
    translated = load_config(["photos", "book.epub", "--target-lang", "de"])
    assert native.mode == "native" and native.target_lang is None
    assert translated.mode == "translated" and translated.target_lang == "de"


@pytest.mark.parametrize("arguments", [
    ["photos", "book.epub", "--workspace", ""],
    ["photos", "book.epub", "--base-revision", ""],
    ["photos", "book.epub", "--correction-set", ""],
    ["", "book.epub"], ["photos", ""],
])
def test_blank_run_paths_and_selectors_reject(arguments):
    with pytest.raises(SystemExit) as exc:
        load_config(arguments)
    assert exc.value.code == 2


def test_config_has_no_correction_command_or_review_gate_surface():
    fields = set(Config.__dataclass_fields__)
    assert {"workspace", "base_revision", "correction_set", "refresh"}.issubset(fields)
    assert "review" not in fields
