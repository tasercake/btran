"""Task 4 configuration and finite-process policy tests."""

from pathlib import Path

import pytest

from btran.config import Config, load_config, resolve_workspace


@pytest.fixture
def clean_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in list(__import__("os").environ):
        if name.startswith("BTRAN_"):
            monkeypatch.delenv(name, raising=False)


def test_absent_target_selects_native(clean_cwd):
    config = load_config(["photos", "book.epub"])
    assert config.target_lang is None
    assert config.mode == "native"


@pytest.mark.parametrize(("environment", "arguments", "expected"), [
    ("ja", ["photos", "book.epub"], "ja"),
    (" ja ", ["photos", "book.epub"], "ja"),
    ("ja", ["photos", "book.epub", "--target-lang", "fr"], "fr"),
])
def test_target_cli_overrides_only_nonblank_environment(clean_cwd, monkeypatch, environment, arguments, expected):
    monkeypatch.setenv("BTRAN_TARGET_LANG", environment)
    config = load_config(arguments)
    assert config.target_lang == expected
    assert config.mode == "translated"


@pytest.mark.parametrize("environment, args", [
    ("", ["photos", "book.epub"]),
    ("   ", ["photos", "book.epub", "--target-lang", "fr"]),
])
def test_blank_environment_target_is_rejected(clean_cwd, monkeypatch, environment, args):
    monkeypatch.setenv("BTRAN_TARGET_LANG", environment)
    with pytest.raises(SystemExit) as exc:
        load_config(args)
    assert exc.value.code == 2


def test_blank_cli_target_is_rejected(clean_cwd):
    with pytest.raises(SystemExit) as exc:
        load_config(["photos", "book.epub", "--target-lang", " \t "])
    assert exc.value.code == 2


@pytest.mark.parametrize(("flag", "value"), [
    ("--concurrency", "0"), ("--concurrency", "33"),
    ("--max-retries", "-1"), ("--max-retries", "6"),
    ("--timeout", "0"), ("--timeout", "3601"),
])
def test_finite_process_bounds_are_rejected_at_parse_time(clean_cwd, flag, value):
    with pytest.raises(SystemExit) as exc:
        load_config(["photos", "book.epub", flag, value])
    assert exc.value.code == 2


def test_timeout_remains_available_for_bounded_utility_processes(clean_cwd):
    config = load_config(["photos", "book.epub", "--timeout", "10", "--max-retries", "3"])
    assert config.timeout == 10
    assert config.retry_backoffs == (1, 2, 4)


def test_new_run_selectors_load_from_cli_and_environment(clean_cwd, monkeypatch):
    monkeypatch.setenv("BTRAN_CORRECTION_SET", "environment-set")
    config = load_config([
        "photos", "book.epub", "--workspace", "state", "--base-revision", "base",
        "--correction-set", "explicit-set", "--refresh",
    ])
    assert config.workspace == Path("state")
    assert config.base_revision == "base"
    assert config.correction_set == "explicit-set"
    assert config.refresh is True


def test_workspace_defaults_beside_output_and_bad_optional_workspace_falls_back(tmp_path):
    output = tmp_path / "output" / "book.epub"
    default = resolve_workspace(Config(output_epub=output))
    assert default.workspace == output.parent / ".btran"
    bad = tmp_path / "not-a-directory"
    bad.write_text("x")
    fallback = resolve_workspace(Config(output_epub=output, workspace=bad))
    assert fallback.workspace == output.parent / ".btran"
    assert fallback.fallback_from == bad


def test_legacy_intermediate_directory_is_only_workspace_fallback_alias(tmp_path):
    legacy = tmp_path / "legacy"
    resolution = resolve_workspace(Config(output_epub=tmp_path / "book.epub", intermediate_dir=legacy))
    assert resolution.workspace == legacy
