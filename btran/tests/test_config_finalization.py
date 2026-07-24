"""Acceptance tests for the final production CLI configuration surface."""

from pathlib import Path

import pytest

from btran.config import Config, load_config


_ARGS = ["photos", "book.epub", "--target-lang", "fr"]


def test_manifest_is_optional_so_integration_uses_input_directory_default():
    config = load_config(_ARGS)

    assert config.manifest_path is None


def test_explicit_manifest_cli_value_overrides_environment(monkeypatch):
    monkeypatch.setenv("BTRAN_MANIFEST_PATH", "/env/manifest.json")

    config = load_config([*_ARGS, "--manifest-path", "/cli/manifest.json"])

    assert config.manifest_path == Path("/cli/manifest.json")


def test_preflight_only_and_review_are_production_controls():
    config = load_config([*_ARGS, "--preflight-only"])
    assert config.preflight_only is True

    config = load_config([*_ARGS, "--review"])
    assert config.review is True


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--concurrency", "0", "concurrency must be positive"),
        ("--max-retries", "0", "max_retries must be positive"),
        ("--timeout", "0", "timeout must be positive"),
        ("--glossary-budget", "0", "glossary_budget must be positive"),
        ("--glossary-budget", "120001", "glossary_budget must not exceed 120000"),
    ],
)
def test_numeric_limits_fail_during_argument_parsing(flag, value, message, capsys):
    with pytest.raises(SystemExit) as exc:
        load_config([*_ARGS, flag, value])

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_glossary_budget_defaults_to_100k_and_accepts_120k_cap():
    assert load_config(_ARGS).glossary_budget == 100_000
    assert load_config([*_ARGS, "--glossary-budget", "120000"]).glossary_budget == 120_000


@pytest.mark.parametrize(
    "args",
    [
        ["--preflight-only", "--review"],
        ["--epub-check-path", "/opt/epubcheck"],
    ],
)
def test_conflicting_or_incomplete_production_modes_are_rejected(args, capsys):
    with pytest.raises(SystemExit) as exc:
        load_config([*_ARGS, *args])

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_strict_epubcheck_accepts_an_explicit_path_when_enabled():
    config = load_config(
        [*_ARGS, "--epub-check", "--epub-check-path", "/opt/epubcheck"]
    )

    assert config.epub_check is True
    assert config.epub_check_path == "/opt/epubcheck"


@pytest.mark.parametrize(
    "args",
    [
        ["photos", "book.epub", "--target-lang", "fr", "--manifest-path", ""],
        ["", "book.epub", "--target-lang", "fr"],
        ["photos", "", "--target-lang", "fr"],
    ],
)
def test_empty_explicit_paths_are_rejected(args):
    with pytest.raises(SystemExit) as exc:
        load_config(args)

    assert exc.value.code == 2


@pytest.mark.parametrize("banned_flag", ["--no-preflight", "--eval-dir", "--reconciliation-rounds"])
def test_banned_production_flags_are_not_accepted(banned_flag):
    with pytest.raises(SystemExit) as exc:
        load_config([*_ARGS, banned_flag])

    assert exc.value.code == 2


def test_unsafe_legacy_preflight_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("BTRAN_NO_PREFLIGHT", "1")

    with pytest.raises(ValueError, match="preflight is always enabled"):
        load_config(_ARGS)


def test_config_has_no_production_eval_or_bypass_fields():
    config_fields = set(Config.__dataclass_fields__)

    assert {"eval_dir", "no_preflight", "glossary_path"}.isdisjoint(config_fields)
