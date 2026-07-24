"""Tests for btran.config."""

from pathlib import Path

import pytest

from btran.config import Config, load_config


class TestDefaults:
    """Config loads sensible defaults when nothing is provided."""

    def test_default_model(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.model == "gemini-2.5-flash"

    def test_default_source_lang(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.source_lang == "en"

    def test_default_concurrency(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.concurrency == 4

    def test_default_max_retries(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.max_retries == 3

    def test_default_timeout(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.timeout == 120

    def test_default_intermediate_dir_is_path(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.intermediate_dir == Path("./intermediate")

    def test_default_cache_db_is_path(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.cache_db == Path("./cache.sqlite")

    def test_default_pi_bin(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.pi_bin == "pi"

    def test_default_title(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.title == "Translated Book"

    def test_default_author(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.author == "Unknown"

    def test_default_embed_images(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.embed_images is False

    def test_default_no_resume(self):
        cfg = load_config(["dummy_in", "dummy_out.epub", "--target-lang", "fr"])
        assert cfg.no_resume is False

    def test_positional_input_dir(self):
        cfg = load_config(["/tmp/books", "out.epub", "--target-lang", "es"])
        assert cfg.input_dir == Path("/tmp/books")

    def test_positional_output_epub(self):
        cfg = load_config(["in", "/tmp/output.epub", "--target-lang", "es"])
        assert cfg.output_epub == Path("/tmp/output.epub")


class TestEnvVars:
    """Environment variables populate config fields."""

    def test_env_model(self, monkeypatch):
        monkeypatch.setenv("BTRAN_MODEL", "gpt-4o")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.model == "gpt-4o"

    def test_env_source_lang(self, monkeypatch):
        monkeypatch.setenv("BTRAN_SOURCE_LANG", "ja")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.source_lang == "ja"

    def test_env_target_lang(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TARGET_LANG", "zh")
        cfg = load_config(["in", "out.epub"])
        assert cfg.target_lang == "zh"

    def test_env_concurrency_int(self, monkeypatch):
        monkeypatch.setenv("BTRAN_CONCURRENCY", "8")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.concurrency == 8
        assert isinstance(cfg.concurrency, int)

    def test_env_max_retries_int(self, monkeypatch):
        monkeypatch.setenv("BTRAN_MAX_RETRIES", "5")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.max_retries == 5
        assert isinstance(cfg.max_retries, int)

    def test_env_timeout_int(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TIMEOUT", "300")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.timeout == 300
        assert isinstance(cfg.timeout, int)

    def test_env_intermediate_dir(self, monkeypatch):
        monkeypatch.setenv("BTRAN_INTERMEDIATE_DIR", "/tmp/work")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.intermediate_dir == Path("/tmp/work")

    def test_env_cache_db(self, monkeypatch):
        monkeypatch.setenv("BTRAN_CACHE_DB", "/tmp/cache.db")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.cache_db == Path("/tmp/cache.db")

    def test_env_pi_bin(self, monkeypatch):
        monkeypatch.setenv("BTRAN_PI_BIN", "/usr/local/bin/pi")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.pi_bin == "/usr/local/bin/pi"

    def test_env_title(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TITLE", "My Book")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.title == "My Book"

    def test_env_author(self, monkeypatch):
        monkeypatch.setenv("BTRAN_AUTHOR", "Krishna")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.author == "Krishna"

    def test_env_input_dir(self, monkeypatch):
        monkeypatch.setenv("BTRAN_INPUT_DIR", "/data/scans")
        # CLI positional overrides env — spec says CLI wins.
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.input_dir == Path("in")

    def test_env_output_epub(self, monkeypatch):
        monkeypatch.setenv("BTRAN_OUTPUT_EPUB", "/out/book.epub")
        # CLI positional overrides env — spec says CLI wins.
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert cfg.output_epub == Path("out.epub")


class TestCLIOverridesEnv:
    """CLI args take precedence over environment variables."""

    def test_cli_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_MODEL", "gpt-4o")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--model", "claude-sonnet"])
        assert cfg.model == "claude-sonnet"

    def test_cli_source_lang_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_SOURCE_LANG", "ja")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--source-lang", "ko"])
        assert cfg.source_lang == "ko"

    def test_cli_target_lang_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TARGET_LANG", "zh")
        cfg = load_config(["in", "out.epub", "--target-lang", "es"])
        assert cfg.target_lang == "es"

    def test_cli_concurrency_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_CONCURRENCY", "8")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--concurrency", "16"])
        assert cfg.concurrency == 16

    def test_cli_max_retries_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_MAX_RETRIES", "2")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--max-retries", "10"])
        assert cfg.max_retries == 10

    def test_cli_timeout_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TIMEOUT", "60")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--timeout", "240"])
        assert cfg.timeout == 240

    def test_cli_embed_images_flag(self, monkeypatch):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--embed-images"])
        assert cfg.embed_images is True

    def test_cli_no_resume_flag(self, monkeypatch):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--no-resume"])
        assert cfg.no_resume is True

    def test_cli_intermediate_dir_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_INTERMEDIATE_DIR", "/tmp/work")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--intermediate-dir", "/custom/int"])
        assert cfg.intermediate_dir == Path("/custom/int")

    def test_cli_title_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_TITLE", "Env Title")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--title", "CLI Title"])
        assert cfg.title == "CLI Title"

    def test_cli_author_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_AUTHOR", "Env Author")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--author", "CLI Author"])
        assert cfg.author == "CLI Author"

    def test_cli_pi_bin_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BTRAN_PI_BIN", "/usr/bin/pi")
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--pi-bin", "/opt/pi"])
        assert cfg.pi_bin == "/opt/pi"


class TestMissingTargetLang:
    """target_lang is required — must raise ValueError if not set anywhere."""

    def test_missing_raises_valueerror(self):
        with pytest.raises(ValueError, match="target_lang"):
            load_config(["in", "out.epub"])

    def test_missing_even_with_other_env(self, monkeypatch):
        """Even with other env vars set, missing target_lang still raises."""
        monkeypatch.setenv("BTRAN_MODEL", "gpt-5")
        monkeypatch.setenv("BTRAN_SOURCE_LANG", "de")
        with pytest.raises(ValueError, match="target_lang"):
            load_config(["in", "out.epub"])


class TestIntParsing:
    """Integer fields are parsed correctly from CLI."""

    def test_concurrency_parsed_as_int(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--concurrency", "32"])
        assert cfg.concurrency == 32
        assert isinstance(cfg.concurrency, int)

    def test_max_retries_parsed_as_int(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--max-retries", "7"])
        assert cfg.max_retries == 7
        assert isinstance(cfg.max_retries, int)

    def test_timeout_parsed_as_int(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr", "--timeout", "999"])
        assert cfg.timeout == 999
        assert isinstance(cfg.timeout, int)


class TestPathObjects:
    """Path fields are Path objects."""

    def test_input_dir_is_path(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert isinstance(cfg.input_dir, Path)

    def test_output_epub_is_path(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert isinstance(cfg.output_epub, Path)

    def test_intermediate_dir_is_path(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert isinstance(cfg.intermediate_dir, Path)

    def test_cache_db_is_path(self):
        cfg = load_config(["in", "out.epub", "--target-lang", "fr"])
        assert isinstance(cfg.cache_db, Path)


class TestDotenvLoading:
    """Integration: .env file values are loaded."""

    def test_dotenv_target_lang(self, monkeypatch, tmp_path):
        """Simulate a .env file with target_lang set."""
        envfile = tmp_path / ".env"
        envfile.write_text("BTRAN_TARGET_LANG=it\n")
        monkeypatch.chdir(tmp_path)
        # load_dotenv uses find_dotenv(usecwd=True) so chdir works.
        from dotenv import load_dotenv as _load
        _load(dotenv_path=str(envfile))
        cfg = load_config(["in", "out.epub"])
        assert cfg.target_lang == "it"

    def test_dotenv_overridden_by_cli(self, monkeypatch, tmp_path):
        """CLI overrides .env values."""
        envfile = tmp_path / ".env"
        envfile.write_text("BTRAN_TARGET_LANG=it\nBTRAN_MODEL=env-model\n")
        monkeypatch.chdir(tmp_path)
        from dotenv import load_dotenv as _load
        _load(dotenv_path=str(envfile))
        cfg = load_config(["in", "out.epub", "--target-lang", "de", "--model", "cli-model"])
        assert cfg.target_lang == "de"
        assert cfg.model == "cli-model"
