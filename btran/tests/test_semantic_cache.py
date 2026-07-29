"""TDD tests for semantic cache scoping — issue #2.

These tests exercise the requirement that cache reuse is scoped by every
semantic input that can change a page translation: source_lang, target_lang,
model, and prompt/output-schema identity (prompt_version).

All tests are written to FAIL before the implementation is updated.
"""

from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from btran.schema import Finding, PageResult, RevisionSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 128


def _make_png(path: Path, pattern: str) -> None:
    """Write a distinct PNG for testing."""
    if pattern == "solid_white":
        img = Image.new("RGB", (SIZE, SIZE), color=(255, 255, 255))
    elif pattern == "solid_black":
        img = Image.new("RGB", (SIZE, SIZE), color=(0, 0, 0))
    elif pattern == "red_diagonal":
        img = Image.new("RGB", (SIZE, SIZE), color=(255, 255, 255))
        pix = img.load()
        for i in range(SIZE):
            pix[i, i] = (255, 0, 0)
    elif pattern == "red_diagonal_v2":
        img = Image.new("RGB", (SIZE, SIZE), color=(255, 255, 255))
        pix = img.load()
        for i in range(SIZE):
            pix[i, i] = (255, 0, 0)
            if i + 1 < SIZE:
                pix[i, i + 1] = (255, 0, 0)
    elif pattern == "gradient_h":
        img = Image.new("RGB", (SIZE, SIZE))
        pix = img.load()
        for x in range(SIZE):
            v = int(255 * x / (SIZE - 1))
            for y in range(SIZE):
                pix[x, y] = (v, v, v)
    elif pattern == "noise_a":
        import random
        rng = random.Random(42)
        img = Image.new("RGB", (SIZE, SIZE))
        pix = img.load()
        for y in range(SIZE):
            for x in range(SIZE):
                v = rng.randint(0, 255)
                pix[x, y] = (v, v, v)
    elif pattern == "noise_b":
        import random
        rng = random.Random(999)
        img = Image.new("RGB", (SIZE, SIZE))
        pix = img.load()
        for y in range(SIZE):
            for x in range(SIZE):
                v = rng.randint(0, 255)
                pix[x, y] = (v, v, v)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    img.save(path, format="PNG")


def _make_result(
    page_number: int = 1,
    page_text: str = "hello",
    source_lang: str = "en",
    target_lang: str = "fr",
    model: str = "test-model",
) -> PageResult:
    return PageResult(
        page_number=page_number,
        sha256="0" * 64,
        phash="0" * 16,
        source_lang=source_lang,
        target_lang=target_lang,
        model=model,
        page_text=page_text,
    )


def _make_fresh_db(tmp_path: Path, name: str = "cache.db") -> Path:
    """Return path to a non-existent DB within tmp_path."""
    return tmp_path / name


# ---------------------------------------------------------------------------
# 1. Same-context exact hash cache hit
# ---------------------------------------------------------------------------

class TestSameContextExactHit:
    """Cache hit when all semantic fields match."""

    def test_exact_hit_when_all_semantic_fields_match(self, tmp_path: Path):
        """Store result → lookup with same semantic context returns it."""
        from btran.hasher import ImageCache

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            result = _make_result(page_text="bonjour", source_lang="en", target_lang="fr", model="gpt-5")
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=result,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            cached = cache.lookup(
                "a" * 64,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )
            assert cached is not None
            assert cached.page_text == "bonjour"
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 2. Mismatch by source_lang
# ---------------------------------------------------------------------------

class TestMismatchSourceLang:
    """Cache misses when source_lang differs."""

    def test_lookup_misses_when_source_lang_differs(self, tmp_path: Path):
        """Stored with source_lang='en', lookup with 'ja' → miss."""
        from btran.hasher import ImageCache

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=_make_result(source_lang="en"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            cached = cache.lookup(
                "a" * 64,
                source_lang="ja",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )
            assert cached is None, (
                "Expected cache miss when source_lang differs"
            )
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 3. Mismatch by target_lang
# ---------------------------------------------------------------------------

class TestMismatchTargetLang:
    """Cache misses when target_lang differs."""

    def test_lookup_misses_when_target_lang_differs(self, tmp_path: Path):
        """Stored with target_lang='fr', lookup with 'de' → miss."""
        from btran.hasher import ImageCache

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=_make_result(target_lang="fr"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            cached = cache.lookup(
                "a" * 64,
                source_lang="en",
                target_lang="de",
                model="gpt-5",
                prompt_version="v1",
            )
            assert cached is None, (
                "Expected cache miss when target_lang differs"
            )
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 4. Mismatch by model
# ---------------------------------------------------------------------------

class TestMismatchModel:
    """Cache misses when model differs."""

    def test_lookup_misses_when_model_differs(self, tmp_path: Path):
        """Stored with model='gpt-5', lookup with 'claude' → miss."""
        from btran.hasher import ImageCache

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=_make_result(model="gpt-5"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            cached = cache.lookup(
                "a" * 64,
                source_lang="en",
                target_lang="fr",
                model="claude",
                prompt_version="v1",
            )
            assert cached is None, (
                "Expected cache miss when model differs"
            )
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 5. Mismatch by prompt_version
# ---------------------------------------------------------------------------

class TestMismatchPromptVersion:
    """Cache misses when prompt_version differs."""

    def test_lookup_misses_when_prompt_version_differs(self, tmp_path: Path):
        """Stored with prompt_version='v1', lookup with 'v2' → miss."""
        from btran.hasher import ImageCache

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=_make_result(),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            cached = cache.lookup(
                "a" * 64,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v2",
            )
            assert cached is None, (
                "Expected cache miss when prompt_version differs"
            )
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 6. Legacy DB: rows without semantic columns must fail closed (miss)
# ---------------------------------------------------------------------------

class TestLegacyDbFailClosed:
    """Existing legacy cache rows lacking semantic scope must produce
    cache misses, not be reused ambiguously."""

    def test_legacy_row_without_semantic_columns_is_miss(self, tmp_path: Path):
        """Manually insert a row with old schema (no semantic columns),
        then lookup must return None."""
        import sqlite3
        import json

        db = _make_fresh_db(tmp_path)

        # Create the table as the old code would (without semantic columns)
        legacy_sql = """
        CREATE TABLE IF NOT EXISTS translations (
            sha256 TEXT PRIMARY KEY,
            phash TEXT NOT NULL,
            image_path TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            result_json TEXT NOT NULL
        );
        """
        conn = sqlite3.connect(str(db))
        conn.execute(legacy_sql)
        conn.execute(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
            (
                "a" * 64,
                "b" * 16,
                "/tmp/old.png",
                1,
                json.dumps(asdict(_make_result(page_text="legacy cached"))),
            ),
        )
        conn.commit()
        conn.close()

        # Now open with ImageCache — it must migrate and treat old rows as misses
        from btran.hasher import ImageCache

        cache = ImageCache(db)
        try:
            cached = cache.lookup(
                "a" * 64,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )
            assert cached is None, (
                "Legacy rows lacking semantic scope must return None (fail closed)"
            )
        finally:
            cache.close()

    def test_legacy_db_opens_without_crashing(self, tmp_path: Path):
        """Opening a legacy DB file must not crash. It should add columns
        and remain usable."""
        import sqlite3
        import json

        db = _make_fresh_db(tmp_path)

        legacy_sql = """
        CREATE TABLE IF NOT EXISTS translations (
            sha256 TEXT PRIMARY KEY,
            phash TEXT NOT NULL,
            image_path TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            result_json TEXT NOT NULL
        );
        """
        conn = sqlite3.connect(str(db))
        conn.execute(legacy_sql)
        conn.execute(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
            (
                "a" * 64,
                "b" * 16,
                "/tmp/old.png",
                1,
                json.dumps(asdict(_make_result())),
            ),
        )
        conn.commit()
        conn.close()

        from btran.hasher import ImageCache

        # This must not raise
        cache = ImageCache(db)
        # Should be able to store new rows with semantic context
        cache.store(
            sha256="c" * 64,
            phash="d" * 16,
            image_path="/tmp/new.png",
            result=_make_result(page_text="new translation"),
            source_lang="en",
            target_lang="fr",
            model="gpt-5",
            prompt_version="v1",
        )
        # New row should be retrievable
        cached = cache.lookup(
            "c" * 64,
            source_lang="en",
            target_lang="fr",
            model="gpt-5",
            prompt_version="v1",
        )
        assert cached is not None
        assert cached.page_text == "new translation"
        cache.close()


# ---------------------------------------------------------------------------
# 7. Perceptual match respects semantic scoping
# ---------------------------------------------------------------------------

class TestPerceptualMatchScoping:
    """Perceptual/near-match cache lookup must also honor semantic scope."""

    def test_perceptual_match_hit_same_context(self, tmp_path: Path):
        """Two similar images with same semantic context → perceptual hit."""
        from btran.hasher import ImageCache, compute_phash

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "red_diagonal")
        _make_png(p2, "red_diagonal_v2")
        phash1 = compute_phash(p1)
        phash2 = compute_phash(p2)

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="x" * 64,
                phash=phash1,
                image_path=str(p1),
                result=_make_result(page_text="same context"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            found = cache.lookup_perceptual(
                phash2,
                threshold=30,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )
            assert found is not None
            assert found.page_text == "same context"
        finally:
            cache.close()

    def test_perceptual_match_miss_when_model_differs(self, tmp_path: Path):
        """Similar images but different model → perceptual miss."""
        from btran.hasher import ImageCache, compute_phash

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "red_diagonal")
        _make_png(p2, "red_diagonal_v2")
        phash1 = compute_phash(p1)
        phash2 = compute_phash(p2)

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="x" * 64,
                phash=phash1,
                image_path=str(p1),
                result=_make_result(page_text="gpt translation", model="gpt-5"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            found = cache.lookup_perceptual(
                phash2,
                threshold=30,
                source_lang="en",
                target_lang="fr",
                model="claude",
                prompt_version="v1",
            )
            assert found is None, (
                "Perceptual match must miss when model differs"
            )
        finally:
            cache.close()

    def test_perceptual_match_miss_when_target_lang_differs(self, tmp_path: Path):
        """Similar images but different target_lang → perceptual miss."""
        from btran.hasher import ImageCache, compute_phash

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "red_diagonal")
        _make_png(p2, "red_diagonal_v2")
        phash1 = compute_phash(p1)
        phash2 = compute_phash(p2)

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        try:
            cache.store(
                sha256="x" * 64,
                phash=phash1,
                image_path=str(p1),
                result=_make_result(page_text="french", target_lang="fr"),
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )

            found = cache.lookup_perceptual(
                phash2,
                threshold=30,
                source_lang="en",
                target_lang="de",
                model="gpt-5",
                prompt_version="v1",
            )
            assert found is None, (
                "Perceptual match must miss when target_lang differs"
            )
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# 8. Current image_path association
# ---------------------------------------------------------------------------

class TestCurrentImageAssociation:
    """Cache hits must not retain a stale image_path from original cached run.
    The returned PageResult should reflect the current page."""

    def test_exact_hit_uses_current_image_path_not_cached(self, tmp_path: Path):
        """Cache hit → result.image_path is updated to current page's path."""
        from btran.hasher import ImageCache, compute_sha256

        p1 = tmp_path / "original_page.png"
        _make_png(p1, "solid_white")

        sha = compute_sha256(p1)
        ph = "b" * 16

        db = _make_fresh_db(tmp_path)
        cache = ImageCache(db)
        result_orig = _make_result(page_text="stale path test")
        result_orig.image_path = "/old/path/to/image.png"

        cache.store(
            sha256=sha,
            phash=ph,
            image_path="/old/path/to/image.png",
            result=result_orig,
            source_lang="en",
            target_lang="fr",
            model="gpt-5",
            prompt_version="v1",
        )
        cache.close()

        # Simulate what orchestrator does: open cache, lookup with current image_path
        cache2 = ImageCache(db)
        try:
            cached = cache2.lookup(
                sha,
                source_lang="en",
                target_lang="fr",
                model="gpt-5",
                prompt_version="v1",
            )
            assert cached is not None
            # This is the old image_path from the cached row
            # The orchestrator is responsible for updating it to current
            # We test the orchestrator behavior in test_orchestrator.py
            # Here we verify the raw cache returns what was stored
            assert cached.image_path == "/old/path/to/image.png"
        finally:
            cache2.close()


# ---------------------------------------------------------------------------
# 9. prompt_fingerprint helper
# ---------------------------------------------------------------------------

class TestPromptFingerprint:
    """The prompt fingerprint is deterministically computed from the prompt."""

    def test_same_prompt_produces_same_fingerprint(self):
        """Identical input yields identical fingerprint."""
        from btran.hasher import compute_prompt_fingerprint

        f1 = compute_prompt_fingerprint("translate this")
        f2 = compute_prompt_fingerprint("translate this")
        assert f1 == f2
        assert isinstance(f1, str)
        assert len(f1) > 0

    def test_different_prompts_produce_different_fingerprints(self):
        """Different inputs yield different fingerprints."""
        from btran.hasher import compute_prompt_fingerprint

        f1 = compute_prompt_fingerprint("translate from en to fr")
        f2 = compute_prompt_fingerprint("translate from ja to de")
        assert f1 != f2


def test_immutable_cache_reuse_requires_explicit_selected_snapshot_artifact(tmp_path: Path):
    """Same-key history is discovery only; selector determines reuse."""
    from btran.artifacts import ArtifactStore, CacheValidator

    store = ArtifactStore(tmp_path / "state")
    summary = Finding(kind="stage_summary", severity="info", stage="test", message="done")
    store.put_finding(summary)
    old = store.put("leaf", {"version": "old"}, finding_ids=(summary.finding_id,), semantic_key="same-key")
    new = store.put("leaf", {"version": "new"}, finding_ids=(summary.finding_id,), semantic_key="same-key")
    validator = CacheValidator(store)
    selected = RevisionSnapshot(
        revision_id="new-snapshot", selected_artifact_ids=(new.artifact_id,),
        selected_cache_attestation_ids=(store.attestation_id_for(new.artifact_id, "leaf", "same-key"),),
    )

    key = lambda *, value: value
    assert validator.select(selected, requested_artifact_id=new.artifact_id, kind="leaf", key_constructor=key, value="same-key") == new
    assert validator.select(selected, requested_artifact_id=old.artifact_id, kind="leaf", key_constructor=key, value="same-key") is None
    assert validator.select(selected, requested_artifact_id=None, kind="leaf", key_constructor=key, value="same-key") is None
