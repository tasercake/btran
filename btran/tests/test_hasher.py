"""Tests for btran.hasher — TDD: tests first, then implementation."""

from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 128  # large enough for meaningful phash DCT


def _make_png(path: Path, pattern: str) -> None:
    """Write a distinct PNG. *pattern* controls pixel content."""
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
        # Same diagonal but slightly thicker — visually similar
        img = Image.new("RGB", (SIZE, SIZE), color=(255, 255, 255))
        pix = img.load()
        for i in range(SIZE):
            pix[i, i] = (255, 0, 0)
            if i + 1 < SIZE:
                pix[i, i + 1] = (255, 0, 0)
    elif pattern == "gradient_h":
        # Horizontal gradient: black → white — low spatial frequency
        img = Image.new("RGB", (SIZE, SIZE))
        pix = img.load()
        for x in range(SIZE):
            v = int(255 * x / (SIZE - 1))
            for y in range(SIZE):
                pix[x, y] = (v, v, v)
    elif pattern == "gradient_v":
        # Vertical gradient: black → white
        img = Image.new("RGB", (SIZE, SIZE))
        pix = img.load()
        for y in range(SIZE):
            v = int(255 * y / (SIZE - 1))
            for x in range(SIZE):
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


def _make_result(page_number: int = 1, page_text: str = "hello") -> "PageResult":
    from btran.schema import PageResult

    return PageResult(
        page_number=page_number,
        sha256="0" * 64,
        phash="0" * 16,
        page_text=page_text,
    )


# ---------------------------------------------------------------------------
# SHA256
# ---------------------------------------------------------------------------

class TestSHA256:
    def test_consistent_result(self, tmp_path: Path):
        """compute_sha256 returns the same hash for the same file every time."""
        from btran.hasher import compute_sha256

        p = tmp_path / "a.png"
        _make_png(p, "solid_white")
        h1 = compute_sha256(p)
        h2 = compute_sha256(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in h1)

    def test_different_files_have_different_hashes(self, tmp_path: Path):
        from btran.hasher import compute_sha256

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "solid_white")
        _make_png(p2, "solid_black")
        assert compute_sha256(p1) != compute_sha256(p2)


# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------

class TestPHash:
    def test_returns_hex_string(self, tmp_path: Path):
        """compute_phash returns a hex string."""
        from btran.hasher import compute_phash

        p = tmp_path / "img.png"
        _make_png(p, "solid_white")
        h = compute_phash(p)
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)

    def test_similar_images_have_similar_hashes(self, tmp_path: Path):
        """Two nearly-identical images should have low Hamming distance."""
        from btran.hasher import compute_phash, hamming_distance

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "red_diagonal")
        _make_png(p2, "red_diagonal_v2")
        h1 = compute_phash(p1)
        h2 = compute_phash(p2)
        dist = hamming_distance(h1, h2)
        # Very similar images — distance should be small
        assert dist <= 30, f"Expected ≤ 30, got {dist}"

    def test_dissimilar_images_have_different_hashes(self, tmp_path: Path):
        """Visually different images produce different phashes."""
        from btran.hasher import compute_phash, hamming_distance

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        # Random noise images have very different DCT coefficients
        _make_png(p1, "noise_a")
        _make_png(p2, "noise_b")
        h1 = compute_phash(p1)
        h2 = compute_phash(p2)
        # Two independent noise patterns should be far apart
        assert hamming_distance(h1, h2) > 10


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------

class TestHammingDistance:
    def test_known_values(self):
        from btran.hasher import hamming_distance

        assert hamming_distance("00", "00") == 0
        assert hamming_distance("0f", "00") == 4  # 0xf = 1111, differs in 4 bits
        # "ff" vs "00" = 8 bits
        assert hamming_distance("ff", "00") == 8
        # "aa" vs "55" = 10101010 vs 01010101 = 8 bits different
        assert hamming_distance("aa", "55") == 8

    def test_symmetric(self):
        from btran.hasher import hamming_distance

        assert hamming_distance("c0ffee", "deadbe") == hamming_distance(
            "deadbe", "c0ffee"
        )

    def test_different_lengths_raises(self):
        from btran.hasher import hamming_distance

        with pytest.raises(ValueError):
            hamming_distance("ab", "cdef")


# ---------------------------------------------------------------------------
# ImageCache
# ---------------------------------------------------------------------------

class TestImageCache:
    def test_store_and_lookup_returns_same_result(self, tmp_path: Path):
        from btran.hasher import ImageCache

        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            result = _make_result(page_number=3, page_text="hello world")
            cache.store(
                sha256="a" * 64,
                phash="b" * 16,
                image_path="/tmp/img.png",
                result=result,
            )
            cached = cache.lookup("a" * 64)
            assert cached is not None
            assert cached.page_number == 3
            assert cached.page_text == "hello world"
        finally:
            cache.close()

    def test_lookup_returns_none_for_unknown_hash(self, tmp_path: Path):
        from btran.hasher import ImageCache

        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            assert cache.lookup("0" * 64) is None
        finally:
            cache.close()

    def test_lookup_perceptual_finds_near_match(self, tmp_path: Path):
        from btran.hasher import ImageCache, compute_phash

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "red_diagonal")
        _make_png(p2, "red_diagonal_v2")
        phash1 = compute_phash(p1)
        phash2 = compute_phash(p2)

        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            result = _make_result(page_number=1, page_text="first page")
            cache.store("a" * 64, phash1, str(p1), result)

            found = cache.lookup_perceptual(phash2, threshold=30)
            assert found is not None
            assert found.page_number == 1
            assert found.page_text == "first page"
        finally:
            cache.close()

    def test_lookup_perceptual_misses_distant(self, tmp_path: Path):
        from btran.hasher import ImageCache, compute_phash

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _make_png(p1, "noise_a")
        _make_png(p2, "noise_b")
        phash1 = compute_phash(p1)
        phash2 = compute_phash(p2)

        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            result = _make_result(page_number=1, page_text="noise a page")
            cache.store("a" * 64, phash1, str(p1), result)

            found = cache.lookup_perceptual(phash2, threshold=5)
            assert found is None
        finally:
            cache.close()

    def test_end_to_end_image_workflow(self, tmp_path: Path):
        """Create image → hash it → store → retrieve → verify."""
        from btran.hasher import ImageCache, compute_sha256, compute_phash

        p = tmp_path / "page.png"
        _make_png(p, "gradient_h")

        sha = compute_sha256(p)
        ph = compute_phash(p)

        from btran.schema import PageResult

        result = PageResult(
            page_number=7,
            sha256=sha,
            phash=ph,
            page_text="gradient translated",
        )
        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            cache.store(sha, ph, str(p), result)

            # Exact lookup
            cached = cache.lookup(sha)
            assert cached is not None
            assert cached.page_number == 7
            assert cached.page_text == "gradient translated"

            # Perceptual lookup on the same hash
            cached2 = cache.lookup_perceptual(ph, threshold=0)
            assert cached2 is not None
            assert cached2.page_number == 7

            # Non-existent exact lookup
            assert cache.lookup("f" * 64) is None
        finally:
            cache.close()

    def test_db_persists_across_instances(self, tmp_path: Path):
        """Data survives close + re-open."""
        from btran.hasher import ImageCache

        db = tmp_path / "cache.db"
        cache1 = ImageCache(db)
        cache1.store("a" * 64, "b" * 16, "/tmp/x.png", _make_result(page_text="hi"))
        cache1.close()

        cache2 = ImageCache(db)
        try:
            cached = cache2.lookup("a" * 64)
            assert cached is not None
            assert cached.page_text == "hi"
        finally:
            cache2.close()

    def test_store_overwrites_existing_sha256(self, tmp_path: Path):
        from btran.hasher import ImageCache

        db = tmp_path / "cache.db"
        cache = ImageCache(db)
        try:
            cache.store(
                "a" * 64, "b" * 16, "/tmp/x.png", _make_result(page_text="v1")
            )
            cache.store(
                "a" * 64, "c" * 16, "/tmp/y.png", _make_result(page_number=2, page_text="v2")
            )
            cached = cache.lookup("a" * 64)
            assert cached is not None
            assert cached.page_number == 2
            assert cached.page_text == "v2"
        finally:
            cache.close()
