"""Unit tests for the pure label-confusion pair helpers."""

from __future__ import annotations

import pytest

from pair_utils import (
    PairCandidate,
    PatchKey,
    global_tile_rc,
    is_adjacent_same_slide,
    parse_patch_filename,
    select_pairs,
)


def test_parse_patch_filename_basic() -> None:
    key = parse_patch_filename("CF14-003233D-4__fovr1-fovc9__pr0-pc1.jpg")
    assert key == PatchKey(
        slide="CF14-003233D-4", fov_r=1, fov_c=9, patch_r=0, patch_c=1
    )


def test_parse_patch_filename_rejects_bad_name() -> None:
    with pytest.raises(ValueError):
        parse_patch_filename("not_a_valid_name.jpg")


def test_global_tile_rc() -> None:
    key = PatchKey(slide="s", fov_r=2, fov_c=3, patch_r=4, patch_c=5)
    assert global_tile_rc(key) == (2 * 8 + 4, 3 * 8 + 5)


def test_is_adjacent_same_slide_true_within_radius() -> None:
    k0 = PatchKey("s", 0, 0, 0, 0)  # tile (0, 0)
    k1 = PatchKey("s", 0, 0, 0, 2)  # tile (0, 2) -> cheb dist 2
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is True


def test_is_adjacent_same_slide_false_when_far() -> None:
    k0 = PatchKey("s", 0, 0, 0, 0)  # tile (0, 0)
    k1 = PatchKey("s", 0, 0, 0, 4)  # tile (0, 4) -> cheb dist 4
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is False


def test_is_adjacent_same_slide_false_for_different_slide() -> None:
    k0 = PatchKey("a", 0, 0, 0, 0)
    k1 = PatchKey("b", 0, 0, 0, 0)
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is False


def _cand(fp0: str, fp1: str, dist: float) -> PairCandidate:
    return PairCandidate(
        fp0=fp0,
        fp1=fp1,
        dist=dist,
        key0=parse_patch_filename(fp0),
        key1=parse_patch_filename(fp1),
    )


def test_select_pairs_sorts_and_dedupes() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    # Reuses the label-0 patch from `a`; must be skipped by dedupe.
    b = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-3__fovr0-fovc0__pr0-pc0.jpg", 0.20)
    c = _cand("CF-4__fovr0-fovc0__pr0-pc0.jpg", "CF-5__fovr0-fovc0__pr0-pc0.jpg", 0.30)
    out = select_pairs([c, a, b], n=2)  # unsorted input on purpose
    assert [p.dist for p in out] == [0.10, 0.30]


def test_select_pairs_applies_accept_filter() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    c = _cand("CF-4__fovr0-fovc0__pr0-pc0.jpg", "CF-5__fovr0-fovc0__pr0-pc0.jpg", 0.30)
    out = select_pairs([a, c], n=2, accept=lambda p: p.dist > 0.2)
    assert [p.dist for p in out] == [0.30]
