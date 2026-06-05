"""Unit tests for the pure label-confusion pair helpers."""

from __future__ import annotations

import pytest

from pair_utils import (
    PairCandidate,
    PatchKey,
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


def _cand(fp0: str, fp1: str, dist: float) -> PairCandidate:
    return PairCandidate(
        fp0=fp0,
        fp1=fp1,
        dist=dist,
        key0=parse_patch_filename(fp0),
        key1=parse_patch_filename(fp1),
    )


def test_select_pairs_sorts_and_caps() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    # Reuses label-0 patch CF-1; allowed because dedupe is on the label-1 side only.
    b = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-3__fovr0-fovc0__pr0-pc0.jpg", 0.20)
    c = _cand("CF-4__fovr0-fovc0__pr0-pc0.jpg", "CF-5__fovr0-fovc0__pr0-pc0.jpg", 0.30)
    out = select_pairs([c, a, b], n=2)  # unsorted input on purpose
    assert [p.dist for p in out] == [0.10, 0.20]


def test_select_pairs_dedupes_label1_patch() -> None:
    # Same label-1 patch (CF-9) twice; only the closer pair is kept.
    near = _cand(
        "CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-9__fovr0-fovc0__pr0-pc0.jpg", 0.05
    )
    far = _cand(
        "CF-2__fovr0-fovc0__pr0-pc0.jpg", "CF-9__fovr0-fovc0__pr0-pc0.jpg", 0.50
    )
    out = select_pairs([far, near], n=5)
    assert [p.dist for p in out] == [0.05]


def test_select_pairs_allows_repeated_label0_patch() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    b = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-3__fovr0-fovc0__pr0-pc0.jpg", 0.20)
    out = select_pairs([a, b], n=5)
    assert len(out) == 2  # label-0 patch CF-1 may appear in both pairs
