"""Pure helpers for the label-confusion pair finder.

No torch / IO here so the ranking and filtering logic stays unit-testable.
A "pair" is one label-0 patch (side 0) and one label-1 patch (side 1).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

#: Patches are an 8x8 grid of 256px tiles per FOV.
PATCHES_PER_FOV_EDGE: int = 8

_NAME_RE = re.compile(
    r"^(?P<slide>.+)__fovr(?P<fov_r>\d+)-fovc(?P<fov_c>\d+)"
    r"__pr(?P<patch_r>\d+)-pc(?P<patch_c>\d+)\.jpg$"
)


@dataclass(frozen=True)
class PatchKey:
    """Identity of one patch, parsed from its JPEG filename."""

    slide: str
    fov_r: int
    fov_c: int
    patch_r: int
    patch_c: int


@dataclass(frozen=True)
class PairCandidate:
    """A candidate (label-0, label-1) pair and its cosine distance."""

    fp0: str
    fp1: str
    dist: float
    key0: PatchKey
    key1: PatchKey


def parse_patch_filename(filename: str) -> PatchKey:
    """Parse a patch JPEG filename into a :class:`PatchKey`.

    Args:
        filename: e.g. ``"CF14-003233D-4__fovr1-fovc9__pr0-pc1.jpg"``.

    Returns:
        The parsed :class:`PatchKey`.

    Raises:
        ValueError: If ``filename`` does not match the expected pattern.
    """
    m = _NAME_RE.match(filename)
    if m is None:
        raise ValueError(f"Unparseable patch filename: {filename!r}")
    return PatchKey(
        slide=m.group("slide"),
        fov_r=int(m.group("fov_r")),
        fov_c=int(m.group("fov_c")),
        patch_r=int(m.group("patch_r")),
        patch_c=int(m.group("patch_c")),
    )


def global_tile_rc(key: PatchKey) -> tuple[int, int]:
    """Return the patch's (row, col) on the slide-wide tile grid.

    Args:
        key: A parsed :class:`PatchKey`.

    Returns:
        ``(row, col)`` in global tile coordinates.
    """
    row = key.fov_r * PATCHES_PER_FOV_EDGE + key.patch_r
    col = key.fov_c * PATCHES_PER_FOV_EDGE + key.patch_c
    return row, col


def is_adjacent_same_slide(k0: PatchKey, k1: PatchKey, adj_tiles: int) -> bool:
    """Return True if both patches are on the same slide within ``adj_tiles`` (Chebyshev).

    Args:
        k0: First patch key.
        k1: Second patch key.
        adj_tiles: Maximum Chebyshev distance for two patches to be considered adjacent.

    Returns:
        True if same slide and Chebyshev distance <= ``adj_tiles``.
    """
    if k0.slide != k1.slide:
        return False
    r0, c0 = global_tile_rc(k0)
    r1, c1 = global_tile_rc(k1)
    return max(abs(r0 - r1), abs(c0 - c1)) <= adj_tiles


def select_pairs(
    candidates: list[PairCandidate],
    n: int,
    accept: Callable[[PairCandidate], bool] | None = None,
) -> list[PairCandidate]:
    """Greedily pick the ``n`` closest pairs, deduping by patch.

    Candidates are sorted ascending by distance. A patch (either side) appears
    in at most one returned pair, so the result is ``n`` distinct examples.

    Args:
        candidates: Candidate pairs (any order).
        n: Number of pairs to return.
        accept: Optional predicate; a candidate is kept only if it returns True.

    Returns:
        Up to ``n`` pairs, ascending by distance.
    """
    used: set[str] = set()
    chosen: list[PairCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.dist):
        if cand.fp0 in used or cand.fp1 in used:
            continue
        if accept is not None and not accept(cand):
            continue
        chosen.append(cand)
        used.add(cand.fp0)
        used.add(cand.fp1)
        if len(chosen) == n:
            break
    return chosen
