"""Pure helpers for the label-confusion pair finder.

No torch / IO here so the ranking and selection logic stays unit-testable.
A "pair" is one label-0 patch (side 0) and one label-1 patch (side 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def select_pairs(candidates: list[PairCandidate], n: int) -> list[PairCandidate]:
    """Greedily pick the ``n`` closest pairs, deduping by the label-1 patch.

    Candidates are sorted ascending by distance. Each label-1 (relevant) patch
    appears in at most one returned pair; a label-0 patch may recur across pairs.

    Args:
        candidates: Candidate pairs (any order).
        n: Number of pairs to return.

    Returns:
        Up to ``n`` pairs, ascending by distance.
    """
    used_label1: set[str] = set()
    chosen: list[PairCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.dist):
        if cand.fp1 in used_label1:
            continue
        chosen.append(cand)
        used_label1.add(cand.fp1)
        if len(chosen) == n:
            break
    return chosen
