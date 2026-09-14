"""Perceptual hashing and near-duplicate grouping.

A 64-bit dHash over a 9x8 greyscale reduction. Cheap, stable under re-encoding
and resizing, and good enough to cluster the twelve nearly identical shots of
the same doorway that every travel backlog is full of.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

HASH_BITS = 64
_MASK = (1 << HASH_BITS) - 1


def dhash(gray: np.ndarray) -> int:
    """Hash a 2-D greyscale array (any size; it is resampled to 9x8).

    Returned as a *signed* 64-bit value, because that is what SQLite stores.
    Every comparison here masks back to unsigned first, so the sign is only
    ever a storage detail.
    """
    small = _resize_nearest(gray, width=9, height=8)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return to_signed(value)


def to_signed(value: int) -> int:
    """Map an unsigned 64-bit hash into SQLite's signed integer range."""
    value &= _MASK
    return value - (1 << HASH_BITS) if value >> (HASH_BITS - 1) else value


def to_unsigned(value: int) -> int:
    return value & _MASK


def _resize_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    rows = (np.linspace(0, arr.shape[0] - 1, height)).round().astype(int)
    cols = (np.linspace(0, arr.shape[1] - 1, width)).round().astype(int)
    return arr[np.ix_(rows, cols)]


def hamming(a: int, b: int) -> int:
    """Bit distance between two hashes, signed or unsigned."""
    return int((to_unsigned(a) ^ to_unsigned(b)).bit_count())


def _buckets(value: int, parts: int = 4) -> list[tuple[int, int]]:
    """Split a hash into ``parts`` chunks for pigeonhole blocking."""
    value = to_unsigned(value)
    width = HASH_BITS // parts
    mask = (1 << width) - 1
    return [(i, (value >> (i * width)) & mask) for i in range(parts)]


def group_near_duplicates(
    items: Sequence[tuple[int, int | None]],
    threshold: int,
) -> dict[int, int]:
    """Cluster ``(media_id, phash)`` pairs into groups of near-identical images.

    Returns a mapping of media id to group id. Items with no hash, or with no
    neighbours, are left out of the mapping entirely.

    Blocking by hash chunks keeps this near-linear: two hashes within
    ``threshold`` bits (threshold < parts) must share at least one chunk, so
    only same-chunk candidates are ever compared.
    """
    parts = max(threshold + 1, 4)
    parts = min(parts, 8)
    blocks: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes: dict[int, int] = {}
    for media_id, value in items:
        if value is None:
            continue
        hashes[media_id] = int(value)
        for block in _buckets(int(value), parts):
            blocks[block].append(media_id)

    parent: dict[int, int] = {mid: mid for mid in hashes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for candidates in blocks.values():
        # A chunk shared by half the library is noise (flat black frames, for
        # instance); comparing it pairwise is quadratic for no benefit.
        if len(candidates) > 400:
            continue
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                if hamming(hashes[a], hashes[b]) <= threshold:
                    union(a, b)

    sizes: dict[int, int] = defaultdict(int)
    for mid in hashes:
        sizes[find(mid)] += 1
    return {mid: find(mid) for mid in hashes if sizes[find(mid)] > 1}


def pack_groups(mapping: dict[int, int]) -> dict[int, int]:
    """Renumber group ids to a dense 1..n range for readable output."""
    order: dict[int, int] = {}
    packed: dict[int, int] = {}
    for mid, gid in sorted(mapping.items()):
        packed[mid] = order.setdefault(gid, len(order) + 1)
    return packed
