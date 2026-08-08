"""Candidate-chunk selection interfaces and conservative filters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np

from packed_corpus import CorpusChunk, PackedCorpusCatalog
from storage import StorageBackend


@dataclass(frozen=True)
class CandidateBuildStats:
    bytes_read: int = 0
    build_seconds: float = 0.0
    indexed_keys: int = 0


class CandidateSelector(ABC):
    """Select stable chunk IDs; verification only sees the returned chunks."""

    def prepare(
        self, catalog: PackedCorpusCatalog, backend: StorageBackend
    ) -> CandidateBuildStats:
        """Build optional selector state from packed bytes, never source files."""
        return CandidateBuildStats()

    @abstractmethod
    def select(self, query: bytes, catalog: PackedCorpusCatalog) -> Sequence[int]:
        raise NotImplementedError


class AllChunksCandidateSelector(CandidateSelector):
    """Baseline selector that verifies every chunk."""

    def select(self, query: bytes, catalog: PackedCorpusCatalog) -> Sequence[int]:
        return range(len(catalog.chunks))


class TrigramCandidateSelector(CandidateSelector):
    """Conservative first-trigram filter with no query-time storage reads.

    Trigrams are ASCII-folded so the same postings safely serve both exact and
    case-insensitive verification. This can add false positives for exact lower-
    case queries, but never drops a valid match. A two-byte build overlap assigns
    a boundary trigram to the chunk that owns its start.
    """

    def __init__(self):
        self._postings: dict[int, list[int]] = {}
        self._ready = False
        self._stats = CandidateBuildStats()

    @staticmethod
    def _key(data: bytes) -> int:
        folded = data[:3].lower()
        return (folded[0] << 16) | (folded[1] << 8) | folded[2]

    def prepare(
        self, catalog: PackedCorpusCatalog, backend: StorageBackend
    ) -> CandidateBuildStats:
        started = time.perf_counter()
        postings: dict[int, list[int]] = {}
        bytes_read = 0
        capacity = max((chunk.valid_length for chunk in catalog.chunks), default=0) + 2
        destination = bytearray(capacity)
        for chunk in catalog.chunks:
            size = min(
                chunk.valid_length + 2,
                catalog.corpus_size - chunk.offset,
            )
            result = backend.read(chunk.offset, size, destination)
            if result.bytes_read != size:
                raise EOFError(
                    f"short candidate-index read at {chunk.offset}: "
                    f"expected {size}, got {result.bytes_read}"
                )
            bytes_read += result.bytes_read
            if size < 3:
                continue
            values = np.frombuffer(destination, dtype=np.uint8, count=size).copy()
            uppercase = (values >= ord("A")) & (values <= ord("Z"))
            values[uppercase] += 32
            starts = min(chunk.valid_length, size - 2)
            keys = (
                (values[:starts].astype(np.uint32) << 16)
                | (values[1:starts + 1].astype(np.uint32) << 8)
                | values[2:starts + 2].astype(np.uint32)
            )
            for key in np.unique(keys):
                postings.setdefault(int(key), []).append(chunk.chunk_id)
        self._postings = postings
        self._ready = True
        self._stats = CandidateBuildStats(
            bytes_read=bytes_read,
            build_seconds=time.perf_counter() - started,
            indexed_keys=len(postings),
        )
        return self._stats

    def select(self, query: bytes, catalog: PackedCorpusCatalog) -> Sequence[int]:
        if len(query) < 3 or not self._ready:
            return range(len(catalog.chunks))
        return self._postings.get(self._key(query), ())


def resolve_candidates(
    selector: CandidateSelector, query: bytes, catalog: PackedCorpusCatalog
) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    seen: set[int] = set()
    for chunk_id in selector.select(query, catalog):
        chunk_id = int(chunk_id)
        if chunk_id in seen:
            continue
        if chunk_id < 0 or chunk_id >= len(catalog.chunks):
            raise ValueError(f"candidate selector returned invalid chunk ID {chunk_id}")
        seen.add(chunk_id)
        chunks.append(catalog.chunks[chunk_id])
    chunks.sort(key=lambda item: item.chunk_id)
    return chunks
