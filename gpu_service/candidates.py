"""Candidate-chunk selection interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from packed_corpus import CorpusChunk, PackedCorpusCatalog


class CandidateSelector(ABC):
    """Select stable chunk IDs; verification only sees the returned chunks."""

    @abstractmethod
    def select(self, query: bytes, catalog: PackedCorpusCatalog) -> Sequence[int]:
        raise NotImplementedError


class AllChunksCandidateSelector(CandidateSelector):
    """Baseline selector used until a pruning index is introduced."""

    def select(self, query: bytes, catalog: PackedCorpusCatalog) -> Sequence[int]:
        return range(len(catalog.chunks))


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
