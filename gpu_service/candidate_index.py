"""Versioned persistent posting arrays for conservative trigram candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import uuid

import numpy as np

from packed_corpus import FORMAT_VERSION as PACKED_FORMAT_VERSION, PackedCorpusCatalog


TRIGRAM_INDEX_FILENAME = "trigrams.idx"
TRIGRAM_INDEX_VERSION = 1
_MAGIC = b"GPTRGM01"
_HEADER = struct.Struct("<8sII32s32sQQQQQ")
_KEY_DTYPE = np.dtype("<u4")
_OFFSET_DTYPE = np.dtype("<u8")


@dataclass(frozen=True)
class TrigramPostingsIndex:
    """Compact sorted keys, posting offsets, and stable uint32 chunk IDs."""

    keys: np.ndarray
    offsets: np.ndarray
    postings: np.ndarray
    serialized_bytes: int
    _raw: bytes | None = None

    @classmethod
    def from_mapping(
        cls, mapping: dict[int, list[int]]
    ) -> "TrigramPostingsIndex":
        ordered_keys = sorted(mapping)
        keys = np.asarray(ordered_keys, dtype=_KEY_DTYPE)
        offsets = np.empty(len(ordered_keys) + 1, dtype=_OFFSET_DTYPE)
        offsets[0] = 0
        if ordered_keys:
            counts = np.fromiter(
                (len(mapping[key]) for key in ordered_keys),
                dtype=_OFFSET_DTYPE,
                count=len(ordered_keys),
            )
            np.cumsum(counts, out=offsets[1:])
        postings = np.empty(int(offsets[-1]), dtype=_KEY_DTYPE)
        cursor = 0
        for key in ordered_keys:
            chunk_ids = mapping[key]
            next_cursor = cursor + len(chunk_ids)
            postings[cursor:next_cursor] = chunk_ids
            cursor = next_cursor
        keys.setflags(write=False)
        offsets.setflags(write=False)
        postings.setflags(write=False)
        serialized_bytes = (
            _HEADER.size + keys.nbytes + offsets.nbytes + postings.nbytes
        )
        return cls(keys, offsets, postings, serialized_bytes)

    def select(self, key: int) -> np.ndarray:
        position = int(np.searchsorted(self.keys, key))
        if position >= len(self.keys) or int(self.keys[position]) != key:
            return self.postings[:0]
        start = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return self.postings[start:end]

    def save_atomic(self, path: str | Path, catalog: PackedCorpusCatalog) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        header = _HEADER.pack(
            _MAGIC,
            TRIGRAM_INDEX_VERSION,
            PACKED_FORMAT_VERSION,
            catalog_identity(catalog),
            _payload_digest(self.keys, self.offsets, self.postings),
            catalog.corpus_size,
            catalog.chunk_size,
            len(catalog.chunks),
            len(self.keys),
            len(self.postings),
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(header)
                handle.write(memoryview(self.keys).cast("B"))
                handle.write(memoryview(self.offsets).cast("B"))
                handle.write(memoryview(self.postings).cast("B"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size

    @classmethod
    def load(
        cls, path: str | Path, catalog: PackedCorpusCatalog
    ) -> "TrigramPostingsIndex":
        raw = Path(path).read_bytes()
        if len(raw) < _HEADER.size:
            raise ValueError("trigram index is truncated")
        (
            magic,
            version,
            packed_version,
            identity,
            payload_digest,
            corpus_size,
            chunk_size,
            chunk_count,
            key_count,
            posting_count,
        ) = _HEADER.unpack_from(raw)
        if magic != _MAGIC or version != TRIGRAM_INDEX_VERSION:
            raise ValueError("unsupported trigram index format")
        if packed_version != PACKED_FORMAT_VERSION:
            raise ValueError("trigram index packed-format version mismatch")
        if identity != catalog_identity(catalog):
            raise ValueError("trigram index does not match the packed corpus")
        if (
            corpus_size != catalog.corpus_size
            or chunk_size != catalog.chunk_size
            or chunk_count != len(catalog.chunks)
        ):
            raise ValueError("trigram index catalog metadata mismatch")

        keys_size = key_count * _KEY_DTYPE.itemsize
        offsets_size = (key_count + 1) * _OFFSET_DTYPE.itemsize
        postings_size = posting_count * _KEY_DTYPE.itemsize
        expected_size = _HEADER.size + keys_size + offsets_size + postings_size
        if len(raw) != expected_size:
            raise ValueError("trigram index payload size mismatch")

        cursor = _HEADER.size
        keys = np.frombuffer(raw, dtype=_KEY_DTYPE, count=key_count, offset=cursor)
        cursor += keys_size
        offsets = np.frombuffer(
            raw, dtype=_OFFSET_DTYPE, count=key_count + 1, offset=cursor
        )
        cursor += offsets_size
        postings = np.frombuffer(
            raw, dtype=_KEY_DTYPE, count=posting_count, offset=cursor
        )
        if payload_digest != _payload_digest(keys, offsets, postings):
            raise ValueError("trigram index payload checksum mismatch")
        _validate_arrays(keys, offsets, postings, chunk_count)
        return cls(keys, offsets, postings, len(raw), raw)


def _payload_digest(
    keys: np.ndarray, offsets: np.ndarray, postings: np.ndarray
) -> bytes:
    digest = hashlib.sha256()
    digest.update(memoryview(keys).cast("B"))
    digest.update(memoryview(offsets).cast("B"))
    digest.update(memoryview(postings).cast("B"))
    return digest.digest()


def catalog_identity(catalog: PackedCorpusCatalog) -> bytes:
    """Hash compact catalog metadata and per-file content digests."""
    digest = hashlib.sha256()
    digest.update(b"gpu-search-trigram-catalog-v1")
    digest.update(
        struct.pack(
            "<IQQQ",
            PACKED_FORMAT_VERSION,
            catalog.corpus_size,
            catalog.chunk_size,
            len(catalog.files),
        )
    )
    for entry in catalog.files:
        path = entry.relative_path.encode("utf-8")
        content_digest = entry.digest.encode("ascii")
        digest.update(struct.pack("<I", len(path)))
        digest.update(path)
        digest.update(
            struct.pack("<IQQI", entry.file_id, entry.offset, entry.length, len(content_digest))
        )
        digest.update(content_digest)
    digest.update(struct.pack("<Q", len(catalog.chunks)))
    for chunk in catalog.chunks:
        digest.update(
            struct.pack("<IQQ", chunk.chunk_id, chunk.offset, chunk.valid_length)
        )
    return digest.digest()


def _validate_arrays(
    keys: np.ndarray,
    offsets: np.ndarray,
    postings: np.ndarray,
    chunk_count: int,
) -> None:
    if len(keys) and (int(keys[-1]) > 0xFFFFFF or np.any(keys[1:] <= keys[:-1])):
        raise ValueError("trigram index keys are invalid")
    if len(offsets) != len(keys) + 1 or int(offsets[0]) != 0:
        raise ValueError("trigram index offsets are invalid")
    if int(offsets[-1]) != len(postings):
        raise ValueError("trigram index posting count is invalid")
    if len(keys) and np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("trigram index contains empty or unordered postings")
    if len(postings) and (
        chunk_count == 0 or int(postings.max()) >= chunk_count
    ):
        raise ValueError("trigram index contains an invalid chunk ID")
    if len(postings) > 1:
        continuation = np.ones(len(postings) - 1, dtype=bool)
        boundaries = offsets[1:-1]
        if len(boundaries):
            continuation[boundaries.astype(np.int64) - 1] = False
        if np.any((postings[1:] <= postings[:-1]) & continuation):
            raise ValueError("trigram index postings are not strictly ordered")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
