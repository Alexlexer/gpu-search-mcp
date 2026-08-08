"""Versioned packed-corpus builder and stable result-address catalog."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable
import uuid


FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024
PACKED_DIRNAME = ".gpusearch"
CORPUS_FILENAME = "corpus.bin"
FILES_INDEX_FILENAME = "files.idx"
CHUNKS_INDEX_FILENAME = "chunks.idx"
FILE_SEPARATOR = b"\x00"
_COPY_BLOCK_SIZE = 1024 * 1024
_REPLACE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class CorpusFile:
    file_id: int
    relative_path: str
    offset: int
    length: int
    newline_offsets: tuple[int, ...] = ()
    size: int = 0
    mtime_ns: int = 0
    digest: str = ""


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: int
    offset: int
    valid_length: int


@dataclass(frozen=True)
class BuildStats:
    total_corpus_size: int
    source_bytes: int
    file_count: int
    chunk_count: int
    chunk_size: int
    build_time_seconds: float


class PackedCorpusCatalog:
    """Small in-memory metadata index; corpus bytes remain in storage."""

    def __init__(
        self,
        root: str | Path,
        packed_dir: str | Path,
        files: list[CorpusFile],
        chunks: list[CorpusChunk],
        corpus_size: int,
        chunk_size: int,
    ):
        self.root = Path(root).resolve()
        self.packed_dir = Path(packed_dir)
        self.files = files
        self.chunks = chunks
        self.corpus_size = corpus_size
        self.chunk_size = chunk_size
        self._file_offsets = [entry.offset for entry in files]
        self._files_by_id = {entry.file_id: entry for entry in files}

    @property
    def corpus_path(self) -> Path:
        return self.packed_dir / CORPUS_FILENAME

    @property
    def source_bytes(self) -> int:
        return sum(entry.length for entry in self.files)

    def absolute_path(self, entry: CorpusFile) -> str:
        return str((self.root / Path(entry.relative_path)).resolve())

    def file_by_id(self, file_id: int) -> CorpusFile:
        return self._files_by_id[file_id]

    def locate(self, corpus_offset: int, match_length: int = 1) -> tuple[CorpusFile, int] | None:
        """Resolve a stable corpus offset and reject separators/cross-file hits."""
        if not self.files or corpus_offset < 0 or match_length < 0:
            return None
        index = bisect_right(self._file_offsets, corpus_offset) - 1
        if index < 0:
            return None
        entry = self.files[index]
        local = corpus_offset - entry.offset
        if local < 0 or local + match_length > entry.length:
            return None
        return entry, local

    def line_for_offset(self, entry: CorpusFile, file_offset: int) -> tuple[int, int, int]:
        """Return one-based line number plus file-relative [start, end)."""
        line_index = bisect_left(entry.newline_offsets, file_offset)
        start = entry.newline_offsets[line_index - 1] + 1 if line_index else 0
        end = (
            entry.newline_offsets[line_index]
            if line_index < len(entry.newline_offsets)
            else entry.length
        )
        return line_index + 1, start, end

    @classmethod
    def load(cls, packed_dir: str | Path) -> "PackedCorpusCatalog":
        packed_dir = Path(packed_dir)
        file_doc = json.loads((packed_dir / FILES_INDEX_FILENAME).read_text(encoding="utf-8"))
        chunk_doc = json.loads((packed_dir / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8"))
        _validate_header(file_doc, "gpu-search-files")
        _validate_header(chunk_doc, "gpu-search-chunks")
        files = [
            CorpusFile(
                file_id=int(item["file_id"]),
                relative_path=str(item["relative_path"]),
                offset=int(item["offset"]),
                length=int(item["length"]),
                newline_offsets=tuple(int(value) for value in item.get("newline_offsets", [])),
                size=int(item.get("size", item["length"])),
                mtime_ns=int(item.get("mtime_ns", 0)),
                digest=str(item.get("digest", "")),
            )
            for item in file_doc.get("files", [])
        ]
        chunks = [
            CorpusChunk(
                chunk_id=int(item["chunk_id"]),
                offset=int(item["offset"]),
                valid_length=int(item["valid_length"]),
            )
            for item in chunk_doc.get("chunks", [])
        ]
        corpus_size = int(file_doc.get("corpus_size", 0))
        chunk_size = int(chunk_doc["chunk_size"])
        actual_size = (packed_dir / CORPUS_FILENAME).stat().st_size
        if corpus_size != actual_size or int(chunk_doc.get("corpus_size", -1)) != actual_size:
            raise ValueError("packed corpus size does not match its indexes")
        _validate_records(files, chunks, actual_size, chunk_size)
        return cls(file_doc["root"], packed_dir, files, chunks, actual_size, chunk_size)


def build_packed_corpus(
    root: str | Path,
    source_files: Iterable[str | Path],
    *,
    packed_dir: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[PackedCorpusCatalog, BuildStats]:
    """Stream source files into a packed corpus without retaining them in RAM."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    started = time.perf_counter()
    root = Path(root).resolve()
    packed_dir = Path(packed_dir) if packed_dir is not None else root / PACKED_DIRNAME
    packed_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = packed_dir / CORPUS_FILENAME
    build_token = f"{os.getpid()}-{uuid.uuid4().hex}"
    corpus_tmp = packed_dir / f".{CORPUS_FILENAME}.{build_token}.tmp"
    files: list[CorpusFile] = []
    corpus_offset = 0
    source_bytes = 0

    try:
        with corpus_tmp.open("wb") as target:
            for file_id, source in enumerate(Path(item).resolve() for item in source_files):
                stat = source.stat()
                digest = hashlib.blake2b(digest_size=16)
                newline_offsets: list[int] = []
                file_length = 0
                with source.open("rb", buffering=0) as handle:
                    while True:
                        block = handle.read(_COPY_BLOCK_SIZE)
                        if not block:
                            break
                        digest.update(block)
                        cursor = block.find(b"\n")
                        while cursor >= 0:
                            newline_offsets.append(file_length + cursor)
                            cursor = block.find(b"\n", cursor + 1)
                        target.write(block)
                        file_length += len(block)
                relative = os.path.relpath(source, root).replace(os.sep, "/")
                files.append(CorpusFile(
                    file_id=file_id,
                    relative_path=relative,
                    offset=corpus_offset,
                    length=file_length,
                    newline_offsets=tuple(newline_offsets),
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    digest=digest.hexdigest(),
                ))
                source_bytes += file_length
                target.write(FILE_SEPARATOR)
                corpus_offset += file_length + len(FILE_SEPARATOR)
            target.flush()
            os.fsync(target.fileno())

        chunks = [
            CorpusChunk(chunk_id, offset, min(chunk_size, corpus_offset - offset))
            for chunk_id, offset in enumerate(range(0, corpus_offset, chunk_size))
        ]
        file_doc = {
            "format": "gpu-search-files",
            "version": FORMAT_VERSION,
            "root": str(root),
            "corpus_size": corpus_offset,
            "separator_hex": FILE_SEPARATOR.hex(),
            "files": [_file_json(entry) for entry in files],
        }
        chunk_doc = {
            "format": "gpu-search-chunks",
            "version": FORMAT_VERSION,
            "corpus_size": corpus_offset,
            "chunk_size": chunk_size,
            "chunks": [
                {"chunk_id": item.chunk_id, "offset": item.offset,
                 "valid_length": item.valid_length}
                for item in chunks
            ],
        }
        _write_json_atomic(packed_dir / FILES_INDEX_FILENAME, file_doc)
        _write_json_atomic(packed_dir / CHUNKS_INDEX_FILENAME, chunk_doc)
        _replace_with_retry(corpus_tmp, corpus_path)
    except Exception:
        corpus_tmp.unlink(missing_ok=True)
        raise

    catalog = PackedCorpusCatalog(root, packed_dir, files, chunks, corpus_offset, chunk_size)
    stats = BuildStats(
        total_corpus_size=corpus_offset,
        source_bytes=source_bytes,
        file_count=len(files),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        build_time_seconds=time.perf_counter() - started,
    )
    return catalog, stats


def _file_json(entry: CorpusFile) -> dict:
    return {
        "file_id": entry.file_id,
        "relative_path": entry.relative_path,
        "offset": entry.offset,
        "length": entry.length,
        "newline_offsets": list(entry.newline_offsets),
        "size": entry.size,
        "mtime_ns": entry.mtime_ns,
        "digest": entry.digest,
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate short Windows sharing windows from active file readers."""
    deadline = time.monotonic() + _REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _validate_header(document: dict, expected_format: str) -> None:
    if document.get("format") != expected_format:
        raise ValueError(f"unexpected packed index format: {document.get('format')!r}")
    if document.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported packed index version: {document.get('version')!r}")


def _validate_records(
    files: list[CorpusFile], chunks: list[CorpusChunk], corpus_size: int, chunk_size: int
) -> None:
    if chunk_size <= 0:
        raise ValueError("invalid chunk size")
    for expected, entry in enumerate(files):
        if entry.file_id != expected or entry.offset < 0 or entry.length < 0:
            raise ValueError("invalid file record")
        if entry.offset + entry.length > corpus_size:
            raise ValueError("file record exceeds corpus")
        if any(value < 0 or value >= entry.length for value in entry.newline_offsets):
            raise ValueError("invalid newline offset")
    for expected, chunk in enumerate(chunks):
        if chunk.chunk_id != expected or chunk.offset != expected * chunk_size:
            raise ValueError("invalid chunk record")
        if chunk.valid_length <= 0 or chunk.offset + chunk.valid_length > corpus_size:
            raise ValueError("chunk record exceeds corpus")
    expected_chunks = (corpus_size + chunk_size - 1) // chunk_size
    if len(chunks) != expected_chunks:
        raise ValueError("chunk index does not cover corpus")
