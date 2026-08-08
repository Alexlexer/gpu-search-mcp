"""Out-of-core packed-corpus, transport, chunking, and compatibility tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from candidates import CandidateSelector, TrigramCandidateSelector
from gpu_buffer import GpuBufferPool
from gpu_index import GpuFileIndex
from packed_corpus import (
    CHUNKS_INDEX_FILENAME,
    CORPUS_FILENAME,
    FILES_INDEX_FILENAME,
    FORMAT_VERSION,
    PackedCorpusCatalog,
    build_packed_corpus,
)
from storage import (
    FileStorageBackend,
    InMemoryStorageBackend,
    MmapStorageBackend,
    ReadResult,
    StorageBackend,
)


class _DeviceReadyStorageBackend(StorageBackend):
    """Test transport that models a completed direct read into a pool buffer."""

    def __init__(self, path: str | Path):
        self._data = Path(path).read_bytes()
        self.device_reads = 0
        self.host_reads = 0

    @property
    def size(self) -> int:
        return len(self._data)

    def read(self, offset: int, size: int, destination) -> ReadResult:
        count = min(size, max(0, self.size - offset))
        payload = self._data[offset:offset + count]
        device = getattr(destination, "device_buffer", None)
        if device is not None:
            device[:count].copy_(torch.tensor(list(payload), dtype=torch.uint8))
            self.device_reads += 1
            return ReadResult(count, device_ready=True)

        memoryview(destination).cast("B")[:count] = payload
        self.host_reads += 1
        return ReadResult(count)


def test_packed_corpus_layout_indexes_empty_utf8_binary_and_large_file(tmp_path: Path):
    empty = tmp_path / "empty.py"
    utf8 = tmp_path / "utf8.py"
    binary = tmp_path / "binary.py"
    large = tmp_path / "large.py"
    empty.write_bytes(b"")
    utf8.write_bytes("snowman = '☃'\n".encode("utf-8"))
    binary.write_bytes(b"prefix\xff\x00suffix\n")
    large.write_bytes(b"L" * (2 * 1024 * 1024 + 17))

    ordered = [empty, utf8, binary, large]
    catalog, stats = build_packed_corpus(tmp_path, ordered, chunk_size=1024 * 1024)

    expected = b"\x00".join(path.read_bytes() for path in ordered) + b"\x00"
    assert (tmp_path / ".gpusearch" / CORPUS_FILENAME).read_bytes() == expected
    assert stats.source_bytes == sum(path.stat().st_size for path in ordered)
    assert stats.total_corpus_size == len(expected)
    assert len(catalog.files) == 4
    assert catalog.files[0].length == 0
    assert [entry.file_id for entry in catalog.files] == list(range(4))
    assert all(
        entry.offset + entry.length < stats.total_corpus_size
        for entry in catalog.files
    )
    assert catalog.chunks[-1].valid_length < catalog.chunk_size

    files_doc = json.loads(
        (tmp_path / ".gpusearch" / FILES_INDEX_FILENAME).read_text(encoding="utf-8")
    )
    chunks_doc = json.loads(
        (tmp_path / ".gpusearch" / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8")
    )
    assert files_doc["version"] == chunks_doc["version"] == FORMAT_VERSION
    assert files_doc["files"][1]["relative_path"] == "utf8.py"
    reloaded = PackedCorpusCatalog.load(tmp_path / ".gpusearch")
    assert reloaded.files == catalog.files
    assert reloaded.chunks == catalog.chunks


@pytest.mark.parametrize("backend_type", [FileStorageBackend, MmapStorageBackend])
def test_file_and_mmap_storage_read_into_destination(tmp_path: Path, backend_type):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(64)))
    with backend_type(corpus) as backend:
        destination = bytearray(12)
        result = backend.read(7, 12, destination)
    assert result.bytes_read == 12
    assert result.device_ready is False
    assert bytes(destination) == bytes(range(7, 19))


def test_device_ready_backend_contract_skips_staging_copy_and_preserves_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "direct.py"
    source.write_bytes(b"prefix needle suffix\nsecond needle\n")
    created: list[_DeviceReadyStorageBackend] = []

    def backend_factory(path: Path) -> StorageBackend:
        backend = _DeviceReadyStorageBackend(path)
        created.append(backend)
        return backend

    index = GpuFileIndex(
        chunk_size=8,
        buffer_count=2,
        storage_backend=backend_factory,
    )
    index.index_directory(str(tmp_path))

    results = index.search("needle", case_sensitive=True)
    assert Path(results[0]["file"]).name == "direct.py"
    assert results[0]["matches"] == [
        {"line": 1, "content": "prefix needle suffix"},
        {"line": 2, "content": "second needle"},
    ]
    assert results[0]["_total_files"] == 1
    metrics = index.stats()["last_query"]
    backend = created[-1]
    assert backend.device_reads > 0
    assert backend.host_reads > 0  # Result-line materialization remains host-based.
    assert metrics["device_ready_bytes"] == metrics["bytes_transferred_to_gpu"]
    assert metrics["host_to_gpu_bytes"] == 0


def test_concurrent_processes_share_one_atomic_packed_build(tmp_path: Path):
    source = tmp_path / "large.py"
    source.write_bytes(b"x" * (2 * 1024 * 1024) + b"needle\n")
    service_dir = REPO_ROOT / "gpu_service"
    program = (
        "import sys;"
        "sys.path.insert(0, sys.argv[2]);"
        "from gpu_index import GpuFileIndex;"
        "index=GpuFileIndex();"
        "print(index.index_directory(sys.argv[1])['cache']);"
        "index.close()"
    )
    commands = [sys.executable, "-c", program, str(tmp_path), str(service_dir)]
    processes = [
        subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    completed = [process.communicate(timeout=60) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], completed
    statuses = sorted(stdout.strip() for stdout, _ in completed)
    assert statuses == ["loaded", "rebuilt"]
    catalog = PackedCorpusCatalog.load(tmp_path / ".gpusearch")
    assert catalog.files[0].length == source.stat().st_size
    assert not [
        path for path in (tmp_path / ".gpusearch").iterdir()
        if path.name.endswith(".tmp")
    ]


def test_in_memory_storage_and_reusable_buffer_pool():
    backend = InMemoryStorageBackend(b"0123456789")
    pool = GpuBufferPool(8, 2, torch.device("cpu"))
    identities = []
    for _ in range(3):
        with pool.acquire() as buffer:
            identities.append(id(buffer))
            transfer = buffer.load(backend, 1, 8)
            assert bytes(buffer.host_view[:8]) == b"12345678"
            assert transfer.host_to_device_bytes == 0
    assert len(set(identities)) <= 2
    pool.ensure_capacity(20)
    assert pool.buffer_size == 20
    pool.close()
    with pytest.raises(RuntimeError):
        with pool.acquire():
            pass


@pytest.mark.parametrize(
    "content, expected_line",
    [
        (b"needle\n", 1),                     # match at chunk start
        (b"x" * 10 + b"needle\n", 1),       # match ends at chunk end
        (b"x" * 14 + b"needle\n", 1),       # match spans the boundary
    ],
)
def test_chunk_boundary_matches_are_found_once(
    tmp_path: Path, content: bytes, expected_line: int
):
    source = tmp_path / "boundary.py"
    source.write_bytes(content)
    index = GpuFileIndex(chunk_size=16, buffer_count=2)
    stats = index.index_directory(str(tmp_path))

    result = index.search("needle", case_sensitive=True)

    assert stats["chunks"] >= 1
    assert len(result) == 1
    assert result[0]["matches"] == [
        {"line": expected_line, "content": content.rstrip().decode()}
    ]


def test_dynamic_overlap_handles_query_larger_than_chunk_and_no_source_reads(
    tmp_path: Path,
):
    query = "query-longer-than-one-chunk"
    source = tmp_path / "long.py"
    source.write_text("abc" + query + "\ntail\n", encoding="utf-8")
    index = GpuFileIndex(chunk_size=8, buffer_count=2)
    index.index_directory(str(tmp_path))
    source.unlink()

    result = index.search(query, case_sensitive=True)

    assert result[0]["file"].endswith("long.py")
    assert result[0]["matches"][0]["content"] == "abc" + query
    assert index._pool.buffer_size >= 8 + len(query.encode()) - 1


def test_adjacent_files_do_not_create_cross_file_matches(tmp_path: Path):
    (tmp_path / "a.py").write_bytes(b"ALPHA")
    (tmp_path / "b.py").write_bytes(b"BETA")
    index = GpuFileIndex(chunk_size=4)
    index.index_directory(str(tmp_path))

    assert index.search("ALPHA", case_sensitive=True)
    assert index.search("BETA", case_sensitive=True)
    assert index.search("ALPHABETA", case_sensitive=True) == []


def test_existing_binary_policy_empty_final_chunk_many_and_zero_matches(tmp_path: Path):
    (tmp_path / "empty.py").write_bytes(b"")
    (tmp_path / "ignored.bin").write_bytes(b"needle")
    (tmp_path / "many.py").write_text(
        "".join(f"needle {number}\n" for number in range(20)), encoding="utf-8"
    )
    index = GpuFileIndex(chunk_size=31)
    stats = index.index_directory(str(tmp_path))

    assert stats["indexed"] == 2
    assert len(index.search("needle")[0]["matches"]) == 10
    assert index.search("absent") == []
    assert index._catalog.chunks[-1].valid_length <= 31
    assert not any(path.endswith("ignored.bin") for path in index._file_names)


class _FixedSelector(CandidateSelector):
    def __init__(self, chunk_ids):
        self.chunk_ids = chunk_ids

    def select(self, query: bytes, catalog: PackedCorpusCatalog):
        return self.chunk_ids


def test_candidate_selector_limits_verification_and_reports_metrics(tmp_path: Path):
    source = tmp_path / "candidate.py"
    source.write_bytes(b"a" * 15 + b"\n" + b"a" * 16 + b"needle\n")
    selector = _FixedSelector([2])
    index = GpuFileIndex(chunk_size=16, candidate_selector=selector)
    index.index_directory(str(tmp_path))

    assert index.search("needle", case_sensitive=True)
    metrics = index.stats()["last_query"]
    assert metrics["candidate_chunks"] == 1
    assert metrics["candidate_percentage"] < 100
    assert metrics["bytes_read_from_storage"] < metrics["total_corpus_size"]
    assert metrics["bytes_transferred_to_gpu"] > 0
    assert metrics["storage_read_seconds"] >= 0
    assert metrics["gpu_search_seconds"] >= 0

    selector.chunk_ids = [0]
    assert index.search("needle", case_sensitive=True) == []


def _legacy_reference(
    paths: list[Path], query: str, case_sensitive: bool, max_files: int = 50
) -> list[dict]:
    pattern = query.encode("utf-8", errors="replace")
    if not case_sensitive:
        pattern = pattern.lower()
    matched_files = []
    for path in sorted(paths):
        raw = path.read_bytes()
        searchable = raw if case_sensitive else raw.lower()
        positions = []
        start = 0
        while pattern:
            position = searchable.find(pattern, start)
            if position < 0:
                break
            positions.append(position)
            start = position + 1
        if not positions:
            continue
        newlines = [index for index, value in enumerate(raw) if value == ord("\n")]
        seen_lines = set()
        matches = []
        for position in positions:
            line_index = sum(newline < position for newline in newlines)
            if line_index in seen_lines:
                continue
            seen_lines.add(line_index)
            line_start = newlines[line_index - 1] + 1 if line_index else 0
            line_end = newlines[line_index] if line_index < len(newlines) else len(raw)
            matches.append({
                "line": line_index + 1,
                "content": raw[line_start:line_end].decode(
                    "utf-8", errors="replace"
                ).rstrip(),
            })
            if len(matches) >= 10:
                break
        matched_files.append({
            "file": str(path.resolve()),
            "matches": matches,
            "_total_files": 0,
        })
    total_files = len(matched_files)
    matched_files = matched_files[:max(0, max_files)]
    for result in matched_files:
        result["_total_files"] = total_files
    return matched_files


@pytest.mark.parametrize(
    "query,case_sensitive",
    [
        ("Needle", True),
        ("needle", False),
        ("é", False),
        ("missing", False),
    ],
)
def test_out_of_core_results_match_legacy_semantics(
    tmp_path: Path, query: str, case_sensitive: bool
):
    paths = [tmp_path / "a.py", tmp_path / "b.py"]
    paths[0].write_text("Needle one\nNEEDLE two\ncafé\n", encoding="utf-8")
    paths[1].write_text("prefix needle suffix\nneedle again\n", encoding="utf-8")
    expected = _legacy_reference(paths, query, case_sensitive)
    index = GpuFileIndex(chunk_size=11, storage_backend="mmap")
    index.index_directory(str(tmp_path))

    assert index.search(query, case_sensitive=case_sensitive) == expected


@pytest.mark.parametrize("storage_backend", ["file", "mmap", "memory"])
@pytest.mark.parametrize("chunk_size", [1, 7, 16, 64])
def test_out_of_core_backend_chunk_matrix_matches_legacy_semantics(
    tmp_path: Path, storage_backend: str, chunk_size: int
) -> None:
    paths = [
        tmp_path / "a.py",
        tmp_path / "b.py",
        tmp_path / "c.py",
        tmp_path / "empty.py",
    ]
    paths[0].write_bytes(
        "Needle one\r\nNEEDLE two\ncafé\nnul".encode("utf-8")
        + b"\x00byte\n"
    )
    paths[1].write_bytes(
        b"aaaaa\nprefix needle suffix\nneedle again\n"
        b"query-longer-than-one-chunk\n"
    )
    paths[2].write_bytes(b"needle third\n\xffneedle invalid\n")
    paths[3].write_bytes(b"")
    cases = [
        ("", False, 50),
        ("Needle", True, 50),
        ("needle", False, 50),
        ("é", False, 50),
        ("aaa", True, 50),
        ("\x00", True, 50),
        ("query-longer-than-one-chunk", True, 50),
        ("needle", False, 2),
        ("missing", False, 50),
    ]
    index = GpuFileIndex(
        chunk_size=chunk_size,
        buffer_count=2,
        storage_backend=storage_backend,
    )
    index.index_directory(str(tmp_path))

    for query, case_sensitive, max_files in cases:
        expected = _legacy_reference(paths, query, case_sensitive, max_files)
        actual = index.search(
            query,
            case_sensitive=case_sensitive,
            max_files=max_files,
        )
        assert actual == expected, (
            storage_backend,
            chunk_size,
            query,
            case_sensitive,
            max_files,
        )


def test_dense_hits_keep_first_ten_lines_and_total_file_count(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("needle\n" * 1000, encoding="utf-8")
    second.write_text("needle\n" * 1000, encoding="utf-8")
    index = GpuFileIndex(chunk_size=64, buffer_count=2)
    index.index_directory(str(tmp_path))

    results = index.search("needle", case_sensitive=True, max_files=1)

    assert len(results) == 1
    assert results[0]["file"] == str(first.resolve())
    assert results[0]["_total_files"] == 2
    assert results[0]["matches"] == [
        {"line": line, "content": "needle"} for line in range(1, 11)
    ]


def test_buffer_read_and_device_stage_are_separate() -> None:
    backend = InMemoryStorageBackend(b"0123456789")
    pool = GpuBufferPool(8, 2, torch.device("cpu"))
    with pool.acquire() as buffer:
        read = buffer.read_from(backend, 1, 8)
        assert bytes(buffer.host_view[:8]) == b"12345678"
        assert read.bytes_read == 8
        transfer = buffer.make_device_ready(8, read)
    assert transfer.host_to_device_bytes == 0
    assert transfer.device_ready is False
    pool.close()


def test_double_buffer_prefetches_next_chunk_during_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pipeline.py"
    source.write_text(("padding line\n" * 20) + "needle\n", encoding="utf-8")
    instances = []

    class ObservedStorage(FileStorageBackend):
        def __init__(self, path):
            super().__init__(path)
            self.read_count = 0
            self.second_read_started = threading.Event()
            instances.append(self)

        def read(self, offset, size, destination):
            self.read_count += 1
            if self.read_count == 2:
                self.second_read_started.set()
            return super().read(offset, size, destination)

    index = GpuFileIndex(
        chunk_size=32,
        buffer_count=2,
        storage_backend=ObservedStorage,
    )
    stats = index.index_directory(str(tmp_path))
    backend = instances[0]
    original_search = index._verifier.search
    calls = 0

    def observed_search(buffer, valid_length, query):
        nonlocal calls
        if calls == 0:
            assert backend.second_read_started.wait(timeout=2)
        calls += 1
        return original_search(buffer, valid_length, query)

    index._verifier.search = observed_search
    assert index.search("needle", case_sensitive=True)
    metrics = index.stats()["last_query"]
    assert stats["chunks"] > 1
    assert metrics["pipeline_enabled"] is True
    assert metrics["prefetched_chunks"] == stats["chunks"] - 1


def test_single_buffer_uses_synchronous_fallback(tmp_path: Path) -> None:
    source = tmp_path / "single.py"
    source.write_text(("padding\n" * 20) + "needle\n", encoding="utf-8")
    index = GpuFileIndex(chunk_size=32, buffer_count=1)
    index.index_directory(str(tmp_path))

    assert index.search("needle", case_sensitive=True)
    metrics = index.stats()["last_query"]
    assert metrics["number_of_chunks"] > 1
    assert metrics["pipeline_enabled"] is False
    assert metrics["prefetched_chunks"] == 0


@pytest.mark.parametrize("storage_backend", ["file", "mmap", "memory"])
def test_trigram_selector_prunes_chunks_without_losing_results(
    tmp_path: Path, storage_backend: str
) -> None:
    source = tmp_path / "selective.py"
    source.write_bytes(b"x" * 32 + b"needle\n" + b"y" * 32)
    index = GpuFileIndex(
        chunk_size=16,
        buffer_count=2,
        storage_backend=storage_backend,
        candidate_selector="trigram",
    )
    stats = index.index_directory(str(tmp_path))

    results = index.search("needle", case_sensitive=True)
    metrics = index.stats()["last_query"]

    assert results[0]["matches"] == [{"line": 1, "content": "x" * 32 + "needle"}]
    assert stats["candidate_selector"] == "TrigramCandidateSelector"
    assert stats["candidate_index_bytes_read"] >= stats["corpus_bytes"]
    assert stats["candidate_index_keys"] > 0
    assert metrics["candidate_chunks"] < metrics["number_of_chunks"]
    assert metrics["candidate_percentage"] < 100
    assert metrics["bytes_read_from_storage"] < metrics["total_corpus_size"]

    assert index.search("missing-token", case_sensitive=True) == []
    missing_metrics = index.stats()["last_query"]
    assert missing_metrics["candidate_chunks"] == 0
    assert missing_metrics["bytes_read_from_storage"] == 0


def test_trigram_selector_keeps_chunk_boundary_and_case_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "boundary.py"
    source.write_bytes(b"x" * 15 + b"NEEDLE\n" + b"z" * 32)
    index = GpuFileIndex(chunk_size=16, candidate_selector=TrigramCandidateSelector())
    stats = index.index_directory(str(tmp_path))

    assert index.search("NEEDLE", case_sensitive=True)
    exact_metrics = index.stats()["last_query"]
    assert exact_metrics["candidate_chunks"] == 1
    assert index.search("needle", case_sensitive=False)
    assert index.stats()["last_query"]["candidate_chunks"] == 1

    assert index.search("NE", case_sensitive=False)
    short_metrics = index.stats()["last_query"]
    assert short_metrics["candidate_chunks"] == stats["chunks"]
