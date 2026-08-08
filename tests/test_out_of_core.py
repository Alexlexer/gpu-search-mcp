"""Out-of-core packed-corpus, transport, chunking, and compatibility tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from candidates import CandidateSelector
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
from storage import FileStorageBackend, InMemoryStorageBackend, MmapStorageBackend


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


def _legacy_reference(paths: list[Path], query: str, case_sensitive: bool) -> list[dict]:
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
    for result in matched_files:
        result["_total_files"] = len(matched_files)
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
