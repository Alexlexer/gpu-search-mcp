"""Tests for concurrent search/update behavior."""
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from gpu_index import GpuFileIndex
from gpu_dep_index import DepIndex


class TestGpuIndexConcurrency:
    def test_search_during_index(self, tmp_path):
        """Search while index_directory runs concurrently — must not crash or return garbage."""
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(f"def func_{i}(): return {i}\n")

        idx = GpuFileIndex()
        idx.index_directory(str(tmp_path))

        errors = []
        results_list = []

        def _search():
            for _ in range(50):
                try:
                    r = idx.search("func_")
                    results_list.append(r)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        def _update():
            for i in range(20):
                try:
                    fpath = str(tmp_path / f"file_{i}.py")
                    (tmp_path / f"file_{i}.py").write_text(f"def func_{i}(): return {i * 2}\n")
                    idx.update_file(fpath)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t_search = threading.Thread(target=_search)
        t_update = threading.Thread(target=_update)
        t_search.start()
        t_update.start()
        t_search.join(timeout=10)
        t_update.join(timeout=10)

        assert not errors, f"Concurrent errors: {errors}"
        # At least some searches should have found results
        assert any(len(r) > 0 for r in results_list)

    def test_index_is_consistent_after_concurrent_updates(self, tmp_path):
        """After concurrent updates settle, a fresh search should find updated content."""
        target = tmp_path / "target.py"
        target.write_text("SENTINEL_OLD = 1\n")

        idx = GpuFileIndex()
        idx.index_directory(str(tmp_path))

        barrier = threading.Barrier(3)

        def _writer(content):
            barrier.wait()
            target.write_text(content)
            idx.update_file(str(target))

        threads = [
            threading.Thread(target=_writer, args=("SENTINEL_NEW = 99\n",)),
            threading.Thread(target=_writer, args=("SENTINEL_NEW = 99\n",)),
        ]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join(timeout=5)

        results = idx.search("SENTINEL_NEW")
        assert any("SENTINEL_NEW" in m["content"] for r in results for m in r["matches"])


class TestDepIndexConcurrency:
    def test_search_during_index(self, tmp_path):
        (tmp_path / "a.py").write_text("import b\n")
        (tmp_path / "b.py").write_text("x = 1\n")

        dep = DepIndex()
        dep.index_directory(str(tmp_path))

        errors = []

        def _impact():
            for _ in range(30):
                try:
                    dep.impact(str(tmp_path / "b.py"))
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        def _update():
            for _ in range(10):
                try:
                    dep.update_file(str(tmp_path / "a.py"))
                except Exception as e:
                    errors.append(e)
                time.sleep(0.003)

        t1 = threading.Thread(target=_impact)
        t2 = threading.Thread(target=_update)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

        assert not errors, f"Concurrent dep errors: {errors}"
