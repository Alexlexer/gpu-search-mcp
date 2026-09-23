import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gpu_service'))
import runtime_metrics
from watcher_queue import Debouncer
import gpu_buffer
from gpu_buffer import GpuBufferPool


class FakeBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.allocated_device_bytes = capacity


def test_pool_counts_checked_out_allocations_and_close(monkeypatch):
    monkeypatch.setattr(gpu_buffer, 'GpuBuffer', FakeBuffer)
    pool = GpuBufferPool(1024, 2, SimpleNamespace(type='cuda'))
    first, second = pool.acquire_buffer(), pool.acquire_buffer()
    assert pool.allocated_device_bytes == 2048
    assert pool.stats()['host_staging_bytes'] == 2048
    assert pool.stats()['leased_buffers'] == 2
    pool.close()
    assert pool.allocated_device_bytes == 2048
    pool.release_buffer(first)
    assert pool.allocated_device_bytes == 1024
    pool.release_buffer(second)
    assert pool.allocated_device_bytes == 0


def test_foreign_and_double_return_rejected(monkeypatch):
    monkeypatch.setattr(gpu_buffer, 'GpuBuffer', FakeBuffer)
    pool = GpuBufferPool(16, 2, None)
    buffer = pool.acquire_buffer()
    with pytest.raises(RuntimeError):
        pool.release_buffer(FakeBuffer(16, None))
    pool.release_buffer(buffer)
    with pytest.raises(RuntimeError):
        pool.release_buffer(buffer)
    pool.close()


def test_resize_failure_keeps_original_pool(monkeypatch):
    monkeypatch.setattr(gpu_buffer, 'GpuBuffer', FakeBuffer)
    pool = GpuBufferPool(16, 2, None)
    def fail(*args):
        raise MemoryError()
    monkeypatch.setattr(gpu_buffer, 'GpuBuffer', fail)
    with pytest.raises(MemoryError):
        pool.ensure_capacity(32)
    assert pool.buffer_size == 16
    assert pool.allocated_device_bytes == 32
    pool.close()


def test_watcher_counters_do_not_start_workers():
    queue = Debouncer(60)
    try:
        assert queue.stats()['worker_alive'] is False
        for _ in range(100):
            queue.submit('same', lambda: None)
        stats = queue.stats()
        assert stats['pending'] == stats['high_water'] == 1
        assert stats['coalesced'] == 99
        assert stats['submitted'] == 100
    finally:
        queue.close()


def test_running_and_completed_counters():
    queue = Debouncer(0)
    entered, release = threading.Event(), threading.Event()
    try:
        queue.submit('blocked', lambda: (entered.set(), release.wait(3)))
        assert entered.wait(2)
        assert queue.stats()['running'] == 1
        release.set()
        queue.close()
        assert queue.stats()['completed'] == 1
        assert queue.stats()['running'] == 0
    finally:
        release.set()
        queue.close()


def test_memory_sampler_does_not_import_torch():
    code = "import sys,json;sys.path.insert(0,sys.argv[1]);import runtime_metrics; print(json.dumps(runtime_metrics.memory_snapshot()));assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code, str(ROOT/'gpu_service')], capture_output=True, text=True, check=True, timeout=10)
    metrics = json.loads(result.stdout)
    if sys.platform == 'win32' or sys.platform.startswith('linux'):
        assert metrics['resident_bytes'] > 0
    assert metrics['cuda_allocated_bytes'] is None


def test_memory_failure_reported_without_fake_zero(monkeypatch):
    monkeypatch.setattr(runtime_metrics.sys, 'platform', 'win32')
    def fail():
        raise OSError()
    monkeypatch.setattr(runtime_metrics, '_windows_memory', fail)
    result = runtime_metrics.memory_snapshot()
    assert result['resident_bytes'] is None
    assert result['warnings']


def test_runtime_does_not_initialize_index(monkeypatch):
    import mcp_server
    def fail():
        raise AssertionError('Index initialized by runtime diagnostics')
    monkeypatch.setattr(mcp_server, 'index', mcp_server._LazyService(fail))
    result = mcp_server.runtime_snapshot()
    assert result['pattern_buffers'] is None
    assert result['watcher']['pending'] >= 0


def test_runtime_http_endpoint_avoids_lazy_initialization(monkeypatch):
    import mcp_server
    from tests.test_diagnostics import _get_http
    def fail():
        raise AssertionError('HTTP runtime endpoint initialized an index')
    monkeypatch.setattr(mcp_server, 'index', mcp_server._LazyService(fail))
    status, result = _get_http('/runtime')
    assert status == 200
    assert result['process']['pid'] > 0
    assert result['pattern_buffers'] is None
