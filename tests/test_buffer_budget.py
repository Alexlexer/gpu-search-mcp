from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'gpu_service'))
import gpu_buffer
from gpu_buffer import GpuBufferPool


@pytest.fixture
def allocations(monkeypatch):
    calls = []
    class Buffer:
        def __init__(self, capacity, device):
            calls.append(capacity)
            self.capacity = capacity
            self.allocated_device_bytes = capacity if device.type != 'cpu' else 0
    monkeypatch.setattr(gpu_buffer, 'GpuBuffer', Buffer)
    return calls


def test_accelerator_budget_counts_host_and_device_before_allocation(allocations):
    with pytest.raises(MemoryError, match='budget'):
        GpuBufferPool(1024, 2, SimpleNamespace(type='cuda'), max_owned_bytes=4095)
    assert allocations == []


def test_cpu_alias_is_not_double_counted(allocations):
    pool = GpuBufferPool(1024, 2, SimpleNamespace(type='cpu'), max_owned_bytes=2048)
    assert len(allocations) == 2
    assert pool.stats()['host_staging_bytes'] == 2048
    assert pool.stats()['device_bytes'] == 0
    pool.close()


def test_resize_checks_peak_and_preserves_original_pool(allocations):
    pool = GpuBufferPool(1024, 2, SimpleNamespace(type='cuda'), max_owned_bytes=8192)
    with pytest.raises(MemoryError):
        pool.ensure_capacity(2048)  # new=8192 plus old=4096
    assert allocations == [1024, 1024]
    assert pool.buffer_size == 1024
    assert pool.allocated_device_bytes == 2048
    with pool.acquire():
        assert pool.stats()['leased_buffers'] == 1
    pool.close()


def test_resize_with_sufficient_peak_budget(allocations):
    pool = GpuBufferPool(1024, 2, SimpleNamespace(type='cuda'), max_owned_bytes=12288)
    pool.ensure_capacity(2048)
    assert allocations == [1024, 1024, 2048, 2048]
    assert pool.stats()['device_bytes'] == 4096
    assert pool.stats()['host_staging_bytes'] == 4096
    pool.close()


@pytest.mark.parametrize('value', ['0', '-1', 'invalid', '1.5'])
def test_invalid_environment_limit_rejected(monkeypatch, value):
    from gpu_index import GpuFileIndex
    monkeypatch.setenv('GPU_SEARCH_BUFFER_BUDGET_MB', value)
    with pytest.raises(ValueError, match='GPU_SEARCH_BUFFER_BUDGET_MB'):
        GpuFileIndex()
