"""Cheap process memory snapshots; never imports torch or starts another process."""
import os
from pathlib import Path
import sys
import threading
import time


def _windows_memory():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    info = Counters()
    info.cb = ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(resident_bytes=info.WorkingSetSize, peak_resident_bytes=info.PeakWorkingSetSize,
                private_committed_bytes=info.PrivateUsage, source='Windows GetProcessMemoryInfo')


def _linux_memory():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            fields[key] = int(value.strip().split()[0]) * 1024
    return dict(resident_bytes=fields.get('VmRSS'), peak_resident_bytes=fields.get('VmHWM'),
                private_committed_bytes=None, source='Linux /proc/self/status')


def memory_snapshot():
    result = dict(pid=os.getpid(), python_threads=threading.active_count(),
                  process_cpu_seconds=time.process_time(), resident_bytes=None,
                  peak_resident_bytes=None, private_committed_bytes=None,
                  cuda_allocated_bytes=None, cuda_reserved_bytes=None, warnings=[])
    try:
        if sys.platform == 'win32':
            result.update(_windows_memory())
        elif sys.platform.startswith('linux'):
            result.update(_linux_memory())
        else:
            result['warnings'].append('Live process memory unavailable on this platform.')
    except (OSError, ValueError, AttributeError) as error:
        result['warnings'].append('Process memory unavailable: ' + type(error).__name__)
    torch = sys.modules.get('torch')
    try:
        if torch is not None and torch.cuda.is_initialized():
            result['cuda_allocated_bytes'] = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
            result['cuda_reserved_bytes'] = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count()))
    except (RuntimeError, AttributeError) as error:
        result['warnings'].append('CUDA counters unavailable: ' + type(error).__name__)
    return result
