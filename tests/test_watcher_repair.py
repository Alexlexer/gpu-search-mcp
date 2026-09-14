import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
import sys
SERVICE = Path(__file__).resolve().parents[1] / 'gpu_service'
sys.path.insert(0, str(SERVICE))
from watcher_queue import Debouncer
from server_config import SKIP_DIRS


class RepairTests(unittest.TestCase):
    def test_fifty_thousand_events_use_one_worker_and_latest_value(self):
        q = Debouncer(.02)
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        values = []
        def blocked():
            entered.set()
            release.wait(5)
        try:
            q.submit('block', blocked)
            self.assertTrue(entered.wait(2))
            for i in range(50000):
                q.submit('pattern:full-corpus', lambda value: (values.append(value), done.set()), i)
            self.assertEqual(len(q._pending), 1)
            self.assertEqual(sum(t.name == 'gpu-search-watch-updates' for t in threading.enumerate()), 1)
            release.set()
            self.assertTrue(done.wait(2))
            self.assertEqual(values, [49999])
        finally:
            release.set()
            q.close()

    def test_pending_capacity_backpressure_preserves_updates(self):
        q = Debouncer(0, max_pending=2)
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        values = []
        q.submit('block', lambda: (entered.set(), release.wait(5)))
        self.assertTrue(entered.wait(2))
        def produce():
            for i in range(20):
                q.submit(str(i), values.append, i)
            submitted.set()
        producer = threading.Thread(target=produce)
        try:
            producer.start()
            self.assertFalse(submitted.wait(.05))
            with q._condition:
                self.assertLessEqual(len(q._pending), 2)
            release.set()
            self.assertTrue(submitted.wait(2))
            done = threading.Event()
            q.submit('done', done.set)
            self.assertTrue(done.wait(2))
            self.assertEqual(values, list(range(20)))
        finally:
            release.set()
            producer.join(2)
            q.close()

    def test_callback_failure_does_not_kill_worker(self):
        q = Debouncer(0)
        done = threading.Event()
        try:
            with self.assertLogs('watcher_queue', level='ERROR'):
                q.submit('fail', lambda: 1/0)
                q.submit('done', done.set)
                self.assertTrue(done.wait(2))
        finally:
            q.close()

    def test_close_releases_blocked_producer_and_cancels_pending(self):
        q = Debouncer(60, max_pending=1)
        q.submit('a', lambda: self.fail('closed queue ran callback'))
        thread = threading.Thread(target=lambda: q.submit('b', lambda: None))
        thread.start()
        q.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(q._worker.is_alive())

    def watcher(self):
        # Execute only watcher definitions, never server startup or GPU imports.
        tree = ast.parse((SERVICE/'mcp_server.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('_is_skipped_path', '_Watcher')]
        calls = []
        ns = dict(Path=Path, SKIP_DIRS=SKIP_DIRS, FileSystemEventHandler=object,
                  _get_effective_indexed_exts=lambda: {'.py', '.cs', '.json'}, _DEP_EXTS={'.py', '.cs'}, _ALLOW_ENV_FILES=False,
                  _debouncer=SimpleNamespace(submit=lambda *args: calls.append(args)))
        for service in ('index', 'semantic', 'deps', 'symbols'):
            ns[service] = SimpleNamespace(update_file=lambda *a: None)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'watcher-under-test', 'exec'), ns)
        return ns['_Watcher'](), calls

    def test_cache_and_generated_events_are_ignored(self):
        watcher, calls = self.watcher()
        for directory in ('.gpu-search-cache', '.gpusearch', 'artifacts', 'datasets', 'tokenizer-venv', '.GPU-SEARCH-CACHE'):
            event = SimpleNamespace(src_path=f'C:\\DEV\\Astra\\{directory}\\cache-meta.json', is_directory=False)
            for method in (watcher.on_created, watcher.on_modified, watcher.on_deleted):
                method(event)
        self.assertEqual(calls, [])

    def test_source_changes_and_atomic_rename_are_retained(self):
        watcher, calls = self.watcher()
        watcher.on_modified(SimpleNamespace(src_path='C:/DEV/Astra/src/A.cs', is_directory=False))
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][0], 'pattern:full-corpus')
        watcher.on_moved(SimpleNamespace(src_path='C:/DEV/Astra/.gpu-search-cache/a.json', dest_path='C:/DEV/Astra/src/B.cs', is_directory=False))
        self.assertEqual(len(calls), 8)


if __name__ == '__main__':
    unittest.main()
