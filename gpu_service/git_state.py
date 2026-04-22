"""Git-state polling for recency-based search result boosting."""
import os
import subprocess
import threading
import time

BOOST_MODIFIED      = 0.30   # file has uncommitted changes (git diff HEAD)
BOOST_RECENT_COMMIT = 0.15   # file touched in last N commits
BOOST_MTIME         = 0.10   # file mtime within the last hour
BOOST_CAP           = 0.40

_RECENT_COMMITS  = 20
_MTIME_WINDOW    = 3_600     # seconds (1 hour)
_REFRESH_SECONDS = 60        # how often to re-run git in the background


def _run_git(args: list[str], cwd: str) -> list[str]:
    """Run a git command, return lines of stdout. Never blocks the caller."""
    try:
        proc = subprocess.Popen(
            ["git"] + args, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        try:
            stdout, _ = proc.communicate(timeout=8)
            if proc.returncode == 0:
                return [l.strip() for l in stdout.decode("utf-8", errors="replace").splitlines() if l.strip()]
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
    except Exception:
        pass
    return []


def _abs_set(root: str, rel_lines: list[str]) -> set[str]:
    return {os.path.normcase(os.path.abspath(os.path.join(root, p))) for p in rel_lines}


class GitState:
    def __init__(self):
        self._modified: set[str] = set()
        self._recent:   set[str] = set()
        self._roots:    list[str] = []
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()
        self._refreshing = False

    def add_root(self, directory: str):
        """Register a directory. Schedules a background refresh — never blocks."""
        directory = os.path.abspath(directory)
        with self._lock:
            if directory not in self._roots:
                self._roots.append(directory)
        self._schedule_refresh()

    def _schedule_refresh(self):
        """Fire-and-forget: start a refresh thread if one isn't already running."""
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self):
        try:
            modified: set[str] = set()
            recent:   set[str] = set()
            with self._lock:
                roots = list(self._roots)
            for root in roots:
                modified |= _abs_set(root, _run_git(["diff", "--name-only", "HEAD"], root))
                recent   |= _abs_set(root, _run_git(
                    ["diff", "--name-only", f"HEAD~{_RECENT_COMMITS}", "HEAD"], root,
                ))
            with self._lock:
                self._modified = modified
                self._recent   = recent
                self._last_refresh = time.time()
        finally:
            with self._lock:
                self._refreshing = False

    def _maybe_refresh(self):
        """Schedule a background refresh if the data is stale. Never blocks."""
        if time.time() - self._last_refresh > _REFRESH_SECONDS:
            self._schedule_refresh()

    def boost(self, filepath: str) -> float:
        """Return score boost in [0, BOOST_CAP]. Reads cached state — never blocks."""
        with self._lock:
            has_roots = bool(self._roots)
        if not has_roots:
            return 0.0
        self._maybe_refresh()
        key = os.path.normcase(os.path.abspath(filepath))
        with self._lock:
            in_modified = key in self._modified
            in_recent   = key in self._recent
        score = 0.0
        if in_modified:
            score += BOOST_MODIFIED
        elif in_recent:
            score += BOOST_RECENT_COMMIT
        try:
            if time.time() - os.path.getmtime(filepath) <= _MTIME_WINDOW:
                score += BOOST_MTIME
        except Exception:
            pass
        return min(score, BOOST_CAP)
