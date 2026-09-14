"""Explain exact-search discovery without reading source contents or loading torch."""
import argparse
import json
import math
import os
from pathlib import Path

try:
    from .server_config import INDEXED_EXTS, SKIP_DIRS
except ImportError:
    from server_config import INDEXED_EXTS, SKIP_DIRS


def file_extension(name):
    lower = name.lower()
    return '.env' if lower == '.env' or lower.startswith('.env.') else Path(lower).suffix


def iter_scope(directory, max_file_mb=5.0, allow_env_files=False):
    """Metadata only. Pruned directories produce one entry, not every descendant."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Scope root must be a directory')
    if not math.isfinite(max_file_mb) or max_file_mb <= 0:
        raise ValueError('max_file_mb must be finite and positive')
    maximum = int(max_file_mb * 1024 * 1024)
    extensions = INDEXED_EXTS | ({'.env'} if allow_env_files else set())

    def walk_error(error):
        raise OSError('Scope traversal failed; refusing a silently incomplete index') from error

    for current, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        keep = []
        for name in sorted(dirs):
            path = Path(current)/name
            relative = str(path.relative_to(root))
            if name in SKIP_DIRS:
                yield dict(path=relative, kind='directory', included=False, reason='excluded-directory', bytes=None)
            elif path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                yield dict(path=relative, kind='directory', included=False, reason='link-directory', bytes=None)
            else:
                keep.append(name)
        dirs[:] = keep
        for name in sorted(files):
            path = Path(current)/name
            row = dict(path=str(path.relative_to(root)), kind='file', included=False, bytes=None)
            if file_extension(name) not in extensions:
                yield dict(row, reason='unsupported-extension-or-env-disabled')
                continue
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    yield dict(row, reason='outside-root-link')
                    continue
                if not resolved.is_file():
                    yield dict(row, reason='not-regular-file')
                    continue
                size = path.stat().st_size
            except OSError:
                yield dict(row, reason='unavailable')
                continue
            yield dict(row, bytes=size, included=size <= maximum,
                       reason='included' if size <= maximum else 'file-size-limit')


def explain_scope(directory, max_entries=500, max_file_mb=5.0, allow_env_files=False):
    if max_entries < 1 or max_entries > 100000:
        raise ValueError('max_entries must be between 1 and 100000')
    rows = []
    truncated = False
    for row in iter_scope(directory, max_file_mb, allow_env_files):
        if len(rows) == max_entries:
            truncated = True
            break
        rows.append(row)
    return dict(root=str(Path(directory).resolve()), engine='pattern', entries=rows,
                truncated=truncated, included_files_shown=sum(r['included'] for r in rows),
                included_bytes_shown=sum(r['bytes'] for r in rows if r['included']),
                excludes=sorted(SKIP_DIRS), max_file_mb=max_file_mb,
                allow_env_files=allow_env_files, gitignore_honored=False,
                limitations=['Metadata snapshot, not a reservation against concurrent edits.',
                             'Totals cover displayed entries only; directories are pruned.',
                             'Gitignore/custom glob rules are not implemented by this policy.',
                             'Semantic/dependency/symbol index policies may differ.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--max-entries', type=int, default=500)
    parser.add_argument('--max-file-mb', type=float, default=5)
    parser.add_argument('--allow-env-files', action='store_true')
    args = parser.parse_args()
    try:
        report = explain_scope(args.directory, args.max_entries, args.max_file_mb, args.allow_env_files)
    except (OSError, ValueError) as error:
        parser.exit(2, 'Scope inspection failed: ' + type(error).__name__ + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
