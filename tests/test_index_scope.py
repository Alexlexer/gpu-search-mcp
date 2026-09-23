import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'gpu_service'))
from index_scope import explain_scope, iter_scope


def test_explains_exclusions_without_reading_contents(tmp_path, monkeypatch):
    (tmp_path/'src').mkdir()
    (tmp_path/'src'/'app.cs').write_text('class C {}')
    (tmp_path/'.env.production').write_text('SECRET=not-read')
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'artifacts'/'generated.cs').write_text('class G {}')
    (tmp_path/'big.py').write_text('x'*2048)
    def fail(*args, **kwargs):
        raise AssertionError('Read source contents')
    monkeypatch.setattr(Path, 'read_text', fail)
    monkeypatch.setattr(Path, 'read_bytes', fail)
    report = explain_scope(tmp_path, max_file_mb=.001)
    rows = {r['path'].replace('\\', '/'): r for r in report['entries']}
    assert rows['src/app.cs']['included']
    assert rows['artifacts']['reason'] == 'excluded-directory'
    assert 'artifacts/generated.cs' not in rows
    assert not rows['.env.production']['included']
    assert rows['big.py']['reason'] == 'file-size-limit'
    assert report['gitignore_honored'] is False


def test_truncation_does_not_claim_complete_totals(tmp_path):
    for i in range(4):
        (tmp_path/f'{i}.py').write_text('x')
    report = explain_scope(tmp_path, max_entries=2)
    assert report['truncated']
    assert report['included_files_shown'] == 2
    assert report['included_bytes_shown'] == 2


def test_exact_limit_and_env_opt_in(tmp_path):
    (tmp_path/'.env').write_text('x')
    report = explain_scope(tmp_path, max_entries=1, allow_env_files=True)
    assert not report['truncated']
    assert report['included_files_shown'] == 1


def test_cli_has_no_gpu_imports_or_cache_writes(tmp_path):
    (tmp_path/'example.cs').write_text('class C {}')
    code = "import sys;sys.path.insert(0,sys.argv.pop(1));import index_scope;index_scope.main();assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code, str(ROOT/'gpu_service'), '--directory', str(tmp_path)],
                            capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout)['included_files_shown'] == 1
    assert not (tmp_path/'.gpu-search-cache').exists()
    assert not (tmp_path/'.gpusearch').exists()


def test_shared_policy_matches_pattern_discovery(tmp_path):
    from gpu_index import GpuFileIndex
    (tmp_path/'source.py').write_text('x')
    (tmp_path/'binary.exe').write_bytes(b'abc')
    index = GpuFileIndex()
    try:
        actual, _ = index._discover_files(str(tmp_path), 5*1024*1024, {'.py'})
        expected = [str(tmp_path/r['path']) for r in iter_scope(tmp_path) if r['included']]
        assert actual == expected
    finally:
        index.close()


@pytest.mark.parametrize('limit', [0, -1, 100001])
def test_invalid_limits_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        explain_scope(tmp_path, max_entries=limit)


def test_external_symlink_not_included(tmp_path):
    outside = tmp_path.parent/'outside-scope.py'
    outside.write_text('private')
    try:
        (tmp_path/'linked.py').symlink_to(outside)
    except OSError:
        pytest.skip('OS does not permit symlink creation')
    report = explain_scope(tmp_path)
    assert report['entries'][0]['reason'] == 'outside-root-link'


def test_directory_alias_is_pruned_without_new_pathlib_apis(tmp_path, monkeypatch):
    # Python 3.10 has no Path.is_junction; resolution detects aliases as well.
    alias = tmp_path/'alias'
    alias.mkdir()
    (alias/'hidden.py').write_text('x')
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == alias:
            return tmp_path.parent/'outside-directory'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    report = explain_scope(tmp_path)
    assert len(report['entries']) == 1
    assert report['entries'][0]['reason'] == 'link-directory'
