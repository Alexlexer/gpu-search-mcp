"""Tests for .env exclusion and secret redaction."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from redact import redact, redact_match, redact_chunk
from gpu_index import GpuFileIndex, INDEXED_EXTS


# ---------------------------------------------------------------------------
# .env extension exclusion
# ---------------------------------------------------------------------------

class TestEnvExclusion:
    def test_env_not_in_default_indexed_exts(self):
        assert ".env" not in INDEXED_EXTS, (
            ".env must not be in INDEXED_EXTS by default to avoid leaking secrets"
        )

    def test_env_file_not_indexed_by_default(self, tmp_path):
        (tmp_path / "app.py").write_text("SECRET_KEY = 'abc'\n")
        (tmp_path / ".env").write_text("DB_PASSWORD=supersecret\n")

        idx = GpuFileIndex()
        stats = idx.index_directory(str(tmp_path))
        assert stats["indexed"] == 1, "Only app.py should be indexed, not .env"
        indexed_names = [Path(f).name for f in idx._file_names]
        assert ".env" not in indexed_names

    def test_env_file_indexed_when_opt_in(self, tmp_path):
        (tmp_path / "app.py").write_text("SECRET_KEY = 'abc'\n")
        (tmp_path / ".env").write_text("DB_PASSWORD=supersecret\n")

        idx = GpuFileIndex()
        stats = idx.index_directory(str(tmp_path), allow_env_files=True)
        assert stats["indexed"] == 2, ".env should be indexed with allow_env_files=True"
        indexed_names = [Path(f).name for f in idx._file_names]
        assert ".env" in indexed_names

    def test_env_file_not_indexed_after_update_by_default(self, tmp_path):
        py_file = tmp_path / "app.py"
        env_file = tmp_path / ".env"
        py_file.write_text("x = 1\n")
        env_file.write_text("SECRET=abc\n")

        idx = GpuFileIndex()
        idx.index_directory(str(tmp_path))
        idx.update_file(str(env_file))
        indexed_names = [Path(f).name for f in idx._file_names]
        assert ".env" not in indexed_names


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_bearer_token(self):
        out = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc")
        assert "Bearer [REDACTED]" in out
        assert "eyJhbGciO" not in out

    def test_api_key_equals(self):
        out = redact('api_key = "sk-1234567890abcdef1234"')
        assert "[REDACTED]" in out
        assert "sk-1234567890abcdef1234" not in out

    def test_password_assignment(self):
        out = redact("password = 'my_secret_pw'")
        assert "[REDACTED]" in out
        assert "my_secret_pw" not in out

    def test_connection_string(self):
        out = redact("postgres://user:p4ssw0rd@localhost:5432/db")
        assert "[CREDENTIALS_REDACTED]" in out
        assert "p4ssw0rd" not in out

    def test_pem_private_key(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = redact(pem)
        assert "[PRIVATE KEY REDACTED]" in out
        assert "MIIEpAIBAAK" not in out

    def test_aws_access_key(self):
        # Raw key not in an assignment context (assignment would match api_key pattern first)
        out = redact("Found leaked key AKIAIOSFODNN7EXAMPLE in logs")
        assert "[AWS_KEY_REDACTED]" in out
        assert "AKIAIOSFODNN7EXAMPLE" not in out

    def test_aws_access_key_in_assignment_still_redacted(self):
        # When assigned, the api_key/access_key pattern fires — key is still redacted
        out = redact("access_key = AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in out

    def test_generic_token(self):
        out = redact("token = ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456")
        assert "[REDACTED]" in out
        assert "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456" not in out

    def test_safe_code_not_redacted(self):
        safe = "def authenticate(user, password_hash): return check(user, password_hash)"
        out = redact(safe)
        # The function signature should survive — no secret value here
        assert "def authenticate" in out

    def test_redact_match_dict(self):
        m = {"line": 5, "content": "api_key = 'sk-secret12345678901234'"}
        out = redact_match(m)
        assert out["line"] == 5
        assert "sk-secret" not in out["content"]

    def test_redact_chunk_dict(self):
        c = {"snippet": "token = 'ghp_abc123def456ghi789jkl012mno34567890'"}
        out = redact_chunk(c)
        assert "ghp_abc123" not in out["snippet"]
