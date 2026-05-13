"""
Secret redaction for search output.

Covers the most common credential formats: API keys, bearer tokens, passwords,
connection strings, and PEM private keys. This is best-effort pattern matching,
not a comprehensive DLP scanner — treat it as a safety net, not a guarantee.
"""
import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Bearer / Authorization header values
    (
        re.compile(r'(Bearer\s+)[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE),
        r'\1[REDACTED]',
    ),
    # API key / secret key / auth token assignments (key = "value" or key: value)
    (
        re.compile(
            r'(?i)(api[_\-]?key|apikey|access[_\-]?key|secret[_\-]?key|'
            r'auth[_\-]?token|auth[_\-]?secret)\s*[=:]\s*["\']?'
            r'([A-Za-z0-9\-._~+/]{16,})["\']?'
        ),
        r'\1=[REDACTED]',
    ),
    # Password assignments: password = "...", PASSWORD=..., passwd=...
    (
        re.compile(
            r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\']{4,})["\']?'
        ),
        r'\1=[REDACTED]',
    ),
    # Connection strings with embedded credentials: proto://user:pass@host
    (
        re.compile(
            r'(?i)(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://'
            r'[^:\s@/]+:[^@\s/]+@'
        ),
        r'\1://[CREDENTIALS_REDACTED]@',
    ),
    # PEM private key blocks (inline, may be multiline)
    (
        re.compile(
            r'-----BEGIN (?:(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY)-----'
            r'.*?'
            r'-----END (?:(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY)-----',
            re.DOTALL,
        ),
        '[PRIVATE KEY REDACTED]',
    ),
    # AWS-style access key IDs (20-char uppercase, starting with AKIA/ASIA/etc.)
    (
        re.compile(r'\b(AKIA|ASIA|AROA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b'),
        '[AWS_KEY_REDACTED]',
    ),
    # Generic long tokens after token/secret/jwt/private_key keywords
    (
        re.compile(
            r'(?i)(token|jwt|client[_\-]?secret|app[_\-]?secret|'
            r'private[_\-]?key|encryption[_\-]?key)\s*[=:]\s*["\']?'
            r'([A-Za-z0-9+/=\-_]{32,})["\']?'
        ),
        r'\1=[REDACTED]',
    ),
]


def redact(text: str) -> str:
    """Apply all secret patterns to text and return the sanitized result."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_match(m: dict) -> dict:
    """Return a copy of a pattern match dict with the 'content' field redacted."""
    out = dict(m)
    if 'content' in out:
        out['content'] = redact(out['content'])
    return out


def redact_chunk(c: dict) -> dict:
    """Return a copy of a semantic chunk dict with the 'snippet' field redacted."""
    out = dict(c)
    if 'snippet' in out:
        out['snippet'] = redact(out['snippet'])
    return out
