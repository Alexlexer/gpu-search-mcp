from backend.auth import AUTH_TTL_SECONDS, token_is_fresh


def test_token_ttl_boundary() -> None:
    assert token_is_fresh(AUTH_TTL_SECONDS - 1)
