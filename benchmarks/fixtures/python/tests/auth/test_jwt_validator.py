from auth.jwt_validator import validate_expiration


def test_rejects_expired_tokens() -> None:
    assert not validate_expiration(10, 11)
