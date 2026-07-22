from .jwt_validator import validate_expiration


def accept_token(expires_at: int, now: int) -> bool:
    return validate_expiration(expires_at, now)
