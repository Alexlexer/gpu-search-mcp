def validate_expiration(expires_at: int, now: int) -> bool:
    return expires_at > now
