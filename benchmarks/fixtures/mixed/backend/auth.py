AUTH_TTL_SECONDS = 3600


def token_is_fresh(age_seconds: int) -> bool:
    return age_seconds < AUTH_TTL_SECONDS
