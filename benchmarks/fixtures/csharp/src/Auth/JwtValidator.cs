namespace QualityFixture.Auth;

public sealed class JwtValidator
{
    public bool ValidateExpiration(long expiresAt, long now)
    {
        return expiresAt > now;
    }
}
