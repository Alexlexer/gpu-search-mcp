namespace QualityFixture.Auth;

public sealed class TokenService
{
    private readonly JwtValidator _validator = new();

    public bool Accept(long expiresAt, long now)
    {
        return _validator.ValidateExpiration(expiresAt, now);
    }
}
