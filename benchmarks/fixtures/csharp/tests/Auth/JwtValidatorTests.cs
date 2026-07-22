using QualityFixture.Auth;

public sealed class JwtValidatorTests
{
    [Fact]
    public void ValidateExpirationRejectsExpiredTokens()
    {
        Assert.False(new JwtValidator().ValidateExpiration(10, 11));
    }
}
