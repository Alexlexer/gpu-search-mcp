namespace Storefront.App.Pricing;

public sealed record Promotion(
    decimal Percentage,
    DateTimeOffset StartsAt,
    DateTimeOffset ExpiresAt);
