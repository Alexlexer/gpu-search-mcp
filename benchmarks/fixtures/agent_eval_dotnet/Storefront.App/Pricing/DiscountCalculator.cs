namespace Storefront.App.Pricing;

public sealed class DiscountCalculator
{
    public decimal Apply(decimal subtotal, Promotion promotion, DateTimeOffset now)
    {
        if (subtotal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(subtotal));
        }

        if (promotion.Percentage is <= 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(promotion));
        }

        var isActive =
            promotion.StartsAt <= now &&
            promotion.ExpiresAt <= now;

        return isActive
            ? decimal.Round(subtotal * (1 - promotion.Percentage), 2)
            : subtotal;
    }
}
